"""
Main script for training VAE models and running inference on ADNI-B data.

This script loads ADNI-B data and can be used for training models or running inference.
"""

import argparse
import json
import pathlib
import re
import torch
import yaml
import itertools
from copy import deepcopy
from neuronumba.tools.filters import BandPassFilter
from sklearn.decomposition import PCA

from .utils import *
from .load_data import load_adni, prepare_data_loaders
from training_tracker import TrainingResultsManager

CACHED_ADNI = None

def load_config(config_path):
    def _normalize_numeric_values(value):
        if isinstance(value, dict):
            return {k: _normalize_numeric_values(v) for k, v in value.items()}
        if isinstance(value, list):
            return [_normalize_numeric_values(item) for item in value]
        if isinstance(value, str):
            stripped = value.strip()
            if re.fullmatch(r"[+-]?\d+", stripped):
                return int(stripped)
            if re.fullmatch(r"[+-]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][+-]?\d+)?", stripped):
                return float(stripped)
        return value

    with open(config_path, 'r') as file:
        conf = yaml.load(file, Loader=yaml.FullLoader)
    return _normalize_numeric_values(conf)


def load_data_from_config(data_dir, data_config):
    global CACHED_ADNI
    if CACHED_ADNI is None:
        # Load ADNI data
        print(f"\nLoading ADNI-B dataset...")

        if 'filter' in data_config and data_config['filter']['type'] == 'BandPassFilter':
            filter_config = data_config['filter']
            filter = BandPassFilter(
                tr=filter_config['tr'],
                flp=filter_config['flp'],
                fhi=filter_config['fhi'],
                k=filter_config['k'],
                remove_artifacts=filter_config['remove_artifacts'],
                apply_demean=filter_config['apply_demean'],
                apply_detrend=filter_config['apply_detrend'],
                apply_finalDetrend=filter_config['apply_finalDetrend'],
            )
        else:
            filter = None

        data_loader = load_adni(
            data_dir=data_dir,
            filter=filter,
        )
        CACHED_ADNI = data_loader

        # Print dataset information
        print(f"\nDataset: ADNI-B")
        print(f"Number of ROIs: {data_loader.N()}")
        print(f"TR: {data_loader.TR()} seconds")
        print(f"\nSubject counts:")
        counts = data_loader.get_subject_count()
        for group, count in counts.items():
            print(f"  {group}: {count}")
    else:
        data_loader = CACHED_ADNI

    # Prepare PyTorch DataLoaders
    print(f"\nPreparing PyTorch DataLoaders...")
    split_mode = data_config['data'].get('datasplit_mode', 'none')
    datasplit_file = data_config['data'].get('datasplit_file')
    loaders = prepare_data_loaders(
        data_loader,
        batch_size=data_config['data'].get('batch_size', 16),
        transpose=data_config['data'].get('transpose', False),
        flatten=data_config['data'].get('flatten', False),
        normalize=data_config['data'].get('normalize', False),
        pad_features=data_config['data'].get('pad_features', False),
        truncate_features=data_config['data'].get('truncate_features', False),
        train_split=data_config['data'].get('train_split', 0.7),
        val_split=data_config['data'].get('val_split', 0.15),
        random_seed=data_config['data'].get('random_seed', 42),
        train_groups=data_config['data'].get('groups', ["HC", "MCI", "AD"]),
        timepoints_as_samples=data_config['data'].get('timepoints_as_samples', False),
        fc_input=data_config['data'].get('fc_input', False),
        split_mode=split_mode,
        datasplit_file=datasplit_file
    )

    print(f"\nDataLoader information:")
    print(f"  Input dimension: {loaders['input_dim']}")
    print(f"  Number of samples:")
    for split, num in loaders['num_samples'].items():
        print(f"    {split}: {num}")

    return loaders


def _build_training_summary(history, mse_pca):
    def _metric_values(split, metric):
        split_metrics = history.get(split)
        if isinstance(split_metrics, dict):
            values = split_metrics.get(metric, [])
            return values if isinstance(values, list) else []
        values = history.get(f"{split}_{metric}", [])
        return values if isinstance(values, list) else []

    val_losses = _metric_values("val", "loss")
    train_losses = _metric_values("train", "loss")
    best_epoch = None
    best_val = None
    significance = None
    if val_losses:
        best_index = min(range(len(val_losses)), key=lambda idx: val_losses[idx])
        best_epoch = best_index + 1
        best_val = float(val_losses[best_index])
        bvl = best_val if 'recon' not in history['val'] else float(history['val']['recon'][best_index])
        significance = 100 * (mse_pca - bvl) / mse_pca

    return {
        'num_epochs': max(len(train_losses), len(val_losses)),
        'best_epoch': best_epoch,
        'val_pca_mse': mse_pca,
        'best_val_loss': best_val,
        'significance': significance,
        'final_train_loss': float(train_losses[-1]) if train_losses else None,
        'final_val_loss': float(val_losses[-1]) if val_losses else None,
    }


def load_model_from_config(model_config, input_dim, device):
    model_name = model_config['model']['name']
    latent_dim = 0

    if model_name == "BasicVAE":
        from .models.basic import BasicVAE
        hidden_dims = model_config['model'].get('hidden_dims', [1024, 512, 256, 128])
        hidden_dims = [int(i) for i in hidden_dims]
        latent_dim = int(model_config['model'].get('latent_dim', 32))
        model = BasicVAE(
            input_dim=input_dim[0],
            hidden_dims=hidden_dims,
            latent_dim=latent_dim,
            device=device)
    elif model_name == "AutoencoderKL":
        from .models.monaiAEKL import AutoencoderKL
        latent_dim = model_config['model'].get('latent_channels', 8)
        channels = model_config['model'].get('channels', [64, 128, 256])
        attention_levels = model_config['model'].get('attention_levels', [False] * len(channels))
        model = AutoencoderKL(
            spatial_dims=1,
            in_channels=input_dim[0],
            out_channels=input_dim[0],
            num_res_blocks=model_config['model'].get('num_res_blocks', 1),
            channels=channels,
            attention_levels=attention_levels,
            latent_channels=latent_dim,
            norm_num_groups=model_config['model'].get('norm_num_groups', 32),
            norm_eps=model_config['model'].get('norm_eps', 1e-6),
            with_encoder_nonlocal_attn=False,
            with_decoder_nonlocal_attn=False,
        )
    elif model_name == "DeterministicAE":
        from .models.determAE import DeterministicAE
        latent_dim = model_config['model'].get('latent_dim', 32)
        model = DeterministicAE(
            input_dim=input_dim[0],
            hidden_dims=model_config['model'].get('hidden_dims', [1024, 256]),
            latent_dim=latent_dim,
            dropout=model_config['model'].get('dropout', 0.0),
            use_batchnorm=model_config['model'].get('use_batchnorm', False),
            final_activation=model_config['model'].get('final_activation', None),
        )
    elif model_name == "Perl2023":
        from .models.perl2023 import Perl2023
        latent_dim = model_config['model'].get('latent_dim', 2)
        model = Perl2023(
            input_dim=input_dim[0],
            intermediate_dim=model_config['model'].get('intermediate_dim', 1028),
            latent_dim=latent_dim,
            output_activation=model_config['model'].get('output_activation', 'sigmoid'),
        )
    elif model_name == "SequentialAE":
        from .models.seqAE import SequentialAE
        latent_dim = model_config['model'].get('latent_dim', 2)
        model = SequentialAE(
            regions=input_dim[1],
            hidden_dim=model_config['model'].get('hidden_dim', 256),
            latent_dim=latent_dim,
            num_layers=model_config['model'].get('num_layers', 1),
            dropout=model_config['model'].get('dropout', 0.0),
            cell=model_config['model'].get('cell', 'lstm')
        )
    elif model_name == "LinearAE":
        from .models.linear import LinearAE
        latent_dim = model_config['model'].get('latent_dim', 2)
        model = LinearAE(
            input_dim=input_dim[0],
            latent_dim=latent_dim,
        )
    else:
        raise ValueError(f"Model name {model_name} not supported")

    torch_device = torch.device(device)
    if "load_path" in model_config['model']:
        model.load_state_dict(torch.load(model_config['model']['load_path'], map_location=torch_device))
        print(f"Model loaded from {model_config['model']['load_path']}")
        if model_config['model']['freeze_encoder']:
            model.freeze_encoder()
        if model_config['model']['reset_decoder']:
            model.reset_decoder()

    model = model.to(torch_device)
    return model, model_name, latent_dim


def get_most_recent_experiment_id(index_path):
    latest_entry = None
    latest_created_at = ""
    with open(index_path, "r") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            created_at = entry.get("created_at", "")
            if created_at >= latest_created_at:
                latest_created_at = created_at
                latest_entry = entry

    if latest_entry is None or "experiment_id" not in latest_entry:
        raise ValueError(f"No experiment entries found in {index_path}")
    return latest_entry["experiment_id"]


def run_training(model, model_name, latent_dim, loaders, training_config, model_config, data_config, device):
    from .train import train_vae

    model.set_loss_fn_params(training_config['training'].get('loss_params', None))

    tracker = TrainingResultsManager(results_dir=project_path / "results")
    experiment_id = tracker.build_experiment_id(
        model_type=model_name,
        model_params=model_config.get('model', {}),
        training_params=training_config.get('training', {}),
        data_params=data_config,
    )
    print(f"Experiment ID: {experiment_id}")

    # PCA for validation fit on the training data
    pca = PCA(latent_dim)
    if len(loaders['train_loader'].dataset.data.shape) > 2:
        pca.fit(loaders['train_loader'].dataset.data.reshape(loaders['train_loader'].dataset.data.shape[0], -1))
    else:
        pca.fit(loaders['train_loader'].dataset.data)

    pathlib.Path(training_config['training']['save_dir']).mkdir(parents=True, exist_ok=True)
    history, mse_pca = train_vae(
        model,
        loaders['train_loader'],
        loaders['val_loader'],
        num_epochs=training_config['training'].get('num_epochs', 50),
        learning_rate=training_config['training'].get('learning_rate', 1e-3),
        weight_decay=training_config['training'].get('weight_decay', 1e-4),
        device=device,
        save_dir=training_config['training']['save_dir'],
        name=experiment_id,
        pca=pca,
        noise=training_config['training'].get("noise", None)
    )
    model_artifact = pathlib.Path(training_config['training']['save_dir']) / f"{experiment_id}_model.pt"
    experiment_metadata = {
        'experiment_id': experiment_id,
        'status': 'completed',
        'model_type': model_name,
        'summary': _build_training_summary(history, mse_pca),
        'model_params': deepcopy(model_config.get('model', {})),
        'training_params': deepcopy(training_config.get('training', {})),
        'data_params': deepcopy(data_config),
        'artifacts': {
            'model_path': str(model_artifact),
        },
    }
    tracked_experiment_id = tracker.register_experiment(
        metadata=experiment_metadata,
        history=history,
    )
    print(f'Tracked training experiment: {tracked_experiment_id}')
    return experiment_metadata['experiment_id']


def run_evaluation(model, latent_dim, loaders, training_config, device, experiment_id=None):
    from .eval import eval_vae

    # Fit PCA on training data and pass it into evaluation for baseline comparison.
    pca = PCA(latent_dim)
    train_data = loaders['train_loader'].dataset.data
    if len(train_data.shape) > 2:
        pca.fit(train_data.reshape(train_data.shape[0], -1))
    else:
        pca.fit(train_data)

    target_experiment_id = experiment_id or get_most_recent_experiment_id(project_path / "results" / "index.jsonl")
    print(f"Evaluating experiment: {target_experiment_id}")
    model_path = pathlib.Path(training_config['training']['save_dir']) / f"{target_experiment_id}_model.pt"
    torch_device = torch.device(device)
    model.load_state_dict(torch.load(model_path, map_location=torch_device))
    model = model.to(torch_device)

    eval_metrics = eval_vae(
        model,
        loaders['test_loader'],
        pca=pca,
        device=device,
    )

    tracker = TrainingResultsManager(results_dir=project_path / "results")
    tracker.set_evaluation_metrics(
        experiment_id=target_experiment_id,
        model_metrics=eval_metrics.get("model", {}),
        pca_metrics=eval_metrics.get("pca"),
    )
    print(f"Stored evaluation metrics for experiment: {target_experiment_id}")
    return eval_metrics


def main():
    """Main function for training and inference."""
    parser = argparse.ArgumentParser(description='Train VAE models or run inference on ADNI-B data')
    parser.add_argument(
        '-m', '--mode',
        type=str,
        default='train',
        choices=['train', 'eval', 'load', 'exp'],
        help='Mode: train, eval, or just load data (default: train)'
    )
    parser.add_argument(
        '-d', '--data-dir',
        type=pathlib.Path,
        default=project_path / "data",
        help='Path to data directory (default: ./data)'
    )
    parser.add_argument(
        '--data-config',
        type=pathlib.Path,
        default=project_path / "config" / "data.yml",
        help='Path to data configuration file (default: ./config/data.yml)'
    )
    parser.add_argument(
        '--model-config',
        type=pathlib.Path,
        default=project_path / "config" / "model.yml",
        help='Path to model configuration file (default: ./config/model.yml)'
    )
    parser.add_argument(
        '--training-config',
        type=pathlib.Path,
        default=project_path / "config" / "training.yml",
        help='Path to training configuration file (default: ./config/training.yml)'
    )
    parser.add_argument(
        '--experiment-config',
        type=pathlib.Path,
        default=project_path / "config" / "experiments.yml",
        help='Path to training configuration file (default: ./config/experiments.yml)'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='mps',
        choices=['cpu', 'cuda', 'mps'],
        help='Device to use for training and inference (default: mps)'
    )
    parser.add_argument(
        '--exp-name',
        type=str,
        help='Name of the experiment, used only in evaluation. If not provided, it uses the last trained model.'
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("ADNI-B VAE")
    print("=" * 60)
    # Mode-specific actions
    if args.mode == 'load':
        # Just load and display data
        data_config = load_config(args.data_config)
        loaders = load_data_from_config(
            data_dir=args.data_dir,
            data_config=data_config,
        )
        print(f"\nTesting data loading...")
        sample_batch = next(iter(loaders['train_loader']))
        if isinstance(sample_batch, tuple) or isinstance(sample_batch, list):
            batch_data, batch_labels = sample_batch
            print(f"  Batch shape: {batch_data.shape}")
            print(f"  Batch labels: {batch_labels[:5]}...")  # Show first 5 labels
        else:
            breakpoint()
            print(f"  Batch shape: {sample_batch.shape}")
        
        print("\n" + "=" * 60)
        print("Data loading complete!")
        print("=" * 60)
        
    elif args.mode == 'train':
        print("\n" + "=" * 60)
        print("Training mode")
        print("=" * 60)

        data_config = load_config(args.data_config)
        loaders = load_data_from_config(
            data_dir=args.data_dir,
            data_config=data_config,
        )

        input_dim = loaders['input_dim']

        model_config = load_config(args.model_config)
        model, model_name, latent_dim = load_model_from_config(
            model_config=model_config,
            input_dim=input_dim,
            device=args.device,
        )
        training_config = load_config(args.training_config)
        run_training(
            model,
            model_name,
            latent_dim,
            loaders,
            training_config,
            model_config,
            data_config,
            device=args.device,
        )
        print("=" * 60)
        
    elif args.mode == 'eval':
        print("\n" + "=" * 60)
        print("Evaluation mode")
        print("=" * 60)

        data_config = load_config(args.data_config)
        loaders = load_data_from_config(
            data_dir=args.data_dir,
            data_config=data_config,
        )

        input_dim = loaders['input_dim']
        model_config = load_config(args.model_config)
        model, model_name, latent_dim = load_model_from_config(
            model_config=model_config,
            input_dim=input_dim,
            device=args.device,
        )
        training_config = load_config(args.training_config)
        run_evaluation(
            model,
            latent_dim,
            loaders,
            training_config,
            device=args.device,
            experiment_id=args.exp_name,
        )
    elif args.mode == 'exp':
        print("\n" + "=" * 60)
        print("Experimentation mode")
        print("=" * 60)
        exp_config = load_config(args.experiment_config)

        def collect_vars(node, name):
            if isinstance(node, dict):
                new_node = {}
                for k,v in node.items():
                    new_node.update(collect_vars(v, f"{name}.{k}" if name != '' else k))
                return new_node
            else:
                return {name: node}
            
        def set_var_value(node, name, value):
            if '.' in name and isinstance(node, dict) and name.split('.')[0] in node:
                # breakpoint()
                node_name = name.split('.')[0]
                ch_name = '.'.join(name.split('.')[1:])
                set_var_value(node[node_name], ch_name, value)
            elif isinstance(node, dict) and '.' not in name:
                if isinstance(value, dict) and name in node and isinstance(node[name], dict):
                    node[name].update(value)
                else:
                    node[name] = value

        for set_name, ec in exp_config.items():
            if set_name == 'default':
                continue
            print(f"Setting up experiments for {set_name}")
            data_config = deepcopy(exp_config['default']['data'])
            model_config = deepcopy(exp_config['default']['model'])
            training_config = deepcopy(exp_config['default']['training'])
            # overwrite the configurations with static parameters
            if 'data' in ec['static_params']:
                data_config.update(ec['static_params']['data'])
            if 'model' in ec['static_params']:
                model_config.update(ec['static_params']['model'])
            if 'training' in ec['static_params']:
                training_config.update(ec['static_params']['training'])

            # collect all experiment variables
            vars = collect_vars(ec['exp_params'],'')
            keys, values = zip(*vars.items())
            permutations_dicts = [dict(zip(keys, v)) for v in itertools.product(*values)]
            for collection in permutations_dicts:
                dc = deepcopy(data_config)
                mc = {'model':deepcopy(model_config)}
                tc = {'training':deepcopy(training_config)}
                for vn, var in collection.items():
                    set_var_value({'data':dc}, vn, var)
                    set_var_value(mc, vn, var)
                    set_var_value(tc, vn, var)

                # load data
                loaders = load_data_from_config(
                    data_dir=args.data_dir,
                    data_config=dc
                )
                input_dim = loaders['input_dim']
                model, model_name, latent_dim = load_model_from_config(
                    model_config=mc,
                    input_dim=input_dim,
                    device=args.device,
                )
                exp_id = run_training(
                    model,
                    model_name,
                    latent_dim,
                    loaders,
                    tc,
                    mc,
                    dc,
                    device=args.device,
                )
                run_evaluation(
                    model,
                    latent_dim,
                    loaders,
                    tc,
                    device=args.device,
                    experiment_id=exp_id,
                )




if __name__ == '__main__':
    main()
