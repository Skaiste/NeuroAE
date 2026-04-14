"""
Main script for training VAE models and running inference on ADNI-B data.

This script loads ADNI-B data and can be used for training models or running inference.
"""

import argparse
import concurrent.futures
import hashlib
import json
import math
import os
import pathlib
import random
import re
import numpy as np
import torch
import yaml
import itertools
from copy import deepcopy
from neuronumba.tools.filters import BandPassFilter
from sklearn.preprocessing import StandardScaler

from .utils import *
from .utils.dict_utils import deepupdate
from .load_data import load_adni, prepare_data_loaders
from .models.archive.pca import PCA, PCA_multi
from training_tracker import TrainingResultsManager

CACHED_ADNI = None
CACHED_FILTER = None


def _get_reproducibility_settings(data_config=None, training_config=None):
    data_config = data_config or {}
    training_config = training_config or {}
    training_section = training_config.get("training", {})
    reproducibility = training_section.get("reproducibility", {})
    data_seed = data_config.get("data", {}).get("random_seed", 42)
    return {
        "enabled": bool(reproducibility.get("enabled", True)),
        "seed": int(reproducibility.get("seed", data_seed)),
        "deterministic": bool(reproducibility.get("deterministic", True)),
        "warn_only": bool(reproducibility.get("warn_only", False)),
        "benchmark": bool(reproducibility.get("benchmark", False)),
    }


def configure_reproducibility(data_config=None, training_config=None):
    settings = _get_reproducibility_settings(data_config, training_config)
    if not settings["enabled"]:
        return settings

    seed = settings["seed"]
    deterministic = settings["deterministic"]
    warn_only = settings["warn_only"]
    benchmark = settings["benchmark"]

    os.environ["PYTHONHASHSEED"] = str(seed)
    if deterministic and torch.cuda.is_available():
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.use_deterministic_algorithms(deterministic, warn_only=warn_only)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = deterministic
        torch.backends.cudnn.benchmark = benchmark
    if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
        torch.backends.cuda.matmul.allow_tf32 = not deterministic
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.allow_tf32 = not deterministic

    print(
        "Reproducibility settings: "
        f"seed={seed}, deterministic={deterministic}, "
        f"warn_only={warn_only}, benchmark={benchmark}"
    )
    return settings


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


def load_data_from_config(data_dir, data_config, num_workers=0):
    global CACHED_ADNI
    if CACHED_ADNI is None:
        # Load ADNI data
        print(f"\nLoading ADNI-B dataset...")
        data_loader = load_adni(data_dir=data_dir)
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

    # setup filter
    global CACHED_FILTER
    if CACHED_FILTER is None and 'filter' in data_config and data_config['filter']['type'] == 'BandPassFilter':
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
        CACHED_FILTER = filter
    elif CACHED_FILTER is not None:
        filter = CACHED_FILTER
    else:
        filter = None


    # Prepare PyTorch DataLoaders
    print(f"\nPreparing PyTorch DataLoaders...")
    split_mode = data_config['data'].get('datasplit_mode', 'none')
    datasplit_file = data_config['data'].get('datasplit_file')
    # setup normaliser
    normaliser = None
    if data_config['data'].get('normalize', False):
        normaliser = StandardScaler()
    loaders = prepare_data_loaders(
        data_loader,
        batch_size=data_config['data'].get('batch_size', 16),
        num_workers=num_workers,
        transpose=data_config['data'].get('transpose', False),
        flatten=data_config['data'].get('flatten', False),
        pad_features=data_config['data'].get('pad_features', False),
        truncate_features=data_config['data'].get('truncate_features', False),
        train_split=data_config['data'].get('train_split', 0.7),
        val_split=data_config['data'].get('val_split', 0.15),
        random_seed=data_config['data'].get('random_seed', 42),
        train_groups=data_config['data'].get('groups', ["HC", "MCI", "AD"]),
        timepoints_as_samples=data_config['data'].get('timepoints_as_samples', False),
        fc_input=data_config['data'].get('fc_input', False),
        preserve_timepoints=data_config['data'].get('preserve_timepoints', False),
        split_mode=split_mode,
        datasplit_file=datasplit_file,
        filter=filter,
        normaliser=normaliser,
        use_bio_levels=data_config['data'].get('use_bio_levels', []),
    )

    print(f"\nDataLoader information:")
    print(f"  Input dimension: {loaders['input_dim']}")
    print(f"  Number of samples:")
    for split, num in loaders['num_samples'].items():
        print(f"    {split}: {num}")

    return loaders


def _build_training_summary(history, mse_pca, checkpoint_selection_metric="val_loss"):
    def _metric_values(split, metric):
        split_metrics = history.get(split)
        if isinstance(split_metrics, dict):
            values = split_metrics.get(metric, [])
            return values if isinstance(values, list) else []
        values = history.get(f"{split}_{metric}", [])
        return values if isinstance(values, list) else []

    def _best_finite(values, maximize=True):
        finite_values = [float(v) for v in values if math.isfinite(float(v))]
        if not finite_values:
            return None
        return max(finite_values) if maximize else min(finite_values)

    val_losses = _metric_values("val", "loss")
    val_swfcd = _metric_values("val", "swfcd_pearson")
    val_logreg = _metric_values("val", "logreg_accuracy")
    train_losses = _metric_values("train", "loss")
    selected_epoch = None
    selected_val_loss = None
    selected_swfcd_pearson = None
    selected_logreg_accuracy = None
    selected_joint_score = None
    best_val = None
    significance = None
    from .train import select_best_checkpoint
    selection = select_best_checkpoint(history, selection_metric=checkpoint_selection_metric)
    if selection is not None:
        selected_epoch = selection["best_epoch"]
        selected_val_loss = selection["loss"]
        selected_swfcd_pearson = selection["swfcd_pearson"]
        selected_logreg_accuracy = selection["logreg_accuracy"]
        selected_joint_score = selection.get("joint_score")

    if val_losses:
        best_index = min(range(len(val_losses)), key=lambda idx: val_losses[idx])
        best_val = float(val_losses[best_index])
        selected_index = selection["best_index"] if selection is not None else best_index
        bvl = best_val if 'recon' not in history['val'] else float(history['val']['recon'][selected_index])
        if mse_pca is not None and math.isfinite(float(mse_pca)) and float(mse_pca) != 0.0:
            significance = 100 * (mse_pca - bvl) / mse_pca
        else:
            significance = None

    return {
        'num_epochs': max(len(train_losses), len(val_losses)),
        'best_epoch': selected_epoch,
        'selected_val_loss': selected_val_loss,
        'selected_swfcd_pearson': selected_swfcd_pearson,
        'selected_logreg_accuracy': selected_logreg_accuracy,
        'selected_joint_score': selected_joint_score,
        'checkpoint_selection_metric': checkpoint_selection_metric,
        'val_pca_mse': mse_pca,
        'best_val_loss': best_val,
        'best_val_swfcd_pearson': _best_finite(val_swfcd, maximize=True),
        'best_val_logreg_accuracy': _best_finite(val_logreg, maximize=True),
        'significance': significance,
        'final_train_loss': float(train_losses[-1]) if train_losses else None,
        'final_val_loss': float(val_losses[-1]) if val_losses else None,
    }


def load_model_from_config(model_config, data_config, input_dim, timepoint_dim, device, preserve_timepoints=False):
    model_name = model_config['model']['name']
    latent_dim = 0

    if model_name == "VAE":
        from .models.variational import VAE
        hidden_dim = model_config['model']['hidden_dims']
        latent_dim = model_config['model']['latent_dim']
        # assuming the input is transposed
        model = VAE(
            region_dim=input_dim[-1],
            timepoint_dim=input_dim[0],
            hidden_dims=hidden_dim,
            latent_dim=latent_dim,
            device=device)
    elif model_name == "LAE":
        from .models.linear import LAE
        latent_dim = model_config['model']['latent_dim']
        hidden_dim = None
        model = LAE(
            region_dim=input_dim[-1],
            timepoint_dim=input_dim[0],
            latent_dim=latent_dim,
        )
    elif model_name == "ConvAE":
        from .models.convAE import ConvAE
        hidden_dim = model_config['model']['hidden_dims']
        latent_dim = model_config['model']['latent_dim']
        kernel_size = model_config['model'].get("kernel_size", 3)
        model = ConvAE(
            regions=input_dim[-1],
            timepoints=input_dim[0],
            latent_dim=latent_dim,
            hidden_channels=hidden_dim,
            kernel_size=kernel_size
        )
    elif model_name == "ConvVAE":
        from .models.convAE import ConvVAE
        hidden_dim = model_config['model']['hidden_dims']
        latent_dim = model_config['model']['latent_dim']
        kernel_size = model_config['model'].get("kernel_size", 3)
        model = ConvVAE(
            regions=input_dim[-1],
            timepoints=input_dim[0],
            latent_dim=latent_dim,
            hidden_channels=hidden_dim,
            kernel_size=kernel_size
        )
    elif model_name == "MonaiAEKL":
        from .models.monaiAE import MonaiAEKL
        hidden_dim = model_config['model']['hidden_dims']
        latent_dim = model_config['model']['latent_dim']
        kernel_size = model_config['model'].get("kernel_size", 3)
        num_res_blocks = model_config['model'].get("num_res_blocks", 1)
        attention_levels = model_config['model'].get("attention_levels")
        latent_channels = model_config['model'].get("latent_channels", 3)
        norm_num_groups = model_config['model'].get("norm_num_groups")
        use_checkpoint = model_config['model'].get("use_checkpoint", False)
        use_convtranspose = model_config['model'].get("use_convtranspose", False)
        with_encoder_nonlocal_attn = model_config['model'].get("with_encoder_nonlocal_attn", True)
        with_decoder_nonlocal_attn = model_config['model'].get("with_decoder_nonlocal_attn", True)
        include_fc = model_config['model'].get("include_fc", True)
        use_combined_linear = model_config['model'].get("use_combined_linear", False)
        use_flash_attention = model_config['model'].get("use_flash_attention", False)
        model = MonaiAEKL(
            regions=input_dim[-1],
            timepoints=input_dim[0],
            latent_dim=latent_dim,
            hidden_channels=hidden_dim,
            kernel_size=kernel_size,
            num_res_blocks=num_res_blocks,
            attention_levels=attention_levels,
            latent_channels=latent_channels,
            norm_num_groups=norm_num_groups,
            with_encoder_nonlocal_attn=with_encoder_nonlocal_attn,
            with_decoder_nonlocal_attn=with_decoder_nonlocal_attn,
            use_checkpoint=use_checkpoint,
            use_convtranspose=use_convtranspose,
            include_fc=include_fc,
            use_combined_linear=use_combined_linear,
            use_flash_attention=use_flash_attention,
            device=device,
        )
    else:
        raise ValueError(f"Model name {model_name} not supported")
    
    print("\n"+"="*60+"\n")
    print(f"Model to train: {model_name}")
    print(f"Parameters: {hidden_dim=}, {latent_dim=}\n")

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


def build_experiment_signature(model_type, model_params, training_params, data_params):
    canonical = {
        "model_type": model_type,
        "model_params": model_params,
        "training_params": training_params,
        "data_params": data_params,
    }
    payload = json.dumps(canonical, sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:8]


def _dedupe_param_choices(values):
    """De-duplicate list-like parameter choices while preserving input order."""
    unique = []
    seen = set()
    for value in values:
        marker = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
        if marker in seen:
            continue
        seen.add(marker)
        unique.append(value)
    return unique


def _has_evaluation_results(metadata):
    evaluation = metadata.get("evaluation")
    if not isinstance(evaluation, dict):
        return False
    model_metrics = evaluation.get("model")
    return isinstance(model_metrics, dict) and len(model_metrics) > 0


def load_completed_experiment_signatures(results_dir):
    print("Loading already existing experiments")
    tracker = TrainingResultsManager(results_dir=results_dir)
    completed_signatures = set()
    for entry in tracker.list_experiments():
        experiment_id = entry.get("experiment_id")
        if not experiment_id:
            continue
        try:
            metadata = tracker.get_experiment(experiment_id)
        except FileNotFoundError:
            continue

        if not _has_evaluation_results(metadata):
            continue

        signature = build_experiment_signature(
            model_type=metadata.get("model_type", "unknown"),
            model_params=metadata.get("model_params", {}),
            training_params=metadata.get("training_params", {}),
            data_params=metadata.get("data_params", {}),
        )
        completed_signatures.add(signature)
    return completed_signatures


def run_training(model, model_name, latent_dim, loaders, training_config, model_config, data_config, device):
    from .train import train_vae

    use_pred_heads = len(data_config['data'].get('use_bio_levels', [])) > 0

    model.set_loss_fn_params(training_config['training'].get('loss_params', None))
    if training_config['training']['loss_params'].get("swfcd_beta", 0.0) > 0:
        from .metrics.swfcd_torch import SwFCD
        window = training_config['training']['loss_params'].get("swfcd_window", 30)
        step = training_config['training']['loss_params'].get("swfcd_step", 3)
        swfcd = SwFCD(loaders['train_loader'].dataset, window, step)
        model.set_swfcd(swfcd)

    tracker = TrainingResultsManager(results_dir=project_path / "results")
    experiment_id = tracker.build_experiment_id(
        model_type=model_name,
        model_params=model_config.get('model', {}),
        training_params=training_config.get('training', {}),
        data_params=data_config,
    )
    print(f"Experiment ID: {experiment_id}")

    reproducibility = _get_reproducibility_settings(data_config, training_config)
    pca_seed = reproducibility["seed"] if reproducibility["enabled"] else None

    # PCA for validation fit on the training data
    # if loaders['preserve_timepoints']:
    #     pca = PCA_multi(loaders['train_loader'].dataset, latent_dim, random_state=pca_seed)
    # else:
    #     pca = PCA(loaders['train_loader'].dataset, latent_dim, random_state=pca_seed)
    # pca.fit(loaders['train_loader'].dataset.data)
    pca = None

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
        noise=training_config['training'].get("noise", None),
        use_pred_heads=use_pred_heads,
        convergence_patience=training_config['training'].get('convergence_patience'),
        convergence_min_delta=training_config['training'].get('convergence_min_delta', 0.0),
        convergence_warmup_epochs=training_config['training'].get('convergence_warmup_epochs', 0),
        checkpoint_selection_metric=training_config['training'].get('checkpoint_selection_metric', 'val_loss'),
    )
    model_artifact = pathlib.Path(training_config['training']['save_dir']) / f"{experiment_id}_model.pt"
    experiment_metadata = {
        'experiment_id': experiment_id,
        'status': 'completed',
        'model_type': model_name,
        'summary': _build_training_summary(
            history,
            mse_pca,
            checkpoint_selection_metric=training_config['training'].get('checkpoint_selection_metric', 'val_loss'),
        ),
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


def run_evaluation(
    model,
    latent_dim,
    loaders,
    training_config,
    data_config,
    device,
    experiment_id=None,
    delete_model_after_eval=True,
):
    from .eval import eval_vae

    reproducibility = _get_reproducibility_settings(data_config, training_config)
    pca_seed = reproducibility["seed"] if reproducibility["enabled"] else None

    # Fit PCA on training data and pass it into evaluation for baseline comparison.
    # if loaders['preserve_timepoints']:
    #     pca = PCA_multi(loaders['train_loader'].dataset, latent_dim, random_state=pca_seed)
    # else:
    #     pca = PCA(loaders['train_loader'].dataset, latent_dim, random_state=pca_seed)
    # pca.fit(loaders['train_loader'].dataset.data)
    pca = None

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
    if delete_model_after_eval:
        if model_path.exists():
            model_path.unlink()
            print(f"Deleted model artifact after evaluation: {model_path}")
        else:
            print(f"Model artifact already missing, nothing to delete: {model_path}")
    return eval_metrics


def run_experiment_pipeline(
    data_dir,
    device,
    data_config,
    model_config,
    training_config,
    delete_model_after_eval=True,
    num_workers=0,
):
    configure_reproducibility(data_config=data_config, training_config=training_config)
    loaders = load_data_from_config(
        data_dir=data_dir,
        data_config=data_config,
        num_workers=num_workers,
    )
    input_dim = loaders['input_dim']
    timepoint_dim = loaders['timepoint_dim']
    model, model_name, latent_dim = load_model_from_config(
        model_config=model_config,
        data_config=data_config,
        input_dim=input_dim,
        timepoint_dim=timepoint_dim,
        device=device,
        preserve_timepoints=loaders.get('preserve_timepoints', False)
    )
    exp_id = run_training(
        model,
        model_name,
        latent_dim,
        loaders,
        training_config,
        model_config,
        data_config,
        device=device,
    )
    run_evaluation(
        model,
        latent_dim,
        loaders,
        training_config,
        data_config,
        device=device,
        experiment_id=exp_id,
        delete_model_after_eval=delete_model_after_eval,
    )
    return exp_id


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
    parser.add_argument(
        '--num-parallel-experiments',
        type=int,
        default=1,
        help='Maximum number of experiments to run concurrently in exp mode (default: 1).'
    )
    parser.add_argument(
        '--delete-model-after-eval',
        action=argparse.BooleanOptionalAction,
        default=True,
        help='Delete model artifact after evaluation completes (default: true). Use --no-delete-model-after-eval to keep it.'
    )
    parser.add_argument(
        '--num-workers',
        type=int,
        default=0,
        help='Number of DataLoader worker processes (default: 0).'
    )
    parser.add_argument(
        '--max-experiment-combinations',
        type=int,
        default=100000,
        help='Safety cap for Cartesian combinations per experiment set in exp mode (default: 100000).'
    )
    
    args = parser.parse_args()
    if args.num_parallel_experiments < 1:
        parser.error('--num-parallel-experiments must be >= 1')
    if args.num_workers < 0:
        parser.error('--num-workers must be >= 0')
    if args.max_experiment_combinations < 1:
        parser.error('--max-experiment-combinations must be >= 1')
    
    print("=" * 60)
    print("ADNI-B VAE")
    print("=" * 60)
    # Mode-specific actions
    if args.mode == 'load':
        # Just load and display data
        data_config = load_config(args.data_config)
        configure_reproducibility(data_config=data_config, training_config={})
        loaders = load_data_from_config(
            data_dir=args.data_dir,
            data_config=data_config,
            num_workers=args.num_workers,
        )
        print(f"\nTesting data loading...")
        sample_batch = next(iter(loaders['train_loader']))
        if isinstance(sample_batch, tuple) or isinstance(sample_batch, list):
            batch_data, batch_labels = sample_batch
            print(f"  Batch shape: {batch_data.shape}")
            print(f"  Batch labels: {batch_labels[:5]}...")  # Show first 5 labels
        else:
            print(f"  Batch shape: {sample_batch.shape}")
        
        print("\n" + "=" * 60)
        print("Data loading complete!")
        print("=" * 60)
        
    elif args.mode == 'train':
        print("\n" + "=" * 60)
        print("Training mode")
        print("=" * 60)

        data_config = load_config(args.data_config)
        model_config = load_config(args.model_config)
        training_config = load_config(args.training_config)
        configure_reproducibility(data_config=data_config, training_config=training_config)
        
        loaders = load_data_from_config(
            data_dir=args.data_dir,
            data_config=data_config,
            num_workers=args.num_workers,
        )
        input_dim = loaders['input_dim']
        timepoint_dim = loaders['timepoint_dim']
        
        model, model_name, latent_dim = load_model_from_config(
            model_config=model_config,
            data_config=data_config,
            input_dim=input_dim,
            timepoint_dim=timepoint_dim,
            device=args.device,
            preserve_timepoints=loaders.get('preserve_timepoints', False)
        )

        run_training(
            model,
            model_name,
            latent_dim,
            loaders,
            training_config,
            model_config,
            data_config,
            device=args.device
        )
        print("=" * 60)
        
    elif args.mode == 'eval':
        print("\n" + "=" * 60)
        print("Evaluation mode")
        print("=" * 60)

        data_config = load_config(args.data_config)
        training_config = load_config(args.training_config)
        configure_reproducibility(data_config=data_config, training_config=training_config)
        loaders = load_data_from_config(
            data_dir=args.data_dir,
            data_config=data_config,
            num_workers=args.num_workers,
        )

        input_dim = loaders['input_dim']
        timepoint_dim = loaders['timepoint_dim']
        model_config = load_config(args.model_config)
        model, model_name, latent_dim = load_model_from_config(
            model_config=model_config,
            data_config=data_config,
            input_dim=input_dim,
            timepoint_dim=timepoint_dim,
            device=args.device,
            preserve_timepoints=loaders.get('preserve_timepoints', False)
        )
        run_evaluation(
            model,
            latent_dim,
            loaders,
            training_config,
            data_config,
            device=args.device,
            experiment_id=args.exp_name,
            delete_model_after_eval=args.delete_model_after_eval,
        )
    elif args.mode == 'exp':
        print("\n" + "=" * 60)
        print("Experimentation mode")
        print("=" * 60)
        print(f"Using experiment config: {args.experiment_config}")
        exp_config = load_config(args.experiment_config)

        def collect_vars(node, name):
            if isinstance(node, dict):
                new_node = {}
                for k,v in node.items():
                    deepupdate(new_node, collect_vars(v, f"{name}.{k}" if name != '' else k))
                return new_node
            else:
                return {name: node}
            
        def set_var_value(node, name, value):
            if '.' in name and isinstance(node, dict) and name.split('.')[0] in node:
                node_name = name.split('.')[0]
                ch_name = '.'.join(name.split('.')[1:])
                set_var_value(node[node_name], ch_name, value)
            elif isinstance(node, dict) and '.' not in name:
                if isinstance(value, dict) and name in node and isinstance(node[name], dict):
                    deepupdate(node[name], value)
                else:
                    node[name] = value

        experiment_specs = []
        completed_signatures = load_completed_experiment_signatures(project_path / "results")
        seen_signatures = set()
        skipped_completed = 0
        skipped_duplicates = 0
        for set_name, ec in exp_config.items():
            if set_name == 'default':
                continue
            print(f"Setting up experiments for {set_name}")
            data_config = deepcopy(exp_config['default']['data'])
            model_config = deepcopy(exp_config['default']['model'])
            training_config = deepcopy(exp_config['default']['training'])
            # overwrite the configurations with static parameters
            if 'data' in ec['static_params']:
                deepupdate(data_config, ec['static_params']['data'])
            if 'model' in ec['static_params']:
                deepupdate(model_config, ec['static_params']['model'])
            if 'training' in ec['static_params']:
                deepupdate(training_config, ec['static_params']['training'])

            # Collect all experiment variables and iterate lazily to avoid
            # materializing a potentially large Cartesian product in memory.
            vars = collect_vars(ec['exp_params'],'')
            if vars:
                keys = list(vars.keys())
                values = []
                for key in keys:
                    candidates = vars[key]
                    if isinstance(candidates, (list, tuple)):
                        raw_candidate_list = list(candidates)
                        candidate_list = _dedupe_param_choices(raw_candidate_list)
                        values.append(candidate_list)
                        print(
                            f"    {key}: {len(raw_candidate_list)} choices ({len(candidate_list)} unique)",
                            flush=True,
                        )
                    else:
                        raise ValueError(
                            f"Experiment parameter '{key}' must be a list/tuple; got {type(candidates).__name__}"
                        )
                total_set_permutations = math.prod(len(v) for v in values)
                if total_set_permutations > args.max_experiment_combinations:
                    raise ValueError(
                        f"{set_name} expands to {total_set_permutations} combinations, "
                        f"which exceeds --max-experiment-combinations={args.max_experiment_combinations}. "
                        "Reduce search space or pass a larger cap explicitly."
                    )
                print(f"  {set_name}: {total_set_permutations} candidate combinations", flush=True)
                permutations_iter = itertools.product(*values)
            else:
                total_set_permutations = 1
                print(f"  {set_name}: 1 candidate combination", flush=True)
                permutations_iter = [()]

            for idx, values_tuple in enumerate(permutations_iter, start=1):
                collection = dict(zip(keys, values_tuple)) if vars else {}
                dc = deepcopy(data_config)
                mc = {'model':deepcopy(model_config)}
                tc = {'training':deepcopy(training_config)}
                for vn, var in collection.items():
                    set_var_value({'data':dc}, vn, var)
                    set_var_value(mc, vn, var)
                    set_var_value(tc, vn, var)

                signature = build_experiment_signature(
                    model_type=mc["model"]["name"],
                    model_params=mc.get("model", {}),
                    training_params=tc.get("training", {}),
                    data_params=dc,
                )
                if signature in completed_signatures:
                    skipped_completed += 1
                    continue
                if signature in seen_signatures:
                    skipped_duplicates += 1
                    continue
                seen_signatures.add(signature)
                experiment_specs.append((dc, mc, tc))
                if idx == 1 or idx == total_set_permutations or idx % 100 == 0:
                    print(f"  {set_name}: processed {idx}/{total_set_permutations}", flush=True)

        total_experiments = len(experiment_specs)
        print(f"Prepared {total_experiments} experiments")
        print(f"Skipped {skipped_completed} experiments with existing evaluation results")
        print(f"Skipped {skipped_duplicates} duplicated experiments in current config")
        if total_experiments == 0:
            print("No experiments were generated from the provided config.")
            return

        max_workers = min(args.num_parallel_experiments, total_experiments)
        if max_workers == 1:
            for i, (dc, mc, tc) in enumerate(experiment_specs, start=1):

                # skip incompatible AutoencoderKL parameters
                if (mc['model']['name'].startswith("AutoencoderKL") and
                        'num_res_blocks' in mc['model'] and
                        'hidden_dims' in mc['model'] and
                        'attention_levels' in mc['model'] and
                        (len(mc['model']['num_res_blocks']) != len(mc['model']['hidden_dims']) or
                         len(mc['model']['num_res_blocks']) != len(mc['model']['attention_levels']))):
                    continue

                print(f"Running experiment {i}/{total_experiments}...")
                run_experiment_pipeline(
                    data_dir=args.data_dir,
                    device=args.device,
                    data_config=dc,
                    model_config=mc,
                    training_config=tc,
                    delete_model_after_eval=args.delete_model_after_eval,
                    num_workers=args.num_workers,
                )
        else:
            if any(_get_reproducibility_settings(dc, tc)["enabled"] for dc, _, tc in experiment_specs):
                raise ValueError(
                    "--num-parallel-experiments > 1 is incompatible with reproducibility-enabled runs "
                    "because thread-based parallelism shares process RNG state. "
                    "Set --num-parallel-experiments=1 for deterministic experiments."
                )
            print(f"Running up to {max_workers} experiments concurrently")
            futures = {}
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                for i, (dc, mc, tc) in enumerate(experiment_specs, start=1):
                    future = executor.submit(
                        run_experiment_pipeline,
                        args.data_dir,
                        args.device,
                        dc,
                        mc,
                        tc,
                        args.delete_model_after_eval,
                        args.num_workers,
                    )
                    futures[future] = i

                failed = []
                for future in concurrent.futures.as_completed(futures):
                    idx = futures[future]
                    try:
                        exp_id = future.result()
                        print(f"Experiment {idx}/{total_experiments} completed: {exp_id}")
                    except Exception as exc:
                        failed.append((idx, exc))
                        print(f"Experiment {idx}/{total_experiments} failed: {exc}")

                if failed:
                    failed_indexes = ", ".join(str(idx) for idx, _ in failed)
                    raise RuntimeError(f"{len(failed)} experiment(s) failed: {failed_indexes}")




if __name__ == '__main__':
    main()
