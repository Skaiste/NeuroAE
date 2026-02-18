"""
Main script for training VAE models and running inference on ADNI-B data.

This script loads ADNI-B data and can be used for training models or running inference.
"""

import argparse
import os
import torch
import pathlib
import yaml
from copy import deepcopy
from neuronumba.tools.filters import BandPassFilter

from .utils import *
from .load_data import load_adni, prepare_data_loaders
from training_tracker import TrainingResultsManager


def load_config(config_path):
    with open(config_path, 'r') as file:
        conf = yaml.load(file, Loader=yaml.FullLoader)
    return conf


def _build_training_summary(history):
    val_losses = history.get('val_loss', [])
    train_losses = history.get('train_loss', [])
    best_epoch = None
    best_val = None
    if val_losses:
        best_index = min(range(len(val_losses)), key=lambda idx: val_losses[idx])
        best_epoch = best_index + 1
        best_val = float(val_losses[best_index])

    return {
        'num_epochs': len(train_losses),
        'best_epoch': best_epoch,
        'best_val_loss': best_val,
        'final_train_loss': float(train_losses[-1]) if train_losses else None,
        'final_val_loss': float(val_losses[-1]) if val_losses else None,
    }

def main():
    """Main function for training and inference."""
    parser = argparse.ArgumentParser(description='Train VAE models or run inference on ADNI-B data')
    parser.add_argument(
        '-m', '--mode',
        type=str,
        default='train',
        choices=['train', 'inference', 'load'],
        help='Mode: train, inference, or just load data (default: train)'
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
        '--device',
        type=str,
        default='mps',
        choices=['cpu', 'cuda', 'mps'],
        help='Device to use for training and inference (default: mps)'
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("ADNI-B VAE Training/Inference")
    print("=" * 60)
    
    # Load ADNI data
    print(f"\nLoading ADNI-B dataset...")
    data_config = load_config(args.data_config)
    if 'filter' in data_config and data_config['filter']['type'] == 'BandPassFilter':
        filter = BandPassFilter(
            tr=float(data_config['filter']['tr']),
            flp=float(data_config['filter']['flp']),
            fhi=float(data_config['filter']['fhi']),
            k=int(data_config['filter']['k']),
            remove_artifacts=data_config['filter']['remove_artifacts'],
            apply_demean=data_config['filter']['apply_demean'],
            apply_detrend=data_config['filter']['apply_detrend'],
            apply_finalDetrend=data_config['filter']['apply_finalDetrend'],
        )
        print(f"Filter configuration found in {args.data_config}")
    else:
        filter = None
        print(f"No filter configuration found in {args.data_config}")
    data_loader = load_adni(
        data_dir=args.data_dir,
        filter=filter,
    )
    
    # Print dataset information
    print(f"\nDataset: ADNI-B")
    print(f"Number of ROIs: {data_loader.N()}")
    print(f"TR: {data_loader.TR()} seconds")
    print(f"\nSubject counts:")
    counts = data_loader.get_subject_count()
    for group, count in counts.items():
        print(f"  {group}: {count}")
    
    # Prepare PyTorch DataLoaders
    print(f"\nPreparing PyTorch DataLoaders...")
    loaders = prepare_data_loaders(
        data_loader,
        batch_size=int(data_config['data']['batch_size']),
        flatten=data_config['data']['flatten'],
        normalize=data_config['data']['normalize'],
        train_split=float(data_config['data']['train_split']),
        val_split=float(data_config['data']['val_split']),
        random_seed=int(data_config['data']['random_seed']),
        train_groups=data_config['data']['groups'], # will use the same for val and test
    )
    
    print(f"\nDataLoader information:")
    print(f"  Input dimension: {loaders['input_dim']}")
    print(f"  Number of samples:")
    for split, num in loaders['num_samples'].items():
        print(f"    {split}: {num}")
    
    # Mode-specific actions
    if args.mode == 'load':
        # Just load and display data
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

        from .models import BasicVAE
        from .train import train_vae_basic, plot_training_history

        model_config = load_config(args.model_config)
        model_name = model_config['model']['name']
        input_dim = loaders['input_dim']
        hidden_dims = model_config['model'].get('hidden_dims', [1024, 512, 256, 128])
        latent_dim = model_config['model'].get('latent_dim', 32)
        if model_name == "BasicVAE":
            model = BasicVAE(input_dim=input_dim, hidden_dims=hidden_dims, latent_dim=latent_dim, device=args.device)
        else:
            raise ValueError(f"Model name {model_name} not supported")

        if "load_path" in model_config['model']:
            model.load_state_dict(torch.load(model_config['model']['load_path']))
            print(f"Model loaded from {model_config['model']['load_path']}")
            if model_config['model']['freeze_encoder']:
                model.freeze_encoder()
            if model_config['model']['reset_decoder']:
                model.reset_decoder()
        else:
            print("No model load path provided, training from scratch")

        training_config = load_config(args.training_config)
        tracker = TrainingResultsManager(results_dir=project_path / "results")
        experiment_id = tracker.build_experiment_id(
            model_type=model_name,
            model_params=model_config.get('model', {}),
            training_params=training_config.get('training', {}),
            data_params=data_config,
        )
        print(f"Experiment ID: {experiment_id}")

        pathlib.Path(training_config['training']['save_dir']).mkdir(parents=True, exist_ok=True)
        history = train_vae_basic(
            model,
            loaders['train_loader'],
            loaders['val_loader'],
            num_epochs=int(training_config['training']['num_epochs']),
            learning_rate=float(training_config['training']['learning_rate']),
            loss_per_feature=training_config['training']['loss_per_feature'],
            kld_weight=float(training_config['training']['kld_weight']),
            device=args.device,
            save_dir=training_config['training']['save_dir'],
            name=experiment_id,
        )
        # os.makedirs('plots', exist_ok=True)
        # plot_training_history(history, save_path=f'plots/{experiment_id}_training_history.png', show=False)
        print(f'Training history saved to plots/{experiment_id}_training_history.png')

        model_artifact = pathlib.Path(training_config['training']['save_dir']) / f"{experiment_id}_model.pt"
        experiment_metadata = {
            'experiment_id': experiment_id,
            'status': 'completed',
            'model_type': model_name,
            'summary': _build_training_summary(history),
            'model_params': deepcopy(model_config.get('model', {})),
            'training_params': deepcopy(training_config.get('training', {})),
            'data_params': deepcopy(data_config),
            'tags': [training_config['training']['name']],
            'artifacts': {
                'model_path': str(model_artifact),
            },
        }
        tracked_experiment_id = tracker.register_experiment(
            metadata=experiment_metadata,
            history=history,
        )
        print(f'Tracked training experiment: {tracked_experiment_id}')
        print("=" * 60)
        
    elif args.mode == 'inference':
        print("\n" + "=" * 60)
        print("Inference mode")
        print("=" * 60)

        from .models import BasicVAE
        from .inference import inference_vae_basic

        model_config = load_config(args.model_config)

        input_dim = loaders['input_dim']
        hidden_dims = model_config['model'].get('hidden_dims', [1024, 512, 256, 128])
        latent_dim = model_config['model'].get('latent_dim', 32)
        model = BasicVAE(input_dim=input_dim, hidden_dims=hidden_dims, latent_dim=latent_dim, device=args.device)

        training_config = load_config(args.training_config)
        model_name = training_config['training']['name'] + "_model.pt"
        model_path = pathlib.Path(training_config['training']['save_dir']) / model_name
        model.load_state_dict(torch.load(model_path))
        result_dir = args.data_dir / "results"
        result_dir.mkdir(parents=True, exist_ok=True)
        inference_vae_basic(
            model,
            loaders['test_loader'],
            device=args.device,
            loss_per_feature=True,
            num_examples=3,
            plot_dir='plots',
            sample_dir=result_dir
        )


if __name__ == '__main__':
    main()
