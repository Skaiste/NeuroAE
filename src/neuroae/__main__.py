"""
Main script for training VAE models and running inference on ADNI-B data.

This script loads ADNI-B data and can be used for training models or running inference.
"""

import argparse
import os
import torch
import pathlib
import configparser
from neuronumba.tools.filters import BandPassFilter

from .utils import *
from .load_data import load_adni, prepare_data_loaders


def load_filter_config(filter_config_path):
    filter_config = configparser.ConfigParser()
    filter_config.read(filter_config_path)
    return filter_config

def main():
    """Main function for training and inference."""
    parser = argparse.ArgumentParser(description='Train VAE models or run inference on ADNI-B data')
    parser.add_argument(
        '--mode',
        type=str,
        default='train',
        choices=['train', 'inference', 'load'],
        help='Mode: train, inference, or just load data'
    )
    parser.add_argument(
        '--data-dir',
        type=pathlib.Path,
        default=project_path / "data",
        help='Path to data directory (default: ./data)'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=32,
        help='Batch size for DataLoaders'
    )
    parser.add_argument(
        '--no-flatten',
        action='store_true',
        help='Keep timeseries as 2D (N_ROIs, T_timepoints). Default is to flatten to 1D.'
    )
    parser.add_argument(
        '--filter-config',
        type=pathlib.Path,
        default=project_path / "config" / "filter.yml",
        help='Path to filter configuration file (default: ./config/filter.yml)'
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("ADNI-B VAE Training/Inference")
    print("=" * 60)
    
    # Load ADNI data
    print(f"\nLoading ADNI-B dataset...")
    filter_config = load_filter_config(args.filter_config)
    if 'BandPassFilter' in filter_config:
        filter = BandPassFilter(
            tr=float(filter_config['BandPassFilter']['tr']),
            flp=float(filter_config['BandPassFilter']['flp']),
            fhi=float(filter_config['BandPassFilter']['fhi']),
            k=int(filter_config['BandPassFilter']['k']),
            remove_artifacts=filter_config['BandPassFilter']['remove_artifacts'],
            apply_demean=filter_config['BandPassFilter']['apply_demean'],
            apply_detrend=filter_config['BandPassFilter']['apply_detrend'],
            apply_finalDetrend=filter_config['BandPassFilter']['apply_finalDetrend'],
        )
        print(f"Filter configuration found in {args.filter_config}")
    else:
        filter = None
        print(f"No filter configuration found in {args.filter_config}")
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
    flatten = not args.no_flatten  # Default is True (flatten)
    loaders = prepare_data_loaders(
        data_loader,
        batch_size=args.batch_size,
        flatten=flatten,
        normalize=False,
    )
    
    print(f"\nDataLoader information:")
    print(f"  Input dimension: {loaders['input_dim']}")
    print(f"  Number of samples:")
    for split, num in loaders['num_samples'].items():
        print(f"    {split}: {num}")

    breakpoint()
    
    # Mode-specific actions
    if args.mode == 'load':
        # Just load and display data
        print(f"\nTesting data loading...")
        sample_batch = next(iter(loaders['train_loader']))
        if isinstance(sample_batch, tuple):
            batch_data, batch_labels = sample_batch
            print(f"  Batch shape: {batch_data.shape}")
            print(f"  Batch labels: {batch_labels[:5]}...")  # Show first 5 labels
        else:
            print(f"  Batch shape: {sample_batch.shape}")
        
        print("\n" + "=" * 60)
        print("Data loading complete!")
        print("=" * 60)
        
    elif args.mode == 'train':
        # TODO: Add training logic here
        print("\n" + "=" * 60)
        print("Training mode")
        print("=" * 60)

        # running on mac
        device = 'mps'

        from .models import BasicVAE
        from .train import train_vae_basic, plot_training_history
        model = BasicVAE(input_dim=78800, device=device)
        # train full model first    
        history = train_vae_basic(
            model,
            loaders['train_loader'],
            loaders['val_loader'],
            num_epochs=40,
            learning_rate=1e-3,
            device=device,
            loss_per_feature=False
        )
        os.makedirs('plots', exist_ok=True)
        plot_training_history(history, save_path='plots/basicVAE_training_history.png', show=False)
        print("Training history saved to plots/basicVAE_training_history.png")
        print("=" * 60)
        
    elif args.mode == 'inference':
        # TODO: Add inference logic here
        print("\n" + "=" * 60)
        print("Inference mode")
        print("=" * 60)

        # running on mac
        device = 'mps'

        from .models import BasicVAE
        from .inference import inference_vae_basic
        model = BasicVAE(input_dim=78800, device=device)
        model.load_state_dict(torch.load('checkpoints/best_model.pt'))
        inference_vae_basic(
            model,
            loaders['test_loader'],
            device=device,
            loss_per_feature=True,
            num_examples=3,
            plot_dir='plots'
        )

    return data_loader, loaders


if __name__ == '__main__':
    data_loader, loaders = main()
