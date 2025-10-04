"""
Main entry point for dataset creation and model training.
This script is called from Julia interface.
"""

import argparse
import os
import json
import torch
from torch.utils.data import DataLoader

from dataset_constructor import (
    GaussianMixtureDataset,
    NonlinearManifoldDataset,
    load_real_dataset,
    create_dataloaders
)
from neural_network import create_model, NETWORK_CONFIGS
from trainer import (
    hyperparameter_search,
    train_final_model,
    Trainer
)


def create_dataset_from_args(args):
    """Create dataset based on arguments."""
    
    if args.dataset_type == 'gaussian_mixture':
        # Generate dataset name based on parameters
        dataset_name = f"GM_n{args.n_samples}_d{args.input_dim}_c{args.output_dim}_comp{args.n_components}_s{args.seed}"
        save_dir = os.path.join(args.output_dir, 'datasets', 'gaussian_mixture')
        
        dataset = GaussianMixtureDataset(
            n_samples=args.n_samples,
            input_dim=args.input_dim,
            output_dim=args.output_dim,
            n_components=args.n_components,
            seed=args.seed
        )
        dataset.save(save_dir, dataset_name)
        return dataset, dataset.input_dim, dataset.output_dim, dataset_name
    
    elif args.dataset_type == 'nonlinear_manifold':
        # Generate dataset name based on parameters
        dataset_name = f"NM_n{args.n_samples}_a{args.input_dim}_m{args.manifold_dim}_c{args.output_dim}_{args.nonlinearity}_s{args.seed}"
        save_dir = os.path.join(args.output_dir, 'datasets', 'nonlinear_manifold')
        
        dataset = NonlinearManifoldDataset(
            n_samples=args.n_samples,
            ambient_dim=args.input_dim,
            manifold_dim=args.manifold_dim,
            output_dim=args.output_dim,
            nonlinearity=args.nonlinearity,
            seed=args.seed
        )
        dataset.save(save_dir, dataset_name)
        return dataset, dataset.ambient_dim, dataset.output_dim, dataset_name
    
    elif args.dataset_type in ['mnist', 'fashionmnist', 'cifar10']:
        save_dir = os.path.join(args.output_dir, 'datasets', args.dataset_type)
        train_dataset, test_dataset, metadata = load_real_dataset(
            dataset_name=args.dataset_type,
            data_dir=save_dir,
            flatten=True
        )
        # Save metadata
        with open(os.path.join(save_dir, 'metadata.json'), 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # For real datasets, we return the train dataset
        # and combine train+test for the full dataset
        from torch.utils.data import ConcatDataset
        full_dataset = ConcatDataset([train_dataset, test_dataset])
        return full_dataset, metadata['input_dim'], metadata['output_dim'], args.dataset_type
    
    else:
        raise ValueError(f"Unknown dataset type: {args.dataset_type}")


def main(args):
    """Main training pipeline."""
    print(f"Starting training pipeline...")
    print(f"Dataset: {args.dataset_type}")
    print(f"Model: {args.model_config}")
    print(f"Device: {args.device}")
    
    # Set random seed
    torch.manual_seed(args.seed)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Create or load dataset
    print("\n[1/5] Creating/Loading dataset...")
    dataset, input_dim, output_dim, dataset_name = create_dataset_from_args(args)
    print(f"Dataset size: {len(dataset)}")
    print(f"Dataset name: {dataset_name}")
    print(f"Input dim: {input_dim}, Output dim: {output_dim}")
    
    # Create data loaders
    print("\n[2/5] Creating data loaders...")
    train_loader, val_loader = create_dataloaders(
        dataset,
        batch_size=args.batch_size,
        train_split=0.8,
        seed=args.seed
    )
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
    
    # Create model
    print("\n[3/5] Creating model...")
    if args.model_config in NETWORK_CONFIGS:
        model_kwargs = NETWORK_CONFIGS[args.model_config].copy()
    else:
        raise ValueError(f"Unknown model config: {args.model_config}")
    
    model = create_model(
        model_kwargs.pop('model_type'),
        input_dim,
        output_dim,
        **model_kwargs
    )
    print(f"Model created with {model.count_parameters()} parameters")
    print(f"Model config: {model.get_config()}")
    
    # Hyperparameter search (if enabled)
    if args.run_hparam_search:
        print("\n[4/5] Running hyperparameter search...")
        
        def model_factory():
            return create_model(
                model_kwargs['model_type'] if 'model_type' in NETWORK_CONFIGS[args.model_config] else NETWORK_CONFIGS[args.model_config]['model_type'],
                input_dim,
                output_dim,
                **{k: v for k, v in model_kwargs.items() if k != 'model_type'}
            )
        
        # Recreate model for search
        model_factory_fn = lambda: create_model(
            NETWORK_CONFIGS[args.model_config]['model_type'],
            input_dim,
            output_dim,
            **{k: v for k, v in NETWORK_CONFIGS[args.model_config].items() if k != 'model_type'}
        )
        
        best_params, best_value = hyperparameter_search(
            model_factory_fn,
            train_loader,
            val_loader,
            n_trials=args.n_trials,
            fixed_args={'hparam_search_epochs': args.hparam_search_epochs},
            device=args.device
        )
        
        print(f"Best hyperparameters: {best_params}")
    else:
        print("\n[4/5] Skipping hyperparameter search, using default parameters...")
        best_params = {
            'learning_rate': args.learning_rate,
            'weight_decay': args.weight_decay,
            'optimizer': args.optimizer
        }
    
    # Train final model
    print("\n[5/5] Training final model...")
    model = create_model(
        NETWORK_CONFIGS[args.model_config]['model_type'],
        input_dim,
        output_dim,
        **{k: v for k, v in NETWORK_CONFIGS[args.model_config].items() if k != 'model_type'}
    )
    
    trainer = train_final_model(
        model,
        train_loader,
        val_loader,
        best_params,
        epochs=args.epochs,
        device=args.device,
        verbose=True
    )
    
    # Save model with better organization
    print("\nSaving model...")
    
    # Organize models by dataset name and model config
    # Structure: output/models/{dataset_name}/{model_config}/
    model_save_dir = os.path.join(
        args.output_dir,
        'models',
        dataset_name,
        args.model_config
    )
    os.makedirs(model_save_dir, exist_ok=True)
    
    model_save_path = os.path.join(
        model_save_dir,
        f"model_seed{args.seed}.pt"
    )
    
    train_args = {
        'dataset_type': args.dataset_type,
        'dataset_name': dataset_name,
        'model_config': args.model_config,
        'n_samples': args.n_samples if hasattr(args, 'n_samples') else None,
        'input_dim': input_dim,
        'output_dim': output_dim,
        'batch_size': args.batch_size,
        'epochs': args.epochs,
        'seed': args.seed,
        'best_hyperparams': best_params,
        'device': args.device
    }
    
    # Save model (including ONNX export)
    trainer.save_model(
        model_save_path,
        model.get_config(),
        train_args,
        save_onnx=True
    )
    
    print(f"\nModel saved to: {model_save_path}")
    print(f"Best validation accuracy: {max(trainer.history['val_acc']):.4f}")
    
    # Return results as JSON for Julia interface
    results = {
        'model_path': model_save_path,
        'onnx_path': model_save_path.replace('.pt', '.onnx'),
        'best_val_acc': max(trainer.history['val_acc']),
        'train_args': train_args,
        'model_config': model.get_config()
    }
    
    results_path = os.path.join(model_save_dir, f"results_seed{args.seed}.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to: {results_path}")
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train neural networks for MadNLP4NN')
    
    # Dataset arguments
    parser.add_argument('--dataset_type', type=str, required=True,
                        choices=['gaussian_mixture', 'nonlinear_manifold', 'mnist', 'fashionmnist', 'cifar10'],
                        help='Type of dataset')
    parser.add_argument('--n_samples', type=int, default=10000,
                        help='Number of samples for synthetic datasets')
    parser.add_argument('--input_dim', type=int, default=784,
                        help='Input dimension')
    parser.add_argument('--output_dim', type=int, default=10,
                        help='Output dimension (number of classes)')
    parser.add_argument('--n_components', type=int, default=20,
                        help='Number of Gaussian components (for gaussian_mixture)')
    parser.add_argument('--manifold_dim', type=int, default=50,
                        help='Manifold dimension (for nonlinear_manifold)')
    parser.add_argument('--nonlinearity', type=str, default='polynomial',
                        choices=['polynomial', 'trigonometric', 'mixed'],
                        help='Type of nonlinearity (for nonlinear_manifold)')
    
    # Model arguments
    parser.add_argument('--model_config', type=str, required=True,
                        choices=list(NETWORK_CONFIGS.keys()),
                        help='Model configuration')
    
    # Training arguments
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-3,
                        help='Learning rate (if not using hparam search)')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='Weight decay (if not using hparam search)')
    parser.add_argument('--optimizer', type=str, default='adam',
                        choices=['adam', 'adamw', 'sgd'],
                        help='Optimizer (if not using hparam search)')
    
    # Hyperparameter search arguments
    parser.add_argument('--run_hparam_search', action='store_true',
                        help='Run hyperparameter search')
    parser.add_argument('--n_trials', type=int, default=50,
                        help='Number of trials for hyperparameter search')
    parser.add_argument('--hparam_search_epochs', type=int, default=20,
                        help='Number of epochs for each trial in hyperparameter search')
    
    # Other arguments
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to train on')
    parser.add_argument('--output_dir', type=str, default='./output',
                        help='Output directory')
    
    args = parser.parse_args()
    main(args)
