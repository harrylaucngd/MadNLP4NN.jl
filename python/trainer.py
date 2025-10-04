"""
Training module with hyperparameter search using Optuna.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import os
import json
from typing import Dict, Optional, Tuple, Callable
from tqdm import tqdm
import optuna
from datetime import datetime

from neural_network import create_model, NETWORK_CONFIGS


class Trainer:
    """Trainer class for neural network training."""
    
    def __init__(
        self,
        model: nn.Module,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        """
        Args:
            model: Neural network model
            device: Device to train on
        """
        self.model = model.to(device)
        self.device = device
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }
        
    def train_epoch(
        self,
        train_loader: DataLoader,
        optimizer: optim.Optimizer,
        criterion: nn.Module
    ) -> Tuple[float, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        for data, target in train_loader:
            data, target = data.to(self.device), target.to(self.device)
            
            optimizer.zero_grad()
            output = self.model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item() * data.size(0)
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += data.size(0)
        
        avg_loss = total_loss / total
        accuracy = correct / total
        return avg_loss, accuracy
    
    def validate(
        self,
        val_loader: DataLoader,
        criterion: nn.Module
    ) -> Tuple[float, float]:
        """Validate the model."""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = criterion(output, target)
                
                total_loss += loss.item() * data.size(0)
                pred = output.argmax(dim=1)
                correct += pred.eq(target).sum().item()
                total += data.size(0)
        
        avg_loss = total_loss / total
        accuracy = correct / total
        return avg_loss, accuracy
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: optim.Optimizer,
        criterion: nn.Module,
        epochs: int,
        scheduler: Optional[optim.lr_scheduler._LRScheduler] = None,
        verbose: bool = True
    ) -> Dict[str, list]:
        """
        Train the model.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            optimizer: Optimizer
            criterion: Loss criterion
            epochs: Number of epochs
            scheduler: Learning rate scheduler
            verbose: Whether to print progress
            
        Returns:
            Training history
        """
        best_val_acc = 0.0
        
        iterator = tqdm(range(epochs), desc="Training") if verbose else range(epochs)
        
        for epoch in iterator:
            train_loss, train_acc = self.train_epoch(train_loader, optimizer, criterion)
            val_loss, val_acc = self.validate(val_loader, criterion)
            
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            
            if scheduler is not None:
                scheduler.step()
            
            if verbose:
                iterator.set_postfix({
                    'train_loss': f'{train_loss:.4f}',
                    'train_acc': f'{train_acc:.4f}',
                    'val_loss': f'{val_loss:.4f}',
                    'val_acc': f'{val_acc:.4f}'
                })
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
        
        return self.history
    
    def save_model(
        self,
        save_path: str,
        config: Dict,
        train_args: Dict,
        save_onnx: bool = True
    ):
        """
        Save model with configuration and training arguments.
        
        Args:
            save_path: Path to save the model
            config: Model configuration
            train_args: Training arguments
            save_onnx: Whether to also save as ONNX format
        """
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        
        # Save model state
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'model_config': config,
            'train_args': train_args,
            'history': self.history
        }, save_path)
        
        # Save parameters in numpy format for NLP interface
        params_dict = self.model.get_parameters_dict()
        np_save_path = save_path.replace('.pt', '_params.npz')
        np.savez(np_save_path, **params_dict)
        
        # Save metadata
        metadata = {
            'model_config': config,
            'train_args': train_args,
            'history': self.history,
            'best_val_acc': max(self.history['val_acc']) if self.history['val_acc'] else 0.0,
            'timestamp': datetime.now().isoformat()
        }
        metadata_path = save_path.replace('.pt', '_metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Save as ONNX format
        if save_onnx:
            try:
                onnx_path = save_path.replace('.pt', '.onnx')
                self.model.eval()
                
                # Create dummy input based on model input dimension
                input_dim = config.get('input_dim', 784)
                dummy_input = torch.randn(1, input_dim).to(self.device)
                
                # Export to ONNX
                torch.onnx.export(
                    self.model,
                    dummy_input,
                    onnx_path,
                    export_params=True,
                    opset_version=11,
                    do_constant_folding=True,
                    input_names=['input'],
                    output_names=['output'],
                    dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
                )
                print(f"Model also saved as ONNX: {onnx_path}")
            except Exception as e:
                print(f"Warning: Failed to save ONNX model: {e}")


def create_optimizer(
    model: nn.Module,
    optimizer_name: str,
    learning_rate: float,
    weight_decay: float = 0.0,
    **kwargs
) -> optim.Optimizer:
    """Create optimizer."""
    if optimizer_name.lower() == 'adam':
        return optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
    elif optimizer_name.lower() == 'adamw':
        return optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
    elif optimizer_name.lower() == 'sgd':
        momentum = kwargs.get('momentum', 0.9)
        return optim.SGD(
            model.parameters(),
            lr=learning_rate,
            momentum=momentum,
            weight_decay=weight_decay
        )
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")


def objective_function(
    trial: optuna.Trial,
    model_factory: Callable,
    train_loader: DataLoader,
    val_loader: DataLoader,
    fixed_args: Dict,
    device: str
) -> float:
    """
    Objective function for Optuna hyperparameter search.
    
    Args:
        trial: Optuna trial object
        model_factory: Function to create model
        train_loader: Training data loader
        val_loader: Validation data loader
        fixed_args: Fixed arguments (not tuned)
        device: Device to train on
        
    Returns:
        Validation accuracy
    """
    # Suggest hyperparameters
    learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)
    weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True)
    optimizer_name = trial.suggest_categorical('optimizer', ['adam', 'adamw', 'sgd'])
    
    # Create model
    model = model_factory()
    
    # Create optimizer
    optimizer = create_optimizer(
        model,
        optimizer_name,
        learning_rate,
        weight_decay
    )
    
    # Create trainer
    trainer = Trainer(model, device)
    
    # Train
    criterion = nn.CrossEntropyLoss()
    epochs = fixed_args.get('hparam_search_epochs', 20)
    
    trainer.train(
        train_loader,
        val_loader,
        optimizer,
        criterion,
        epochs,
        verbose=False
    )
    
    # Return best validation accuracy
    return max(trainer.history['val_acc'])


def hyperparameter_search(
    model_factory: Callable,
    train_loader: DataLoader,
    val_loader: DataLoader,
    n_trials: int = 50,
    fixed_args: Optional[Dict] = None,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
) -> Tuple[Dict, float]:
    """
    Perform hyperparameter search using Optuna.
    
    Args:
        model_factory: Function to create model
        train_loader: Training data loader
        val_loader: Validation data loader
        n_trials: Number of trials for hyperparameter search
        fixed_args: Fixed arguments
        device: Device to train on
        
    Returns:
        best_params, best_value
    """
    if fixed_args is None:
        fixed_args = {}
    
    study = optuna.create_study(direction='maximize')
    
    def objective(trial):
        return objective_function(
            trial,
            model_factory,
            train_loader,
            val_loader,
            fixed_args,
            device
        )
    
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    
    print(f"\nBest trial:")
    print(f"  Value: {study.best_trial.value:.4f}")
    print(f"  Params: {study.best_trial.params}")
    
    return study.best_trial.params, study.best_trial.value


def train_final_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    hyperparams: Dict,
    epochs: int = 100,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
    verbose: bool = True
) -> Trainer:
    """
    Train final model with best hyperparameters.
    
    Args:
        model: Neural network model
        train_loader: Training data loader
        val_loader: Validation data loader
        hyperparams: Hyperparameters (from search or manual)
        epochs: Number of epochs
        device: Device to train on
        verbose: Whether to print progress
        
    Returns:
        Trained trainer object
    """
    # Create optimizer
    optimizer = create_optimizer(
        model,
        hyperparams['optimizer'],
        hyperparams['learning_rate'],
        hyperparams['weight_decay']
    )
    
    # Create learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    # Create trainer
    trainer = Trainer(model, device)
    
    # Train
    criterion = nn.CrossEntropyLoss()
    trainer.train(
        train_loader,
        val_loader,
        optimizer,
        criterion,
        epochs,
        scheduler=scheduler,
        verbose=verbose
    )
    
    return trainer


if __name__ == '__main__':
    # Example usage
    from dataset_constructor import GaussianMixtureDataset, create_dataloaders
    from neural_network import create_model
    
    print("Creating dataset...")
    dataset = GaussianMixtureDataset(n_samples=5000, input_dim=784, output_dim=10)
    train_loader, val_loader = create_dataloaders(dataset, batch_size=64)
    
    print("Creating model...")
    model = create_model('mlp', 784, 10, hidden_dims=[256, 128, 64], dropout=0.2)
    
    print("Training...")
    trainer = Trainer(model)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    
    history = trainer.train(train_loader, val_loader, optimizer, criterion, epochs=10)
    
    print(f"\nFinal train acc: {history['train_acc'][-1]:.4f}")
    print(f"Final val acc: {history['val_acc'][-1]:.4f}")
