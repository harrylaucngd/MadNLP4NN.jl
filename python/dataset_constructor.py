"""
Dataset construction module for MadNLP4NN.
Supports both synthetic datasets (with non-trivial distributions) and real-life datasets.
"""

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import datasets, transforms
from sklearn.datasets import make_classification
import json
import os
from typing import Tuple, Dict, Optional


class GaussianMixtureDataset(Dataset):
    """
    Synthetic dataset based on high-dimensional Gaussian mixture.
    This creates a non-trivial, non-uniform distribution suitable for adversarial testing.
    """
    
    def __init__(
        self,
        n_samples: int = 10000,
        input_dim: int = 784,
        output_dim: int = 10,
        n_components: int = 20,
        seed: int = 42
    ):
        """
        Args:
            n_samples: Number of samples to generate
            input_dim: Input dimension (e.g., 784 for 28x28 images)
            output_dim: Number of classes
            n_components: Number of Gaussian components per class (higher = more complex)
            seed: Random seed for reproducibility
        """
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.n_components = n_components
        
        # Generate data with complex, non-uniform distribution
        self.data, self.labels = self._generate_mixture_data(n_samples)
        
    def _generate_mixture_data(self, n_samples: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate data from Gaussian mixture model."""
        samples_per_class = n_samples // self.output_dim
        
        all_data = []
        all_labels = []
        
        for class_idx in range(self.output_dim):
            class_data = []
            
            # Create multiple Gaussian components for each class
            for _ in range(self.n_components):
                # Random mean for each component
                mean = np.random.randn(self.input_dim) * 2.0
                
                # Random covariance (using diagonal for efficiency)
                # Add non-uniformity by varying scales
                scale = np.exp(np.random.randn(self.input_dim) * 0.5)
                
                # Generate samples from this component
                component_samples = samples_per_class // self.n_components
                samples = np.random.randn(component_samples, self.input_dim) * scale + mean
                class_data.append(samples)
            
            class_data = np.vstack(class_data)
            all_data.append(class_data)
            all_labels.extend([class_idx] * len(class_data))
        
        data = np.vstack(all_data)
        labels = np.array(all_labels)
        
        # Normalize to [0, 1] range
        data = (data - data.min()) / (data.max() - data.min() + 1e-8)
        
        # Shuffle
        indices = np.random.permutation(len(data))
        data = data[indices]
        labels = labels[indices]
        
        return torch.FloatTensor(data), torch.LongTensor(labels)
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.data[idx], self.labels[idx]
    
    def save(self, save_dir: str):
        """Save dataset to disk."""
        os.makedirs(save_dir, exist_ok=True)
        torch.save(self.data, os.path.join(save_dir, 'data.pt'))
        torch.save(self.labels, os.path.join(save_dir, 'labels.pt'))
        
        metadata = {
            'n_samples': len(self.data),
            'input_dim': self.input_dim,
            'output_dim': self.output_dim,
            'n_components': self.n_components,
            'dataset_type': 'gaussian_mixture'
        }
        with open(os.path.join(save_dir, 'metadata.json'), 'w') as f:
            json.dump(metadata, f, indent=2)


class NonlinearManifoldDataset(Dataset):
    """
    Synthetic dataset where data lies on a nonlinear manifold.
    This represents a more challenging scenario for adversarial attacks.
    """
    
    def __init__(
        self,
        n_samples: int = 10000,
        ambient_dim: int = 784,
        manifold_dim: int = 50,
        output_dim: int = 10,
        nonlinearity: str = 'polynomial',
        seed: int = 42
    ):
        """
        Args:
            n_samples: Number of samples
            ambient_dim: Dimension of the ambient space
            manifold_dim: Intrinsic dimension of the manifold
            output_dim: Number of classes
            nonlinearity: Type of nonlinearity ('polynomial', 'trigonometric', 'mixed')
            seed: Random seed
        """
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        self.ambient_dim = ambient_dim
        self.manifold_dim = manifold_dim
        self.output_dim = output_dim
        self.nonlinearity = nonlinearity
        
        self.data, self.labels = self._generate_manifold_data(n_samples)
    
    def _generate_manifold_data(self, n_samples: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate data lying on a nonlinear manifold."""
        samples_per_class = n_samples // self.output_dim
        
        all_data = []
        all_labels = []
        
        for class_idx in range(self.output_dim):
            # Low-dimensional latent representation
            latent = np.random.randn(samples_per_class, self.manifold_dim)
            
            # Random projection matrix
            projection = np.random.randn(self.manifold_dim, self.ambient_dim) / np.sqrt(self.manifold_dim)
            
            # Linear projection
            data = latent @ projection
            
            # Add nonlinearity based on type
            if self.nonlinearity == 'polynomial':
                # Add polynomial terms
                data = data + 0.3 * (latent @ projection) ** 2
                data = data + 0.1 * (latent @ projection) ** 3
            elif self.nonlinearity == 'trigonometric':
                # Add trigonometric terms
                data = data + 0.5 * np.sin(latent @ projection * 2)
            elif self.nonlinearity == 'mixed':
                # Mix of both
                data = data + 0.2 * (latent @ projection) ** 2 + 0.3 * np.sin(latent @ projection)
            
            # Add class-specific shift
            class_shift = np.random.randn(self.ambient_dim) * 2.0
            data = data + class_shift
            
            all_data.append(data)
            all_labels.extend([class_idx] * samples_per_class)
        
        data = np.vstack(all_data)
        labels = np.array(all_labels)
        
        # Normalize
        data = (data - data.min()) / (data.max() - data.min() + 1e-8)
        
        # Shuffle
        indices = np.random.permutation(len(data))
        data = data[indices]
        labels = labels[indices]
        
        return torch.FloatTensor(data), torch.LongTensor(labels)
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.data[idx], self.labels[idx]
    
    def save(self, save_dir: str):
        """Save dataset to disk."""
        os.makedirs(save_dir, exist_ok=True)
        torch.save(self.data, os.path.join(save_dir, 'data.pt'))
        torch.save(self.labels, os.path.join(save_dir, 'labels.pt'))
        
        metadata = {
            'n_samples': len(self.data),
            'ambient_dim': self.ambient_dim,
            'manifold_dim': self.manifold_dim,
            'output_dim': self.output_dim,
            'nonlinearity': self.nonlinearity,
            'dataset_type': 'nonlinear_manifold'
        }
        with open(os.path.join(save_dir, 'metadata.json'), 'w') as f:
            json.dump(metadata, f, indent=2)


def load_real_dataset(
    dataset_name: str = 'mnist',
    data_dir: str = './data',
    flatten: bool = True
) -> Tuple[Dataset, Dataset, Dict]:
    """
    Load real-life datasets (MNIST, FashionMNIST, CIFAR10).
    
    Args:
        dataset_name: Name of the dataset
        data_dir: Directory to store/load data
        flatten: Whether to flatten images
        
    Returns:
        train_dataset, test_dataset, metadata
    """
    if dataset_name.lower() == 'mnist':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.flatten() if flatten else x)
        ])
        train_dataset = datasets.MNIST(data_dir, train=True, download=True, transform=transform)
        test_dataset = datasets.MNIST(data_dir, train=False, download=True, transform=transform)
        input_dim = 784 if flatten else (1, 28, 28)
        output_dim = 10
        
    elif dataset_name.lower() == 'fashionmnist':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.flatten() if flatten else x)
        ])
        train_dataset = datasets.FashionMNIST(data_dir, train=True, download=True, transform=transform)
        test_dataset = datasets.FashionMNIST(data_dir, train=False, download=True, transform=transform)
        input_dim = 784 if flatten else (1, 28, 28)
        output_dim = 10
        
    elif dataset_name.lower() == 'cifar10':
        if flatten:
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Lambda(lambda x: x.flatten())
            ])
            input_dim = 3072
        else:
            transform = transforms.ToTensor()
            input_dim = (3, 32, 32)
        
        train_dataset = datasets.CIFAR10(data_dir, train=True, download=True, transform=transform)
        test_dataset = datasets.CIFAR10(data_dir, train=False, download=True, transform=transform)
        output_dim = 10
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    metadata = {
        'dataset_name': dataset_name,
        'input_dim': input_dim,
        'output_dim': output_dim,
        'dataset_type': 'real',
        'flatten': flatten
    }
    
    return train_dataset, test_dataset, metadata


def create_dataloaders(
    dataset: Dataset,
    batch_size: int = 64,
    train_split: float = 0.8,
    seed: int = 42
) -> Tuple[DataLoader, DataLoader]:
    """
    Create train and validation dataloaders from a dataset.
    
    Args:
        dataset: The dataset to split
        batch_size: Batch size for dataloaders
        train_split: Fraction of data for training
        seed: Random seed
        
    Returns:
        train_loader, val_loader
    """
    train_size = int(train_split * len(dataset))
    val_size = len(dataset) - train_size
    
    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(seed)
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )
    
    return train_loader, val_loader


if __name__ == '__main__':
    # Example usage
    print("Creating Gaussian Mixture Dataset...")
    gm_dataset = GaussianMixtureDataset(
        n_samples=10000,
        input_dim=784,
        output_dim=10,
        n_components=20
    )
    print(f"Created dataset with {len(gm_dataset)} samples")
    print(f"Data shape: {gm_dataset.data.shape}")
    print(f"Labels shape: {gm_dataset.labels.shape}")
    
    print("\nCreating Nonlinear Manifold Dataset...")
    nm_dataset = NonlinearManifoldDataset(
        n_samples=10000,
        ambient_dim=784,
        manifold_dim=50,
        output_dim=10
    )
    print(f"Created dataset with {len(nm_dataset)} samples")
    
    print("\nLoading MNIST Dataset...")
    train_ds, test_ds, metadata = load_real_dataset('mnist')
    print(f"Train size: {len(train_ds)}, Test size: {len(test_ds)}")
    print(f"Metadata: {metadata}")
