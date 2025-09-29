"""
Neural network architectures for MadNLP4NN.
All networks use ReLU activation as required for the NLP formulation.
"""

import torch
import torch.nn as nn
from typing import List, Tuple


class MLPClassifier(nn.Module):
    """
    Multi-layer perceptron classifier with ReLU activation.
    Designed for medium-complexity tasks suitable for adversarial optimization.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        output_dim: int,
        dropout: float = 0.0
    ):
        """
        Args:
            input_dim: Input dimension
            hidden_dims: List of hidden layer dimensions
            output_dim: Output dimension (number of classes)
            dropout: Dropout probability
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        self.dropout_p = dropout
        
        # Build layers
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.network(x)
    
    def get_config(self) -> dict:
        """Get network configuration."""
        return {
            'model_type': 'MLPClassifier',
            'input_dim': self.input_dim,
            'hidden_dims': self.hidden_dims,
            'output_dim': self.output_dim,
            'dropout': self.dropout_p
        }
    
    def get_parameters_dict(self) -> dict:
        """
        Get parameters in a format suitable for NLP formulation.
        Returns a dictionary with layer-wise parameters.
        """
        params_dict = {}
        for name, param in self.named_parameters():
            params_dict[name] = param.detach().cpu().numpy()
        return params_dict
    
    def count_parameters(self) -> int:
        """Count total number of parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class ResidualBlock(nn.Module):
    """Residual block with ReLU activation."""
    
    def __init__(self, dim: int, dropout: float = 0.0):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = torch.relu(self.fc1(x))
        out = self.dropout(out)
        out = self.fc2(out)
        out = out + residual
        out = torch.relu(out)
        return out


class ResMLPClassifier(nn.Module):
    """
    Residual MLP classifier for more complex tasks.
    Uses residual connections to enable deeper networks.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_blocks: int,
        output_dim: int,
        dropout: float = 0.0
    ):
        """
        Args:
            input_dim: Input dimension
            hidden_dim: Hidden dimension (same for all residual blocks)
            num_blocks: Number of residual blocks
            output_dim: Output dimension
            dropout: Dropout probability
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_blocks = num_blocks
        self.output_dim = output_dim
        self.dropout_p = dropout
        
        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Residual blocks
        self.blocks = nn.Sequential(*[
            ResidualBlock(hidden_dim, dropout) for _ in range(num_blocks)
        ])
        
        # Output layer
        self.output_layer = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        x = self.input_proj(x)
        x = self.blocks(x)
        x = self.output_layer(x)
        return x
    
    def get_config(self) -> dict:
        """Get network configuration."""
        return {
            'model_type': 'ResMLPClassifier',
            'input_dim': self.input_dim,
            'hidden_dim': self.hidden_dim,
            'num_blocks': self.num_blocks,
            'output_dim': self.output_dim,
            'dropout': self.dropout_p
        }
    
    def get_parameters_dict(self) -> dict:
        """Get parameters for NLP formulation."""
        params_dict = {}
        for name, param in self.named_parameters():
            params_dict[name] = param.detach().cpu().numpy()
        return params_dict
    
    def count_parameters(self) -> int:
        """Count total number of parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_model(
    model_type: str,
    input_dim: int,
    output_dim: int,
    **kwargs
) -> nn.Module:
    """
    Factory function to create models.
    
    Args:
        model_type: Type of model ('mlp', 'resmlp')
        input_dim: Input dimension
        output_dim: Output dimension
        **kwargs: Additional model-specific arguments
        
    Returns:
        Neural network model
    """
    if model_type.lower() == 'mlp':
        hidden_dims = kwargs.get('hidden_dims', [256, 128, 64])
        dropout = kwargs.get('dropout', 0.0)
        return MLPClassifier(input_dim, hidden_dims, output_dim, dropout)
    
    elif model_type.lower() == 'resmlp':
        hidden_dim = kwargs.get('hidden_dim', 256)
        num_blocks = kwargs.get('num_blocks', 4)
        dropout = kwargs.get('dropout', 0.0)
        return ResMLPClassifier(input_dim, hidden_dim, num_blocks, output_dim, dropout)
    
    else:
        raise ValueError(f"Unknown model type: {model_type}")


# Predefined network configurations for different complexity levels
NETWORK_CONFIGS = {
    'small_mlp': {
        'model_type': 'mlp',
        'hidden_dims': [128, 64],
        'dropout': 0.1
    },
    'medium_mlp': {
        'model_type': 'mlp',
        'hidden_dims': [256, 128, 64],
        'dropout': 0.2
    },
    'large_mlp': {
        'model_type': 'mlp',
        'hidden_dims': [512, 256, 128, 64],
        'dropout': 0.2
    },
    'small_resmlp': {
        'model_type': 'resmlp',
        'hidden_dim': 128,
        'num_blocks': 2,
        'dropout': 0.1
    },
    'medium_resmlp': {
        'model_type': 'resmlp',
        'hidden_dim': 256,
        'num_blocks': 4,
        'dropout': 0.2
    },
    'large_resmlp': {
        'model_type': 'resmlp',
        'hidden_dim': 512,
        'num_blocks': 6,
        'dropout': 0.2
    }
}


if __name__ == '__main__':
    # Test models
    print("Testing MLP Classifier:")
    mlp = MLPClassifier(784, [256, 128, 64], 10, dropout=0.2)
    print(f"Parameters: {mlp.count_parameters()}")
    
    x = torch.randn(32, 784)
    y = mlp(x)
    print(f"Input shape: {x.shape}, Output shape: {y.shape}")
    
    print("\nTesting Residual MLP Classifier:")
    resmlp = ResMLPClassifier(784, 256, 4, 10, dropout=0.2)
    print(f"Parameters: {resmlp.count_parameters()}")
    
    y = resmlp(x)
    print(f"Input shape: {x.shape}, Output shape: {y.shape}")
