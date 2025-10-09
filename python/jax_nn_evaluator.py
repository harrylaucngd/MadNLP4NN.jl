"""
JAX-based neural network evaluator for optimization.

This module provides functions to evaluate neural networks and their derivatives
using JAX, callable from Julia via PythonCall. This enables using JAX automatic
differentiation while keeping MadNLP as the solver in Julia.

The evaluator computes:
- Neural network forward pass
- Objective function value
- Gradient via JAX automatic differentiation
- Constraint values
- Constraint Jacobian via JAX
- Hessian of Lagrangian via JAX

Author: MadNLP4NN
License: MIT
"""

import jax
import jax.numpy as jnp
import numpy as np
import torch
from typing import Tuple, Callable, Dict, Optional, Union
from functools import partial

# Enable float64 for numerical consistency with Julia
jax.config.update("jax_enable_x64", True)

# Enable deterministic operations for reproducibility
# This ensures that operations like reductions are deterministic
jax.config.update("jax_default_matmul_precision", "highest")

# Disable JIT compilation randomness (if needed)
# Note: JAX operations are deterministic by default when given the same inputs
# and random seeds, but this ensures platform consistency


def smooth_relu(x: jnp.ndarray, alpha: float = 1e-3) -> jnp.ndarray:
    """
    Smoothed ReLU activation: log(1 + exp(x/α)) * α
    
    This provides a smooth approximation to ReLU that is differentiable
    everywhere, which is required for second-order optimization methods.
    Uses jax.nn.softplus which is numerically stable and JAX-friendly.
    
    Args:
        x: Input array
        alpha: Smoothing parameter (smaller = closer to ReLU, default 1e-3)
    
    Returns:
        Smoothed ReLU applied element-wise
    """
    # Use JAX's built-in softplus: softplus(x) = log(1 + exp(x))
    # Our smooth_relu is: alpha * softplus(x/alpha)
    return alpha * jax.nn.softplus(x / alpha)


def load_pytorch_model_to_jax(model_path: str) -> Tuple[Callable, Dict, Dict]:
    """
    Load PyTorch model and convert to JAX format.
    
    Args:
        model_path: Path to saved PyTorch model (.pt file)
    
    Returns:
        jax_forward: JAX function (params, x) -> output
        params: PyTree of model parameters (dict)
        config: Model configuration dict
    
    Example:
        >>> jax_forward, params, config = load_pytorch_model_to_jax("model.pt")
        >>> output = jax_forward(params, x)
    """
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location='cpu')
    state_dict = checkpoint['model_state_dict']
    config = checkpoint['model_config']
    
    # Convert based on model type
    model_type = config['model_type']
    if model_type in ['mlp', 'MLPClassifier']:
        jax_forward, params = _build_mlp_jax(state_dict, config)
    elif model_type in ['resmlp', 'ResMLPClassifier']:
        jax_forward, params = _build_resmlp_jax(state_dict, config)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    return jax_forward, params, config


def _build_mlp_jax(state_dict: Dict, config: Dict) -> Tuple[Callable, Dict]:
    """
    Build JAX MLP from PyTorch state dict.
    
    Args:
        state_dict: PyTorch state dictionary
        config: Model configuration
    
    Returns:
        forward: JAX forward function
        params: Parameter dictionary
    """
    # Extract layer indices from state dict keys
    # Handle both "network.X" and "layers.X" formats
    layer_indices = []
    for key in state_dict.keys():
        if 'weight' in key:
            if key.startswith('network.'):
                idx = int(key.split('.')[1])
                layer_indices.append(idx)
            elif key.startswith('layers.'):
                idx = int(key.split('.')[1])
                layer_indices.append(idx)
    
    layer_indices = sorted(set(layer_indices))
    
    if not layer_indices:
        raise ValueError("No layers found in state_dict")
    
    # Determine prefix
    prefix = 'network' if any(k.startswith('network.') for k in state_dict.keys()) else 'layers'
    
    # Extract parameters
    params = {}
    for i, idx in enumerate(layer_indices):
        W_key = f'{prefix}.{idx}.weight'
        b_key = f'{prefix}.{idx}.bias'
        
        W = state_dict[W_key].detach().cpu().numpy()
        b = state_dict[b_key].detach().cpu().numpy()
        
        # Convert to JAX arrays
        params[f'layer_{i}'] = {
            'W': jnp.array(W, dtype=jnp.float64),
            'b': jnp.array(b, dtype=jnp.float64)
        }
    
    n_layers = len(params)
    
    # Build forward function
    def forward(params_dict, x):
        """MLP forward pass with smooth ReLU activation."""
        out = x
        for i in range(n_layers):
            W = params_dict[f'layer_{i}']['W']
            b = params_dict[f'layer_{i}']['b']
            out = jnp.dot(W, out) + b
            if i < n_layers - 1:  # Apply activation except last layer
                out = smooth_relu(out)
        return out
    
    return forward, params


def _build_resmlp_jax(state_dict: Dict, config: Dict) -> Tuple[Callable, Dict]:
    """
    Build JAX Residual MLP from PyTorch state dict.
    
    Args:
        state_dict: PyTorch state dictionary
        config: Model configuration
    
    Returns:
        forward: JAX forward function
        params: Parameter dictionary
    """
    # Extract initial projection
    params = {
        'init_proj': {
            'W': jnp.array(state_dict['input_proj.0.weight'].detach().cpu().numpy(), dtype=jnp.float64),
            'b': jnp.array(state_dict['input_proj.0.bias'].detach().cpu().numpy(), dtype=jnp.float64)
        },
        'blocks': [],
        'output': {
            'W': jnp.array(state_dict['output_layer.weight'].detach().cpu().numpy(), dtype=jnp.float64),
            'b': jnp.array(state_dict['output_layer.bias'].detach().cpu().numpy(), dtype=jnp.float64)
        }
    }
    
    # Extract residual blocks
    n_blocks = config.get('num_blocks', config.get('n_blocks', 0))
    for i in range(n_blocks):
        block_params = {
            'fc1': {
                'W': jnp.array(state_dict[f'blocks.{i}.fc1.weight'].detach().cpu().numpy(), dtype=jnp.float64),
                'b': jnp.array(state_dict[f'blocks.{i}.fc1.bias'].detach().cpu().numpy(), dtype=jnp.float64)
            },
            'fc2': {
                'W': jnp.array(state_dict[f'blocks.{i}.fc2.weight'].detach().cpu().numpy(), dtype=jnp.float64),
                'b': jnp.array(state_dict[f'blocks.{i}.fc2.bias'].detach().cpu().numpy(), dtype=jnp.float64)
            }
        }
        params['blocks'].append(block_params)
    
    # Build forward function
    def forward(params_dict, x):
        """Residual MLP forward pass."""
        # Initial projection
        out = jnp.dot(params_dict['init_proj']['W'], x) + params_dict['init_proj']['b']
        out = smooth_relu(out)
        
        # Residual blocks
        for block in params_dict['blocks']:
            identity = out
            # First layer of block
            res = jnp.dot(block['fc1']['W'], out) + block['fc1']['b']
            res = smooth_relu(res)
            # Second layer of block
            res = jnp.dot(block['fc2']['W'], res) + block['fc2']['b']
            # Residual connection
            out = res + identity
            out = smooth_relu(out)
        
        # Output layer
        out = jnp.dot(params_dict['output']['W'], out) + params_dict['output']['b']
        return out
    
    return forward, params


class JAXNeuralNetworkEvaluator:
    """
    Evaluator for neural network and derivatives using JAX.
    
    This class provides functions to evaluate:
    - Neural network forward pass
    - Objective function (NN-based + regularization)
    - Gradient via JAX automatic differentiation
    - Constraints (box, spherical)
    - Constraint Jacobian via JAX
    - Hessian of Lagrangian via JAX
    
    Designed to be called from Julia via PythonCall, enabling use of
    JAX derivatives with MadNLP solver.
    
    Example:
        >>> evaluator = JAXNeuralNetworkEvaluator(
        ...     model_path="model.pt",
        ...     target=np.array([1, 0, 0]),
        ...     x0=np.zeros(500),
        ...     bounds=(-1.0, 1.0)
        ... )
        >>> grad = evaluator.evaluate_grad(x)
    """
    
    def __init__(
        self,
        model_path: str,
        target: np.ndarray,
        x0: np.ndarray,
        objective_weight: float = 1.0,
        regularization_weight: float = 0.0,
        bounds: Optional[Union[Tuple[float, float], Tuple[np.ndarray, np.ndarray]]] = None,
        radius: Optional[float] = None
    ):
        """
        Initialize JAX evaluator.
        
        Args:
            model_path: Path to PyTorch model (.pt file)
            target: Target output for neural network objective
            x0: Initial point (used as center for regularization)
            objective_weight: Weight for NN objective (default 1.0)
            regularization_weight: Weight for quadratic regularization (default 0.0)
            bounds: Box constraints as (x_min, x_max), scalars or arrays
            radius: Spherical constraint radius (optional)
        """
        # Load model
        print(f"Loading model from {model_path}...")
        self.jax_forward, self.params, self.config = load_pytorch_model_to_jax(model_path)
        print(f"  Model type: {self.config['model_type']}")
        print(f"  Input dim: {self.config['input_dim']}")
        print(f"  Output dim: {self.config['output_dim']}")
        
        # JIT compile forward pass for performance
        self.jax_forward_jit = jax.jit(self.jax_forward)
        
        # Store problem data (convert to JAX arrays)
        self.target = jnp.array(target, dtype=jnp.float64)
        self.x0 = jnp.array(x0, dtype=jnp.float64)
        self.objective_weight = float(objective_weight)
        self.regularization_weight = float(regularization_weight)
        
        # Process bounds
        self.bounds = None
        if bounds is not None:
            if isinstance(bounds, tuple) and len(bounds) == 2:
                x_min, x_max = bounds
                if np.isscalar(x_min):
                    # Scalar bounds
                    x_min = np.full(len(x0), float(x_min))
                    x_max = np.full(len(x0), float(x_max))
                self.bounds = (
                    jnp.array(x_min, dtype=jnp.float64),
                    jnp.array(x_max, dtype=jnp.float64)
                )
        
        self.radius = float(radius) if radius is not None else None
        
        # Dimensions
        self.n = len(x0)
        self.m = self._compute_num_constraints()
        
        print(f"  Problem dimensions: n={self.n}, m={self.m}")
        
        # Compile derivative functions
        print("Compiling JAX derivative functions...")
        self._compile_derivatives()
        print("✓ JAX evaluator ready!")
    
    def _compute_num_constraints(self) -> int:
        """Compute number of constraints."""
        m = 0
        if self.bounds is not None:
            m += 2 * self.n  # Box constraints: x <= x_max and x >= x_min
        if self.radius is not None:
            m += 1  # Spherical constraint
        return m
    
    def _compile_derivatives(self):
        """Pre-compile JAX derivative functions (JIT for performance)."""
        
        # Store params and other data as local variables
        # Use functools.partial to properly bind them instead of closures
        params = self.params
        forward = self.jax_forward
        target = self.target
        x0 = self.x0
        obj_weight = self.objective_weight
        reg_weight = self.regularization_weight
        bounds = self.bounds
        radius = self.radius
        
        # Define objective function that takes ONLY x as argument
        # All other variables are bound via functools.partial
        def obj_func_template(params_arg, forward_fn, target_arg, x0_arg, obj_w, reg_w, x):
            # Neural network objective
            nn_out = forward_fn(params_arg, x)
            f = obj_w * jnp.sum((nn_out - target_arg) ** 2)
            
            # Regularization term
            if reg_w > 0:
                f += reg_w * jnp.sum((x - x0_arg) ** 2)
            
            return f
        
        # Bind all parameters except x using partial
        obj_func = partial(obj_func_template, params, forward, target, x0, obj_weight, reg_weight)
        
        # Constraint function for AD
        def cons_func_template(bounds_arg, x0_arg, radius_arg, x):
            constraints = []
            
            # Box constraints: x <= x_max and x >= x_min
            if bounds_arg is not None:
                x_min, x_max = bounds_arg
                constraints.append(x - x_max)      # x <= x_max → x - x_max <= 0
                constraints.append(x_min - x)      # x >= x_min → x_min - x <= 0
            
            # Spherical constraint: ||x - x0||^2 <= radius^2
            if radius_arg is not None:
                diff = x - x0_arg
                constraints.append(jnp.array([jnp.sum(diff ** 2) - radius_arg ** 2]))
            
            return jnp.concatenate(constraints) if constraints else jnp.array([])
        
        # Bind constraint parameters using partial
        cons_func = partial(cons_func_template, bounds, x0, radius)
        
        # Lagrangian for Hessian computation
        def lagrangian(x, y, obj_weight_arg):
            L = obj_weight_arg * obj_func(x)
            if self.m > 0 and y is not None:
                c = cons_func(x)
                L += jnp.dot(y, c)
            return L
        
        # Compile derivatives
        # Now obj_func takes only x, so jax.grad will differentiate w.r.t. x
        self.obj_func = jax.jit(obj_func)
        self.grad_func = jax.grad(obj_func)  # Don't JIT compile grad for now
        
        if self.m > 0:
            self.cons_func = jax.jit(cons_func)
            self.jac_func = jax.jacobian(cons_func)  # Don't JIT compile jacobian for now
            
            # Hessian of Lagrangian
            def hess_lagrangian_func(x, y, obj_weight_arg):
                """Compute Hessian of Lagrangian."""
                def lag_func(x_):
                    return lagrangian(x_, y, obj_weight_arg)
                return jax.hessian(lag_func)(x)
            
            self.hess_func_lagrangian = hess_lagrangian_func
        
        # Hessian of objective
        self.hess_func_objective = jax.hessian(obj_func)
    
    # ============================================================================
    # Evaluation Functions (Called from Julia)
    # ============================================================================
    
    def evaluate_nn(self, x: np.ndarray) -> np.ndarray:
        """
        Evaluate neural network at x.
        
        Args:
            x: Input point (numpy array)
        
        Returns:
            Neural network output (numpy array)
        """
        x_jax = jnp.array(x, dtype=jnp.float64)
        output = self.jax_forward_jit(self.params, x_jax)
        return np.array(output)
    
    def evaluate_obj(self, x: np.ndarray) -> float:
        """
        Evaluate objective function at x.
        
        Args:
            x: Input point (numpy array)
        
        Returns:
            Objective value (float)
        """
        x_jax = jnp.array(x, dtype=jnp.float64)
        f = self.obj_func(x_jax)
        return float(f)
    
    def evaluate_grad(self, x: np.ndarray) -> np.ndarray:
        """
        Evaluate gradient at x using JAX automatic differentiation.
        
        Args:
            x: Input point (numpy array)
        
        Returns:
            Gradient vector (numpy array)
        """
        x_jax = jnp.array(x, dtype=jnp.float64)
        g = self.grad_func(x_jax)
        return np.array(g)
    
    def evaluate_cons(self, x: np.ndarray) -> np.ndarray:
        """
        Evaluate constraints at x.
        
        Args:
            x: Input point (numpy array)
        
        Returns:
            Constraint values (numpy array, empty if no constraints)
        """
        if self.m == 0:
            return np.array([])
        
        x_jax = jnp.array(x, dtype=jnp.float64)
        c = self.cons_func(x_jax)
        return np.array(c)
    
    def evaluate_jac(self, x: np.ndarray) -> np.ndarray:
        """
        Evaluate constraint Jacobian at x using JAX automatic differentiation.
        
        Args:
            x: Input point (numpy array)
        
        Returns:
            Jacobian matrix (m x n numpy array)
        """
        if self.m == 0:
            return np.array([]).reshape(0, self.n)
        
        x_jax = jnp.array(x, dtype=jnp.float64)
        J = self.jac_func(x_jax)
        return np.array(J)
    
    def evaluate_hess(
        self,
        x: np.ndarray,
        y: Optional[np.ndarray] = None,
        obj_weight: float = 1.0
    ) -> np.ndarray:
        """
        Evaluate Hessian of Lagrangian at x.
        
        Lagrangian: L(x, y) = obj_weight * f(x) + y' * c(x)
        
        Args:
            x: Input point (numpy array)
            y: Lagrange multipliers (numpy array, can be None if no constraints)
            obj_weight: Weight for objective in Lagrangian (default 1.0)
        
        Returns:
            Hessian matrix (n x n symmetric numpy array)
        """
        x_jax = jnp.array(x, dtype=jnp.float64)
        
        if self.m > 0 and y is not None and not np.allclose(y, 0):
            # Hessian of Lagrangian
            y_jax = jnp.array(y, dtype=jnp.float64)
            H = self.hess_func_lagrangian(x_jax, y_jax, obj_weight)
        else:
            # Hessian of objective only
            H = self.hess_func_objective(x_jax)
            H = obj_weight * H
        
        return np.array(H)


def create_evaluator(
    model_path: str,
    target: np.ndarray,
    x0: np.ndarray,
    **kwargs
) -> JAXNeuralNetworkEvaluator:
    """
    Factory function to create JAX evaluator.
    
    Convenient interface for Julia/Python usage.
    
    Args:
        model_path: Path to PyTorch model
        target: Target output for objective
        x0: Initial point
        **kwargs: Additional arguments passed to JAXNeuralNetworkEvaluator
    
    Returns:
        JAXNeuralNetworkEvaluator instance
    
    Example:
        >>> evaluator = create_evaluator(
        ...     "model.pt",
        ...     np.array([1, 0, 0]),
        ...     np.zeros(500),
        ...     bounds=(-1.0, 1.0)
        ... )
    """
    return JAXNeuralNetworkEvaluator(model_path, target, x0, **kwargs)


# =============================================================================
# Standalone Testing
# =============================================================================

if __name__ == '__main__':
    import sys
    
    if len(sys.argv) > 1:
        model_path = sys.argv[1]
        
        # Load model
        jax_forward, params, config = load_pytorch_model_to_jax(model_path)
        print(f"Model loaded: {config['model_type']}, input_dim={config['input_dim']}, output_dim={config['output_dim']}")
        
        # Test evaluator
        x_test = np.random.randn(config['input_dim'])
        target = np.zeros(config['output_dim'])
        target[0] = 1.0
        
        evaluator = create_evaluator(model_path, target, x_test, bounds=(-1.0, 1.0))
        
        # Test evaluations
        obj_val = evaluator.evaluate_obj(x_test)
        grad_val = evaluator.evaluate_grad(x_test)
        
        print(f"Objective: {obj_val:.4f}, Gradient norm: {np.linalg.norm(grad_val):.4f}")
        print("✓ Evaluator test passed!")
    else:
        print("Usage: python jax_nn_evaluator.py <model_path>")

