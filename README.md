# MadNLP4NN.jl

**A Julia/Python framework for optimization on neural networks using MadNLP**

## Overview

MadNLP4NN.jl enables constrained optimization problems involving neural networks by formulating them as nonlinear programs (NLP) and solving with MadNLP, a GPU-accelerated second-order NLP solver.

**Problem Formulation:**
```
minimize  f(x, θ)
subject to  g(x) ≤ 0
```

where:
- `x`: decision variables (optimization inputs)
- `θ`: neural network parameters (trained model)
- `f(x, θ)`: objective function using neural network output and x-related expressions
- `g(x)`: constraint functions (e.g., bounds on x, spherical constraints)

The framework provides:
- **Modular objective functions**: Combine neural network outputs with custom expressions
- **Flexible constraints**: Box constraints, spherical constraints, or custom formulations
- **Smooth optimization**: Handles non-smooth ReLU activations appropriately
- **Dual backend support**: Choose between Julia/Flux or Python/JAX for NN evaluation and derivatives
- **Automatic differentiation**: Gradients and Hessians computed via ForwardDiff.jl or JAX
- **Second-order optimization**: Leverages MadNLP's interior-point method for efficiency

## Features

### Python Framework (PyTorch-based)
- ✅ **Synthetic dataset generation** with non-trivial distributions:
  - Gaussian mixture models (high-dimensional, multi-modal)
  - Nonlinear manifold datasets (polynomial, trigonometric, mixed)
- ✅ **Real-life datasets**: MNIST, Fashion-MNIST, CIFAR-10
- ✅ **Neural network architectures** (all with ReLU activation):
  - Multi-layer perceptron (MLP)
  - Residual MLP (ResidualMLP)
  - Configurable sizes (small, medium, large)
- ✅ **Automated hyperparameter search** using Optuna
- ✅ **Comprehensive training pipeline** with validation
- ✅ **Model saving** with architecture and parameters

### Julia Interface
- ✅ **Seamless Python integration** via PythonCall.jl
- ✅ **Dataset creation** from Julia
- ✅ **Model training** from Julia
- ✅ **Batch processing** for all dataset/model combinations
- ✅ **NLP formulation** with modular objectives and constraints
- ✅ **MadNLP integration** with automatic differentiation
- ✅ **Dual backend support**:
  - **Flux + ForwardDiff** (default): Pure Julia implementation
  - **JAX + JAX AD** (optional): Python/JAX for NN evaluation and derivatives
- ✅ **Model loading** from PyTorch to Flux.jl or JAX
- ✅ **Smooth ReLU** handling for differentiability

## Installation

### Prerequisites
- Julia 1.9 or later
- Python 3.8 or later
- CUDA (optional, for GPU acceleration)

### Step 1: Clone the Repository
```bash
cd ~/Desktop/Optimization
git clone <repository-url> MadNLP4NN.jl
cd MadNLP4NN.jl
```

### Step 2: Set Up Python Environment

#### Option A: Using conda (recommended)
```bash
conda create -n madnlp4nn python=3.10
conda activate madnlp4nn
pip install -r python/requirements.txt
```

#### Option B: Using venv
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r python/requirements.txt
```

### Step 3: Install Julia Package
```julia
using Pkg
Pkg.activate(".")
Pkg.instantiate()
```

### Step 4: Configure PythonCall
PythonCall.jl needs to know which Python to use:

```julia
using PythonCall
# Point to your conda/venv python
ENV["JULIA_CONDAPKG_BACKEND"] = "Null"  # Use system Python
ENV["JULIA_PYTHONCALL_EXE"] = "/path/to/your/python"  # e.g., conda env python

# Or let PythonCall manage it (installs packages automatically)
# Just use the package, it will set up its own environment
```

## Quick Start

### Example 1: Create a Dataset

```julia
using MadNLP4NN

# Create a Gaussian mixture dataset
metadata = create_dataset(
    dataset_type="gaussian_mixture",
    n_samples=10000,
    input_dim=784,
    output_dim=10,
    n_components=20,
    seed=42
)

println("Dataset created: ", metadata)
```

### Example 2: Train a Model

```julia
using MadNLP4NN

# Train a medium MLP on MNIST
results = train_model(
    dataset_type="mnist",
    model_config="medium_mlp",
    epochs=50,
    batch_size=128,
    run_hparam_search=false,  # Use default hyperparameters
    seed=42
)

println("Training complete! Best accuracy: ", results[:best_val_acc])
```

### Example 3: Train All Combinations

```julia
using MadNLP4NN

# Train all combinations of datasets and models
all_results = train_all_combinations(
    dataset_types=["gaussian_mixture", "mnist"],
    model_configs=["medium_mlp", "medium_resmlp"],
    epochs=50,
    n_samples=10000,  # For synthetic datasets
    input_dim=784,
    output_dim=10,
    run_hparam_search=true,  # Enable hyperparameter search
    n_trials=30,
    seed=42
)

# Results are saved in ./output/models/
```

### Example 4: Load a Trained Model

```julia
using MadNLP4NN

model_path = "./output/models/mnist_medium_mlp/model_seed42.pt"
state_dict, config, args, history = load_trained_model(model_path)

println("Model architecture: ", config)
println("Best validation accuracy: ", maximum(history["val_acc"]))
```

### Example 5: NLP Optimization with Neural Networks

```julia
using MadNLP4NN

# Load a trained model
model_path = "output/models/GM_n10000_d500_c10_comp10_s123/small_mlp/model_seed123.pt"

# Define optimization problem
input_dim = 500
output_dim = 10

# Target: minimize distance to class 0
target = zeros(output_dim)
target[1] = 1.0

# Initial point
x0 = randn(input_dim) .* 0.1

# Create NLP with box constraints (default: Julia/Flux backend)
nlp = create_simple_nlp(
    model_path,
    target,
    x0,
    bounds=(-1.0, 1.0)
)

# Solve with MadNLP
result = solve_nlp(nlp, max_iter=100, tol=1e-4)

println("Status: ", result[:status])
println("Objective: ", result[:objective])
println("Solution norm: ", norm(result[:solution]))
```

### Example 6: Using Python/JAX Backend

```julia
using MadNLP4NN

# Same problem as Example 5, but using Python/JAX for NN evaluation
model_path = "output/models/GM_n10000_d500_c10_comp10_s123/small_mlp/model_seed123.pt"

target = zeros(10)
target[1] = 1.0
x0 = randn(500) .* 0.1

# Create NLP with Python/JAX backend
nlp_jax = create_simple_nlp(
    model_path,
    target,
    x0,
    bounds=(-1.0, 1.0),
    use_python=true  # Use JAX for NN evaluation and derivatives!
)

# Solve with MadNLP (same solver, different evaluation backend)
result_jax = solve_nlp(nlp_jax, max_iter=100, tol=1e-4)

println("JAX Backend - Status: ", result_jax[:status])
println("JAX Backend - Objective: ", result_jax[:objective])
```

**Note**: Both backends use the same MadNLP solver. The only difference is how neural network evaluations and derivatives are computed:
- **Flux backend** (default): Uses Flux.jl for inference and ForwardDiff.jl for derivatives
- **JAX backend** (`use_python=true`): Uses JAX (Python) for inference and JAX AD for derivatives

### Example 7: Custom Objectives and Constraints

```julia
using MadNLP4NN

# Load model
model_path = "output/models/dataset/model.pt"

# Create composite objective
obj = CompositeObjective(
    NeuralNetworkObjective(target, weight=1.0),
    QuadraticRegularization(x0, weight=0.01)
)

# Create composite constraints
cons = CompositeConstraint(
    BoxConstraints(fill(-2.0, n), fill(2.0, n)),
    SphericalConstraint(x0, radius=1.0)
)

# Create and solve NLP
nlp = NeuralNetworkNLPModel(model_path, obj, cons, x0)
result = solve_nlp(nlp)
```

## Project Structure

```
MadNLP4NN.jl/
├── Project.toml              # Julia package definition
├── README.md                 # This file
├── docs/
│   ├── TUTORIAL.md          # Detailed tutorials
│   └── JULIA_PYTHON.md      # Julia/Python integration guide
├── src/
│   ├── MadNLP4NN.jl        # Main module
│   ├── python_interface.jl  # Julia → Python interface
│   ├── nlp_interface.jl     # Objective/constraint definitions
│   └── nlp_model.jl         # NLPModels implementation
├── python/
│   ├── requirements.txt     # Python dependencies
│   ├── dataset_constructor.py  # Dataset generation
│   ├── neural_network.py    # Network architectures
│   ├── trainer.py           # Training with hyperparameter search
│   └── main.py              # Main entry point
├── examples/
│   ├── basic_usage.jl       # Basic examples
│   ├── dataset_all.jl       # Generate all datasets
│   ├── train_all_parallel.jl # Train all combinations (parallel)
│   ├── nlp_optimization.jl  # NLP optimization (redirects to evaluation)
│   └── nlp_evaluation.jl    # Comprehensive NLP evaluation
└── output/                   # Generated data and models
    ├── datasets/            # Saved datasets
    └── models/              # Saved models
```

## Dataset Types

### Synthetic Datasets

#### 1. Gaussian Mixture
High-dimensional Gaussian mixture with multiple components per class.
- **Purpose**: Creates non-uniform, complex distributions
- **Parameters**: `n_samples`, `input_dim`, `output_dim`, `n_components`
- **Use case**: Testing robustness to multi-modal distributions

```julia
create_dataset(
    dataset_type="gaussian_mixture",
    n_samples=10000,
    input_dim=784,
    output_dim=10,
    n_components=20
)
```

#### 2. Nonlinear Manifold
Data lying on a nonlinear manifold embedded in high-dimensional space.
- **Purpose**: Simulates real data with lower intrinsic dimensionality
- **Parameters**: `ambient_dim`, `manifold_dim`, `nonlinearity`
- **Nonlinearity types**: "polynomial", "trigonometric", "mixed"

```julia
create_dataset(
    dataset_type="nonlinear_manifold",
    n_samples=10000,
    input_dim=784,
    manifold_dim=50,
    output_dim=10,
    nonlinearity="polynomial"
)
```

### Real Datasets

- **MNIST**: 28×28 grayscale images of handwritten digits (flattened to 784)
- **Fashion-MNIST**: 28×28 grayscale images of fashion items
- **CIFAR-10**: 32×32 RGB images (flattened to 3072)

## Model Configurations

### MLP Architectures
- `small_mlp`: [128, 64] hidden layers
- `medium_mlp`: [256, 128, 64] hidden layers
- `large_mlp`: [512, 256, 128, 64] hidden layers

### Residual MLP Architectures
- `small_resmlp`: 128-dim, 2 residual blocks
- `medium_resmlp`: 256-dim, 4 residual blocks
- `large_resmlp`: 512-dim, 6 residual blocks

All networks use ReLU activation functions (required for NLP formulation).

## Hyperparameter Search

The framework uses Optuna for automated hyperparameter search:

**Tuned hyperparameters:**
- Learning rate (log scale: 1e-4 to 1e-2)
- Weight decay (log scale: 1e-6 to 1e-3)
- Optimizer (Adam, AdamW, SGD)

**Fixed parameters** (set via command line):
- Network architecture
- Batch size
- Number of epochs
- Dataset parameters

To enable hyperparameter search:
```julia
train_model(
    dataset_type="mnist",
    model_config="medium_mlp",
    run_hparam_search=true,
    n_trials=50,
    hparam_search_epochs=20,  # Shorter for search
    epochs=100  # Full training with best params
)
```

## NLP Formulation

The NLP interface is now fully implemented! The framework provides a modular approach to defining optimization problems with neural networks.

### Key Components

**1. Model Loading**
- Converts PyTorch models to Flux.jl format
- Supports MLP and Residual MLP architectures
- Preserves trained weights and biases

**2. Smooth ReLU Activation**
- Uses `smooth_relu(x) = α * log(1 + exp(x/α))` approximation
- Ensures differentiability everywhere
- Parameter α controls smoothness (default: 1e-3)

**3. Modular Objective Functions**
- `NeuralNetworkObjective`: Minimize distance to target output
- `QuadraticRegularization`: Regularization term
- `CompositeObjective`: Combine multiple objectives
- Easy to create custom objectives

**4. Modular Constraint Functions**
- `BoxConstraints`: Element-wise bounds on x
- `SphericalConstraint`: L2 ball constraint
- `CompositeConstraint`: Combine multiple constraints
- Easy to create custom constraints

**5. Automatic Differentiation**
- Gradients computed via ForwardDiff.jl
- Hessians computed via ForwardDiff.jl
- Supports second-order optimization methods

**6. NLPModels Integration**
- Full implementation of NLPModels.jl interface
- Callbacks: `obj`, `grad!`, `cons!`, `jac_coord!`, `hess_coord!`
- Compatible with all NLPModels-based solvers

**7. MadNLP Solver**
- Interior-point method for constrained optimization
- Configurable tolerances and iteration limits
- Support for different linear solvers

### Usage

```julia
using MadNLP4NN

# Load model
model_path = "output/models/dataset/model.pt"

# Define objective
target = zeros(10)
target[1] = 1.0
obj = NeuralNetworkObjective(target)

# Define constraints
cons = BoxConstraints(fill(-1.0, 500), fill(1.0, 500))

# Create NLP model
x0 = randn(500) .* 0.1
nlp = NeuralNetworkNLPModel(model_path, obj, cons, x0)

# Solve
result = solve_nlp(nlp)
```

See `examples/nlp_evaluation.jl` for comprehensive examples.

## Julia/Python Integration

This project demonstrates a clean Julia/Python workflow:

1. **Python** handles:
   - Deep learning (PyTorch)
   - Dataset generation
   - Model training
   - Hyperparameter search (Optuna)

2. **Julia** handles:
   - High-level orchestration
   - NLP formulation (TODO)
   - Optimization (MadNLP)
   - Analysis and visualization

3. **Communication**:
   - Julia calls Python via PythonCall.jl
   - Python saves models/data to disk
   - Julia loads models for NLP formulation

See `docs/JULIA_PYTHON.md` for detailed integration patterns.

## Performance Considerations

### For Training (Python)
- Use GPU if available (automatically detected)
- Adjust batch size based on GPU memory
- Use larger networks for complex datasets
- Enable hyperparameter search for best results

### For Optimization (Julia/MadNLP) - TODO
- Use GPU linear solver for large problems
- Exploit sparsity in Jacobian/Hessian
- Start with box constraints (L∞) - simpler than L2
- Use warm starts from simpler attacks

## Contributing

This project is ready for use and open for contributions. Key areas for enhancement:

1. **Additional Objective Functions**
   - Custom loss functions for specific applications
   - Multi-objective optimization
   - Robust optimization objectives

2. **Additional Constraint Types**
   - Non-convex constraints
   - Probabilistic constraints
   - Differential constraints

3. **Network Architectures**
   - Convolutional networks
   - Attention mechanisms
   - Batch normalization handling

4. **Optimization Enhancements**
   - GPU acceleration for large-scale problems
   - Sparse Hessian exploitation
   - Warm-start strategies
   - Adaptive smoothing parameter for ReLU

5. **Additional Datasets**
   - More synthetic distribution types
   - Domain-specific datasets
   - Custom dataset loaders

6. **Applications**
   - Adversarial example generation
   - Input optimization for specific outputs
   - Certified robustness verification
   - Neural network interpretability

## License

[Add your license here]

## Citation

If you use this framework in your research, please cite:

```bibtex
@software{madnlp4nn2024,
  title={MadNLP4NN.jl: Adversarial Optimization on Neural Networks},
  author={[Your Name]},
  year={2024},
  url={https://github.com/[your-username]/MadNLP4NN.jl}
}
```

## References

- [MadNLP.jl](https://github.com/MadNLP/MadNLP.jl): GPU-accelerated NLP solver
- [NLPModels.jl](https://github.com/JuliaSmoothOptimizers/NLPModels.jl): NLP modeling
- [PythonCall.jl](https://github.com/cjdoris/PythonCall.jl): Julia-Python integration
- [PyTorch](https://pytorch.org/): Deep learning framework
- [Optuna](https://optuna.org/): Hyperparameter optimization

## Contact

[Add your contact information]
