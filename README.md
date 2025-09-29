# MadNLP4NN.jl

**A Julia/Python framework for adversarial optimization on neural networks using MadNLP**

## Overview

MadNLP4NN.jl enables optimization-based adversarial attacks on neural networks by formulating the problem as a nonlinear program (NLP) and solving it with MadNLP, a GPU-accelerated second-order NLP solver.

**Problem Formulation:**
```
maximize  ||f(x; θ) - y||²
subject to  ||x - x*|| ≤ ε
```

where:
- `f(x; θ)` is a trained neural network with parameters θ
- `x*` is the original input point
- `y` is the target output
- `ε` is the perturbation radius

This approach leverages second-order optimization to find adversarial examples more efficiently than traditional gradient-based methods.

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
- 🚧 **NLP formulation** (placeholder - requires implementation)
- 🚧 **MadNLP integration** (placeholder - requires implementation)

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
│   └── nlp_interface.jl     # NLP formulation (TODO)
├── python/
│   ├── requirements.txt     # Python dependencies
│   ├── dataset_constructor.py  # Dataset generation
│   ├── neural_network.py    # Network architectures
│   ├── trainer.py           # Training with hyperparameter search
│   └── main.py              # Main entry point
├── examples/
│   ├── basic_usage.jl       # Basic examples
│   ├── train_all.jl         # Train all combinations
│   └── nlp_optimization.jl  # NLP optimization (TODO)
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

## NLP Formulation (TODO)

The next step is to implement the NLP interface that formulates the adversarial problem for MadNLP. See `src/nlp_interface.jl` for detailed implementation hints.

**Key components needed:**
1. Neural network forward pass
2. Objective function: `||f(x; θ) - y||²`
3. Gradient computation (backpropagation)
4. Hessian computation (exact or Hessian-vector products)
5. Constraint implementation (L∞ or L2)
6. NLPModels.jl integration
7. MadNLP solver configuration

**Recommended approach:**
```julia
# Load trained model
model_path = "./output/models/mnist_medium_mlp/model_seed42.pt"

# Define adversarial problem
x_star = randn(784)  # Original input
y_target = randn(10)  # Target output
epsilon = 0.1  # Perturbation radius

# Create NLP model (TO BE IMPLEMENTED)
nlp = create_adversarial_nlp(model_path, x_star, y_target, epsilon=epsilon)

# Solve with MadNLP (TO BE IMPLEMENTED)
result = solve_adversarial(nlp)

# Get adversarial example
x_adv = result.solution
```

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

This project is in active development. Key areas for contribution:

1. **NLP Interface Implementation** (Priority!)
   - Forward pass through saved PyTorch models
   - Gradient and Hessian computation
   - NLPModels.jl integration
   - MadNLP solver configuration

2. **Additional Datasets**
   - More synthetic distribution types
   - Support for non-image data
   - Custom dataset loaders

3. **Network Architectures**
   - Convolutional networks (after NLP interface works)
   - Other activation functions
   - Batch normalization handling

4. **Optimization Enhancements**
   - Multiple constraint types
   - Multi-target adversarial examples
   - Certified adversarial robustness

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
