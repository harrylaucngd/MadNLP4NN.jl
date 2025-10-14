# MadNLP4NN.jl

A Julia/Python framework for constrained optimization on neural networks using MadNLP, featuring dual evaluation backends and flexible automatic differentiation.

## Overview

MadNLP4NN.jl enables solving constrained optimization problems involving trained neural networks by formulating them as nonlinear programs (NLP) and solving with MadNLP, a GPU-capable second-order interior-point solver.

**Problem Formulation:**
```
minimize   f(x, θ)
subject to g(x) ≤ 0
```

where:
- `x ∈ ℝⁿ`: decision variables (optimization inputs)
- `θ`: neural network parameters (trained weights)
- `f(x, θ)`: objective function combining neural network output with custom expressions
- `g(x) ≤ 0`: constraint functions (box constraints, spherical constraints, or custom)

**Key Capabilities:**
- **Dual Evaluation Backends**: Julia/Flux or Python/JAX for neural network evaluation
- **Multiple AD Backends**: ForwardDiff (small-scale), Zygote (medium-scale), or JAX (large-scale)
- **Modular Problem Formulation**: Composable objectives and constraints
- **Advanced Solver Configuration**: Multiple linear solvers, KKT systems, GPU support
- **Smooth ReLU Approximation**: Ensures differentiability for second-order methods
- **Comprehensive Model Training**: PyTorch-based training with hyperparameter optimization

## Features

### Neural Network Evaluation Backends

**1. Julia/Flux Backend (Default)**
- **Neural Network**: Flux.jl inference
- **Automatic Differentiation**: 
  - ForwardDiff: Best for `n < 100` (simple, stable)
  - Zygote: Best for `100 ≤ n < 500` (forward-over-reverse mode, 5-10× faster)
- **Advantages**: Pure Julia, no Python overhead during optimization
- **Use Case**: General-purpose, ideal for medium-scale problems

**2. Python/JAX Backend (Optional)**
- **Neural Network**: JAX inference
- **Automatic Differentiation**: JAX native AD
- **Advantages**: Cross-validation, GPU acceleration via JAX, optimal for `n > 500`
- **Use Case**: Large-scale problems, validation, Python ecosystem integration

Both backends use the same MadNLP solver; the difference is only in how neural network evaluations and derivatives are computed.

### MadNLP Solver Configuration

**Linear Solvers:**
- **CPU**: Umfpack, Lapack, HSL (Ma27/57/77/86/97), MUMPS, Pardiso
- **GPU**: LapackGPU, CuCholesky, CUDSS, RF, GLU (requires MadNLPGPU)

**KKT Systems:**
- Sparse, SparseUnreduced, SparseCondensed
- Dense, DenseCondensed
- Automatic selection based on problem structure

**Performance Options:**
- Multi-threaded BLAS
- Garbage collection control
- Custom convergence tolerances
- GPU acceleration

### Problem Formulation

**Objective Functions:**
- `NeuralNetworkObjective`: Minimize distance to target output
- `QuadraticRegularization`: Regularization term
- `CompositeObjective`: Weighted combination of objectives

**Constraint Functions:**
- `BoxConstraints`: Element-wise bounds `x_min ≤ x ≤ x_max`
- `SphericalConstraint`: L2 ball `‖x - center‖² ≤ radius²`
- `CompositeConstraint`: Multiple constraints combined

### Training Pipeline (Python/PyTorch)

- **Datasets**: Gaussian mixture, nonlinear manifold, MNIST, Fashion-MNIST, CIFAR-10
- **Architectures**: MLP, Residual MLP (configurable sizes: small/medium/large)
- **Hyperparameter Search**: Optuna-based automated tuning
- **Activation**: ReLU (converted to smooth ReLU for optimization)

## Installation

### Prerequisites
- Julia 1.9 or later
- Python 3.8 or later
- CUDA (optional, for GPU acceleration)

### Step 1: Clone Repository
```bash
git clone <repository-url> MadNLP4NN.jl
cd MadNLP4NN.jl
```

### Step 2: Install Python Dependencies
```bash
# Using conda (recommended)
conda create -n madnlp4nn python=3.10
conda activate madnlp4nn
pip install -r python/requirements.txt

# Or using venv
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r python/requirements.txt
```

### Step 3: Install Julia Package
```julia
using Pkg
Pkg.activate(".")
Pkg.instantiate()
```

### Step 4: Configure PythonCall
```julia
using PythonCall

# Option A: Use your conda/venv Python
ENV["JULIA_CONDAPKG_BACKEND"] = "Null"
ENV["JULIA_PYTHONCALL_EXE"] = "/path/to/your/python"  # Replace with your Python path

# Option B: Let PythonCall manage its own environment (simpler)
# Just skip the ENV settings above
```

**Tip**: Add these ENV settings to `~/.julia/config/startup.jl` for persistence.

### Optional: GPU Support
For GPU-accelerated solving, install MadNLPGPU:
```julia
using Pkg
Pkg.add("MadNLPGPU")
```

## Quick Start

### Example 1: Train a Model
```julia
using MadNLP4NN

# Train a model on MNIST
results = train_model(
    dataset_type="mnist",
    model_config="medium_mlp",
    epochs=50,
    batch_size=128,
    seed=42
)
```

### Example 2: Optimization with Julia/Flux Backend
```julia
using MadNLP4NN

# Load trained model
model_path = "output/models/mnist/small_mlp/model_seed42.pt"

# Define problem
input_dim = 784
output_dim = 10
target = zeros(output_dim)
target[1] = 1.0  # Target class 1
x0 = randn(input_dim) .* 0.1

# Create NLP with box constraints (default: Zygote AD)
nlp = create_simple_nlp(
    model_path,
    target,
    x0,
    bounds=(-1.0, 1.0),
    use_python=false  # Use Julia/Flux backend
)

# Solve with MadNLP
result = solve_nlp(nlp, max_iter=1000, tol=1e-4)

println("Status: ", result[:status])
println("Objective: ", result[:objective])
```

### Example 3: Optimization with Python/JAX Backend
```julia
using MadNLP4NN

# Same problem as Example 2, but using Python/JAX
nlp_jax = create_simple_nlp(
    model_path,
    target,
    x0,
    bounds=(-1.0, 1.0),
    use_python=true  # Use Python/JAX backend
)

result_jax = solve_nlp(nlp_jax, max_iter=1000, tol=1e-4)
```

### Example 4: Choosing AD Backend (Julia/Flux)
```julia
using MadNLP4NN

# ForwardDiff: Best for n < 100
nlp_fd = create_simple_nlp(
    model_path, target, x0,
    bounds=(-1.0, 1.0),
    ad_backend=:forwarddiff
)

# Zygote: Best for 100 ≤ n < 500 (default)
nlp_zygote = create_simple_nlp(
    model_path, target, x0,
    bounds=(-1.0, 1.0),
    ad_backend=:zygote
)
```

### Example 5: Advanced Solver Configuration
```julia
using MadNLP4NN
using MadNLP

# Create NLP
nlp = create_simple_nlp(model_path, target, x0, bounds=(-1.0, 1.0))

# Solve with custom solver options
result = solve_nlp(
    nlp,
    max_iter=2000,
    tol=1e-6,
    print_level=MadNLP.INFO,
    linear_solver=MadNLP.UmfpackSolver,
    kkt_system=MadNLP.SparseKKTSystem,
    blas_num_threads=4
)
```

### Example 6: Composite Objectives and Constraints
```julia
using MadNLP4NN

# Custom composite objective
obj = CompositeObjective(
    NeuralNetworkObjective(target, weight=1.0),
    QuadraticRegularization(x0, weight=0.01)
)

# Composite constraints
cons = CompositeConstraint(
    BoxConstraints(fill(-2.0, input_dim), fill(2.0, input_dim)),
    SphericalConstraint(x0, radius=10.0)
)

# Create and solve NLP
nlp = NeuralNetworkNLPModel(model_path, obj, cons, x0)
result = solve_nlp(nlp)
```

## Project Structure

```
MadNLP4NN.jl/
├── Project.toml               # Julia package definition
├── README.md                  # This file
├── src/
│   ├── MadNLP4NN.jl           # Main module (exports)
│   ├── python_interface.jl    # Julia → Python bridge (training, datasets)
│   ├── nlp_interface.jl       # Objective/constraint definitions, model loading
│   └── nlp_model.jl           # NLPModels implementation, solver interface
├── python/
│   ├── requirements.txt       # Python dependencies
│   ├── dataset_constructor.py # Dataset generation
│   ├── neural_network.py      # Network architectures (MLP, ResidualMLP)
│   ├── trainer.py             # Training with hyperparameter search
│   ├── jax_nn_evaluator.py    # JAX backend evaluator
│   └── main.py                # CLI entry point
├── examples/
│   ├── create_dataset_all.jl  # Generate all datasets
│   ├── train_all.jl           # Train all combinations
│   ├── run_opt.jl             # Configurable optimization script
│   └── run_opt_ablation.jl    # Ablation study script
└── output/                    # Generated data and models
    ├── datasets/              # Saved datasets
    ├── models/                # Trained models
    └── optimization_results/  # Optimization results (JSON)
```

## NLP Formulation Details

### Smooth ReLU Activation
To ensure differentiability everywhere (required for second-order methods), ReLU activations are replaced with a smooth approximation:

```
smooth_relu(x; α=1e-3) = α · log(1 + exp(x/α))
```

For small α, this behaves like ReLU but is C^∞ smooth. This enables Hessian computation via automatic differentiation.

### Objective Function Interface
All objectives implement:
```julia
evaluate(obj::AbstractObjectiveFunction, x::Vector, nn_output::Vector) → Float64
```

**Example**: Neural network objective minimizes squared distance to target:
```
f(x) = ‖nn(x) - target‖²
```

### Constraint Function Interface
All constraints implement:
```julia
evaluate(cons::AbstractConstraintFunction, x::Vector) → Vector{Float64}  # g(x) ≤ 0
num_constraints(cons::AbstractConstraintFunction) → Int
```

**Example**: Box constraints are formulated as:
```
g(x) = [x - x_max; x_min - x]  (all components ≤ 0)
```

### Automatic Differentiation
The framework computes:
- **Gradient**: `∇f(x)` via ForwardDiff, Zygote, or JAX
- **Jacobian**: `∇g(x)` via ForwardDiff or JAX
- **Hessian**: `∇²L(x, λ)` where `L(x, λ) = σ·f(x) + λᵀ·g(x)` is the Lagrangian

**Optimization**: For box and spherical constraints, specialized Hessian computation exploits their simple structure (linear or quadratic).

### NLPModels Integration
The `NeuralNetworkNLPModel` implements the full NLPModels.jl interface:
- `obj(nlp, x)`: Objective value
- `grad!(nlp, x, g)`: Objective gradient
- `cons!(nlp, x, c)`: Constraint values
- `jac_coord!(nlp, x, vals)`: Jacobian (coordinate format)
- `hess_coord!(nlp, x, y, vals)`: Hessian of Lagrangian (coordinate format)

This makes it compatible with any NLPModels-based solver, though MadNLP is recommended for performance.

## Backend Selection Guide

| Problem Size | Input Dimension | Recommended Backend | AD Method | Rationale |
|--------------|-----------------|---------------------|-----------|-----------|
| Small | n < 100 | Julia/Flux | ForwardDiff | Simple, stable, sufficient speed |
| Medium | 100 ≤ n < 500 | Julia/Flux | **Zygote** | Forward-over-reverse 5-10× faster |
| Large | n ≥ 500 | Python/JAX | JAX AD | Optimal for large-scale, GPU support |

**All backends use the same MadNLP solver.** The backend only affects neural network evaluation and derivative computation.

## Model Architectures

### MLP (Multi-Layer Perceptron)
- **small_mlp**: [128, 64] hidden layers
- **medium_mlp**: [256, 128, 64] hidden layers
- **large_mlp**: [512, 256, 128, 64] hidden layers

### Residual MLP
- **small_resmlp**: 128-dim, 2 residual blocks
- **medium_resmlp**: 256-dim, 4 residual blocks
- **large_resmlp**: 512-dim, 6 residual blocks

All networks use ReLU activation (converted to smooth ReLU during optimization).

## Datasets

### Synthetic Datasets
1. **Gaussian Mixture**: High-dimensional, multi-modal distributions
   - Parameters: `n_samples`, `input_dim`, `output_dim`, `n_components`
   
2. **Nonlinear Manifold**: Data on nonlinear manifolds in high-dimensional space
   - Parameters: `ambient_dim`, `manifold_dim`, `nonlinearity` (polynomial/trigonometric/mixed)

### Real Datasets
- **MNIST**: 28×28 grayscale handwritten digits (784-dim)
- **Fashion-MNIST**: 28×28 grayscale fashion items (784-dim)
- **CIFAR-10**: 32×32 RGB images (3072-dim)

All real datasets are automatically downloaded and flattened.

## Example Workflows

### Workflow 1: Train and Optimize
```julia
using MadNLP4NN

# Step 1: Train a model
train_model(
    dataset_type="mnist",
    model_config="medium_mlp",
    epochs=50,
    seed=42
)

# Step 2: Define optimization problem
model_path = "output/models/mnist/medium_mlp/model_seed42.pt"
target = zeros(10); target[1] = 1.0
x0 = randn(784) .* 0.1

# Step 3: Solve
nlp = create_simple_nlp(model_path, target, x0, bounds=(-1.0, 1.0))
result = solve_nlp(nlp, max_iter=1000, tol=1e-4)
```

### Workflow 2: Compare Backends
```julia
using MadNLP4NN

# Create problem
model_path = "output/models/mnist/medium_mlp/model_seed42.pt"
target = zeros(10); target[1] = 1.0
x0 = randn(784) .* 0.1

# Solve with Julia/Flux (Zygote)
nlp_flux = create_simple_nlp(model_path, target, x0, bounds=(-1.0, 1.0), use_python=false)
result_flux = solve_nlp(nlp_flux, max_iter=1000, tol=1e-4)

# Solve with Python/JAX
nlp_jax = create_simple_nlp(model_path, target, x0, bounds=(-1.0, 1.0), use_python=true)
result_jax = solve_nlp(nlp_jax, max_iter=1000, tol=1e-4)

# Compare results
println("Flux objective: ", result_flux[:objective])
println("JAX objective: ", result_jax[:objective])
```

### Workflow 3: Batch Training
```julia
using MadNLP4NN

# Train all combinations
results = train_all_combinations(
    dataset_types=["mnist", "fashionmnist"],
    model_configs=["small_mlp", "medium_mlp"],
    epochs=50,
    run_hparam_search=true,
    n_trials=30
)
```

## Troubleshooting

### PythonCall Configuration
If you encounter Python import errors:
```julia
using PythonCall
ENV["JULIA_CONDAPKG_BACKEND"] = "Null"
ENV["JULIA_PYTHONCALL_EXE"] = "/path/to/your/python"
```

### GPU Solver Not Found
For GPU solvers, ensure you have:
```julia
using Pkg
Pkg.add("MadNLPGPU")
```
And a CUDA-capable GPU with appropriate drivers.

### NaN/Inf in Gradients
- Check that your initial point `x0` is reasonable
- Reduce smoothing parameter: `smooth_relu(x; α=1e-4)`
- Use a smaller tolerance: `solve_nlp(nlp, tol=1e-3)`

## References

- **MadNLP.jl**: [https://github.com/MadNLP/MadNLP.jl](https://github.com/MadNLP/MadNLP.jl)
- **NLPModels.jl**: [https://github.com/JuliaSmoothOptimizers/NLPModels.jl](https://github.com/JuliaSmoothOptimizers/NLPModels.jl)
- **PythonCall.jl**: [https://github.com/cjdoris/PythonCall.jl](https://github.com/cjdoris/PythonCall.jl)
- **PyTorch**: [https://pytorch.org/](https://pytorch.org/)
- **JAX**: [https://github.com/google/jax](https://github.com/google/jax)
- **Optuna**: [https://optuna.org/](https://optuna.org/)
- **Zygote.jl**: [https://fluxml.ai/Zygote.jl/](https://fluxml.ai/Zygote.jl/)
