# MadNLP4NN.jl

A GPU-oriented reduced-space interior-point framework for constrained optimization over trained neural networks, with two proposal-aligned validation case studies.

## Overview

MadNLP4NN.jl solves nonlinear programs of the form

```
min_{x ∈ ℝⁿ}  F(x, N_θ(x))
subject to    c(x, N_θ(x)) ≤ 0,   ℓ ≤ x ≤ u
```

where `N_θ` is one or more fixed trained neural networks and `x` is the decision variable.  The optimization methodology is fixed: a **reduced-space interior-point solver** (MadNLP.jl) with derivatives supplied by a **JIT-compiled JAX backend** through PythonCall.jl, with a Julia/Flux+Zygote fallback.

Two structured case studies from the accompanying proposal are fully implemented:

| Case Study | Problem | Key constraints |
|---|---|---|
| **I: Darcy Inversion** | Recover permeability field from FNO pressure surrogate | box + budget (1ᵀx ≤ B) + smoothness (xᵀLx ≤ τ) |
| **II: Pareto Tracing** | Weighted-sum constrained front over learned f₁, f₂, h | learned feasibility h(x) ≤ 0 |

## Key Capabilities

- **Reduced-space NLP**: KKT system size depends only on the decision variable dimension, not the network size
- **Dual AD backends**: Python/JAX (primary, GPU-capable, JIT-compiled) and Julia/Flux+Zygote (baseline)
- **Explicit CPU/NVIDIA GPU paths**: single `device="cpu"|"gpu"` flag controls both JAX device and MadNLP linear solver selection
- **Composable problem types**: `ProblemSpec` decouples network paths, objectives, and constraints from the solver
- **New constraint types**: `BudgetConstraint`, `SmoothnessConstraint`, `LearnedFeasibilityConstraint`
- **New objective types**: `SurrogateInversionObjective`, `WeightedScalarizationObjective`
- **FNO2D in JAX**: pure functional implementation supporting checkpoint loading from our format and neuraloperator-style checkpoints

## Project Structure

```
MadNLP4NN.jl/
├── Project.toml
├── README.md
├── startup.jl
├── configs/
│   ├── darcy_config.toml          # Darcy case study configuration
│   └── pareto_config.toml         # Pareto tracing configuration
├── src/
│   ├── MadNLP4NN.jl               # Main module
│   ├── nlp_interface.jl           # Objectives, constraints, model loading
│   ├── problem_spec.jl            # ProblemSpec + new types + Laplacian helpers
│   ├── nlp_model.jl               # NeuralNetworkNLPModel, ProblemSpec constructor
│   ├── python_interface.jl        # Julia → Python bridge (training)
│   ├── darcy_problem.jl           # DarcyProblemConfig + staged constructors
│   └── pareto_problem.jl          # ParetoProblemConfig + sweep + quality metrics
├── python/
│   ├── jax_nn_evaluator.py        # Backward-compatible facade + factory functions
│   ├── darcy_data.py              # Darcy data generation + FNO training
│   ├── pareto_data.py             # Pareto surrogate data + training
│   ├── backend/
│   │   ├── device.py              # DeviceManager (CPU / NVIDIA GPU)
│   │   └── model_loader.py        # Unified checkpoint → JAX loader
│   ├── evaluators/
│   │   ├── base.py                # BaseEvaluator interface
│   │   ├── standard.py            # Single-network target-matching
│   │   ├── multi_network.py       # Multi-network Pareto evaluator
│   │   └── darcy.py               # FNO inversion evaluator
│   └── models/
│       ├── mlp.py                 # MLP / ResMLP builders
│       ├── resnet.py              # CIFAR-style ResNet builder
│       └── fno.py                 # FNO2D builder (new)
├── examples/
│   ├── run_opt.jl                 # Configurable single-run (original)
│   ├── run_opt_ablation.jl        # Classification ablation study (original)
│   ├── run_darcy_inversion.jl     # Case Study I: staged Darcy experiments
│   ├── run_pareto_tracing.jl      # Case Study II: Pareto sweep
│   ├── run_benchmarks.jl          # Unified CPU/GPU benchmark harness
│   ├── create_dataset_all.jl
│   ├── train_all.jl
│   └── train_all_parallel.jl
└── test/
    └── runtests.jl                # Regression + correctness tests
```

## Installation

### Prerequisites
- Julia 1.9+
- Python 3.9+ with JAX, PyTorch, NumPy, SciPy
- CUDA (optional, for NVIDIA GPU paths)

### Step 1: Julia packages
```julia
using Pkg; Pkg.activate("."); Pkg.instantiate()
```

### Step 2: Python dependencies
```bash
pip install -r python/requirements.txt
# For GPU JAX:
pip install "jax[cuda12]"
# For FNO training:
pip install optax
```

### Step 3: Configure PythonCall
```julia
# In startup.jl or before using MadNLP4NN:
ENV["JULIA_CONDAPKG_BACKEND"] = "Null"
ENV["JULIA_PYTHONCALL_EXE"] = "/path/to/your/python"
```

### Step 4 (optional): GPU solver
```julia
using Pkg; Pkg.add("MadNLPGPU")
```

## Quick Start

### Run tests
```julia
using Pkg; Pkg.test()
```

### Case Study I: Darcy FNO Inversion
```julia
# Stage A only (auto-trains a small FNO if no checkpoint found)
julia examples/run_darcy_inversion.jl --stage A --grid 32

# All stages with an existing FNO checkpoint
julia examples/run_darcy_inversion.jl --stage all --grid 64 \
    --fno_path output/darcy/models/fno_darcy_64x64_dv32_seed42.npz
```

### Case Study II: Pareto Tracing
```julia
# Stage B coarse sweep (auto-trains surrogates)
julia examples/run_pareto_tracing.jl --stage B --n_dim 20

# Stage C dense sweep + multi-start
julia examples/run_pareto_tracing.jl --stage C --n_dim 20 --n_alpha 21 --n_starts 5
```

### Unified Benchmark
```julia
# Quick smoke test
julia examples/run_benchmarks.jl --quick

# Full benchmark on CPU
julia examples/run_benchmarks.jl --case all --device cpu

# Both CPU and GPU
julia examples/run_benchmarks.jl --case all --device both
```

## Programmatic API

### Darcy Inversion (Stage A)
```julia
using MadNLP4NN

cfg = DarcyProblemConfig(
    "output/darcy/models/fno_darcy_64x64_dv32_seed42.npz",
    target_pressure,   # Vector{Float64}, length n
    x0;                # initial permeability field
    x_lb = 0.0,
    x_ub = 5.0,
    lambda_reg = 1e-2,
    device = "gpu"     # or "cpu"
)
nlp = create_darcy_nlp(cfg)
result = solve_nlp(nlp, max_iter=500, tol=1e-4)
```

### Darcy Inversion (Stage B: with budget + smoothness)
```julia
cfg_b = DarcyProblemConfig(
    fno_path, y_tar, x0;
    x_lb=0.0, x_ub=5.0, lambda_reg=1e-2,
    budget = 50.0,          # 1ᵀx ≤ B
    tau = 1e3,              # xᵀLx ≤ τ  (L = 2-D Laplacian, auto-built)
    grid_nx = 64,
    device = "cpu",
)
nlp = create_darcy_nlp(cfg_b)
```

### Pareto Sweep
```julia
results = pareto_front_sweep(
    "output/models/pareto/f1_net_n20_seed42.pt",
    "output/models/pareto/f2_net_n20_seed42.pt",
    x0;
    h_path = "output/models/pareto/h_net_n20_seed42.pt",
    alpha_values = collect(range(0, 1; length=21)),
    x_lb = -1.0,
    x_ub = 1.0,
    device = "cpu",
    warm_start = true,
)
# results is a Vector of Dicts with :alpha, :f1_val, :f2_val, :h_val, :feasible, …
hv = pareto_hypervolume(
    [(r[:f1_val], r[:f2_val]) for r in results if r[:feasible]],
    (2.0, 2.0)  # reference point
)
```

### Generic single-network (original API, unchanged)
```julia
nlp = create_simple_nlp(
    "output/models/mnist/small_mlp/model_seed42.pt",
    target, x0;
    bounds = (-1.0, 1.0),
    use_python = true,
    ad_backend = :zygote,
)
result = solve_nlp(nlp, max_iter=1000, tol=1e-4)
```

### ProblemSpec (generic multi-network constructor)
```julia
spec = ProblemSpec(
    ["f1.pt", "f2.pt", "h.pt"],
    WeightedScalarizationObjective(0.3),
    x0;
    constraints = CompositeConstraint(
        BoxConstraints(fill(-1.0, n), fill(1.0, n)),
        LearnedFeasibilityConstraint(; network_index=3)
    ),
    problem_type = "pareto",
)
nlp = NeuralNetworkNLPModel(spec; device="gpu")
```

## Constraint and Objective Types

| Type | Formulation |
|---|---|
| `BoxConstraints(x_min, x_max)` | `x_min ≤ x ≤ x_max` (→ lvar/uvar) |
| `SphericalConstraint(c, r)` | `‖x - c‖² ≤ r²` |
| `BudgetConstraint(B)` | `1ᵀx ≤ B` |
| `SmoothnessConstraint(L, τ)` | `xᵀLx ≤ τ` |
| `LearnedFeasibilityConstraint(; network_index, threshold)` | `h_{θ}(x) ≤ threshold` |
| `CompositeConstraint(c1, c2, …)` | stacked |
| `NeuralNetworkObjective(target)` | `‖N(x) − target‖²` |
| `QuadraticRegularization(c; weight)` | `weight ‖x − c‖²` |
| `SurrogateInversionObjective(target, x0; weight, reg_weight)` | `w‖N(x)−target‖² + λ‖x−x₀‖²` |
| `WeightedScalarizationObjective(α)` | `α f₁(x) + (1−α) f₂(x)` |
| `CompositeObjective(o1, o2, …)` | sum |

## Backend Selection

| Problem size | Recommended backend | AD method |
|---|---|---|
| n < 100 | Julia/Flux | ForwardDiff |
| 100 ≤ n < 500 | Julia/Flux | Zygote (default) |
| n ≥ 500 | Python/JAX | JAX native (JIT, GPU) |
| FNO / multi-network | Python/JAX | JAX native (required) |

## References

- **MadNLP.jl**: Shin, Pacaud, Zavala — GPU-accelerated interior-point methods
- **ExaModels.jl / condensed IPM**: Pacaud & Shin (arXiv:2403.15913, arXiv:2405.14236)
- **FNO**: Li et al. (ICLR 2021)
- **NLPModels.jl**: JuliaSmoothOptimizers
- **JAX**: Bradbury et al.
- **Zygote.jl / Flux.jl**: Innes
