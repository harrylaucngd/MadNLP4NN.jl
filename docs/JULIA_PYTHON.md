# Julia/Python Integration Guide

This document explains how Julia and Python work together in MadNLP4NN.jl, the conventions used, and best practices for mixed Julia/Python projects.

## Overview

MadNLP4NN.jl is a **Julia-first** project that leverages Python for specific tasks (deep learning). The general architecture is:

```
User
  ↓
Julia (Main Interface)
  ↓
PythonCall.jl (Bridge)
  ↓
Python (PyTorch, Optuna)
  ↓
Disk (Saved Models/Data)
  ↓
Julia (NLP Optimization)
```

## Why Julia + Python?

### Python Strengths
- **PyTorch**: Mature, well-tested deep learning framework
- **Optuna**: Powerful hyperparameter optimization
- **Ecosystem**: Extensive ML/DL libraries
- **GPU Support**: Excellent CUDA integration

### Julia Strengths
- **Performance**: Near-C speed for optimization code
- **MadNLP**: GPU-accelerated second-order NLP solver
- **Automatic Differentiation**: Powerful AD ecosystem
- **Mathematical Notation**: Clean, readable code
- **Type System**: Strong typing for numerical computing

### The Best of Both Worlds
By combining them, we get:
1. Easy model training (Python/PyTorch)
2. Fast optimization (Julia/MadNLP)
3. Clean user interface (Julia)
4. Interoperability without overhead

## Integration Approaches

### 1. PythonCall.jl (Our Choice)

PythonCall.jl is the modern, recommended way to call Python from Julia.

**Advantages:**
- Fast (minimal overhead)
- Stable (well-maintained)
- Pythonic (natural Python syntax in Julia)
- Flexible (can use any Python package)

**Installation:**
```julia
using Pkg
Pkg.add("PythonCall")
```

**Basic Usage:**
```julia
using PythonCall

# Import Python modules
sys = pyimport("sys")
np = pyimport("numpy")

# Call Python functions
arr = np.array([1, 2, 3, 4, 5])
mean_val = np.mean(arr)

# Convert to Julia
julia_arr = pyconvert(Vector, arr)
```

**In MadNLP4NN:**
```julia
# Import custom Python modules
sys = pyimport("sys")
pylist(sys.path).insert(0, "/path/to/python/modules")

dataset_module = pyimport("dataset_constructor")
dataset = dataset_module.GaussianMixtureDataset(
    n_samples=10000,
    input_dim=784,
    output_dim=10
)
```

### 2. Calling Python Scripts (Our Approach for Training)

For complex workflows, we call Python scripts directly:

```julia
# Build command-line arguments
cmd_args = [
    "--dataset_type", "mnist",
    "--model_config", "medium_mlp",
    "--epochs", "100"
]

# Run Python script
run(`python python/main.py $cmd_args`)

# Load results from disk
results = JSON3.read(read("output/results.json", String))
```

**Advantages:**
- Clean separation of concerns
- Easy to debug (can run Python script independently)
- Persistent results (saved to disk)
- No memory overhead (processes separate)

**Disadvantages:**
- Requires disk I/O
- Less interactive than direct calls

## Python Environment Setup

### Option 1: Conda Environment (Recommended)

Conda provides isolated environments and better CUDA support.

```bash
# Create environment
conda create -n madnlp4nn python=3.10
conda activate madnlp4nn

# Install packages
pip install -r python/requirements.txt

# Or use conda for PyTorch with CUDA
conda install pytorch torchvision pytorch-cuda=11.8 -c pytorch -c nvidia
pip install optuna tensorboard scikit-learn
```

**Using from Julia:**
```julia
using PythonCall

# Point to conda Python
ENV["JULIA_CONDAPKG_BACKEND"] = "Null"
ENV["JULIA_PYTHONCALL_EXE"] = "/path/to/conda/envs/madnlp4nn/bin/python"

# Now use PythonCall
sys = pyimport("sys")
```

### Option 2: System Python with venv

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install packages
pip install -r python/requirements.txt
```

**Using from Julia:**
```julia
ENV["JULIA_PYTHONCALL_EXE"] = "/path/to/venv/bin/python"
```

### Option 3: Let PythonCall Manage (Easiest)

PythonCall can manage its own Python environment:

```julia
using PythonCall

# PythonCall will install packages automatically
# when you first import them
torch = pyimport("torch")  # Installs torch if needed
```

**Note:** This may not give you the exact versions you want, especially for CUDA-enabled PyTorch.

## Best Practices for Mixed Projects

### 1. Clear Separation of Concerns

**Python does:**
- Data preprocessing
- Model training
- Hyperparameter search
- Neural network forward/backward passes

**Julia does:**
- High-level orchestration
- NLP formulation
- Optimization (MadNLP)
- Analysis and visualization

### 2. Communication via Files

For large data (models, datasets), use files:

```python
# Python: Save model
torch.save({
    'model_state_dict': model.state_dict(),
    'config': config
}, 'model.pt')

# Also save as numpy for Julia
import numpy as np
params = {k: v.cpu().numpy() for k, v in model.state_dict().items()}
np.savez('model_params.npz', **params)
```

```julia
# Julia: Load model
using NPZ
params = npzread("model_params.npz")
weights = params["layer1.weight"]
```

### 3. JSON for Metadata

Use JSON for configuration and metadata:

```python
import json

metadata = {
    'model_type': 'mlp',
    'hidden_dims': [256, 128, 64],
    'input_dim': 784,
    'output_dim': 10
}

with open('metadata.json', 'w') as f:
    json.dump(metadata, f, indent=2)
```

```julia
using JSON3

metadata = JSON3.read(read("metadata.json", String))
model_type = metadata.model_type
hidden_dims = metadata.hidden_dims
```

### 4. Command-Line Interface

Expose Python functionality via CLI:

```python
# python/main.py
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_type', type=str, required=True)
parser.add_argument('--epochs', type=int, default=100)
args = parser.parse_args()

# Do work...
```

```julia
# Julia
run(`python python/main.py --dataset_type mnist --epochs 100`)
```

### 5. Error Handling

Wrap Python calls in try-catch:

```julia
try
    run(`python python/main.py $args`)
catch e
    @error "Python script failed: $e"
    # Handle error
end
```

## Common Patterns

### Pattern 1: Import and Call

```julia
using PythonCall

function create_dataset_python(n_samples::Int)
    # Import module
    dataset_module = pyimport("dataset_constructor")
    
    # Create dataset
    dataset = dataset_module.GaussianMixtureDataset(
        n_samples=n_samples,
        input_dim=784,
        output_dim=10
    )
    
    # Convert to Julia (if needed)
    data = pyconvert(Array, dataset.data)
    labels = pyconvert(Array, dataset.labels)
    
    return data, labels
end
```

### Pattern 2: Script Execution

```julia
function train_model_python(dataset_type::String, model_config::String; kwargs...)
    # Build arguments
    args = [
        "--dataset_type", dataset_type,
        "--model_config", model_config
    ]
    
    for (key, val) in kwargs
        push!(args, "--$(key)")
        push!(args, string(val))
    end
    
    # Run script
    script_path = joinpath(@__DIR__, "..", "python", "main.py")
    run(`python $script_path $args`)
    
    # Load results
    results_file = "output/results.json"
    return JSON3.read(read(results_file, String))
end
```

### Pattern 3: Load Saved Data

```julia
function load_pytorch_model(path::String)
    # Use Python to load PyTorch model
    torch = pyimport("torch")
    checkpoint = torch.load(path)
    
    # Convert to Julia types
    config = pyconvert(Dict, checkpoint["model_config"])
    state_dict = pyconvert(Dict, checkpoint["model_state_dict"])
    
    # Or load numpy version directly
    using NPZ
    params = npzread(replace(path, ".pt" => "_params.npz"))
    
    return params, config
end
```

## Directory Structure Convention

```
ProjectName.jl/
├── Project.toml          # Julia package
├── src/
│   ├── ProjectName.jl   # Main Julia module
│   ├── interface.jl     # Julia code
│   └── python_interface.jl  # Julia → Python bridge
├── python/
│   ├── requirements.txt  # Python dependencies
│   ├── module1.py       # Python modules
│   └── main.py          # Python CLI entry point
├── test/
│   ├── runtests.jl      # Julia tests
│   └── test_python.py   # Python tests (optional)
└── docs/
    └── README.md
```

**Key points:**
- Julia code in `src/`
- Python code in `python/` subdirectory
- Clear separation, but can call each other
- Both can be tested independently

## Environment Variables

Useful environment variables for Julia/Python integration:

```bash
# Python executable for PythonCall
export JULIA_PYTHONCALL_EXE="/path/to/python"

# Conda package backend
export JULIA_CONDAPKG_BACKEND="Null"  # Use system Python

# Python path
export PYTHONPATH="/path/to/your/python/modules:$PYTHONPATH"

# CUDA (if using GPU)
export CUDA_VISIBLE_DEVICES=0
```

## Troubleshooting

### Problem: PythonCall can't find modules

**Solution:**
```julia
sys = pyimport("sys")
pylist(sys.path).insert(0, "/path/to/your/python/modules")
```

### Problem: Import errors for Python packages

**Solution:**
```bash
# Check which Python PythonCall is using
julia -e 'using PythonCall; println(pyimport("sys").executable)'

# Install packages to that Python
/path/to/that/python -m pip install -r requirements.txt
```

### Problem: PyTorch CUDA not available

**Solution:**
```bash
# Install CUDA-enabled PyTorch
conda install pytorch torchvision pytorch-cuda=11.8 -c pytorch -c nvidia

# Or with pip (check PyTorch website for correct command)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### Problem: Slow PythonCall startup

**Solution:**
```julia
# Use system Python instead of CondaPkg
ENV["JULIA_CONDAPKG_BACKEND"] = "Null"
ENV["JULIA_PYTHONCALL_EXE"] = "/path/to/fast/python"
```

## Performance Tips

1. **Minimize data transfer**: Keep large arrays in Python or save to disk
2. **Use numpy for interop**: Convert PyTorch → numpy → Julia
3. **Batch operations**: Do many operations in one Python call
4. **Cache imports**: Import modules once, reuse them
5. **Use separate processes**: For long-running tasks, use scripts

## Testing

### Test Python code independently:
```bash
cd python
python -m pytest test_*.py
```

### Test Julia code independently:
```julia
using Pkg
Pkg.test("MadNLP4NN")
```

### Test integration:
```julia
using Test
using MadNLP4NN

@testset "Python Integration" begin
    # Test dataset creation
    metadata = create_dataset(
        dataset_type="gaussian_mixture",
        n_samples=100
    )
    @test metadata isa Dict
    
    # Test model training
    results = train_model(
        dataset_type="mnist",
        model_config="small_mlp",
        epochs=1  # Quick test
    )
    @test results isa Dict
end
```

## Summary

**For MadNLP4NN.jl:**

1. **Use PythonCall.jl** for direct Python calls (loading data, small tasks)
2. **Use Python scripts** for heavy tasks (training, hyperparameter search)
3. **Use files** for communication (models, datasets, results)
4. **Use JSON** for metadata and configuration
5. **Keep it simple**: Clear separation, minimal coupling

This approach gives you:
- ✅ Easy to develop (work in each language naturally)
- ✅ Easy to debug (can run Python/Julia separately)
- ✅ Easy to maintain (clear boundaries)
- ✅ Fast (minimal overhead)
- ✅ Flexible (can change either side easily)

## Further Reading

- [PythonCall.jl Documentation](https://cjdoris.github.io/PythonCall.jl/stable/)
- [Julia Calling C/Python](https://docs.julialang.org/en/v1/manual/calling-c-and-fortran-code/)
- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
- [MadNLP.jl Documentation](https://madnlp.github.io/MadNLP.jl/stable/)
