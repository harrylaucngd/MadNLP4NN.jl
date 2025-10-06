# Setup Guide for MadNLP4NN.jl

Quick setup guide to get started with MadNLP4NN.jl.

## Prerequisites

- **Julia 1.9+**: Download from [julialang.org](https://julialang.org/downloads/)
- **Python 3.8+**: Anaconda or system Python
- **Git**: For cloning the repository
- **CUDA** (optional): For GPU acceleration

## Step-by-Step Setup

### 1. Clone Repository

```bash
cd ~/Desktop/Optimization
# If you haven't cloned yet:
# git clone <your-repo-url> MadNLP4NN.jl

cd MadNLP4NN.jl
```

### 2. Set Up Python Environment

#### Using Conda (Recommended)

```bash
# Create environment
conda create -n madnlp4nn python=3.10
conda activate madnlp4nn

# Install PyTorch with CUDA (if you have GPU)
conda install pytorch torchvision pytorch-cuda=11.8 -c pytorch -c nvidia

# Install other dependencies (including JAX for dual backend support)
pip install -r python/requirements.txt

# Note: requirements.txt includes JAX for the Python/JAX backend option
# If you only want to use the Julia/Flux backend, JAX is optional

# Note the Python path for later
which python
# Example output: /Users/yourusername/anaconda3/envs/madnlp4nn/bin/python
```

#### Using pip + venv

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r python/requirements.txt

# Note the Python path
which python
```

### 3. Set Up Julia Package

```bash
# Start Julia in the project directory
julia
```

In Julia REPL:

```julia
# Activate the project
using Pkg
Pkg.activate(".")

# Install dependencies
Pkg.instantiate()

# This will install:
# - PythonCall.jl (for Python integration)
# - MadNLP.jl (for optimization)
# - NLPModels.jl (for NLP modeling)
# - JSON3.jl (for data exchange)å
# - LinearAlgebra.jl (for linear algebra operations)

# Wait for installation to complete (may take a few minutes)
```

### 4. Configure PythonCall

```julia
# Still in Julia REPL

using PythonCall

# Option A: Use your conda/venv Python
ENV["JULIA_CONDAPKG_BACKEND"] = "Null"
ENV["JULIA_PYTHONCALL_EXE"] = "/path/to/your/conda/or/venv/python"
# Replace with the path you noted in Step 2

# Option B: Let PythonCall manage Python (simpler but less control)
# Just skip the ENV settings above

# Test Python integration
torch = pyimport("torch")
println("PyTorch version: ", torch.__version__)
println("CUDA available: ", torch.cuda.is_available())

# If you see PyTorch version and CUDA status, you're good!
```

### 5. Test Installation

```julia
# Load the package
using MadNLP4NN

println("MadNLP4NN.jl loaded successfully!")

# Try creating a small dataset
metadata = create_dataset(
    dataset_type="gaussian_mixture",
    n_samples=1000,
    input_dim=100,
    output_dim=5,
    seed=42
)

println("Dataset created: ", metadata)

# If this works, installation is complete! ✓
```

### 6. Make Configuration Persistent (Optional)

To avoid setting `ENV` variables every time, add to your Julia startup file:

```bash
# Create/edit Julia startup file
mkdir -p ~/.julia/config
nano ~/.julia/config/startup.jl
```

Add these lines:

```julia
# Python configuration for MadNLP4NN
ENV["JULIA_CONDAPKG_BACKEND"] = "Null"
ENV["JULIA_PYTHONCALL_EXE"] = "/path/to/your/python"
```

Now Python will be configured automatically when you start Julia.

## Verification Checklist

Run this to verify everything is working:

```julia
using MadNLP4NN
using PythonCall

# Check Python
println("Testing Python integration...")
torch = pyimport("torch")
println("✓ PyTorch ", torch.__version__)
if torch.cuda.is_available()
    println("✓ CUDA available: ", torch.cuda.get_device_name(0))
else
    println("⚠ CUDA not available (CPU only)")
end

# Check Julia packages
println("\nTesting Julia packages...")
using MadNLP
println("✓ MadNLP loaded")
using NLPModels
println("✓ NLPModels loaded")
using JSON3
println("✓ JSON3 loaded")

# Check MadNLP4NN functions
println("\nTesting MadNLP4NN functions...")
metadata = create_dataset(
    dataset_type="gaussian_mixture",
    n_samples=100,
    input_dim=50,
    output_dim=3,
    seed=42
)
println("✓ Dataset creation works")

println("\n" * "="^60)
println("All checks passed! You're ready to use MadNLP4NN.jl")
println("="^60)
```

## Troubleshooting

### Problem: "cannot open shared object file: libpython.so"

**Solution:**
```julia
# Set Python executable explicitly
ENV["JULIA_PYTHONCALL_EXE"] = "/full/path/to/python"
```

### Problem: "PyTorch module not found"

**Solution:**
```bash
# Make sure you're in the right environment
conda activate madnlp4nn  # or: source venv/bin/activate

# Install PyTorch
pip install torch torchvision

# Verify installation
python -c "import torch; print(torch.__version__)"
```

### Problem: "PythonCall takes forever to load"

**Solution:**
```julia
# Use system Python instead of CondaPkg
ENV["JULIA_CONDAPKG_BACKEND"] = "Null"
ENV["JULIA_PYTHONCALL_EXE"] = "/usr/local/bin/python3"
```

### Problem: "CUDA out of memory"

**Solution:**
- Use smaller batch sizes (e.g., `batch_size=32`)
- Use smaller models (e.g., `model_config="small_mlp"`)
- Train on CPU (PyTorch will automatically fall back)

### Problem: Module not found in Python

**Solution:**
```julia
using PythonCall
sys = pyimport("sys")
# Add the python directory
python_dir = joinpath(@__DIR__, "python")
pylist(sys.path).insert(0, python_dir)
```

## Dual Backend Support (Julia/Flux vs Python/JAX)

MadNLP4NN supports two backends for neural network evaluation and automatic differentiation:

### Backend 1: Julia/Flux (Default)
- **Inference**: Flux.jl
- **Derivatives**: ForwardDiff.jl
- **Pros**: Pure Julia, no Python dependency for optimization
- **Usage**: Default, no special flags needed

### Backend 2: Python/JAX (Optional)
- **Inference**: JAX (Python)
- **Derivatives**: JAX automatic differentiation
- **Pros**: Cross-validation, GPU support via JAX
- **Usage**: Add `use_python=true` flag

**Both backends use the same MadNLP solver in Julia!**

### Example: Choosing Backend

```julia
using MadNLP4NN

model_path = "output/models/dataset/model.pt"
target = [1.0, 0.0, 0.0, ...]
x0 = randn(500)

# Option 1: Julia/Flux backend (default)
nlp_flux = create_simple_nlp(
    model_path, target, x0,
    bounds=(-1, 1),
    use_python=false  # or omit (default)
)

# Option 2: Python/JAX backend
nlp_jax = create_simple_nlp(
    model_path, target, x0,
    bounds=(-1, 1),
    use_python=true  # Use JAX!
)

# Both solve with MadNLP
result_flux = solve_nlp(nlp_flux)
result_jax = solve_nlp(nlp_jax)
```

### When to Use Each Backend?

**Use Julia/Flux (default) when:**
- You want pure Julia workflow
- You don't need to validate derivatives
- You're comfortable with the Julia ecosystem

**Use Python/JAX when:**
- You want to cross-validate with Julia implementation
- You need GPU acceleration via JAX
- You want to compare ForwardDiff vs JAX AD performance
- You're more comfortable with Python for debugging

## Next Steps

Once setup is complete:

1. **Try the examples:**
   ```bash
   julia examples/basic_usage.jl
   ```

2. **Try NLP optimization:**
   ```bash
   julia examples/nlp_evaluation.jl
   ```

3. **Read the tutorial:**
   ```bash
   cat docs/TUTORIAL.md
   ```

4. **Train your first model:**
   ```julia
   using MadNLP4NN
   
   results = train_model(
       dataset_type="mnist",
       model_config="medium_mlp",
       epochs=50
   )
   ```

5. **Try both backends:**
   ```julia
   # See Example 5 and 6 in README.md
   ```

## Getting Help

- Check the documentation in `docs/`
- Review examples in `examples/`
- See README.md for dual backend examples
- Check PROJECT_SUMMARY.md for detailed feature list

Happy coding! 🚀
