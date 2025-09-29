# MadNLP4NN.jl Tutorial

Complete tutorial for using MadNLP4NN.jl, from installation to adversarial optimization.

## Table of Contents

1. [Installation and Setup](#installation-and-setup)
2. [Creating Datasets](#creating-datasets)
3. [Training Models](#training-models)
4. [Hyperparameter Search](#hyperparameter-search)
5. [Batch Training](#batch-training)
6. [Loading and Inspecting Models](#loading-and-inspecting-models)
7. [NLP Formulation (Future)](#nlp-formulation-future)

## Installation and Setup

### Step 1: Install Julia

Download and install Julia 1.9+ from [julialang.org](https://julialang.org/downloads/).

### Step 2: Set Up Python Environment

```bash
# Create conda environment
conda create -n madnlp4nn python=3.10
conda activate madnlp4nn

# Install PyTorch with CUDA (if you have GPU)
conda install pytorch torchvision pytorch-cuda=11.8 -c pytorch -c nvidia

# Install other dependencies
cd ~/Desktop/Optimization/MadNLP4NN.jl
pip install -r python/requirements.txt
```

### Step 3: Set Up Julia Package

```bash
cd ~/Desktop/Optimization/MadNLP4NN.jl
julia
```

```julia
using Pkg
Pkg.activate(".")
Pkg.instantiate()
```

### Step 4: Configure PythonCall

```julia
using PythonCall

# Tell PythonCall to use your conda environment
ENV["JULIA_CONDAPKG_BACKEND"] = "Null"
ENV["JULIA_PYTHONCALL_EXE"] = "/path/to/conda/envs/madnlp4nn/bin/python"

# On macOS/Linux, find path with: which python (in activated env)
# On Windows: where python

# Test it works
torch = pyimport("torch")
println("PyTorch version: ", torch.__version__)
println("CUDA available: ", torch.cuda.is_available())
```

### Step 5: Verify Installation

```julia
using MadNLP4NN

# This should show no errors
println("MadNLP4NN.jl loaded successfully!")
```

## Creating Datasets

### Synthetic Dataset: Gaussian Mixture

This creates a high-dimensional dataset with non-trivial distribution:

```julia
using MadNLP4NN

# Create Gaussian mixture dataset
metadata = create_dataset(
    dataset_type="gaussian_mixture",
    output_dir="./output",
    n_samples=10000,
    input_dim=784,        # 28×28 = 784 (MNIST-like)
    output_dim=10,        # 10 classes
    n_components=20,      # 20 Gaussian components per class
    seed=42
)

println("Dataset created!")
println("  Input dimension: ", metadata[:input_dim])
println("  Output dimension: ", metadata[:output_dim])
println("  Number of samples: ", metadata[:n_samples])
```

**What's happening:**
- Creates 10 classes
- Each class has 20 Gaussian components (multi-modal)
- Data is in 784-dimensional space
- Normalized to [0, 1]
- Saved to `./output/datasets/gaussian_mixture/`

### Synthetic Dataset: Nonlinear Manifold

This creates data on a nonlinear manifold:

```julia
metadata = create_dataset(
    dataset_type="nonlinear_manifold",
    output_dir="./output",
    n_samples=10000,
    input_dim=784,        # Ambient dimension
    manifold_dim=50,      # Intrinsic dimension
    output_dim=10,
    nonlinearity="polynomial",  # or "trigonometric", "mixed"
    seed=42
)
```

**What's happening:**
- Data lies on a 50-dimensional manifold
- Embedded in 784-dimensional space
- Nonlinear mapping (polynomial, trigonometric, or mixed)
- More realistic than uniform distributions

### Real Dataset: MNIST

```julia
metadata = create_dataset(
    dataset_type="mnist",
    output_dir="./output"
)

println("MNIST loaded!")
println("  Train + Test size: ", metadata[:dataset_name])
```

**Other real datasets:**
- `"fashionmnist"`: Fashion items (clothing)
- `"cifar10"`: Color images (32×32×3)

## Training Models

### Basic Training

```julia
using MadNLP4NN

# Train a medium MLP on Gaussian mixture data
results = train_model(
    dataset_type="gaussian_mixture",
    model_config="medium_mlp",
    output_dir="./output",
    epochs=50,
    batch_size=64,
    learning_rate=1e-3,
    weight_decay=1e-4,
    optimizer="adam",
    seed=42
)

println("Training complete!")
println("  Best validation accuracy: ", results[:best_val_acc])
println("  Model saved to: ", results[:model_path])
```

**Model configurations available:**
- `"small_mlp"`: [128, 64] - for quick experiments
- `"medium_mlp"`: [256, 128, 64] - good balance
- `"large_mlp"`: [512, 256, 128, 64] - for complex datasets
- `"small_resmlp"`: Residual connections, 2 blocks
- `"medium_resmlp"`: Residual connections, 4 blocks
- `"large_resmlp"`: Residual connections, 6 blocks

### Training on MNIST

```julia
# First, create/download the dataset
create_dataset(dataset_type="mnist", output_dir="./output")

# Then train
results = train_model(
    dataset_type="mnist",
    model_config="medium_mlp",
    epochs=50,
    batch_size=128,
    learning_rate=1e-3,
    optimizer="adam",
    seed=42
)

# Should achieve ~97-98% accuracy on MNIST
println("MNIST accuracy: ", results[:best_val_acc])
```

## Hyperparameter Search

Instead of manually setting hyperparameters, let Optuna find the best ones:

```julia
using MadNLP4NN

# Train with hyperparameter search
results = train_model(
    dataset_type="mnist",
    model_config="medium_mlp",
    run_hparam_search=true,
    n_trials=50,                    # Try 50 different hyperparameter combinations
    hparam_search_epochs=20,        # Train each trial for 20 epochs
    epochs=100,                     # After search, train best config for 100 epochs
    batch_size=128,
    seed=42
)

println("Best hyperparameters found:")
println("  ", results[:train_args][:best_hyperparams])
println("Final validation accuracy: ", results[:best_val_acc])
```

**What Optuna tunes:**
- Learning rate (1e-4 to 1e-2, log scale)
- Weight decay (1e-6 to 1e-3, log scale)
- Optimizer (Adam, AdamW, SGD)

**What you still control:**
- Network architecture
- Dataset
- Batch size
- Number of epochs

**Typical workflow:**
1. Run hyperparameter search with `n_trials=30-50`
2. Each trial trains for `hparam_search_epochs=20` epochs
3. Best hyperparameters are selected
4. Final model trains with best hyperparameters for `epochs=100` epochs
5. Model is saved with its hyperparameters

## Batch Training

Train multiple dataset/model combinations in one go:

```julia
using MadNLP4NN

# Define what to train
dataset_types = ["gaussian_mixture", "nonlinear_manifold", "mnist"]
model_configs = ["medium_mlp", "medium_resmlp"]

# Train all combinations (3 datasets × 2 models = 6 models)
all_results = train_all_combinations(
    dataset_types=dataset_types,
    model_configs=model_configs,
    output_dir="./output",
    epochs=50,
    n_samples=10000,              # For synthetic datasets
    input_dim=784,
    output_dim=10,
    batch_size=64,
    run_hparam_search=false,      # Set to true for better results (slower)
    seed=42
)

println("Trained $(length(all_results)) models:")
for result in all_results
    println("  $(result.dataset) + $(result.model): $(result.results[:best_val_acc])")
end
```

**This will create:**
```
output/
├── datasets/
│   ├── gaussian_mixture/
│   ├── nonlinear_manifold/
│   └── mnist/
└── models/
    ├── gaussian_mixture_medium_mlp/
    ├── gaussian_mixture_medium_resmlp/
    ├── nonlinear_manifold_medium_mlp/
    ├── nonlinear_manifold_medium_resmlp/
    ├── mnist_medium_mlp/
    └── mnist_medium_resmlp/
```

**With hyperparameter search (recommended but slower):**

```julia
all_results = train_all_combinations(
    dataset_types=["mnist", "fashionmnist"],
    model_configs=["medium_mlp", "large_mlp"],
    epochs=100,
    run_hparam_search=true,
    n_trials=30,
    hparam_search_epochs=20,
    batch_size=128
)
```

This takes longer but gives better models!

## Loading and Inspecting Models

### Load a Trained Model

```julia
using MadNLP4NN

# Path to saved model
model_path = "./output/models/mnist_medium_mlp/model_seed42.pt"

# Load it
state_dict, config, train_args, history = load_trained_model(model_path)

# Inspect architecture
println("Model architecture:")
println("  Type: ", config["model_type"])
println("  Input dim: ", config["input_dim"])
println("  Hidden dims: ", config["hidden_dims"])
println("  Output dim: ", config["output_dim"])

# Inspect training
println("\nTraining info:")
println("  Dataset: ", train_args["dataset_type"])
println("  Epochs: ", train_args["epochs"])
println("  Best hyperparameters: ", train_args["best_hyperparams"])

# Inspect performance
println("\nPerformance:")
println("  Final train accuracy: ", history["train_acc"][end])
println("  Final val accuracy: ", history["val_acc"][end])
println("  Best val accuracy: ", maximum(history["val_acc"]))
```

### Access Model Parameters

Parameters are saved in two formats:

1. **PyTorch format** (`.pt` file): Full checkpoint
2. **NumPy format** (`.npz` file): Just parameters, easier for Julia

```julia
using NPZ

# Load numpy parameters
params_path = "./output/models/mnist_medium_mlp/model_seed42_params.npz"
params = npzread(params_path)

# Access specific layers
layer1_weights = params["network.0.weight"]  # Shape: (256, 784)
layer1_biases = params["network.0.bias"]     # Shape: (256,)

println("Layer 1 weights shape: ", size(layer1_weights))
println("Layer 1 biases shape: ", size(layer1_biases))

# List all parameters
println("\nAll parameters:")
for key in keys(params)
    println("  $key: ", size(params[key]))
end
```

**Parameter naming convention:**
- `network.0.weight`: First layer weights
- `network.0.bias`: First layer biases  
- `network.2.weight`: Second layer weights (index 1 is ReLU)
- `network.2.bias`: Second layer biases
- etc.

### Visualize Training History

```julia
using Plots

# Load model
_, _, _, history = load_trained_model(model_path)

# Extract arrays
train_loss = history["train_loss"]
val_loss = history["val_loss"]
train_acc = history["train_acc"]
val_acc = history["val_acc"]

# Plot loss
p1 = plot(train_loss, label="Train Loss", xlabel="Epoch", ylabel="Loss")
plot!(p1, val_loss, label="Val Loss")

# Plot accuracy
p2 = plot(train_acc, label="Train Acc", xlabel="Epoch", ylabel="Accuracy")
plot!(p2, val_acc, label="Val Acc")

# Combine plots
plot(p1, p2, layout=(2, 1), size=(800, 600))
savefig("training_history.png")
```

## NLP Formulation (Future)

**Note:** This section describes the planned functionality. Implementation is TODO.

Once the NLP interface is complete, you'll be able to:

### 1. Load a Trained Model for Optimization

```julia
using MadNLP4NN
using MadNLP

# Load trained model
model_path = "./output/models/mnist_medium_mlp/model_seed42.pt"

# Define the adversarial problem
x_star = randn(784)      # Original input (or load from dataset)
y_target = randn(10)     # Target output (or load from dataset)
epsilon = 0.1            # Perturbation radius

# Create NLP model
nlp = create_adversarial_nlp(
    model_path,
    x_star,
    y_target,
    epsilon=epsilon,
    constraint_type="l2"  # or "linf"
)
```

### 2. Solve with MadNLP

```julia
# Configure solver
solver_options = Dict(
    "max_iter" => 1000,
    "tol" => 1e-6,
    "print_level" => MadNLP.INFO,
    "linear_solver" => MadNLPMumps  # Or MadNLPGPU for GPU
)

# Solve
result = solve_adversarial(nlp, solver_options=solver_options)

# Get adversarial example
x_adv = result[:solution]
obj_value = result[:objective]
status = result[:status]

println("Optimization complete!")
println("  Status: $status")
println("  Objective value: $obj_value")
println("  Perturbation norm: ", norm(x_adv - x_star))
```

### 3. Verify Adversarial Example

```julia
# Check perturbation constraint
perturbation = x_adv - x_star
@assert norm(perturbation) <= epsilon + 1e-6

# Compute network outputs (requires Python)
py_model = load_pytorch_model_for_inference(model_path)
y_original = forward_pass(py_model, x_star)
y_adversarial = forward_pass(py_model, x_adv)

println("Original output: ", y_original)
println("Adversarial output: ", y_adversarial)
println("Distance to target: ", norm(y_adversarial - y_target))
```

### 4. Batch Adversarial Optimization

```julia
# Find adversarial examples for multiple points
x_stars = [randn(784) for _ in 1:100]
y_targets = [randn(10) for _ in 1:100]

adversarial_results = []

for (x_star, y_target) in zip(x_stars, y_targets)
    nlp = create_adversarial_nlp(model_path, x_star, y_target, epsilon=0.1)
    result = solve_adversarial(nlp)
    push!(adversarial_results, result)
end

# Analyze success rate
success_rate = count(r -> r[:status] == :solved, adversarial_results) / length(adversarial_results)
println("Success rate: $(success_rate * 100)%")
```

## Tips and Best Practices

### 1. Start Small

```julia
# For testing, use small datasets and few epochs
results = train_model(
    dataset_type="gaussian_mixture",
    model_config="small_mlp",
    n_samples=1000,      # Small dataset
    epochs=10,           # Few epochs
    batch_size=32
)
```

### 2. Use Hyperparameter Search for Important Models

```julia
# For models you care about, use hyperparameter search
results = train_model(
    dataset_type="mnist",
    model_config="medium_mlp",
    run_hparam_search=true,
    n_trials=50,
    hparam_search_epochs=20,
    epochs=100
)
```

### 3. Save Intermediate Results

```julia
# Train and immediately check results
results = train_model(...)

# Save results summary
using JSON3
open("my_results.json", "w") do f
    JSON3.write(f, results)
end
```

### 4. GPU Usage

```julia
# Check if GPU is available
using PythonCall
torch = pyimport("torch")
if torch.cuda.is_available()
    println("GPU available: ", torch.cuda.get_device_name(0))
    # Models will automatically use GPU
else
    println("No GPU, using CPU")
end
```

### 5. Organize Your Experiments

```julia
# Use descriptive output directories
for seed in [42, 43, 44]
    results = train_model(
        dataset_type="mnist",
        model_config="medium_mlp",
        output_dir="./experiments/mnist_study",
        epochs=50,
        seed=seed
    )
end
```

## Common Issues and Solutions

### Issue: "Python module not found"

```julia
# Solution: Add Python directory to path
using PythonCall
sys = pyimport("sys")
pylist(sys.path).insert(0, "/full/path/to/MadNLP4NN.jl/python")
```

### Issue: "CUDA out of memory"

```julia
# Solution: Reduce batch size
results = train_model(
    dataset_type="mnist",
    model_config="medium_mlp",
    batch_size=32,  # Instead of 128
    epochs=50
)
```

### Issue: Training is slow

```julia
# Solution 1: Use smaller model
model_config="small_mlp"

# Solution 2: Reduce dataset size (for synthetic data)
n_samples=5000

# Solution 3: Fewer epochs
epochs=30

# Solution 4: Skip hyperparameter search for experiments
run_hparam_search=false
```

## Next Steps

1. **Complete the NLP interface** (see `src/nlp_interface.jl`)
2. **Run adversarial optimization experiments**
3. **Compare with gradient-based attacks** (PGD, FGSM)
4. **Analyze adversarial robustness**
5. **Try certified defenses**

Happy optimizing! 🚀
