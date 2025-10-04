# Basic Usage Examples for MadNLP4NN.jl

using MadNLP4NN
using Printf

println("="^80)
println("MadNLP4NN.jl - Basic Usage Examples")
println("="^80)

# =============================================================================
# Example 1: Create a synthetic dataset
# =============================================================================
println("\n[Example 1] Creating a Gaussian Mixture Dataset")
println("-"^80)

metadata = create_dataset(
    dataset_type="gaussian_mixture",
    output_dir="./output",
    n_samples=5000,
    input_dim=784,
    output_dim=10,
    n_components=20,
    seed=42
)

println("✓ Dataset created successfully!")
println("  Location: ./output/datasets/gaussian_mixture/")
println("  Samples: $(metadata[:n_samples])")
println("  Input dimension: $(metadata[:input_dim])")
println("  Output dimension: $(metadata[:output_dim])")

# =============================================================================
# Example 2: Create a real dataset (MNIST)
# =============================================================================
println("\n[Example 2] Loading MNIST Dataset")
println("-"^80)

metadata_mnist = create_dataset(
    dataset_type="mnist",
    output_dir="./output"
)

println("✓ MNIST loaded successfully!")
println("  Type: $(metadata_mnist["dataset_type"])")
println("  Input dimension: $(metadata_mnist["input_dim"])")

# =============================================================================
# Example 3: Train a simple model (no hyperparameter search)
# =============================================================================
println("\n[Example 3] Training a Small MLP on Gaussian Mixture")
println("-"^80)
println("This will take a few minutes...")

results = train_model(
    dataset_type="gaussian_mixture",
    model_config="small_mlp",
    output_dir="./output",
    epochs=30,
    batch_size=64,
    learning_rate=1e-3,
    weight_decay=1e-4,
    optimizer="adam",
    run_hparam_search=false,
    seed=42
)

println("✓ Training complete!")
println("  Best validation accuracy: $(@sprintf("%.4f", results[:best_val_acc]))")
println("  Model saved to: $(results[:model_path])")

# =============================================================================
# Example 4: Load and inspect a trained model
# =============================================================================
println("\n[Example 4] Loading and Inspecting Trained Model")
println("-"^80)

model_path = results[:model_path]
state_dict, config, train_args, history = load_trained_model(model_path)

println("✓ Model loaded successfully!")
println("\nModel Architecture:")
println("  Type: $(config["model_type"])")
println("  Input dimension: $(config["input_dim"])")
println("  Hidden dimensions: $(config["hidden_dims"])")
println("  Output dimension: $(config["output_dim"])")

println("\nTraining Information:")
println("  Dataset: $(train_args["dataset_type"])")
println("  Epochs trained: $(train_args["epochs"])")
println("  Batch size: $(train_args["batch_size"])")
println("  Optimizer: $(train_args["best_hyperparams"]["optimizer"])")
println("  Learning rate: $(@sprintf("%.2e", train_args["best_hyperparams"]["learning_rate"]))")

println("\nTraining History:")
println("  Final train accuracy: $(@sprintf("%.4f", history["train_acc"][end]))")
println("  Final val accuracy: $(@sprintf("%.4f", history["val_acc"][end]))")
println("  Best val accuracy: $(@sprintf("%.4f", maximum(history["val_acc"])))")

# =============================================================================
# Summary
# =============================================================================
println("\n" * "="^80)
println("Examples Complete!")
println("="^80)
println("""
Next steps:
1. Generate all datasets (see examples/dataset_all.jl)
2. Train multiple model/dataset combinations (see examples/train_all_parallel.jl)
3. Implement NLP interface for adversarial optimization (see src/nlp_interface.jl)

For more information, see:
- README.md: Overview and quick start
- docs/TUTORIAL.md: Comprehensive tutorial
- docs/JULIA_PYTHON.md: Julia/Python integration guide
""")
