# Train All Combinations Example
# This script trains models for all combinations of datasets and architectures

using MadNLP4NN
using Printf
using Dates

println("="^80)
println("MadNLP4NN.jl - Batch Training Example")
println("="^80)

# =============================================================================
# Configuration
# =============================================================================

# Define datasets to use
# Note: Real datasets (mnist, fashionmnist) will be downloaded automatically
dataset_types = [
    "gaussian_mixture",
    "nonlinear_manifold",
    "mnist"
]

# Define model architectures
model_configs = [
    "medium_mlp",
    "medium_resmlp"
]

# Training configuration
training_config = Dict(
    "output_dir" => "./output",
    "epochs" => 50,
    "batch_size" => 64,
    "run_hparam_search" => false,  # Set to true for better results (takes longer)
    "seed" => 42,
    # Parameters for synthetic datasets
    "n_samples" => 10000,
    "input_dim" => 784,
    "output_dim" => 10,
    "n_components" => 20,
    "manifold_dim" => 50,
    "nonlinearity" => "polynomial"
)

println("\nConfiguration:")
println("  Datasets: $(join(dataset_types, ", "))")
println("  Models: $(join(model_configs, ", "))")
println("  Total combinations: $(length(dataset_types) × length(model_configs))")
println("  Epochs per model: $(training_config["epochs"])")
println("  Hyperparameter search: $(training_config["run_hparam_search"])")

# =============================================================================
# Training
# =============================================================================

println("\n" * "="^80)
println("Starting batch training...")
println("Start time: $(now())")
println("="^80)

start_time = time()

# Convert Dict to kwargs
kwargs = pairs(training_config)

# Train all combinations
all_results = train_all_combinations(
    dataset_types=dataset_types,
    model_configs=model_configs;
    kwargs...
)

end_time = time()
elapsed = end_time - start_time

# =============================================================================
# Results Summary
# =============================================================================

println("\n" * "="^80)
println("Training Complete!")
println("="^80)

println("\nTiming:")
println("  Total time: $(@sprintf("%.2f", elapsed)) seconds")
println("  Average per model: $(@sprintf("%.2f", elapsed / length(all_results))) seconds")

println("\nResults Summary:")
println("-"^80)
println(@sprintf("%-30s %-20s %s", "Dataset", "Model", "Val Accuracy"))
println("-"^80)

for result in all_results
    dataset = result.dataset
    model = result.model
    acc = result.results[:best_val_acc]
    println(@sprintf("%-30s %-20s %.4f", dataset, model, acc))
end

println("-"^80)

# Find best model
best_result = all_results[argmax([r.results[:best_val_acc] for r in all_results])]
println("\nBest Model:")
println("  Dataset: $(best_result.dataset)")
println("  Architecture: $(best_result.model)")
println("  Validation accuracy: $(@sprintf("%.4f", best_result.results[:best_val_acc]))")
println("  Model path: $(best_result.results[:model_path])")

# =============================================================================
# Save Summary
# =============================================================================

using JSON3

summary = Dict(
    "training_config" => training_config,
    "dataset_types" => dataset_types,
    "model_configs" => model_configs,
    "elapsed_time_seconds" => elapsed,
    "results" => [
        Dict(
            "dataset" => r.dataset,
            "model" => r.model,
            "val_accuracy" => r.results[:best_val_acc],
            "model_path" => r.results[:model_path]
        )
        for r in all_results
    ],
    "timestamp" => string(now())
)

summary_path = joinpath(training_config["output_dir"], "training_summary.json")
open(summary_path, "w") do f
    JSON3.pretty(f, summary)
end

println("\n✓ Summary saved to: $summary_path")

println("\n" * "="^80)
println("All models saved to: $(training_config["output_dir"])/models/")
println("All datasets saved to: $(training_config["output_dir"])/datasets/")
println("="^80)
