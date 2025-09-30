# Train All Models on All Datasets
# This script trains all model architectures on all generated datasets
# with hyperparameter search and multiple seeds, saving only the best

using MadNLP4NN
using Printf
using Dates
using JSON3

println("="^80)
println("MadNLP4NN.jl - Comprehensive Training Pipeline")
println("="^80)
println("Start time: $(now())")
println()

# =============================================================================
# Configuration
# =============================================================================

# Dataset configuration (matching dataset_all.jl)
sample_sizes = [10000, 20000]
input_dims = [500]
output_dims = [10]
n_components_list = [10, 20, 30]
manifold_dims = [20, 50, 100]
nonlinearity_types = ["polynomial", "trigonometric", "mixed"]
dataset_seed = 42

# All model architectures
all_model_configs = [
    "small_mlp",
    "medium_mlp",
    "large_mlp",
    "small_resmlp",
    "medium_resmlp",
    "large_resmlp"
]

# Training seeds (we train 3 times and keep the best)
training_seeds = [42, 123, 456]

# Training hyperparameters
epochs = 100
batch_size = 128
run_hparam_search = true
n_trials = 30  # For hyperparameter search
hparam_search_epochs = 20

output_dir = "./output"

# =============================================================================
# Generate Dataset List
# =============================================================================

println("="^80)
println("DATASET CONFIGURATION")
println("="^80)

all_datasets = []

# Gaussian Mixture datasets
println("\n1. Gaussian Mixture Datasets:")
for n_samples in sample_sizes
    for input_dim in input_dims
        for output_dim in output_dims
            for n_components in n_components_list
                dataset_info = Dict(
                    "type" => "gaussian_mixture",
                    "name" => "GM_n$(n_samples)_d$(input_dim)_c$(output_dim)_comp$(n_components)_s$(dataset_seed)",
                    "n_samples" => n_samples,
                    "input_dim" => input_dim,
                    "output_dim" => output_dim,
                    "n_components" => n_components,
                    "seed" => dataset_seed
                )
                push!(all_datasets, dataset_info)
                println("  - $(dataset_info["name"])")
            end
        end
    end
end
gm_count = length([d for d in all_datasets if d["type"] == "gaussian_mixture"])
println("  Total: $gm_count datasets")

# Nonlinear Manifold datasets
println("\n2. Nonlinear Manifold Datasets:")
for n_samples in sample_sizes
    for ambient_dim in input_dims
        for output_dim in output_dims
            for manifold_dim in manifold_dims
                if manifold_dim >= ambient_dim
                    continue
                end
                for nonlinearity in nonlinearity_types
                    dataset_info = Dict(
                        "type" => "nonlinear_manifold",
                        "name" => "NM_n$(n_samples)_a$(ambient_dim)_m$(manifold_dim)_c$(output_dim)_$(nonlinearity)_s$(dataset_seed)",
                        "n_samples" => n_samples,
                        "ambient_dim" => ambient_dim,
                        "manifold_dim" => manifold_dim,
                        "output_dim" => output_dim,
                        "nonlinearity" => nonlinearity,
                        "seed" => dataset_seed
                    )
                    push!(all_datasets, dataset_info)
                    println("  - $(dataset_info["name"])")
                end
            end
        end
    end
end
nm_count = length([d for d in all_datasets if d["type"] == "nonlinear_manifold"]) - gm_count
println("  Total: $nm_count datasets")

# Real datasets
println("\n3. Real Datasets:")
for dataset_name in ["mnist", "fashionmnist", "cifar10"]
    dataset_info = Dict(
        "type" => "real",
        "name" => dataset_name
    )
    push!(all_datasets, dataset_info)
    println("  - $(dataset_name)")
end
real_count = 3
println("  Total: $real_count datasets")

total_datasets = length(all_datasets)
total_models = length(all_model_configs)
total_seeds = length(training_seeds)
total_training_runs = total_datasets * total_models * total_seeds
total_saved_models = total_datasets * total_models  # Only best per (dataset, model)

println("\n" * "="^80)
println("TRAINING PLAN")
println("="^80)
println("  Total datasets: $total_datasets")
println("  Total model architectures: $total_models")
println("  Training seeds per (dataset, model): $total_seeds")
println("  Total training runs: $total_training_runs")
println("  Models to save (best per combo): $total_saved_models")
println("  Hyperparameter search: $run_hparam_search")
println("  Epochs per training: $epochs")
println("  Batch size: $batch_size")
println()

# Estimate time
est_minutes_per_run = 5  # Rough estimate
total_est_minutes = total_training_runs * est_minutes_per_run
println("Estimated time: ~$(@sprintf("%.1f", total_est_minutes / 60)) hours")
println()

# =============================================================================
# Training Loop
# =============================================================================

println("="^80)
println("STARTING TRAINING")
println("="^80)
println()

start_time = time()
all_results = []
training_count = 0
saved_models_count = 0

for (dataset_idx, dataset_info) in enumerate(all_datasets)
    dataset_type = dataset_info["type"]
    dataset_name = dataset_info["name"]
    
    println("\n" * "="^80)
    println("DATASET [$dataset_idx/$total_datasets]: $dataset_name")
    println("="^80)
    
    for (model_idx, model_config) in enumerate(all_model_configs)
        println("\n  MODEL [$model_idx/$total_models]: $model_config")
        println("  " * "-"^76)
        
        # Train with multiple seeds
        seed_results = []
        
        for (seed_idx, train_seed) in enumerate(training_seeds)
            global training_count += 1
            
            println("\n    SEED [$seed_idx/$total_seeds]: $train_seed")
            println("    Training run $training_count / $total_training_runs")
            
            try
                # Prepare training arguments based on dataset type
                if dataset_type == "gaussian_mixture"
                    results = train_model(
                        dataset_type=dataset_type,
                        model_config=model_config,
                        output_dir=output_dir,
                        n_samples=dataset_info["n_samples"],
                        input_dim=dataset_info["input_dim"],
                        output_dim=dataset_info["output_dim"],
                        n_components=dataset_info["n_components"],
                        seed=train_seed,
                        epochs=epochs,
                        batch_size=batch_size,
                        run_hparam_search=run_hparam_search,
                        n_trials=n_trials,
                        hparam_search_epochs=hparam_search_epochs
                    )
                elseif dataset_type == "nonlinear_manifold"
                    results = train_model(
                        dataset_type=dataset_type,
                        model_config=model_config,
                        output_dir=output_dir,
                        n_samples=dataset_info["n_samples"],
                        input_dim=dataset_info["ambient_dim"],
                        manifold_dim=dataset_info["manifold_dim"],
                        output_dim=dataset_info["output_dim"],
                        nonlinearity=dataset_info["nonlinearity"],
                        seed=train_seed,
                        epochs=epochs,
                        batch_size=batch_size,
                        run_hparam_search=run_hparam_search,
                        n_trials=n_trials,
                        hparam_search_epochs=hparam_search_epochs
                    )
                else  # Real dataset
                    results = train_model(
                        dataset_type=dataset_name,
                        model_config=model_config,
                        output_dir=output_dir,
                        seed=train_seed,
                        epochs=epochs,
                        batch_size=batch_size,
                        run_hparam_search=run_hparam_search,
                        n_trials=n_trials,
                        hparam_search_epochs=hparam_search_epochs
                    )
                end
                
                push!(seed_results, (
                    seed=train_seed,
                    val_acc=results[:best_val_acc],
                    results=results,
                    status="✓"
                ))
                
                println("    ✓ Complete! Val accuracy: $(@sprintf("%.4f", results[:best_val_acc]))")
                
            catch e
                println("    ✗ Failed: $e")
                push!(seed_results, (
                    seed=train_seed,
                    val_acc=0.0,
                    results=nothing,
                    status="✗"
                ))
            end
        end
        
        # Find best seed for this (dataset, model) combination
        successful_runs = filter(r -> r.status == "✓", seed_results)
        
        if !isempty(successful_runs)
            best_run = successful_runs[argmax([r.val_acc for r in successful_runs])]
            saved_models_count += 1
            
            println("\n  📊 Best seed for $model_config: $(best_run.seed)")
            println("     Validation accuracy: $(@sprintf("%.4f", best_run.val_acc))")
            
            push!(all_results, (
                dataset=dataset_name,
                dataset_type=dataset_type,
                model=model_config,
                best_seed=best_run.seed,
                best_val_acc=best_run.val_acc,
                all_seeds=[(r.seed, r.val_acc) for r in seed_results],
                model_path=best_run.results[:model_path],
                train_args=best_run.results[:train_args]
            ))
        else
            println("\n  ✗ All seeds failed for $model_config")
            push!(all_results, (
                dataset=dataset_name,
                dataset_type=dataset_type,
                model=model_config,
                best_seed=nothing,
                best_val_acc=0.0,
                all_seeds=[(r.seed, r.val_acc) for r in seed_results],
                model_path=nothing,
                train_args=nothing
            ))
        end
    end
end

end_time = time()
elapsed = end_time - start_time

# =============================================================================
# Results Summary
# =============================================================================

println("\n" * "="^80)
println("TRAINING COMPLETE!")
println("="^80)

println("\nTiming:")
println("  Total time: $(@sprintf("%.2f", elapsed / 3600)) hours")
println("  Total runs: $training_count")
println("  Successful runs: $(count(r -> r.best_val_acc > 0, all_results) * total_seeds)")
println("  Models saved: $saved_models_count / $total_saved_models")
println("  Average time per run: $(@sprintf("%.2f", elapsed / training_count)) seconds")

println("\n" * "="^80)
println("RESULTS BY DATASET")
println("="^80)

# Group by dataset
for dataset_info in all_datasets
    dataset_name = dataset_info["name"]
    dataset_results = filter(r -> r.dataset == dataset_name, all_results)
    
    if !isempty(dataset_results)
        println("\n$dataset_name:")
        println("-"^80)
        println(@sprintf("  %-20s %-8s %-10s %s", "Model", "Seed", "Val Acc", "All Seeds"))
        println("-"^80)
        
        for result in dataset_results
            if result.best_val_acc > 0
                seeds_str = join([@sprintf("%.4f", acc) for (s, acc) in result.all_seeds], ", ")
                println(@sprintf("  %-20s %-8s %-10s [%s]",
                    result.model,
                    result.best_seed,
                    @sprintf("%.4f", result.best_val_acc),
                    seeds_str
                ))
            else
                println(@sprintf("  %-20s %-8s %-10s %s",
                    result.model,
                    "FAILED",
                    "N/A",
                    "All seeds failed"
                ))
            end
        end
    end
end

# =============================================================================
# Top Models Summary
# =============================================================================

println("\n" * "="^80)
println("TOP 10 MODELS")
println("="^80)

successful_results = filter(r -> r.best_val_acc > 0, all_results)
sorted_results = sort(successful_results, by=r -> r.best_val_acc, rev=true)

println(@sprintf("\n%-40s %-20s %-8s %s", "Dataset", "Model", "Seed", "Val Acc"))
println("-"^80)

for (i, result) in enumerate(sorted_results[1:min(10, length(sorted_results))])
    println(@sprintf("%-40s %-20s %-8s %.4f",
        result.dataset,
        result.model,
        result.best_seed,
        result.best_val_acc
    ))
end

# =============================================================================
# Analysis by Model Architecture
# =============================================================================

println("\n" * "="^80)
println("PERFORMANCE BY MODEL ARCHITECTURE")
println("="^80)

for model_config in all_model_configs
    model_results = filter(r -> r.model == model_config && r.best_val_acc > 0, all_results)
    
    if !isempty(model_results)
        accs = [r.best_val_acc for r in model_results]
        println("\n$model_config:")
        println("  Count: $(length(accs))")
        println("  Mean accuracy: $(@sprintf("%.4f", sum(accs) / length(accs)))")
        println("  Best accuracy: $(@sprintf("%.4f", maximum(accs)))")
        println("  Worst accuracy: $(@sprintf("%.4f", minimum(accs)))")
    end
end

# =============================================================================
# Analysis by Dataset Type
# =============================================================================

println("\n" * "="^80)
println("PERFORMANCE BY DATASET TYPE")
println("="^80)

for dataset_type in ["gaussian_mixture", "nonlinear_manifold", "real"]
    type_results = filter(r -> r.dataset_type == dataset_type && r.best_val_acc > 0, all_results)
    
    if !isempty(type_results)
        accs = [r.best_val_acc for r in type_results]
        println("\n$(uppercase(dataset_type)):")
        println("  Count: $(length(accs))")
        println("  Mean accuracy: $(@sprintf("%.4f", sum(accs) / length(accs)))")
        println("  Best accuracy: $(@sprintf("%.4f", maximum(accs)))")
        println("  Worst accuracy: $(@sprintf("%.4f", minimum(accs)))")
    end
end

# =============================================================================
# Save Complete Results
# =============================================================================

println("\n" * "="^80)
println("SAVING RESULTS")
println("="^80)

results_summary = Dict(
    "metadata" => Dict(
        "total_datasets" => total_datasets,
        "total_models" => total_models,
        "training_seeds" => training_seeds,
        "total_training_runs" => training_count,
        "models_saved" => saved_models_count,
        "elapsed_time_hours" => elapsed / 3600,
        "timestamp" => string(now())
    ),
    "configuration" => Dict(
        "epochs" => epochs,
        "batch_size" => batch_size,
        "run_hparam_search" => run_hparam_search,
        "n_trials" => n_trials,
        "hparam_search_epochs" => hparam_search_epochs
    ),
    "results" => [
        Dict(
            "dataset" => r.dataset,
            "dataset_type" => r.dataset_type,
            "model" => r.model,
            "best_seed" => r.best_seed,
            "best_val_acc" => r.best_val_acc,
            "all_seeds" => [Dict("seed" => s, "val_acc" => acc) for (s, acc) in r.all_seeds],
            "model_path" => r.model_path
        )
        for r in all_results
    ]
)

summary_path = joinpath(output_dir, "training_complete_summary.json")
open(summary_path, "w") do f
    JSON3.pretty(f, results_summary)
end

println("✓ Complete results saved to: $summary_path")

# =============================================================================
# Final Summary
# =============================================================================

println("\n" * "="^80)
println("FINAL SUMMARY")
println("="^80)

println("""
Training Pipeline Complete!

Statistics:
  - Total training runs: $training_count
  - Successful models: $saved_models_count / $total_saved_models
  - Total time: $(@sprintf("%.2f", elapsed / 3600)) hours
  - Average time per run: $(@sprintf("%.1f", elapsed / training_count / 60)) minutes

Best Model Overall:
  $(sorted_results[1].dataset) + $(sorted_results[1].model)
  Seed: $(sorted_results[1].best_seed)
  Accuracy: $(@sprintf("%.4f", sorted_results[1].best_val_acc))
  Path: $(sorted_results[1].model_path)

All models saved to: $output_dir/models/
Complete results: $summary_path

Next steps:
1. Review results in training_complete_summary.json
2. Use trained models for NLP optimization
3. Analyze model performance across different datasets
""")

println("="^80)