# Train All Models on All Datasets - PARALLEL VERSION
# This script trains all model architectures on all generated datasets
# with hyperparameter search and multiple seeds, using parallel processing

using Distributed

# =============================================================================
# Setup Parallel Workers
# =============================================================================

# Determine number of workers (leave 1-2 cores free for system)
n_cores = 12  # Adjust based on your system
n_workers = n_cores - 1  # Leave one core for the main process

println("="^80)
println("MadNLP4NN.jl - PARALLEL Training Pipeline")
println("="^80)
println("Detected cores: $n_cores")
println("Starting workers: $n_workers")
println()

# Add worker processes
if nworkers() == 1
    addprocs(n_workers)
    println("✓ Started $(nworkers()) worker processes")
else
    println("✓ Using existing $(nworkers()) worker processes")
end

# Load packages on all workers
@everywhere using MadNLP4NN
@everywhere using Printf
@everywhere using Dates
@everywhere using JSON3

println("✓ Packages loaded on all workers")
println()

# =============================================================================
# Configuration (on main process)
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
n_trials = 30
hparam_search_epochs = 20

output_dir = "./output"

println("Start time: $(now())")
println()

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
total_saved_models = total_datasets * total_models

println("\n" * "="^80)
println("TRAINING PLAN")
println("="^80)
println("  Total datasets: $total_datasets")
println("  Total model architectures: $total_models")
println("  Training seeds per (dataset, model): $total_seeds")
println("  Total training runs: $total_training_runs")
println("  Models to save (best per combo): $total_saved_models")
println("  Parallel workers: $(nworkers())")
println("  Hyperparameter search: $run_hparam_search")
println("  Epochs per training: $epochs")
println()

# Estimate time with parallelization
est_minutes_per_run = 5
serial_hours = (total_training_runs * est_minutes_per_run) / 60
parallel_hours = serial_hours / nworkers()
println("Estimated time:")
println("  Serial: ~$(@sprintf("%.1f", serial_hours)) hours")
println("  Parallel ($n_workers workers): ~$(@sprintf("%.1f", parallel_hours)) hours")
println("  Speedup: ~$(nworkers())x")
println()

# =============================================================================
# Create Training Jobs
# =============================================================================

# Generate all training jobs (dataset × model × seed combinations)
all_jobs = []
job_id = 0

for dataset_info in all_datasets
    for model_config in all_model_configs
        for train_seed in training_seeds
            global job_id += 1
            push!(all_jobs, (
                id=job_id,
                dataset_info=dataset_info,
                model_config=model_config,
                train_seed=train_seed
            ))
        end
    end
end

println("="^80)
println("STARTING PARALLEL TRAINING")
println("="^80)
println("Total jobs: $(length(all_jobs))")
println("Workers: $(nworkers())")
println()

# =============================================================================
# Define Training Function (available on all workers)
# =============================================================================

@everywhere function train_single_job(
    job_id::Int,
    dataset_info::Dict,
    model_config::String,
    train_seed::Int,
    output_dir::String,
    epochs::Int,
    batch_size::Int,
    run_hparam_search::Bool,
    n_trials::Int,
    hparam_search_epochs::Int
)
    """
    Train a single model configuration.
    Returns a tuple of (success, job_id, results_dict)
    """
    
    dataset_type = dataset_info["type"]
    dataset_name = dataset_info["name"]
    
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
        
        return (
            success=true,
            job_id=job_id,
            dataset=dataset_name,
            dataset_type=dataset_type,
            model=model_config,
            seed=train_seed,
            val_acc=results[:best_val_acc],
            model_path=results[:model_path],
            train_args=results[:train_args]
        )
        
    catch e
        return (
            success=false,
            job_id=job_id,
            dataset=dataset_name,
            dataset_type=dataset_type,
            model=model_config,
            seed=train_seed,
            val_acc=0.0,
            model_path=nothing,
            train_args=nothing,
            error=string(e)
        )
    end
end

# =============================================================================
# Run Parallel Training with Progress Monitoring
# =============================================================================

start_time = time()

# Use pmap for parallel execution with load balancing
println("Starting parallel execution...")
println("Jobs will be distributed across $(nworkers()) workers")
println()

# Track progress
completed = Threads.Atomic{Int}(0)
total_jobs = length(all_jobs)

# Create progress monitoring task
progress_task = @async begin
    while completed[] < total_jobs
        sleep(30)  # Update every 30 seconds
        elapsed = time() - start_time
        pct = 100 * completed[] / total_jobs
        rate = completed[] / (elapsed / 60)  # jobs per minute
        eta = (total_jobs - completed[]) / rate
        
        println("[$(now())] Progress: $(completed[])/$total_jobs ($(@sprintf("%.1f", pct))%) - " *
                "$(@sprintf("%.1f", rate)) jobs/min - ETA: $(@sprintf("%.1f", eta)) min")
    end
end

# Run all jobs in parallel
job_results = pmap(all_jobs) do job
    result = train_single_job(
        job.id,
        job.dataset_info,
        job.model_config,
        job.train_seed,
        output_dir,
        epochs,
        batch_size,
        run_hparam_search,
        n_trials,
        hparam_search_epochs
    )
    
    # Update progress counter
    Threads.atomic_add!(completed, 1)
    
    # Print completion message
    if result.success
        println("[Worker $(myid())] ✓ Job $(result.job_id): $(result.dataset) + $(result.model) [seed $(result.seed)] = $(@sprintf("%.4f", result.val_acc))")
    else
        println("[Worker $(myid())] ✗ Job $(result.job_id): $(result.dataset) + $(result.model) [seed $(result.seed)] FAILED")
    end
    
    return result
end

end_time = time()
elapsed = end_time - start_time

println("\n" * "="^80)
println("PARALLEL TRAINING COMPLETE!")
println("="^80)
println("Total time: $(@sprintf("%.2f", elapsed / 3600)) hours")
println("Average time per job: $(@sprintf("%.1f", elapsed / total_jobs)) seconds")
println("Jobs per hour: $(@sprintf("%.1f", total_jobs / (elapsed / 3600)))")
println()

# =============================================================================
# Aggregate Results by (Dataset, Model)
# =============================================================================

println("="^80)
println("AGGREGATING RESULTS")
println("="^80)

all_results = []

for dataset_info in all_datasets
    dataset_name = dataset_info["name"]
    dataset_type = dataset_info["type"]
    
    for model_config in all_model_configs
        # Get all seed results for this (dataset, model) combo
        seed_results = filter(r -> r.dataset == dataset_name && r.model == model_config, job_results)
        
        if !isempty(seed_results)
            # Get successful runs
            successful_runs = filter(r -> r.success, seed_results)
            
            if !isempty(successful_runs)
                # Find best seed
                best_run = successful_runs[argmax([r.val_acc for r in successful_runs])]
                
                push!(all_results, (
                    dataset=dataset_name,
                    dataset_type=dataset_type,
                    model=model_config,
                    best_seed=best_run.seed,
                    best_val_acc=best_run.val_acc,
                    all_seeds=[(r.seed, r.val_acc) for r in seed_results],
                    model_path=best_run.model_path,
                    train_args=best_run.train_args,
                    n_successful=length(successful_runs)
                ))
            else
                # All seeds failed
                push!(all_results, (
                    dataset=dataset_name,
                    dataset_type=dataset_type,
                    model=model_config,
                    best_seed=nothing,
                    best_val_acc=0.0,
                    all_seeds=[(r.seed, r.val_acc) for r in seed_results],
                    model_path=nothing,
                    train_args=nothing,
                    n_successful=0
                ))
            end
        end
    end
end

saved_models_count = count(r -> r.best_val_acc > 0, all_results)

println("✓ Aggregation complete")
println("  Successful models: $saved_models_count / $total_saved_models")
println()

# =============================================================================
# Results Summary
# =============================================================================

println("="^80)
println("RESULTS BY DATASET")
println("="^80)

for dataset_info in all_datasets
    dataset_name = dataset_info["name"]
    dataset_results = filter(r -> r.dataset == dataset_name, all_results)
    
    if !isempty(dataset_results)
        println("\n$dataset_name:")
        println("-"^80)
        println(@sprintf("  %-20s %-8s %-10s %s", "Model", "Seed", "Val Acc", "Success Rate"))
        println("-"^80)
        
        for result in dataset_results
            if result.best_val_acc > 0
                println(@sprintf("  %-20s %-8s %-10s %d/$(length(training_seeds))",
                    result.model,
                    result.best_seed,
                    @sprintf("%.4f", result.best_val_acc),
                    result.n_successful
                ))
            else
                println(@sprintf("  %-20s %-8s %-10s %s",
                    result.model,
                    "FAILED",
                    "N/A",
                    "0/$(length(training_seeds))"
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
# Performance Analysis
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
# Save Results
# =============================================================================

println("\n" * "="^80)
println("SAVING RESULTS")
println("="^80)

results_summary = Dict(
    "metadata" => Dict(
        "total_datasets" => total_datasets,
        "total_models" => total_models,
        "training_seeds" => training_seeds,
        "total_training_jobs" => total_jobs,
        "models_saved" => saved_models_count,
        "elapsed_time_hours" => elapsed / 3600,
        "n_workers" => nworkers(),
        "speedup" => nworkers(),
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
            "model_path" => r.model_path,
            "n_successful" => r.n_successful
        )
        for r in all_results
    ]
)

summary_path = joinpath(output_dir, "training_complete_summary_parallel.json")
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

speedup = (serial_hours / parallel_hours)
time_saved = serial_hours - (elapsed / 3600)

println("""
Parallel Training Complete!

Performance:
  - Workers used: $(nworkers())
  - Total jobs: $total_jobs
  - Time taken: $(@sprintf("%.2f", elapsed / 3600)) hours
  - Estimated serial time: ~$(@sprintf("%.1f", serial_hours)) hours
  - Time saved: ~$(@sprintf("%.1f", time_saved)) hours
  - Effective speedup: $(@sprintf("%.1f", speedup))x

Results:
  - Successful models: $saved_models_count / $total_saved_models
  - Success rate: $(@sprintf("%.1f", 100 * saved_models_count / total_saved_models))%

Best Model Overall:
  $(sorted_results[1].dataset) + $(sorted_results[1].model)
  Seed: $(sorted_results[1].best_seed)
  Accuracy: $(@sprintf("%.4f", sorted_results[1].best_val_acc))
  Path: $(sorted_results[1].model_path)

All models saved to: $output_dir/models/
Complete results: $summary_path
""")

println("="^80)

# Clean up workers
println("\nCleaning up worker processes...")
rmprocs(workers())
println("✓ Workers terminated")
println("\nDone!")
println("="^80)
