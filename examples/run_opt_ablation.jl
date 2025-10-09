#!/usr/bin/env julia

"""
run_opt_ablation.jl

Ablation study script for testing Flux and JAX backends on multiple neural network models.
This script runs optimization experiments across all models in a given dataset directory,
testing each model with multiple initial points and target classes.

Usage:
    julia run_opt_ablation.jl --backend flux --dataset mnist
    julia run_opt_ablation.jl --backend jax --dataset cifar

Arguments:
    --backend: Backend to use ("flux" or "jax")
    --dataset: Dataset to test ("mnist" or "cifar")
"""

using MadNLP4NN
using MadNLP
using LinearAlgebra
using Random
using JSON3
using Printf
using Dates
using PythonCall
using ArgParse
using FileIO
using Images

println("="^80)
println("MadNLP4NN - Ablation Study for Neural Network Optimization")
println("="^80)
println("Started at: $(Dates.now())")
println()

# =============================================================================
# Argument Parsing
# =============================================================================

function parse_commandline()
    s = ArgParseSettings(
        description = "Run ablation study for MadNLP4NN optimization",
        epilog = "Example: julia run_opt_ablation.jl --backend flux --dataset mnist"
    )

    @add_arg_table! s begin
        "--backend"
            help = "Backend to use: 'flux' or 'jax'"
            arg_type = String
            default = "flux"
        "--dataset"
            help = "Dataset to test: 'mnist' or 'cifar'"
            arg_type = String
            default = "mnist"
        "--max-iter"
            help = "Maximum number of iterations"
            arg_type = Int
            default = 1000
        "--tolerance"
            help = "Convergence tolerance"
            arg_type = Float64
            default = 1e-4
        "--print-level"
            help = "Print level: 'ERROR', 'WARN', 'INFO', 'DEBUG'"
            arg_type = String
            default = "ERROR"
    end

    return parse_args(s)
end

args = parse_commandline()

# =============================================================================
# Configuration
# =============================================================================

const BACKEND = args["backend"]
const DATASET = args["dataset"]
const MAX_ITER = args["max-iter"]
const TOLERANCE = args["tolerance"]

# Validate arguments
if !(BACKEND in ["flux", "jax"])
    error("Invalid backend: $BACKEND. Must be 'flux' or 'jax'")
end

if !(DATASET in ["mnist", "cifar"])
    error("Invalid dataset: $DATASET. Must be 'mnist' or 'cifar'")
end

# Print level mapping
const PRINT_LEVEL_MAP = Dict(
    "ERROR" => MadNLP.ERROR,
    "WARN" => MadNLP.WARN,
    "INFO" => MadNLP.INFO,
    "DEBUG" => MadNLP.DEBUG
)
const PRINT_LEVEL = PRINT_LEVEL_MAP[args["print-level"]]

# Fixed optimization settings
const RANDOM_SEED = 42
const CONSTRAINT_TYPE = "both"
const BOX_LOWER = -100.0
const BOX_UPPER = 100.0
const SPHERE_RADIUS = 50.0
const INIT_SCALE = 1.0  # For random initialization

# Dataset-specific configurations
const DATASET_CONFIG = Dict(
    "mnist" => Dict(
        "model_dir" => "output/models_ablation/mnist",
        "output_dir" => "output/models_ablation/mnist/optimization_results",
        "input_dim" => 784,  # 28x28
        "output_dim" => 10,
        "image_shape" => (28, 28),
        "models" => [
            "small_mlp_seed42.pt",
            "medium_mlp_seed42.pt",
            "large_mlp_seed42.pt",
            "small_resmlp_seed42.pt",
            "medium_resmlp_seed42.pt",
            "larg_resmlp_seed42.pt"
        ]
    ),
    "cifar" => Dict(
        "model_dir" => "output/models_ablation/cifar",
        "output_dir" => "output/models_ablation/cifar/optimization_results",
        "input_dim" => 3072,  # 32x32x3
        "output_dim" => 10,
        "image_shape" => (32, 32, 3),
        "models" => [
            "cifar10_vgg11_bn.pt",
            "cifar10_vgg13_bn.pt",
            "cifar10_vgg16_bn-6ee7ea24.pt",
            "cifar10_vgg19_bn-57191229.pt",
            "cifar10_resnet20.pt",
            "cifar10_resnet32.pt",
            "cifar10_resnet44.pt",
            "cifar10_resnet56.pt"
        ]
    )
)

const CONFIG = DATASET_CONFIG[DATASET]
const MODEL_DIR = CONFIG["model_dir"]
const OUTPUT_DIR = CONFIG["output_dir"]
const INPUT_DIM = CONFIG["input_dim"]
const OUTPUT_DIM = CONFIG["output_dim"]
const IMAGE_SHAPE = CONFIG["image_shape"]
const MODELS = CONFIG["models"]

println("[Configuration]")
println("  Backend: $BACKEND")
println("  Dataset: $DATASET")
println("  Model directory: $MODEL_DIR")
println("  Output directory: $OUTPUT_DIR")
println("  Number of models: $(length(MODELS))")
println("  Input dimension: $INPUT_DIM")
println("  Output dimension: $OUTPUT_DIM")
println("  Max iterations: $MAX_ITER")
println("  Tolerance: $TOLERANCE")
println("  Random seed: $RANDOM_SEED")
println()

# =============================================================================
# Setup Random Seeds
# =============================================================================

Random.seed!(RANDOM_SEED)

if BACKEND == "jax"
    pyimport("numpy").random.seed(RANDOM_SEED)
    pyimport("random").seed(RANDOM_SEED)
end

println("[Random Seeds Set]")
println("  Julia seed: $RANDOM_SEED")
if BACKEND == "jax"
    println("  Python/NumPy seed: $RANDOM_SEED")
end
println()

# =============================================================================
# Generate Fixed Initial Points
# =============================================================================

println("[Generating Initial Points]")
println("-"^80)

# Generate 10 fixed random initial points (one for each target class)
# These will be the same across all models for fair comparison
const INITIAL_POINTS = [rand(INPUT_DIM) .* (2 * INIT_SCALE) .- INIT_SCALE for _ in 1:OUTPUT_DIM]

println("✓ Generated $(length(INITIAL_POINTS)) initial points")
for (i, x0) in enumerate(INITIAL_POINTS)
    println("  Initial point $i: ||x0|| = $(@sprintf("%.4f", norm(x0)))")
end
println()

# =============================================================================
# Generate Target Outputs
# =============================================================================

println("[Generating Target Outputs]")
println("-"^80)

# Generate one-hot targets for each class
const TARGETS = [begin
    t = zeros(OUTPUT_DIM)
    t[i] = 1.0
    t
end for i in 1:OUTPUT_DIM]

println("✓ Generated $(length(TARGETS)) one-hot targets")
for (i, target) in enumerate(TARGETS)
    println("  Target $i: class $i (one-hot)")
end
println()

# =============================================================================
# Helper Functions
# =============================================================================

"""
    save_image(x, filepath, image_shape)

Save optimization result as an image.
"""
function save_image(x::Vector{Float64}, filepath::String, image_shape::Tuple)
    if length(image_shape) == 2
        # Grayscale image (e.g., MNIST)
        H, W = image_shape
        # Reshape and normalize to [0, 1]
        img_data = reshape(x, H, W)
        img_data = clamp.(img_data, 0.0, 1.0)
        img = Gray.(img_data')  # Transpose for correct orientation
        
    elseif length(image_shape) == 3
        # RGB image (e.g., CIFAR-10)
        H, W, C = image_shape
        # Reshape from flat vector to (H, W, C)
        img_data = reshape(x, C, H, W)  # PyTorch format: (C, H, W)
        
        # Transpose to (H, W, C) for Images.jl
        img_data = permutedims(img_data, (2, 3, 1))
        
        # Normalize to [0, 1]
        img_data = clamp.(img_data, 0.0, 1.0)
        
        # Create RGB image
        img = colorview(RGB, permutedims(img_data, (3, 1, 2)))
    else
        error("Unsupported image shape: $image_shape")
    end
    
    # Save image
    save(filepath, img)
end

"""
    run_single_optimization(model_path, x0, target, target_class, model_name, backend)

Run a single optimization experiment and save results.
"""
function run_single_optimization(
    model_path::String,
    x0::Vector{Float64},
    target::Vector{Float64},
    target_class::Int,
    model_name::String,
    backend::String
)
    println("  → Optimizing for target class $target_class")
    
    # Create objective
    objective_func = NeuralNetworkObjective(target, weight=1.0)
    
    # Create constraints (box + spherical)
    x_min = fill(BOX_LOWER, length(x0))
    x_max = fill(BOX_UPPER, length(x0))
    sphere_center = copy(x0)
    
    constraint_func = CompositeConstraint(
        BoxConstraints(x_min, x_max),
        SphericalConstraint(sphere_center, SPHERE_RADIUS)
    )
    
    # Create NLP model
    use_python = (backend == "jax")
    
    try
        nlp_model = NeuralNetworkNLPModel(
            model_path,
            objective_func,
            constraint_func,
            x0;
            use_python=use_python
        )
        
        # Solve
        solver_options = Dict{Symbol, Any}(
            :max_iter => MAX_ITER,
            :tol => TOLERANCE,
            :print_level => PRINT_LEVEL
        )
        
        start_time = time()
        result = solve_nlp(nlp_model; solver_options...)
        solve_time = time() - start_time
        
        # Extract results
        x_sol = result[:solution]
        timing_stats = result[:timing_stats]
        total_eval_time = timing_stats.total_eval_time
        madnlp_internal_time = solve_time - total_eval_time
        
        # Evaluate neural network output
        nn_output = nlp_model.neural_network !== nothing ? 
            nlp_model.neural_network(x_sol) : 
            nlp_model.python_evaluator.evaluate_nn(x_sol)
        
        if nlp_model.neural_network === nothing
            nn_output = pyconvert(Vector{Float64}, nn_output)
        end
        
        dist_to_target = norm(nn_output - target)
        pred_class = argmax(nn_output)
        
        # Generate filename
        timestamp = Dates.format(Dates.now(), "yyyymmdd_HHMMSS")
        base_filename = "$(model_name)_target$(target_class)_seed$(RANDOM_SEED)_$(backend)_$(timestamp)"
        
        # Save results JSON
        results_dict = Dict(
            "timestamp" => string(Dates.now()),
            "configuration" => Dict(
                "model_path" => model_path,
                "model_name" => model_name,
                "backend" => backend,
                "dataset" => DATASET,
                "target_class" => target_class,
                "random_seed" => RANDOM_SEED
            ),
            "problem" => Dict(
                "input_dim" => INPUT_DIM,
                "output_dim" => OUTPUT_DIM,
                "n_variables" => nlp_model.meta.nvar,
                "n_constraints" => nlp_model.meta.ncon
            ),
            "parameters" => Dict(
                "max_iter" => MAX_ITER,
                "tolerance" => TOLERANCE,
                "constraint_type" => CONSTRAINT_TYPE,
                "box_bounds" => [BOX_LOWER, BOX_UPPER],
                "sphere_radius" => SPHERE_RADIUS
            ),
            "results" => Dict(
                "status" => string(result[:status]),
                "objective" => result[:objective],
                "iterations" => result[:iter_count],
                "solve_time" => solve_time,
                "solution_norm" => norm(x_sol),
                "solution_norm_inf" => maximum(abs.(x_sol)),
                "dist_to_target" => dist_to_target,
                "predicted_class" => pred_class,
                "target_class" => target_class,
                "classification_correct" => (pred_class == target_class)
            ),
            "timing" => Dict(
                "total_solve_time" => solve_time,
                "total_eval_time" => total_eval_time,
                "total_madnlp_internal_time" => madnlp_internal_time,
                "eval_count" => timing_stats.eval_count,
                "statistics" => Dict(
                    "avg_eval_time" => total_eval_time / max(timing_stats.eval_count, 1),
                    "eval_time_percentage" => 100 * total_eval_time / solve_time,
                    "madnlp_internal_percentage" => 100 * madnlp_internal_time / solve_time
                )
            ),
            "solution" => Dict(
                "nn_output" => nn_output,
                "target" => target,
                "initial_point_norm" => norm(x0)
            )
        )
        
        # Save JSON
        json_filepath = joinpath(OUTPUT_DIR, "$(base_filename).json")
        open(json_filepath, "w") do io
            JSON3.pretty(io, results_dict)
        end
        
        # Save image
        image_filepath = joinpath(OUTPUT_DIR, "$(base_filename).png")
        save_image(x_sol, image_filepath, IMAGE_SHAPE)
        
        # Print summary
        println("    Status: $(result[:status])")
        println("    Objective: $(@sprintf("%.6e", result[:objective]))")
        println("    Iterations: $(result[:iter_count])")
        println("    Time: $(@sprintf("%.3f", solve_time))s")
        println("    Predicted class: $pred_class (target: $target_class)")
        println("    Distance to target: $(@sprintf("%.6e", dist_to_target))")
        println("    ✓ Saved: $(basename(json_filepath))")
        println("    ✓ Saved: $(basename(image_filepath))")
        
        return true
        
    catch e
        @error "Optimization failed for target class $target_class" exception=(e, catch_backtrace())
        return false
    end
end

# =============================================================================
# Main Ablation Study Loop
# =============================================================================

println("="^80)
println("[Starting Ablation Study]")
println("="^80)
println()

# Create output directory
mkpath(OUTPUT_DIR)

# Statistics
total_experiments = length(MODELS) * OUTPUT_DIM
completed_experiments = 0
failed_experiments = 0

# Iterate over all models
for (model_idx, model_file) in enumerate(MODELS)
    model_path = joinpath(MODEL_DIR, model_file)
    model_name = replace(splitext(model_file)[1], r"\.pt$" => "")
    
    println("="^80)
    println("[Model $model_idx/$(length(MODELS)): $model_name]")
    println("="^80)
    println("  Path: $model_path")
    
    # Check if model exists
    if !isfile(model_path)
        @warn "Model file not found: $model_path. Skipping..."
        failed_experiments += OUTPUT_DIM
        continue
    end
    
    println()
    
    # Run optimization for each target class
    for target_class in 1:OUTPUT_DIM
        println("[Target Class $target_class/$OUTPUT_DIM]")
        
        x0 = INITIAL_POINTS[target_class]
        target = TARGETS[target_class]
        
        success = run_single_optimization(
            model_path,
            x0,
            target,
            target_class,
            model_name,
            BACKEND
        )
        
        if success
            completed_experiments += 1
        else
            failed_experiments += 1
        end
        
        println()
    end
    
    # Print progress
    progress = 100 * (model_idx) / length(MODELS)
    println("Progress: $(@sprintf("%.1f", progress))% ($model_idx/$(length(MODELS)) models completed)")
    println()
end

# =============================================================================
# Final Summary
# =============================================================================

println("="^80)
println("[Ablation Study Complete]")
println("="^80)
println()
println("Summary:")
println("  Backend: $BACKEND")
println("  Dataset: $DATASET")
println("  Total experiments: $total_experiments")
println("  Completed: $completed_experiments")
println("  Failed: $failed_experiments")
println("  Success rate: $(@sprintf("%.1f", 100 * completed_experiments / total_experiments))%")
println()
println("Results saved to: $OUTPUT_DIR")
println()
println("Completed at: $(Dates.now())")
println("="^80)

