#!/usr/bin/env julia

"""
run_optimization.jl

A flexible script for running neural network optimization with MadNLP4NN.

Usage:
    julia run_optimization.jl

Configuration:
    Edit the CONFIGURATION section below to customize the optimization problem.
    All major parameters are exposed as variables for easy modification.
"""

using MadNLP4NN
using MadNLP
using LinearAlgebra
using Random
using JSON3
using Printf
using Dates
using PythonCall

println("="^80)
println("MadNLP4NN - Configurable Neural Network Optimization")
println("="^80)
println("Started at: $(Dates.now())")
println()

# =============================================================================
# CONFIGURATION - Edit these parameters to customize your optimization
# =============================================================================

# -----------------------------------------------------------------------------
# Model Configuration
# -----------------------------------------------------------------------------
const MODEL_PATH = "output/models/mnist/small_mlp/model_seed42.pt"
# Alternative examples:
# const MODEL_PATH = "output/models/mnist/small_mlp/model_seed42.pt"
# const MODEL_PATH = "output/models/fashionmnist/medium_mlp/model_seed123.pt"

# -----------------------------------------------------------------------------
# Backend Configuration
# -----------------------------------------------------------------------------
# Autodiff backend: "flux" (ForwardDiff) or "jax" (Python/JAX)
const BACKEND = "jax"  # Options: "flux", "jax"

# -----------------------------------------------------------------------------
# Optimization Problem Configuration
# -----------------------------------------------------------------------------

# Objective type: "target_distance", "regularized", or "custom"
const OBJECTIVE_TYPE = "target_distance"  # Options: "target_distance", "regularized", "custom"

# Constraint type: "box", "spherical", "both", or "none"
const CONSTRAINT_TYPE = "both"  # Options: "box", "spherical", "both", "none"

# -----------------------------------------------------------------------------
# Target Output Configuration
# -----------------------------------------------------------------------------
# Target output type: "one_hot", "uniform", "zeros", "ones", or "custom"
const TARGET_TYPE = "one_hot"  # Options: "one_hot", "uniform", "zeros", "ones", "custom"
const TARGET_CLASS = 1  # For "one_hot" type, which class to target (1-indexed)
const CUSTOM_TARGET = nothing  # For "custom" type, provide Vector{Float64}

# -----------------------------------------------------------------------------
# Initial Point Configuration
# -----------------------------------------------------------------------------
# Initial point type: "random", "zeros", "randn", or "custom"
const INIT_TYPE = "random"  # Options: "random", "zeros", "randn", "custom"
const INIT_SCALE = 1.0  # Scaling factor for random initialization
const CUSTOM_INIT = nothing  # For "custom" type, provide Vector{Float64}

# -----------------------------------------------------------------------------
# Constraint Parameters
# -----------------------------------------------------------------------------
# Box constraints
const BOX_LOWER = -100.0  # Lower bound for all variables
const BOX_UPPER = 100.0   # Upper bound for all variables
const CUSTOM_BOX_BOUNDS = nothing  # Custom bounds: (x_min::Vector, x_max::Vector) or nothing

# Spherical constraints
const SPHERE_RADIUS = 50.0  # Radius for spherical constraint
const SPHERE_CENTER = nothing  # Center point (nothing = use initial point)

# -----------------------------------------------------------------------------
# Regularization Parameters
# -----------------------------------------------------------------------------
const REGULARIZATION_WEIGHT = 0.01  # Weight for quadratic regularization term
const REGULARIZATION_CENTER = nothing  # Center for regularization (nothing = use initial point)

# -----------------------------------------------------------------------------
# Custom Objective Weights (for "custom" objective type)
# -----------------------------------------------------------------------------
const CUSTOM_NN_WEIGHT = 1.0  # Weight for neural network objective
const CUSTOM_REG_WEIGHT = 0.01  # Weight for regularization term

# -----------------------------------------------------------------------------
# Solver Parameters
# -----------------------------------------------------------------------------
const MAX_ITER = 1000  # Maximum number of iterations
const TOLERANCE = 1e-4  # Convergence tolerance
const PRINT_LEVEL = MadNLP.ERROR  # Options: ERROR, WARN, INFO, DEBUG
const LINEAR_SOLVER = nothing  # Linear solver (nothing = default, or e.g., LapackCPUSolver)

# Additional MadNLP options (add more as needed)
const EXTRA_SOLVER_OPTIONS = Dict{Symbol, Any}(
    # :mu_init => 1e-1,
    # :kappa_d => 1e-5,
    # :acceptable_tol => 1e-3,
)

# -----------------------------------------------------------------------------
# Output Configuration
# -----------------------------------------------------------------------------
const SAVE_RESULTS = true  # Whether to save results to file
const OUTPUT_DIR = "output/optimization_results"  # Directory for saving results
const RESULT_FILENAME = nothing  # Custom filename (nothing = auto-generate)

# Random seed for reproducibility
const RANDOM_SEED = 42

# Verbose output
const VERBOSE = true  # Print detailed information during setup

# =============================================================================
# End of Configuration
# =============================================================================

println("[Configuration Summary]")
println("  Model: $MODEL_PATH")
println("  Backend: $BACKEND")
println("  Objective: $OBJECTIVE_TYPE")
println("  Constraints: $CONSTRAINT_TYPE")
println("  Target type: $TARGET_TYPE")
println("  Initial point: $INIT_TYPE")
println("  Max iterations: $MAX_ITER")
println("  Tolerance: $TOLERANCE")
println()

# Set random seed
Random.seed!(RANDOM_SEED)

# Set Python random seeds if using JAX backend
if BACKEND == "jax"
    pyimport("numpy").random.seed(RANDOM_SEED)
    pyimport("random").seed(RANDOM_SEED)
end

# =============================================================================
# Load Model and Extract Configuration
# =============================================================================

println("[Step 1] Loading Model")
println("-"^80)

if !isfile(MODEL_PATH)
    error("""
    Model not found at: $MODEL_PATH
    
    Please either:
    1. Train a model first by running: julia examples/train_all.jl
    2. Update the MODEL_PATH variable to point to an existing model
    """)
end

state_dict, model_config, train_args, history = load_trained_model(MODEL_PATH)

input_dim = model_config["input_dim"]
output_dim = model_config["output_dim"]

if VERBOSE
    println("✓ Model loaded successfully")
    println("  Model type: $(model_config["model_type"])")
    println("  Input dimension: $input_dim")
    println("  Output dimension: $output_dim")
    if haskey(model_config, "hidden_dims")
        println("  Hidden layers: $(model_config["hidden_dims"])")
    end
    if haskey(history, "best_val_acc")
        println("  Training accuracy: $(round(history["best_val_acc"], digits=4))")
    end
end
println()

# =============================================================================
# Setup Target Output
# =============================================================================

println("[Step 2] Setting up Target Output")
println("-"^80)

target = if TARGET_TYPE == "one_hot"
    t = zeros(output_dim)
    if TARGET_CLASS < 1 || TARGET_CLASS > output_dim
        error("TARGET_CLASS must be between 1 and $output_dim")
    end
    t[TARGET_CLASS] = 1.0
    if VERBOSE
        println("  Target: one-hot encoding for class $TARGET_CLASS")
    end
    t
elseif TARGET_TYPE == "uniform"
    t = ones(output_dim) ./ output_dim
    if VERBOSE
        println("  Target: uniform distribution")
    end
    t
elseif TARGET_TYPE == "zeros"
    t = zeros(output_dim)
    if VERBOSE
        println("  Target: all zeros")
    end
    t
elseif TARGET_TYPE == "ones"
    t = ones(output_dim)
    if VERBOSE
        println("  Target: all ones")
    end
    t
elseif TARGET_TYPE == "custom"
    if CUSTOM_TARGET === nothing
        error("TARGET_TYPE is 'custom' but CUSTOM_TARGET is not set")
    end
    if length(CUSTOM_TARGET) != output_dim
        error("CUSTOM_TARGET length ($(length(CUSTOM_TARGET))) must match output_dim ($output_dim)")
    end
    t = Float64.(CUSTOM_TARGET)
    if VERBOSE
        println("  Target: custom ($(round.(t, digits=3)))")
    end
    t
else
    error("Unknown TARGET_TYPE: $TARGET_TYPE")
end

println()

# =============================================================================
# Setup Initial Point
# =============================================================================

println("[Step 3] Setting up Initial Point")
println("-"^80)

x0 = if INIT_TYPE == "random"
    x = rand(input_dim) .* (2 * INIT_SCALE) .- INIT_SCALE
    if VERBOSE
        println("  Initial point: random uniform in [$(-INIT_SCALE), $INIT_SCALE]")
        println("  ||x0||: $(round(norm(x), digits=4))")
    end
    x
elseif INIT_TYPE == "zeros"
    x = zeros(input_dim)
    if VERBOSE
        println("  Initial point: all zeros")
    end
    x
elseif INIT_TYPE == "randn"
    x = randn(input_dim) .* INIT_SCALE
    if VERBOSE
        println("  Initial point: random normal N(0, $(INIT_SCALE^2))")
        println("  ||x0||: $(round(norm(x), digits=4))")
    end
    x
elseif INIT_TYPE == "custom"
    if CUSTOM_INIT === nothing
        error("INIT_TYPE is 'custom' but CUSTOM_INIT is not set")
    end
    if length(CUSTOM_INIT) != input_dim
        error("CUSTOM_INIT length ($(length(CUSTOM_INIT))) must match input_dim ($input_dim)")
    end
    x = Float64.(CUSTOM_INIT)
    if VERBOSE
        println("  Initial point: custom")
        println("  ||x0||: $(round(norm(x), digits=4))")
    end
    x
else
    error("Unknown INIT_TYPE: $INIT_TYPE")
end

println()

# =============================================================================
# Setup Objective Function
# =============================================================================

println("[Step 4] Setting up Objective Function")
println("-"^80)

objective_func = if OBJECTIVE_TYPE == "target_distance"
    obj = NeuralNetworkObjective(target, weight=1.0)
    if VERBOSE
        println("  Objective: minimize ||nn(x) - target||²")
    end
    obj
elseif OBJECTIVE_TYPE == "regularized"
    reg_center = REGULARIZATION_CENTER !== nothing ? REGULARIZATION_CENTER : x0
    obj = CompositeObjective(
        NeuralNetworkObjective(target, weight=1.0),
        QuadraticRegularization(reg_center, weight=REGULARIZATION_WEIGHT)
    )
    if VERBOSE
        println("  Objective: ||nn(x) - target||² + λ||x - center||²")
        println("  Regularization weight: $REGULARIZATION_WEIGHT")
    end
    obj
elseif OBJECTIVE_TYPE == "custom"
    reg_center = REGULARIZATION_CENTER !== nothing ? REGULARIZATION_CENTER : x0
    obj = CompositeObjective(
        NeuralNetworkObjective(target, weight=CUSTOM_NN_WEIGHT),
        QuadraticRegularization(reg_center, weight=CUSTOM_REG_WEIGHT)
    )
    if VERBOSE
        println("  Objective: custom composite")
        println("  NN weight: $CUSTOM_NN_WEIGHT")
        println("  Regularization weight: $CUSTOM_REG_WEIGHT")
    end
    obj
else
    error("Unknown OBJECTIVE_TYPE: $OBJECTIVE_TYPE")
end

println()

# =============================================================================
# Setup Constraints
# =============================================================================

println("[Step 5] Setting up Constraints")
println("-"^80)

constraint_func = if CONSTRAINT_TYPE == "box"
    if CUSTOM_BOX_BOUNDS !== nothing
        x_min, x_max = CUSTOM_BOX_BOUNDS
    else
        x_min = fill(BOX_LOWER, input_dim)
        x_max = fill(BOX_UPPER, input_dim)
    end
    cons = BoxConstraints(x_min, x_max)
    if VERBOSE
        println("  Constraints: box constraints")
        println("  Bounds: [$BOX_LOWER, $BOX_UPPER]")
    end
    cons
elseif CONSTRAINT_TYPE == "spherical"
    sphere_center = SPHERE_CENTER !== nothing ? SPHERE_CENTER : x0
    cons = SphericalConstraint(sphere_center, SPHERE_RADIUS)
    if VERBOSE
        println("  Constraints: spherical constraint")
        println("  Radius: $SPHERE_RADIUS")
    end
    cons
elseif CONSTRAINT_TYPE == "both"
    if CUSTOM_BOX_BOUNDS !== nothing
        x_min, x_max = CUSTOM_BOX_BOUNDS
    else
        x_min = fill(BOX_LOWER, input_dim)
        x_max = fill(BOX_UPPER, input_dim)
    end
    sphere_center = SPHERE_CENTER !== nothing ? SPHERE_CENTER : x0
    cons = CompositeConstraint(
        BoxConstraints(x_min, x_max),
        SphericalConstraint(sphere_center, SPHERE_RADIUS)
    )
    if VERBOSE
        println("  Constraints: box + spherical")
        println("  Box bounds: [$BOX_LOWER, $BOX_UPPER]")
        println("  Sphere radius: $SPHERE_RADIUS")
    end
    cons
elseif CONSTRAINT_TYPE == "none"
    if VERBOSE
        println("  Constraints: none (unconstrained)")
    end
    nothing
else
    error("Unknown CONSTRAINT_TYPE: $CONSTRAINT_TYPE")
end

println()

# =============================================================================
# Create NLP Model
# =============================================================================

println("[Step 6] Creating NLP Model")
println("-"^80)

use_python = (BACKEND == "jax")

if use_python
    println("  Using Python/JAX backend for automatic differentiation")
else
    println("  Using Julia/Flux backend with ForwardDiff")
end

nlp_model = NeuralNetworkNLPModel(
    MODEL_PATH,
    objective_func,
    constraint_func,
    x0;
    use_python=use_python
)

if VERBOSE
    println("✓ NLP model created successfully")
    println("  Number of variables: $(nlp_model.meta.nvar)")
    println("  Number of constraints: $(nlp_model.meta.ncon)")
end
println()

# =============================================================================
# Solve Optimization Problem
# =============================================================================

println("[Step 7] Solving Optimization Problem")
println("-"^80)
println()

# Build solver options
solver_options = Dict{Symbol, Any}(
    :max_iter => MAX_ITER,
    :tol => TOLERANCE,
    :print_level => PRINT_LEVEL
)

if LINEAR_SOLVER !== nothing
    solver_options[:linear_solver] = LINEAR_SOLVER
end

# Merge extra options
merge!(solver_options, EXTRA_SOLVER_OPTIONS)

# Solve
start_time = time()
result = solve_nlp(nlp_model; solver_options...)
solve_time = time() - start_time

# Extract timing statistics
timing_stats = result[:timing_stats]
total_eval_time = timing_stats.total_eval_time
madnlp_internal_time = solve_time - total_eval_time

println()
println("="^80)
println("[Optimization Complete]")
println("="^80)
println("  Status: $(result[:status])")
println("  Final objective: $(@sprintf("%.8e", result[:objective]))")
println("  Iterations: $(result[:iter_count])")
println("  Solve time: $(@sprintf("%.3f", solve_time)) seconds")
println("  Eval time: $(@sprintf("%.3f", total_eval_time)) seconds ($(@sprintf("%.1f", 100*total_eval_time/solve_time))%)")
println("  MadNLP internal time: $(@sprintf("%.3f", madnlp_internal_time)) seconds ($(@sprintf("%.1f", 100*madnlp_internal_time/solve_time))%)")
println()

# =============================================================================
# Verify and Analyze Solution
# =============================================================================

println("[Step 8] Solution Analysis")
println("-"^80)

x_sol = result[:solution]

# Basic statistics
println("\nSolution statistics:")
println("  ||x||₂: $(@sprintf("%.6f", norm(x_sol)))")
println("  ||x||∞: $(@sprintf("%.6f", maximum(abs.(x_sol))))")
println("  min(x): $(@sprintf("%.6f", minimum(x_sol)))")
println("  max(x): $(@sprintf("%.6f", maximum(x_sol)))")

# Constraint verification
if CONSTRAINT_TYPE == "box" || CONSTRAINT_TYPE == "both"
    box_satisfied = all(BOX_LOWER .<= x_sol .<= BOX_UPPER)
    println("\nBox constraint verification:")
    println("  All constraints satisfied: $(box_satisfied ? "✓ Yes" : "✗ No")")
    if !box_satisfied
        n_violated = sum((x_sol .< BOX_LOWER) .| (x_sol .> BOX_UPPER))
        println("  Number of violations: $n_violated")
        max_violation = max(
            maximum(BOX_LOWER .- x_sol),
            maximum(x_sol .- BOX_UPPER)
        )
        println("  Maximum violation: $(@sprintf("%.6e", max_violation))")
    end
end

if CONSTRAINT_TYPE == "spherical" || CONSTRAINT_TYPE == "both"
    sphere_center = SPHERE_CENTER !== nothing ? SPHERE_CENTER : x0
    dist_from_center = norm(x_sol - sphere_center)
    sphere_satisfied = dist_from_center <= SPHERE_RADIUS * (1 + TOLERANCE)
    println("\nSpherical constraint verification:")
    println("  ||x - center||: $(@sprintf("%.6f", dist_from_center))")
    println("  Radius: $(@sprintf("%.6f", SPHERE_RADIUS))")
    println("  Constraint satisfied: $(sphere_satisfied ? "✓ Yes" : "✗ No")")
    if !sphere_satisfied
        println("  Violation: $(@sprintf("%.6e", dist_from_center - SPHERE_RADIUS))")
    end
end

# Evaluate neural network output
println("\nNeural network evaluation:")
nn_output = nlp_model.neural_network !== nothing ? 
    nlp_model.neural_network(x_sol) : 
    nlp_model.python_evaluator.evaluate_nn(x_sol)

if nlp_model.neural_network === nothing
    nn_output = pyconvert(Vector{Float64}, nn_output)
end

println("  Output: $(round.(nn_output, digits=4))")
println("  Target: $(round.(target, digits=4))")

dist_to_target = norm(nn_output - target)
println("  ||output - target||: $(@sprintf("%.6e", dist_to_target))")

# Predicted class (if classification problem)
if output_dim > 1
    pred_class = argmax(nn_output)
    target_class = argmax(target)
    println("\nClassification analysis:")
    println("  Predicted class: $pred_class")
    println("  Target class: $target_class")
    println("  Match: $(pred_class == target_class ? "✓ Yes" : "✗ No")")
    println("  Confidence: $(@sprintf("%.4f", maximum(nn_output)))")
end

println()

# =============================================================================
# Save Results
# =============================================================================

if SAVE_RESULTS
    println("[Step 9] Saving Results")
    println("-"^80)
    
    mkpath(OUTPUT_DIR)
    
    # Generate filename if not provided
    filename = if RESULT_FILENAME !== nothing
        RESULT_FILENAME
    else
        timestamp = Dates.format(Dates.now(), "yyyymmdd_HHMMSS")
        backend_str = BACKEND
        obj_str = OBJECTIVE_TYPE
        cons_str = CONSTRAINT_TYPE
        "opt_$(backend_str)_$(obj_str)_$(cons_str)_$(timestamp).json"
    end
    
    filepath = joinpath(OUTPUT_DIR, filename)
    
    # Compute timing statistics
    per_iter_times = timing_stats.per_iteration_times
    avg_eval_time = length(per_iter_times) > 0 ? total_eval_time / length(per_iter_times) : 0.0
    
    # Prepare per-iteration timing data (group evaluations by approximate iteration)
    # Note: MadNLP may call multiple evaluations per iteration
    # We'll report individual evaluation times
    per_iteration_data = [
        Dict(
            "call" => i,
            "eval_time" => per_iter_times[i]
        )
        for i in 1:min(length(per_iter_times), 1000)  # Limit to first 1000 to avoid huge files
    ]
    
    # Prepare results dictionary
    results_dict = Dict(
        "timestamp" => string(Dates.now()),
        "configuration" => Dict(
            "model_path" => MODEL_PATH,
            "backend" => BACKEND,
            "objective_type" => OBJECTIVE_TYPE,
            "constraint_type" => CONSTRAINT_TYPE,
            "target_type" => TARGET_TYPE,
            "init_type" => INIT_TYPE,
            "random_seed" => RANDOM_SEED
        ),
        "problem" => Dict(
            "input_dim" => input_dim,
            "output_dim" => output_dim,
            "n_variables" => nlp_model.meta.nvar,
            "n_constraints" => nlp_model.meta.ncon
        ),
        "parameters" => Dict(
            "max_iter" => MAX_ITER,
            "tolerance" => TOLERANCE,
            "box_bounds" => CONSTRAINT_TYPE in ["box", "both"] ? [BOX_LOWER, BOX_UPPER] : nothing,
            "sphere_radius" => CONSTRAINT_TYPE in ["spherical", "both"] ? SPHERE_RADIUS : nothing,
            "regularization_weight" => OBJECTIVE_TYPE in ["regularized", "custom"] ? 
                (OBJECTIVE_TYPE == "regularized" ? REGULARIZATION_WEIGHT : CUSTOM_REG_WEIGHT) : 
                nothing
        ),
        "results" => Dict(
            "status" => string(result[:status]),
            "objective" => result[:objective],
            "iterations" => result[:iter_count],
            "solve_time" => solve_time,
            "solution_norm" => norm(x_sol),
            "solution_norm_inf" => maximum(abs.(x_sol)),
            "dist_to_target" => dist_to_target
        ),
        "timing" => Dict(
            "total_solve_time" => solve_time,
            "total_eval_time" => total_eval_time,
            "total_madnlp_internal_time" => madnlp_internal_time,
            "eval_count" => timing_stats.eval_count,
            "per_iteration" => per_iteration_data,
            "statistics" => Dict(
                "avg_eval_time" => avg_eval_time,
                "eval_time_percentage" => 100 * total_eval_time / solve_time,
                "madnlp_internal_percentage" => 100 * madnlp_internal_time / solve_time
            )
        ),
        "solution" => Dict(
            "x" => x_sol,
            "nn_output" => nn_output,
            "target" => target,
            "initial_point" => x0
        )
    )
    
    # Save to file
    open(filepath, "w") do io
        JSON3.pretty(io, results_dict)
    end
    
    println("✓ Results saved to: $filepath")
    println()
end

# =============================================================================
# Summary
# =============================================================================

println("="^80)
println("[Summary]")
println("="^80)
println("""
Optimization completed successfully!

Configuration:
  Model: $(basename(MODEL_PATH))
  Backend: $BACKEND
  Objective: $OBJECTIVE_TYPE
  Constraints: $CONSTRAINT_TYPE

Performance:
  Status: $(result[:status])
  Objective: $(@sprintf("%.6e", result[:objective]))
  Iterations: $(result[:iter_count])
  Time: $(@sprintf("%.3f", solve_time))s

Solution Quality:
  Distance to target: $(@sprintf("%.6e", dist_to_target))
  Solution norm: $(@sprintf("%.6f", norm(x_sol)))
""")

if output_dim > 1
    pred_class = argmax(nn_output)
    target_class = argmax(target)
    println("  Predicted class: $pred_class (target: $target_class)")
end

println()
println("Completed at: $(Dates.now())")
println("="^80)

