# NLP Evaluation Example
# Demonstrates how to use MadNLP4NN for optimization with neural networks

using MadNLP4NN
using MadNLP
using LinearAlgebra
using Random
using JSON3

println("="^80)
println("MadNLP4NN.jl - Neural Network Optimization Evaluation")
println("="^80)

# =============================================================================
# Configuration
# =============================================================================

# Set random seed for reproducibility
Random.seed!(42)

# Model selection - use one of the trained models
# You can modify this to use different models from output/models/
dataset_name = "GM_n10000_d500_c10_comp10_s123"  # Gaussian mixture dataset
model_type = "small_mlp"  # Options: small_mlp, medium_mlp, large_mlp, small_resmlp, medium_resmlp, large_resmlp
seed = 123

model_path = joinpath("output", "models", dataset_name, model_type, "model_seed$(seed).pt")

println("\n[Configuration]")
println("  Dataset: $dataset_name")
println("  Model: $model_type")
println("  Seed: $seed")
println("  Model path: $model_path")

# Check if model exists
if !isfile(model_path)
    println("\n⚠ Error: Model not found at $model_path")
    println("\nPlease train the model first by running:")
    println("  julia examples/train_all.jl")
    println("\nOr modify the dataset_name and model_type variables above to use an existing model.")
    exit(1)
end

# =============================================================================
# Load Model and Inspect
# =============================================================================

println("\n" * "="^80)
println("[Step 1] Loading Neural Network Model")
println("="^80)

# Load model metadata
state_dict, model_config, train_args, history = load_trained_model(model_path)

println("\n✓ Model loaded successfully!")
println("  Model type: $(model_config["model_type"])")
println("  Input dimension: $(model_config["input_dim"])")
println("  Output dimension: $(model_config["output_dim"])")
println("  Hidden layers: $(model_config["hidden_dims"])")

if haskey(history, "best_val_acc")
    println("  Training accuracy: $(round(history["best_val_acc"], digits=4))")
end

input_dim = model_config["input_dim"]
output_dim = model_config["output_dim"]

# =============================================================================
# Example 1: Simple Minimization with Box Constraints
# =============================================================================

println("\n" * "="^80)
println("[Example 1] Minimize Distance to Target with Box Constraints")
println("="^80)

println("""
Problem formulation:
    min_x  ||nn(x) - target||²
    s.t.   -1 ≤ x ≤ 1
    
where nn(x) is the neural network output.
""")

# Define target output (e.g., one-hot encoding for class 0)
target = zeros(output_dim)
target[1] = 1.0

println("Target output: ", target)

# Initial point (random initialization within bounds)
x0_ex1 = rand(input_dim) .* 2 .- 1  # Random in [-1, 1]

println("Initial point: random in [-1, 1]")
println("  Dimension: $input_dim")

# Create NLP model
println("\nCreating NLP model...")
nlp_ex1 = create_simple_nlp(
    model_path,
    target,
    x0_ex1,
    bounds=(-1.0, 1.0)
)

println("✓ NLP model created")
println("  Variables: $(nlp_ex1.meta.nvar)")
println("  Constraints: $(nlp_ex1.meta.ncon)")

# Solve
println("\nSolving with MadNLP...")
result_ex1 = solve_nlp(
    nlp_ex1,
    max_iter=100,
    tol=1e-4,
    print_level=MadNLP.ERROR  # Suppress detailed output
)

println("\n✓ Optimization complete!")
println("  Status: $(result_ex1[:status])")
println("  Final objective: $(round(result_ex1[:objective], digits=6))")
println("  Iterations: $(result_ex1[:iter_count])")

# Verify solution
x_sol_ex1 = result_ex1[:solution]
println("\nSolution verification:")
println("  ||x||_∞: $(round(maximum(abs.(x_sol_ex1)), digits=6))")
println("  Constraint satisfied: $(maximum(abs.(x_sol_ex1)) <= 1.0)")

# Evaluate neural network at solution
nn_output_ex1 = nlp_ex1.neural_network(x_sol_ex1)
println("  Neural network output: ", round.(nn_output_ex1, digits=4))
println("  Distance to target: $(round(norm(nn_output_ex1 - target), digits=6))")

# =============================================================================
# Example 2: Regularized Optimization with Spherical Constraint
# =============================================================================

println("\n" * "="^80)
println("[Example 2] Regularized Optimization with Spherical Constraint")
println("="^80)

println("""
Problem formulation:
    min_x  ||nn(x) - target||² + λ||x - x0||²
    s.t.   ||x - x0||² ≤ r²
    
where λ is the regularization weight and r is the radius.
""")

# Parameters
target_ex2 = zeros(output_dim)
target_ex2[end] = 1.0  # Target last class
x0_ex2 = randn(input_dim) .* 0.1  # Small random initialization
radius = 0.5
reg_weight = 0.01

println("Target output: ", target_ex2)
println("Initial point: N(0, 0.01)")
println("Radius: $radius")
println("Regularization weight: $reg_weight")

# Create NLP model
println("\nCreating NLP model...")
nlp_ex2 = create_simple_nlp(
    model_path,
    target_ex2,
    x0_ex2,
    radius=radius,
    regularization_weight=reg_weight
)

println("✓ NLP model created")
println("  Variables: $(nlp_ex2.meta.nvar)")
println("  Constraints: $(nlp_ex2.meta.ncon)")

# Solve
println("\nSolving with MadNLP...")
result_ex2 = solve_nlp(
    nlp_ex2,
    max_iter=100,
    tol=1e-4,
    print_level=MadNLP.ERROR
)

println("\n✓ Optimization complete!")
println("  Status: $(result_ex2[:status])")
println("  Final objective: $(round(result_ex2[:objective], digits=6))")
println("  Iterations: $(result_ex2[:iter_count])")

# Verify solution
x_sol_ex2 = result_ex2[:solution]
dist_from_center = norm(x_sol_ex2 - x0_ex2)
println("\nSolution verification:")
println("  ||x - x0||: $(round(dist_from_center, digits=6))")
println("  Radius constraint: $(dist_from_center <= radius ? "✓ Satisfied" : "✗ Violated")")

# Evaluate neural network at solution
nn_output_ex2 = nlp_ex2.neural_network(x_sol_ex2)
println("  Neural network output: ", round.(nn_output_ex2, digits=4))
println("  Distance to target: $(round(norm(nn_output_ex2 - target_ex2), digits=6))")

# =============================================================================
# Example 3: Custom Composite Objective
# =============================================================================

println("\n" * "="^80)
println("[Example 3] Custom Composite Objective and Constraints")
println("="^80)

println("""
Problem formulation:
    min_x  w1 * ||nn(x) - target||² + w2 * ||x||²
    s.t.   -2 ≤ x ≤ 2
           ||x||² ≤ r²
    
Combining both box and spherical constraints.
""")

# Parameters
target_ex3 = ones(output_dim) ./ output_dim  # Uniform distribution
x0_ex3 = zeros(input_dim)
radius_ex3 = 1.0
reg_weight_ex3 = 0.001

println("Target output: uniform distribution")
println("Initial point: zeros")
println("Box bounds: [-2, 2]")
println("Spherical radius: $radius_ex3")
println("Regularization weight: $reg_weight_ex3")

# Create composite objective manually
obj_ex3 = CompositeObjective(
    NeuralNetworkObjective(target_ex3, weight=1.0),
    QuadraticRegularization(x0_ex3, weight=reg_weight_ex3)
)

# Create composite constraints
cons_ex3 = CompositeConstraint(
    BoxConstraints(fill(-2.0, input_dim), fill(2.0, input_dim)),
    SphericalConstraint(x0_ex3, radius_ex3)
)

# Create NLP model
println("\nCreating NLP model with custom objectives and constraints...")
nlp_ex3 = NeuralNetworkNLPModel(
    model_path,
    obj_ex3,
    cons_ex3,
    x0_ex3
)

println("✓ NLP model created")
println("  Variables: $(nlp_ex3.meta.nvar)")
println("  Constraints: $(nlp_ex3.meta.ncon)")

# Solve
println("\nSolving with MadNLP...")
result_ex3 = solve_nlp(
    nlp_ex3,
    max_iter=100,
    tol=1e-4,
    print_level=MadNLP.ERROR
)

println("\n✓ Optimization complete!")
println("  Status: $(result_ex3[:status])")
println("  Final objective: $(round(result_ex3[:objective], digits=6))")
println("  Iterations: $(result_ex3[:iter_count])")

# Verify solution
x_sol_ex3 = result_ex3[:solution]
println("\nSolution verification:")
println("  ||x||_∞: $(round(maximum(abs.(x_sol_ex3)), digits=6))")
println("  Box constraint: $(maximum(abs.(x_sol_ex3)) <= 2.0 ? "✓ Satisfied" : "✗ Violated")")
println("  ||x||: $(round(norm(x_sol_ex3), digits=6))")
println("  Spherical constraint: $(norm(x_sol_ex3) <= radius_ex3 ? "✓ Satisfied" : "✗ Violated")")

# Evaluate neural network at solution
nn_output_ex3 = nlp_ex3.neural_network(x_sol_ex3)
println("  Neural network output: ", round.(nn_output_ex3, digits=4))
println("  Distance to target: $(round(norm(nn_output_ex3 - target_ex3), digits=6))")

# =============================================================================
# Summary and Export Results
# =============================================================================

println("\n" * "="^80)
println("[Summary] All Examples Completed")
println("="^80)

summary_results = Dict(
    "model_path" => model_path,
    "model_config" => model_config,
    "examples" => [
        Dict(
            "name" => "Box Constraints",
            "status" => string(result_ex1[:status]),
            "objective" => result_ex1[:objective],
            "iterations" => result_ex1[:iter_count],
            "solution_norm_inf" => maximum(abs.(x_sol_ex1)),
            "output_distance" => norm(nn_output_ex1 - target)
        ),
        Dict(
            "name" => "Spherical Constraint + Regularization",
            "status" => string(result_ex2[:status]),
            "objective" => result_ex2[:objective],
            "iterations" => result_ex2[:iter_count],
            "solution_distance_from_center" => norm(x_sol_ex2 - x0_ex2),
            "output_distance" => norm(nn_output_ex2 - target_ex2)
        ),
        Dict(
            "name" => "Composite Objectives and Constraints",
            "status" => string(result_ex3[:status]),
            "objective" => result_ex3[:objective],
            "iterations" => result_ex3[:iter_count],
            "solution_norm" => norm(x_sol_ex3),
            "output_distance" => norm(nn_output_ex3 - target_ex3)
        )
    ]
)

# Save results
output_dir = "output/evaluation_results"
mkpath(output_dir)
results_file = joinpath(output_dir, "nlp_evaluation_$(dataset_name)_$(model_type).json")

open(results_file, "w") do io
    JSON3.pretty(io, summary_results)
end

println("\n✓ Results saved to: $results_file")

println("\n" * "="^80)
println("Evaluation Complete!")
println("="^80)

println("""
Key Takeaways:
1. The NLP formulation successfully optimizes neural network inputs
2. Smooth ReLU ensures differentiability for second-order methods
3. Both box and spherical constraints can be enforced
4. Composite objectives allow flexible problem formulations
5. MadNLP efficiently solves the constrained optimization problems

Next Steps:
- Try different model architectures (ResMLPs, larger networks)
- Experiment with different constraint types
- Implement custom objective functions for specific applications
- Use the framework for adversarial example generation
- Explore GPU acceleration for larger problems
""")

