# NLP Optimization Example (TODO - Requires Implementation)
# This is a placeholder showing the intended API

using MadNLP4NN
using MadNLP
using LinearAlgebra

println("="^80)
println("MadNLP4NN.jl - Adversarial NLP Optimization")
println("="^80)

println("""
NOTE: This example is currently a placeholder.
The NLP interface needs to be implemented in src/nlp_interface.jl

This file shows the intended API and workflow.
""")

# =============================================================================
# Step 1: Load a trained model
# =============================================================================
println("\n[Step 1] Load Trained Model")
println("-"^80)

# You would use a real model path here
model_path = "./output/models/mnist_medium_mlp/model_seed42.pt"

# Check if model exists
if !isfile(model_path)
    println("⚠ Model not found: $model_path")
    println("Please train a model first using:")
    println("  julia examples/basic_usage.jl")
    exit(1)
end

# Load model
state_dict, config, train_args, history = load_trained_model(model_path)
println("✓ Model loaded: $(config["model_type"])")
println("  Input dim: $(config["input_dim"])")
println("  Output dim: $(config["output_dim"])")

# =============================================================================
# Step 2: Define adversarial problem
# =============================================================================
println("\n[Step 2] Define Adversarial Problem")
println("-"^80)

# Original input (in practice, load from dataset)
input_dim = config["input_dim"]
output_dim = config["output_dim"]

x_star = rand(input_dim)  # Original input
y_target = rand(output_dim)  # Target output (to maximize distance from)

# Perturbation radius
epsilon = 0.1

println("  Original input dimension: $input_dim")
println("  Target output dimension: $output_dim")
println("  Perturbation radius ε: $epsilon")

# =============================================================================
# Step 3: Create NLP model (TODO - Not yet implemented)
# =============================================================================
println("\n[Step 3] Create NLP Model")
println("-"^80)

println("TODO: Implement create_adversarial_nlp")
println("""
This function should:
1. Load network parameters from the model
2. Create an NLPModel that implements:
   - obj(x): -||f(x; θ) - y||² (negative for maximization)
   - grad!(x, g): Gradient via backpropagation
   - hess_structure!: Hessian sparsity pattern
   - hess_coord!: Hessian values
   - cons!(x, c): Constraint ||x - x*|| ≤ ε
   - jac_structure!: Jacobian sparsity pattern
   - jac_coord!: Jacobian values
3. Return NLPModels.jl compatible model
""")

# This would be the actual call:
# nlp = create_adversarial_nlp(
#     model_path,
#     x_star,
#     y_target,
#     epsilon=epsilon,
#     constraint_type="l2"  # or "linf"
# )

# =============================================================================
# Step 4: Solve with MadNLP (TODO - Not yet implemented)
# =============================================================================
println("\n[Step 4] Solve with MadNLP")
println("-"^80)

println("TODO: Implement solve_adversarial")
println("""
This function should:
1. Configure MadNLP solver with appropriate options
2. Solve the NLP: result = MadNLP.solve!(solver)
3. Extract and return solution

Example configuration:
  solver = MadNLPSolver(
      nlp;
      max_iter=1000,
      tol=1e-6,
      print_level=MadNLP.INFO,
      linear_solver=MadNLPMumps  # Or MadNLPGPU for GPU
  )
""")

# This would be the actual call:
# solver_options = Dict(
#     "max_iter" => 1000,
#     "tol" => 1e-6,
#     "linear_solver" => MadNLPMumps
# )
# result = solve_adversarial(nlp, solver_options=solver_options)

# =============================================================================
# Step 5: Verify and analyze results (TODO)
# =============================================================================
println("\n[Step 5] Verify and Analyze Results")
println("-"^80)

println("""
After solving, you would:
1. Extract adversarial point: x_adv = result[:solution]
2. Verify constraint: ||x_adv - x_star|| ≤ ε
3. Compute network output: y_adv = f(x_adv; θ)
4. Check objective: ||y_adv - y_target||²
5. Analyze success rate, convergence, etc.
""")

# =============================================================================
# Implementation Roadmap
# =============================================================================
println("\n" * "="^80)
println("Implementation Roadmap")
println("="^80)

println("""
To complete the NLP interface, you need to:

1. Parse Network Architecture
   - Extract layer sizes from saved model
   - Identify weight/bias parameters for each layer
   - Handle different architectures (MLP, ResidualMLP)

2. Implement Forward Pass
   - Matrix multiplication: z = Wx + b
   - ReLU activation: a = max(0, z)
   - Track intermediate activations for backprop

3. Implement Backward Pass (Gradient)
   - Backpropagation through network
   - Chain rule through ReLU
   - Compute ∇f(x) = ∂||f(x) - y||²/∂x

4. Implement Hessian
   - Option A: Exact Hessian (expensive)
   - Option B: Hessian-vector products (efficient)
   - Option C: Quasi-Newton approximation

5. Implement Constraints
   - L2 constraint: ||x - x*||² ≤ ε²
     Gradient: ∇c(x) = 2(x - x*)
     Hessian: ∇²c(x) = 2I
   - L∞ constraint: x* - ε ≤ x ≤ x* + ε (box constraints)

6. Create NLPModel
   - Subtype NLPModels.AbstractNLPModel
   - Implement required methods (obj, grad!, cons!, etc.)
   - Define sparsity patterns

7. Integrate with MadNLP
   - Create solver instance
   - Configure for GPU (if available)
   - Set appropriate tolerances
   - Solve and extract results

See src/nlp_interface.jl for detailed hints and template code.
""")

println("\n" * "="^80)
println("For help implementing this, refer to:")
println("  - src/nlp_interface.jl (implementation hints)")
println("  - docs/TUTORIAL.md (planned API)")
println("  - NLPModels.jl documentation")
println("  - MadNLP.jl documentation")
println("="^80)
