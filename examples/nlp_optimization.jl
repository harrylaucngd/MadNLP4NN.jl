# NLP Optimization Example - Redirects to nlp_evaluation.jl
# 
# This file has been replaced by a more comprehensive evaluation example.
# Please use nlp_evaluation.jl instead.

println("="^80)
println("MadNLP4NN.jl - NLP Optimization")
println("="^80)

println("""

NOTE: This example has been replaced by a more comprehensive version.

Please run the new evaluation script:
    julia examples/nlp_evaluation.jl

The new script includes:
1. Loading trained neural networks from disk
2. Creating NLP models with custom objectives and constraints
3. Solving with MadNLP
4. Three detailed examples:
   - Box constraints
   - Spherical constraints with regularization
   - Composite objectives and constraints

The NLP formulation is now fully implemented with:
- Modular objective functions (f(x, θ))
- Modular constraint functions (g(x) ≤ 0)
- Smooth ReLU handling for differentiability
- Automatic differentiation for gradients and Hessians
- Full NLPModels interface implementation

See also:
- examples/nlp_evaluation.jl (comprehensive evaluation)
- src/nlp_interface.jl (objective/constraint definitions)
- src/nlp_model.jl (NLPModels implementation)
- README.md (updated documentation)

""")

println("="^80)
println("Redirecting to comprehensive evaluation...")
println("="^80)

# Run the comprehensive evaluation
include("nlp_evaluation.jl")
