#!/usr/bin/env julia

"""
run_optimization_examples.jl

Example configurations for run_optimization.jl

This file demonstrates various configuration options for the optimization script.
Copy any of these configurations to run_optimization.jl to use them.
"""

# =============================================================================
# Example 1: Simple Box-Constrained Optimization (Default)
# =============================================================================
"""
Minimize distance to one-hot target with box constraints [-1, 1]

Configuration:
"""
const EXAMPLE_1 = """
const MODEL_PATH = "output/models/GM_n10000_d500_c10_comp10_s123/small_mlp/model_seed123.pt"
const BACKEND = "flux"
const OBJECTIVE_TYPE = "target_distance"
const CONSTRAINT_TYPE = "box"
const TARGET_TYPE = "one_hot"
const TARGET_CLASS = 1
const INIT_TYPE = "random"
const BOX_LOWER = -1.0
const BOX_UPPER = 1.0
const MAX_ITER = 100
const TOLERANCE = 1e-4
"""

# =============================================================================
# Example 2: Regularized Optimization with Spherical Constraint
# =============================================================================
"""
Minimize distance to target with regularization, constrained to a sphere

Configuration:
"""
const EXAMPLE_2 = """
const MODEL_PATH = "output/models/GM_n10000_d500_c10_comp10_s123/small_mlp/model_seed123.pt"
const BACKEND = "flux"
const OBJECTIVE_TYPE = "regularized"
const CONSTRAINT_TYPE = "spherical"
const TARGET_TYPE = "uniform"
const INIT_TYPE = "randn"
const INIT_SCALE = 0.1
const SPHERE_RADIUS = 0.5
const REGULARIZATION_WEIGHT = 0.01
const MAX_ITER = 200
const TOLERANCE = 1e-6
"""

# =============================================================================
# Example 3: Box + Spherical Constraints with JAX Backend
# =============================================================================
"""
Use Python/JAX for automatic differentiation with both constraint types

Configuration:
"""
const EXAMPLE_3 = """
const MODEL_PATH = "output/models/mnist/medium_mlp/model_seed42.pt"
const BACKEND = "jax"  # Use Python/JAX backend
const OBJECTIVE_TYPE = "regularized"
const CONSTRAINT_TYPE = "both"
const TARGET_TYPE = "one_hot"
const TARGET_CLASS = 5
const INIT_TYPE = "zeros"
const BOX_LOWER = -2.0
const BOX_UPPER = 2.0
const SPHERE_RADIUS = 1.0
const REGULARIZATION_WEIGHT = 0.001
const MAX_ITER = 150
const TOLERANCE = 1e-5
const PRINT_LEVEL = MadNLP.INFO  # More verbose output
"""

# =============================================================================
# Example 4: Unconstrained Optimization
# =============================================================================
"""
Unconstrained optimization to find input that maximizes specific output

Configuration:
"""
const EXAMPLE_4 = """
const MODEL_PATH = "output/models/fashionmnist/large_mlp/model_seed123.pt"
const BACKEND = "flux"
const OBJECTIVE_TYPE = "target_distance"
const CONSTRAINT_TYPE = "none"  # No constraints
const TARGET_TYPE = "one_hot"
const TARGET_CLASS = 3
const INIT_TYPE = "randn"
const INIT_SCALE = 0.5
const MAX_ITER = 300
const TOLERANCE = 1e-6
"""

# =============================================================================
# Example 5: Custom Objective with Tight Constraints
# =============================================================================
"""
Custom weighted objective with tight box and spherical constraints

Configuration:
"""
const EXAMPLE_5 = """
const MODEL_PATH = "output/models/GM_n20000_d500_c10_comp20_s42/small_resmlp/model_seed42.pt"
const BACKEND = "flux"
const OBJECTIVE_TYPE = "custom"
const CONSTRAINT_TYPE = "both"
const TARGET_TYPE = "uniform"
const INIT_TYPE = "random"
const INIT_SCALE = 0.5
const BOX_LOWER = -0.8
const BOX_UPPER = 0.8
const SPHERE_RADIUS = 0.3
const CUSTOM_NN_WEIGHT = 2.0
const CUSTOM_REG_WEIGHT = 0.05
const MAX_ITER = 200
const TOLERANCE = 1e-5
const PRINT_LEVEL = MadNLP.DEBUG
"""

# =============================================================================
# Example 6: High-Precision Optimization
# =============================================================================
"""
High-precision optimization with tight tolerance

Configuration:
"""
const EXAMPLE_6 = """
const MODEL_PATH = "output/models/GM_n10000_d500_c10_comp10_s123/small_mlp/model_seed123.pt"
const BACKEND = "flux"
const OBJECTIVE_TYPE = "regularized"
const CONSTRAINT_TYPE = "box"
const TARGET_TYPE = "one_hot"
const TARGET_CLASS = 7
const INIT_TYPE = "zeros"
const BOX_LOWER = -1.5
const BOX_UPPER = 1.5
const REGULARIZATION_WEIGHT = 0.001
const MAX_ITER = 500
const TOLERANCE = 1e-8  # Very tight tolerance
const PRINT_LEVEL = MadNLP.INFO
"""

# =============================================================================
# Example 7: Adversarial Example Generation
# =============================================================================
"""
Generate adversarial examples by minimizing confidence of correct class
while staying close to original input

Configuration:
"""
const EXAMPLE_7 = """
# For adversarial examples, you would:
# 1. Load an image from test set as CUSTOM_INIT
# 2. Set target to different class
# 3. Use small sphere radius to stay close to original
# 4. Use regularization to maintain similarity

const MODEL_PATH = "output/models/mnist/large_mlp/model_seed42.pt"
const BACKEND = "flux"
const OBJECTIVE_TYPE = "regularized"
const CONSTRAINT_TYPE = "both"
const TARGET_TYPE = "one_hot"
const TARGET_CLASS = 8  # Target wrong class
const INIT_TYPE = "custom"
# const CUSTOM_INIT = original_image  # Set to actual image
const BOX_LOWER = 0.0  # Valid pixel range
const BOX_UPPER = 1.0
const SPHERE_RADIUS = 0.1  # Stay close to original
const REGULARIZATION_WEIGHT = 0.1
const MAX_ITER = 200
const TOLERANCE = 1e-5
"""

# =============================================================================
# Usage Instructions
# =============================================================================

println("""
================================================================================
run_optimization.jl - Example Configurations
================================================================================

To use any of these examples:

1. Open run_optimization.jl in your editor
2. Copy the configuration variables from one of the examples above
3. Paste them into the CONFIGURATION section of run_optimization.jl
4. Run: julia run_optimization.jl

Available Examples:
-------------------

Example 1: Simple Box-Constrained Optimization (Default)
  - One-hot target with box constraints
  - Good starting point for most problems

Example 2: Regularized Optimization with Spherical Constraint
  - Uniform target distribution
  - Spherical constraint to limit distance from origin
  - Regularization to encourage small solutions

Example 3: Box + Spherical Constraints with JAX Backend
  - Uses Python/JAX for automatic differentiation
  - Combines both constraint types
  - Good for testing JAX backend

Example 4: Unconstrained Optimization
  - No constraints on variables
  - Useful for exploration of solution space

Example 5: Custom Objective with Tight Constraints
  - Custom weights for objective terms
  - Tight constraints for controlled solutions

Example 6: High-Precision Optimization
  - Very tight tolerance for precise solutions
  - More iterations for convergence

Example 7: Adversarial Example Generation
  - Template for generating adversarial examples
  - Requires custom initial point from dataset

Quick Configuration Guide:
--------------------------

MODEL_PATH: Path to trained model (.pt file)
  - Check output/models/ directory for available models

BACKEND: "flux" or "jax"
  - "flux": Julia/ForwardDiff (faster, default)
  - "jax": Python/JAX (more flexible, requires Python setup)

OBJECTIVE_TYPE: "target_distance", "regularized", or "custom"
  - "target_distance": minimize ||nn(x) - target||²
  - "regularized": add regularization term λ||x - center||²
  - "custom": custom weights for each objective term

CONSTRAINT_TYPE: "box", "spherical", "both", or "none"
  - "box": element-wise bounds on variables
  - "spherical": ||x - center|| ≤ radius
  - "both": combine box and spherical
  - "none": unconstrained

TARGET_TYPE: "one_hot", "uniform", "zeros", "ones", or "custom"
  - "one_hot": target specific class (set TARGET_CLASS)
  - "uniform": uniform distribution over all classes
  - "zeros": target all zeros
  - "ones": target all ones
  - "custom": provide custom target vector

INIT_TYPE: "random", "zeros", "randn", or "custom"
  - "random": uniform random in [-INIT_SCALE, INIT_SCALE]
  - "zeros": all zeros
  - "randn": normal distribution N(0, INIT_SCALE²)
  - "custom": provide custom initial point

Solver Parameters:
  MAX_ITER: maximum iterations (default: 100)
  TOLERANCE: convergence tolerance (default: 1e-4)
  PRINT_LEVEL: MadNLP.ERROR, WARN, INFO, or DEBUG

For more details, see the comments in run_optimization.jl

================================================================================
""")

