module MadNLP4NN

using PythonCall
using JSON3
using MadNLP
using NLPModels
using Flux
using LinearAlgebra
using ForwardDiff
using SparseArrays
using Statistics
using CUDA
using CUDSS
using DLPack
using MadNLPGPU

# ------------------------------------------------------------------
# Python interface (dataset creation, training)
# ------------------------------------------------------------------
export setup_python_env, train_darcy_fno, train_pareto_surrogates, load_trained_model

# ------------------------------------------------------------------
# Core NLP interface: activation, model loading
# ------------------------------------------------------------------
export load_pytorch_model_to_flux
export smooth_relu

# ------------------------------------------------------------------
# Objective functions (original + proposal extensions)
# ------------------------------------------------------------------
export AbstractObjectiveFunction
export NeuralNetworkObjective, QuadraticRegularization, CompositeObjective
export SurrogateInversionObjective, WeightedScalarizationObjective

# ------------------------------------------------------------------
# Constraint functions (original + proposal extensions)
# ------------------------------------------------------------------
export AbstractConstraintFunction
export BoxConstraints, SphericalConstraint, CompositeConstraint
export BudgetConstraint, SmoothnessConstraint, LearnedFeasibilityConstraint

# ------------------------------------------------------------------
# Evaluation and Hessian helpers
# ------------------------------------------------------------------
export evaluate, num_constraints
export hessian_contribution, has_closed_form_hessian

# ------------------------------------------------------------------
# Problem specification
# ------------------------------------------------------------------
export ProblemSpec
export laplacian_1d, laplacian_2d
export DarcyProblemConfig, create_darcy_nlp, create_darcy_stage_a, create_darcy_stage_b, create_darcy_stage_c
export ParetoProblemConfig, create_pareto_nlp, pareto_front_sweep
export evaluate_pareto_objectives, pareto_hypervolume, pareto_spread

# ------------------------------------------------------------------
# NLP model and solver interface
# ------------------------------------------------------------------
export NeuralNetworkNLPModel
export DLPackGPUModel, verify_jax_dlpack
export TimingStats
export solve_nlp, create_simple_nlp
export select_linear_solver

include("python_interface.jl")
include("nlp_interface.jl")
include("problem_spec.jl")
include("nlp_model.jl")
include("gpu_interop.jl")
include("gpu_nlp_model.jl")
include("darcy_problem.jl")
include("pareto_problem.jl")

end # module
