module MadNLP4NN

using PythonCall
using JSON3
using MadNLP
using NLPModels
using Flux
using LinearAlgebra
using ForwardDiff

# Python interface exports
export setup_python_env, create_dataset, train_model, load_trained_model

# NLP interface exports
export load_pytorch_model_to_flux
export smooth_relu
export AbstractObjectiveFunction, AbstractConstraintFunction
export NeuralNetworkObjective, QuadraticRegularization, CompositeObjective
export BoxConstraints, SphericalConstraint, CompositeConstraint
export NeuralNetworkNLPModel
export solve_nlp, create_simple_nlp
export evaluate, num_constraints

include("python_interface.jl")
include("nlp_interface.jl")
include("nlp_model.jl")

end # module
