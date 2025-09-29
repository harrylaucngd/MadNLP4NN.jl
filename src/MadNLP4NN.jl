module MadNLP4NN

using PythonCall
using JSON3
using MadNLP
using NLPModels

export setup_python_env, create_dataset, train_model, load_trained_model
export AdversarialNLPModel

include("python_interface.jl")
include("nlp_interface.jl")

end # module
