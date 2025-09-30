module MadNLP4NN

using PythonCall
using JSON3
using MadNLP
using NLPModels

export setup_python_env, create_dataset, train_model, load_trained_model
export AdversarialNLPModel

include("python_interface.jl")
include("nlp_interface.jl")

# Module initialization: set up Python path
function __init__()
    try
        # Add Python directory to Python's sys.path
        python_dir = joinpath(@__DIR__, "..", "python")
        abs_python_dir = abspath(python_dir)
        
        sys = pyimport("sys")
        
        # Check if already in path (convert all paths to strings for comparison)
        path_strings = [string(p) for p in sys.path]
        
        if !(abs_python_dir in path_strings)
            # Use pylist to get a mutable reference and insert
            # Must use sys.path.insert directly, not on a temporary pylist
            sys.path.insert(0, abs_python_dir)
            @info "✓ Added Python directory to sys.path: $abs_python_dir"
        else
            @info "✓ Python directory already in sys.path: $abs_python_dir"
        end
    catch e
        @warn "Failed to set up Python path in __init__" exception=(e, catch_backtrace())
    end
end

end # module
