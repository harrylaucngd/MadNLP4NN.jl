"""
Julia interface to Python training scripts.
Uses PythonCall.jl to execute Python code.
"""

using PythonCall
using JSON3

# Python environment setup
const PYTHON_DIR = joinpath(@__DIR__, "..", "python")

"""
    setup_python_env(; force_reinstall=false)

Set up Python environment for MadNLP4NN.
This creates a conda environment and installs required packages.

# Arguments
- `force_reinstall::Bool`: Whether to force reinstallation of packages

# Example
```julia
using MadNLP4NN
setup_python_env()
```
"""
function setup_python_env(; force_reinstall=false)
    @info "Setting up Python environment for MadNLP4NN..."
    
    # Add Python directory to path (use absolute path)
    abs_python_dir = abspath(PYTHON_DIR)
    sys = pyimport("sys")
    path_strings = [string(p) for p in sys.path]
    
    if !(abs_python_dir in path_strings)
        sys.path.insert(0, abs_python_dir)
        @info "Added Python directory to sys.path: $abs_python_dir"
    else
        @info "Python directory already in sys.path: $abs_python_dir"
    end
    
    # Check if required packages are installed
    requirements_file = joinpath(PYTHON_DIR, "requirements.txt")
    
    if force_reinstall
        @info "Force reinstalling Python packages..."
        run(`pip install -r $requirements_file`)
    else
        @info "Python path configured. Make sure you have installed requirements:"
        @info "  pip install -r $requirements_file"
    end
    
    @info "Python environment setup complete!"
    return nothing
end


"""
    create_dataset(;
        dataset_type::String,
        output_dir::String="./output",
        n_samples::Int=10000,
        input_dim::Int=784,
        output_dim::Int=10,
        kwargs...
    )

Create and save a dataset using Python backend.

# Arguments
- `dataset_type::String`: Type of dataset ("gaussian_mixture", "nonlinear_manifold", "mnist", "fashionmnist", "cifar10")
- `output_dir::String`: Directory to save dataset
- `n_samples::Int`: Number of samples (for synthetic datasets)
- `input_dim::Int`: Input dimension
- `output_dim::Int`: Output dimension
- `kwargs...`: Additional dataset-specific arguments

# Returns
- Dictionary with dataset metadata

# Example
```julia
metadata = create_dataset(
    dataset_type="gaussian_mixture",
    n_samples=10000,
    input_dim=784,
    output_dim=10,
    n_components=20
)
```
"""
function create_dataset(;
    dataset_type::String,
    output_dir::String="./output",
    n_samples::Int=10000,
    input_dim::Int=784,
    output_dim::Int=10,
    kwargs...
)
    @info "Creating dataset: $dataset_type"
    
    # Import Python modules
    # Ensure Python directory is in path (use absolute path)
    abs_python_dir = abspath(PYTHON_DIR)
    sys = pyimport("sys")
    path_strings = [string(p) for p in sys.path]
    
    if !(abs_python_dir in path_strings)
        sys.path.insert(0, abs_python_dir)
        @debug "Added Python directory to sys.path: $abs_python_dir"
    end
    
    dataset_module = pyimport("dataset_constructor")
    
    # Create dataset based on type
    if dataset_type == "gaussian_mixture"
        n_components = get(kwargs, :n_components, 20)
        seed = get(kwargs, :seed, 42)
        
        dataset = dataset_module.GaussianMixtureDataset(
            n_samples=n_samples,
            input_dim=input_dim,
            output_dim=output_dim,
            n_components=n_components,
            seed=seed
        )
        
    elseif dataset_type == "nonlinear_manifold"
        manifold_dim = get(kwargs, :manifold_dim, 50)
        nonlinearity = get(kwargs, :nonlinearity, "polynomial")
        seed = get(kwargs, :seed, 42)
        
        dataset = dataset_module.NonlinearManifoldDataset(
            n_samples=n_samples,
            ambient_dim=input_dim,
            manifold_dim=manifold_dim,
            output_dim=output_dim,
            nonlinearity=nonlinearity,
            seed=seed
        )
        
    elseif dataset_type in ["mnist", "fashionmnist", "cifar10"]
        flatten = get(kwargs, :flatten, true)
        
        train_ds, test_ds, metadata = dataset_module.load_real_dataset(
            dataset_name=dataset_type,
            data_dir=joinpath(output_dir, "datasets", dataset_type),
            flatten=flatten
        )
        
        return pyconvert(Dict, metadata)
        
    else
        error("Unknown dataset type: $dataset_type")
    end
    
    # Save dataset
    save_dir = joinpath(output_dir, "datasets", dataset_type)
    dataset.save(save_dir)
    
    @info "Dataset saved to: $save_dir"
    
    # Load and return metadata
    metadata_file = joinpath(save_dir, "metadata.json")
    metadata = JSON3.read(read(metadata_file, String))
    
    return metadata
end


"""
    train_model(;
        dataset_type::String,
        model_config::String,
        output_dir::String="./output",
        run_hparam_search::Bool=false,
        kwargs...
    )

Train a neural network model using Python backend.

# Arguments
- `dataset_type::String`: Type of dataset
- `model_config::String`: Model configuration ("small_mlp", "medium_mlp", "large_mlp", "small_resmlp", "medium_resmlp", "large_resmlp")
- `output_dir::String`: Output directory
- `run_hparam_search::Bool`: Whether to run hyperparameter search
- `kwargs...`: Additional training arguments

# Returns
- Dictionary with training results

# Example
```julia
results = train_model(
    dataset_type="gaussian_mixture",
    model_config="medium_mlp",
    epochs=100,
    run_hparam_search=true
)
```
"""
function train_model(;
    dataset_type::String,
    model_config::String,
    output_dir::String="./output",
    run_hparam_search::Bool=false,
    kwargs...
)
    @info "Training model: $model_config on $dataset_type"
    
    # Build command line arguments
    cmd_args = [
        "--dataset_type", dataset_type,
        "--model_config", model_config,
        "--output_dir", output_dir,
    ]
    
    if run_hparam_search
        push!(cmd_args, "--run_hparam_search")
    end
    
    # Add optional arguments
    for (key, value) in kwargs
        key_str = "--" * string(key)
        push!(cmd_args, key_str)
        push!(cmd_args, string(value))
    end
    
    # Run Python script
    main_script = joinpath(PYTHON_DIR, "main.py")
    
    @info "Running training with arguments: $(join(cmd_args, " "))"
    
    try
        run(`python $main_script $cmd_args`)
    catch e
        @error "Training failed: $e"
        rethrow(e)
    end
    
    # Load and return results
    model_save_dir = joinpath(output_dir, "models", "$(dataset_type)_$(model_config)")
    seed = get(kwargs, :seed, 42)
    results_file = joinpath(model_save_dir, "results_seed$(seed).json")
    
    if isfile(results_file)
        results = JSON3.read(read(results_file, String))
        @info "Training complete! Best validation accuracy: $(results[:best_val_acc])"
        return results
    else
        @warn "Results file not found: $results_file"
        return nothing
    end
end


"""
    train_all_combinations(;
        dataset_types::Vector{String},
        model_configs::Vector{String},
        output_dir::String="./output",
        run_hparam_search::Bool=false,
        kwargs...
    )

Train models for all combinations of datasets and model configurations.
This is useful for generating a comprehensive set of trained models.

# Arguments
- `dataset_types::Vector{String}`: List of dataset types
- `model_configs::Vector{String}`: List of model configurations
- `output_dir::String`: Output directory
- `run_hparam_search::Bool`: Whether to run hyperparameter search
- `kwargs...`: Additional training arguments

# Returns
- Vector of dictionaries with training results

# Example
```julia
results = train_all_combinations(
    dataset_types=["gaussian_mixture", "mnist"],
    model_configs=["medium_mlp", "medium_resmlp"],
    epochs=100
)
```
"""
function train_all_combinations(;
    dataset_types::Vector{String},
    model_configs::Vector{String},
    output_dir::String="./output",
    run_hparam_search::Bool=false,
    kwargs...
)
    @info "Training $(length(dataset_types)) datasets × $(length(model_configs)) models = $(length(dataset_types) * length(model_configs)) combinations"
    
    all_results = []
    
    for dataset_type in dataset_types
        # Create dataset first
        @info "Creating dataset: $dataset_type"
        
        if dataset_type in ["mnist", "fashionmnist", "cifar10"]
            create_dataset(dataset_type=dataset_type, output_dir=output_dir)
        else
            # For synthetic datasets, create with specified parameters
            create_dataset(
                dataset_type=dataset_type,
                output_dir=output_dir;
                kwargs...
            )
        end
        
        for model_config in model_configs
            @info "Training combination: $dataset_type + $model_config"
            
            try
                results = train_model(
                    dataset_type=dataset_type,
                    model_config=model_config,
                    output_dir=output_dir,
                    run_hparam_search=run_hparam_search;
                    kwargs...
                )
                
                push!(all_results, (
                    dataset=dataset_type,
                    model=model_config,
                    results=results
                ))
            catch e
                @error "Failed to train $dataset_type + $model_config: $e"
            end
        end
    end
    
    @info "Training complete! Trained $(length(all_results)) models successfully."
    return all_results
end


"""
    load_trained_model(model_path::String)

Load a trained model from disk.

# Arguments
- `model_path::String`: Path to the saved model

# Returns
- Tuple of (model_state_dict, model_config, train_args, history)

# Example
```julia
state_dict, config, args, history = load_trained_model("./output/models/gaussian_mixture_medium_mlp/model_seed42.pt")
```
"""
function load_trained_model(model_path::String)
    @info "Loading model from: $model_path"
    
    # Import Python modules
    # Ensure Python directory is in path (use absolute path)
    abs_python_dir = abspath(PYTHON_DIR)
    sys = pyimport("sys")
    path_strings = [string(p) for p in sys.path]
    
    if !(abs_python_dir in path_strings)
        sys.path.insert(0, abs_python_dir)
        @debug "Added Python directory to sys.path: $abs_python_dir"
    end
    
    torch = pyimport("torch")
    
    # Load checkpoint
    checkpoint = torch.load(model_path)
    
    model_state = pyconvert(Dict, checkpoint["model_state_dict"])
    model_config = pyconvert(Dict, checkpoint["model_config"])
    train_args = pyconvert(Dict, checkpoint["train_args"])
    history = pyconvert(Dict, checkpoint["history"])
    
    @info "Model loaded successfully"
    @info "  Model type: $(model_config["model_type"])"
    @info "  Input dim: $(model_config["input_dim"])"
    @info "  Output dim: $(model_config["output_dim"])"
    
    return model_state, model_config, train_args, history
end
