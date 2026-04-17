"""
Julia interface to the proposal-aligned Python tooling.

The repository now focuses on two workflows only:
- Darcy/FNO data generation + surrogate training
- Pareto surrogate data generation + training
"""

using PythonCall

const PYTHON_DIR = joinpath(@__DIR__, "..", "python")

function _ensure_python_path!()
    abs_python_dir = abspath(PYTHON_DIR)
    sys = pyimport("sys")
    path_strings = [string(p) for p in sys.path]
    if !(abs_python_dir in path_strings)
        sys.path.insert(0, abs_python_dir)
        @info "Added Python directory to sys.path: $abs_python_dir"
    end
    return abs_python_dir
end

function _python_executable()
    return get(ENV, "JULIA_PYTHONCALL_EXE", "python")
end

"""
    setup_python_env(; force_reinstall=false)

Configure PythonCall to see the repository's Python package directory.
If `force_reinstall=true`, reinstall `python/requirements.txt` into the
currently selected Python executable.
"""
function setup_python_env(; force_reinstall::Bool=false)
    @info "Setting up Python environment for MadNLP4NN..."
    _ensure_python_path!()
    requirements_file = joinpath(PYTHON_DIR, "requirements.txt")

    if force_reinstall
        python_exe = _python_executable()
        @info "Reinstalling Python packages with $python_exe"
        run(`$python_exe -m pip install -r $requirements_file`)
    else
        @info "Python path configured. Requirements file:"
        @info "  $requirements_file"
    end

    return nothing
end

"""
    train_darcy_fno(; kwargs...)

Generate Darcy data (train/test) and train an FNO surrogate via
`python/darcy_data.py`.

Returns the checkpoint path of the trained `.npz` FNO model.
"""
function train_darcy_fno(;
    data_dir::String="./output/darcy",
    grid_n::Int=64,
    n_samples::Int=1000,
    seed::Int=42,
    epochs::Int=200,
    d_v::Int=32,
    n_layers::Int=4,
    k_max::Int=min(16, grid_n ÷ 4),
    lr::Float64=1e-3,
    batch_size::Int=32,
    device::String="cpu",
)
    setup_python_env()
    darcy_module = pyimport("darcy_data")

    darcy_module.generate_darcy_dataset(
        n_samples=n_samples,
        grid_n=grid_n,
        seed=seed,
        output_dir=data_dir,
        split="train",
    )
    darcy_module.generate_darcy_dataset(
        n_samples=max(1, n_samples ÷ 5),
        grid_n=grid_n,
        seed=seed + 1,
        output_dir=data_dir,
        split="test",
    )

    return pyconvert(String, darcy_module.train_fno(
        data_dir=data_dir,
        grid_n=grid_n,
        d_v=d_v,
        n_layers=n_layers,
        k_max=k_max,
        lr=lr,
        epochs=epochs,
        batch_size=batch_size,
        seed=seed,
        output_dir=joinpath(data_dir, "models"),
        device=device,
    ))
end

"""
    train_pareto_surrogates(; kwargs...)

Generate Pareto training data and train the `f1`, `f2`, and `h` surrogate
networks via `python/pareto_data.py`.

Returns a Dict with keys `\"f1_path\"`, `\"f2_path\"`, and `\"h_path\"`.
"""
function train_pareto_surrogates(;
    n::Int=20,
    n_samples::Int=5000,
    seed::Int=42,
    data_dir::String="./output/pareto",
    output_dir::String="./output/models/pareto",
    epochs::Int=200,
    hidden::Int=64,
    device::String="cpu",
)
    setup_python_env()
    pareto_module = pyimport("pareto_data")
    return pyconvert(Dict, pareto_module.train_all_surrogates(
        n=n,
        n_samples=n_samples,
        seed=seed,
        data_dir=data_dir,
        output_dir=output_dir,
        epochs=epochs,
        hidden=hidden,
        device=device,
    ))
end

"""
    load_trained_model(model_path::String)

Load a PyTorch `.pt` checkpoint and return:
`(model_state_dict, model_config, train_args, history)`.

This helper is only for Torch checkpoints. Proposal-era FNO checkpoints use
`.npz` and are loaded through the JAX model loader instead.
"""
function load_trained_model(model_path::String)
    endswith(lowercase(model_path), ".npz") &&
        error("load_trained_model only supports PyTorch .pt checkpoints, not .npz FNO checkpoints.")

    @info "Loading model from: $model_path"
    _ensure_python_path!()
    torch = pyimport("torch")
    checkpoint = torch.load(model_path, map_location="cpu")

    model_state = pyconvert(Dict, checkpoint["model_state_dict"])
    model_config = pyconvert(Dict, checkpoint["model_config"])
    train_args = pyconvert(Dict, get(checkpoint, "train_args", Dict()))
    history = pyconvert(Dict, get(checkpoint, "history", Dict()))

    return model_state, model_config, train_args, history
end
