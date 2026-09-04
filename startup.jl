# Startup script for MadNLP4NN.jl
# This ensures the correct Python environment is used

# Set environment variables BEFORE loading any packages
# Prefer an explicitly provided Python, otherwise use the active environment.
# This file must stay portable: never bake a developer-specific conda path into it.
if !haskey(ENV, "JULIA_PYTHONCALL_EXE")
    python_executable = something(Sys.which("python3"), Sys.which("python"), nothing)
    python_executable === nothing && error(
        "No Python executable found. Activate the project environment or set " *
        "JULIA_PYTHONCALL_EXE before loading MadNLP4NN.",
    )
    ENV["JULIA_PYTHONCALL_EXE"] = python_executable
end
ENV["JULIA_CONDAPKG_BACKEND"] = get(
    ENV,
    "JULIA_CONDAPKG_BACKEND",
    "Null",
)

# Activate the project
using Pkg
Pkg.activate(@__DIR__)

# Load the package
println("Loading MadNLP4NN with madnlp4nn Python environment...")
using MadNLP4NN

println("\n✓ MadNLP4NN loaded successfully!")
