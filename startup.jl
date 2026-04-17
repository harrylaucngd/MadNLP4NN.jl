# Startup script for MadNLP4NN.jl
# This ensures the correct Python environment is used

# Set environment variables BEFORE loading any packages
# Prefer an explicitly provided Python, otherwise use the local `madnlp4nn` conda env.
ENV["JULIA_PYTHONCALL_EXE"] = get(
    ENV,
    "JULIA_PYTHONCALL_EXE",
    "/opt/anaconda3/envs/madnlp4nn/bin/python",
)
ENV["JULIA_CONDAPKG_BACKEND"] = "Null"

# Activate the project
using Pkg
Pkg.activate(@__DIR__)

# Load the package
println("Loading MadNLP4NN with madnlp4nn Python environment...")
using MadNLP4NN

println("\n✓ MadNLP4NN loaded successfully!")
