"""
NLP interface for optimization using neural networks and MadNLP.

This module provides structures and functions to formulate optimization problems
involving neural networks as NLPs and solve them using MadNLP.

Problem Formulation:
    min_x  f(x, θ)
    s.t.   g(x) ≤ 0

where:
- x: decision variables
- θ: neural network parameters
- f(x, θ): objective function using neural network output and x-related expressions
- g(x): constraint functions (e.g., bounds on x)

The formulation uses NLPModels callbacks to provide numerical values only,
not symbolic representations.
"""

using NLPModels
using MadNLP
using Flux
using LinearAlgebra
using ForwardDiff

# ============================================================================
# Smoothed Activation Functions
# ============================================================================

"""
    smooth_relu(x; α=1e-3)

Smoothed ReLU activation for differentiability.

Uses the approximation: smooth_relu(x) ≈ log(1 + exp(x/α)) * α

This ensures the function is smooth everywhere, which is required for
second-order optimization methods like those used in MadNLP.

For practical purposes with α small, this behaves like ReLU but is
differentiable at x=0.

# Arguments
- `x`: Input value
- `α`: Smoothing parameter (smaller = closer to ReLU, but less smooth)
"""
function smooth_relu(x::T; α=1e-3) where T<:Real
    # Numerically stable implementation
    # Use the same type as input for numerical stability
    # Use softplus: log(1 + exp(x/α)) * α
    alpha = convert(T, α)
    
    # For numerical stability:
    # if x/α > 20, softplus(x/α) ≈ x/α, so result ≈ x
    # Use ifelse to ensure differentiability
    scaled = x / alpha
    return ifelse(scaled > 20, x, alpha * log(one(T) + exp(scaled)))
end

# Array version with element-wise application
smooth_relu(x::AbstractArray; α=1e-3) = smooth_relu.(x; α=α)


# ============================================================================
# Neural Network Model Loading and Conversion
# ============================================================================

"""
    load_pytorch_model_to_flux(model_path::String)

Load a PyTorch model and convert it to Flux.jl format.

# Arguments
- `model_path::String`: Path to the saved PyTorch model (.pt file)

# Returns
- `model`: Flux Chain representing the neural network
- `config`: Dictionary with model configuration

# Example
```julia
model, config = load_pytorch_model_to_flux("output/models/dataset/model.pt")
```
"""
function load_pytorch_model_to_flux(model_path::String)
    # Load PyTorch model using our existing function
    state_dict, model_config, train_args, history = load_trained_model(model_path)
    
    model_type = model_config["model_type"]
    
    # Handle both Python class names and lowercase versions
    if model_type in ["mlp", "MLPClassifier"]
        model = build_mlp_from_state_dict(state_dict, model_config)
    elseif model_type in ["resmlp", "ResMLPClassifier"]
        model = build_resmlp_from_state_dict(state_dict, model_config)
    else
        error("Unknown model type: $model_type")
    end
    
    return model, model_config
end


"""
    build_mlp_from_state_dict(state_dict, config)

Build a Flux MLP from PyTorch state dict.

Note: ReLU activations are handled with a smoothed version to ensure differentiability.
"""
function build_mlp_from_state_dict(state_dict, config)
    layers = []
    
    # Extract layer indices from keys (handle both "layers.X" and "network.X" formats)
    all_keys = collect(keys(state_dict))
    layer_indices = Int[]
    for key in all_keys
        # Match either "layers.X.weight" or "network.X.weight"
        m = match(r"^(?:layers|network)\.(\d+)\.weight$", key)
        if m !== nothing
            push!(layer_indices, parse(Int, m.captures[1]))
        end
    end
    sort!(layer_indices)
    
    if isempty(layer_indices)
        error("No layers found in state_dict. Available keys: $(collect(keys(state_dict)))")
    end
    
    # Determine the prefix (layers or network)
    prefix = occursin("network.", first(all_keys)) ? "network" : "layers"
    
    # Build layers
    for layer_idx in layer_indices
        weight_key = "$prefix.$layer_idx.weight"
        bias_key = "$prefix.$layer_idx.bias"
        
        # Convert PyTorch tensors to Julia arrays
        # Both PyTorch and Flux use (out, in) format, so no transpose needed
        W = Float64.(pyconvert(Array, state_dict[weight_key]))
        b = Float64.(vec(pyconvert(Array, state_dict[bias_key])))
        
        # Add dense layer
        push!(layers, Dense(W, b))
        
        # Add activation (ReLU for hidden layers, except last)
        if layer_idx != layer_indices[end]
            push!(layers, x -> smooth_relu.(x))
        end
    end
    
    return Chain(layers...)
end


"""
    build_resmlp_from_state_dict(state_dict, config)

Build a Flux Residual MLP from PyTorch state dict.
"""
function build_resmlp_from_state_dict(state_dict, config)
    # Initial projection
    # No transpose needed - both PyTorch and Flux use (out, in)
    W_init = pyconvert(Array, state_dict["init_proj.weight"])
    b_init = pyconvert(Array, state_dict["init_proj.bias"])
    W_init = Float64.(W_init)
    b_init = Float64.(vec(b_init))
    
    init_proj = Dense(W_init, b_init)
    
    # Build residual blocks
    n_blocks = config["n_blocks"]
    blocks = []
    
    for i in 0:(n_blocks-1)
        # Block layers - no transpose needed
        W1 = pyconvert(Array, state_dict["blocks.$i.fc1.weight"])
        b1 = pyconvert(Array, state_dict["blocks.$i.fc1.bias"])
        W2 = pyconvert(Array, state_dict["blocks.$i.fc2.weight"])
        b2 = pyconvert(Array, state_dict["blocks.$i.fc2.bias"])
        
        W1 = Float64.(W1)
        b1 = Float64.(vec(b1))
        W2 = Float64.(W2)
        b2 = Float64.(vec(b2))
        
        # Create residual block as a function
        block = function(x)
            identity = x
            out = Dense(W1, b1)(x)
            out = smooth_relu.(out)
            out = Dense(W2, b2)(out)
            return out .+ identity
        end
        
        push!(blocks, block)
    end
    
    # Output layer - no transpose needed
    W_out = pyconvert(Array, state_dict["output_layer.weight"])
    b_out = pyconvert(Array, state_dict["output_layer.bias"])
    W_out = Float64.(W_out)
    b_out = Float64.(vec(b_out))
    
    output_layer = Dense(W_out, b_out)
    
    # Combine into a single model
    model = function(x)
        out = init_proj(x)
        out = smooth_relu.(out)
        for block in blocks
            out = block(out)
            out = smooth_relu.(out)
        end
        out = output_layer(out)
        return out
    end
    
    return model
end


# ============================================================================
# Objective and Constraint Function Modules
# ============================================================================

"""
    AbstractObjectiveFunction

Abstract type for objective functions f(x, θ).

Subtypes should implement:
- `evaluate(obj, x, nn_output)`: Compute objective value
"""
abstract type AbstractObjectiveFunction end

"""
    AbstractConstraintFunction

Abstract type for constraint functions g(x).

Subtypes should implement:
- `evaluate(cons, x)`: Compute constraint values (should be ≤ 0)
- `num_constraints(cons)`: Number of constraints
"""
abstract type AbstractConstraintFunction end


"""
    NeuralNetworkObjective

Objective function based on neural network output.

Example: Minimize squared distance to a target:
    f(x) = ||nn(x) - target||²
"""
struct NeuralNetworkObjective <: AbstractObjectiveFunction
    target::Vector{Float64}
    weight::Float64
    
    NeuralNetworkObjective(target; weight=1.0) = new(target, weight)
end

function evaluate(obj::NeuralNetworkObjective, x::AbstractVector, nn_output::AbstractVector)
    diff = nn_output .- obj.target
    return obj.weight * dot(diff, diff)
end


"""
    QuadraticRegularization

Regularization term: f(x) = weight * ||x - center||²
"""
struct QuadraticRegularization <: AbstractObjectiveFunction
    center::Vector{Float64}
    weight::Float64
    
    QuadraticRegularization(center; weight=0.01) = new(center, weight)
end

function evaluate(obj::QuadraticRegularization, x::AbstractVector, nn_output::AbstractVector)
    diff = x .- obj.center
    return obj.weight * dot(diff, diff)
end


"""
    CompositeObjective

Composite objective function combining multiple objectives.

    f(x) = sum(weight_i * f_i(x))
"""
struct CompositeObjective <: AbstractObjectiveFunction
    objectives::Vector{AbstractObjectiveFunction}
    
    CompositeObjective(objs...) = new([objs...])
end

function evaluate(obj::CompositeObjective, x::AbstractVector, nn_output::AbstractVector)
    # Use eltype(x) to support automatic differentiation
    T = promote_type(eltype(x), eltype(nn_output))
    total = zero(T)
    for sub_obj in obj.objectives
        total += evaluate(sub_obj, x, nn_output)
    end
    return total
end


"""
    BoxConstraints

Box constraints: x_min ≤ x ≤ x_max

Formulated as: x - x_max ≤ 0 and x_min - x ≤ 0
"""
struct BoxConstraints <: AbstractConstraintFunction
    x_min::Vector{Float64}
    x_max::Vector{Float64}
    
    function BoxConstraints(x_min, x_max)
        @assert length(x_min) == length(x_max)
        @assert all(x_min .<= x_max)
        new(x_min, x_max)
    end
end

function evaluate(cons::BoxConstraints, x::AbstractVector)
    n = length(x)
    # Use eltype(x) to support automatic differentiation (ForwardDiff.Dual)
    T = eltype(x)
    c = zeros(T, 2n)
    c[1:n] .= x .- cons.x_max  # x ≤ x_max
    c[n+1:2n] .= cons.x_min .- x  # x_min ≤ x
    return c
end

num_constraints(cons::BoxConstraints) = 2 * length(cons.x_min)


"""
    SphericalConstraint

Spherical constraint: ||x - center||² ≤ radius²

Formulated as: ||x - center||² - radius² ≤ 0
"""
struct SphericalConstraint <: AbstractConstraintFunction
    center::Vector{Float64}
    radius::Float64
    
    SphericalConstraint(center, radius) = new(center, radius)
end

function evaluate(cons::SphericalConstraint, x::AbstractVector)
    diff = x .- cons.center
    return [dot(diff, diff) - cons.radius^2]
end

num_constraints(cons::SphericalConstraint) = 1


"""
    CompositeConstraint

Composite constraint combining multiple constraints.
"""
struct CompositeConstraint <: AbstractConstraintFunction
    constraints::Vector{AbstractConstraintFunction}
    
    CompositeConstraint(cons...) = new([cons...])
end

function evaluate(cons::CompositeConstraint, x::AbstractVector)
    # Use eltype(x) to support automatic differentiation
    T = eltype(x)
    c = T[]
    for sub_cons in cons.constraints
        append!(c, evaluate(sub_cons, x))
    end
    return c
end

function num_constraints(cons::CompositeConstraint)
    return sum(num_constraints(c) for c in cons.constraints)
end


# Continued in next part...

