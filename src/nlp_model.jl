"""
NLP Model implementation for NeuralNetworkNLPModel.
This file contains the core NLPModel structure and NLPModels callback implementations.
"""

using NLPModels
using MadNLP
using Flux
using ForwardDiff
using LinearAlgebra

# ============================================================================
# NLP Model with Neural Network
# ============================================================================

"""
    NeuralNetworkNLPModel <: AbstractNLPModel

NLP model integrating a neural network with custom objective and constraints.

This model uses automatic differentiation (ForwardDiff) to compute gradients
and Hessians numerically, which works with the smoothed ReLU activations.

# Fields
- `meta`: NLP model metadata
- `counters`: Function evaluation counters
- `neural_network`: Flux model representing the neural network
- `objective_func`: Objective function f(x, θ)
- `constraint_func`: Constraint function g(x)
- `x0`: Initial point
"""
mutable struct NeuralNetworkNLPModel <: AbstractNLPModel{Float64, Vector{Float64}}
    meta::NLPModelMeta{Float64, Vector{Float64}}
    counters::Counters
    neural_network::Any  # Flux model
    objective_func::AbstractObjectiveFunction
    constraint_func::Union{AbstractConstraintFunction, Nothing}
    x0::Vector{Float64}
end


"""
    NeuralNetworkNLPModel(
        model_path::String,
        objective_func::AbstractObjectiveFunction,
        constraint_func::Union{AbstractConstraintFunction, Nothing},
        x0::Vector{Float64}
    )

Create an NLP model with a neural network.
"""
function NeuralNetworkNLPModel(
    model_path::String,
    objective_func::AbstractObjectiveFunction,
    constraint_func::Union{AbstractConstraintFunction, Nothing},
    x0::Vector{Float64}
)
    # Load neural network
    nn_model, nn_config = load_pytorch_model_to_flux(model_path)
    
    # Determine dimensions
    n = length(x0)
    m = constraint_func === nothing ? 0 : num_constraints(constraint_func)
    
    # Set up bounds
    lvar = fill(-Inf, n)
    uvar = fill(Inf, n)
    
    # Set up constraint bounds (all g(x) ≤ 0)
    lcon = fill(-Inf, m)
    ucon = fill(0.0, m)
    
    # Create metadata
    meta = NLPModelMeta(
        n,
        x0=copy(x0),
        lvar=lvar,
        uvar=uvar,
        ncon=m,
        lcon=lcon,
        ucon=ucon,
        nnzj=m * n,  # Dense Jacobian
        nnzh=div(n * (n + 1), 2),  # Dense Hessian (lower triangle)
        minimize=true
    )
    
    counters = Counters()
    
    return NeuralNetworkNLPModel(
        meta,
        counters,
        nn_model,
        objective_func,
        constraint_func,
        copy(x0)
    )
end


# ============================================================================
# NLPModels Callback Implementations
# ============================================================================

"""
    NLPModels.obj(nlp::NeuralNetworkNLPModel, x::AbstractVector)

Evaluate objective function at x.
"""
function NLPModels.obj(nlp::NeuralNetworkNLPModel, x::AbstractVector)
    NLPModels.increment!(nlp, :neval_obj)
    
    # Forward pass through neural network
    nn_output = nlp.neural_network(x)
    
    # Evaluate objective function
    f = evaluate(nlp.objective_func, x, nn_output)
    
    return f
end


"""
    NLPModels.grad!(nlp::NeuralNetworkNLPModel, x::AbstractVector, g::AbstractVector)

Evaluate objective gradient at x using automatic differentiation.
"""
function NLPModels.grad!(nlp::NeuralNetworkNLPModel, x::AbstractVector, g::AbstractVector)
    NLPModels.increment!(nlp, :neval_grad)
    
    # Use ForwardDiff for gradient computation
    g .= ForwardDiff.gradient(x_val -> begin
        nn_output = nlp.neural_network(x_val)
        evaluate(nlp.objective_func, x_val, nn_output)
    end, x)
    
    return g
end


"""
    NLPModels.cons!(nlp::NeuralNetworkNLPModel, x::AbstractVector, c::AbstractVector)

Evaluate constraint functions at x.
"""
function NLPModels.cons!(nlp::NeuralNetworkNLPModel, x::AbstractVector, c::AbstractVector)
    NLPModels.increment!(nlp, :neval_cons)
    
    if nlp.constraint_func === nothing
        # No constraints
        return c
    end
    
    c_vals = evaluate(nlp.constraint_func, x)
    c .= c_vals
    
    return c
end


"""
    NLPModels.jac_structure!(nlp::NeuralNetworkNLPModel, rows, cols)

Define the sparsity structure of the Jacobian.
"""
function NLPModels.jac_structure!(
    nlp::NeuralNetworkNLPModel,
    rows::AbstractVector{<:Integer},
    cols::AbstractVector{<:Integer}
)
    n = nlp.meta.nvar
    m = nlp.meta.ncon
    
    idx = 1
    for i in 1:m
        for j in 1:n
            rows[idx] = i
            cols[idx] = j
            idx += 1
        end
    end
    
    return rows, cols
end


"""
    NLPModels.jac_coord!(nlp::NeuralNetworkNLPModel, x, vals)

Evaluate Jacobian values at x using automatic differentiation.
"""
function NLPModels.jac_coord!(
    nlp::NeuralNetworkNLPModel,
    x::AbstractVector,
    vals::AbstractVector
)
    NLPModels.increment!(nlp, :neval_jac)
    
    if nlp.meta.ncon == 0
        return vals
    end
    
    # Compute Jacobian using ForwardDiff
    jac_func = x_val -> begin
        # Use eltype(x_val) to support automatic differentiation
        T = eltype(x_val)
        c = zeros(T, nlp.meta.ncon)
        c_vals = evaluate(nlp.constraint_func, x_val)
        c .= c_vals
        return c
    end
    
    J = ForwardDiff.jacobian(jac_func, x)
    
    # Fill vals in row-major order
    idx = 1
    for i in 1:nlp.meta.ncon
        for j in 1:nlp.meta.nvar
            vals[idx] = J[i, j]
            idx += 1
        end
    end
    
    return vals
end


"""
    NLPModels.hess_structure!(nlp::NeuralNetworkNLPModel, rows, cols)

Define the sparsity structure of the Hessian (lower triangle).
"""
function NLPModels.hess_structure!(
    nlp::NeuralNetworkNLPModel,
    rows::AbstractVector{<:Integer},
    cols::AbstractVector{<:Integer}
)
    n = nlp.meta.nvar
    
    idx = 1
    for j in 1:n
        for i in j:n  # Lower triangle
            rows[idx] = i
            cols[idx] = j
            idx += 1
        end
    end
    
    return rows, cols
end


"""
    NLPModels.hess_coord!(nlp, x, vals, y; obj_weight=1.0)

Evaluate Hessian of the Lagrangian at x.

Lagrangian: L(x, y) = obj_weight * f(x) + y' * c(x)
"""
function NLPModels.hess_coord!(
    nlp::NeuralNetworkNLPModel,
    x::AbstractVector,
    vals::AbstractVector,
    y::AbstractVector=zeros(nlp.meta.ncon);
    obj_weight=1.0
)
    NLPModels.increment!(nlp, :neval_hess)
    
    # Compute Hessian of objective
    obj_func = x_val -> begin
        nn_output = nlp.neural_network(x_val)
        evaluate(nlp.objective_func, x_val, nn_output)
    end
    hess_obj = ForwardDiff.hessian(obj_func, x)
    
    # Initialize Lagrangian Hessian
    hess_lag = obj_weight .* hess_obj
    
    # Add constraint Hessians
    if nlp.meta.ncon > 0 && any(y .!= 0)
        for i in 1:nlp.meta.ncon
            if y[i] != 0
                # Compute Hessian of constraint i
                cons_i_func = x_val -> begin
                    c_vals = evaluate(nlp.constraint_func, x_val)
                    return c_vals[i]
                end
                hess_ci = ForwardDiff.hessian(cons_i_func, x)
                hess_lag .+= y[i] .* hess_ci
            end
        end
    end
    
    # Extract lower triangle
    idx = 1
    n = nlp.meta.nvar
    for j in 1:n
        for i in j:n
            vals[idx] = hess_lag[i, j]
            idx += 1
        end
    end
    
    return vals
end


# ============================================================================
# High-Level Solving Interface
# ============================================================================

"""
    solve_nlp(nlp; kwargs...)

Solve the NLP using MadNLP.
"""
function solve_nlp(
    nlp::NeuralNetworkNLPModel;
    max_iter=1000,
    tol=1e-6,
    print_level=MadNLP.INFO,
    linear_solver=nothing,
    kwargs...
)
    @info "Solving NLP with MadNLP..."
    @info "  Variables: $(nlp.meta.nvar)"
    @info "  Constraints: $(nlp.meta.ncon)"
    
    # Build solver options  
    options = Dict{Symbol, Any}(
        :max_iter => max_iter,
        :tol => tol,
        :print_level => print_level
    )
    
    # Add linear solver if specified
    if linear_solver !== nothing
        options[:linear_solver] = linear_solver
    end
    
    # Merge additional options
    merge!(options, Dict(kwargs))
    
    # Create and solve
    solver = MadNLPSolver(nlp; options...)
    result = MadNLP.solve!(solver)
    
    # Extract solution information
    status = result.status
    solution = copy(result.solution)
    objective = result.objective
    iter_count = result.iter
    
    @info "Solver finished with status: $status"
    @info "  Objective: $objective"
    @info "  Iterations: $iter_count"
    
    return Dict(
        :status => status,
        :solution => solution,
        :objective => objective,
        :iter_count => iter_count,
        :nlp_model => nlp
    )
end


"""
    create_simple_nlp(model_path, target, x0; kwargs...)

Create a simple NLP model with standard objective and constraints.
"""
function create_simple_nlp(
    model_path::String,
    target::Vector{Float64},
    x0::Vector{Float64};
    bounds=nothing,
    radius=nothing,
    regularization_weight=0.0
)
    n = length(x0)
    
    # Create objective
    if regularization_weight > 0
        obj = CompositeObjective(
            NeuralNetworkObjective(target),
            QuadraticRegularization(x0, weight=regularization_weight)
        )
    else
        obj = NeuralNetworkObjective(target)
    end
    
    # Create constraints
    cons = nothing
    if bounds !== nothing && radius !== nothing
        # Both types of constraints
        if isa(bounds, Tuple) && length(bounds) == 2
            x_min = fill(Float64(bounds[1]), n)
            x_max = fill(Float64(bounds[2]), n)
        else
            x_min, x_max = bounds
        end
        cons = CompositeConstraint(
            BoxConstraints(x_min, x_max),
            SphericalConstraint(x0, Float64(radius))
        )
    elseif bounds !== nothing
        # Only box constraints
        if isa(bounds, Tuple) && length(bounds) == 2
            x_min = fill(Float64(bounds[1]), n)
            x_max = fill(Float64(bounds[2]), n)
        else
            x_min, x_max = bounds
        end
        cons = BoxConstraints(x_min, x_max)
    elseif radius !== nothing
        # Only spherical constraint
        cons = SphericalConstraint(x0, Float64(radius))
    end
    
    return NeuralNetworkNLPModel(model_path, obj, cons, x0)
end

