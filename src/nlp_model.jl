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

This model supports two backends for neural network evaluation and derivatives:
1. Julia/Flux backend: Uses Flux for inference and ForwardDiff for derivatives (default)
2. Python/JAX backend: Uses JAX for inference and JAX AD for derivatives (optional)

Both backends use the same MadNLP solver for optimization.

# Fields
- `meta`: NLP model metadata
- `counters`: Function evaluation counters
- `neural_network`: Flux model (only used if use_python=false)
- `objective_func`: Objective function f(x, θ)
- `constraint_func`: Constraint function g(x)
- `x0`: Initial point
- `use_python`: If true, use Python/JAX backend; if false, use Flux/ForwardDiff
- `python_evaluator`: Python JAX evaluator (only used if use_python=true)
"""
mutable struct NeuralNetworkNLPModel <: AbstractNLPModel{Float64, Vector{Float64}}
    meta::NLPModelMeta{Float64, Vector{Float64}}
    counters::Counters
    neural_network::Any  # Flux model (or nothing if using Python)
    objective_func::AbstractObjectiveFunction
    constraint_func::Union{AbstractConstraintFunction, Nothing}
    x0::Vector{Float64}
    use_python::Bool  # Whether to use Python/JAX backend
    python_evaluator::Union{Py, Nothing}  # Python evaluator (or nothing if using Flux)
end


"""
    NeuralNetworkNLPModel(
        model_path::String,
        objective_func::AbstractObjectiveFunction,
        constraint_func::Union{AbstractConstraintFunction, Nothing},
        x0::Vector{Float64};
        use_python::Bool=false
    )

Create an NLP model with a neural network.

# Arguments
- `model_path`: Path to trained PyTorch model
- `objective_func`: Objective function
- `constraint_func`: Constraint function (or nothing)
- `x0`: Initial point
- `use_python`: If true, use Python/JAX backend; if false, use Flux/ForwardDiff (default: false)
"""
function NeuralNetworkNLPModel(
    model_path::String,
    objective_func::AbstractObjectiveFunction,
    constraint_func::Union{AbstractConstraintFunction, Nothing},
    x0::Vector{Float64};
    use_python::Bool=false
)
    if use_python
        @info "Creating NLP model with Python/JAX backend"
        return _create_python_nlp_model(model_path, objective_func, constraint_func, x0)
    else
        @info "Creating NLP model with Julia/Flux backend"
        return _create_flux_nlp_model(model_path, objective_func, constraint_func, x0)
    end
end


"""
    _create_flux_nlp_model(...)

Internal function to create NLP model with Flux/ForwardDiff backend.
"""
function _create_flux_nlp_model(
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
        copy(x0),
        false,    # use_python = false
        nothing   # python_evaluator = nothing
    )
end


"""
    _create_python_nlp_model(...)

Internal function to create NLP model with Python/JAX backend.
"""
function _create_python_nlp_model(
    model_path::String,
    objective_func::AbstractObjectiveFunction,
    constraint_func::Union{AbstractConstraintFunction, Nothing},
    x0::Vector{Float64}
)
    # Import Python JAX evaluator module
    jax_evaluator_module = pyimport("jax_nn_evaluator")
    
    # Extract parameters for Python evaluator
    # Get target from objective (assuming NeuralNetworkObjective)
    target = if objective_func isa NeuralNetworkObjective
        objective_func.target
    elseif objective_func isa CompositeObjective
        # Find NeuralNetworkObjective in composite
        nn_obj = findfirst(obj -> obj isa NeuralNetworkObjective, objective_func.objectives)
        if nn_obj !== nothing
            objective_func.objectives[nn_obj].target
        else
            error("Could not find NeuralNetworkObjective in CompositeObjective")
        end
    else
        error("Unsupported objective type for Python backend: $(typeof(objective_func))")
    end
    
    # Get regularization weight if present
    regularization_weight = if objective_func isa CompositeObjective
        reg_obj = findfirst(obj -> obj isa QuadraticRegularization, objective_func.objectives)
        if reg_obj !== nothing
            objective_func.objectives[reg_obj].weight
        else
            0.0
        end
    else
        0.0
    end
    
    # Extract bounds and radius from constraints
    bounds = nothing
    radius = nothing
    
    if constraint_func isa BoxConstraints
        bounds = (constraint_func.x_min, constraint_func.x_max)
    elseif constraint_func isa SphericalConstraint
        radius = constraint_func.radius
    elseif constraint_func isa CompositeConstraint
        # Extract from composite constraint
        for c in constraint_func.constraints
            if c isa BoxConstraints
                bounds = (c.x_min, c.x_max)
            elseif c isa SphericalConstraint
                radius = c.radius
            end
        end
    end
    
    # Create Python evaluator
    py_eval = jax_evaluator_module.create_evaluator(
        model_path,
        target,
        x0;
        objective_weight=1.0,
        regularization_weight=regularization_weight,
        bounds=bounds,
        radius=radius
    )
    
    # Get dimensions from evaluator
    n = pyconvert(Int, py_eval.n)
    m = pyconvert(Int, py_eval.m)
    
    # Set up bounds
    lvar = fill(-Inf, n)
    uvar = fill(Inf, n)
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
        nnzj=m * n,
        nnzh=div(n * (n + 1), 2),
        minimize=true
    )
    
    counters = Counters()
    
    return NeuralNetworkNLPModel(
        meta,
        counters,
        nothing,  # neural_network = nothing (using Python)
        objective_func,
        constraint_func,
        copy(x0),
        true,     # use_python = true
        py_eval   # python_evaluator
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
    
    if nlp.use_python
        # Use Python/JAX backend
        f = pyconvert(Float64, nlp.python_evaluator.evaluate_obj(x))
        return f
    else
        # Use Flux backend
        nn_output = nlp.neural_network(x)
        f = evaluate(nlp.objective_func, x, nn_output)
        return f
    end
end


"""
    NLPModels.grad!(nlp::NeuralNetworkNLPModel, x::AbstractVector, g::AbstractVector)

Evaluate objective gradient at x using automatic differentiation.
"""
function NLPModels.grad!(nlp::NeuralNetworkNLPModel, x::AbstractVector, g::AbstractVector)
    NLPModels.increment!(nlp, :neval_grad)
    
    if nlp.use_python
        # Use Python/JAX backend
        g_py = nlp.python_evaluator.evaluate_grad(x)
        g .= pyconvert(Vector{Float64}, g_py)
        
        # Check for invalid values
        if any(isnan.(g)) || any(isinf.(g))
            @warn "Gradient contains NaN or Inf values! This will cause solver failure."
            @warn "  NaN count: $(sum(isnan.(g)))"
            @warn "  Inf count: $(sum(isinf.(g)))"
        end
        
        return g
    else
        # Use ForwardDiff for gradient computation
        g .= ForwardDiff.gradient(x_val -> begin
            nn_output = nlp.neural_network(x_val)
            evaluate(nlp.objective_func, x_val, nn_output)
        end, x)
        return g
    end
end


"""
    NLPModels.cons!(nlp::NeuralNetworkNLPModel, x::AbstractVector, c::AbstractVector)

Evaluate constraint functions at x.
"""
function NLPModels.cons!(nlp::NeuralNetworkNLPModel, x::AbstractVector, c::AbstractVector)
    NLPModels.increment!(nlp, :neval_cons)
    
    if nlp.use_python
        # Use Python/JAX backend
        if nlp.meta.ncon > 0
            c_py = nlp.python_evaluator.evaluate_cons(x)
            c .= pyconvert(Vector{Float64}, c_py)
        end
        return c
    else
        # Use Flux backend
        if nlp.constraint_func === nothing
            return c
        end
        c_vals = evaluate(nlp.constraint_func, x)
        c .= c_vals
        return c
    end
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
    
    if nlp.use_python
        # Use Python/JAX backend
        J_py = nlp.python_evaluator.evaluate_jac(x)
        J = pyconvert(Matrix{Float64}, J_py)
        
        # Fill vals in row-major order
        idx = 1
        for i in 1:nlp.meta.ncon
            for j in 1:nlp.meta.nvar
                vals[idx] = J[i, j]
                idx += 1
            end
        end
        return vals
    else
        # Use ForwardDiff
        jac_func = x_val -> begin
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
    NLPModels.hess_coord!(nlp, x, y, vals; obj_weight=1.0)

Evaluate Hessian of the Lagrangian at x.

Lagrangian: L(x, y) = obj_weight * f(x) + y' * c(x)

Note: The correct NLPModels signature has y (Lagrange multipliers) before vals (Hessian values).
"""
function NLPModels.hess_coord!(
    nlp::NeuralNetworkNLPModel,
    x::AbstractVector,
    y::AbstractVector,
    vals::AbstractVector;
    obj_weight=1.0
)
    NLPModels.increment!(nlp, :neval_hess)
    
    if nlp.use_python
        # Use Python/JAX backend
        H_py = nlp.python_evaluator.evaluate_hess(x, y, obj_weight)
        H = pyconvert(Matrix{Float64}, H_py)
        
        # Extract lower triangle
        idx = 1
        n = nlp.meta.nvar
        for j in 1:n
            for i in j:n
                vals[idx] = H[i, j]
                idx += 1
            end
        end
        return vals
    else
        # Use ForwardDiff
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
    # Extract only primal variables (not dual variables)
    # MadNLP's result.solution contains [x; y] where x are primal and y are dual
    solution = copy(result.solution[1:nlp.meta.nvar])
    objective = result.objective
    iter_count = result.iter
    
    @info "Solver finished with status: $status"
    @info "  Objective: $objective"
    @info "  Iterations: $iter_count"
    
    return Dict(
        :status => status,
        :solution => solution,  # Only primal variables
        :objective => objective,
        :iter_count => iter_count,
        :nlp_model => nlp
    )
end


"""
    create_simple_nlp(model_path, target, x0; kwargs...)

Create a simple NLP model with standard objective and constraints.

# Arguments
- `model_path`: Path to trained PyTorch model
- `target`: Target output for neural network objective
- `x0`: Initial point
- `bounds`: Box constraints (tuple or arrays)
- `radius`: Spherical constraint radius
- `regularization_weight`: Weight for quadratic regularization
- `use_python`: If true, use Python/JAX backend; if false, use Flux/ForwardDiff (default: false)
"""
function create_simple_nlp(
    model_path::String,
    target::Vector{Float64},
    x0::Vector{Float64};
    bounds=nothing,
    radius=nothing,
    regularization_weight=0.0,
    use_python::Bool=false
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
    
    return NeuralNetworkNLPModel(model_path, obj, cons, x0; use_python=use_python)
end

