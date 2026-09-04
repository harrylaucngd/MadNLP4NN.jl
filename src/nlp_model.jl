"""
NLP Model implementation for NeuralNetworkNLPModel.
This file contains the core NLPModel structure and NLPModels callback implementations.
"""

using NLPModels
using MadNLP
using Flux
using ForwardDiff
using LinearAlgebra
using Zygote
using SparseArrays

# ============================================================================
# Timing Statistics
# ============================================================================

"""
    TimingStats

Mutable struct to track timing statistics during optimization with detailed breakdown by operation type.

# Fields
## Categorized timing
- `obj_time`: Time spent evaluating objective function
- `grad_time`: Time spent evaluating gradients
- `cons_time`: Time spent evaluating constraints
- `jac_time`: Time spent evaluating Jacobian matrices
- `hess_time`: Time spent evaluating Hessian matrices

## Categorized counts
- `obj_count`: Number of objective evaluations
- `grad_count`: Number of gradient evaluations
- `cons_count`: Number of constraint evaluations
- `jac_count`: Number of Jacobian evaluations
- `hess_count`: Number of Hessian evaluations

## Total statistics (for backward compatibility)
- `total_eval_time`: Total time spent in all evaluation callbacks
- `eval_count`: Total number of evaluation calls
- `per_iteration_times`: Vector of eval times per iteration (approximated by tracking evaluation calls)
- `iteration_start_time`: Time when current iteration started
"""
mutable struct TimingStats
    # Categorized timing
    obj_time::Float64
    grad_time::Float64
    cons_time::Float64
    jac_time::Float64
    hess_time::Float64
    
    # Categorized counts
    obj_count::Int
    grad_count::Int
    cons_count::Int
    jac_count::Int
    hess_count::Int
    
    # Total statistics (backward compatibility)
    total_eval_time::Float64
    eval_count::Int
    per_iteration_times::Vector{Float64}
    iteration_start_time::Float64
    
    TimingStats() = new(
        0.0, 0.0, 0.0, 0.0, 0.0,  # times
        0, 0, 0, 0, 0,              # counts
        0.0, 0, Float64[], time()   # totals
    )
end

"""
    record_eval_time!(stats::TimingStats, elapsed::Float64, eval_type::Symbol)

Record evaluation time in timing stats with categorization by operation type.

# Arguments
- `stats`: TimingStats object to update
- `elapsed`: Time elapsed for this evaluation
- `eval_type`: Type of evaluation - one of :obj, :grad, :cons, :jac, :hess
"""
function record_eval_time!(stats::TimingStats, elapsed::Float64, eval_type::Symbol)
    # Update total statistics
    stats.total_eval_time += elapsed
    stats.eval_count += 1
    push!(stats.per_iteration_times, elapsed)
    
    # Update categorized statistics
    if eval_type == :obj
        stats.obj_time += elapsed
        stats.obj_count += 1
    elseif eval_type == :grad
        stats.grad_time += elapsed
        stats.grad_count += 1
    elseif eval_type == :cons
        stats.cons_time += elapsed
        stats.cons_count += 1
    elseif eval_type == :jac
        stats.jac_time += elapsed
        stats.jac_count += 1
    elseif eval_type == :hess
        stats.hess_time += elapsed
        stats.hess_count += 1
    else
        @warn "Unknown eval_type: $eval_type, not categorized"
    end
end

# ============================================================================
# NLP Model with Neural Network
# ============================================================================

"""
    NeuralNetworkNLPModel <: AbstractNLPModel

NLP model integrating a neural network with custom objective and constraints.

This model supports multiple backends for neural network evaluation and derivatives:
1. Julia/Flux backend with ForwardDiff: Uses Flux for inference and ForwardDiff for derivatives
2. Julia/Flux backend with Zygote: Uses Flux for inference and Zygote for derivatives (faster for large n)
3. Python/JAX backend: Uses JAX for inference and JAX AD for derivatives (optional)

All backends use the same MadNLP solver for optimization.

# Fields
- `meta`: NLP model metadata
- `counters`: Function evaluation counters
- `neural_network`: Flux model (only used if use_python=false)
- `objective_func`: Objective function f(x, θ)
- `constraint_func`: Constraint function g(x)
- `x0`: Initial point
- `use_python`: If true, use Python/JAX backend; if false, use Flux
- `python_evaluator`: Python JAX evaluator (only used if use_python=true)
- `timing_stats`: Timing statistics for performance analysis
- `ad_backend`: AD backend for Flux (:forwarddiff or :zygote), ignored if use_python=true
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
    timing_stats::TimingStats  # Timing statistics
    ad_backend::Symbol  # AD backend: :forwarddiff or :zygote (only for Flux backend)
end


"""
    NeuralNetworkNLPModel(
        model_path::String,
        objective_func::AbstractObjectiveFunction,
        constraint_func::Union{AbstractConstraintFunction, Nothing},
        x0::Vector{Float64};
        use_python::Bool=false,
        ad_backend::Symbol=:zygote
    )

Create an NLP model with a neural network.

# Arguments
- `model_path`: Path to trained PyTorch model
- `objective_func`: Objective function
- `constraint_func`: Constraint function (or nothing)
- `x0`: Initial point
- `use_python`: If true, use Python/JAX backend; if false, use Flux (default: false)
- `ad_backend`: AD backend for Flux (:forwarddiff or :zygote, default: :zygote)
                Ignored if use_python=true

# Performance Guide
- For n < 100: use_python=false, ad_backend=:forwarddiff (simple and stable)
- For 100 < n < 500: use_python=false, ad_backend=:zygote (faster gradients/Hessians) ⭐ DEFAULT
- For n > 500: use_python=true (JAX is optimal for large-scale problems)
"""
function NeuralNetworkNLPModel(
    model_path::String,
    objective_func::AbstractObjectiveFunction,
    constraint_func::Union{AbstractConstraintFunction, Nothing},
    x0::Vector{Float64};
    use_python::Bool=false,
    ad_backend::Symbol=:zygote
)
    if use_python
        @info "Creating NLP model with Python/JAX backend"
        return _create_python_nlp_model(model_path, objective_func, constraint_func, x0)
    else
        @info "Creating NLP model with Julia/Flux backend (AD: $ad_backend)"
        return _create_flux_nlp_model(model_path, objective_func, constraint_func, x0, ad_backend)
    end
end


"""
    _create_flux_nlp_model(...)

Internal function to create NLP model with Flux backend and specified AD.
"""
function _create_flux_nlp_model(
    model_path::String,
    objective_func::AbstractObjectiveFunction,
    constraint_func::Union{AbstractConstraintFunction, Nothing},
    x0::Vector{Float64},
    ad_backend::Symbol=:zygote
)
    # Load neural network
    nn_model, nn_config = load_pytorch_model_to_flux(model_path)
    
    # Determine dimensions
    n = length(x0)
    
    # Extract box bounds and remaining constraints
    # Treat BoxConstraints as variable bounds (lvar/uvar) instead of general constraints
    # This matches the JAX backend behavior and is more efficient
    lvar = fill(-Inf, n)
    uvar = fill(Inf, n)
    remaining_constraint = nothing
    
    if constraint_func isa BoxConstraints
        # Box constraints -> variable bounds
        lvar = copy(constraint_func.x_min)
        uvar = copy(constraint_func.x_max)
        remaining_constraint = nothing
    elseif constraint_func isa CompositeConstraint
        # Extract box constraints from composite
        non_box_constraints = AbstractConstraintFunction[]
        for c in constraint_func.constraints
            if c isa BoxConstraints
                lvar = copy(c.x_min)
                uvar = copy(c.x_max)
            else
                push!(non_box_constraints, c)
            end
        end
        
        # Remaining constraints (non-box)
        if length(non_box_constraints) == 0
            remaining_constraint = nothing
        elseif length(non_box_constraints) == 1
            remaining_constraint = non_box_constraints[1]
        else
            remaining_constraint = CompositeConstraint(non_box_constraints...)
        end
    else
        # Other constraint types (spherical, etc.)
        remaining_constraint = constraint_func
    end
    
    # Count only non-box constraints
    m = remaining_constraint === nothing ? 0 : num_constraints(remaining_constraint)
    
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
    timing_stats = TimingStats()
    
    # Validate AD backend
    if !(ad_backend in [:forwarddiff, :zygote])
        @warn "Unknown AD backend: $ad_backend, falling back to :forwarddiff"
        ad_backend = :forwarddiff
    end
    
    return NeuralNetworkNLPModel(
        meta,
        counters,
        nn_model,
        objective_func,
        remaining_constraint,  # Use remaining constraint (without box)
        copy(x0),
        false,    # use_python = false
        nothing,  # python_evaluator = nothing
        timing_stats,
        ad_backend  # AD backend
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
    # Ensure Python path includes project python directory so local modules can be imported
    setup_python_env()
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
    # Treat BoxConstraints as variable bounds (lvar/uvar) instead of general constraints
    x_min = nothing
    x_max = nothing
    radius = nothing
    
    if constraint_func isa BoxConstraints
        x_min = constraint_func.x_min
        x_max = constraint_func.x_max
    elseif constraint_func isa SphericalConstraint
        radius = constraint_func.radius
    elseif constraint_func isa CompositeConstraint
        # Extract from composite constraint
        for c in constraint_func.constraints
            if c isa BoxConstraints
                x_min = c.x_min
                x_max = c.x_max
            elseif c isa SphericalConstraint
                radius = c.radius
            end
        end
    end
    
    # Do NOT pass box bounds as inequality constraints to Python evaluator
    bounds_for_evaluator = nothing
    
    # Create Python evaluator
    py_eval = jax_evaluator_module.create_evaluator(
        model_path,
        target,
        x0;
        objective_weight=1.0,
        regularization_weight=regularization_weight,
        bounds=bounds_for_evaluator,
        radius=radius
    )
    
    # Get dimensions from evaluator
    n = pyconvert(Int, py_eval.n)
    m = pyconvert(Int, py_eval.m)
    
    # Set up variable bounds (use box as variable bounds if provided)
    lvar = x_min === nothing ? fill(-Inf, n) : copy(x_min)
    uvar = x_max === nothing ? fill(Inf, n)  : copy(x_max)
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
    timing_stats = TimingStats()
    
    return NeuralNetworkNLPModel(
        meta,
        counters,
        nothing,  # neural_network = nothing (using Python)
        objective_func,
        constraint_func,
        copy(x0),
        true,     # use_python = true
        py_eval,  # python_evaluator
        timing_stats,
        :forwarddiff  # ad_backend (unused for Python backend, just a placeholder)
    )
end


# ============================================================================
# Helper Functions for AD
# ============================================================================

"""
    _compute_hessian(f, x, ad_backend::Symbol)

Compute Hessian matrix using specified AD backend.

# Arguments
- `f`: Scalar-valued function f: R^n → R
- `x`: Point at which to compute Hessian
- `ad_backend`: AD backend (:forwarddiff or :zygote, default is :zygote)

# Returns
- Hessian matrix (n × n symmetric matrix)

# Details
- ForwardDiff: Uses forward-mode AD, complexity O(n²)
  - Best for n < 100
  - Simple and stable
- Zygote: Uses forward-over-reverse mode, complexity O(n) ⭐ DEFAULT
  - Best for n ≥ 100
  - 5-10x faster for large n
  - Computes Jacobian of gradient (forward-over-reverse)
"""
function _compute_hessian(f::Function, x::Vector{Float64}, ad_backend::Symbol)
    if ad_backend == :zygote
        # Zygote: forward-over-reverse mode
        # Compute Jacobian of the gradient (forward-over-reverse)
        # This is O(n) reverse passes, more efficient than O(n²) forward passes
        n = length(x)
        H = zeros(n, n)
        
        # Compute gradient function
        grad_f = x_val -> Zygote.gradient(f, x_val)[1]
        
        # Compute Jacobian of gradient using ForwardDiff
        # This gives us the Hessian via forward-over-reverse
        H = ForwardDiff.jacobian(grad_f, x)
        
        # Ensure symmetry (due to potential numerical errors)
        H = 0.5 * (H + H')
        
        return H
    else
        # ForwardDiff: forward-mode AD (default)
        return ForwardDiff.hessian(f, x)
    end
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
    
    eval_time = @elapsed begin
        if nlp.use_python
            # Use Python/JAX backend
            f = pyconvert(Float64, nlp.python_evaluator.evaluate_obj(x))
        else
            # Use Flux backend
            nn_output = nlp.neural_network(x)
            f = evaluate(nlp.objective_func, x, nn_output)
        end
    end
    
    record_eval_time!(nlp.timing_stats, eval_time, :obj)
    return f
end


"""
    NLPModels.grad!(nlp::NeuralNetworkNLPModel, x::AbstractVector, g::AbstractVector)

Evaluate objective gradient at x using automatic differentiation.
"""
function NLPModels.grad!(nlp::NeuralNetworkNLPModel, x::AbstractVector, g::AbstractVector)
    NLPModels.increment!(nlp, :neval_grad)
    
    eval_time = @elapsed begin
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
        else
            # Use specified AD backend for gradient computation
            obj_func = x_val -> begin
                nn_output = nlp.neural_network(x_val)
                evaluate(nlp.objective_func, x_val, nn_output)
            end
            
            if nlp.ad_backend == :zygote
                # Zygote (reverse-mode AD, faster for large n)
                g .= Zygote.gradient(obj_func, x)[1]
            else
                # ForwardDiff (forward-mode AD, default)
                g .= ForwardDiff.gradient(obj_func, x)
            end
        end
    end
    
    record_eval_time!(nlp.timing_stats, eval_time, :grad)
    return g
end


"""
    NLPModels.cons!(nlp::NeuralNetworkNLPModel, x::AbstractVector, c::AbstractVector)

Evaluate constraint functions at x.
"""
function NLPModels.cons!(nlp::NeuralNetworkNLPModel, x::AbstractVector, c::AbstractVector)
    NLPModels.increment!(nlp, :neval_cons)
    
    eval_time = @elapsed begin
        if nlp.use_python
            # Use Python/JAX backend
            if nlp.meta.ncon > 0
                c_py = nlp.python_evaluator.evaluate_cons(x)
                c .= pyconvert(Vector{Float64}, c_py)
            end
        else
            # Use Flux backend
            if nlp.constraint_func !== nothing
                c_vals = evaluate(nlp.constraint_func, x)
                c .= c_vals
            end
        end
    end
    
    record_eval_time!(nlp.timing_stats, eval_time, :cons)
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
    if nlp.use_python
        structure = nlp.python_evaluator.jacobian_structure()
        rows .= pyconvert(Vector{Int}, structure[0]) .+ 1
        cols .= pyconvert(Vector{Int}, structure[1]) .+ 1
    else
        n = nlp.meta.nvar
        m = nlp.meta.ncon
        idx = 1
        for i in 1:m, j in 1:n
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
    
    eval_time = @elapsed begin
        if nlp.use_python
            # Use Python/JAX backend
            coordinates = nlp.python_evaluator.evaluate_jac_coord(x)
            vals .= pyconvert(Vector{Float64}, coordinates)
        else
            # Use specified AD backend
            jac_func = x_val -> begin
                T = eltype(x_val)
                c = zeros(T, nlp.meta.ncon)
                c_vals = evaluate(nlp.constraint_func, x_val)
                c .= c_vals
                return c
            end
            
            if nlp.ad_backend == :zygote
                # Zygote with Jacobian computation
                # For vector-valued functions, use ForwardDiff over Zygote gradient
                J = ForwardDiff.jacobian(jac_func, x)
            else
                # ForwardDiff
                J = ForwardDiff.jacobian(jac_func, x)
            end
            
            # Fill vals in row-major order
            idx = 1
            for i in 1:nlp.meta.ncon
                for j in 1:nlp.meta.nvar
                    vals[idx] = J[i, j]
                    idx += 1
                end
            end
        end
    end
    
    record_eval_time!(nlp.timing_stats, eval_time, :jac)
    return vals
end


function NLPModels.jac_dense!(
    nlp::NeuralNetworkNLPModel,
    x::AbstractVector,
    jacobian::AbstractMatrix,
)
    NLPModels.increment!(nlp, :neval_jac)
    isempty(jacobian) && return jacobian
    elapsed = @elapsed begin
        if nlp.use_python
            py_jacobian = nlp.python_evaluator.evaluate_jac(x)
            jacobian .= pyconvert(Matrix{Float64}, py_jacobian)
        else
            jacobian_function = x_value -> evaluate(nlp.constraint_func, x_value)
            jacobian .= ForwardDiff.jacobian(jacobian_function, x)
        end
    end
    record_eval_time!(nlp.timing_stats, elapsed, :jac)
    return jacobian
end


function NLPModels.hess_dense!(
    nlp::NeuralNetworkNLPModel,
    x::AbstractVector,
    y::AbstractVector,
    hessian::AbstractMatrix;
    obj_weight=1.0,
)
    NLPModels.increment!(nlp, :neval_hess)
    elapsed = @elapsed begin
        if nlp.use_python
            py_hessian = nlp.python_evaluator.evaluate_hess(x, y, obj_weight)
            hessian .= pyconvert(Matrix{Float64}, py_hessian)
        else
            lagrangian = x_value -> begin
                output = nlp.neural_network(x_value)
                value = obj_weight * evaluate(nlp.objective_func, x_value, output)
                if nlp.constraint_func !== nothing && !isempty(y)
                    value += dot(y, evaluate(nlp.constraint_func, x_value))
                end
                return value
            end
            hessian .= _compute_hessian(lagrangian, collect(x), nlp.ad_backend)
        end
    end
    record_eval_time!(nlp.timing_stats, elapsed, :hess)
    return hessian
end


"""Matrix-free constraint Jacobian product required by quasi-Newton MadNLP."""
function NLPModels.jprod!(
    nlp::NeuralNetworkNLPModel,
    x::AbstractVector,
    vector::AbstractVector,
    product::AbstractVector,
)
    NLPModels.increment!(nlp, :neval_jprod)
    isempty(product) && return product
    elapsed = @elapsed begin
        if nlp.use_python
            py_product = nlp.python_evaluator.evaluate_jprod(x, vector)
            product .= pyconvert(Vector{Float64}, py_product)
        else
            jacobian_function = x_value -> evaluate(nlp.constraint_func, x_value)
            jacobian = ForwardDiff.jacobian(jacobian_function, x)
            mul!(product, jacobian, vector)
        end
    end
    record_eval_time!(nlp.timing_stats, elapsed, :jac)
    return product
end


"""Matrix-free transpose-Jacobian product required by quasi-Newton MadNLP."""
function NLPModels.jtprod!(
    nlp::NeuralNetworkNLPModel,
    x::AbstractVector,
    vector::AbstractVector,
    product::AbstractVector,
)
    NLPModels.increment!(nlp, :neval_jtprod)
    fill!(product, 0.0)
    isempty(vector) && return product
    elapsed = @elapsed begin
        if nlp.use_python
            py_product = nlp.python_evaluator.evaluate_jtprod(x, vector)
            product .= pyconvert(Vector{Float64}, py_product)
        else
            jacobian_function = x_value -> evaluate(nlp.constraint_func, x_value)
            jacobian = ForwardDiff.jacobian(jacobian_function, x)
            mul!(product, transpose(jacobian), vector)
        end
    end
    record_eval_time!(nlp.timing_stats, elapsed, :jac)
    return product
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
    if nlp.use_python
        structure = nlp.python_evaluator.hessian_structure()
        rows .= pyconvert(Vector{Int}, structure[0]) .+ 1
        cols .= pyconvert(Vector{Int}, structure[1]) .+ 1
    else
        n = nlp.meta.nvar
        idx = 1
        for j in 1:n, i in j:n
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
    
    eval_time = @elapsed begin
        if nlp.use_python
            # Use Python/JAX backend
            coordinates = nlp.python_evaluator.evaluate_hess_coord(
                x, y, obj_weight,
            )
            vals .= pyconvert(Vector{Float64}, coordinates)
        else
            # Use specified AD backend (ForwardDiff or Zygote)
            n = nlp.meta.nvar
            
            # Special handling for specific constraint types (optimization)
            # Note: BoxConstraints are now handled as variable bounds (lvar/uvar),
            # not as general constraints, so nlp.constraint_func will be nothing or non-box
            if nlp.constraint_func === nothing || nlp.meta.ncon == 0
                # No general constraints: only objective Hessian
                # (Box constraints are handled as variable bounds by MadNLP)
                obj_func = x_val -> begin
                    nn_output = nlp.neural_network(x_val)
                    evaluate(nlp.objective_func, x_val, nn_output)
                end
                hess_lag = obj_weight .* _compute_hessian(obj_func, x, nlp.ad_backend)
                
            elseif nlp.constraint_func isa SphericalConstraint
                # Spherical constraint: ∇²g(x) = 2I (constant Hessian)
                obj_func = x_val -> begin
                    nn_output = nlp.neural_network(x_val)
                    evaluate(nlp.objective_func, x_val, nn_output)
                end
                hess_lag = obj_weight .* _compute_hessian(obj_func, x, nlp.ad_backend)
                
                # Add spherical constraint contribution: λ * 2I
                if abs(y[1]) > 1e-12  # Only if multiplier is non-zero
                    for i in 1:n
                        hess_lag[i, i] += 2.0 * y[1]
                    end
                end
                
            elseif nlp.constraint_func isa CompositeConstraint
                # Composite constraint: check components
                # Note: BoxConstraints are now handled as variable bounds, so they won't appear here
                has_sphere = any(c -> c isa SphericalConstraint, nlp.constraint_func.constraints)
                has_other = any(c -> !(c isa SphericalConstraint), nlp.constraint_func.constraints)
                
                if !has_other && has_sphere
                    # Only Spherical: can use fast path
                    obj_func = x_val -> begin
                        nn_output = nlp.neural_network(x_val)
                        evaluate(nlp.objective_func, x_val, nn_output)
                    end
                    hess_lag = obj_weight .* _compute_hessian(obj_func, x, nlp.ad_backend)
                    
                    # Add spherical constraint contribution
                    # Find spherical constraint index
                    sphere_idx = 0
                    current_idx = 0
                    for c in nlp.constraint_func.constraints
                        n_cons = num_constraints(c)
                        if c isa SphericalConstraint
                            sphere_idx = current_idx + 1  # Spherical has 1 constraint
                            break
                        end
                        current_idx += n_cons
                    end
                    
                    # Add contribution: λ * 2I
                    if sphere_idx > 0 && abs(y[sphere_idx]) > 1e-12
                        for i in 1:n
                            hess_lag[i, i] += 2.0 * y[sphere_idx]
                        end
                    end
                else
                    # General case: compute Lagrangian Hessian directly (FIXED APPROACH)
                    lagrangian_func = x_val -> begin
                        # Objective term
                        nn_output = nlp.neural_network(x_val)
                        L = obj_weight * evaluate(nlp.objective_func, x_val, nn_output)
                        
                        # Constraint term: λᵀ·g(x)
                        if nlp.meta.ncon > 0
                            c_vals = evaluate(nlp.constraint_func, x_val)
                            L += dot(y, c_vals)
                        end
                        
                        return L
                    end
                    
                    hess_lag = _compute_hessian(lagrangian_func, x, nlp.ad_backend)
                end
                
            else
                # General constraint type: compute Lagrangian Hessian directly (FIXED APPROACH)
                # This is the corrected implementation that computes the Hessian in one pass
                lagrangian_func = x_val -> begin
                    # Objective term
                    nn_output = nlp.neural_network(x_val)
                    L = obj_weight * evaluate(nlp.objective_func, x_val, nn_output)
                    
                    # Constraint term: λᵀ·g(x)
                    if nlp.meta.ncon > 0
                        c_vals = evaluate(nlp.constraint_func, x_val)
                        L += dot(y, c_vals)
                    end
                    
                    return L
                end
                
                hess_lag = _compute_hessian(lagrangian_func, x, nlp.ad_backend)
            end
            
            # Extract lower triangle
            idx = 1
            for j in 1:n
                for i in j:n
                    vals[idx] = hess_lag[i, j]
                    idx += 1
                end
            end
        end
    end
    
    record_eval_time!(nlp.timing_stats, eval_time, :hess)
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
    kkt_device::String="cpu",
    kwargs...
)
    kkt_device in ("cpu", "gpu") ||
        throw(ArgumentError("kkt_device must be \"cpu\" or \"gpu\", got $kkt_device"))
    @info "Solving NLP with MadNLP..."
    @info "  Variables: $(nlp.meta.nvar)"
    @info "  Constraints: $(nlp.meta.ncon)"
    @info "  KKT device: $kkt_device"
    
    # Build solver options  
    options = Dict{Symbol, Any}(
        :max_iter => max_iter,
        :tol => tol,
        :print_level => print_level
    )
    
    # The current NeuralNetworkNLPModel owns host vectors. SparseWrapperModel
    # makes the KKT path GPU resident while retaining host callbacks. This is a
    # deliberately named host-staged baseline: x is copied to the host for each
    # callback and derivatives are copied back to the device. The zero-copy
    # DLPack model is implemented separately and must not be conflated with this
    # path in benchmark labels.
    solver_model = nlp
    host_staged = false
    if kkt_device == "gpu"
        CUDA.functional() || error("GPU KKT requested but CUDA.jl is not functional")
        callback_type = get(kwargs, :callback, MadNLP.SparseCallback)
        solver_model = if callback_type === MadNLP.DenseCallback
            MadNLP.DenseWrapperModel(CuArray, nlp)
        else
            MadNLP.SparseWrapperModel(CuArray, nlp)
        end
        host_staged = true
        linear_solver === nothing &&
            (linear_solver = MadNLPGPU.CUDSSSolver)
        @info "  Transfer path: host-staged SparseWrapperModel"
    end

    # Add linear solver if specified.
    linear_solver !== nothing && (options[:linear_solver] = linear_solver)
    
    # Merge additional options
    merge!(options, Dict(kwargs))
    
    # Create and solve. Convert Dict -> NamedTuple so keyword forwarding is explicit.
    solver_kwargs = (; pairs(options)...)
    solver = MadNLPSolver(solver_model; solver_kwargs...)
    result = MadNLP.solve!(solver)
    complementarity = MadNLP.get_inf_compl(solver)
    kkt_error = MadNLP.get_kkt_error(solver)
    
    # Extract solution information
    status = result.status
    # Extract only primal variables (not dual variables)
    # MadNLP's result.solution contains [x; y] where x are primal and y are dual
    raw_solution = result.solution[1:nlp.meta.nvar]
    solution = raw_solution isa CuArray ? Array(raw_solution) : copy(raw_solution)
    raw_multipliers = result.multipliers
    multipliers = raw_multipliers isa CuArray ?
        Array(raw_multipliers) : copy(raw_multipliers)
    objective = result.objective
    iter_count = result.iter
    
    @info "Solver finished with status: $status"
    @info "  Objective: $objective"
    @info "  Iterations: $iter_count"
    
    # Extract timing statistics
    timing_stats = nlp.timing_stats
    
    return Dict(
        :status => status,
        :solution => solution,  # Only primal variables
        :multipliers => multipliers,
        :objective => objective,
        :iter_count => iter_count,
        :primal_feas => result.primal_feas,
        :dual_feas => result.dual_feas,
        :complementarity => complementarity,
        :kkt_error => kkt_error,
        :solver_counters => result.counters,
        :nlp_model => nlp,
        :timing_stats => timing_stats,
        :kkt_device => kkt_device,
        :host_staged => host_staged,
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
- `use_python`: If true, use Python/JAX backend; if false, use Flux (default: false)
- `ad_backend`: AD backend for Flux (:forwarddiff or :zygote, default: :zygote)
"""
function create_simple_nlp(
    model_path::String,
    target::Vector{Float64},
    x0::Vector{Float64};
    bounds=nothing,
    radius=nothing,
    regularization_weight=0.0,
    use_python::Bool=false,
    ad_backend::Symbol=:zygote
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
    
    return NeuralNetworkNLPModel(model_path, obj, cons, x0; use_python=use_python, ad_backend=ad_backend)
end


# ============================================================================
# ProblemSpec-based NLP construction
# ============================================================================

"""
    NeuralNetworkNLPModel(spec::ProblemSpec; device="cpu", kwargs...)

Construct an NLP model from a high-level `ProblemSpec`. This is the
recommended constructor for the Darcy inversion and Pareto tracing case
studies.

# Arguments
- `spec`: ProblemSpec describing the problem (networks, objectives, constraints)
- `device`: Execution device for Python/JAX backend ("cpu" or "gpu")
- `kwargs...`: Additional keyword arguments passed to the underlying constructor

# Notes
- Always uses the Python/JAX backend (required for multi-network problems)
- The `problem_type` field of `spec` selects the appropriate Python evaluator
"""
function NeuralNetworkNLPModel(spec::ProblemSpec; device::String="cpu", kwargs...)
    @info "Creating NLP model from ProblemSpec (type=$(spec.problem_type))"
    return _create_spec_nlp_model(spec, device)
end


"""
    _create_spec_nlp_model(spec, device)

Internal: build NeuralNetworkNLPModel from a ProblemSpec using the
Python/JAX backend.
"""
function _create_spec_nlp_model(spec::ProblemSpec, device::String)
    setup_python_env()
    jax_evaluator_module = pyimport("jax_nn_evaluator")

    n = spec.n

    # Extract constraint count.  LearnedFeasibilityConstraint counts handled
    # by the Python backend which also knows about box constraints.
    remaining_constraint, lvar, uvar = _extract_bounds_and_remaining(
        spec.constraints, n, spec.x_lb, spec.x_ub
    )

    # Determine m (number of general inequality constraints)
    m_julia = remaining_constraint === nothing ? 0 : num_constraints(remaining_constraint)

    # Dispatch to appropriate Python evaluator factory
    py_eval = _build_python_evaluator(jax_evaluator_module, spec, device, remaining_constraint)

    # m from Python evaluator takes priority (it may add its own constraints)
    m = pyconvert(Int, py_eval.m)

    lcon = fill(-Inf, m)
    ucon = fill(0.0, m)

    meta = NLPModelMeta(
        n,
        x0=copy(spec.x0),
        lvar=lvar,
        uvar=uvar,
        ncon=m,
        lcon=lcon,
        ucon=ucon,
        nnzj=m * n,
        nnzh=div(n * (n + 1), 2),
        minimize=true,
    )

    counters = Counters()
    timing_stats = TimingStats()

    return NeuralNetworkNLPModel(
        meta,
        counters,
        nothing,           # neural_network (Python backend)
        spec.objective,
        remaining_constraint,
        copy(spec.x0),
        true,              # use_python
        py_eval,
        timing_stats,
        :forwarddiff,      # ad_backend (unused for Python)
    )
end


"""
    _extract_bounds_and_remaining(constraint_func, n, x_lb, x_ub)

Merge ProblemSpec box bounds with any BoxConstraints found inside the
constraint tree, returning (remaining_constraint, lvar, uvar).
"""
function _extract_bounds_and_remaining(constraint_func, n, x_lb, x_ub)
    lvar = copy(x_lb)
    uvar = copy(x_ub)
    remaining = nothing

    if constraint_func === nothing
        return nothing, lvar, uvar
    elseif constraint_func isa BoxConstraints
        # Tighten bounds
        lvar .= max.(lvar, constraint_func.x_min)
        uvar .= min.(uvar, constraint_func.x_max)
        return nothing, lvar, uvar
    elseif constraint_func isa CompositeConstraint
        non_box = AbstractConstraintFunction[]
        for c in constraint_func.constraints
            if c isa BoxConstraints
                lvar .= max.(lvar, c.x_min)
                uvar .= min.(uvar, c.x_max)
            else
                push!(non_box, c)
            end
        end
        remaining = if isempty(non_box)
            nothing
        elseif length(non_box) == 1
            non_box[1]
        else
            CompositeConstraint(non_box...)
        end
        return remaining, lvar, uvar
    else
        return constraint_func, lvar, uvar
    end
end


"""
    _build_python_evaluator(mod, spec, device, remaining_constraint)

Dispatch to the correct Python evaluator factory based on spec.problem_type.
"""
function _build_python_evaluator(mod, spec::ProblemSpec, device::String, remaining_constraint)
    ptype = spec.problem_type

    if ptype == "darcy"
        return _build_darcy_evaluator(mod, spec, device, remaining_constraint)
    elseif ptype == "pareto"
        return _build_pareto_evaluator(mod, spec, device, remaining_constraint)
    else
        # Generic single-network path
        return _build_generic_evaluator(mod, spec, device, remaining_constraint)
    end
end


function _build_generic_evaluator(mod, spec::ProblemSpec, device::String, remaining_constraint)
    @assert length(spec.network_paths) == 1 "Generic evaluator requires exactly one network"
    model_path = spec.network_paths[1]

    # Extract target and regularization from objective
    target, reg_weight, reg_center = _unpack_inversion_objective(spec.objective, spec.x0)

    # Extract spherical radius if present
    radius = _extract_sphere_radius(remaining_constraint)

    return mod.create_evaluator(
        model_path,
        target,
        spec.x0;
        objective_weight=1.0,
        regularization_weight=reg_weight,
        radius=radius,
        device=device,
    )
end


function _build_darcy_evaluator(mod, spec::ProblemSpec, device::String, remaining_constraint)
    @assert length(spec.network_paths) >= 1 "Darcy evaluator requires at least one FNO network"
    model_path = spec.network_paths[1]

    target, reg_weight, _ = _unpack_inversion_objective(spec.objective, spec.x0)

    # Extract budget and smoothness constraint parameters
    budget = nothing
    tau = nothing
    L_flat = nothing
    L_n = 0

    for c in _flatten_constraints(remaining_constraint)
        if c isa BudgetConstraint
            budget = c.budget
        elseif c isa SmoothnessConstraint
            tau = c.tau
            L_flat = vec(c.L)
            L_n = size(c.L, 1)
        end
    end

    return mod.create_darcy_evaluator(
        model_path,
        target,
        spec.x0;
        regularization_weight=reg_weight,
        budget=budget,
        tau=tau,
        L_flat=L_flat,
        L_n=L_n,
        device=device,
    )
end


function _build_pareto_evaluator(mod, spec::ProblemSpec, device::String, remaining_constraint)
    @assert length(spec.network_paths) >= 2 "Pareto evaluator requires at least 2 networks"

    @assert spec.objective isa WeightedScalarizationObjective "Pareto problems need WeightedScalarizationObjective"
    alpha = spec.objective.alpha

    has_feasibility = any(
        c -> c isa LearnedFeasibilityConstraint,
        _flatten_constraints(remaining_constraint)
    )
    f3_path = length(spec.network_paths) >= 3 ? spec.network_paths[3] : nothing

    return mod.create_pareto_evaluator(
        spec.network_paths[1],
        spec.network_paths[2],
        spec.x0;
        f3_path=f3_path,
        alpha=alpha,
        device=device,
    )
end


# ============================================================================
# Helpers for objective/constraint inspection
# ============================================================================

function _unpack_inversion_objective(obj::AbstractObjectiveFunction, x0)
    if obj isa NeuralNetworkObjective
        return obj.target, 0.0, x0
    elseif obj isa SurrogateInversionObjective
        return obj.target, obj.reg_weight, obj.x0
    elseif obj isa CompositeObjective
        target = nothing
        reg_weight = 0.0
        for sub in obj.objectives
            if sub isa NeuralNetworkObjective
                target = sub.target
            elseif sub isa SurrogateInversionObjective
                target = sub.target
                reg_weight += sub.reg_weight
            elseif sub isa QuadraticRegularization
                reg_weight += sub.weight
            end
        end
        target === nothing && error("Cannot find a target-based objective in CompositeObjective")
        return target, reg_weight, x0
    else
        error("Unsupported objective type for Python backend: $(typeof(obj))")
    end
end

function _extract_sphere_radius(cons)
    cons === nothing && return nothing
    if cons isa SphericalConstraint
        return cons.radius
    elseif cons isa CompositeConstraint
        for c in cons.constraints
            c isa SphericalConstraint && return c.radius
        end
    end
    return nothing
end

function _flatten_constraints(cons)
    cons === nothing && return AbstractConstraintFunction[]
    cons isa CompositeConstraint && return cons.constraints
    return [cons]
end


# ============================================================================
# Solver backend selection utilities
# ============================================================================

"""
    select_linear_solver(device::String)

Return the recommended MadNLP linear solver type for the given KKT device
("cpu" or "gpu"). MadNLP 0.10 includes MUMPS directly; NVIDIA GPU KKT
systems use cuDSS.
"""
function select_linear_solver(device::String)
    device in ("cpu", "gpu") ||
        throw(ArgumentError("device must be \"cpu\" or \"gpu\", got $device"))
    if device == "gpu"
        CUDA.functional() || error("GPU solver requested but CUDA.jl is not functional")
        @info "GPU KKT: using MadNLPGPU.CUDSSSolver"
        return MadNLPGPU.CUDSSSolver
    end
    return MadNLP.MumpsSolver
end
