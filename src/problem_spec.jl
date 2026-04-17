"""
problem_spec.jl

High-level problem specification types for proposal case studies.

Adds:
- Extended constraint types: BudgetConstraint, SmoothnessConstraint,
  LearnedFeasibilityConstraint
- Extended objective types: SurrogateInversionObjective,
  WeightedScalarizationObjective
- ProblemSpec: a composable problem description for multi-network setups

All types remain compatible with the existing AbstractObjectiveFunction /
AbstractConstraintFunction dispatch used by NeuralNetworkNLPModel.
"""

# ============================================================================
# Extended Constraint Types
# ============================================================================

"""
    BudgetConstraint(budget)

Linear budget constraint: 1ᵀx ≤ budget.

Used in the Darcy inversion case study to enforce an integral or average
bound on the permeability field.
"""
struct BudgetConstraint <: AbstractConstraintFunction
    budget::Float64
    BudgetConstraint(b::Real) = new(Float64(b))
end

function evaluate(cons::BudgetConstraint, x::AbstractVector)
    T = eltype(x)
    return [sum(x) - convert(T, cons.budget)]
end

num_constraints(::BudgetConstraint) = 1

function hessian_contribution(::BudgetConstraint, ::AbstractVector, ::Real)
    # Linear constraint: zero Hessian contribution
    return nothing
end


"""
    SmoothnessConstraint(L, tau)

Quadratic smoothness constraint: xᵀ L x ≤ tau.

`L` is a positive semi-definite discrete regularity operator (e.g., the
graph Laplacian or a finite-difference Laplacian matrix). Used in the Darcy
inversion case study to control spatial roughness of the permeability field.
"""
struct SmoothnessConstraint <: AbstractConstraintFunction
    L::Matrix{Float64}
    tau::Float64

    function SmoothnessConstraint(L::AbstractMatrix, tau::Real)
        @assert issymmetric(Matrix(L)) "Regularity operator L must be symmetric"
        new(Matrix{Float64}(L), Float64(tau))
    end
end

function evaluate(cons::SmoothnessConstraint, x::AbstractVector)
    T = eltype(x)
    Lx = cons.L * x
    return [dot(x, Lx) - convert(T, cons.tau)]
end

num_constraints(::SmoothnessConstraint) = 1

"""Return the constant Hessian contribution 2*lambda*L for the smoothness
constraint."""
function hessian_contribution(cons::SmoothnessConstraint, ::AbstractVector, lambda::Real)
    return 2.0 * Float64(lambda) .* cons.L
end


"""
    LearnedFeasibilityConstraint(; network_index=3, threshold=0.0)

Constraint via a learned feasibility surrogate: h_{θ}(x) ≤ threshold.

`network_index` identifies which network in a multi-network evaluator
provides h. Used in the constrained Pareto tracing case study.
"""
struct LearnedFeasibilityConstraint <: AbstractConstraintFunction
    network_index::Int
    threshold::Float64

    LearnedFeasibilityConstraint(; network_index::Int = 3, threshold::Real = 0.0) =
        new(network_index, Float64(threshold))
end

function evaluate(cons::LearnedFeasibilityConstraint, x::AbstractVector)
    # Actual evaluation is delegated to the backend evaluator.
    # This stub keeps the interface consistent; the backend overrides
    # the raw cons! callback when this constraint type is detected.
    error(
        "LearnedFeasibilityConstraint must be evaluated through the Python/JAX " *
        "multi-network backend. Use NeuralNetworkNLPModel with use_python=true."
    )
end

num_constraints(::LearnedFeasibilityConstraint) = 1


# ============================================================================
# Extended Objective Types
# ============================================================================

"""
    SurrogateInversionObjective(target, x0; weight=1.0, reg_weight=0.0)

Inversion objective for neural operator surrogates:

    f(x) = weight * ‖N_θ(x) - target‖² + reg_weight * ‖x - x0‖²

Used in the Darcy/FNO inversion case study where `target` is the observed
pressure field and the network N_θ is the FNO surrogate.
"""
struct SurrogateInversionObjective <: AbstractObjectiveFunction
    target::Vector{Float64}
    x0::Vector{Float64}
    weight::Float64
    reg_weight::Float64

    function SurrogateInversionObjective(
        target::AbstractVector,
        x0::AbstractVector;
        weight::Real = 1.0,
        reg_weight::Real = 0.0,
    )
        new(Float64.(target), Float64.(x0), Float64(weight), Float64(reg_weight))
    end
end

function evaluate(
    obj::SurrogateInversionObjective,
    x::AbstractVector,
    nn_output::AbstractVector,
)
    T = promote_type(eltype(x), eltype(nn_output))
    diff = nn_output .- obj.target
    val = obj.weight * dot(diff, diff)
    if obj.reg_weight > 0
        dx = x .- obj.x0
        val += obj.reg_weight * dot(dx, dx)
    end
    return val
end


"""
    WeightedScalarizationObjective(alpha)

Weighted-sum scalarization objective for multi-network Pareto tracing:

    f(x; α) = α · f₁(x) + (1-α) · f₂(x)

where `alpha ∈ [0,1]` is the scalarization weight. The backend is
responsible for evaluating `f₁` and `f₂` and returning their concatenation
as `nn_output = [f₁(x); f₂(x)]`.
"""
struct WeightedScalarizationObjective <: AbstractObjectiveFunction
    alpha::Float64

    function WeightedScalarizationObjective(alpha::Real)
        @assert 0.0 ≤ Float64(alpha) ≤ 1.0 "alpha must be in [0, 1]"
        new(Float64(alpha))
    end
end

function evaluate(
    obj::WeightedScalarizationObjective,
    x::AbstractVector,
    nn_output::AbstractVector,
)
    if length(nn_output) < 2
        error(
            "WeightedScalarizationObjective requires at least 2 network outputs " *
            "stacked as nn_output = [f1(x); f2(x)]"
        )
    end
    return obj.alpha * nn_output[1] + (1 - obj.alpha) * nn_output[2]
end


# ============================================================================
# ProblemSpec: composable problem description
# ============================================================================

"""
    ProblemSpec

High-level descriptor for a reduced-space NLP involving one or more trained
neural networks. Passed to specialized constructors for each case study.

# Fields
- `n`: decision variable dimension
- `network_paths`: paths to trained PyTorch/checkpoint files (one or more)
- `objective`: objective function (subtypes AbstractObjectiveFunction)
- `constraints`: constraint function or `nothing`
- `x0`: initial point
- `x_lb`: lower variable bounds (default -∞)
- `x_ub`: upper variable bounds (default +∞)
- `problem_type`: string tag identifying case study (e.g., "darcy", "pareto",
                  or "generic")

# Constructor
    ProblemSpec(network_paths, objective, x0; constraints, x_lb, x_ub, problem_type)
"""
struct ProblemSpec
    n::Int
    network_paths::Vector{String}
    objective::AbstractObjectiveFunction
    constraints::Union{AbstractConstraintFunction, Nothing}
    x0::Vector{Float64}
    x_lb::Vector{Float64}
    x_ub::Vector{Float64}
    problem_type::String

    function ProblemSpec(
        network_paths::Vector{String},
        objective::AbstractObjectiveFunction,
        x0::AbstractVector{<:Real};
        constraints::Union{AbstractConstraintFunction, Nothing} = nothing,
        x_lb::AbstractVector{<:Real} = fill(-Inf, length(x0)),
        x_ub::AbstractVector{<:Real} = fill(Inf, length(x0)),
        problem_type::String = "generic",
    )
        n = length(x0)
        @assert length(x_lb) == n "x_lb must have length n=$n"
        @assert length(x_ub) == n "x_ub must have length n=$n"
        new(n, network_paths, objective, constraints, Float64.(x0),
            Float64.(x_lb), Float64.(x_ub), problem_type)
    end

    # Convenience: single network path
    function ProblemSpec(
        network_path::String,
        objective::AbstractObjectiveFunction,
        x0::AbstractVector{<:Real};
        kwargs...
    )
        ProblemSpec([network_path], objective, x0; kwargs...)
    end
end

Base.show(io::IO, spec::ProblemSpec) = print(
    io,
    "ProblemSpec(type=$(spec.problem_type), n=$(spec.n), " *
    "networks=$(length(spec.network_paths)), " *
    "constraints=$(spec.constraints === nothing ? "none" : typeof(spec.constraints)))"
)


# ============================================================================
# Discrete regularity helpers for SmoothnessConstraint
# ============================================================================

"""
    laplacian_1d(n)

Build the 1-D finite-difference Laplacian matrix of size n×n (zero boundary
conditions). Useful for constructing SmoothnessConstraint on 1-D or
flattened 2-D grids.
"""
function laplacian_1d(n::Int)
    L = zeros(n, n)
    for i in 1:n
        L[i, i] = 2.0
        if i > 1
            L[i, i-1] = -1.0
        end
        if i < n
            L[i, i+1] = -1.0
        end
    end
    return L
end


"""
    laplacian_2d(nx, ny)

Build the 2-D finite-difference Laplacian (Kronecker product) of size
(nx*ny)×(nx*ny). Used for grid-based smoothness regularization in the
Darcy inversion case study.
"""
function laplacian_2d(nx::Int, ny::Int = nx)
    Lx = laplacian_1d(nx)
    Ly = laplacian_1d(ny)
    Ix = I(nx)
    Iy = I(ny)
    return kron(Ix, Ly) + kron(Lx, Iy)
end
