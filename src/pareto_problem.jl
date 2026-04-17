"""
pareto_problem.jl

Convenience constructors for Case Study II: constrained multi-network
Pareto tracing.

Problem (from proposal §3):

    min_{x ∈ X}  α · f₁(x) + (1 − α) · f₂(x)
    s.t.         h(x) ≤ 0

where f₁, f₂ are competing learned objectives and h is a learned feasibility
surrogate, all implemented as frozen neural networks. Solving for a grid of
α values yields an approximate Pareto front.

Exports
-------
- `ParetoProblemConfig` : parameter struct
- `create_pareto_nlp`   : build NeuralNetworkNLPModel for one α
- `pareto_front_sweep`  : run a full α sweep and collect results
"""

# ============================================================================
# Configuration struct
# ============================================================================

"""
    ParetoProblemConfig

Parameters for a single scalarized Pareto subproblem.

# Fields
- `f1_path`, `f2_path`: checkpoint paths for the two objective networks
- `h_path`: checkpoint path for the feasibility surrogate (or `nothing`)
- `alpha`: scalarization weight α ∈ [0,1]
- `x0`: initial point
- `x_lb`, `x_ub`: box bounds
- `device`: execution device ("cpu" or "gpu")
"""
struct ParetoProblemConfig
    f1_path::String
    f2_path::String
    h_path::Union{String, Nothing}
    alpha::Float64
    x0::Vector{Float64}
    x_lb::Vector{Float64}
    x_ub::Vector{Float64}
    device::String

    function ParetoProblemConfig(
        f1_path::String,
        f2_path::String,
        x0::AbstractVector;
        h_path::Union{String, Nothing} = nothing,
        alpha::Real = 0.5,
        x_lb::Union{Real, AbstractVector} = -Inf,
        x_ub::Union{Real, AbstractVector} = Inf,
        device::String = "cpu",
    )
        n = length(x0)
        lb = x_lb isa Real ? fill(Float64(x_lb), n) : Float64.(x_lb)
        ub = x_ub isa Real ? fill(Float64(x_ub), n) : Float64.(x_ub)
        @assert 0.0 ≤ Float64(alpha) ≤ 1.0 "alpha must be in [0,1]"
        new(f1_path, f2_path, h_path, Float64(alpha), Float64.(x0), lb, ub, device)
    end
end

Base.show(io::IO, cfg::ParetoProblemConfig) = print(io,
    "ParetoProblemConfig(α=$(cfg.alpha), n=$(length(cfg.x0)), " *
    "feasibility=$(cfg.h_path !== nothing), device=$(cfg.device))"
)


# ============================================================================
# Single-problem constructor
# ============================================================================

"""
    create_pareto_nlp(cfg::ParetoProblemConfig) → NeuralNetworkNLPModel

Build the NLP for one scalarization weight α.

# Example
```julia
cfg = ParetoProblemConfig(
    "output/models/pareto/f1_net.pt",
    "output/models/pareto/f2_net.pt",
    x0;
    h_path = "output/models/pareto/h_net.pt",
    alpha = 0.3,
    x_lb = 0.0,
    x_ub = 1.0,
    device = "cpu"
)
nlp = create_pareto_nlp(cfg)
result = solve_nlp(nlp, max_iter=500, tol=1e-4)
```
"""
function create_pareto_nlp(cfg::ParetoProblemConfig)
    n = length(cfg.x0)

    # ---- objective ----
    obj = WeightedScalarizationObjective(cfg.alpha)

    # ---- constraints ----
    constraint_list = AbstractConstraintFunction[]

    if any(isfinite, cfg.x_lb) || any(isfinite, cfg.x_ub)
        push!(constraint_list, BoxConstraints(cfg.x_lb, cfg.x_ub))
    end

    if cfg.h_path !== nothing
        push!(constraint_list, LearnedFeasibilityConstraint(; network_index=3))
    end

    constraint_func = if isempty(constraint_list)
        nothing
    elseif length(constraint_list) == 1
        constraint_list[1]
    else
        CompositeConstraint(constraint_list...)
    end

    # ---- network paths ----
    network_paths = [cfg.f1_path, cfg.f2_path]
    cfg.h_path !== nothing && push!(network_paths, cfg.h_path)

    # ---- ProblemSpec ----
    spec = ProblemSpec(
        network_paths,
        obj,
        cfg.x0;
        constraints = constraint_func,
        x_lb = cfg.x_lb,
        x_ub = cfg.x_ub,
        problem_type = "pareto",
    )

    return NeuralNetworkNLPModel(spec; device=cfg.device)
end


# ============================================================================
# Front sweep orchestrator
# ============================================================================

"""
    pareto_front_sweep(f1_path, f2_path, x0; kwargs...) → Vector{Dict}

Run a weighted-sum Pareto sweep over a grid of α values.

Returns a vector of result dictionaries (one per α), each containing:
- `:alpha` — scalarization weight
- `:status` — MadNLP termination status
- `:solution` — primal solution x⋆
- `:objective` — scalarized objective value
- `:f1_val`, `:f2_val` — individual objectives at x⋆
- `:h_val` — feasibility surrogate value at x⋆ (or `nothing`)
- `:timing_stats` — TimingStats object
- `:feasible` — whether h(x⋆) ≤ 0 (or `true` when h_path is nothing)

# Keyword Arguments
- `h_path`: feasibility surrogate checkpoint (optional)
- `alpha_values`: vector of α values to sweep (default: 11 uniform values)
- `x_lb`, `x_ub`: box bounds
- `device`: "cpu" or "gpu"
- `max_iter`, `tol`: solver parameters
- `warm_start`: if true, use the previous solution as x0 for the next α
- `linear_solver`: MadNLP linear solver type (default: auto-select)
"""
function pareto_front_sweep(
    f1_path::String,
    f2_path::String,
    x0::AbstractVector{Float64};
    h_path::Union{String, Nothing} = nothing,
    alpha_values::AbstractVector{Float64} = collect(range(0.0, 1.0; length=11)),
    x_lb::Union{Real, AbstractVector} = -Inf,
    x_ub::Union{Real, AbstractVector} = Inf,
    device::String = "cpu",
    max_iter::Int = 500,
    tol::Float64 = 1e-4,
    warm_start::Bool = false,
    linear_solver = nothing,
)
    @info "Pareto sweep: $(length(alpha_values)) α values on $(device)"

    results = Dict{Symbol, Any}[]
    current_x0 = copy(x0)

    for (idx, alpha) in enumerate(sort(alpha_values))
        @info "  [$(idx)/$(length(alpha_values))] α = $(round(alpha, digits=4))"

        cfg = ParetoProblemConfig(
            f1_path, f2_path, current_x0;
            h_path, alpha, x_lb, x_ub, device,
        )

        nlp = create_pareto_nlp(cfg)

        solver_opts = Dict{Symbol, Any}(
            :max_iter => max_iter,
            :tol => tol,
            :print_level => MadNLP.ERROR,
        )
        linear_solver !== nothing && (solver_opts[:linear_solver] = linear_solver)

        result = solve_nlp(nlp; solver_opts...)

        x_sol = result[:solution]

        # Evaluate individual objectives at x⋆ via Python backend
        f1_val, f2_val, h_val = _evaluate_pareto_objectives(nlp, x_sol, h_path !== nothing)

        feasible = h_val === nothing || h_val <= 0.0

        push!(results, Dict(
            :alpha => alpha,
            :status => result[:status],
            :solution => x_sol,
            :objective => result[:objective],
            :f1_val => f1_val,
            :f2_val => f2_val,
            :h_val => h_val,
            :feasible => feasible,
            :timing_stats => result[:timing_stats],
            :iter_count => result[:iter_count],
        ))

        # Warm start: use current solution as next x0
        if warm_start
            current_x0 = copy(x_sol)
        end
    end

    n_feasible = sum(r -> r[:feasible], results)
    @info "Pareto sweep complete: $(n_feasible)/$(length(results)) feasible points"

    return results
end


"""
    _evaluate_pareto_objectives(nlp, x_sol, has_h)

Call the Python evaluator to get raw [f1, f2] (and optionally h) at x_sol.
"""
function _evaluate_pareto_objectives(nlp::NeuralNetworkNLPModel, x_sol::Vector{Float64}, has_h::Bool)
    if nlp.python_evaluator === nothing
        return nothing, nothing, nothing
    end

    try
        # The multi-network evaluator exposes evaluate_individuals
        f_vals = pyconvert(
            Vector{Float64},
            nlp.python_evaluator.evaluate_individuals(x_sol)
        )
        f1 = length(f_vals) >= 1 ? f_vals[1] : nothing
        f2 = length(f_vals) >= 2 ? f_vals[2] : nothing
        h  = (has_h && length(f_vals) >= 3) ? f_vals[3] : nothing
        return f1, f2, h
    catch e
        @warn "Could not evaluate individual objectives: $e"
        return nothing, nothing, nothing
    end
end


# ============================================================================
# Pareto quality metrics
# ============================================================================

"""
    pareto_hypervolume(front, ref_point) → Float64

Compute the dominated hypervolume for a 2-D Pareto front approximation.

`front` is a vector of (f1, f2) tuples (or pairs), and `ref_point` is the
reference point (typically the nadir point or a user-supplied upper bound).
"""
function pareto_hypervolume(
    front::AbstractVector,
    ref_point::Tuple{Float64, Float64},
)
    pts = [(Float64(p[1]), Float64(p[2])) for p in front]
    filter!(p -> p[1] <= ref_point[1] && p[2] <= ref_point[2], pts)
    isempty(pts) && return 0.0

    sort!(pts; by = p -> p[1])

    hv = 0.0
    prev_f2 = ref_point[2]
    for p in pts
        f1, f2 = p
        hv += (ref_point[1] - f1) * (prev_f2 - f2)
        prev_f2 = f2
    end
    return hv
end


"""
    pareto_spread(front) → Float64

Compute the Δ spread metric for a 2-D Pareto front approximation.
"""
function pareto_spread(front::AbstractVector)
    pts = sort([(Float64(p[1]), Float64(p[2])) for p in front]; by = p -> p[1])
    length(pts) < 2 && return 0.0

    distances = [sqrt((pts[i+1][1] - pts[i][1])^2 + (pts[i+1][2] - pts[i][2])^2)
                 for i in 1:length(pts)-1]
    d_mean = sum(distances) / length(distances)
    d_extremes = distances[1] + distances[end]
    d_middle_sum = length(distances) > 2 ? sum(abs(d - d_mean) for d in distances[2:end-1]) : 0.0

    n = length(distances)
    return (d_extremes + d_middle_sum) / (d_extremes + (n - 1) * d_mean)
end
