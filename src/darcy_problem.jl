"""
darcy_problem.jl

Convenience constructors for Case Study I: neural-operator inversion of
2-D Darcy flow.

Problem (from proposal §2):

    min_{x ∈ ℝⁿ}  (1/2)‖G_θ(x) − y_tar‖² + (λ/2)‖x − x₀‖²

    subject to
        ℓ ≤ x ≤ u                     (box bounds)
        1ᵀx ≤ B                       (budget)
        xᵀ L x ≤ τ                    (smoothness / TV proxy)

where G_θ is a Fourier Neural Operator surrogate mapping permeability
fields to Darcy pressure solutions, x is the discretized permeability
field, and y_tar is the target pressure field.

Exports
-------
- `DarcyProblemConfig` : parameter struct
- `create_darcy_nlp`   : main constructor returning NeuralNetworkNLPModel
"""

# ============================================================================
# Configuration struct
# ============================================================================

"""
    DarcyProblemConfig

All parameters needed to fully specify a Darcy inversion NLP.

# Fields
- `fno_path`: path to the FNO surrogate checkpoint
- `target_pressure`: target pressure field y_tar (length n)
- `x0`: initial permeability field (length n)
- `x_lb`, `x_ub`: element-wise permeability bounds (scalars or length-n vectors)
- `lambda_reg`: Tikhonov regularization weight
- `budget`: budget bound B for the linear constraint; `nothing` = inactive
- `tau`: smoothness bound τ; `nothing` = inactive
- `L`: regularity operator (n×n matrix); if `nothing` and `tau` is given, the
       2-D Laplacian for the grid implied by `grid_nx × grid_ny` is used
- `grid_nx`, `grid_ny`: grid dimensions (needed when L is auto-built)
- `device`: execution device ("cpu" or "gpu")
"""
struct DarcyProblemConfig
    fno_path::String
    target_pressure::Vector{Float64}
    x0::Vector{Float64}
    x_lb::Vector{Float64}
    x_ub::Vector{Float64}
    lambda_reg::Float64
    budget::Union{Float64, Nothing}
    tau::Union{Float64, Nothing}
    L::Union{Matrix{Float64}, Nothing}
    grid_nx::Int
    grid_ny::Int
    device::String

    function DarcyProblemConfig(
        fno_path::String,
        target_pressure::AbstractVector,
        x0::AbstractVector;
        x_lb::Union{Real, AbstractVector} = 0.0,
        x_ub::Union{Real, AbstractVector} = 5.0,
        lambda_reg::Real = 1e-2,
        budget = nothing,
        tau = nothing,
        L = nothing,
        grid_nx::Int = isqrt(length(x0)),
        grid_ny::Int = isqrt(length(x0)),
        device::String = "cpu",
    )
        n = length(x0)
        @assert length(target_pressure) > 0 "target_pressure must be non-empty"

        lb = x_lb isa Real ? fill(Float64(x_lb), n) : Float64.(x_lb)
        ub = x_ub isa Real ? fill(Float64(x_ub), n) : Float64.(x_ub)
        @assert length(lb) == n && length(ub) == n

        # Build 2-D Laplacian if tau is active but no L supplied
        L_mat = if tau !== nothing && L === nothing
            @info "DarcyProblemConfig: building 2-D Laplacian for $(grid_nx)×$(grid_ny) grid"
            Matrix{Float64}(laplacian_2d(grid_nx, grid_ny))
        elseif L !== nothing
            Matrix{Float64}(L)
        else
            nothing
        end

        new(
            fno_path,
            Float64.(target_pressure),
            Float64.(x0),
            lb, ub,
            Float64(lambda_reg),
            budget === nothing ? nothing : Float64(budget),
            tau === nothing ? nothing : Float64(tau),
            L_mat,
            grid_nx, grid_ny,
            device,
        )
    end
end

Base.show(io::IO, cfg::DarcyProblemConfig) = print(io,
    "DarcyProblemConfig(n=$(length(cfg.x0)), " *
    "grid=$(cfg.grid_nx)×$(cfg.grid_ny), " *
    "budget=$(cfg.budget !== nothing), " *
    "smoothness=$(cfg.tau !== nothing), " *
    "device=$(cfg.device))"
)


# ============================================================================
# Constructor
# ============================================================================

"""
    create_darcy_nlp(cfg::DarcyProblemConfig) → NeuralNetworkNLPModel

Build the full Darcy inversion NLP from a `DarcyProblemConfig`.

The returned model conforms to the NLPModels interface and can be passed
directly to `solve_nlp`.

# Example
```julia
cfg = DarcyProblemConfig(
    "output/darcy/models/fno_darcy_64x64_dv32_seed42.npz",
    target_pressure,
    x0;
    x_lb=0.1, x_ub=5.0,
    lambda_reg=1e-2,
    budget=50.0,
    tau=1e3,
    grid_nx=64,
    device="cpu"
)
nlp = create_darcy_nlp(cfg)
result = solve_nlp(nlp, max_iter=500, tol=1e-4)
```
"""
function create_darcy_nlp(cfg::DarcyProblemConfig)
    n = length(cfg.x0)

    # ---- objective ----
    obj = SurrogateInversionObjective(
        cfg.target_pressure,
        cfg.x0;
        weight = 0.5,
        reg_weight = cfg.lambda_reg / 2,
    )

    # ---- constraints ----
    constraint_list = AbstractConstraintFunction[]
    push!(constraint_list, BoxConstraints(cfg.x_lb, cfg.x_ub))

    if cfg.budget !== nothing
        push!(constraint_list, BudgetConstraint(cfg.budget))
    end

    if cfg.tau !== nothing && cfg.L !== nothing
        push!(constraint_list, SmoothnessConstraint(cfg.L, cfg.tau))
    end

    constraint_func = if length(constraint_list) == 1
        constraint_list[1]
    else
        CompositeConstraint(constraint_list...)
    end

    # ---- ProblemSpec ----
    spec = ProblemSpec(
        [cfg.fno_path],
        obj,
        cfg.x0;
        constraints = constraint_func,
        x_lb = cfg.x_lb,
        x_ub = cfg.x_ub,
        problem_type = "darcy",
    )

    return NeuralNetworkNLPModel(spec; device=cfg.device)
end


# ============================================================================
# Stage-wise constructors (proposal §2, Stages A / B / C)
# ============================================================================

"""
    create_darcy_stage_a(fno_path, target, x0; kwargs...)

Stage A: box constraints only. Used for correctness / backend cross-checks.
"""
function create_darcy_stage_a(
    fno_path::String,
    target_pressure::AbstractVector,
    x0::AbstractVector;
    x_lb = 0.0,
    x_ub = 5.0,
    lambda_reg = 1e-2,
    device = "cpu",
)
    cfg = DarcyProblemConfig(
        fno_path, target_pressure, x0;
        x_lb, x_ub, lambda_reg, budget=nothing, tau=nothing, device,
    )
    return create_darcy_nlp(cfg)
end


"""
    create_darcy_stage_b(fno_path, target, x0; budget, tau, grid_nx, kwargs...)

Stage B: box + budget + smoothness constraints.
"""
function create_darcy_stage_b(
    fno_path::String,
    target_pressure::AbstractVector,
    x0::AbstractVector;
    x_lb = 0.0,
    x_ub = 5.0,
    lambda_reg = 1e-2,
    budget::Real = sum(x0),
    tau::Real = 1e3,
    grid_nx::Int = isqrt(length(x0)),
    device = "cpu",
)
    cfg = DarcyProblemConfig(
        fno_path, target_pressure, x0;
        x_lb, x_ub, lambda_reg, budget=Float64(budget),
        tau=Float64(tau), grid_nx, device,
    )
    return create_darcy_nlp(cfg)
end


"""
    create_darcy_stage_c(fno_path, target, x0; kwargs...)

Stage C: same as Stage B; the distinction is in the experiment driver
(scaling study) not in the NLP formulation.
"""
create_darcy_stage_c = create_darcy_stage_b
