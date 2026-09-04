#!/usr/bin/env julia
"""
run_pareto_tracing.jl

Case Study II (proposal §3): constrained multi-network Pareto tracing.

Stages:
  A — constrained single-objective solves (α ∈ {0, 1})
  B — coarse Pareto sweep (11 uniform α values)
  C — dense sweep + multi-start robustness test

Usage
-----
    julia examples/run_pareto_tracing.jl [--stage A|B|C|all] [--device cpu|gpu]
        [--n_dim 20] [--f1 path] [--f2 path] [--h path]

If --f1/--f2/--h are not supplied, surrogates are trained automatically.
"""

using MadNLP4NN
using MadNLP
using LinearAlgebra
using Random
using Statistics: mean
using JSON3
using Printf
using Dates
using ArgParse
using PythonCall

const GPU_AVAILABLE = try
    @eval using MadNLPGPU; true
catch; false; end

const MUMPS_AVAILABLE = true  # MUMPS is built into MadNLP >= 0.10

println("="^80)
println("MadNLP4NN — Case Study II: Constrained Pareto Tracing")
println("="^80)
println("Started: $(Dates.now())")
println()


# ============================================================================
# Argument parsing
# ============================================================================

function parse_cli_args()
    s = ArgParseSettings()
    @add_arg_table! s begin
        "--stage";    default = "B"
        "--n_dim";    arg_type = Int;    default = 20
        "--device";   default = "cpu"
        "--f1";       default = ""
        "--f2";       default = ""
        "--h";        default = ""
        "--output_dir"; default = "output/pareto/results"
        "--max_iter"; arg_type = Int;    default = 500
        "--tol";      arg_type = Float64; default = 1e-4
        "--seed";     arg_type = Int;    default = 42
        "--n_alpha";  arg_type = Int;    default = 11
        "--n_starts"; arg_type = Int;    default = 5
        "--n_samples"; arg_type = Int;   default = 3000
        "--epochs";   arg_type = Int;    default = 150
        "--x_lb";     arg_type = Float64; default = -1.0
        "--x_ub";     arg_type = Float64; default = 1.0
    end
    return ArgParse.parse_args(ARGS, s)
end

args = parse_cli_args()

const STAGE      = args["stage"]
const N_DIM      = args["n_dim"]
const DEVICE     = args["device"]
const OUTPUT_DIR = args["output_dir"]
const MAX_ITER   = args["max_iter"]
const TOLERANCE  = args["tol"]
const SEED       = args["seed"]
const N_ALPHA    = args["n_alpha"]
const N_STARTS   = args["n_starts"]
const X_LB       = args["x_lb"]
const X_UB       = args["x_ub"]

Random.seed!(SEED)
mkpath(OUTPUT_DIR)

println("[Configuration]")
println("  Stage: $(STAGE), n=$(N_DIM), device=$(DEVICE)")
println("  α values: $(N_ALPHA), multi-start: $(N_STARTS)")
println()


# ============================================================================
# Ensure surrogate checkpoints exist
# ============================================================================

function ensure_surrogates(f1_arg, f2_arg, h_arg, n_dim, seed, n_samples, epochs)
    if !isempty(f1_arg) && isfile(f1_arg) && !isempty(f2_arg) && isfile(f2_arg)
        println("[Surrogates] Using existing checkpoints")
        return f1_arg, f2_arg, (isempty(h_arg) ? nothing : h_arg)
    end

    println("[Surrogates] Training f1, f2, h surrogates (n=$(n_dim))...")
    setup_python_env()
    pareto_module = pyimport("pareto_data")

    paths = pyconvert(Dict, pareto_module.train_all_surrogates(
        n=n_dim, n_samples=n_samples, seed=seed,
        data_dir="output/pareto",
        output_dir="output/models/pareto",
        epochs=epochs,
        hidden=64,
        device=DEVICE,
    ))

    f1_path = String(paths["f1_path"])
    f2_path = String(paths["f2_path"])
    h_path  = String(paths["h_path"])
    println("  f1: $(f1_path)")
    println("  f2: $(f2_path)")
    println("  h:  $(h_path)")
    return f1_path, f2_path, h_path
end

f1_path, f2_path, h_path = ensure_surrogates(
    args["f1"], args["f2"], args["h"],
    N_DIM, SEED, args["n_samples"], args["epochs"]
)


# ============================================================================
# Shared solver helper
# ============================================================================

function build_solver_opts(device)
    opts = Dict{Symbol,Any}(
        :max_iter => MAX_ITER,
        :tol => TOLERANCE,
        :print_level => MadNLP.ERROR,
    )
    ls = select_linear_solver(device)
    ls !== nothing && (opts[:linear_solver] = ls)
    MUMPS_AVAILABLE && device == "cpu" &&
        (opts[:linear_solver] = MadNLP.MumpsSolver)
    return opts
end

function solve_single(alpha::Float64, x0::Vector{Float64})
    cfg = ParetoProblemConfig(
        f1_path, f2_path, x0;
        h_path = h_path,
        alpha = alpha,
        x_lb = X_LB,
        x_ub = X_UB,
        device = DEVICE,
    )
    nlp = create_pareto_nlp(cfg)
    opts = build_solver_opts(DEVICE)
    t0 = time()
    result = solve_nlp(nlp; kkt_device=DEVICE, opts...)
    elapsed = time() - t0

    x_sol = result[:solution]
    f_vals = evaluate_pareto_objectives(nlp, x_sol, h_path !== nothing)
    f1_val, f2_val, h_val = f_vals

    return Dict(
        :alpha => alpha,
        :status => string(result[:status]),
        :objective => result[:objective],
        :f1_val => f1_val,
        :f2_val => f2_val,
        :h_val => h_val,
        :feasible => (h_val === nothing || h_val <= 0.0),
        :solve_time => elapsed,
        :iter_count => result[:iter_count],
        :solution => x_sol,
    )
end


# ============================================================================
# Stage A: single-objective solves
# ============================================================================

function run_stage_a()
    println("="^60)
    println("[Stage A] Single-objective solves (α ∈ {0, 1})")
    println("="^60)

    x0 = zeros(N_DIM)
    results = []

    for (alpha_str, alpha_val) in [("f2 only (α=0)", 0.0), ("f1 only (α=1)", 1.0)]
        println("  Solving: $(alpha_str)")
        r = solve_single(alpha_val, x0)
        println("    Status: $(r[:status])")
        println("    f1=$(r[:f1_val] !== nothing ? @sprintf("%.4f", r[:f1_val]) : "N/A"), " *
                "f2=$(r[:f2_val] !== nothing ? @sprintf("%.4f", r[:f2_val]) : "N/A")")
        println("    h=$(r[:h_val] !== nothing ? @sprintf("%.4f", r[:h_val]) : "N/A") " *
                "($(r[:feasible] ? "✓ feasible" : "✗ infeasible"))")
        push!(results, r)
    end

    return results
end


# ============================================================================
# Stage B: coarse Pareto sweep
# ============================================================================

function run_stage_b()
    println("="^60)
    println("[Stage B] Coarse Pareto sweep ($(N_ALPHA) α values)")
    println("="^60)

    alpha_values = collect(range(0.0, 1.0; length=N_ALPHA))
    x0 = zeros(N_DIM)

    results = pareto_front_sweep(
        f1_path, f2_path, x0;
        h_path = h_path,
        alpha_values = alpha_values,
        x_lb = X_LB,
        x_ub = X_UB,
        device = DEVICE,
        max_iter = MAX_ITER,
        tol = TOLERANCE,
        warm_start = true,
        linear_solver = (MUMPS_AVAILABLE && DEVICE == "cpu") ?
            MadNLP.MumpsSolver : nothing,
    )

    # Summary table
    n_feas = sum(r[:feasible] for r in results)
    println("\n  α       f1       f2      h       Feasible  Status")
    for r in results
        f1s = r[:f1_val] !== nothing ? @sprintf("%.4f", r[:f1_val]) : "  N/A  "
        f2s = r[:f2_val] !== nothing ? @sprintf("%.4f", r[:f2_val]) : "  N/A  "
        hs  = r[:h_val]  !== nothing ? @sprintf("%.4f", r[:h_val])  : "  N/A  "
        @printf("  %.2f  %s  %s  %s  %-6s  %s\n",
            r[:alpha], f1s, f2s, hs,
            r[:feasible] ? "✓" : "✗",
            r[:status])
    end
    println("\n  Feasible points: $(n_feas)/$(length(results))")

    # Compute Pareto quality metrics on feasible points
    feasible_pts = [(r[:f1_val], r[:f2_val]) for r in results
                    if r[:feasible] && r[:f1_val] !== nothing && r[:f2_val] !== nothing]

    if length(feasible_pts) >= 2
        f1_max = maximum(p[1] for p in feasible_pts)
        f2_max = maximum(p[2] for p in feasible_pts)
        ref = (f1_max * 1.1, f2_max * 1.1)
        hv = pareto_hypervolume(feasible_pts, ref)
        sp = pareto_spread(feasible_pts)
        println("  Hypervolume: $(@sprintf("%.4f", hv))")
        println("  Spread Δ:    $(@sprintf("%.4f", sp))")
    end

    return results
end


# ============================================================================
# Stage C: dense sweep + multi-start
# ============================================================================

function run_stage_c()
    println("="^60)
    println("[Stage C] Dense sweep + multi-start robustness test")
    println("="^60)

    alpha_dense = collect(range(0.0, 1.0; length=21))
    rng = Random.MersenneTwister(SEED)

    all_results = []

    for start_idx in 1:N_STARTS
        x0 = rand(rng, N_DIM) .* (X_UB - X_LB) .+ X_LB
        println("  Start $(start_idx)/$(N_STARTS): ‖x0‖=$(round(norm(x0), digits=3))")

        sweep_res = pareto_front_sweep(
            f1_path, f2_path, x0;
            h_path = h_path,
            alpha_values = alpha_dense,
            x_lb = X_LB,
            x_ub = X_UB,
            device = DEVICE,
            max_iter = MAX_ITER,
            tol = TOLERANCE,
            warm_start = true,
        )
        append!(all_results, sweep_res)
    end

    # Aggregate Pareto metrics
    n_total   = length(all_results)
    n_feas    = sum(r[:feasible] for r in all_results)
    feas_rate = 100 * n_feas / n_total

    println("\n  Multi-start summary:")
    println("  Total subproblems: $(n_total)")
    println("  Feasible: $(n_feas) ($(round(feas_rate, digits=1))%)")

    avg_time = mean(r[:solve_time] for r in all_results)
    println("  Avg solve time: $(@sprintf("%.3f", avg_time))s per subproblem")

    return all_results
end


# ============================================================================
# Run stages
# ============================================================================

all_results = Dict{String, Any}()
stages_to_run = STAGE == "all" ? ["A", "B", "C"] : [STAGE]

for stage in stages_to_run
    if stage == "A"
        all_results["A"] = run_stage_a()
    elseif stage == "B"
        all_results["B"] = run_stage_b()
    elseif stage == "C"
        all_results["C"] = run_stage_c()
    else
        @warn "Unknown stage: $stage"
    end
end


# ============================================================================
# Save results
# ============================================================================

timestamp = Dates.format(Dates.now(), "yyyymmdd_HHMMSS")
result_file = joinpath(OUTPUT_DIR,
    "pareto_stage$(STAGE)_n$(N_DIM)_$(DEVICE)_$(timestamp).json")

function to_serializable(v)
    v isa Vector{Float64} && return "(length=$(length(v)))"
    v isa Vector && return map(to_serializable, v)
    v isa Dict  && return Dict(string(k) => to_serializable(val) for (k, val) in v)
    return v
end

open(result_file, "w") do io
    JSON3.pretty(io, Dict(
        "timestamp" => string(Dates.now()),
        "configuration" => Dict(
            "stage" => STAGE,
            "n_dim" => N_DIM,
            "device" => DEVICE,
            "f1_path" => f1_path,
            "f2_path" => f2_path,
            "h_path" => h_path !== nothing ? h_path : "none",
            "max_iter" => MAX_ITER,
            "tol" => TOLERANCE,
            "n_alpha" => N_ALPHA,
        ),
        "results" => to_serializable(all_results),
    ))
end

println("\nResults saved: $(result_file)")
println("Completed: $(Dates.now())")
println("="^80)
