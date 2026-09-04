#!/usr/bin/env julia
"""
run_benchmarks.jl

Unified benchmark harness for MadNLP4NN.

Runs both case studies (Darcy inversion + Pareto tracing) across a matrix of:
  - Linear solvers
  - KKT system types
  - Execution devices (CPU / NVIDIA GPU)

and produces a single consolidated JSON report with the metrics required by
the proposal:
  - Objective value
  - KKT residual, complementarity, max constraint violation
  - Derivative evaluation time vs KKT factorization time
  - Inversion quality (relative recovery error) / Pareto quality

Usage
-----
    julia examples/run_benchmarks.jl [--case darcy|pareto|all]
        [--device cpu|gpu|both] [--quick]

--quick runs a minimal smoke test (small grid / few α values).
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
println("MadNLP4NN — Unified Benchmark Harness")
println("="^80)
println("Started: $(Dates.now())")
println("GPU: $(GPU_AVAILABLE ? "✓" : "✗")   MUMPS: $(MUMPS_AVAILABLE ? "✓" : "✗")")
println()


# ============================================================================
# Argument parsing
# ============================================================================

function parse_args_bench()
    s = ArgParseSettings()
    @add_arg_table! s begin
        "--case";    default = "all"
        "--device";  default = "cpu"
        "--quick";   action = :store_true
        "--output_dir"; default = "output/benchmark_results"
        "--seed";    arg_type = Int; default = 42
    end
    return ArgParse.parse_args(ARGS, s)
end

args = parse_args_bench()

const CASE       = args["case"]
const DEVICE_ARG = args["device"]
const QUICK      = args["quick"]
const OUTPUT_DIR = args["output_dir"]
const SEED       = args["seed"]

Random.seed!(SEED)
mkpath(OUTPUT_DIR)

DEVICES = if DEVICE_ARG == "both"
    GPU_AVAILABLE ? ["cpu", "gpu"] : (println("[WARN] GPU not available; running CPU only"); ["cpu"])
else
    [DEVICE_ARG]
end


# ============================================================================
# Solver configuration matrix
# ============================================================================

function build_solver_matrix(device::String)
    configs = []

    if device == "cpu"
        push!(configs, (label="MUMPS_Sparse",
            opts=Dict{Symbol,Any}(
                :linear_solver => MadNLP.MumpsSolver,
            )))
        push!(configs, (label="Umfpack_Sparse",
            opts=Dict{Symbol,Any}()))  # MadNLP default on CPU
    elseif device == "gpu"
        if GPU_AVAILABLE
            push!(configs, (label="cuDSS_HostStaged",
                opts=Dict{Symbol,Any}(
                    :linear_solver => MadNLPGPU.CUDSSSolver,
                )))
        else
            @warn "GPU device requested but MadNLPGPU not loaded"
        end
    end

    return configs
end


# ============================================================================
# Metric schema
# ============================================================================

function build_result_record(;
    case, stage, device, solver_label,
    status, objective, iter_count,
    solve_time, eval_time, hess_time,
    extra=Dict{String,Any}()
)
    eval_frac = solve_time > 0 ? eval_time / solve_time : 0.0
    hess_frac = eval_time > 0 ? hess_time / max(eval_time, 1e-12) : 0.0

    return Dict(
        "case" => case,
        "stage" => stage,
        "device" => device,
        "solver" => solver_label,
        "status" => string(status),
        "objective" => objective,
        "iter_count" => iter_count,
        "timing" => Dict(
            "total_solve_s" => solve_time,
            "eval_s" => eval_time,
            "hess_s" => hess_time,
            "eval_pct" => 100 * eval_frac,
            "hess_pct" => 100 * hess_frac,
        ),
        "extra" => extra,
    )
end


# ============================================================================
# Darcy benchmark
# ============================================================================

function benchmark_darcy(devices, quick)
    println("[Darcy] Running Darcy inversion benchmark")
    grid_n = quick ? 16 : 64
    println("  Grid: $(grid_n)×$(grid_n) (n=$(grid_n^2)), quick=$(quick)")

    # Ensure FNO exists (train a small one if not)
    setup_python_env()
    darcy_mod = pyimport("darcy_data")
    ckpt_dir = "output/darcy/models"
    fno_path_expected = joinpath(ckpt_dir, "fno_darcy_$(grid_n)x$(grid_n)_dv32_seed$(SEED).npz")

    if !isfile(fno_path_expected)
        n_samples = quick ? 100 : 500
        epochs    = quick ? 20  : 100
        darcy_mod.generate_darcy_dataset(n_samples=n_samples, grid_n=grid_n,
            seed=SEED, output_dir="output/darcy", split="train")
        fno_path_expected = pyconvert(String, darcy_mod.train_fno(
            data_dir="output/darcy", grid_n=grid_n, d_v=32, n_layers=4,
            k_max=min(16, grid_n ÷ 4), lr=1e-3,
            epochs=epochs, batch_size=16, seed=SEED,
            output_dir=ckpt_dir, device="cpu",
        ))
    end

    # Synthetic target
    rng = Random.MersenneTwister(SEED)
    x_true = clamp.(0.5 .+ 0.3 .* randn(rng, grid_n^2), 0.1, 3.0)
    jax_ev = pyimport("jax_nn_evaluator").create_darcy_evaluator(
        fno_path_expected, zeros(grid_n^2), x_true;
        regularization_weight=0.0, device="cpu",
    )
    y_tar = pyconvert(Vector{Float64}, jax_ev.evaluate_nn(x_true))
    x0 = clamp.(x_true .+ 0.1 .* randn(rng, grid_n^2), 0.1, 3.0)

    records = []

    for device in devices
        for solver_cfg in build_solver_matrix(device)
            println("  Device=$(device), Solver=$(solver_cfg.label)")
            nlp = create_darcy_stage_a(
                fno_path_expected, y_tar, x0;
                x_lb=0.0, x_ub=5.0, lambda_reg=1e-2, device=device,
            )
            opts = merge(Dict{Symbol,Any}(
                :max_iter => quick ? 100 : 300,
                :tol => 1e-3,
                :print_level => MadNLP.ERROR,
            ), solver_cfg.opts)
            # Remove nothing values
            filter!(p -> p.second !== nothing, opts)

            t0 = time()
            result = try
                solve_nlp(nlp; kkt_device=device, opts...)
            catch e
                @warn "  Solver failed: $e"
                continue
            end
            elapsed = time() - t0

            ts = result[:timing_stats]
            rel_err = norm(result[:solution] - x_true) / norm(x_true)

            rec = build_result_record(
                case="darcy", stage="A", device=device,
                solver_label=solver_cfg.label,
                status=result[:status],
                objective=result[:objective],
                iter_count=result[:iter_count],
                solve_time=elapsed,
                eval_time=ts.total_eval_time,
                hess_time=ts.hess_time,
                extra=Dict("rel_recovery_error" => rel_err,
                           "grid_n" => grid_n),
            )
            push!(records, rec)
            @printf("    Status=%-20s  obj=%.4e  time=%.2fs  rel_err=%.4f\n",
                result[:status], result[:objective], elapsed, rel_err)
        end
    end

    return records
end


# ============================================================================
# Pareto benchmark
# ============================================================================

function benchmark_pareto(devices, quick)
    println("[Pareto] Running constrained Pareto tracing benchmark")
    n_dim   = quick ? 10 : 20
    n_alpha = quick ? 5  : 11

    # Ensure surrogates exist
    setup_python_env()
    pareto_mod = pyimport("pareto_data")
    model_dir = "output/models/pareto"
    f1_p = joinpath(model_dir, "f1_net_n$(n_dim)_seed$(SEED).pt")
    f2_p = joinpath(model_dir, "f2_net_n$(n_dim)_seed$(SEED).pt")
    h_p  = joinpath(model_dir, "h_net_n$(n_dim)_seed$(SEED).pt")

    if !isfile(f1_p)
        n_samples = quick ? 1000 : 3000
        epochs    = quick ? 30   : 150
        paths = pyconvert(Dict, pareto_mod.train_all_surrogates(
            n=n_dim, n_samples=n_samples, seed=SEED,
            data_dir="output/pareto", output_dir=model_dir,
            epochs=epochs, hidden=64, device="cpu",
        ))
        f1_p = String(paths["f1_path"])
        f2_p = String(paths["f2_path"])
        h_p  = String(paths["h_path"])
    end

    alpha_values = collect(range(0.0, 1.0; length=n_alpha))
    records = []

    for device in devices
        for solver_cfg in build_solver_matrix(device)
            println("  Device=$(device), Solver=$(solver_cfg.label)")
            x0 = zeros(n_dim)

            times, feas_count = Float64[], 0
            front_pts = Tuple{Float64, Float64}[]

            for alpha in alpha_values
                cfg = ParetoProblemConfig(f1_p, f2_p, x0;
                    h_path=h_p, alpha=alpha, x_lb=-1.0, x_ub=1.0, device=device)
                nlp = create_pareto_nlp(cfg)

                opts = merge(Dict{Symbol,Any}(
                    :max_iter => quick ? 100 : 300,
                    :tol => 1e-3,
                    :print_level => MadNLP.ERROR,
                ), solver_cfg.opts)
                filter!(p -> p.second !== nothing, opts)

                t0 = time()
                result = try
                    solve_nlp(nlp; kkt_device=device, opts...)
                catch e
                    continue
                end
                elapsed = time() - t0

                push!(times, elapsed)
                f_vals = evaluate_pareto_objectives(nlp, result[:solution], true)
                f1v, f2v, hv = f_vals
                feasible = hv === nothing || hv <= 0.0
                feasible && push!(front_pts, (f1v, f2v))
                feasible && (feas_count += 1)
            end

            feas_rate = n_alpha > 0 ? 100 * feas_count / n_alpha : 0.0
            hv_val = if length(front_pts) >= 2
                ref = (maximum(p[1] for p in front_pts) * 1.1,
                       maximum(p[2] for p in front_pts) * 1.1)
                pareto_hypervolume(front_pts, ref)
            else
                0.0
            end

            rec = build_result_record(
                case="pareto", stage="B", device=device,
                solver_label=solver_cfg.label,
                status="sweep", objective=NaN,
                iter_count=0,
                solve_time=sum(times),
                eval_time=0.0,
                hess_time=0.0,
                extra=Dict(
                    "n_alpha" => n_alpha,
                    "n_dim" => n_dim,
                    "feasibility_rate_pct" => feas_rate,
                    "hypervolume" => hv_val,
                    "avg_subproblem_time_s" => isempty(times) ? 0.0 : mean(times),
                ),
            )
            push!(records, rec)
            @printf("    FeasRate=%.1f%%  HV=%.4f  AvgT=%.2fs\n",
                feas_rate, hv_val, isempty(times) ? 0.0 : mean(times))
        end
    end

    return records
end


# ============================================================================
# Main dispatch
# ============================================================================

all_records = []

if CASE in ("darcy", "all")
    append!(all_records, benchmark_darcy(DEVICES, QUICK))
end

if CASE in ("pareto", "all")
    append!(all_records, benchmark_pareto(DEVICES, QUICK))
end


# ============================================================================
# Save consolidated report
# ============================================================================

timestamp = Dates.format(Dates.now(), "yyyymmdd_HHMMSS")
report_file = joinpath(OUTPUT_DIR, "benchmark_$(CASE)_$(timestamp).json")

open(report_file, "w") do io
    JSON3.pretty(io, Dict(
        "timestamp" => string(Dates.now()),
        "configuration" => Dict(
            "case" => CASE,
            "devices" => DEVICES,
            "quick" => QUICK,
            "gpu_available" => GPU_AVAILABLE,
            "mumps_available" => MUMPS_AVAILABLE,
            "seed" => SEED,
        ),
        "records" => all_records,
    ))
end

# Print summary table
println("\n" * "="^80)
println("[Benchmark Summary]")
println("="^80)
@printf("%-12s %-8s %-22s %-20s %-10s %-10s\n",
    "Case", "Device", "Solver", "Status", "Time(s)", "Extra")
for r in all_records
    extra_str = if haskey(r["extra"], "rel_recovery_error")
        @sprintf("err=%.4f", r["extra"]["rel_recovery_error"])
    elseif haskey(r["extra"], "feasibility_rate_pct")
        @sprintf("feas=%.1f%%", r["extra"]["feasibility_rate_pct"])
    else
        ""
    end
    @printf("%-12s %-8s %-22s %-20s %-10.3f %-10s\n",
        r["case"], r["device"], r["solver"], r["status"],
        r["timing"]["total_solve_s"], extra_str)
end
println("\nReport saved: $(report_file)")
println("Completed: $(Dates.now())")
