#!/usr/bin/env julia
"""
run_darcy_inversion.jl

Case Study I (proposal §2): neural-operator inversion of 2-D Darcy flow.

Staged experiment driver:
  Stage A — box constraints only (correctness / backend cross-check)
  Stage B — add budget and smoothness constraints
  Stage C — scaling study across grid sizes

Usage
-----
    julia examples/run_darcy_inversion.jl [--stage A|B|C|all] [--device cpu|gpu]
        [--grid 64] [--backend jax] [--fno_path path/to/fno.npz]

The script generates synthetic Darcy data and trains an FNO surrogate if
no checkpoint is provided.  Results are written to
    output/darcy/inversion_results/
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

# ---- Optional GPU ----
const GPU_AVAILABLE = try
    @eval using MadNLPGPU
    true
catch
    false
end

# ---- Optional MUMPS ----
const MUMPS_AVAILABLE = try
    @eval using MadNLPMumps
    true
catch
    false
end

println("="^80)
println("MadNLP4NN — Case Study I: Darcy Flow FNO Inversion")
println("="^80)
println("Started: $(Dates.now())")
println()


# ============================================================================
# Argument parsing
# ============================================================================

function parse_cli_args()
    s = ArgParseSettings()
    @add_arg_table! s begin
        "--stage"
            help = "Which stage to run: A, B, C, or all"
            default = "A"
        "--grid"
            arg_type = Int
            default = 64
            help = "Grid size N (decision dim = N²)"
        "--device"
            default = "cpu"
            help = "Execution device: cpu or gpu"
        "--fno_path"
            default = ""
            help = "Path to trained FNO checkpoint.  Empty = auto-generate."
        "--data_dir"
            default = "output/darcy"
        "--output_dir"
            default = "output/darcy/inversion_results"
        "--max_iter"
            arg_type = Int
            default = 500
        "--tol"
            arg_type = Float64
            default = 1e-4
        "--lambda_reg"
            arg_type = Float64
            default = 1e-2
        "--seed"
            arg_type = Int
            default = 42
        "--n_train_samples"
            arg_type = Int
            default = 500
            help = "FNO training samples when generating data"
        "--fno_epochs"
            arg_type = Int
            default = 100
            help = "FNO training epochs when generating data"
        "--compare_backends"
            action = :store_true
            help = "Run both JAX and Julia/Flux and compare"
    end
    return ArgParse.parse_args(ARGS, s)
end

args = parse_cli_args()

const STAGE        = args["stage"]
const GRID_N       = args["grid"]
const DEVICE       = args["device"]
const FNO_PATH_ARG = args["fno_path"]
const DATA_DIR     = args["data_dir"]
const OUTPUT_DIR   = args["output_dir"]
const MAX_ITER     = args["max_iter"]
const TOLERANCE    = args["tol"]
const LAMBDA_REG   = args["lambda_reg"]
const RANDOM_SEED  = args["seed"]
const N_DIM        = GRID_N * GRID_N

Random.seed!(RANDOM_SEED)
mkpath(OUTPUT_DIR)

println("[Configuration]")
println("  Stage: $(STAGE), Grid: $(GRID_N)×$(GRID_N) (n=$(N_DIM))")
println("  Device: $(DEVICE), Max iter: $(MAX_ITER), tol: $(TOLERANCE)")
println("  GPU available: $(GPU_AVAILABLE)")
println()


# ============================================================================
# Step 1: Ensure FNO checkpoint exists
# ============================================================================

function ensure_fno(fno_path_arg::String, grid_n::Int, data_dir::String,
                    n_train::Int, fno_epochs::Int, seed::Int) :: String
    if !isempty(fno_path_arg) && isfile(fno_path_arg)
        println("[FNO] Using existing checkpoint: $(fno_path_arg)")
        return fno_path_arg
    end

    # Auto-generate data and train FNO via Python
    setup_python_env()
    darcy_module = pyimport("darcy_data")

    ckpt_dir = joinpath(data_dir, "models")
    # Pattern-match the expected filename
    expected = joinpath(ckpt_dir, "fno_darcy_$(grid_n)x$(grid_n)_dv32_seed$(seed).npz")
    if isfile(expected)
        println("[FNO] Found existing checkpoint: $(expected)")
        return expected
    end

    println("[FNO] Generating Darcy training data ($(n_train) samples)...")
    darcy_module.generate_darcy_dataset(
        n_samples=n_train, grid_n=grid_n, seed=seed, output_dir=data_dir, split="train"
    )
    darcy_module.generate_darcy_dataset(
        n_samples=n_train ÷ 5, grid_n=grid_n, seed=seed+1, output_dir=data_dir, split="test"
    )

    println("[FNO] Training FNO surrogate (epochs=$(fno_epochs))...")
    ckpt_path = pyconvert(String, darcy_module.train_fno(
        data_dir=data_dir,
        grid_n=grid_n,
        d_v=32,
        n_layers=4,
        k_max=min(16, grid_n ÷ 4),
        lr=1e-3,
        epochs=fno_epochs,
        batch_size=16,
        seed=seed,
        output_dir=ckpt_dir,
        device=DEVICE,
    ))
    println("[FNO] Checkpoint saved: $(ckpt_path)")
    return ckpt_path
end

fno_path = ensure_fno(
    FNO_PATH_ARG, GRID_N, DATA_DIR,
    args["n_train_samples"], args["fno_epochs"], RANDOM_SEED
)


# ============================================================================
# Step 2: Build a synthetic "true" permeability and target pressure
# ============================================================================

function make_synthetic_problem(fno_path::String, grid_n::Int, seed::Int)
    rng = Random.MersenneTwister(seed)

    # True permeability: smooth GRF-like field, bounded
    x_true = 0.5 .+ 0.3 .* randn(rng, grid_n * grid_n)
    x_true = clamp.(x_true, 0.1, 3.0)

    # Forward pass through FNO to get "observed" pressure
    setup_python_env()
    jax_mod = pyimport("jax_nn_evaluator")
    ev = jax_mod.create_darcy_evaluator(
        fno_path, zeros(grid_n * grid_n), x_true;
        regularization_weight=0.0, device=DEVICE,
    )
    y_tar = pyconvert(Vector{Float64}, ev.evaluate_nn(x_true))

    # Noisy initial point
    x0 = clamp.(x_true .+ 0.1 .* randn(rng, grid_n * grid_n), 0.1, 3.0)

    return x_true, y_tar, x0
end

println("[Setup] Building synthetic Darcy problem (grid=$(GRID_N)×$(GRID_N))...")
x_true, y_tar, x0 = make_synthetic_problem(fno_path, GRID_N, RANDOM_SEED)
println("  True field  : mean=$(round(mean(x_true), digits=4)), " *
        "‖x_true‖=$(round(norm(x_true), digits=2))")
println("  Target pres : ‖y_tar‖=$(round(norm(y_tar), digits=4))")
println("  Initial x0  : ‖x0 - x_true‖=$(round(norm(x0 - x_true), digits=4))")
println()


# ============================================================================
# Step 3: Solve helper
# ============================================================================

function run_darcy_solve(
    nlp, label::String;
    max_iter=MAX_ITER, tol=TOLERANCE, device=DEVICE,
    x_true_ref=x_true, y_tar_ref=y_tar
)
    println("  Solving: $(label)")
    solver_opts = Dict{Symbol,Any}(
        :max_iter => max_iter,
        :tol => tol,
        :print_level => MadNLP.ERROR,
    )

    ls = select_linear_solver(device)
    ls !== nothing && (solver_opts[:linear_solver] = ls)

    MUMPS_AVAILABLE && device == "cpu" &&
        (solver_opts[:linear_solver] = MadNLPMumps.MumpsSolver)

    t0 = time()
    result = solve_nlp(nlp; solver_opts...)
    elapsed = time() - t0

    x_sol = result[:solution]
    ts = result[:timing_stats]

    rel_err = norm(x_sol .- x_true_ref) / norm(x_true_ref)
    println("    Status:      $(result[:status])")
    println("    Objective:   $(@sprintf("%.6e", result[:objective]))")
    println("    Iterations:  $(result[:iter_count])")
    println("    Solve time:  $(@sprintf("%.3f", elapsed))s")
    println("    Eval %:      $(@sprintf("%.1f", 100*ts.total_eval_time/elapsed))%")
    println("    Rel recovery: $(@sprintf("%.4f", rel_err))")
    println()

    return Dict(
        :label => label,
        :status => string(result[:status]),
        :objective => result[:objective],
        :iter_count => result[:iter_count],
        :solve_time => elapsed,
        :eval_time => ts.total_eval_time,
        :hess_time => ts.hess_time,
        :rel_recovery_error => rel_err,
        :solution => x_sol,
        :x_true => x_true_ref,
        :y_tar => y_tar_ref,
    )
end


# ============================================================================
# Stage A: box-only, correctness and backend cross-check
# ============================================================================

function run_stage_a()
    println("="^60)
    println("[Stage A] Box constraints only — correctness verification")
    println("="^60)

    nlp = create_darcy_stage_a(
        fno_path, y_tar, x0;
        x_lb=0.0, x_ub=5.0,
        lambda_reg=LAMBDA_REG,
        device=DEVICE,
    )
    result = run_darcy_solve(nlp, "Stage A (JAX, box only)")

    # Verify constraints
    x_sol = result[:solution]
    box_ok = all(0.0 .<= x_sol .<= 5.0)
    println("  Box constraints satisfied: $(box_ok ? "✓" : "✗")")

    return result
end


# ============================================================================
# Stage B: box + budget + smoothness
# ============================================================================

function run_stage_b()
    println("="^60)
    println("[Stage B] Box + Budget + Smoothness constraints")
    println("="^60)

    budget = sum(x_true) * 1.1  # slack relative to true field
    tau    = dot(x_true, Matrix(laplacian_2d(GRID_N)) * x_true) * 1.5

    println("  Budget B = $(@sprintf("%.2f", budget))")
    println("  Smoothness τ = $(@sprintf("%.2f", tau))")

    nlp = create_darcy_stage_b(
        fno_path, y_tar, x0;
        x_lb=0.0, x_ub=5.0,
        lambda_reg=LAMBDA_REG,
        budget=budget,
        tau=tau,
        grid_nx=GRID_N,
        device=DEVICE,
    )
    result = run_darcy_solve(nlp, "Stage B (JAX, box + budget + smoothness)")

    # Verify constraints
    x_sol = result[:solution]
    L = Matrix(laplacian_2d(GRID_N))
    println("  Box:       $(all(0 .<= x_sol .<= 5) ? "✓" : "✗")")
    println("  Budget:    1ᵀx = $(@sprintf("%.2f", sum(x_sol))) ≤ $(@sprintf("%.2f", budget)) $(sum(x_sol) <= budget + 1e-4 ? "✓" : "✗")")
    println("  Smoothness: xᵀLx = $(@sprintf("%.2e", dot(x_sol, L * x_sol))) ≤ $(@sprintf("%.2e", tau)) $(dot(x_sol, L * x_sol) <= tau * 1.01 ? "✓" : "✗")")
    println()

    return result
end


# ============================================================================
# Stage C: scaling study
# ============================================================================

function run_stage_c()
    println("="^60)
    println("[Stage C] Scaling study across grid sizes")
    println("="^60)

    grid_sizes = [16, 32, 64]
    GRID_N > 64 && push!(grid_sizes, GRID_N)
    unique!(sort!(grid_sizes))
    filter!(g -> g <= GRID_N, grid_sizes)

    scaling_results = []

    for gn in grid_sizes
        println("  Grid: $(gn)×$(gn) (n=$(gn^2))")

        # Build a minimal problem for this grid
        fno_g = ensure_fno("", gn, DATA_DIR, 200, 50, RANDOM_SEED)
        x_t, y_t, x_i = make_synthetic_problem(fno_g, gn, RANDOM_SEED)

        nlp = create_darcy_stage_a(
            fno_g, y_t, x_i;
            x_lb=0.0, x_ub=5.0,
            lambda_reg=LAMBDA_REG,
            device=DEVICE,
        )
        res = run_darcy_solve(
            nlp,
            "Stage C, grid $(gn)×$(gn)";
            max_iter=200,
            tol=1e-3,
            x_true_ref=x_t,
            y_tar_ref=y_t,
        )
        ts = nlp.timing_stats
        push!(scaling_results, Dict(
            :grid_n => gn,
            :n => gn^2,
            :solve_time => res[:solve_time],
            :eval_frac => res[:eval_time] / res[:solve_time],
            :hess_frac => ts.hess_time / max(res[:solve_time], 1e-6),
            :rel_err => res[:rel_recovery_error],
        ))
    end

    println("\n  Scaling summary:")
    println("  Grid    n       Time[s]  Eval%  Hess%   RelErr")
    for r in scaling_results
        @printf("  %4d  %6d  %7.2f  %5.1f  %5.1f  %.4f\n",
            r[:grid_n], r[:n], r[:solve_time],
            100 * r[:eval_frac], 100 * r[:hess_frac],
            r[:rel_err])
    end

    return scaling_results
end


# ============================================================================
# Run stages
# ============================================================================

all_results = Dict{String, Any}()

stages_to_run = if STAGE == "all"
    ["A", "B", "C"]
else
    [STAGE]
end

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
    "darcy_stage$(STAGE)_grid$(GRID_N)_$(DEVICE)_$(timestamp).json")

# Serialize only scalar / metadata parts (omit large solution vectors for summary)
serializable = Dict{String, Any}()
for (k, v) in all_results
    if v isa Dict
        serializable[k] = Dict(
            string(sk) => (sv isa AbstractVector ? "(length=$(length(sv)))" : sv)
            for (sk, sv) in v
            if !(sk in (:solution, :x_true, :y_tar))
        )
    else
        serializable[k] = v
    end
end

open(result_file, "w") do io
    JSON3.pretty(io, Dict(
        "timestamp" => string(Dates.now()),
        "configuration" => Dict(
            "stage" => STAGE,
            "grid_n" => GRID_N,
            "n" => N_DIM,
            "device" => DEVICE,
            "fno_path" => fno_path,
            "max_iter" => MAX_ITER,
            "tol" => TOLERANCE,
            "lambda_reg" => LAMBDA_REG,
        ),
        "results" => serializable,
    ))
end

println("\nResults saved: $(result_file)")
println("Completed: $(Dates.now())")
println("="^80)
