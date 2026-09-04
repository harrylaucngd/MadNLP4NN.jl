#!/usr/bin/env julia

"""Compare CPU KKT, host-staged GPU KKT, and DLPack GPU KKT paths.

The planted smooth MLP problem is a controlled systems benchmark. It is not an
application result. All paths share the same JAX evaluator, initial point,
bounds, constraint, tolerances, and curvature setting.
"""

using ArgParse
using CUDA
using JSON3
using MadNLP
using MadNLP4NN
using MadNLPGPU
using NLPModels
using PythonCall
using Random
using Statistics
using Dates
using Sockets


function parse_cli()
    settings = ArgParseSettings()
    @add_arg_table! settings begin
        "--config"
            required = true
        "--output"
            required = true
        "--repetitions"
            arg_type = Int
            default = 3
        "--seed"
            arg_type = Int
            default = 20260821
        "--max_iter"
            arg_type = Int
            default = 200
        "--tol"
            arg_type = Float64
            default = 1e-6
        "--paths"
            default = "gpu_ad_cpu_kkt,host_staged_gpu_kkt,dlpack_gpu_kkt"
        "--curvatures"
            default = "exact,compact_lbfgs"
        "--lbfgs_history"
            arg_type = Int
            default = 20
    end
    return parse_args(settings)
end


function python_mlp_builder()
    namespace = PythonCall.pybuiltins.dict()
    PythonCall.pybuiltins.exec(
        """
import jax.numpy as jnp

def forward(params, x):
    value = x
    for index, layer in enumerate(params):
        weight, bias = layer
        value = weight @ value + bias
        if index + 1 < len(params):
            value = jnp.tanh(value)
    return value
""",
        namespace,
    )
    return namespace["forward"]
end


struct ProblemTemplate
    evaluator::Py
    n::Int
    m_out::Int
    parameter_count::Int
    x0::Vector{Float64}
    target::Vector{Float64}
    radius::Float64
end


function build_template(config, seed::Int)
    n = Int(config["n"])
    m_out = Int(config["m_out"])
    width = Int(config["width"])
    depth = Int(config["depth"])
    rng = Random.MersenneTwister(seed)
    dm = pyimport("backend.device").DeviceManager("gpu")
    np = pyimport("numpy")
    forward = python_mlp_builder()

    dimensions = [n; fill(width, depth); m_out]
    params = PythonCall.pybuiltins.list()
    parameter_count = 0
    for (d_in, d_out) in zip(dimensions[1:end-1], dimensions[2:end])
        weight = randn(rng, d_out, d_in) ./ sqrt(d_in)
        bias = 0.05 .* randn(rng, d_out)
        params.append((dm.array(np.asarray(weight)), dm.array(np.asarray(bias))))
        parameter_count += d_out * d_in + d_out
    end

    x0 = zeros(n)
    x_star = 0.15 .* randn(rng, n)
    radius = 0.5 * sqrt(n)
    target_py = forward(params, dm.array(np.asarray(x_star)))
    target_py.block_until_ready()
    target = pyconvert(Vector{Float64}, np.asarray(target_py))
    evaluator = pyimport("evaluators.standard").StandardEvaluator(
        forward,
        params,
        target,
        x0;
        regularization_weight=1e-3,
        radius=radius,
        device_manager=dm,
    )
    return ProblemTemplate(evaluator, n, m_out, parameter_count, x0, target, radius)
end


function fresh_inner(template::ProblemTemplate)
    n = template.n
    meta = NLPModelMeta(
        n;
        x0=copy(template.x0),
        lvar=fill(-1.0, n),
        uvar=fill(1.0, n),
        ncon=1,
        lcon=[-Inf],
        ucon=[0.0],
        nnzj=n,
        nnzh=div(n * (n + 1), 2),
        minimize=true,
    )
    return NeuralNetworkNLPModel(
        meta,
        NLPModels.Counters(),
        nothing,
        NeuralNetworkObjective(template.target),
        SphericalConstraint(template.x0, template.radius),
        copy(template.x0),
        true,
        template.evaluator,
        TimingStats(),
        :forwarddiff,
    )
end


function curvature_options(curvature::String, lbfgs_history::Int)
    if curvature == "exact"
        return Dict{Symbol,Any}()
    elseif curvature == "compact_lbfgs"
        return Dict{Symbol,Any}(
            :hessian_approximation => MadNLP.CompactLBFGS,
            :quasi_newton_options => MadNLP.QuasiNewtonOptions(
                ; max_history=lbfgs_history,
            ),
        )
    end
    error("Unknown curvature: $curvature")
end


function solve_once(
    template::ProblemTemplate,
    path::String,
    curvature::String,
    max_iter::Int,
    tol::Float64,
    lbfgs_history::Int,
)
    inner = fresh_inner(template)
    options = curvature_options(curvature, lbfgs_history)
    result = nothing
    failure = nothing
    CUDA.synchronize()
    elapsed = @elapsed begin
        try
            if path == "gpu_ad_cpu_kkt"
                result = solve_nlp(
                    inner;
                    kkt_device="cpu",
                    linear_solver=MadNLP.MumpsSolver,
                    max_iter=max_iter,
                    tol=tol,
                    print_level=MadNLP.ERROR,
                    options...,
                )
            elseif path == "host_staged_gpu_kkt"
                result = solve_nlp(
                    inner;
                    kkt_device="gpu",
                    max_iter=max_iter,
                    tol=tol,
                    print_level=MadNLP.ERROR,
                    options...,
                )
            elseif path == "dlpack_gpu_kkt"
                gpu_model = DLPackGPUModel(inner; verify_interop=false)
                result = solve_nlp(
                    gpu_model;
                    max_iter=max_iter,
                    tol=tol,
                    print_level=MadNLP.ERROR,
                    options...,
                )
            else
                error("Unknown path: $path")
            end
        catch exception
            failure = sprint(showerror, exception, catch_backtrace())
        end
        CUDA.synchronize()
    end

    kkt_system = if curvature == "compact_lbfgs" || path == "gpu_ad_cpu_kkt"
        "SparseKKTSystem"
    else
        "SparseCondensedKKTSystem"
    end
    linear_solver = path == "gpu_ad_cpu_kkt" ? "MumpsSolver" : "CUDSSSolver"
    if failure !== nothing
        return Dict(
            "path" => path,
            "curvature" => curvature,
            "kkt_system" => kkt_system,
            "linear_solver" => linear_solver,
            "elapsed_s" => elapsed,
            "status" => "EXCEPTION",
            "error" => failure,
        )
    end

    counters = result[:solver_counters]
    timing = result[:timing_stats]
    return Dict(
        "path" => path,
        "curvature" => curvature,
        "kkt_system" => kkt_system,
        "linear_solver" => linear_solver,
        "elapsed_s" => elapsed,
        "status" => string(result[:status]),
        "objective" => result[:objective],
        "iterations" => result[:iter_count],
        "primal_feas" => result[:primal_feas],
        "dual_feas" => result[:dual_feas],
        "complementarity" => result[:complementarity],
        "kkt_error" => result[:kkt_error],
        "solution_norm" => sqrt(sum(abs2, result[:solution])),
        "callback_time_s" => timing.total_eval_time,
        "objective_time_s" => timing.obj_time,
        "gradient_time_s" => timing.grad_time,
        "constraint_time_s" => timing.cons_time,
        "jacobian_time_s" => timing.jac_time,
        "hessian_time_s" => timing.hess_time,
        "madnlp_total_time_s" => counters.total_time,
        "madnlp_init_time_s" => counters.init_time,
        "madnlp_eval_time_s" => counters.eval_function_time,
        "madnlp_linear_solver_time_s" => counters.linear_solver_time,
        "factorizations" => counters.factorization_cnt,
        "backsolves" => counters.backsolve_cnt,
        "objective_evaluations" => counters.obj_cnt,
        "gradient_evaluations" => counters.obj_grad_cnt,
        "constraint_evaluations" => counters.con_cnt,
        "jacobian_evaluations" => counters.con_jac_cnt,
        "hessian_evaluations" => counters.lag_hess_cnt,
    )
end


function json_safe(value)
    if value isa AbstractFloat
        return isfinite(value) ? value : nothing
    elseif value isa AbstractDict
        return Dict(string(key) => json_safe(item) for (key, item) in pairs(value))
    elseif value isa AbstractVector
        return [json_safe(item) for item in value]
    end
    return value
end


function summarize_repetitions(records)
    elapsed = [Float64(record["elapsed_s"]) for record in records]
    return Dict(
        "median_elapsed_s" => median(elapsed),
        "q1_elapsed_s" => quantile(elapsed, 0.25),
        "q3_elapsed_s" => quantile(elapsed, 0.75),
        "successes" => count(record -> record["status"] == "SOLVE_SUCCEEDED", records),
        "repetitions" => length(records),
    )
end


function main()
    args = parse_cli()
    args["repetitions"] > 0 || error("--repetitions must be positive")
    CUDA.functional() || error("A functional CUDA device is required")
    setup_python_env()
    configs = JSON3.read(read(args["config"], String))
    paths = String.(split(args["paths"], ','))
    curvatures = String.(split(args["curvatures"], ','))

    all_results = Any[]
    for (config_index, config) in enumerate(configs)
        println(
            "[", config_index, "/", length(configs), "] ", config["name"],
            ": n=", config["n"], " width=", config["width"],
            " depth=", config["depth"],
        )
        template = build_template(config, args["seed"] + config_index - 1)

        for curvature in curvatures, path in paths
            if curvature == "compact_lbfgs" && path != "gpu_ad_cpu_kkt"
                reason = (
                    "MadNLP 0.10/MadNLPGPU 0.10.2 CompactLBFGS is not " *
                    "compatible with the default SparseCondensedKKTSystem; " *
                    "the GPU SparseKKT and DenseKKT fallbacks also contain " *
                    "CPU-only/scalar paths. Recorded as unsupported pending " *
                    "an upstream-quality implementation."
                )
                push!(all_results, Dict(
                    "config" => Dict(string(key) => value for (key, value) in pairs(config)),
                    "parameter_count" => template.parameter_count,
                    "path" => path,
                    "curvature" => curvature,
                    "cold" => Dict("status" => "UNSUPPORTED", "reason" => reason),
                    "warm" => Any[],
                    "summary" => Dict(
                        "median_elapsed_s" => nothing,
                        "successes" => 0,
                        "repetitions" => 0,
                    ),
                ))
                continue
            end
            # First solve warms JAX shapes, Julia specializations, and the linear
            # solver. It is retained as cold data but excluded from warm medians.
            println("  warmup path=", path, " curvature=", curvature)
            cold = solve_once(
                template, path, curvature, args["max_iter"], args["tol"],
                args["lbfgs_history"],
            )
            warm = Any[]
            for repetition in 1:args["repetitions"]
                println("    repetition ", repetition, "/", args["repetitions"])
                push!(warm, solve_once(
                    template, path, curvature, args["max_iter"], args["tol"],
                    args["lbfgs_history"],
                ))
            end
            push!(all_results, Dict(
                "config" => Dict(string(key) => value for (key, value) in pairs(config)),
                "parameter_count" => template.parameter_count,
                "path" => path,
                "curvature" => curvature,
                "cold" => cold,
                "warm" => warm,
                "summary" => summarize_repetitions(warm),
            ))
        end
    end

    payload = Dict(
        "schema_version" => 1,
        "created_at" => string(Dates.now()),
        "configuration_file" => abspath(args["config"]),
        "repetitions" => args["repetitions"],
        "seed" => args["seed"],
        "max_iter" => args["max_iter"],
        "tol" => args["tol"],
        "lbfgs_history" => args["lbfgs_history"],
        "provenance" => Dict(
            "hostname" => gethostname(),
            "julia" => string(VERSION),
            "madnlp" => string(pkgversion(MadNLP)),
            "madnlpgpu" => string(pkgversion(MadNLPGPU)),
            "cuda" => string(pkgversion(CUDA)),
            "python" => pyconvert(String, pyimport("sys").version),
            "jax" => pyconvert(String, pyimport("jax").__version__),
            "cuda_device" => string(CUDA.device()),
            "cpu_affinity" => read("/proc/self/status", String),
            "environment" => Dict(
                key => get(ENV, key, nothing)
                for key in (
                    "CUDA_VISIBLE_DEVICES",
                    "JULIA_NUM_THREADS",
                    "OPENBLAS_NUM_THREADS",
                    "OMP_NUM_THREADS",
                    "MKL_NUM_THREADS",
                    "XLA_PYTHON_CLIENT_PREALLOCATE",
                )
            ),
        ),
        "results" => all_results,
    )
    output = args["output"]
    mkpath(dirname(output))
    open(output, "w") do io
        JSON3.pretty(io, json_safe(payload))
        write(io, '\n')
    end
    println("Wrote ", output)
end


main()
