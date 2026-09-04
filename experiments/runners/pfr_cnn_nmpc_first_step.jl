#!/usr/bin/env julia

"""First-step solver paths for published PFR CNN-NMPC cases 2 and 3."""

using ArgParse
using CUDA
using Dates
using JSON3
using LinearAlgebra
using MadNLP
using MadNLP4NN
using NLPModels
using PythonCall
using Sockets
using Statistics


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
        "--max_iter"
            arg_type = Int
            default = 500
        "--tol"
            arg_type = Float64
            default = 1e-6
        "--methods"
            default = "exact_cpu,exact_dlpack_gpu,lbfgs_cpu"
    end
    return parse_args(settings)
end


function make_model(evaluator, initial_guess)
    n = pyconvert(Int, evaluator.n)
    m = pyconvert(Int, evaluator.m)
    lower = pyconvert(Vector{Float64}, evaluator.variable_lower)
    upper = pyconvert(Vector{Float64}, evaluator.variable_upper)
    meta = NLPModelMeta(
        n;
        x0=copy(initial_guess),
        lvar=lower,
        uvar=upper,
        ncon=m,
        lcon=zeros(m),
        ucon=zeros(m),
        nnzj=pyconvert(Int, evaluator.jacobian_nnz),
        nnzh=pyconvert(Int, evaluator.hessian_nnz),
        minimize=true,
    )
    return NeuralNetworkNLPModel(
        meta,
        NLPModels.Counters(),
        nothing,
        NeuralNetworkObjective(Float64[]),
        nothing,
        copy(initial_guess),
        true,
        evaluator,
        TimingStats(),
        :forwarddiff,
    )
end


function solve_method(evaluator, initial_guess, method, max_iter, tol)
    inner = make_model(evaluator, initial_guess)
    CUDA.synchronize()
    result = nothing
    elapsed = @elapsed begin
        if method == "exact_cpu"
            result = solve_nlp(
                inner;
                kkt_device="cpu",
                linear_solver=MadNLP.MumpsSolver,
                max_iter=max_iter,
                tol=tol,
                print_level=MadNLP.ERROR,
            )
        elseif method == "exact_dlpack_gpu"
            result = solve_nlp(
                DLPackGPUModel(inner; verify_interop=false);
                max_iter=max_iter,
                tol=tol,
                print_level=MadNLP.ERROR,
            )
        elseif method == "lbfgs_cpu"
            result = solve_nlp(
                inner;
                kkt_device="cpu",
                linear_solver=MadNLP.MumpsSolver,
                hessian_approximation=MadNLP.CompactLBFGS,
                quasi_newton_options=MadNLP.QuasiNewtonOptions(; max_history=20),
                max_iter=max_iter,
                tol=tol,
                print_level=MadNLP.ERROR,
            )
        else
            error("Unknown method " * method)
        end
        CUDA.synchronize()
    end
    return result, elapsed
end


function record_result(method, result, elapsed, evaluator, published_control)
    unpacked = evaluator.unpack(result[:solution])
    controls = pyconvert(Matrix{Float64}, unpacked[1])
    first_control = vec(controls[1, :])
    constraint_inf = norm(
        pyconvert(Vector{Float64}, evaluator.evaluate_cons(result[:solution])), Inf,
    )
    timing = result[:timing_stats]
    counters = result[:solver_counters]
    return Dict(
        "method" => method,
        "status" => string(result[:status]),
        "elapsed_s" => elapsed,
        "objective" => result[:objective],
        "iterations" => result[:iter_count],
        "primal_feas" => result[:primal_feas],
        "dual_feas" => result[:dual_feas],
        "complementarity" => result[:complementarity],
        "kkt_error" => result[:kkt_error],
        "raw_constraint_inf" => constraint_inf,
        "first_control" => first_control,
        "published_first_control" => published_control,
        "first_control_absolute_error" => abs.(first_control - published_control),
        "callback_time_s" => timing.total_eval_time,
        "jacobian_time_s" => timing.jac_time,
        "hessian_time_s" => timing.hess_time,
        "madnlp_init_time_s" => counters.init_time,
        "madnlp_linear_solver_time_s" => counters.linear_solver_time,
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


function main()
    args = parse_cli()
    config = JSON3.read(read(args["config"], String))
    methods = String.(split(args["methods"], ','))
    setup_python_env()
    module_ = pyimport("jax_nn_evaluator")
    case_data = module_.load_pfr_cnn_published_case(
        String(config["published_output"]), Int(config["case"]),
    )
    current_state = pyconvert(Vector{Float64}, case_data["current_state"])
    previous_control = pyconvert(Vector{Float64}, case_data["previous_control"])
    published_control = pyconvert(
        Vector{Float64}, case_data["published_first_control"],
    )
    evaluator = module_.create_pfr_cnn_nmpc_evaluator(
        String(config["model"]),
        current_state,
        previous_control,
        Int.(config["tracking_channels"]),
        Int.(config["tracking_indices"]),
        Float64.(config["tracking_targets"]),
        Float64.(config["tracking_scales"]),
        Float64.(config["control_scales"]);
        prediction_horizon=Int(config["prediction_horizon"]),
        control_horizon=Int(config["control_horizon"]),
        device="gpu",
    )
    initial_guess = pyconvert(Vector{Float64}, evaluator.initial_guess())
    results = Any[]
    for method in methods
        println("PFR CNN case=", config["case"], " method=", method)
        cold_result, cold_elapsed = solve_method(
            evaluator, initial_guess, method, args["max_iter"], args["tol"],
        )
        cold = record_result(
            method, cold_result, cold_elapsed, evaluator, published_control,
        )
        warm = Any[]
        for _ in 1:args["repetitions"]
            result, elapsed = solve_method(
                evaluator, initial_guess, method, args["max_iter"], args["tol"],
            )
            push!(warm, record_result(
                method, result, elapsed, evaluator, published_control,
            ))
        end
        successful = [
            record for record in warm
            if record["status"] in ("SOLVE_SUCCEEDED", "SOLVED_TO_ACCEPTABLE_LEVEL") &&
               record["kkt_error"] <= 1e-6 &&
               record["raw_constraint_inf"] <= 1e-6
        ]
        push!(results, Dict(
            "method" => method,
            "cold" => cold,
            "warm" => warm,
            "common_success_count" => length(successful),
            "median_elapsed_s" => isempty(successful) ? nothing : median(
                [record["elapsed_s"] for record in successful]
            ),
        ))
    end
    payload = Dict(
        "schema_version" => 1,
        "created_at" => string(Dates.now()),
        "hostname" => gethostname(),
        "config" => Dict(string(key) => value for (key, value) in pairs(config)),
        "n" => pyconvert(Int, evaluator.n),
        "m" => pyconvert(Int, evaluator.m),
        "jacobian_nnz" => pyconvert(Int, evaluator.jacobian_nnz),
        "hessian_nnz" => pyconvert(Int, evaluator.hessian_nnz),
        "repetitions" => args["repetitions"],
        "tol" => args["tol"],
        "common_audit_tol" => 1e-6,
        "results" => results,
    )
    mkpath(dirname(args["output"]))
    open(args["output"], "w") do io
        JSON3.pretty(io, json_safe(payload))
        write(io, '\n')
    end
    println("Wrote ", args["output"])
end


main()
