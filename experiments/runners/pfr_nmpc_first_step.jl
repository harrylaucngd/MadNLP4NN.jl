#!/usr/bin/env julia

"""Reproduce the first published PFR-NMPC case-study-1 gray-box solve."""

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
            default = "exact_cpu,exact_host_staged_gpu,exact_dlpack_gpu,lbfgs_cpu"
    end
    return parse_args(settings)
end


function load_published_controls(path)
    builtins = pyimport("builtins")
    pickle = pyimport("pickle")
    np = pyimport("numpy")
    handle = builtins.open(path, "rb")
    try
        payload = pickle.load(handle)
        flow = pyconvert(Vector{Float64}, np.asarray(payload["F"], dtype=np.float64))
        inlet = pyconvert(
            Vector{Float64}, np.asarray(payload["C_in"], dtype=np.float64),
        )
        return [flow[2], inlet[2]]
    finally
        handle.close()
    end
end


function make_model(evaluator, config, initial_guess)
    prediction_horizon = Int(config["prediction_horizon"])
    control_horizon = Int(config["control_horizon"])
    current_state = Float64.(config["current_state"])
    state_dimension = length(current_state)
    control_dimension = length(config["previous_control"])
    state_size = (prediction_horizon + 1) * state_dimension
    n = state_size + control_horizon * control_dimension
    m = prediction_horizon * state_dimension
    lower = vcat(
        fill(Float64(config["state_lower"]), state_size),
        fill(Float64(config["control_lower"]), n - state_size),
    )
    upper = vcat(
        fill(Float64(config["state_upper"]), state_size),
        fill(Float64(config["control_upper"]), n - state_size),
    )
    lower[1:state_dimension] .= current_state
    upper[1:state_dimension] .= current_state
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


function solve_method(evaluator, config, initial_guess, method, max_iter, tol)
    inner = make_model(evaluator, config, initial_guess)
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
        elseif method == "exact_host_staged_gpu"
            result = solve_nlp(
                inner;
                kkt_device="gpu",
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
    states = pyconvert(Matrix{Float64}, unpacked[0])
    controls = pyconvert(Matrix{Float64}, unpacked[1])
    first_control = vec(controls[1, :])
    counters = result[:solver_counters]
    timing = result[:timing_stats]
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
        "first_control" => first_control,
        "published_first_control" => published_control,
        "first_control_absolute_error" => abs.(first_control - published_control),
        "predicted_first_outlet" => states[2, end],
        "terminal_outlet" => states[end, end],
        "callback_time_s" => timing.total_eval_time,
        "jacobian_time_s" => timing.jac_time,
        "hessian_time_s" => timing.hess_time,
        "madnlp_total_time_s" => counters.total_time,
        "madnlp_init_time_s" => counters.init_time,
        "madnlp_eval_time_s" => counters.eval_function_time,
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
    factory = pyimport("jax_nn_evaluator").create_pfr_nmpc_evaluator
    evaluator = factory(
        String(config["model"]),
        Float64.(config["current_state"]),
        Float64.(config["previous_control"]),
        Float64(config["setpoint"]);
        prediction_horizon=Int(config["prediction_horizon"]),
        control_horizon=Int(config["control_horizon"]),
        device="gpu",
    )
    initial_guess = pyconvert(Vector{Float64}, evaluator.initial_guess())
    published_control = load_published_controls(String(config["published_output"]))
    results = Any[]
    for method in methods
        println("PFR first step method=", method)
        warmup_result, warmup_elapsed = solve_method(
            evaluator,
            config,
            initial_guess,
            method,
            args["max_iter"],
            args["tol"],
        )
        warmup = record_result(
            method, warmup_result, warmup_elapsed, evaluator, published_control,
        )
        warm = Any[]
        for _ in 1:args["repetitions"]
            result, elapsed = solve_method(
                evaluator,
                config,
                initial_guess,
                method,
                args["max_iter"],
                args["tol"],
            )
            push!(warm, record_result(
                method, result, elapsed, evaluator, published_control,
            ))
        end
        successful = [
            record for record in warm
            if record["status"] == "SOLVE_SUCCEEDED" &&
               record["kkt_error"] <= args["tol"]
        ]
        push!(results, Dict(
            "method" => method,
            "warmup" => warmup,
            "warm" => warm,
            "common_success_count" => length(successful),
            "median_elapsed_s" => (
                isempty(successful) ? nothing : median(
                    [record["elapsed_s"] for record in successful]
                )
            ),
        ))
    end
    payload = Dict(
        "schema_version" => 1,
        "created_at" => string(Dates.now()),
        "config" => Dict(string(key) => value for (key, value) in pairs(config)),
        "hostname" => gethostname(),
        "repetitions" => args["repetitions"],
        "max_iter" => args["max_iter"],
        "tol" => args["tol"],
        "initial_objective" => pyconvert(Float64, evaluator.evaluate_obj(initial_guess)),
        "initial_constraint_inf_norm" => norm(
            pyconvert(Vector{Float64}, evaluator.evaluate_cons(initial_guess)), Inf,
        ),
        "published_first_control" => published_control,
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
