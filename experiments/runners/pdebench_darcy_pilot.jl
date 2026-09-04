#!/usr/bin/env julia

"""Matched solver pilot on held-out latent-field FNO inverse instances."""

using ArgParse
using CUDA
using JSON3
using MadNLP
using MadNLP4NN
using NLPModels
using PythonCall
using Statistics
using Dates
using LinearAlgebra
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
            default = 1
        "--max_iter"
            arg_type = Int
            default = 200
        "--tol"
            arg_type = Float64
            default = 1e-6
        "--methods"
            default = "exact_cpu,exact_host_staged_gpu,exact_dlpack_gpu,lbfgs_cpu"
        "--starts"
            default = ""
        "--regularization_weight"
            arg_type = Float64
            default = NaN
    end
    return parse_args(settings)
end


function load_instance(data_path::String, sample::Int)
    np = pyimport("numpy")
    if endswith(data_path, ".pt")
        torch = pyimport("torch")
        payload = torch.load(data_path, map_location="cpu", weights_only=false)
        coefficient = pyconvert(
            Matrix{Float64},
            np.asarray(payload["x"][sample].numpy(), dtype=np.float64),
        )
        pressure = pyconvert(
            Matrix{Float64},
            np.asarray(payload["y"][sample].numpy(), dtype=np.float64),
        )
        return coefficient, pressure
    end
    h5py = pyimport("h5py")
    handle = h5py.File(data_path, "r")
    try
        coefficient = pyconvert(
            Matrix{Float64},
            np.asarray(handle["nu"][sample], dtype=np.float64),
        )
        pressure = pyconvert(
            Matrix{Float64},
            np.asarray(handle["tensor"][sample, 0], dtype=np.float64),
        )
        return coefficient, pressure
    finally
        handle.close()
    end
end


"""Flatten a Julia matrix in the row-major order expected by NumPy/JAX."""
row_major_vec(field::AbstractMatrix) = vec(permutedims(field))


function starts_for_resolution(registry, latent_resolution, reference)
    if registry === nothing
        return [Dict(
            "id" => "constant",
            "sha256" => nothing,
            "values" => copy(reference),
        )]
    end
    entries = registry["starts"][string(latent_resolution)]
    return [
        Dict(
            "id" => String(entry["id"]),
            "sha256" => String(entry["sha256"]),
            "values" => Float64.(entry["values"]),
        )
        for entry in entries
    ]
end


function make_inner(
    evaluator::Py,
    latent_resolution::Int,
    x0::Vector{Float64},
    lower_bound::Float64,
    upper_bound::Float64,
)
    n = latent_resolution^2
    m = pyconvert(Int, evaluator.m)
    meta = NLPModelMeta(
        n;
        x0=copy(x0),
        lvar=fill(lower_bound, n),
        uvar=fill(upper_bound, n),
        ncon=m,
        lcon=fill(-Inf, m),
        ucon=zeros(m),
        nnzj=m * n,
        nnzh=div(n * (n + 1), 2),
        minimize=true,
    )
    return NeuralNetworkNLPModel(
        meta,
        NLPModels.Counters(),
        nothing,
        NeuralNetworkObjective(Float64[]),
        nothing,
        copy(x0),
        true,
        evaluator,
        TimingStats(),
        :forwarddiff,
    )
end


function solve_method(
    evaluator::Py,
    latent_resolution::Int,
    x0::Vector{Float64},
    method::String,
    max_iter::Int,
    tol::Float64,
    lower_bound::Float64,
    upper_bound::Float64,
)
    if startswith(method, "gauss_newton")
        evaluator.set_hessian_mode("gauss_newton")
    else
        evaluator.set_hessian_mode("exact")
    end
    inner = make_inner(
        evaluator, latent_resolution, x0, lower_bound, upper_bound,
    )
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
        elseif method == "gauss_newton_cpu"
            result = solve_nlp(
                inner;
                kkt_device="cpu",
                linear_solver=MadNLP.MumpsSolver,
                max_iter=max_iter,
                tol=tol,
                print_level=MadNLP.ERROR,
            )
        elseif method == "gauss_newton_dlpack_gpu"
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
            error("Unknown method $method")
        end
        CUDA.synchronize()
    end
    return result, elapsed
end


function metrics(
    evaluator::Py,
    solution,
    truth_coefficient,
    target_pressure,
    lower_bound::Float64,
    upper_bound::Float64,
    simulator_replay::Bool,
)
    field = pyconvert(Matrix{Float64}, evaluator.evaluate_field(solution))
    prediction = pyconvert(
        Matrix{Float64},
        evaluator.evaluate_nn(solution).reshape(size(target_pressure)...),
    )
    constraints = pyconvert(Vector{Float64}, evaluator.evaluate_cons(solution))
    result = Dict(
        "coefficient_relative_l2" => norm(field - truth_coefficient) / norm(truth_coefficient),
        "coefficient_rmse" => sqrt(sum(abs2, field - truth_coefficient) / length(field)),
        "binary_accuracy" => mean(
            (field .>= (lower_bound + upper_bound) / 2) .==
            (truth_coefficient .>= (lower_bound + upper_bound) / 2)
        ),
        "binaryness" => mean(
            ((field .- lower_bound) .* (field .- upper_bound)) .^ 2
        ),
        "surrogate_target_relative_l2" => (
            norm(prediction - target_pressure) / norm(target_pressure)
        ),
        "surrogate_target_rmse" => sqrt(
            sum(abs2, prediction - target_pressure) / length(prediction)
        ),
        "field_mean" => mean(field),
        "field_min" => minimum(field),
        "field_max" => maximum(field),
        "max_constraint_violation" => isempty(constraints) ? 0.0 : max(0.0, maximum(constraints)),
    )
    if simulator_replay
        simulator = pyimport("darcy_data")
        replay = nothing
        replay_time = @elapsed replay = pyconvert(
            Matrix{Float64}, simulator.darcy_fd_solve(field),
        )
        result["simulator_target_relative_l2"] = norm(replay - target_pressure) / norm(target_pressure)
        result["simulator_target_rmse"] = sqrt(
            sum(abs2, replay - target_pressure) / length(target_pressure)
        )
        result["simulator_replay_time_s"] = replay_time
    end
    return result
end


function record_result(
    method,
    result,
    elapsed,
    evaluator,
    truth,
    target,
    lower_bound,
    upper_bound,
    instance_baselines,
    simulator_replay,
)
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
        "callback_time_s" => timing.total_eval_time,
        "hessian_time_s" => timing.hess_time,
        "madnlp_total_time_s" => counters.total_time,
        "madnlp_init_time_s" => counters.init_time,
        "madnlp_eval_time_s" => counters.eval_function_time,
        "madnlp_linear_solver_time_s" => counters.linear_solver_time,
        "hessian_evaluations" => counters.lag_hess_cnt,
        "instance_baselines" => instance_baselines,
        "metrics" => metrics(
            evaluator,
            result[:solution],
            truth,
            target,
            lower_bound,
            upper_bound,
            simulator_replay,
        ),
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
    factory = pyimport("jax_nn_evaluator").create_pdebench_darcy_evaluator
    results = Any[]
    full_resolution = Int(get(config, "full_resolution", 128))
    lower_bound = Float64(get(config, "lower_bound", 0.1))
    upper_bound = Float64(get(config, "upper_bound", 1.0))
    initial_value = Float64(get(config, "initial_value", 0.541621900172992))
    start_registry = (
        isempty(args["starts"]) ? nothing : JSON3.read(read(args["starts"], String))
    )
    simulator_replay = Bool(get(config, "simulator_replay", false))
    regularization_weight = isnan(args["regularization_weight"]) ?
        Float64(config["regularization_weight"]) : args["regularization_weight"]

    for sample in config["samples"]
        truth, target = load_instance(String(config["data"]), Int(sample))
        for latent_resolution in config["latent_resolutions"]
            latent_resolution = Int(latent_resolution)
            reference = fill(initial_value, latent_resolution^2)
            mean_lower = (
                haskey(config, "mean_lower") ? Float64(config["mean_lower"]) : nothing
            )
            mean_upper = (
                haskey(config, "mean_upper") ? Float64(config["mean_upper"]) : nothing
            )
            evaluator = factory(
                String(config["model"]),
                row_major_vec(target),
                reference;
                reference_field=reference,
                full_resolution=full_resolution,
                regularization_weight=regularization_weight,
                binary_weight=Float64(get(config, "binary_weight", 0.0)),
                mean_lower=mean_lower,
                mean_upper=mean_upper,
                device="gpu",
            )
            projection = pyconvert(
                Dict{String,Any},
                evaluator.projection_metrics(truth, lower_bound, upper_bound),
            )
            true_prediction = pyconvert(
                Vector{Float64}, evaluator.evaluate_nn_full_field(truth),
            )
            instance_baselines = merge(
                projection,
                Dict(
                    "true_coefficient_surrogate_target_relative_l2" => (
                        norm(true_prediction - row_major_vec(target)) / norm(target)
                    ),
                    "constant_start_coefficient_relative_l2" => (
                        norm(fill(initial_value, size(truth)) - truth) / norm(truth)
                    ),
                ),
            )
            if simulator_replay
                simulator = pyimport("darcy_data")
                truth_replay = pyconvert(
                    Matrix{Float64}, simulator.darcy_fd_solve(truth),
                )
                instance_baselines["truth_simulator_target_relative_l2"] = (
                    norm(truth_replay - target) / norm(target)
                )
            end
            for start in starts_for_resolution(
                start_registry, latent_resolution, reference,
            )
                x0 = start["values"]
                length(x0) == latent_resolution^2 || error(
                    "Start " * start["id"] * " has the wrong dimension",
                )
                start_objective = pyconvert(Float64, evaluator.evaluate_obj(x0))
                for method in methods
                    println(
                        "sample=", sample,
                        " latent=", latent_resolution,
                        " start=", start["id"],
                        " method=", method,
                    )
                    # Retain one warm-up solve and exclude it from the timed summary.
                    cold_result = nothing
                    cold_elapsed = 0.0
                    cold_error = nothing
                    cold_elapsed = @elapsed try
                        cold_result, _ = solve_method(
                            evaluator,
                            latent_resolution,
                            x0,
                            method,
                            args["max_iter"],
                            args["tol"],
                            lower_bound,
                            upper_bound,
                        )
                    catch exception
                        cold_error = sprint(showerror, exception, catch_backtrace())
                    end
                    common = Dict(
                        "sample" => sample,
                        "latent_resolution" => latent_resolution,
                        "n" => latent_resolution^2,
                        "start_id" => start["id"],
                        "start_sha256" => start["sha256"],
                        "method" => method,
                        "start_objective" => start_objective,
                    )
                    if cold_error !== nothing
                        push!(results, merge(common, Dict(
                            "cold" => Dict(
                                "status" => "EXCEPTION",
                                "elapsed_s" => cold_elapsed,
                                "error" => cold_error,
                            ),
                            "warm" => Any[],
                            "median_elapsed_s" => nothing,
                        )))
                        continue
                    end
                    cold = record_result(
                        method,
                        cold_result,
                        cold_elapsed,
                        evaluator,
                        truth,
                        target,
                        lower_bound,
                        upper_bound,
                        instance_baselines,
                        simulator_replay,
                    )
                    warm = Any[]
                    for _ in 1:args["repetitions"]
                        try
                            result, elapsed = solve_method(
                                evaluator,
                                latent_resolution,
                                x0,
                                method,
                                args["max_iter"],
                                args["tol"],
                                lower_bound,
                                upper_bound,
                            )
                            push!(warm, record_result(
                                method,
                                result,
                                elapsed,
                                evaluator,
                                truth,
                                target,
                                lower_bound,
                                upper_bound,
                                instance_baselines,
                                simulator_replay,
                            ))
                        catch exception
                            push!(warm, Dict(
                                "method" => method,
                                "status" => "EXCEPTION",
                                "error" => sprint(
                                    showerror, exception, catch_backtrace(),
                                ),
                            ))
                        end
                    end
                    successful_times = [
                        record["elapsed_s"] for record in warm
                        if get(record, "status", "") == "SOLVE_SUCCEEDED" &&
                           get(record, "kkt_error", Inf) <= args["tol"]
                    ]
                    push!(results, merge(common, Dict(
                        "cold" => cold,
                        "warm" => warm,
                        "common_success_count" => length(successful_times),
                        "median_elapsed_s" => (
                            isempty(successful_times) ? nothing : median(successful_times)
                        ),
                    )))
                end
            end
        end
    end

    payload = Dict(
        "schema_version" => 1,
        "created_at" => string(Dates.now()),
        "config" => Dict(string(key) => value for (key, value) in pairs(config)),
        "repetitions" => args["repetitions"],
        "max_iter" => args["max_iter"],
        "tol" => args["tol"],
        "effective_regularization_weight" => regularization_weight,
        "starts_registry" => (isempty(args["starts"]) ? nothing : args["starts"]),
        "hostname" => gethostname(),
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
