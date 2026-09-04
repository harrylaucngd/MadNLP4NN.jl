#!/usr/bin/env julia

"""Closest-prior adversarial MNIST benchmark with matched solver paths."""

using ArgParse
using CUDA
using Dates
using JSON3
using LinearAlgebra
using MadNLP
using MadNLP4NN
using NLPModels
using PythonCall
using Random
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
            default = 1000
        "--tol"
            arg_type = Float64
            default = 1e-6
        "--methods"
            default = "exact_cpu,exact_host_staged_gpu,exact_dlpack_gpu,lbfgs_cpu"
        "--start-count"
            arg_type = Int
            default = 1
        "--start-seed"
            arg_type = Int
            default = 20260821
        "--start-radius"
            arg_type = Float64
            default = 0.05
    end
    return parse_args(settings)
end


function load_test_image(data_dir, sample)
    torchvision = pyimport("torchvision")
    transforms = pyimport("torchvision.transforms")
    np = pyimport("numpy")
    dataset = torchvision.datasets.MNIST(
        root=data_dir,
        train=false,
        transform=transforms.ToTensor(),
        download=false,
    )
    item = dataset[sample]
    image = pyconvert(
        Vector{Float64},
        np.asarray(item[0].reshape(-1).numpy(), dtype=np.float64),
    )
    label = pyconvert(Int, item[1])
    return image, label
end


function make_model(evaluator, initial_guess, threshold)
    n = pyconvert(Int, evaluator.n)
    m = pyconvert(Int, evaluator.m)
    pixel_size = 784
    output_size = 10
    positive_offset = pixel_size + output_size
    lower = vcat(
        zeros(pixel_size),
        fill(-Inf, output_size),
        zeros(2 * pixel_size),
    )
    upper = vcat(
        ones(pixel_size),
        fill(Inf, output_size),
        fill(Inf, 2 * pixel_size),
    )
    equality_count = output_size + pixel_size
    lower_constraints = vcat(
        zeros(equality_count),
        [threshold],
        zeros(output_size),
    )
    upper_constraints = vcat(
        zeros(equality_count),
        [Inf],
        ones(output_size),
    )
    @assert length(lower) == n
    @assert length(lower_constraints) == m
    meta = NLPModelMeta(
        n;
        x0=copy(initial_guess),
        lvar=lower,
        uvar=upper,
        ncon=m,
        lcon=lower_constraints,
        ucon=upper_constraints,
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


function solve_method(evaluator, initial_guess, threshold, method, max_iter, tol)
    inner = make_model(evaluator, initial_guess, threshold)
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


function record_result(method, result, elapsed, evaluator, reference, adversarial_label)
    unpacked = evaluator.unpack(result[:solution])
    pixels = pyconvert(Vector{Float64}, unpacked[0])
    outputs = pyconvert(Vector{Float64}, unpacked[1])
    constraints = pyconvert(
        Vector{Float64}, evaluator.evaluate_cons(result[:solution]),
    )
    network_violation = norm(constraints[1:10], Inf)
    slack_violation = norm(constraints[11:794], Inf)
    target_violation = max(
        0.0, pyconvert(Float64, evaluator.threshold) - constraints[795],
    )
    output_bound_violation = maximum(
        max.(0.0, max.(-constraints[796:805], constraints[796:805] .- 1.0))
    )
    max_raw_violation = maximum((
        network_violation,
        slack_violation,
        target_violation,
        output_bound_violation,
    ))
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
        "target_probability" => outputs[adversarial_label + 1],
        "predicted_label" => argmax(outputs) - 1,
        "actual_l1_perturbation" => norm(pixels - reference, 1),
        "raw_network_equality_inf" => network_violation,
        "raw_slack_equality_inf" => slack_violation,
        "raw_target_violation" => target_violation,
        "raw_output_bound_violation" => output_bound_violation,
        "max_raw_constraint_violation" => max_raw_violation,
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
    reference, label = load_test_image(String(config["data_dir"]), Int(config["sample"]))
    label == Int(config["expected_label"]) || error("Unexpected MNIST label")
    factory = pyimport("jax_nn_evaluator").create_mnist_adversarial_evaluator
    args["start-count"] >= 1 || error("--start-count must be positive")
    rng = MersenneTwister(args["start-seed"])
    pixel_starts = [("published", copy(reference))]
    for index in 1:(args["start-count"] - 1)
        perturbed = clamp.(
            reference .+ args["start-radius"] .* randn(rng, length(reference)),
            0.0,
            1.0,
        )
        push!(pixel_starts, ("gaussian_$(index)", perturbed))
    end
    results = Any[]
    for model_path in config["models"]
        evaluator = factory(
            String(model_path),
            reference,
            Int(config["adversarial_label"]);
            threshold=Float64(config["threshold"]),
            device="gpu",
        )
        for (start_id, pixels) in pixel_starts
            initial_guess = pyconvert(
                Vector{Float64}, evaluator.initial_guess_from_pixels(pixels),
            )
            initial_constraints = pyconvert(
                Vector{Float64}, evaluator.evaluate_cons(initial_guess),
            )
            initial_objective = pyconvert(
                Float64, evaluator.evaluate_obj(initial_guess),
            )
            initial_raw_violation = maximum((
                norm(initial_constraints[1:10], Inf),
                norm(initial_constraints[11:794], Inf),
                max(0.0, Float64(config["threshold"]) - initial_constraints[795]),
                maximum(max.(
                    0.0,
                    max.(-initial_constraints[796:805], initial_constraints[796:805] .- 1.0),
                )),
            ))
            for method in methods
                println(
                    "MNIST model=", basename(String(model_path)),
                    " start=", start_id,
                    " method=", method,
                )
                warmup_result, warmup_elapsed = solve_method(
                    evaluator,
                    initial_guess,
                    Float64(config["threshold"]),
                    method,
                    args["max_iter"],
                    args["tol"],
                )
                warmup = record_result(
                    method,
                    warmup_result,
                    warmup_elapsed,
                    evaluator,
                    reference,
                    Int(config["adversarial_label"]),
                )
                warm = Any[]
                for _ in 1:args["repetitions"]
                    result, elapsed = solve_method(
                        evaluator,
                        initial_guess,
                        Float64(config["threshold"]),
                        method,
                        args["max_iter"],
                        args["tol"],
                    )
                    push!(warm, record_result(
                        method,
                        result,
                        elapsed,
                        evaluator,
                        reference,
                        Int(config["adversarial_label"]),
                    ))
                end
                successful = [
                    record for record in warm
                    if record["status"] == "SOLVE_SUCCEEDED" &&
                       record["kkt_error"] <= args["tol"] &&
                       record["max_raw_constraint_violation"] <= 1e-6
                ]
                push!(results, Dict(
                    "model" => String(model_path),
                    "parameter_count" => pyconvert(Int, evaluator.parameter_count),
                    "start_id" => start_id,
                    "initial_objective" => initial_objective,
                    "initial_raw_constraint_violation" => initial_raw_violation,
                    "method" => method,
                    "warmup" => warmup,
                    "warm" => warm,
                    "common_success_count" => length(successful),
                    "median_elapsed_s" => isempty(successful) ? nothing : median(
                        [record["elapsed_s"] for record in successful]
                    ),
                ))
            end
        end
    end
    payload = Dict(
        "schema_version" => 1,
        "created_at" => string(Dates.now()),
        "hostname" => gethostname(),
        "config" => Dict(string(key) => value for (key, value) in pairs(config)),
        "repetitions" => args["repetitions"],
        "max_iter" => args["max_iter"],
        "tol" => args["tol"],
        "start_count" => args["start-count"],
        "start_seed" => args["start-seed"],
        "start_radius" => args["start-radius"],
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
