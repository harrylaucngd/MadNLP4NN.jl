#!/usr/bin/env julia

"""Run the published PFR sequence with controlled cold/warm-start policies."""

using ArgParse
using CUDA
using Dates
using JSON3
using LinearAlgebra
using MadNLP
using MadNLP4NN
using MadNLPGPU
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
        "--steps"
            arg_type = Int
            default = 100
        "--max_iter"
            arg_type = Int
            default = 500
        "--tol"
            arg_type = Float64
            default = 1e-6
        "--method"
            default = "exact_dlpack_gpu"
        "--state_mode"
            default = "auto"
        "--warm_start_mode"
            default = "auto"
        "--rebuild_interval"
            arg_type = Int
            default = 0
        "--reuse_watchdog_s"
            arg_type = Float64
            default = Inf
    end
    return parse_args(settings)
end


row_major_vec(field::AbstractMatrix) = vec(permutedims(field))


function mechanistic_rk4(state, control; sample_time=0.1, substeps=100)
    flow, inlet = control
    spatial_step = 1.0 / length(state)
    function derivative(value)
        upstream = vcat(inlet, value[1:end-1])
        return -flow .* (value - upstream) ./ spatial_step .- value .^ 2
    end
    value = copy(state)
    step = sample_time / substeps
    for _ in 1:substeps
        k1 = derivative(value)
        k2 = derivative(value + 0.5 * step * k1)
        k3 = derivative(value + 0.5 * step * k2)
        k4 = derivative(value + step * k3)
        value .+= step / 6 .* (k1 + 2k2 + 2k3 + k4)
    end
    return value
end


function load_published(path)
    builtins = pyimport("builtins")
    pickle = pyimport("pickle")
    np = pyimport("numpy")
    handle = builtins.open(path, "rb")
    try
        payload = pickle.load(handle)
        return Dict(
            "flow" => pyconvert(
                Vector{Float64}, np.asarray(payload["F"], dtype=np.float64),
            ),
            "inlet" => pyconvert(
                Vector{Float64}, np.asarray(payload["C_in"], dtype=np.float64),
            ),
            "state" => pyconvert(
                Matrix{Float64}, np.asarray(payload["C"], dtype=np.float64),
            ),
        )
    finally
        handle.close()
    end
end


function make_model(
    evaluator,
    config,
    initial_guess,
    dual_guess,
    current_state,
    previous_control,
    setpoint,
)
    prediction_horizon = Int(config["prediction_horizon"])
    control_horizon = Int(config["control_horizon"])
    state_dimension = length(current_state)
    control_dimension = length(previous_control)
    state_size = (prediction_horizon + 1) * state_dimension
    base_size = state_size + control_horizon * control_dimension
    n = base_size + 3
    m = prediction_horizon * state_dimension
    lower = vcat(
        fill(Float64(config["state_lower"]), state_size),
        fill(Float64(config["control_lower"]), base_size - state_size),
        [setpoint],
        previous_control,
    )
    upper = vcat(
        fill(Float64(config["state_upper"]), state_size),
        fill(Float64(config["control_upper"]), base_size - state_size),
        [setpoint],
        previous_control,
    )
    lower[1:state_dimension] .= current_state
    upper[1:state_dimension] .= current_state
    meta = NLPModelMeta(
        n;
        x0=copy(initial_guess),
        lvar=lower,
        uvar=upper,
        ncon=m,
        y0=copy(dual_guess),
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


function solve_step(
    inner,
    method,
    max_iter,
    tol;
    dual_initialized=false,
    warm_barrier=false,
)
    warm_options = warm_barrier ?
        (dual_initialized=dual_initialized,
         barrier=MadNLP.MonotoneUpdate(; mu_init=1e-7)) :
        (dual_initialized=dual_initialized,)
    CUDA.synchronize()
    result = nothing
    elapsed = @elapsed begin
        if method == "exact_dlpack_gpu"
            result = solve_nlp(
                DLPackGPUModel(inner; verify_interop=false);
                max_iter=max_iter,
                tol=tol,
                print_level=MadNLP.ERROR,
                warm_options...,
            )
        elseif method == "exact_cpu"
            result = solve_nlp(
                inner;
                kkt_device="cpu",
                linear_solver=MadNLP.MumpsSolver,
                max_iter=max_iter,
                tol=tol,
                print_level=MadNLP.ERROR,
                warm_options...,
            )
        else
            error("Unsupported closed-loop method " * method)
        end
        CUDA.synchronize()
    end
    return result, elapsed
end


function build_reuse_solver(inner, max_iter, tol)
    gpu_model = DLPackGPUModel(inner; verify_interop=false)
    solver = MadNLP.MadNLPSolver(
        gpu_model;
        max_iter=max_iter,
        tol=tol,
        print_level=MadNLP.ERROR,
        linear_solver=MadNLPGPU.CUDSSSolver,
    )
    return gpu_model, solver
end


function solve_reuse!(
    solver,
    gpu_model,
    updated_inner,
    initial_guess;
    first_solve=false,
    watchdog_s=Inf,
)
    gpu_model.meta.x0 .= CuArray(initial_guess)
    gpu_model.meta.lvar .= CuArray(updated_inner.meta.lvar)
    gpu_model.meta.uvar .= CuArray(updated_inner.meta.uvar)
    gpu_model.inner.meta.x0 .= initial_guess
    gpu_model.inner.meta.lvar .= updated_inner.meta.lvar
    gpu_model.inner.meta.uvar .= updated_inner.meta.uvar
    if !first_solve
        solver.cnt = MadNLP.MadNLPCounters(start_time=time())
        solver.status = MadNLP.INITIAL
        solver.del_w = 0.0
        solver.del_c = 0.0
        solver.del_w_last = 0.0
        solver.alpha = 0.0
        solver.alpha_z = 0.0
        solver.RR = nothing
        MadNLP.variable(solver.x) .= gpu_model.meta.x0
        MadNLP.variable(solver.xl) .= gpu_model.meta.lvar
        MadNLP.variable(solver.xu) .= gpu_model.meta.uvar
    end
    solver.opt.max_wall_time = first_solve ? Inf : watchdog_s
    callback_before = gpu_model.timing_stats.total_eval_time
    hessian_before = gpu_model.timing_stats.hess_time
    CUDA.synchronize()
    stats = nothing
    elapsed = @elapsed begin
        stats = MadNLP.solve!(solver)
        CUDA.synchronize()
    end
    n = gpu_model.meta.nvar
    return Dict(
        :status => stats.status,
        :solution => Array(stats.solution[1:n]),
        :multipliers => Array(stats.multipliers),
        :objective => stats.objective,
        :iter_count => stats.iter,
        :primal_feas => stats.primal_feas,
        :dual_feas => stats.dual_feas,
        :kkt_error => MadNLP.get_kkt_error(solver),
        :callback_time_step => (
            gpu_model.timing_stats.total_eval_time - callback_before
        ),
        :hessian_time_step => gpu_model.timing_stats.hess_time - hessian_before,
        :init_time_step => 0.0,
    ), elapsed
end


function shifted_multipliers(multipliers, state_dimension)
    isempty(multipliers) && return Float64[]
    length(multipliers) % state_dimension == 0 || error(
        "Multiplier dimension is not divisible by the state dimension",
    )
    blocks = reshape(multipliers, state_dimension, :)
    shifted = hcat(blocks[:, 2:end], blocks[:, end:end])
    return vec(shifted)
end


function shifted_guess(evaluator, solution, current_state, previous_control, setpoint)
    unpacked = evaluator.unpack(solution)
    states = pyconvert(Matrix{Float64}, unpacked[0])
    controls = pyconvert(Matrix{Float64}, unpacked[1])
    shifted_states = vcat(
        permutedims(current_state),
        states[3:end, :],
        states[end:end, :],
    )
    shifted_controls = vcat(controls[2:end, :], controls[end:end, :])
    return vcat(
        row_major_vec(shifted_states),
        row_major_vec(shifted_controls),
        [setpoint],
        previous_control,
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


function write_payload(path, payload)
    mkpath(dirname(path))
    open(path, "w") do io
        JSON3.pretty(io, json_safe(payload))
        write(io, '\n')
    end
end


function main()
    args = parse_cli()
    config = JSON3.read(read(args["config"], String))
    setup_python_env()
    factory = pyimport("jax_nn_evaluator").create_pfr_nmpc_evaluator
    initial_state = Float64.(config["current_state"])
    initial_control = Float64.(config["previous_control"])
    evaluator = factory(
        String(config["model"]),
        initial_state,
        initial_control,
        Float64(config["setpoint"]);
        prediction_horizon=Int(config["prediction_horizon"]),
        control_horizon=Int(config["control_horizon"]),
        parameters_as_variables=true,
        device="gpu",
    )
    published = load_published(String(config["published_output"]))
    state_mode = args["state_mode"] == "auto" ?
        (args["method"] == "exact_dlpack_gpu_reuse" ? "published_replay" : "scipy") :
        args["state_mode"]
    state_mode in ("published_replay", "scipy", "rk4") || error(
        "state_mode must be published_replay, scipy, or rk4",
    )
    warm_start_mode = args["warm_start_mode"] == "auto" ?
        (args["method"] == "exact_dlpack_gpu_reuse" ? "persistent" : "primal") :
        args["warm_start_mode"]
    warm_start_mode in (
        "cold", "primal", "primal_barrier", "primal_dual",
        "primal_dual_barrier", "persistent",
    ) || error(
        "warm_start_mode must be cold, primal, primal_barrier, " *
        "primal_dual, primal_dual_barrier, or persistent",
    )
    args["method"] == "exact_dlpack_gpu_reuse" &&
        warm_start_mode != "persistent" && error(
            "exact_dlpack_gpu_reuse requires warm_start_mode=persistent",
        )
    args["method"] != "exact_dlpack_gpu_reuse" &&
        warm_start_mode == "persistent" && error(
            "warm_start_mode=persistent requires exact_dlpack_gpu_reuse",
        )
    current_state = copy(initial_state)
    previous_control = copy(initial_control)
    initial_guess = pyconvert(Vector{Float64}, evaluator.initial_guess())
    dual_guess = zeros(Float64, pyconvert(Int, evaluator.m))
    previous_solution = nothing
    previous_multipliers = nothing
    records = Any[]
    reuse_gpu_model = nothing
    reuse_solver = nothing
    reuse_setup_time = 0.0

    for step in 0:(args["steps"] - 1)
        setpoint = step > 50 ? 0.3 : 0.4
        if state_mode == "published_replay"
            current_state = vec(published["state"][step + 1, :])
            previous_control = [
                published["flow"][step + 1], published["inlet"][step + 1],
            ]
        end
        if step > 0 && warm_start_mode != "cold"
            initial_guess = shifted_guess(
                evaluator,
                previous_solution,
                current_state,
                previous_control,
                setpoint,
            )
        else
            initial_guess = pyconvert(
                Vector{Float64},
                evaluator.initial_guess_for(
                    current_state,
                    previous_control,
                    setpoint,
                ),
            )
        end
        dual_initialized = step > 0 && warm_start_mode in (
            "primal_dual", "primal_dual_barrier",
        )
        dual_guess = dual_initialized ?
            shifted_multipliers(previous_multipliers, length(current_state)) :
            zeros(Float64, pyconvert(Int, evaluator.m))
        initial_constraint_inf = norm(
            pyconvert(Vector{Float64}, evaluator.evaluate_cons(initial_guess)), Inf,
        )
        inner = make_model(
            evaluator,
            config,
            initial_guess,
            dual_guess,
            current_state,
            previous_control,
            setpoint,
        )
        proactive_rebuild = false
        proactive_setup_time = 0.0
        if args["method"] == "exact_dlpack_gpu_reuse" &&
           reuse_solver !== nothing &&
           args["rebuild_interval"] > 0 &&
           step % args["rebuild_interval"] == 0
            proactive_rebuild = true
            proactive_setup_time = @elapsed begin
                reuse_gpu_model, reuse_solver = build_reuse_solver(
                    inner, args["max_iter"], args["tol"],
                )
            end
        end
        result, elapsed = if args["method"] == "exact_dlpack_gpu_reuse"
            if reuse_solver === nothing
                reuse_setup_time = @elapsed begin
                    reuse_gpu_model, reuse_solver = build_reuse_solver(
                        inner, args["max_iter"], args["tol"],
                    )
                end
            end
            solve_reuse!(
                reuse_solver,
                reuse_gpu_model,
                inner,
                initial_guess;
                first_solve=(step == 0 || proactive_rebuild),
                watchdog_s=args["reuse_watchdog_s"],
            )
        else
            solve_step(
                inner,
                args["method"],
                args["max_iter"],
                args["tol"];
                dual_initialized=dual_initialized,
                warm_barrier=(step > 0 && warm_start_mode in (
                    "primal_barrier", "primal_dual_barrier",
                )),
            )
        end
        elapsed += proactive_setup_time
        fallback_rebuild = false
        fallback_setup_time = 0.0
        fallback_solve_time = 0.0
        if args["method"] == "exact_dlpack_gpu_reuse" &&
           !(result[:status] == MadNLP.SOLVE_SUCCEEDED &&
             result[:kkt_error] <= args["tol"])
            fallback_rebuild = true
            fallback_setup_time = @elapsed begin
                reuse_gpu_model, reuse_solver = build_reuse_solver(
                    inner, args["max_iter"], args["tol"],
                )
            end
            result, fallback_solve_time = solve_reuse!(
                reuse_solver,
                reuse_gpu_model,
                inner,
                initial_guess;
                first_solve=true,
                watchdog_s=args["reuse_watchdog_s"],
            )
            elapsed += fallback_setup_time + fallback_solve_time
        end
        unpacked = evaluator.unpack(result[:solution])
        controls = pyconvert(Matrix{Float64}, unpacked[1])
        applied_control = vec(controls[1, :])
        published_control = [published["flow"][step + 2], published["inlet"][step + 2]]
        published_state = vec(published["state"][step + 2, :])
        state_replay = state_mode == "published_replay"
        next_state = if state_replay
            copy(published_state)
        elseif state_mode == "rk4"
            mechanistic_rk4(current_state, applied_control)
        else
            pyconvert(
                Vector{Float64},
                evaluator.mechanistic_step(current_state, applied_control, 0.1),
            )
        end
        callback_time = haskey(result, :callback_time_step) ?
            result[:callback_time_step] : result[:timing_stats].total_eval_time
        hessian_time = haskey(result, :hessian_time_step) ?
            result[:hessian_time_step] : result[:timing_stats].hess_time
        init_time = haskey(result, :init_time_step) ?
            result[:init_time_step] : result[:solver_counters].init_time
        record = Dict(
            "step" => step,
            "setpoint" => setpoint,
            "status" => string(result[:status]),
            "elapsed_s" => elapsed,
            "iterations" => result[:iter_count],
            "objective" => result[:objective],
            "kkt_error" => result[:kkt_error],
            "primal_feas" => result[:primal_feas],
            "dual_feas" => result[:dual_feas],
            "applied_control" => applied_control,
            "published_control" => published_control,
            "control_inf_error" => norm(applied_control - published_control, Inf),
            "next_state" => next_state,
            "published_next_state" => published_state,
            "state_inf_error" => norm(next_state - published_state, Inf),
            "state_relative_l2_error" => norm(next_state - published_state) / norm(published_state),
            "published_state_replay" => state_replay,
            "state_mode" => state_mode,
            "warm_start_mode" => warm_start_mode,
            "dual_initialized" => dual_initialized,
            "initial_dual_inf" => norm(dual_guess, Inf),
            "initial_constraint_inf" => initial_constraint_inf,
            "one_time_solver_setup_s" => step == 0 ? reuse_setup_time : 0.0,
            "fallback_rebuild" => fallback_rebuild,
            "fallback_setup_time_s" => fallback_setup_time,
            "fallback_solve_time_s" => fallback_solve_time,
            "proactive_rebuild" => proactive_rebuild,
            "proactive_setup_time_s" => proactive_setup_time,
            "callback_time_s" => callback_time,
            "hessian_time_s" => hessian_time,
            "madnlp_init_time_s" => init_time,
            "solution" => result[:solution],
            "multipliers" => result[:multipliers],
        )
        push!(records, record)
        previous_solution = result[:solution]
        previous_multipliers = result[:multipliers]
        current_state = next_state
        previous_control = applied_control
        println(
            "step=", step,
            " status=", record["status"],
            " time=", round(elapsed, digits=3),
            " control_err=", record["control_inf_error"],
            " state_err=", record["state_inf_error"],
        )

        # Persist a checkpoint without the large warm-start vectors.
        public_records = [
            Dict(
                key => value for (key, value) in pairs(item)
                if key ∉ ("solution", "multipliers")
            )
            for item in records
        ]
        write_payload(args["output"], Dict(
            "schema_version" => 1,
            "created_at" => string(Dates.now()),
            "complete" => false,
            "method" => args["method"],
            "warm_start_mode" => warm_start_mode,
            "hostname" => gethostname(),
            "config" => Dict(string(key) => value for (key, value) in pairs(config)),
            "records" => public_records,
        ))
    end

    public_records = [
        Dict(
            key => value for (key, value) in pairs(item)
            if key ∉ ("solution", "multipliers")
        )
        for item in records
    ]
    successes = [
        item for item in public_records
        if item["status"] == "SOLVE_SUCCEEDED" && item["kkt_error"] <= args["tol"]
    ]
    payload = Dict(
        "schema_version" => 1,
        "created_at" => string(Dates.now()),
        "complete" => true,
        "method" => args["method"],
        "warm_start_mode" => warm_start_mode,
        "published_state_replay" => state_mode == "published_replay",
        "state_mode" => state_mode,
        "one_time_solver_setup_s" => reuse_setup_time,
        "hostname" => gethostname(),
        "config" => Dict(string(key) => value for (key, value) in pairs(config)),
        "steps" => args["steps"],
        "common_success_count" => length(successes),
        "median_elapsed_s" => median([item["elapsed_s"] for item in successes]),
        "median_iterations" => median([item["iterations"] for item in successes]),
        "mean_iterations" => mean([item["iterations"] for item in successes]),
        "max_iterations" => maximum(item["iterations"] for item in successes),
        "max_control_inf_error" => maximum(item["control_inf_error"] for item in public_records),
        "max_state_inf_error" => maximum(item["state_inf_error"] for item in public_records),
        "records" => public_records,
    )
    write_payload(args["output"], payload)
    println("Wrote ", args["output"])
end


main()
