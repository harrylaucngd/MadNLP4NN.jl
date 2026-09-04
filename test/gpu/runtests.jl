using Test
using CUDA
using MadNLP
using MadNLP4NN
using NLPModels
using Random
using PythonCall

CUDA.functional() || error("GPU integration tests require a functional CUDA device")


function make_gpu_smoke_nlp(; n=12, m_out=4, seed=11)
    rng = Random.MersenneTwister(seed)
    W = randn(rng, m_out, n)
    b = randn(rng, m_out)
    nn_model = x -> W * x .+ b
    target = zeros(m_out)
    x0 = zeros(n)
    objective = NeuralNetworkObjective(target)
    meta = NLPModelMeta(
        n;
        x0=copy(x0),
        lvar=fill(-5.0, n),
        uvar=fill(5.0, n),
        ncon=0,
        lcon=Float64[],
        ucon=Float64[],
        nnzj=0,
        nnzh=div(n * (n + 1), 2),
        minimize=true,
    )
    return NeuralNetworkNLPModel(
        meta,
        NLPModels.Counters(),
        nn_model,
        objective,
        nothing,
        copy(x0),
        false,
        nothing,
        TimingStats(),
        :forwarddiff,
    )
end


function make_jax_gpu_smoke_nlp(; n=12, m_out=4, seed=17)
    setup_python_env()
    rng = Random.MersenneTwister(seed)
    W = randn(rng, m_out, n)
    b = randn(rng, m_out)
    x0 = zeros(n)
    target = zeros(m_out)

    dm = pyimport("backend.device").DeviceManager("gpu")
    np = pyimport("numpy")
    namespace = PythonCall.pybuiltins.dict()
    PythonCall.pybuiltins.exec(
        "def forward(params, x):\n    return params['W'] @ x + params['b']\n",
        namespace,
    )
    params = PythonCall.pybuiltins.dict(
        W=dm.array(np.asarray(W)),
        b=dm.array(np.asarray(b)),
    )
    evaluator = pyimport("evaluators.standard").StandardEvaluator(
        namespace["forward"],
        params,
        target,
        x0;
        regularization_weight=1e-2,
        radius=2.0,
        device_manager=dm,
    )

    meta = NLPModelMeta(
        n;
        x0=copy(x0),
        lvar=fill(-5.0, n),
        uvar=fill(5.0, n),
        ncon=1,
        lcon=[-Inf],
        ucon=[0.0],
        nnzj=n,
        nnzh=div(n * (n + 1), 2),
        minimize=true,
    )
    inner = NeuralNetworkNLPModel(
        meta,
        NLPModels.Counters(),
        nothing,
        NeuralNetworkObjective(target),
        SphericalConstraint(x0, 2.0),
        copy(x0),
        true,
        evaluator,
        TimingStats(),
        :forwarddiff,
    )
    return inner, evaluator
end


@testset "host-staged GPU KKT" begin
    nlp = make_gpu_smoke_nlp()
    result = solve_nlp(
        nlp;
        kkt_device="gpu",
        max_iter=100,
        tol=1e-7,
        print_level=MadNLP.ERROR,
    )
    @test result[:kkt_device] == "gpu"
    @test result[:host_staged]
    @test all(isfinite, result[:solution])
    @test isfinite(result[:objective])
    @test result[:objective] <= NLPModels.obj(nlp, nlp.meta.x0) + 1e-8
end


@testset "zero-copy JAX/MadNLP GPU callbacks" begin
    inner, evaluator = make_jax_gpu_smoke_nlp()
    nlp = DLPackGPUModel(inner)
    n = nlp.meta.nvar
    x = copy(nlp.meta.x0)
    y = CuArray([0.3])

    gradient = CUDA.zeros(Float64, n)
    constraints = CUDA.zeros(Float64, 1)
    jacobian = CUDA.zeros(Float64, n)
    hessian = CUDA.zeros(Float64, div(n * (n + 1), 2))
    direction = CuArray(collect(range(-0.2, 0.2; length=n)))
    constraint_weight = CuArray([0.7])
    jacobian_product = CUDA.zeros(Float64, 1)
    transpose_product = CUDA.zeros(Float64, n)
    NLPModels.grad!(nlp, x, gradient)
    NLPModels.cons!(nlp, x, constraints)
    NLPModels.jac_coord!(nlp, x, jacobian)
    NLPModels.hess_coord!(nlp, x, y, hessian; obj_weight=1.0)
    NLPModels.jprod!(nlp, x, direction, jacobian_product)
    NLPModels.jtprod!(nlp, x, constraint_weight, transpose_product)

    x_host = Array(x)
    y_host = Array(y)
    gradient_host = pyconvert(Vector{Float64}, evaluator.evaluate_grad(x_host))
    constraints_host = pyconvert(Vector{Float64}, evaluator.evaluate_cons(x_host))
    jacobian_host = pyconvert(Matrix{Float64}, evaluator.evaluate_jac(x_host))
    hessian_host = pyconvert(Matrix{Float64}, evaluator.evaluate_hess(x_host, y_host, 1.0))
    rows = vcat([collect(column:n) for column in 1:n]...)
    cols = vcat([fill(column, n - column + 1) for column in 1:n]...)

    @test Array(gradient) ≈ gradient_host
    @test Array(constraints) ≈ constraints_host
    @test Array(jacobian) ≈ vec(permutedims(jacobian_host))
    @test Array(hessian) ≈ hessian_host[CartesianIndex.(rows, cols)]
    @test Array(jacobian_product) ≈ jacobian_host * Array(direction)
    @test Array(transpose_product) ≈ transpose(jacobian_host) * Array(constraint_weight)

    result = solve_nlp(
        nlp;
        max_iter=100,
        tol=1e-7,
        print_level=MadNLP.ERROR,
    )
    @test result[:zero_copy]
    @test !result[:host_staged]
    @test all(isfinite, result[:solution])
    @test isfinite(result[:objective])
end
