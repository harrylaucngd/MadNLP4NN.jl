"""
runtests.jl

Test suite for MadNLP4NN.jl.

Covers:
  - Callback correctness on small synthetic problems (CPU only, no Python)
  - Problem specification and constraint accounting
  - New objective and constraint types
  - Discrete Laplacian helpers
  - Pareto quality metrics (pure Julia)
  - ProblemSpec construction
"""

using Test
using MadNLP4NN
using MadNLP
using NLPModels
using LinearAlgebra
using Random

# ============================================================================
# Helper: build a tiny synthetic NLP without a real checkpoint
# ============================================================================

"""
Build a small NeuralNetworkNLPModel wrapping a trivial Flux network
(a single linear layer mapping R^n → R^m).
"""
function make_tiny_flux_nlp(; n=10, m_out=4, seed=1)
    rng = Random.MersenneTwister(seed)
    W = randn(rng, m_out, n)
    b = randn(rng, m_out)

    # Create a trivial Flux-style callable
    nn_model = x -> W * x .+ b

    target = zeros(m_out)
    target[1] = 1.0
    x0 = randn(rng, n) .* 0.1

    obj = NeuralNetworkObjective(target)
    cons = BoxConstraints(fill(-5.0, n), fill(5.0, n))

    # Build meta manually (bypass load_pytorch_model_to_flux)
    meta = NLPModelMeta(
        n,
        x0=copy(x0),
        lvar=fill(-5.0, n),
        uvar=fill(5.0, n),
        ncon=0,
        lcon=Float64[],
        ucon=Float64[],
        nnzj=0,
        nnzh=div(n*(n+1), 2),
        minimize=true,
    )

    return NeuralNetworkNLPModel(
        meta,
        NLPModels.Counters(),
        nn_model,
        obj,
        nothing,      # constraint (box → lvar/uvar)
        copy(x0),
        false,        # use_python
        nothing,      # python_evaluator
        TimingStats(),
        :zygote,
    ), W, b, x0, target
end


# ============================================================================
# Test 1: New objective types evaluate correctly
# ============================================================================

@testset "Extended objective types" begin
    n, m = 8, 3
    target = rand(m)
    x0 = rand(n)
    nn_out = rand(m)

    # SurrogateInversionObjective
    obj = SurrogateInversionObjective(target, x0; weight=2.0, reg_weight=0.5)
    expected = 2.0 * sum((nn_out .- target).^2) + 0.5 * sum((x0 .- x0).^2)
    @test isapprox(evaluate(obj, x0, nn_out), expected)

    # reg_weight > 0, x ≠ x0
    x = rand(n)
    expected2 = 2.0 * sum((nn_out .- target).^2) + 0.5 * sum((x .- x0).^2)
    @test isapprox(evaluate(obj, x, nn_out), expected2)

    # WeightedScalarizationObjective
    alpha = 0.3
    w_obj = WeightedScalarizationObjective(alpha)
    f1, f2 = 1.5, 2.0
    nn_out2 = [f1, f2]
    expected_w = alpha * f1 + (1 - alpha) * f2
    @test isapprox(evaluate(w_obj, rand(n), nn_out2), expected_w)

    # Edge cases
    @test_throws AssertionError WeightedScalarizationObjective(1.5)   # α > 1
    @test_throws AssertionError WeightedScalarizationObjective(-0.1)  # α < 0
end


# ============================================================================
# Test 2: New constraint types evaluate correctly
# ============================================================================

@testset "Extended constraint types" begin
    n = 6

    # BudgetConstraint
    budget = 3.0
    bc = BudgetConstraint(budget)
    x1 = ones(n)  # sum = 6 > budget
    x2 = fill(0.4, n)  # sum = 2.4 < budget
    @test num_constraints(bc) == 1
    @test evaluate(bc, x1)[1] ≈ 6.0 - 3.0
    @test evaluate(bc, x2)[1] ≈ 2.4 - 3.0

    # SmoothnessConstraint with Laplacian
    L1 = laplacian_1d(n)
    tau = 5.0
    sc = SmoothnessConstraint(L1, tau)
    @test num_constraints(sc) == 1
    x_flat = ones(n)
    # ones' L ones = 0 for 1-D Laplacian (Dirichlet) → contribution from boundary
    val = dot(x_flat, L1 * x_flat)
    @test isapprox(evaluate(sc, x_flat)[1], val - tau)

    # LearnedFeasibilityConstraint
    lfc = LearnedFeasibilityConstraint(; network_index=3, threshold=0.1)
    @test num_constraints(lfc) == 1
    @test_throws ErrorException evaluate(lfc, ones(n))  # must use Python backend

    # Hessian contribution helpers
    SphereCons = SphericalConstraint(zeros(n), 2.0)
    @test has_closed_form_hessian(SphereCons)
    @test !has_closed_form_hessian(bc)
    contrib = hessian_contribution(SphereCons, ones(n), 0.5)
    @test isapprox(contrib, 2.0 * 0.5)  # scalar 2λ

    @test hessian_contribution(bc, ones(n), 1.0) === nothing
end


# ============================================================================
# Test 3: laplacian_1d / laplacian_2d
# ============================================================================

@testset "Discrete Laplacian helpers" begin
    # 1-D Laplacian of size n should have tridiagonal structure
    n = 5
    L = laplacian_1d(n)
    @test size(L) == (n, n)
    @test issymmetric(L)
    @test all(diag(L) .== 2.0)
    @test all(diag(L, 1) .== -1.0)

    # 2-D Laplacian: n×n grid → n²×n² matrix
    L2 = laplacian_2d(4, 4)
    @test size(L2) == (16, 16)
    @test issymmetric(L2)
    # All eigenvalues ≥ 0 (PSD)
    λ = eigvals(Symmetric(L2))
    @test all(λ .>= -1e-10)

    # Consistency: rectangular grid
    L_rect = laplacian_2d(3, 4)
    @test size(L_rect) == (12, 12)
    @test issymmetric(L_rect)
end


# ============================================================================
# Test 4: ProblemSpec construction
# ============================================================================

@testset "ProblemSpec construction" begin
    n = 10
    x0 = zeros(n)
    target = ones(3)

    obj = SurrogateInversionObjective(target, x0; weight=1.0)
    cons = BoxConstraints(fill(-1.0, n), fill(1.0, n))

    spec = ProblemSpec(
        "dummy/model.pt",
        obj,
        x0;
        constraints=cons,
        x_lb=fill(-1.0, n),
        x_ub=fill(1.0, n),
        problem_type="darcy",
    )

    @test spec.n == n
    @test length(spec.network_paths) == 1
    @test spec.problem_type == "darcy"
    @test spec.constraints isa BoxConstraints

    # Multi-network spec
    spec2 = ProblemSpec(
        ["f1.pt", "f2.pt", "h.pt"],
        WeightedScalarizationObjective(0.4),
        x0;
        problem_type="pareto",
    )
    @test length(spec2.network_paths) == 3
    @test spec2.objective isa WeightedScalarizationObjective
end


# ============================================================================
# Test 5: DarcyProblemConfig construction
# ============================================================================

@testset "DarcyProblemConfig" begin
    n = 64
    x0 = ones(n)
    y_tar = randn(n)

    # Stage A: box only
    cfg_a = DarcyProblemConfig(
        "dummy_fno.npz", y_tar, x0;
        x_lb=0.0, x_ub=5.0, lambda_reg=1e-2,
    )
    @test length(cfg_a.x0) == n
    @test cfg_a.tau === nothing
    @test cfg_a.budget === nothing

    # Stage B: with smoothness
    cfg_b = DarcyProblemConfig(
        "dummy_fno.npz", y_tar, x0;
        x_lb=0.0, x_ub=5.0, lambda_reg=1e-2,
        tau=100.0, grid_nx=8,
    )
    @test cfg_b.tau ≈ 100.0
    @test cfg_b.L !== nothing
    @test size(cfg_b.L) == (64, 64)
end


# ============================================================================
# Test 6: ParetoProblemConfig construction
# ============================================================================

@testset "ParetoProblemConfig" begin
    n = 10
    x0 = zeros(n)

    cfg = ParetoProblemConfig(
        "f1.pt", "f2.pt", x0;
        h_path="h.pt", alpha=0.3,
        x_lb=-1.0, x_ub=1.0,
    )
    @test cfg.alpha ≈ 0.3
    @test cfg.h_path == "h.pt"
    @test cfg.x_lb == fill(-1.0, n)

    @test_throws AssertionError ParetoProblemConfig("f1.pt", "f2.pt", x0; alpha=1.5)
end


# ============================================================================
# Test 7: Pareto quality metrics
# ============================================================================

@testset "Pareto quality metrics" begin
    # Simple 2-D front
    front = [(1.0, 4.0), (2.0, 2.0), (4.0, 1.0)]
    ref = (5.0, 5.0)
    hv = pareto_hypervolume(front, ref)
    # Expected: sort by f1, accumulate dominance
    # point (1,4): width=(5-1)=4, height=(5-4)=1 → area=4
    # point (2,2): width=(5-2)=3, height=(4-2)=2 → area=6
    # point (4,1): width=(5-4)=1, height=(2-1)=1 → area=1
    @test isapprox(hv, 4.0 + 6.0 + 1.0)

    # Empty front → 0
    @test pareto_hypervolume([], (5.0, 5.0)) == 0.0

    # Spread on collinear front should be low
    collinear = [(Float64(i), Float64(10 - i)) for i in 0:10]
    sp = pareto_spread(collinear)
    @test 0.0 <= sp <= 2.0  # sanity bound
end


# ============================================================================
# Test 8: NLPModels callback correctness on synthetic Flux model
# ============================================================================

@testset "Flux NLP callbacks" begin
    nlp, W, b, x0, target = make_tiny_flux_nlp(n=8, m_out=3, seed=7)

    x = x0 + 0.05 .* ones(8)

    # Objective: ‖Wx + b - target‖²
    nn_out = W * x + b
    expected_obj = sum((nn_out .- target).^2)
    @test isapprox(NLPModels.obj(nlp, x), expected_obj; rtol=1e-8)

    # Gradient: finite-difference check
    g = zeros(8)
    NLPModels.grad!(nlp, x, g)
    h_fd = 1e-5
    for i in 1:8
        xp, xm = copy(x), copy(x)
        xp[i] += h_fd; xm[i] -= h_fd
        g_fd_i = (NLPModels.obj(nlp, xp) - NLPModels.obj(nlp, xm)) / (2h_fd)
        @test isapprox(g[i], g_fd_i; rtol=1e-4)
    end

    # Hessian: must be symmetric PSD (for squared misfit, H = 2WᵀW)
    y_mult = zeros(0)
    nnzh = nlp.meta.nnzh
    vals = zeros(nnzh)
    rows = zeros(Int, nnzh); cols = zeros(Int, nnzh)
    NLPModels.hess_structure!(nlp, rows, cols)
    NLPModels.hess_coord!(nlp, x, y_mult, vals)

    n = nlp.meta.nvar
    H = zeros(n, n)
    for k in 1:nnzh
        H[rows[k], cols[k]] = vals[k]
        H[cols[k], rows[k]] = vals[k]
    end
    λ = eigvals(Symmetric(H))
    @test all(λ .>= -1e-8)  # PSD
end


# ============================================================================
# Test 9: _flatten_constraints helper
# ============================================================================

@testset "_flatten_constraints helper" begin
    n = 5
    bc = BudgetConstraint(10.0)
    sc = SmoothnessConstraint(laplacian_1d(n), 1.0)
    sph = SphericalConstraint(zeros(n), 2.0)

    # Test via the exported composite constraint behaviour
    @test num_constraints(bc) == 1
    @test num_constraints(sc) == 1

    comp = CompositeConstraint(bc, sc, sph)
    @test num_constraints(comp) == 3

    # Box extraction
    box = BoxConstraints(fill(-1.0, n), fill(1.0, n))
    comp2 = CompositeConstraint(box, sph)
    # After _extract_bounds_and_remaining, box should become lvar/uvar
    remaining, lvar, uvar = MadNLP4NN._extract_bounds_and_remaining(comp2, n, fill(-Inf, n), fill(Inf, n))
    @test all(lvar .== -1.0)
    @test all(uvar .== 1.0)
    @test remaining isa SphericalConstraint
end

println("\nAll tests passed!")
