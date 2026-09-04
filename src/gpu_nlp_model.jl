"""
GPU-resident NLPModels adapter for the JAX evaluator.

Unlike `MadNLP.SparseWrapperModel(CuArray, host_model)`, this adapter shares the
MadNLP primal vector with JAX through DLPack and keeps derivative coordinates on
the GPU. The first implementation synchronizes at framework boundaries for
correctness; stream-aware asynchronous exchange is a later optimization.
"""

mutable struct DLPackGPUModel{VT<:CuVector{Float64}, I<:NeuralNetworkNLPModel} <:
               AbstractNLPModel{Float64, VT}
    meta::NLPModelMeta{Float64, VT}
    counters::Counters
    inner::I
    jac_rows::CuVector{Int}
    jac_cols::CuVector{Int}
    hess_rows::CuVector{Int}
    hess_cols::CuVector{Int}
    timing_stats::TimingStats
end


function DLPackGPUModel(inner::NeuralNetworkNLPModel; verify_interop::Bool=true)
    inner.use_python || error("DLPackGPUModel requires a Python/JAX evaluator")
    CUDA.functional() || error("DLPackGPUModel requires a functional CUDA device")
    evaluator_is_gpu = pyconvert(Bool, inner.python_evaluator.dm.is_gpu)
    evaluator_is_gpu || error("The JAX evaluator must be configured with device=\"gpu\"")

    n = inner.meta.nvar
    m = inner.meta.ncon
    x0 = CuArray(inner.meta.x0)
    meta = NLPModelMeta(
        n;
        x0=x0,
        lvar=CuArray(inner.meta.lvar),
        uvar=CuArray(inner.meta.uvar),
        ncon=m,
        y0=CuArray(inner.meta.y0),
        lcon=CuArray(inner.meta.lcon),
        ucon=CuArray(inner.meta.ucon),
        nnzj=inner.meta.nnzj,
        nnzh=inner.meta.nnzh,
        minimize=inner.meta.minimize,
    )

    jacobian_structure = inner.python_evaluator.jacobian_structure()
    jac_rows_host = pyconvert(Vector{Int}, jacobian_structure[0]) .+ 1
    jac_cols_host = pyconvert(Vector{Int}, jacobian_structure[1]) .+ 1
    hessian_structure = inner.python_evaluator.hessian_structure()
    hess_rows_host = pyconvert(Vector{Int}, hessian_structure[0]) .+ 1
    hess_cols_host = pyconvert(Vector{Int}, hessian_structure[1]) .+ 1

    model = DLPackGPUModel(
        meta,
        Counters(),
        inner,
        CuArray(jac_rows_host),
        CuArray(jac_cols_host),
        CuArray(hess_rows_host),
        CuArray(hess_cols_host),
        TimingStats(),
    )
    verify_interop && !verify_jax_dlpack(x0) &&
        error("CUDA.jl/JAX DLPack pointer or mutation check failed")
    return model
end


function NLPModels.obj(nlp::DLPackGPUModel, x::CuVector)
    NLPModels.increment!(nlp, :neval_obj)
    objective = 0.0
    elapsed = @elapsed begin
        py_x = _share_to_jax(x)
        py_objective = nlp.inner.python_evaluator.evaluate_obj_device(py_x)
        py_objective.block_until_ready()
        objective = pyconvert(Float64, py_objective.item())
        PythonCall.pydel!(py_x)
        PythonCall.pydel!(py_objective)
    end
    record_eval_time!(nlp.timing_stats, elapsed, :obj)
    return objective
end


function NLPModels.grad!(nlp::DLPackGPUModel, x::CuVector, gradient::CuVector)
    NLPModels.increment!(nlp, :neval_grad)
    elapsed = @elapsed begin
        py_x = _share_to_jax(x)
        py_gradient = nlp.inner.python_evaluator.evaluate_grad_device(py_x)
        _copy_jax_to_cuda!(gradient, py_gradient)
        PythonCall.pydel!(py_x)
    end
    record_eval_time!(nlp.timing_stats, elapsed, :grad)
    return gradient
end


function NLPModels.cons!(nlp::DLPackGPUModel, x::CuVector, constraints::CuVector)
    NLPModels.increment!(nlp, :neval_cons)
    isempty(constraints) && return constraints
    elapsed = @elapsed begin
        py_x = _share_to_jax(x)
        py_constraints = nlp.inner.python_evaluator.evaluate_cons_device(py_x)
        _copy_jax_to_cuda!(constraints, py_constraints)
        PythonCall.pydel!(py_x)
    end
    record_eval_time!(nlp.timing_stats, elapsed, :cons)
    return constraints
end


function NLPModels.jac_structure!(
    nlp::DLPackGPUModel,
    rows::CuVector{<:Integer},
    cols::CuVector{<:Integer},
)
    copyto!(rows, nlp.jac_rows)
    copyto!(cols, nlp.jac_cols)
    return rows, cols
end


function NLPModels.jac_coord!(
    nlp::DLPackGPUModel,
    x::CuVector,
    values::CuVector,
)
    NLPModels.increment!(nlp, :neval_jac)
    isempty(values) && return values
    elapsed = @elapsed begin
        py_x = _share_to_jax(x)
        py_jacobian = nlp.inner.python_evaluator.evaluate_jac_coord_device(py_x)
        _copy_jax_to_cuda!(values, py_jacobian)
        PythonCall.pydel!(py_x)
    end
    record_eval_time!(nlp.timing_stats, elapsed, :jac)
    return values
end


function NLPModels.jac_dense!(
    nlp::DLPackGPUModel,
    x::CuVector,
    jacobian::CuMatrix,
)
    NLPModels.increment!(nlp, :neval_jac)
    isempty(jacobian) && return jacobian
    elapsed = @elapsed begin
        py_x = _share_to_jax(x)
        py_jacobian = nlp.inner.python_evaluator.evaluate_jac_dense_device(py_x)
        py_jacobian.block_until_ready()
        # A row-major JAX (m,n) array is exposed as a column-major Julia (n,m)
        # view. Its transpose has the requested semantic shape (m,n).
        source = DLPack.from_dlpack(py_jacobian)
        copyto!(jacobian, transpose(source))
        CUDA.synchronize()
        PythonCall.pydel!(py_x)
    end
    record_eval_time!(nlp.timing_stats, elapsed, :jac)
    return jacobian
end


function NLPModels.hess_dense!(
    nlp::DLPackGPUModel,
    x::CuVector,
    y::CuVector,
    hessian::CuMatrix;
    obj_weight=1.0,
)
    NLPModels.increment!(nlp, :neval_hess)
    elapsed = @elapsed begin
        py_x = _share_to_jax(x)
        py_y = _share_to_jax(y)
        py_hessian = nlp.inner.python_evaluator.evaluate_hess_dense_device(
            py_x,
            py_y,
            obj_weight,
        )
        py_hessian.block_until_ready()
        source = DLPack.from_dlpack(py_hessian)
        # The Hessian is symmetric, so DLPack's row/column-major transpose does
        # not change its values.
        copyto!(hessian, source)
        CUDA.synchronize()
        PythonCall.pydel!(py_x)
        PythonCall.pydel!(py_y)
    end
    record_eval_time!(nlp.timing_stats, elapsed, :hess)
    return hessian
end


function NLPModels.jprod!(
    nlp::DLPackGPUModel,
    x::CuVector,
    vector::CuVector,
    product::CuVector,
)
    NLPModels.increment!(nlp, :neval_jprod)
    isempty(product) && return product
    elapsed = @elapsed begin
        py_x = _share_to_jax(x)
        py_vector = _share_to_jax(vector)
        py_product = nlp.inner.python_evaluator.evaluate_jprod_device(py_x, py_vector)
        _copy_jax_to_cuda!(product, py_product)
        PythonCall.pydel!(py_x)
        PythonCall.pydel!(py_vector)
    end
    record_eval_time!(nlp.timing_stats, elapsed, :jac)
    return product
end


function NLPModels.jtprod!(
    nlp::DLPackGPUModel,
    x::CuVector,
    vector::CuVector,
    product::CuVector,
)
    NLPModels.increment!(nlp, :neval_jtprod)
    fill!(product, 0.0)
    isempty(vector) && return product
    elapsed = @elapsed begin
        py_x = _share_to_jax(x)
        py_vector = _share_to_jax(vector)
        py_product = nlp.inner.python_evaluator.evaluate_jtprod_device(py_x, py_vector)
        _copy_jax_to_cuda!(product, py_product)
        PythonCall.pydel!(py_x)
        PythonCall.pydel!(py_vector)
    end
    record_eval_time!(nlp.timing_stats, elapsed, :jac)
    return product
end


function NLPModels.hess_structure!(
    nlp::DLPackGPUModel,
    rows::CuVector{<:Integer},
    cols::CuVector{<:Integer},
)
    copyto!(rows, nlp.hess_rows)
    copyto!(cols, nlp.hess_cols)
    return rows, cols
end


function NLPModels.hess_coord!(
    nlp::DLPackGPUModel,
    x::CuVector,
    y::CuVector,
    values::CuVector;
    obj_weight=1.0,
)
    NLPModels.increment!(nlp, :neval_hess)
    elapsed = @elapsed begin
        py_x = _share_to_jax(x)
        py_y = _share_to_jax(y)
        py_hessian = nlp.inner.python_evaluator.evaluate_hess_coord_device(
            py_x,
            py_y,
            obj_weight,
        )
        _copy_jax_to_cuda!(values, py_hessian)
        PythonCall.pydel!(py_x)
        PythonCall.pydel!(py_y)
    end
    record_eval_time!(nlp.timing_stats, elapsed, :hess)
    return values
end


function solve_nlp(
    nlp::DLPackGPUModel;
    max_iter=1000,
    tol=1e-6,
    print_level=MadNLP.INFO,
    linear_solver=MadNLPGPU.CUDSSSolver,
    kwargs...,
)
    options = Dict{Symbol,Any}(
        :max_iter => max_iter,
        :tol => tol,
        :print_level => print_level,
        :linear_solver => linear_solver,
    )
    merge!(options, Dict(kwargs))
    solver = MadNLPSolver(nlp; (; pairs(options)...)...)
    result = MadNLP.solve!(solver)
    complementarity = MadNLP.get_inf_compl(solver)
    kkt_error = MadNLP.get_kkt_error(solver)
    solution = Array(result.solution[1:nlp.meta.nvar])
    multipliers = Array(result.multipliers)
    return Dict(
        :status => result.status,
        :solution => solution,
        :multipliers => multipliers,
        :objective => result.objective,
        :iter_count => result.iter,
        :primal_feas => result.primal_feas,
        :dual_feas => result.dual_feas,
        :complementarity => complementarity,
        :kkt_error => kkt_error,
        :solver_counters => result.counters,
        :nlp_model => nlp,
        :timing_stats => nlp.timing_stats,
        :kkt_device => "gpu",
        :host_staged => false,
        :zero_copy => true,
    )
end
