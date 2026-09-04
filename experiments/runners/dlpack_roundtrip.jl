#!/usr/bin/env julia

"""
Validate and benchmark zero-copy CUDA buffer exchange between CUDA.jl and JAX.

This is an interoperability probe, not yet a MadNLP callback implementation.
It verifies pointer identity and mutation visibility in both directions before
comparing DLPack wrapping with the host-staged path used by the current code.
"""

using CUDA
using DLPack
using JSON3
using PythonCall
using Statistics
using Dates
using Sockets

CUDA.allowscalar(false)

# DLPack.jl 0.3.1 still inspects the pre-CUDA-6 `CUDA.Mem` buffer names. The
# concrete CuArray method below is more specific than its generic
# `StridedCuArray` method and supplies the same DLPack device descriptor using
# the public CUDA 6 device API. Remove this shim once upstream supports CUDA 6.
function DLPack.dldevice(x::CuArray)
    return DLPack.DLDevice(
        DLPack.kDLCUDA,
        Cint(CUDA.deviceid(CUDA.device(x))),
    )
end

const jax = pyimport("jax")
const jnp = pyimport("jax.numpy")
const np = pyimport("numpy")
const jax_dlpack = pyimport("jax.dlpack")
jax.config.update("jax_enable_x64", true)
jax.config.update("jax_default_matmul_precision", "highest")

# JAX 0.10 no longer accepts a bare legacy PyCapsule. DLPack.jl creates that
# capsule and its PythonCall extension has a protocol-wrapper fallback, but the
# fallback currently only recognizes the older AttributeError rather than the
# TypeError raised by JAX 0.10. This small Python adapter provides the current
# protocol without copying the buffer.
const _bridge_namespace = PythonCall.pybuiltins.dict()
PythonCall.pybuiltins.exec(
    """
class _JuliaDLPackArray:
    def __init__(self, capsule, device_id):
        self.capsule = capsule
        self.device_id = device_id

    def __dlpack__(self, *, stream=None, max_version=None,
                   dl_device=None, copy=None):
        return self.capsule

    def __dlpack_device__(self):
        return (2, self.device_id)  # kDLCUDA

def _jax_from_julia_capsule(capsule):
    import jax.dlpack
    return jax.dlpack.from_dlpack(_JuliaDLPackArray(capsule, 0), copy=False)
""",
    _bridge_namespace,
)
const jax_from_julia_capsule = _bridge_namespace["_jax_from_julia_capsule"]

share_to_jax(x::CuArray) = DLPack.share(x, jax_from_julia_capsule)


function parse_args(args)
    output = "output/runs/dlpack_roundtrip.json"
    repetitions = 20
    for (index, arg) in enumerate(args)
        if arg == "--output"
            output = args[index + 1]
        elseif arg == "--repetitions"
            repetitions = parse(Int, args[index + 1])
        end
    end
    repetitions > 0 || error("--repetitions must be positive")
    return output, repetitions
end


function distribution(values::Vector{Float64})
    return Dict(
        "median_s" => median(values),
        "q1_s" => quantile(values, 0.25),
        "q3_s" => quantile(values, 0.75),
        "min_s" => minimum(values),
        "max_s" => maximum(values),
    )
end


function jax_pointer(array::Py)
    return pyconvert(UInt, array.unsafe_buffer_pointer())
end


function verify_interop()
    x = CuArray(Float64.(1:16))
    CUDA.synchronize()
    py_x = share_to_jax(x)
    julia_input_pointer = UInt(pointer(x))
    jax_input_pointer = jax_pointer(py_x)
    input_pointer_equal = julia_input_pointer == jax_input_pointer

    x .+= 10.0
    CUDA.synchronize()
    julia_input_mutation_visible = pyconvert(Float64, py_x[0].item()) == 11.0
    x .-= 10.0
    CUDA.synchronize()

    # JAX reads the Julia-owned CUDA buffer and allocates a new result buffer.
    py_y = jnp.square(py_x)
    py_y.block_until_ready()
    y = DLPack.from_dlpack(py_y)
    y isa CuArray || error("JAX CUDA output was not wrapped as a CuArray")
    julia_output_pointer = UInt(pointer(y))
    jax_output_pointer = jax_pointer(py_y)
    output_pointer_equal = julia_output_pointer == jax_output_pointer
    values_correct = Array(y) == collect(Float64, 1:16) .^ 2

    # Mutating the Julia view must be visible through the original JAX object.
    y .+= 3.0
    CUDA.synchronize()
    mutation_visible = pyconvert(Float64, py_y[0].item()) == 4.0

    return Dict{String,Any}(
        "input_pointer_equal" => input_pointer_equal,
        "julia_input_mutation_visible_in_jax" => julia_input_mutation_visible,
        "output_pointer_equal" => output_pointer_equal,
        "values_correct" => values_correct,
        "julia_mutation_visible_in_jax" => mutation_visible,
        "julia_input_pointer" => string(julia_input_pointer),
        "jax_input_pointer" => string(jax_input_pointer),
        "julia_output_pointer" => string(julia_output_pointer),
        "jax_output_pointer" => string(jax_output_pointer),
    )
end


function benchmark_size(n::Int, repetitions::Int)
    x = CUDA.rand(Float64, n)
    CUDA.synchronize()
    py_x = share_to_jax(x)
    py_y = jnp.sin(py_x) + 2.0 * py_x
    py_y.block_until_ready()
    y_view = DLPack.from_dlpack(py_y)
    input_pointer_equal = UInt(pointer(x)) == jax_pointer(py_x)
    output_pointer_equal = UInt(pointer(y_view)) == jax_pointer(py_y)

    share_times = Float64[]
    wrap_times = Float64[]
    staged_times = Float64[]

    for _ in 1:repetitions
        CUDA.synchronize()
        start = time_ns()
        py_shared = share_to_jax(x)
        py_shared.block_until_ready()
        push!(share_times, (time_ns() - start) / 1e9)

        py_y.block_until_ready()
        start = time_ns()
        y_shared = DLPack.from_dlpack(py_y)
        CUDA.synchronize()
        push!(wrap_times, (time_ns() - start) / 1e9)

        py_y.block_until_ready()
        start = time_ns()
        host_numpy = np.asarray(py_y)
        host_vector = pyconvert(Vector{Float64}, host_numpy)
        y_staged = CuArray(host_vector)
        CUDA.synchronize()
        push!(staged_times, (time_ns() - start) / 1e9)

        # Keep the objects live until their work has completed.
        length(y_shared) == n || error("DLPack result length mismatch")
        length(y_staged) == n || error("staged result length mismatch")
    end

    return Dict(
        "n" => n,
        "bytes" => n * sizeof(Float64),
        "input_pointer_equal" => input_pointer_equal,
        "output_pointer_equal" => output_pointer_equal,
        "julia_to_jax_dlpack" => distribution(share_times),
        "jax_to_julia_dlpack" => distribution(wrap_times),
        "jax_to_host_to_julia_gpu" => distribution(staged_times),
    )
end


function main()
    output, repetitions = parse_args(ARGS)
    CUDA.functional() || error("CUDA.jl is not functional")
    checks = verify_interop()
    println("DLPack checks: ", checks)
    required_checks = (
        "input_pointer_equal",
        "julia_input_mutation_visible_in_jax",
        "output_pointer_equal",
        "values_correct",
        "julia_mutation_visible_in_jax",
    )
    all(Bool(checks[key]) for key in required_checks) ||
        error("DLPack correctness checks failed: $checks")

    sizes = [1_024, 65_536, 1_048_576, 4_194_304]
    results = [benchmark_size(n, repetitions) for n in sizes]
    payload = Dict(
        "schema_version" => 1,
        "created_at" => string(Dates.now()),
        "repetitions" => repetitions,
        "checks" => checks,
        "provenance" => Dict(
            "hostname" => gethostname(),
            "julia" => string(VERSION),
            "cuda" => string(pkgversion(CUDA)),
            "dlpack" => string(pkgversion(DLPack)),
            "pythoncall" => string(pkgversion(PythonCall)),
            "jax" => pyconvert(String, jax.__version__),
            "jax_devices" => string(jax.devices()),
            "cuda_device" => string(CUDA.device()),
        ),
        "results" => results,
    )

    mkpath(dirname(output))
    open(output, "w") do io
        JSON3.pretty(io, payload)
        write(io, '\n')
    end
    println("Wrote ", output)
end


main()
