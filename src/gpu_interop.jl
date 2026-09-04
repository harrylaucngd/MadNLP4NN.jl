"""CUDA.jl/JAX zero-copy interoperability helpers based on DLPack."""

const _JAX_BRIDGE_LOCK = ReentrantLock()
const _JAX_CAPSULE_BRIDGES = Dict{Int, Py}()

# DLPack.jl 0.3.1 refers to CUDA.Mem, which was removed in CUDA.jl 6. This
# concrete method is more specific than DLPack's StridedCuArray method. Keep the
# workaround local and remove it after DLPack.jl gains CUDA 6 support.
function DLPack.dldevice(x::CuArray)
    return DLPack.DLDevice(
        DLPack.kDLCUDA,
        Cint(CUDA.deviceid(CUDA.device(x))),
    )
end

function _jax_capsule_bridge(device_id::Int)
    lock(_JAX_BRIDGE_LOCK) do
        return get!(_JAX_CAPSULE_BRIDGES, device_id) do
            namespace = PythonCall.pybuiltins.dict()
            PythonCall.pybuiltins.exec(
                """
class _MadNLP4NNDLPackArray:
    def __init__(self, capsule, device_id):
        self.capsule = capsule
        self.device_id = device_id

    def __dlpack__(self, *, stream=None, max_version=None,
                   dl_device=None, copy=None):
        return self.capsule

    def __dlpack_device__(self):
        return (2, self.device_id)  # kDLCUDA

def _madnlp4nn_jax_from_capsule(capsule, device_id):
    import jax
    import jax.dlpack
    jax.config.update("jax_enable_x64", True)
    return jax.dlpack.from_dlpack(
        _MadNLP4NNDLPackArray(capsule, device_id), copy=False
    )
""",
                namespace,
            )
            functools = pyimport("functools")
            return functools.partial(
                namespace["_madnlp4nn_jax_from_capsule"],
                device_id=device_id,
            )
        end
    end
end


"""
    _share_to_jax(x::CuArray; synchronize=true)

Expose a CUDA.jl array to JAX without copying. Synchronization is conservative
because the current protocol adapter does not yet exchange producer/consumer
stream events.
"""
function _share_to_jax(x::CuArray; synchronize::Bool=true)
    synchronize && CUDA.synchronize()
    device_id = CUDA.deviceid(CUDA.device(x))
    return DLPack.share(x, _jax_capsule_bridge(device_id))
end


"""
    _copy_jax_to_cuda!(destination, source; synchronize=true)

Wrap a JAX CUDA result without a host copy, then copy it device-to-device into
the preallocated MadNLP callback buffer. Matrix dimensions are reversed by
DLPack to preserve row-major memory order; flattening the view therefore gives
the row-major coordinate order used by the dense Jacobian callback.
"""
function _copy_jax_to_cuda!(
    destination::CuVector,
    source::Py;
    synchronize::Bool=true,
)
    source.block_until_ready()
    source_view = DLPack.from_dlpack(source)
    length(source_view) == length(destination) || error(
        "JAX result has $(length(source_view)) values, expected $(length(destination))",
    )
    copyto!(destination, vec(source_view))
    synchronize && CUDA.synchronize()
    return destination
end


function _jax_pointer(array::Py)
    return pyconvert(UInt, array.unsafe_buffer_pointer())
end


"""Verify that a CUDA vector is shared with JAX by pointer and mutation."""
function verify_jax_dlpack(x::CuVector)
    isempty(x) && return true
    py_x = _share_to_jax(x)
    pointer_equal = UInt(pointer(x)) == _jax_pointer(py_x)
    original = Array(view(x, 1:1))[1]
    view(x, 1:1) .+= one(eltype(x))
    CUDA.synchronize()
    mutation_visible = pyconvert(Float64, py_x[0].item()) == original + 1
    view(x, 1:1) .-= one(eltype(x))
    CUDA.synchronize()
    PythonCall.pydel!(py_x)
    return pointer_equal && mutation_visible
end
