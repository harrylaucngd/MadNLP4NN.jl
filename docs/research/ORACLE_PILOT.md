# JAX oracle pilot on the primary H100 system

Date: 2026-08-21. This is a diagnostic pilot, not a paper result.

Hardware: NVIDIA H100 80GB HBM3 and two Intel Xeon Platinum 8562Y+ sockets.
Software: Python 3.11.15, JAX/JAXlib 0.10.0, float64 enabled. GPU memory
preallocation was disabled. Each GPU median below uses five post-compilation
calls; each CPU median uses three. CPU affinity and thread count were not yet
pinned, so CPU/GPU ratios are directional rather than final.

## What was measured

A smooth tanh MLP defines an objective and `m` inequality constraints. The
benchmark JIT-compiles and separately times network evaluation, gradient,
constraint Jacobian, and the dense Hessian of the scalar Lagrangian. GPU kernel
completion is synchronized. Device-to-host materialization is timed only after
the device result is complete.

Raw files are in the ignored run directory:

- `output/runs/oracle_quick_cpu.json`
- `output/runs/oracle_quick_gpu.json`
- `output/runs/oracle_regime_map_cpu.json`
- `output/runs/oracle_regime_map_gpu.json`

## Decision-dimension slice

Here `m=16`, width is 512, and depth is four. `p` varies modestly because the
input layer grows with `n`.

| n | p | CPU Hessian (ms) | GPU Hessian kernel (ms) | GPU D2H (ms) | Kernel speedup | Host-staged speedup |
|---:|---:|---:|---:|---:|---:|---:|
| 64 | 0.829M | 2.413 | 0.173 | 0.084 | 13.9x | 9.4x |
| 256 | 0.928M | 4.573 | 0.249 | 0.247 | 18.4x | 9.2x |
| 784 | 1.198M | 9.288 | 0.318 | 0.546 | 29.2x | 10.7x |
| 1024 | 1.321M | 14.840 | 0.370 | 0.920 | 40.1x | 11.5x |
| 2048 | 1.845M | 56.385 | 0.739 | 8.124 | 76.3x | 6.4x |

The dense Hessian is 32 MiB at `n=2048`. Its host transfer is 11 times longer
than the H100 Hessian kernel and causes the staged speedup to fall even though
the raw compute speedup continues to increase.

## Network-parameter slice

Here `n=784`, `m=10`, and depth is four.

| Width | p | CPU Hessian (ms) | GPU Hessian kernel (ms) | GPU D2H (ms) | Host-staged speedup |
|---:|---:|---:|---:|---:|---:|
| 64 | 0.063M | 2.605 | 0.158 | 0.559 | 3.6x |
| 256 | 0.401M | 5.375 | 0.310 | 0.647 | 5.6x |
| 1024 | 3.963M | 26.870 | 0.616 | 0.584 | 22.4x |
| 4096 | 53.600M | 225.729 | 5.618 | 0.613 | 36.2x |

Unlike the `n` slice, the transferred Hessian size is fixed. GPU benefit grows
with neural compute and is not erased by staging. This confirms that `n` and
`p` must be varied independently.

## Output-dimension slice

At `n=256`, width 512, and depth three, increasing `m` from 1 to 256 changed
the GPU Lagrangian-Hessian kernel only from 0.161 ms to 0.180 ms because the
weighted constraint outputs are differentiated as one scalar. The GPU Jacobian
kernel changed from 0.092 ms to 0.112 ms, while CPU Jacobian time grew from
0.260 ms to 1.086 ms. This slice must be extended to larger `m` and checked
against the actual benchmark architectures before drawing a general conclusion.

## Immediate consequences

1. A zero-copy GPU path is scientifically motivated for large `n`; it is not
   merely an implementation cleanup.
2. Exact Hessians remain plausible for moderate `n` with large `p`, but explicit
   storage/transfer is the wrong scaling path for neural-operator inputs.
3. L-BFGS or a structured/matrix-free curvature model is mandatory before
   attempting 64x64 and larger field inversions.
4. Cold JIT times were roughly 0.3--0.7 seconds for most shapes and 5.2 seconds
   for the 53.6M-parameter case. Cold and warm workloads must be reported
   separately.
5. Small first-order oracle calls can favor CPU because dispatch dominates;
   “GPU is faster” is not a universal claim.

## DLPack interoperability pilot

The CUDA.jl/JAX DLPack probe verified pointer identity and mutation visibility
in both directions. JAX float64 must be enabled before importing a Julia
`CuArray{Float64}`; with x64 disabled, the same check correctly detected a
conversion/copy.

For JAX output transferred back into Julia/CUDA, the measured median wrapping
time was 12 microseconds for 8 KiB, 15 microseconds for 0.5 MiB, 30 microseconds
for 8 MiB, and 40 microseconds for 32 MiB. The existing staged path (JAX to
NumPy to a Julia host vector to `CuArray`) took respectively 24 microseconds,
210 microseconds, 2.68 milliseconds, and 10.90 milliseconds. At 32 MiB the
zero-copy wrapper is about 272x faster than staging. Pointer identity passed at
every tested size.

This validates the interoperability mechanism but not yet an end-to-end solver
speedup. MadNLP callbacks must still copy the foreign JAX device view into their
preallocated derivative buffers, and the effect of synchronization and cuDSS
factorization remains to be measured.
