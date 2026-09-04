# Published PFR-NMPC benchmark audit

Source: companion repository for Elorza Casas, Ricardez-Sandoval, and Pulsipher
(2025), cloned from
<https://git.uwaterloo.ca/ricardez_group/benchmarking-nn-surrogate-embedding-methods-in-nmpc>.

## Case study 1 structure

- Ten discretized plug-flow-reactor states and two manipulated variables.
- One-step tanh surrogate: 12 inputs, six hidden layers of width 24, ten
  outputs; normalization bounds are stored in the PyTorch state dict.
- Published gray-box MPC uses prediction horizon 40 and control horizon 10.
- NLP input vector contains 41x10 states plus 10x2 controls (430 variables),
  with 40x10 learned dynamic equality constraints.
- The receding-horizon artifact contains 100 optimization solves and the
  corresponding mechanistic closed-loop trajectory.

The saved `GreyboxNN/UsingFunc` artifact reports median wall-clock solve time
0.269 s, median Jacobian time 0.058 s, and median Hessian time 0.207 s on the
authors' CPU environment. The full-space NN, reduced-space NN, gray-box NN, and
mechanistic trajectories are all present for cross-checking.

## Reproducibility issues to resolve

- Scripts contain absolute Windows paths and do not provide a locked
  environment or useful repository README.
- The gray-box script forces PyTorch to CPU even when CUDA is available.
- Pyomo/PyNumero/Ipopt versions are stated in the paper but not installed in
  this repository's current environment.
- Timer arrays mix Python process time and wall time; our port will preserve raw
  published outputs but use one synchronized protocol.

These are porting issues, not reasons to discard the benchmark. It remains much
more representative than the synthetic Pareto case because it has a published
problem definition, pretrained models, mechanistic simulator, and full closed-
loop reference trajectories.

## Reproduction gates

1. Load the released state dict and match PyTorch one-step predictions in
   float64.
2. Run the upstream Pyomo gray-box case on Linux and match the saved first MPC
   solution and closed-loop trajectory within declared tolerances.
3. Express the same 430-variable/400-constraint NLP through the generic JAX
   oracle; compare objective, constraints, Jacobian, and Lagrangian-Hessian
   directional products.
4. Compare Ipopt/PyNumero, MadNLP CPU KKT, host-staged GPU KKT, and DLPack GPU
   KKT without changing horizon, warm start, or constraints.
5. Only after gates 1--4 pass, consider a control-only unrolled reduced-space
   formulation as a separate formulation ablation.

## Gate results on the primary H100 system

- Gate 1 passes: JAX/PyTorch relative errors are `6.1e-7` in float32 and
  `9.3e-16` in float64.
- Gate 2 passes for the gray-box formulation. The first MadNLP control differs
  from the published Pyomo/PyNumero control by at most `4.7e-7`. A 100-step
  closed loop reaches common KKT on every step and stays within about `3.6e-5`
  of the published state trajectory; part of this difference is the published
  script's default-tolerance `solve_ivp` integration.
- Gate 3 passes. Central directional errors are `3.7e-11` (objective),
  `3.4e-10` (constraint JVP), and `4.6e-11` (Lagrangian Hessian-vector).
- Gate 4 passes for same-oracle MadNLP and cyipopt exact/limited-memory paths.
  The exact methods agree in objective, KKT residuals, and first control.

The true gray-box structure is 5,200 Jacobian and 3,049 lower-Hessian entries,
not the 172,000/92,665 dense patterns used in the first port. Declaring correct
sparsity reduces first-step MadNLP CPU exact from about 0.59 s to 0.043 s and
cyipopt exact from 0.56 s to 0.044 s. This invalidates any PFR comparison based
on the initial dense declaration.

The OMLT full-space port is numerically validated at horizon 5: its objective
differs from the gray-box solution by `4.6e-11`. It takes 1.24 s versus 0.019 s
for that smoke problem and about 21.2 s at the published horizon 40. OMLT
reduced-space on maintained OMLT 1.1 and 1.2 does not finish building even the
horizon-5 model within 180 s in this environment. The authors' saved old-
environment reduced-space trajectory remains useful historical evidence, but
is not labeled a same-node reproduction.

## Cases 2 and 3 scale gate

The released case-2 and case-3 CNN state dictionaries were translated directly
instead of approximating them with new models. Float64 JAX/PyTorch relative
forward errors are `3.1e-16` and `1.4e-16`. Both structured NLP ports pass
finite-difference objective, JVP, and Lagrangian Hessian-vector gates; the
largest relative discrepancy is `1.6e-7`.

| case | variables | equalities | Jacobian nnz | lower-Hessian nnz | exact CPU warm | exact DLPack warm |
|---:|---:|---:|---:|---:|---:|---:|
| 2 | 860 | 800 | 20,000 | 11,738 | 0.204 s | **0.130 s** |
| 3 | 7,780 | 7,500 | 1,905,000 | 963,840 | 19.71 s | **2.27 s** |

All retained solves meet a `1e-6` common KKT/raw-residual audit and reproduce
the published first controls. Case 3 is the first established application
where moving the KKT system, rather than only neural AD, is decisive: cuDSS is
8.7x faster than CPU MUMPS in the five-run warm comparison. The resource trade-off is
equally material: 43.90 GiB peak GPU memory for DLPack/cuDSS versus 1.14 GiB
when JAX AD is staged to CPU KKT. Cold DLPack elapsed time is 71.7 s, so the
result applies to persistent or amortized service, not one-shot execution.

Same-oracle cyipopt exact takes 0.499 s for case 2 and 27.84 s for case 3;
MadNLP CPU exact takes 0.204 s and 19.71 s. This independently confirms the
case-3 CPU factorization regime. Approximate curvature is not competitive:
cyipopt limited-memory fails after 500 iterations on case 2, while MadNLP
compact L-BFGS reaches the case-3 common `1e-6` audit in 271 iterations and
530.36 s. Its linear solves consume 516.24 s even though no Hessian callback is
made, versus 2.27 s total for exact DLPack/cuDSS.

Only the first control problem is ported for cases 2/3. Case 1 remains the
closed-loop deployment benchmark because its 100-step mechanistic replay has
been independently run; cases 2/3 are used as a published, architecture- and
horizon-preserving scale axis rather than as closed-loop claims.
