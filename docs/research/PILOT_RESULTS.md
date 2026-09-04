# Pilot results and decisions

Status: finalized decision log for the completed August 2026 snapshot. Early
sections record diagnostic pilots; later sections record the registered sweeps
used by the paper. The controlled exact-path table uses 10 warm repetitions,
NUMA node 0, and one BLAS/OMP thread.

## 1. Controlled MLP: exact-Hessian execution paths

All 18 warm solves reached the same objective in seven iterations with KKT
error about `9.1e-8`. `gpu_ad_cpu_kkt` already evaluates the JAX network and
derivatives on H100; only the KKT path is on CPU. Thus this comparison isolates
KKT placement and host staging rather than relabeling all GPU AD as an
end-to-end GPU solver.

| n | parameters | GPU AD + CPU KKT | host-staged GPU KKT | DLPack GPU KKT |
|---:|---:|---:|---:|---:|
| 64 | 0.042M | 22.6 ms | 56.8 ms | 43.1 ms |
| 256 | 0.201M | 36.2 ms | 67.0 ms | 56.5 ms |
| 784 | 1.195M | 198.8 ms | 136.1 ms | 95.2 ms |

At `n=784`, DLPack reduces median callback time from 49.8 ms to 17.7 ms and
total time by about 30% relative to host staging; it is about 2.1x faster than
the CPU-KKT path. GPU initialization is still about 47 ms and limits small-
instance performance. At `n=64` and `n=256`, CPU KKT is the clear winner.

## 2. Controlled MLP: exact versus compact L-BFGS on CPU KKT

The AD oracle remains on H100. Compact L-BFGS removes all Hessian calls but
needs more IPM iterations.

| n | exact time | L-BFGS time | iterations exact / L-BFGS | decision |
|---:|---:|---:|---:|---|
| 64 | 22.6 ms | 107.1 ms | 7 / 29 | exact wins |
| 256 | 36.2 ms | 183.1 ms | 7 / 35 | exact wins |
| 784 | 198.8 ms | 103.9 ms | 7 / 21 | L-BFGS wins |

GPU compact L-BFGS is recorded as unsupported in MadNLP/MadNLPGPU 0.10. The
default condensed KKT rejects it, while augmented sparse and dense paths contain
CPU-only/scalar code and do not pass the required deterministic KKT audit. We
do not silently substitute a different algorithm.

## 3. PDEBench artifact validation

- Dataset: official beta=1 Darcy data, 10,000 coefficient/pressure pairs at
  128x128, MD5 `81694ed31306ff2e5f6b76349b0b4389`.
- Model: official PDEBench FNO checkpoint, MD5
  `933cab7a056b94a062288d1a35f8579c` for the five-checkpoint archive.
- The new JAX forward matches the official PyTorch forward at 32, 64, and 128
  resolution. Relative L2 discrepancies are below `2.3e-7` in float32 and
  below `1.5e-7` after promoting both implementations to float64.
- A central directional finite-difference audit on the latent-8 NLP gives
  relative errors `1.8e-12` for the objective directional derivative,
  `7.7e-13` for the constraint JVP, and `2.6e-12` for a Hessian-vector product.
- Disabling high-precision JAX matmul caused a reproducible `~5e-4` relative
  discrepancy. Precision settings are therefore part of provenance, not an
  incidental implementation detail.

Across all 1,000 official held-out beta=1 samples, the checkpoint's full-
resolution pressure relative L2 error has quartiles 14.8%, 17.3%, and 20.9%
(range 7.2%--50.7%). Predeclared representative instances are sample 9153
(q25), 9257 (median), and 9373 (q75).

## 4. PDEBench target-layout correction and benchmark rejection

The first Julia pilot flattened a 128x128 target with Julia's column-major
`vec`, while the JAX FNO flattens arrays in NumPy row-major order. MadNLP was
therefore fitting the transpose of the target; the Python/SciPy baseline was
fitting the intended field. This invalidates every earlier Julia PDEBench
solution-quality number, including the apparent 19% versus 6% solver gap and
the q25/median/q75 inversion table. Those artifacts remain in `output/runs` for
audit but are explicitly excluded from results.

After replacing the boundary conversion by `vec(permutedims(target))`, the
latent-8 MadNLP compact-L-BFGS solve and SLSQP agree closely from the same start:

| method | warm time | iterations | surrogate objective | coefficient rel. error | surrogate-to-target rel. error |
|---|---:|---:|---:|---:|---:|
| MadNLP compact L-BFGS | 0.260 s | 57 | 0.0018677 | 53.97% | 6.101% |
| SciPy SLSQP | 0.860 s | 279 | 0.0018653 | 53.99% | 6.097% |

For the corrected objective, `sqrt(2*objective)` agrees with the reported
relative pressure residual up to the small regularization term. This invariant
is now a required layout test. The SLSQP result is still above the common
stationarity target (`7.8e-6` versus `1e-6`) and is not a matched-KKT success.

The exact-Hessian memory observation survives the layout correction as a
systems result: at latent 32x32 (`n=1024`), JAX requested about 40 GiB of
working memory for the FNO Hessian even though the final dense Hessian is only
8 MiB. Exact neural-operator Hessians cannot be treated as the scalable default.

A separate source audit then found that PDEBench's beta=1 artifact is not a
clean inverse-design benchmark. The official generator evolves a random
initial field with a variable-coefficient diffusion equation for finite time;
the released FNO receives only the coefficient, and the random initial field
is not stored in the HDF5 pair. Consequently the coefficient-to-label map is
confounded and cannot be replayed independently from a released sample. We
retain the official 128x128 FNO for parity, AD, transfer, and memory smoke tests,
but move the main inverse experiment to NeuralOperator's deterministic elliptic
Darcy dataset.

## Decisions triggered by the pilots

1. The paper must center simulator-valid optimization, not only solver speed.
2. Retain three independent scale axes: neural parameter count, NLP decision
   dimension, and surrogate output dimension.
3. Exact GPU KKT is not the default recommendation. It is tested as a regime,
   with CPU KKT and compact L-BFGS mandatory.
4. Require explicit array-layout invariants at every Julia/Python boundary and
   simulator replay before any application result is admitted.
5. Add a surrogate-quality intervention (a better checkpoint or trust-region /
   validity constraint) before claiming improved inverse design.
6. Cold CUDA/cuDSS specialization is roughly 60--75 seconds in fresh Julia
   processes. Long-lived workers or a sysimage are required for honest repeated
   service-style timing; warm and cold numbers remain separate.

## 5. Discarded PDEBench regularization pilot

The earlier double-well penalty sweep used the transposed Julia target and is
invalidated with the other PDEBench inversion results. No hyperparameter will
be selected from it. Any trust-region, support, or uncertainty intervention is
selected only on the new elliptic-Darcy validation split and frozen before test.

## 6. Published PFR checkpoint translation

For PFR-NMPC case study 1, the JAX implementation matches the released PyTorch
tanh network on 100 seeded inputs. Relative L2 discrepancies are `6.1e-7` in
float32 and `9.3e-16` in float64. The network-translation gate passes; the next
gate is reproducing the published Pyomo trajectory and objective before adding
MadNLP.

Those remaining gates now pass. The 430-variable/400-equality first solve
matches the published first control within `4.7e-7`, and the derivative audit
has relative errors below `3.4e-10`. The final structure-aware first-step table
on the primary H100 system is:

| method | warm median | KKT success | note |
|---|---:|---:|---|
| MadNLP, exact, GPU AD + CPU KKT | 43.3 ms | 5/5 | 5,200 Jacobian nonzeros |
| cyipopt/Ipopt, exact, same JAX oracle | 43.9 ms | 3/3 | separate conda baseline env |
| MadNLP, exact, DLPack GPU KKT | 68.9 ms | 5/5 | faster callbacks, higher setup |
| MadNLP, exact, host-staged GPU KKT | 89.1 ms | 5/5 | transfer ablation |
| MadNLP compact L-BFGS, CPU KKT | 385 ms | 5/5 | 61 iterations |
| Ipopt limited-memory | 880 ms | 3/3 | common residual audit passes |
| OMLT full-space + Ipopt | 21.2 s | 3/3 optimal | 40 replicated networks |

The first port incorrectly declared dense Jacobian/Hessian patterns and made
the GPU path appear preferable. Correct sparsity reduces MadNLP CPU exact from
0.59 s to 0.043 s. This reversal is retained as a primary ablation: structure
disclosure matters more than device placement in this regime.

For the full 100-step mechanistic closed loop, structure-aware CPU exact reaches
100/100 KKT success with 29.3 ms median, 30.4 ms mean, and 77.5 ms maximum. A
persistent DLPack/cuDSS path has a lower 24.8 ms median, but one reuse failure
requires a fresh-solver fallback; its mean is 62.1 ms and maximum 3.24 s. The
recommended PFR path is therefore CPU KKT unless the GPU reuse tail is fixed.
Both trajectories remain within about `3.6e-5` state infinity error of the
published trajectory under the stricter RK4 plant replay.

OMLT full-space is numerically cross-checked at horizon 5 and is about 65x
slower there (1.24 s versus 19 ms). At horizon 40 it takes about 21.2 s. OMLT
1.1 and 1.2 reduced-space model construction exceeds 180 s even at horizon 5
on this stack, so the authors' older saved reduced-space timings are kept as
historical, cross-machine context rather than a same-node result.

The two released CNN cases now pass the same gates. Their translated JAX
forwards match the released PyTorch checkpoints to `3.1e-16` (case 2) and
`1.4e-16` (case 3) in float64. Directional derivative checks are below
`1.6e-7`, including sparse Lagrangian Hessian-vector products. The resulting
scale axis is not synthetic:

| published case | n / equalities | Jacobian / lower-Hessian nnz | exact CPU KKT | exact DLPack GPU KKT |
|---:|---:|---:|---:|---:|
| 1 | 430 / 400 | 5,200 / 3,049 | **43.3 ms**, 7 it | 68.9 ms, 7 it |
| 2 | 860 / 800 | 20,000 / 11,738 | 204 ms, 28 it | **130 ms**, 16 it |
| 3 | 7,780 / 7,500 | 1,905,000 / 963,840 | 19.71 s, 17 it | **2.27 s**, 11 it |

Every entry passes the common KKT and raw-constraint audit. First controls
match the released trajectories to at most `4.6e-6`, `1.8e-3`, and `1.7e-8`
in the native control units for cases 1--3; the case-2 maximum is about
`1e-6` relative to that control's magnitude. Case 3 therefore supplies an
application-level placement crossover: GPU KKT is 8.7x faster after warm-up.
Five-run IQRs are 2.25--2.32 s (GPU) and 19.71--19.91 s (CPU), while CPU
factorization consumes about 17.3--17.9 s. It is not a free win. The
case-3 exact DLPack process peaks at 43.90 GiB GPU memory versus 1.14 GiB for
CPU KKT, and its fresh-process solve takes 71.7 s because specialization and
cuDSS initialization dominate the cold path.

The same-oracle external baseline confirms that this is not a weak CPU
implementation artifact. cyipopt/Ipopt exact takes 0.499 s on case 2 and
27.84 s median on case 3, versus MadNLP CPU's 0.204 s and 19.71 s. Ipopt
limited-memory fails its 500-iteration/raw-residual audit on case 2. MadNLP
compact L-BFGS reaches the common `1e-6` case-3 audit only at the acceptable
level: 271 iterations and 530.36 s, of which 516.24 s is linear-solver time.
It peaks at only 1.14 GiB GPU memory, but is 213x slower than exact DLPack.
Approximate curvature is therefore not a universal large-problem default when
the KKT system remains large and must still be factorized every iteration.

## 7. Deterministic NeuralOperator Darcy gate

The official 64x64 Zenodo artifact contains 5,000 training and 1,000 held-out
coefficient/solution pairs. A 1.19M-parameter FNO was trained with a fixed
4,500/500 train/validation split and NeuralOperator's left-endpoint positional
grid. The selected checkpoint has mean/median held-out relative L2 errors of
3.17%/2.40%. Its Torch/JAX relative discrepancies are `1.2e-7` in float32 and
`2.4e-16` in float64.

For test instances at the q25, median, and q75 FNO-error strata, increasing the
latent field from 8x8 to 32x32 lowers the optimized surrogate residual to
0.9%--1.0%, below the FNO residual at the true coefficient (1.8%--3.6%). Yet
the recovered coefficient remains 26.8%--41.4% from truth and exceeds the best
coefficient error representable by the same latent parameterization by
11.0--24.6 percentage points. This is valid evidence that extra decision
capacity is used to exploit surrogate error; it is not confounded by the
PDEBench target-transpose bug. Multi-start and an independently solved PDE
residual remain required for the final application table.

On the median instance, all curvature/placement paths reach KKT below `1e-6`
and essentially the same local objective. Warm timings are:

| latent | exact CPU KKT | exact host-staged GPU KKT | exact DLPack GPU KKT | compact L-BFGS CPU KKT |
|---:|---:|---:|---:|---:|
| 8x8 | 0.125 s / 12 it | 0.164 s / 12 it | 0.158 s / 12 it | **0.105 s / 33 it** |
| 16x16 | 0.605 s / 18 it | 0.648 s / 18 it | 0.636 s / 18 it | **0.216 s / 38 it** |
| 32x32 | 4.716 s / 22 it | 2.898 s / 22 it | 2.752 s / 22 it | **0.961 s / 46 it** |

At latent 32, GPU exact curvature is 1.7x faster than CPU exact and DLPack is
about 5% faster than host staging. Nevertheless compact L-BFGS is another 2.9x
faster than the best exact path. Thus end-to-end GPU placement is a real win
within the exact-Hessian regime, but choosing the curvature model matters more.
The first GPU specialization took 62.8 s and remains a separately reported
cold-start cost.

The five-start audit exposed a robustness trade-off hidden by the constant
start. Compact L-BFGS succeeded on 43/45 instance-resolution starts. Both
failures occurred on the q25 instance: one at latent 16 and one at latent 32,
each reaching the 500-iteration cap with KKT errors above `3e-3`. Exact DLPack
GPU solved the same ten q25 starts 10/10. Its median times were 1.10 s versus
0.85 s for successful L-BFGS runs at latent 16, and 3.38 s versus 1.39 s at
latent 32. Exact curvature improves robustness but is not uniformly better in
solution quality: at latent 32 it converged to several distinct KKT points.

A simple registered routing policy is therefore justified for the next sweep:
run compact L-BFGS first and restart with exact curvature only on common-KKT
failure. Retrospectively on these two hard groups it preserves 5/5 success and
uses 9% (latent 16) and 30% (latent 32) less total warm time than exact on every
start. This is a pilot-derived policy and must be evaluated prospectively on
new instances; it is not yet a headline result.

That policy was then evaluated prospectively on 12 new held-out samples drawn
with seed `20260822`, excluding all three pilot samples. Across latent 16/32
and the same five starts, compact L-BFGS reached common KKT on 115/120 solves
(95.8%); failures occurred on 3/12 instances. Exact DLPack GPU reached common
KKT on all 30 starts from those three difficult instances, including all five
failed L-BFGS starts. Retrying only failures raises success to 120/120 while
adding 7.7% to total warm solve time relative to L-BFGS alone. This is now a
prospective result rather than a policy fitted and evaluated on the same cases.

The larger set also confirms surrogate exploitation. Among successful solves,
the median coefficient-error gap above the best representable latent field is
8.6 percentage points at latent 16 and 15.6 points at latent 32. The optimized
surrogate residual is 85% of the true-coefficient FNO residual at latent 16 but
only 42% at latent 32: additional capacity improves surrogate fit much faster
than coefficient recovery.

## 8. Independently replayable Darcy validation

Because the public archive's historical preprocessing remains ambiguous, a
separate registered split was generated rather than fitting undocumented
scalings. It contains 5,000 training and 1,000 test pairs from
`-div(exp(a) grad u) = 1`, zero Dirichlet boundaries, and a five-point sparse
finite-difference operator with harmonic face averaging. Train/test seeds,
coefficient sampler, clipping, checksums, and the exact simulator are recorded.
The 1.19M-parameter FNO reaches 1.32% mean and 1.16% median held-out relative
L2 error; Torch/JAX discrepancies are `1.4e-7` in float32 and `2.4e-16` in
float64. Replaying a stored truth through the simulator agrees with its target
to about `3e-8` relative error.

The registered optimization sweep uses q25/median/q75 FNO-error instances,
latent 16/32, five shared starts, and exact DLPack, Gauss--Newton DLPack, and
CPU L-BFGS. All 90 warm solves pass the common KKT audit.

| method | latent | warm median | iterations | FNO residual | PDE residual | coefficient error |
|---|---:|---:|---:|---:|---:|---:|
| exact DLPack | 16 | 0.517 s | 15 | 0.81% | 4.13% | 25.0% |
| Gauss--Newton DLPack | 16 | **0.237 s** | 12 | 0.81% | 4.06% | 24.4% |
| L-BFGS CPU | 16 | 0.979 s | 70 | 0.81% | **3.29%** | 24.9% |
| exact DLPack | 32 | 1.946 s | 16 | 0.71% | 5.03% | 37.0% |
| Gauss--Newton DLPack | 32 | **0.705 s** | 12 | **0.42%** | 5.66% | **25.6%** |
| L-BFGS CPU | 32 | 1.635 s | 46 | 0.49% | **2.37%** | 29.9% |

Every optimized FNO residual is below the FNO residual at the true coefficient,
yet PDE residuals remain 2.4%--5.7% in median. More importantly, the method
with the lowest surrogate residual is also the method with the lowest
simulator residual in only 2/30 matched instance-resolution-start groups.
Gauss--Newton is the best solver-side compromise in time and surrogate fit,
but choosing it from those metrics alone would select the worst latent-32 PDE
residual. This converts the earlier surrogate-exploitation inference into an
independently replayed application result.

The comparison was then frozen and evaluated prospectively on 12 new test
instances drawn with seed `20260827`, excluding the three pilots. All 360
method/instance/resolution/start solves pass common KKT. Surrogate and PDE
method rankings agree descriptively in only 35/120 matched groups. At latent 32,
Gauss--Newton wins the FNO ranking in 59/60 groups, while independent PDE
replay selects L-BFGS 25 times, Gauss--Newton 22 times, and exact DLPack 13
times. Median latent-32 FNO/PDE residuals are 0.37%/5.09%
(Gauss--Newton), 0.49%/4.13% (L-BFGS), and 0.76%/4.68% (exact). Using one
median per physical instance, Gauss--Newton reduces FNO residual relative to
L-BFGS on all 12 instances by 0.095 percentage points (bootstrap 95% CI
0.065--0.133; Wilcoxon `p=0.00049`). Its paired PDE change is +1.20 points
with CI -1.10--2.96 (`p=0.266`). The defensible conclusion is failure of
surrogate gains to transfer consistently, not statistically established
physical superiority of L-BFGS.

Gauss--Newton is not a memory cure in the current explicit implementation.
With JAX preallocation disabled, fresh latent-32 exact and Gauss--Newton
DLPack runs peak at 33.54 and 33.51 GiB GPU memory, respectively (5.76/5.75
GiB host). Forming the full FNO residual Jacobian with 1,024 forward tangents
has essentially the same accelerator capacity requirement as the exact scalar
Hessian. Only the matrix-free/compact L-BFGS path avoids that regime today.

## 9. Direct-prior adversarial MNIST replication

The exact public recipe from `Robbybp/moai-examples` was used: seed 101,
four hidden-to-hidden tanh layers, softmax output, ten epochs, and the paper's
sample 1 mapping (torchvision test index 6, true label 4, target label 9). The
128/512/1024-width models have 0.168M/1.46M/5.01M parameters and test
accuracies 97.20%/97.12%/97.01%. Torch/JAX relative discrepancies are below
`7.1e-8` in float32 and `1.8e-15` in float64.

The lifted reduced problem has 2,362 variables, 805 constraints, 10,213
Jacobian entries, and 307,720 lower-Hessian entries, matching the closest
paper's structural table. Directional objective/JVP/Hessian-vector errors are
below `7e-8`. With solver tolerance `1e-7`, every listed solve also has raw
network/slack/target/output-bound violation below `2e-7`.

| parameters | MadNLP exact CPU | MadNLP exact DLPack GPU | MadNLP L-BFGS CPU | cyipopt exact | cyipopt L-BFGS |
|---:|---:|---:|---:|---:|---:|
| 0.168M | 0.895 s | 0.401 s | 0.403 s | 1.739 s | 1.535 s |
| 1.46M | 2.968 s | 0.567 s | **0.231 s** | 3.412 s | 1.651 s |
| 5.01M | 1.511 s | 1.327 s | **0.275 s** | 4.555 s | 2.447 s |

Times alone do not rank the nonconvex methods. At 1.46M parameters, MadNLP
exact CPU and L-BFGS find objective 3.723 while exact GPU reaches 4.079. At
5.01M, exact CPU finds 4.644, exact GPU/L-BFGS/cyipopt exact reach about 4.736,
and cyipopt L-BFGS reaches 5.625. All are KKT-feasible local solutions. The
paper must show objective and success beside timing, and should add multiple
starts before treating these differences as algorithmic robustness.

That five-start audit is now complete for the 5.01M-parameter model. The
published image plus four seed-`20260821` Gaussian perturbations are converted
to network/slack-consistent shared starts. All 15 method/start solves pass the
`1e-7` KKT and `1e-6` raw-residual audits.

| method | successes | median warm time | objective median / range |
|---|---:|---:|---:|
| exact CPU KKT | 5/5 | 2.98 s | 4.736 / 4.644--5.590 |
| exact DLPack GPU | 5/5 | 1.31 s | 4.736 / 4.736--4.999 |
| compact L-BFGS CPU | 5/5 | **0.47 s** | 5.590 / 4.736--5.625 |

The published start is unusually favorable to exact CPU, while one perturbed
start reaches the worse 5.590 basin. L-BFGS is the fastest on every aggregate
but sends all four perturbed starts to objectives at least 5.590. This supports
the paper's narrower conclusion: report time and local solution quality
together; five nearby starts do not establish a global robustness ranking.

## 10. Peak-memory audit

A 50 ms process-tree RSS and `nvidia-smi` sampler was used around fresh
representative runs. These are whole-process cold-run peaks, not just final
matrix sizes.

| case | method | peak host | peak GPU |
|---|---|---:|---:|
| PFR case 1 | exact, CPU KKT | 3.72 GiB | 0.86 GiB |
| PFR case 1 | exact, DLPack GPU KKT | 4.17 GiB | 1.08 GiB |
| MNIST 5.01M | exact DLPack | 4.91 GiB | 2.77 GiB |
| MNIST 5.01M | L-BFGS CPU KKT | 8.27 GiB | 1.05 GiB |
| Darcy latent 32 | exact DLPack | 5.71 GiB | 33.54 GiB |
| Darcy latent 32 | L-BFGS CPU KKT | 3.96 GiB | 0.99 GiB |
| PFR case 3 | exact, CPU KKT | 4.47 GiB | 1.14 GiB |
| PFR case 3 | exact, DLPack GPU KKT | 6.15 GiB | 43.90 GiB |
| PFR case 3 | L-BFGS, CPU KKT | 4.15 GiB | 1.14 GiB |
| replay Darcy latent 32 | exact DLPack | 5.76 GiB | 33.54 GiB |
| replay Darcy latent 32 | Gauss--Newton DLPack | 5.75 GiB | 33.51 GiB |

The Darcy exact path's 33.5 GiB peak confirms that second-order AD
intermediates, not the 8 MiB final Hessian, set the memory regime. Curvature
routing therefore reduces both total time and required accelerator capacity.
PFR again shows no GPU-memory rationale for moving its sparse KKT system.
That statement is regime-specific: case 3 has enough coupled structure for
GPU factorization to win decisively in time, but only on a 48-GiB-class GPU.
