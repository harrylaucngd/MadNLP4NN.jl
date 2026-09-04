# Experiment schedule

Status updated 2026-08-24. A phase advances only after its listed dependency
passes. H100 jobs are serialized; CPU-only analysis and downloads may overlap.

| Priority | Job | Resource | Dependency | Status |
|---:|---|---|---|---|
| 0 | Repository/PDF/method audit | CPU | none | complete |
| 0 | Literature and public-benchmark audit | CPU/network | none | complete |
| 0 | JAX oracle `(n,m,p)` quick regime map | H100 + CPU | timing harness | complete |
| 0 | CUDA.jl/JAX DLPack pointer and mutation audit | H100 | CUDA 6 / DLPack | complete |
| 0 | MadNLP 0.10 + cuDSS host-staged smoke | H100 | package upgrade | complete |
| 0 | DLPackGPUModel callback/solve integration tests | H100 | DLPack audit | complete (10/10) |
| 0 | PDEBench artifact checksum and FNO PyTorch/JAX parity | H100/network | public artifact download | complete |
| 0 | PDEBench held-out FNO quality over 1,000 samples | H100 | parity | complete |
| 0 | PDEBench median instance latent 8/16/32 pilot | H100 + CPU KKT | evaluator parity | invalidated by row/column-major target bug |
| 1 | Repeat controlled exact path table with pinned CPU affinity, 10 warm reps | H100 + one CPU socket | quick regime map | complete |
| 1 | Compact L-BFGS history 6/20/50 sensitivity | H100 + CPU KKT | controlled gate | complete; history 6 is 3.9--9.8x slower |
| 1 | Corrected PDEBench target-order smoke | H100 + CPU KKT | layout invariant | latent-8 L-BFGS complete |
| 1 | NeuralOperator elliptic Darcy-64 data/model gate | H100 + CPU | public artifact | complete; historical preprocessing prevents public-artifact replay |
| 1 | Original FNO Darcy-241/421 artifact recovery | network/disk | historical solve_gwf source | optional/blocked by Drive quota; replay split removes it from critical path |
| 1 | NeuralOperator Darcy exact/L-BFGS placement pilot | H100 + CPU KKT | model/parity gate | complete |
| 1 | NeuralOperator Darcy q25/median/q75, 8/16/32, five starts | H100 + CPU KKT | single-start strata | complete for L-BFGS |
| 1 | Prospective L-BFGS -> exact-on-failure curvature routing | H100 + mixed KKT | multi-start pilot | complete: 120/120, +7.7% time |
| 1 | Prospective Darcy 12-instance x 2-resolution x 5-start audit | H100 + CPU KKT | frozen seed 20260822 | complete |
| 1 | Same-oracle SLSQP/Ipopt plus OMLT formulation baselines | CPU + H100 AD | oracle schema | complete; MathOptAI retained as related software, no cross-AD timing claim |
| 1 | Surrogate-validity intervention: local trust region / ensemble uncertainty | H100 | replayable Darcy evidence | deferred to follow-up; current paper characterizes rather than hides mismatch |
| 2 | Published PFR NMPC case 1 reproduction | CPU | upstream environment lock | complete |
| 2 | PFR JAX/MadNLP/cyipopt first-step and 100-step validation | H100 + CPU | reproduction agreement | complete |
| 2 | PFR Jacobian/Hessian sparsity ablation | mixed | derivative audit | complete; reverses device ranking |
| 2 | OMLT full/reduced formulation reproduction | CPU | isolated Ipopt env | full-space complete; maintained reduced-space build timeout |
| 2 | MathOptAI/Parker adversarial MNIST direct replication | H100 + CPU | public recipe | complete: three scales + five starts at 5.01M |
| 3 | PFR cases 2/3 published CNN scale grid | mixed | phase-2 success | complete: parity/derivatives/Ipopt/exact/L-BFGS; exact paths repeated five times |
| 3 | Replayable elliptic-Darcy split and independent simulator audit | CPU + H100 | public archive metadata gap | complete: 360/360 prospective KKT, ranking agreement 35/120 |
| 3 | Content-addressed registry and environment capture | CPU | validated artifacts | complete: 187 digests, 12 invalidated retained |
| 3 | Representative peak host/GPU memory audit | mixed | resource monitor | complete for PFR/MNIST/Darcy, including PFR case 3 |
| 3 | PFR cold/primal/dual/barrier/persistent 100-step replay | H100 + mixed KKT | closed-loop gate | complete: 1,100/1,100 at $10^{-6}$; 400/400 at $10^{-8}$ |
| 3 | Parameter-matched FNO depth/width Hessian-memory sweep | H100, RTX PRO 6000 | resource monitor | complete: six H100 graphs; cross-device peaks within 0.22 GiB |
| 3 | Blind aggregation and first paper figures | CPU | registered sweeps | complete for current snapshot |
| 3 | Julia package and GPU integration regression | CPU + H100 | final source | complete: 60/60 CPU and 15/15 GPU |
| 3 | ICLR-format anonymous/author manuscripts, appendix, and claim audit | CPU | final registered sweeps | complete: 9-page main text, references page 10, 31/31 claim checks |

Monitoring command:

```bash
python experiments/monitor.py
```

The monitor shows active runner processes, GPU memory/utilization, and recent
artifacts. Long registered jobs will additionally use per-run manifests and
heartbeats; the current diagnostic pilots run in the foreground so failures are
visible immediately.
