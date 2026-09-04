# ICLR workshop submission strategy

Status: frozen scientific positioning, 2026-08-21.

## One-sentence thesis

Efficient optimization through neural surrogates is a joint
structure--curvature--transfer--KKT placement problem, and neither a lower
surrogate KKT residual nor a lower surrogate objective guarantees a better
physical solution.

## What is genuinely novel

1. **The systems integration is end to end and audited.** Prior GPU gray-box
   work accelerates PyTorch derivatives while retaining CPU Ipopt. This work
   connects JAX device buffers to GPU-resident MadNLP/cuDSS through DLPack and
   separately measures host staging, device copies, KKT placement, and cold
   specialization.
2. **Derivative structure is treated as an experimental variable.** The PFR
   correction is not a small optimization: disclosing 5,200/3,049 rather than
   172,000/92,665 Jacobian/Hessian coordinates reverses the CPU/GPU conclusion.
   This is the cleanest evidence that hardware benchmarks can be structurally
   confounded.
3. **Curvature routing is evaluated prospectively.** The L-BFGS-first,
   exact-on-common-KKT-failure policy is chosen on three public-Darcy pilots and
   evaluated on twelve disjoint instances. It raises success from 115/120 to
   120/120 for 7.7% additional warm time.
4. **Application validity is optimization-aware.** The independently replayed
   prospective Darcy experiment contains 360/360 successful KKT solves.
   Gauss--Newton improves FNO residual over L-BFGS on all twelve physical
   instances (paired median reduction 0.095 percentage points, bootstrap 95%
   CI 0.065--0.133), while the paired PDE change has CI -1.10--2.96 points.
   The descriptive method ranking agrees in only 35/120 matched groups. This
   connects constrained NLP systems work to the
   optimizer-induced distribution shift studied in offline model-based
   optimization.

## What the paper deliberately does not claim

- It does not introduce reduced-space or gray-box neural NLP formulations.
- It does not claim JAX is faster than PyTorch; PyTorch is a parity oracle.
- It does not claim a portable numeric crossover from one H100 node.
- It does not claim global optimality for nonconvex MNIST, Darcy, or PFR NLPs.
- It does not report a fitted simulator residual on the public NeuralOperator
  archive whose preprocessing metadata are incomplete.
- It does not compare single-instance latency with jaxipm's batched throughput.
- It does not present the replay split as an established community benchmark.

## Claim--evidence matrix

| Claim | Primary evidence | Stress test / independent check |
|---|---|---|
| DLPack removes real staging cost | 32 MiB transfer microbenchmark; controlled exact paths | pointer identity and bidirectional mutation tests |
| GPU KKT has a real large-constraint regime | PFR case 3: 2.27 s vs 19.71 s CPU, five warm runs | same-oracle Ipopt 27.84 s; released first-control match |
| Sparse disclosure can dominate hardware | PFR case-1 dense/sparse ablation | same functions, starts, precision, and JAX oracle |
| Approximate curvature is not uniformly scalable | PFR case-3 L-BFGS: 271 iterations, 530.36 s | history 6/20/50 controlled ablation; Darcy exact fallback |
| Routing improves reliability economically | public-Darcy 115/120 -> 120/120, +7.7% | policy frozen before twelve-instance prospective list |
| Surrogate gain need not transfer physically | FNO paired reduction 0.095 points, PDE-change CI spans zero | 12 new instances, 5 starts, 2 resolutions, 360/360 KKT |
| Timing needs local-solution context | MNIST 5.01M five-start objectives | 15/15 raw/KKT success; same models and starts |
| Explicit exact/GN curvature has a capacity limit | Darcy exact/GN both about 33.5 GiB | JAX preallocation-disabled process-tree sampling |

## Likely reviewer objections

### “This is an engineering paper, not a new optimization algorithm.”

The response is to avoid pretending otherwise. The main contribution is an
empirical systems and evaluation protocol with two prospective falsification
tests. The paper is workshop-appropriate because it establishes when existing
algorithmic components change rank and identifies a gap for validation-aware
model management. The routing rule is useful but intentionally modest.

### “The benchmark suite is heterogeneous.”

That is intentional: neural parameter count, NLP dimension, and constraint
dimension are independent axes. The controlled family isolates mechanisms;
MNIST reproduces the closest prior; all three PFR cases provide a published
constraint/horizon scale; public and replay Darcy separate external
comparability from physical validation. Every within-problem comparison uses a
common oracle and termination audit.

### “The simulator result is only three physical instances.”

This was true of the pilot and motivated the new frozen prospective sweep.
The main result now uses twelve disjoint physical instances and 120 matched
instance/resolution/start groups.

### “Why not use a stronger prior to prevent exploitation?”

A tenfold Tikhonov increase was tested and worsened latent-32 median PDE
residual from 2.37% to 7.67%. A generic distance-to-mean prior does not encode
the field distribution. Principled trust-region, uncertainty, or active
high-fidelity model management is future algorithmic work rather than an
unvalidated patch in this paper.

### “Why not compare every new GPU NLP solver?”

The paper uses same-oracle Ipopt to isolate the NLP solver and OMLT to isolate
formulation. jaxipm optimizes batch throughput; ipax/sqpdax do not yet match the
same sparse constrained feature set and termination audit. Adding mismatched
numbers would weaken rather than strengthen fairness.

## Venue posture

The official ICLR 2027 workshop calls and templates are not yet public. The
manuscript therefore uses the unmodified latest official ICLR (2026) style,
keeps the main text to nine pages, and isolates the year-specific files. The
paper is strongest for an optimization-for-ML, ML-for-science systems, or
scientific-ML reliability workshop. A main-track submission would benefit from
second-architecture hardware validation and a new validation-aware algorithm
with prospective simulator gains.
