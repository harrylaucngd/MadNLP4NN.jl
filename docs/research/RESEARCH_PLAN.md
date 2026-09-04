# MadNLP4NN research plan

Status: frozen protocol for the completed August 2026 research snapshot.

## Venue and scope

The URL `opt-ml.org` currently refers to the NeurIPS Optimization for Machine
Learning (OPT) workshop, not an ICLR workshop. The 2027 call has not been
announced. We therefore develop the work against a venue-independent scientific
standard and revisit the venue after the 2027 calls are public.

The project is no longer framed as the first reduced-space nonlinear program
with a neural network embedded. That claim is precluded by OMLT, MathOptAI, and
recent gray-box/reduced-space studies. The intended paper asks a narrower and
more useful question:

> When does a GPU-resident primal-dual method actually outperform CPU and
> first-order alternatives for constrained optimization through a fixed neural
> surrogate, and which curvature model should be used in each problem regime?

The working title is:

> **A Regime Map for GPU Second-Order Optimization through Neural Surrogates**

## Proposed contribution

The target contribution has three parts. Each part has an explicit go/no-go
criterion so that an unsupported claim is removed rather than rationalized.

1. **End-to-end GPU gray-box NLP interface.** Neural evaluation and AD run in
   JAX, while MadNLP performs KKT assembly and factorization on the same GPU.
   Device buffers are exchanged without host staging where technically
   possible. A host-staged implementation remains as a measured ablation.

   Go criterion: zero-copy/end-to-end execution must be demonstrated by both a
   transfer audit and a measurable benefit on at least one established task.

2. **Curvature strategy matched to the problem regime.** Compare exact
   Lagrangian Hessians, limited-memory BFGS, and a structure-aware approximation
   (Gauss--Newton for least-squares inverse problems, or an equivalent justified
   model). The study varies decision dimension `n`, neural output/constraint
   dimension `m`, and parameter count `p` independently.

   Go criterion: the proposed strategy must improve time-to-solution or memory
   at matched feasibility and stationarity on more than one task family. If it
   does not, this becomes an empirical characterization paper rather than a new
   method paper.

3. **A reproducible neural-embedded NLP benchmark protocol.** Use public data,
   published problem definitions, held-out targets, multiple starts, independent
   high-fidelity validation, and solver-independent success metrics. Synthetic
   cases are retained only for controlled scaling and derivative correctness.

## Research questions

- RQ1: At fixed `n` and `m`, how large must `p` be before GPU AD amortizes JIT,
  Python/Julia dispatch, and synchronization overhead?
- RQ2: At fixed `p`, when does the explicit `O(n^2)` Hessian become the dominant
  memory/time bottleneck?
- RQ3: Does GPU KKT assembly/factorization help after accounting separately for
  GPU AD, or does the host/device interface erase the gain?
- RQ4: How much robustness or iteration reduction is obtained from exact
  curvature relative to L-BFGS/Gauss--Newton at matched KKT tolerances?
- RQ5: Do solutions optimized through the surrogate remain feasible and useful
  under the high-fidelity simulator, or only under the surrogate?

## Benchmark suite

The suite spans qualitatively different `(n, m, p)` regimes. No single task is
allowed to carry the paper.

### B0: Planted smooth oracle (correctness and controlled scaling)

- Smooth MLPs with independently controlled input dimension, output dimension,
  depth, and width.
- A planted feasible stationary point and analytically checkable small cases.
- Purpose: derivative tests, solver agreement, memory scaling, and clean
  isolation of `n`, `m`, and `p`.
- This is a systems microbenchmark, not evidence of application value.

### B1: Adversarial MNIST (direct-prior-work replication)

- Match the smooth tanh/softmax formulation used by Parker et al. where
  possible: 784 image variables, target-class constraints, and increasing
  hidden-network size.
- Purpose: reproduce the regime in the closest prior GPU gray-box paper and
  isolate the neural parameter-count axis.
- It is not a headline scientific application and will not be described as
  inverse design.
- The located reproduction code fixes seed 101, uses seven-layer tanh/softmax
  classifiers with widths 128--8192, and solves an L1 minimal perturbation with
  target probability at least 0.6. The local 128/512/1024 models reproduce the
  reported 167k/approximately-1M/5M parameter points and exceed 95% test
  accuracy.

### B2: NeuralOperator Darcy/FNO inverse problem (high-dimensional decision axis)

- Use the public, deterministic steady-state Darcy-flow data distributed by
  NeuralOperator on Zenodo. This is the established FNO benchmark for the
  elliptic map `a -> u`, not PDEBench's misleadingly named transient-diffusion
  file whose target also depends on an unobserved random initial condition.
- Use fixed train/validation/test splits and a standard FNO training recipe.
  No maintained public checkpoint was found for this exact artifact, so the
  training script, seeds, normalization state, and all selected checkpoints
  are part of the experiment artifact.
- The public split uses coefficient truth and representability floors because
  its historical scaling/downsampling metadata is not recoverable. A separate
  registered 5,000/1,000 split uses a fully specified sparse finite-difference
  generator; it is trained and optimized with the same FNO size and replayed
  exactly through the generating PDE solver.
- Report coefficient error, surrogate residual, simulator residual where
  admissible, and rankings across exact, Gauss--Newton, and L-BFGS. Partial
  observations/noise are follow-up stress tests rather than prerequisites for
  the current full-observation claim.

PDEBench beta=1 remains a systems smoke test because it has an official FNO
checkpoint and stresses the 128x128 oracle. It is excluded from inverse-design
claims: the official generator evolves a random initial field under a
variable-coefficient diffusion equation, while the released FNO receives only
the coefficient. This makes the inverse map confounded and prevents a clean
independent simulator replay from the released pair alone.

### B3: Published PDE-surrogate NMPC (constraint and horizon axis)

- Adapt the open benchmark accompanying Elorza Casas et al. (2025), which
  provides three plug-flow-reactor NMPC cases and pretrained tanh MLP/CNN
  surrogates.
- Begin with case study 1; advance to cases 2 and 3 only after agreement with
  the published mechanistic and Ipopt trajectories.
- Compare per-control-step latency, closed-loop objective, surrogate constraint
  violation, and mechanistic-simulator constraint violation.
- Preserve receding-horizon warm starts. Cold-start timing is reported
  separately and is never mixed with steady closed-loop latency.

### Deferred benchmark: transient-stability SCOPF

SCOPF is highly relevant and directly used by the closest prior work, but their
trained transient-stability networks/data are not currently available in the
public repositories located during the initial audit. It is deferred until a
fully reproducible instance is obtained. We will not train a weak substitute
and label it an equivalent benchmark.

### Removed from the paper suite: synthetic Pareto tracing

The current Pareto task learns surrogates of newly generated quadratic
functions and then optimizes those surrogates. It lacks an established use case,
and its hypervolume reference changes with the returned points. It remains a
small multi-network API test but is excluded from performance and application
claims.

## Baselines

Every comparison uses the same mathematical objective, constraints, bounds,
start, precision where supported, and termination audit. Solver-reported
success alone is insufficient.

### Formulation and solver baselines

- MadNLP + JAX gray box, exact Hessian (CPU callbacks/CPU KKT).
- MadNLP + JAX gray box, exact Hessian (GPU AD/CPU KKT; host staged).
- MadNLP + JAX gray box, exact Hessian (GPU AD/GPU KKT; zero-copy target).
- MadNLP + JAX gray box, compact L-BFGS, on CPU and GPU.
- MadNLP + structure-aware curvature, if the approximation passes derivative
  and convergence checks.
- cyipopt/Ipopt with the identical JAX oracle, structures, bounds, and starts.
  This isolates the NLP solver instead of changing solver and AD engine at once.
- The exact public PyTorch architectures/checkpoints are forward-parity gated.
  MathOptAI remains closest related software, but is not used as a timing point:
  its framework/formulation changes would confound the placement ablation. The
  paper makes no JAX-versus-PyTorch speed claim.
- Published Pyomo/OMLT full-space and reduced-space formulations on the NMPC
  benchmark for sizes that fit in memory.
- jaxipm is required only if a batched-throughput claim is added; the current
  paper studies single-instance/receding-horizon latency and makes no such claim.
- ipax/sqpdax are excluded from the frozen table because their current feature
  and maturity levels do not match the sparse constrained problems and common
  termination audit. They remain tracked in the literature map.

### Algorithmic baselines

- SciPy L-BFGS-B only for genuinely box-constrained variants.
- `trust-constr` or Ipopt for general smooth constraints.
- Projected Adam only where projection is exact and cheap. For nonlinear
  constraints, a penalty-Adam result is labeled as a penalty method and is not
  declared feasible from the penalized loss alone.
- A derivative-free method is included only on low-dimensional instances; it is
  not a meaningful high-dimensional scalability baseline.

The existing comparison between MadNLP with explicit nonlinear constraints and
L-BFGS-B/Adam with hand-tuned penalties is retired because it changes the
problem and termination definition.

## Ablations

1. Formulation: full-space vs algebraic reduced-space vs gray-box.
2. Curvature: exact vs compact L-BFGS history 6/20 vs structure-aware model.
3. Placement: CPU/CPU, GPU-AD/CPU-KKT, GPU-AD/GPU-KKT.
4. Transfer: host-staged vs DLPack/zero-copy, with bytes and synchronizations.
5. AD engine: JAX is primary and every released PyTorch checkpoint passes
   forward parity. No AD-engine speed claim is made, so solver timings use one
   oracle to avoid conflating two axes.
6. Precision: float64; mixed float32-network/float64-solver only as an ablation
   with a KKT and solution-quality audit.
7. Compilation: cold end-to-end time, warm solve time, and amortized time over a
   parameterized sequence.
8. Scale axes: vary one of `n`, `m`, or `p` while holding the other two fixed.
9. Initialization: fixed published start plus at least five seeded starts for
   nonconvex application instances.
10. Constraint activity and observation density/noise for the inverse task.

## Metrics and statistical protocol

Primary solver success requires all of:

- finite objective and iterates;
- normalized primal infeasibility at or below the declared tolerance;
- normalized dual/stationarity residual at or below the declared tolerance;
- complementarity at or below the declared tolerance for primal-dual methods;
- no more than `1e-6` relative bound violation unless a looser benchmark
  tolerance is predeclared;
- application-specific high-fidelity validation.

Report:

- cold time, warm time, median solve time, and time-to-KKT target;
- median and interquartile range over repeated timed runs;
- success fraction over instances/starts;
- objective gap only among solutions meeting the common feasibility threshold;
- evaluation counts, iterations, peak host memory, peak GPU memory;
- objective/gradient/Jacobian/Hessian, transfer, KKT assembly, factorization,
  and backsolve time;
- performance profiles rather than winner-only averages when enough instances
  are available.

JIT compilation and data/model loading are excluded from warm solve time but
included in cold time. Every reported timing states which definition it uses.
GPU timings synchronize the device at measurement boundaries. The first run is
a warm-up and is never silently included in steady-state medians.

## Reproducibility and artifact layout

Planned layout:

```text
experiments/
  configs/          # immutable TOML/YAML experiment definitions
  runners/          # one entry point per benchmark family
  baselines/        # solver adapters sharing the same oracle/problem data
  analysis/         # tables, profiles, and paper figures
  schemas/          # result and provenance schemas
output/
  runs/<run_id>/    # ignored raw artifacts, one manifest per run
  registry.jsonl    # append-only local run registry
docs/research/      # protocol, literature map, and decision log
```

Each run manifest records the git commit/diff hash, hostname, CPU/GPU, package
versions, model/data checksums, seed, complete solver options, start/stop time,
exit status, and raw residuals. Failed and timed-out runs stay in the registry.

## Compute schedule

### Phase 1: audit and correctness

1. Upgrade to a current MadNLP/MadNLPGPU stack and add GPU integration tests.
2. Implement independent finite-difference/directional derivative checks.
3. Reproduce small problems with MadNLP and Ipopt to a common KKT tolerance.
4. Build synchronized microbenchmarks for each oracle call and transfer path.

No application sweep starts until this phase passes.

### Phase 2: feasibility pilots

1. B0 `(n,m,p)` pilot to locate memory/time boundaries.
2. B1 small and medium MNIST replication.
3. B2 one held-out Darcy instance at two resolutions.
4. B3 published NMPC case 1 for one closed-loop trajectory.

The pilots determine the final grid; they are not automatically promoted to
paper results.

### Phase 3: registered sweeps

- CPU-only jobs are distributed across physical cores with BLAS/thread counts
  pinned to avoid oversubscription.
- The single H100 runs one memory-intensive exact-Hessian process at a time.
  Small quasi-Newton jobs may be packed only after utilization measurements.
- Jobs are ordered shortest-first within each dependency layer so failures are
  discovered before long sweeps.
- A monitor checks process liveness, last result timestamp, GPU utilization and
  memory, disk use, and failure summaries. Stalled jobs are diagnosed rather
  than blindly restarted.

### Phase 4: blind aggregation

The analysis scripts consume the registered result schema without method-
specific exclusions. Any excluded run must match a predeclared rule and remain
visible in the failure table.

## Post-snapshot follow-ups (not required by the current claims)

1. Add partial-observation/noise stress tests to the replayable Darcy split if
   they change the surrogate/simulator ranking conclusion.
2. Treat cases 2/3 as first-step scale evidence unless a full mechanistic replay
   can be ported without changing their published plant definitions.
3. Extend peak-memory sampling with allocator traces/energy only if those
   measurements are stable; the current whole-process peaks are already
   registered and sufficient for capacity claims.
4. Validate timing on a second GPU architecture before claiming a portable
   crossover rule. The current paper deliberately makes no such claim.
