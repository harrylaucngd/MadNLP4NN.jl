# Literature and benchmark map

Status: search frozen for the completed August 2026 snapshot. Future venue or
software updates are outside this snapshot. Links point to primary papers,
project pages, or official documentation.

## Closest prior work

| Work | Main result | Consequence for MadNLP4NN |
|---|---|---|
| [OMLT (JMLR 2022)](https://www.jmlr.org/papers/v23/22-0277.html) | Open-source full/reduced formulations for ML predictors in Pyomo | A generic NN-in-NLP interface is useful software but not a novel scientific claim. |
| [Optimization with NN feasibility surrogates (2023)](https://doi.org/10.3390/en16165913) | Smooth full-space, reduced-space, and complementarity NN formulations on SCOPF | Formulation comparisons and power-system NN constraints predate this project. |
| [Parker et al., formulations and scalability (2024)](https://arxiv.org/abs/2412.11403) | Full, reduced, and PyTorch gray-box formulations up to 590M parameters; approximate Hessians are decisive | This is the closest formulation study. Exact-vs-approximate curvature and a direct MathOptAI/Ipopt comparison are mandatory. |
| [Parker et al., GPU-accelerated NN constraints (2025)](https://arxiv.org/abs/2509.22462) | GPU PyTorch derivatives with Ipopt on MNIST and transient SCOPF, up to 592M parameters | The original MadNLP4NN headline is already demonstrated. Their paper explicitly leaves GPU NLP solver integration and large input/output dimensions open. |
| [Parker et al. reproduction repository](https://github.com/Robbybp/moai-examples) | Exact MNIST training seed/architecture, adversarial formulation, sample mapping, and full/reduced-space scripts | Direct source for B1. Models are trained locally because checkpoints are not committed. |
| [MathOptAI.jl](https://github.com/lanl-ansi/MathOptAI.jl) | Current Julia package for embedding predictors in JuMP and ExaModels, including gray-box PyTorch/Flux paths | Required software baseline and a source of compatible MNIST formulations. |
| [Elorza Casas et al., NN embedding in NMPC (2025)](https://arxiv.org/abs/2501.06335) | Compares explicit full/reduced and external-function embeddings on three PDE-constrained PFR NMPC cases | Supplies an established constrained dynamic benchmark and warns that NN surrogates do not automatically improve solve time. |
| [Companion NMPC code](https://git.uwaterloo.ca/ricardez_group/benchmarking-nn-surrogate-embedding-methods-in-nmpc) | Pretrained tanh MLP/CNN models, mechanistic baselines, and saved outputs | Candidate application benchmark B3. Preserve the published problem rather than inventing another control case. |
| [Process flowsheet optimization with Gibbs reactor surrogates (2023)](https://arxiv.org/abs/2310.09307) | Reliability/time/accuracy comparison among full-space, implicit, polynomial, and NN surrogates over 64 instances | Motivates reporting convergence reliability and high-fidelity error, not runtime alone. |
| [Yang, Balaprakash, and Leyffer (2021)](https://arxiv.org/abs/2111.10489) | Analyzes embedded, mixed-integer, and complementarity formulations and their stationarity for neural surrogates | Prevents novelty claims around generic embedding; supports the smooth-oracle scope. |

## Optimization shift and application validity

| Work | Relevance |
|---|---|
| [Conservative Objective Models (ICML 2021)](https://proceedings.mlr.press/v139/trabucco21a.html) | Direct optimization of a learned objective can exploit out-of-distribution overestimation; motivates simulator replay rather than surrogate-only rankings. |
| [Design-Bench (ICML 2022)](https://proceedings.mlr.press/v162/trabucco22a.html) | Standardizes offline model-based optimization tasks and evaluation; reinforces held-out oracles, shared protocols, and visible simple baselines. |
| [Neural Adjoint (NeurIPS 2020)](https://proceedings.neurips.cc/paper/2020/hash/007ff380ee5ac49ffc34442f5c2a2b86-Abstract.html) | Establishes gradient-based inverse search through learned forward models and time-aware evaluation; closest inverse-design context. |
| [Trust-region model management (2000)](https://doi.org/10.1016/S0377-0427(00)00424-6) | Classical framework for restricting and validating low-fidelity surrogate optimization; frames high-fidelity replay as part of method design, not post-hoc decoration. |

## GPU NLP and solver structure

| Work | Relevance |
|---|---|
| [MadNLP GPU tutorial](https://madsuite.org/MadNLP.jl/stable/tutorials/gpu/) | Documents true GPU-resident ExaModels execution, CPU-model GPU wrappers, cuDSS, Lifted-KKT, and HybridKKT. It exposes the gap between current MadNLP4NN claims and its host-backed implementation. |
| [Condensed-space methods for NLP on GPUs](https://arxiv.org/abs/2405.14236) | Establishes Lifted-KKT and HyKKT trade-offs and the role of cuDSS. |
| [GPU-accelerated dynamic nonlinear optimization](https://doi.org/10.1007/s12532-024-00261-7) | End-to-end ExaModels/MadNLP GPU results for structured dynamic NLPs. |
| [MadNLP quasi-Newton tutorial](https://madsuite.org/MadNLP.jl/stable/tutorials/quasi_newton/) | Compact L-BFGS can remove Hessian callbacks and is a first-class MadNLP option. |
| [JAX DLPack API](https://docs.jax.dev/en/latest/_autosummary/jax.dlpack.from_dlpack.html) | Defines when JAX can share an external device buffer without a copy. |
| [CUDA.jl memory documentation](https://cuda.juliagpu.org/stable/usage/memory/) | Reference for auditing host/device copies of `CuArray` data. |
| [jaxipm (2026)](https://arxiv.org/abs/2606.26341) | A JAX-native, GPU-batched IPOPT-style solver with up to 32.85x throughput gains on batched quadrotor NLPs. It is a required throughput baseline if we claim benefits for many simultaneous inverse/control solves. |
| [ipax](https://github.com/wahln/ipax) | Beta Array-API primal-dual solver with JAX/PyTorch/CuPy GPU, matrix-free and sparse-direct paths, and L-BFGS curvature. It is an implementation baseline for large-`n` GPU problems when its supported feature set matches. |
| [slsqp-jax / sqpdax](https://github.com/lucianopaz/slsqp-jax) | Emerging matrix-free JAX SQP implementation targeting 5k--50k variables. Track during the project; compare only a tagged, tested release and label its maturity. |
| [Accelerator HiOp port (2026)](https://arxiv.org/abs/2605.13736) | Confirms that GPU NLP is now a crowded systems area; reinforces the need for a neural-oracle-specific contribution rather than a generic GPU claim. |
| [Single-precision differentiable IPM (2026)](https://arxiv.org/abs/2605.17913) | Shows that standard complementarity systems can become unreliable in float32. Mixed precision must be validated, not assumed safe. |

`jaxipm` is not placed in the frozen latency table: it studies many-problem
throughput with heterogeneous iteration fusion, whereas this paper studies one
large solve or a receding-horizon sequence. A direct comparison would change
the performance objective. It becomes mandatory if a batched-throughput claim
is added later.

## Public benchmarks and data

| Resource | Assets | Decision |
|---|---|---|
| [PDEBench paper](https://arxiv.org/abs/2210.07182) and [code](https://github.com/pdebench/PDEBench) | Public PDE data and pretrained FNO/U-Net models | Retain its beta=1 128x128 checkpoint as an oracle/system smoke only. Audit of the official generator shows that the file named DarcyFlow is a transient diffusion problem with a random, unreleased initial field; it is not a clean deterministic coefficient-to-solution inverse benchmark. |
| [NeuralOperator Darcy dataset](https://zenodo.org/records/12784353) and [official loader](https://github.com/neuraloperator/neuraloperator/blob/main/neuralop/data/datasets/darcy.py) | Public train/test tensors at 16, 32, 64, 128, and 421 resolution for the standard deterministic elliptic Darcy map | Primary B2 data. Resolution 64 is the initial gate; advance to 128 after model and independent PDE-solver validation. |
| [NeuralOperator Darcy/FNO documentation](https://neuraloperator.github.io/dev/auto_examples/data/plot_darcy_flow.html) | Maintained data loader and FNO training workflow | Basis for the reproducible training protocol; use the full public split rather than only tutorial-sized subsets. |
| [PDE Control Gym](https://proceedings.mlr.press/v242/bhan24a.html) | Public transport, reaction-diffusion, and Navier--Stokes control environments | Possible future control validation, but it is not directly an NN-embedded NLP benchmark. Lower priority than the published PFR NMPC suite. |
| [Deep inverse-model benchmark / Neural Adjoint](https://papers.neurips.cc/paper/2020/hash/007ff380ee5ac49ffc34442f5c2a2b86-Abstract.html) | Four inverse-design tasks including metamaterials | Relevant inverse-design reference, but its low-dimensional unconstrained focus does not stress MadNLP's intended regime. Not in the primary suite. |
| [SurrogateLIB](https://doi.org/10.5281/zenodo.11231147) | Public MIP instances with embedded ML predictors | Discrete/MIP scope is outside this smooth NLP paper; useful only for broader related-work context. |

## Historical audit that motivated the refactor

The following findings were resolved in the new implementation or by retiring
the affected claim before any old figure was reused:

- The current Darcy target is generated by the same FNO that is inverted and
  the initial point is a small perturbation of the hidden truth. This is an
  inverse crime and makes the seven-evaluation L-BFGS-B result unsurprising.
- FNO derivative callbacks explicitly construct a dense Hessian and convert it
  to NumPy on every callback. GPU execution is therefore not end to end.
- The quadratic smoothness constraint stores the discrete Laplacian as a dense
  matrix in the JAX evaluator.
- The repository does not contain the raw results or runnable baselines behind
  the current PDF's full-space and first-order figures.
- `select_linear_solver("gpu")` selects a dense GPU LAPACK solver when loaded
  and otherwise falls back to a CPU solver. It does not establish a cuDSS path.
- Timing is measured around high-level callbacks without explicit GPU timing
  instrumentation or a complete KKT/factorization breakdown.
- Current tests do not exercise JAX, CUDA, derivative agreement, KKT residuals,
  or benchmark data integrity.
- The synthetic Pareto hypervolume uses a reference point derived from the
  returned front, so values are not comparable across methods.

These are reasons to retire the old performance claims, not merely caveats to
append to them.
