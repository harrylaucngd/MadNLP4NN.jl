# NeuralOperator Darcy data and simulator-replay audit

Status: public-artifact semantics remain unresolved; the registered replay
replacement and its current claims are complete (August 2026).

The maintained NeuralOperator loader points to Zenodo record 12784353 and
describes `x` as the diffusion coefficient and `y` as the Darcy solution, but
the archive does not include a generation script, coefficient-value mapping,
forcing scale, or grid/interpolation metadata. At resolution 64, `x` is stored
as binary 0/1 while `y` ranges roughly from -0.43 to 2.23.

The historical FNO generation code was recovered from early forks of the
authors' repository. Its MATLAB `solve_gwf.m`:

- solves `-div(a grad u) = f` with zero Dirichlet boundary conditions;
- spline-interpolates a cell-centered coefficient to a nodal grid;
- uses arithmetic face coefficients in a sparse five-point operator;
- solves the interior sparse linear system and spline-interpolates back;
- demonstrates threshold coefficients 4/12 and forcing one.

A faithful SciPy port reproduces the stated discretization, but no documented
affine mapping from the Zenodo binary `x` to coefficients makes its solution
match the released 64x64 `y` within a defensible tolerance. A 0.1/1 mapping is
strongly correlated after a fitted scale, but still has about 14% relative
error on a representative sample; 4/12 is worse. This is too large to call an
independent high-fidelity replay and suggests that the maintained archive was
rescaled, regenerated, or downsampled from a finer solve without preserving
the necessary metadata.

Decision:

1. Do not report a simulator residual for this artifact yet.
2. Treat held-out coefficient recovery and the best-representable latent
   coefficient projection as the independent inverse metrics; neither uses the
   trained FNO to define truth.
3. Contact the NeuralOperator maintainers or locate the exact Zenodo
   preprocessing script. Alternatively download the 421x421 artifact and test
   the historical generator/downsampling hypothesis.
4. If the mapping remains unavailable, generate and publish a new split with
   the historical solver and use the public Zenodo data only for forward-model
   comparability. Do not tune a mapping on test targets and call it physics.

This gate is intentionally stricter than merely showing that an optimized
coefficient lowers the FNO residual.

## Registered replay fallback

Decision 4 has now been executed without changing the status of the public
artifact gate. A new split uses the fully specified repository solver for
`-div(exp(a) grad u) = 1`, zero Dirichlet boundaries, harmonic face averaging,
and clipped continuous Gaussian-random-field log permeability. It contains
5,000 training and 1,000 test pairs with seeds `20260823`/`20260824` and data
checksums recorded in `output/runs/darcy_fd_replay_dataset.json`.

A 1.19M-parameter FNO trained on this split has 1.32% mean test relative L2
error. Three FNO-error strata, two latent resolutions, five starts, and three
curvature paths first gave 90/90 pilot common-KKT successes. A frozen follow-up
on twelve disjoint instances gives 360/360 successes; the method ranking by FNO
residual agrees with the ranking by independently replayed PDE residual in only
35/120 matched groups. These simulator results are admissible because no hidden
scaling or fitted preprocessing is involved. They are labeled a registered
standard-equation split, not a public community benchmark, while the public
NeuralOperator split remains the external-comparability dataset.
