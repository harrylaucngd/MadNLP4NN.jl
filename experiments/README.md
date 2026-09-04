# Experiment workspace

This directory contains reproducible benchmark definitions, solver adapters,
run schemas, and analysis scripts for the paper.

Raw data, model checkpoints, logs, and run outputs belong under the ignored
`output/` tree. Small immutable configs, checksums, and analysis code are
versioned here.

No result is considered paper-ready unless it has:

1. a complete run manifest and model/data checksums;
2. an independent feasibility and KKT audit;
3. synchronized GPU timing when CUDA is used;
4. at least one repeated warm timing after compilation;
5. a visible record of failures and timeouts.

Current validated artifacts can be aggregated with:

```bash
python experiments/analysis/summarize_results.py \
  --output output/analysis/results_snapshot.json
```

The PFR Ipopt/OMLT baselines use the isolated `madnlp4nn-ipopt` conda
environment; the primary JAX/MadNLP environment is intentionally unchanged.

Capture provenance and append any new raw artifacts to the content-addressed
registry with:

```bash
CUDA_VISIBLE_DEVICES=0 python experiments/analysis/capture_environment.py \
  --output output/environment_primary.json
python experiments/analysis/register_artifacts.py \
  --registry output/registry.jsonl
```

`experiments/invalidated_runs.json` is authoritative for known-invalid runs.
Invalid and malformed artifacts remain in the registry rather than being
deleted or silently filtered.

## Archive boundary

Git tracks the immutable configurations, runners, aggregation code, generated
paper tables/figures, and manuscript. It intentionally does not track the
multi-GB datasets, checkpoints, raw JSON runs, machine traces, or local
content-addressed registry under `output/`. Consequently, a clean clone can
build and inspect the checked-in manuscript, but rerunning the 31-claim audit
requires reacquiring the public inputs and regenerating the registered local
artifacts. Dataset/model checksums and all known invalidations remain recorded
in the research notes and experiment definitions.
