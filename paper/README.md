# Paper builds

Only the anonymous manuscript and its anonymous source bundle are versioned.
Identity-bearing metadata and author-visible builds must remain outside the
tracked repository.

The ICLR 2027 workshop template is not public as of 2026-08-24. The directory
`iclr2026/` therefore contains the unmodified latest official ICLR style and
bibliography files downloaded from the ICLR Master Template repository. The
year-specific dependency is isolated so it can be replaced without rewriting
the paper.

The currently published `opt-ml.org` call is OPT 2026 at NeurIPS, with its own
style and a five-page soft limit; it is not an ICLR 2027 call. This repository
retains the requested full ICLR-format manuscript until the intended 2027
venue publishes its instructions.

From the repository root:

```bash
python experiments/analysis/build_iclr_artifacts.py \
  --run-dir output/runs --paper-dir paper
python experiments/analysis/analyze_sequence_and_topology.py
cd paper
latexmk -pdf -interaction=nonstopmode -halt-on-error iclr_workshop.tex
cd ..
python experiments/analysis/build_submission_archive.py
python experiments/analysis/build_submission_archive.py --check
```

The current main text occupies nine ICLR pages. References begin on page 10;
appendices follow the references.

`iclr_workshop_source.zip` is a deterministic, anonymous, self-contained
submission-source bundle with the official style files, modular sections,
bibliography, generated macros, and referenced PDF figures.
