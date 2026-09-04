#!/usr/bin/env bash

# Check that the sequence conclusion is not specific to the 1e-6 stopping test.
set -u

gpu_index="${1:-0}"
julia_exe="${JULIA_EXE:-}"
if [[ -z "$julia_exe" ]]; then
  julia_exe="$(command -v julia)" || {
    echo "Julia not found; set JULIA_EXE" >&2
    exit 2
  }
fi
python_exe="${PYTHON_EXE:-${JULIA_PYTHONCALL_EXE:-}}"
if [[ -z "$python_exe" ]]; then
  python_exe="$(command -v python3 || command -v python)" || {
    echo "Python not found; set PYTHON_EXE or JULIA_PYTHONCALL_EXE" >&2
    exit 2
  }
fi
runner="experiments/runners/pfr_nmpc_closed_loop.jl"
config="experiments/configs/pfr_nmpc_case1_first_step.json"

for method in exact_cpu exact_dlpack_gpu; do
  for warm_mode in primal primal_barrier; do
    output_path="output/runs/pfr_warm_start_tolerance/${method}_${warm_mode}_tol1e8.json"
    echo "START method=${method} warm=${warm_mode} tol=1e-8"
    CUDA_VISIBLE_DEVICES="$gpu_index" \
    JAX_ENABLE_X64=True \
    XLA_PYTHON_CLIENT_PREALLOCATE=false \
    JULIA_PYTHONCALL_EXE="$python_exe" \
      "$julia_exe" --project=. "$runner" \
        --config "$config" \
        --output "$output_path" \
        --steps 100 \
        --method "$method" \
        --state_mode published_replay \
        --warm_start_mode "$warm_mode" \
        --max_iter 100 \
        --tol 1e-8
    status="$?"
    echo "END method=${method} warm=${warm_mode} status=${status}"
  done
done
