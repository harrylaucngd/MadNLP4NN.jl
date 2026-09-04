#!/usr/bin/env bash

# Matched 100-step replay: same sequence, initializer, tolerance, and oracle.
set -u

gpu_index="${1:-0}"
hardware_label="${2:-h100_80gb}"
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
  for warm_mode in cold primal primal_barrier primal_dual primal_dual_barrier; do
    output_path="output/runs/pfr_warm_start/${hardware_label}_${method}_${warm_mode}.json"
    echo "START method=${method} warm=${warm_mode}"
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
        --tol 1e-6
    status="$?"
    echo "END method=${method} warm=${warm_mode} status=${status}"
  done
done

output_path="output/runs/pfr_warm_start/${hardware_label}_exact_dlpack_gpu_persistent.json"
echo "START method=exact_dlpack_gpu_reuse warm=persistent"
CUDA_VISIBLE_DEVICES="$gpu_index" \
JAX_ENABLE_X64=True \
XLA_PYTHON_CLIENT_PREALLOCATE=false \
JULIA_PYTHONCALL_EXE="$python_exe" \
  "$julia_exe" --project=. "$runner" \
    --config "$config" \
    --output "$output_path" \
    --steps 100 \
    --method exact_dlpack_gpu_reuse \
    --state_mode published_replay \
    --warm_start_mode persistent \
    --max_iter 100 \
    --tol 1e-6 \
    --reuse_watchdog_s 2.0
status="$?"
echo "END method=exact_dlpack_gpu_reuse warm=persistent status=${status}"
