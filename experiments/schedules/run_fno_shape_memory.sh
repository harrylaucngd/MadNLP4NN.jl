#!/usr/bin/env bash

# Fresh-process exact-Hessian memory sweep. Run one copy per physical GPU.
set -u

if [ "$#" -lt 3 ]; then
  echo "usage: $0 HARDWARE_LABEL GPU_INDEX MODEL [MODEL ...]" >&2
  exit 2
fi

hardware_label="$1"
gpu_index="$2"
shift 2

python_exe="${PYTHON_EXE:-${JULIA_PYTHONCALL_EXE:-}}"
if [[ -z "$python_exe" ]]; then
  python_exe="$(command -v python3 || command -v python)" || {
    echo "Python not found; set PYTHON_EXE or JULIA_PYTHONCALL_EXE" >&2
    exit 2
  }
fi
data_path="output/external/pdebench/data/2D_DarcyFlow_beta1.0_Train.hdf5"
runner="experiments/runners/benchmark_fno_hessian_memory.py"
monitor="experiments/runners/run_with_resource_monitor.py"

for model_path in "$@"; do
  model_stem="$(basename "$model_path" .pt)"
  for latent_resolution in 8 16 24 32; do
    result_path="output/runs/fno_shape_memory/${hardware_label}_${model_stem}_z${latent_resolution}.json"
    monitor_path="output/runs/fno_shape_memory/${hardware_label}_${model_stem}_z${latent_resolution}_monitor.json"
    echo "START hardware=${hardware_label} gpu=${gpu_index} model=${model_stem} latent=${latent_resolution}"
    CUDA_VISIBLE_DEVICES="$gpu_index" \
    JAX_ENABLE_X64=True \
    XLA_PYTHON_CLIENT_PREALLOCATE=false \
    PYTHONPATH=python \
      "$python_exe" "$monitor" \
        --output "$monitor_path" \
        --interval 0.02 \
        -- \
        "$python_exe" "$runner" \
          --model "$model_path" \
          --data "$data_path" \
          --output "$result_path" \
          --sample 9257 \
          --latent-resolution "$latent_resolution" \
          --repetitions 3
    status="$?"
    echo "END hardware=${hardware_label} model=${model_stem} latent=${latent_resolution} status=${status}"
  done
done
