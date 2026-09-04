#!/usr/bin/env bash

# Exact-Hessian topology sweep matched to the replay FNO in the paper.
set -u

hardware_label="${1:-h100_80gb}"
gpu_index="${2:-0}"
python_exe="${PYTHON_EXE:-${JULIA_PYTHONCALL_EXE:-}}"
if [[ -z "$python_exe" ]]; then
  python_exe="$(command -v python3 || command -v python)" || {
    echo "Python not found; set PYTHON_EXE or JULIA_PYTHONCALL_EXE" >&2
    exit 2
  }
fi
data_path="output/external/neuraloperator_darcy/darcy_test_64.pt"
runner="experiments/runners/benchmark_fno_hessian_memory.py"
monitor="experiments/runners/run_with_resource_monitor.py"

models=(
  "output/checkpoints/fno_memory_shapes_replay/fno_l1_w64_m12.pt"
  "output/checkpoints/fno_memory_shapes_replay/fno_l2_w45_m12.pt"
  "output/checkpoints/fno_memory_shapes_replay/fno_l3_w37_m12.pt"
  "output/checkpoints/fno_memory_shapes_replay/fno_l4_w32_m12.pt"
  "output/external/neuraloperator_darcy/fno64_width32_m12_leftgrid_seed20260821.pt"
  "output/checkpoints/fno_memory_shapes_replay/fno_l6_w26_m12.pt"
)

for model_path in "${models[@]}"; do
  model_stem="$(basename "$model_path" .pt)"
  result_path="output/runs/fno_replay_shape_memory/${hardware_label}_${model_stem}_z32.json"
  monitor_path="output/runs/fno_replay_shape_memory/${hardware_label}_${model_stem}_z32_monitor.json"
  echo "START hardware=${hardware_label} gpu=${gpu_index} model=${model_stem} latent=32"
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
        --sample 124 \
        --latent-resolution 32 \
        --full-resolution 64 \
        --repetitions 3
  status="$?"
  echo "END hardware=${hardware_label} model=${model_stem} status=${status}"
done
