#!/usr/bin/env python3
"""Summarize the PFR warm-start and parameter-matched FNO topology ablations."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


POLICY_LABELS = {
    "cold": "cold rollout",
    "primal": "shifted primal",
    "primal_barrier": r"primal + $\mu_0=10^{-7}$",
    "primal_dual": "shifted primal--dual",
    "primal_dual_barrier": r"primal--dual + $\mu_0=10^{-7}$",
    "persistent": "persistent solver",
}


def quantile(values, probability):
    return float(np.quantile(np.asarray(values, dtype=float), probability))


def sequence_records(directory: Path):
    rows = []
    traces = {}
    for path in sorted(directory.glob("*.json")):
        payload = json.loads(path.read_text())
        if not payload.get("complete") or len(payload.get("records", [])) != 100:
            continue
        records = payload["records"]
        policy = payload["warm_start_mode"]
        method = payload["method"]
        kkt_device = "GPU" if "dlpack" in method else "CPU"
        if method.endswith("reuse"):
            kkt_device = "GPU reuse"
        steady = [record for record in records if record["step"] not in (0, 51)]
        iterations = [record["iterations"] for record in steady]
        elapsed_ms = [1e3 * record["elapsed_s"] for record in steady]
        successes = [
            record
            for record in records
            if record["status"] == "SOLVE_SUCCEEDED"
            and record["kkt_error"] <= 1e-6
        ]
        row = {
            "path": str(path),
            "method": method,
            "kkt_device": kkt_device,
            "policy": policy,
            "policy_label": POLICY_LABELS[policy],
            "success_count": len(successes),
            "steady_iteration_median": quantile(iterations, 0.5),
            "steady_iteration_mean": float(np.mean(iterations)),
            "steady_iteration_p95": quantile(iterations, 0.95),
            "steady_iteration_max": int(max(iterations)),
            "transition_iterations": int(records[51]["iterations"]),
            "steady_time_median_ms": quantile(elapsed_ms, 0.5),
            "steady_time_p95_ms": quantile(elapsed_ms, 0.95),
            "steady_time_max_ms": float(max(elapsed_ms)),
            "initial_constraint_median": quantile(
                [record["initial_constraint_inf"] for record in steady], 0.5
            ),
            "max_kkt_error": float(max(record["kkt_error"] for record in records)),
            "max_control_inf_error": float(
                max(record["control_inf_error"] for record in records)
            ),
            "fallback_count": sum(record["fallback_rebuild"] for record in records),
        }
        rows.append(row)
        traces[(kkt_device, policy)] = [record["iterations"] for record in records]

    comparisons = []
    for policy in (
        "cold",
        "primal",
        "primal_barrier",
        "primal_dual",
        "primal_dual_barrier",
    ):
        cpu = traces.get(("CPU", policy))
        gpu = traces.get(("GPU", policy))
        if cpu is None or gpu is None:
            continue
        differences = np.asarray(gpu) - np.asarray(cpu)
        comparisons.append(
            {
                "policy": policy,
                "equal_iteration_steps": int(np.sum(differences == 0)),
                "median_gpu_minus_cpu_iterations": float(np.median(differences)),
                "max_abs_iteration_difference": int(np.max(np.abs(differences))),
            }
        )
    return rows, traces, comparisons


def topology_records(directory: Path):
    rows = []
    for monitor_path in sorted(directory.glob("*_monitor.json")):
        monitor = json.loads(monitor_path.read_text())
        result_path = Path(str(monitor_path).replace("_monitor.json", ".json"))
        result = json.loads(result_path.read_text()) if result_path.exists() else None
        command = monitor.get("command", [])
        model_path = command[command.index("--model") + 1] if "--model" in command else ""
        row = {
            "monitor_path": str(monitor_path),
            "result_path": str(result_path) if result else None,
            "model": model_path,
            "exit_code": monitor["exit_code"],
            "peak_gpu_gib": monitor["peak_gpu_memory_bytes"] / 2**30,
            "peak_host_gib": monitor["peak_process_tree_rss_bytes"] / 2**30,
            "process_elapsed_s": monitor["elapsed_s"],
        }
        if result:
            row.update(
                {
                    "hardware": result["device_kind"],
                    "checkpoint_sha256": result["checkpoint_sha256"],
                    "layer_count": result["layer_count"],
                    "width": result["widths"][0],
                    "modes": result["modes1"],
                    "parameter_tensor_elements": result["parameter_tensor_elements"],
                    "parameter_real_scalars": result["parameter_real_scalars"],
                    "latent_resolution": result["latent_resolution"],
                    "returned_hessian_mib": result["returned_hessian_bytes"] / 2**20,
                    "cold_hessian_s": result["cold_compile_and_evaluate_s"],
                    "warm_hessian_median_s": float(
                        np.median(result["warm_evaluate_s"])
                    ),
                    "trained": result["trained"],
                }
            )
        rows.append(row)
    return rows


def tolerance_records(directory: Path):
    rows = []
    if not directory.exists():
        return rows
    for path in sorted(directory.glob("*.json")):
        payload = json.loads(path.read_text())
        if not payload.get("complete") or len(payload.get("records", [])) != 100:
            continue
        records = payload["records"]
        steady = [record for record in records if record["step"] not in (0, 51)]
        method = payload["method"]
        rows.append(
            {
                "path": str(path),
                "kkt_device": "GPU" if "dlpack" in method else "CPU",
                "policy": payload["warm_start_mode"],
                "success_count": payload["common_success_count"],
                "steady_iteration_median": quantile(
                    [record["iterations"] for record in steady], 0.5
                ),
                "steady_iteration_p95": quantile(
                    [record["iterations"] for record in steady], 0.95
                ),
                "steady_time_median_ms": 1e3
                * quantile([record["elapsed_s"] for record in steady], 0.5),
                "max_kkt_error": max(record["kkt_error"] for record in records),
                "max_control_inf_error": max(
                    record["control_inf_error"] for record in records
                ),
            }
        )
    return rows


def write_tex(path: Path, sequences, topology, tolerance):
    sequence_order = {
        "cold": 0,
        "primal": 1,
        "primal_barrier": 2,
        "primal_dual": 3,
        "primal_dual_barrier": 4,
        "persistent": 5,
    }
    sequence_rows = sorted(
        sequences,
        key=lambda row: (row["kkt_device"], sequence_order[row["policy"]]),
    )
    topology_rows = sorted(
        [row for row in topology if row.get("exit_code") == 0],
        key=lambda row: row["layer_count"],
    )
    lines = [
        "% Generated by experiments/analysis/analyze_sequence_and_topology.py",
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{PFR case-1 initialization ablation on one fixed 100-step replay. Steady statistics exclude the first solve and the setpoint change at step 51; iterations are median (95th percentile). Every entry uses the same exact derivatives and passes the common KKT audit.}",
        r"\label{tab:warm-start}",
        r"\small",
        r"\begin{tabular}{llrrrr}",
        r"\toprule",
        r"KKT & initialization & success & iterations & median ms & step-51 iter.\\",
        r"\midrule",
    ]
    for row in sequence_rows:
        lines.append(
            f"{row['kkt_device']} & {row['policy_label']} & "
            f"{row['success_count']}/100 & {row['steady_iteration_median']:.0f} "
            f"({row['steady_iteration_p95']:.0f}) & "
            f"{row['steady_time_median_ms']:.1f} & "
            f"{row['transition_iterations']}\\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])
    if topology_rows:
        lines.extend(
            [
                r"\begin{table}[t]",
                r"\centering",
                r"\caption{Exact-Hessian memory under parameter-matched FNO topology changes. All networks use 12 Fourier modes and $32^2$ latent variables; parameter counts differ by at most 1.1\%. Random checkpoints are used only to instantiate static AD graphs; the two four-layer rows verify that trained and random weights yield the same allocation.}",
                r"\label{tab:topology-memory}",
                r"\small",
                r"\begin{tabular}{rrrrrr}",
                r"\toprule",
                r"layers & width & parameters & GPU GiB & warm Hessian s & trained\\",
                r"\midrule",
            ]
        )
        for row in topology_rows:
            lines.append(
                f"{row['layer_count']} & {row['width']} & "
                f"{row['parameter_tensor_elements']/1e6:.3f}M & "
                f"{row['peak_gpu_gib']:.2f} & {row['warm_hessian_median_s']:.3f} & "
                f"{'yes' if row['trained'] else 'no'}\\\\"
            )
        lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])
    if tolerance:
        lines.extend(
            [
                r"\begin{table}[t]",
                r"\centering",
                r"\caption{Tolerance sensitivity on the same replay. Steady iterations are median (95th percentile); all solves satisfy $10^{-8}$.}",
                r"\label{tab:warm-start-tolerance}",
                r"\small",
                r"\begin{tabular}{llrrr}",
                r"\toprule",
                r"KKT & initialization & success & iterations & median ms\\",
                r"\midrule",
            ]
        )
        for row in sorted(tolerance, key=lambda item: (item["kkt_device"], item["policy"])):
            lines.append(
                f"{row['kkt_device']} & {POLICY_LABELS[row['policy']]} & "
                f"{row['success_count']}/100 & "
                f"{row['steady_iteration_median']:.0f} "
                f"({row['steady_iteration_p95']:.0f}) & "
                f"{row['steady_time_median_ms']:.1f}\\\\"
            )
        lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def write_macros(path: Path, sequences, comparisons, topology, all_topology, tolerance):
    by_key = {(row["kkt_device"], row["policy"]): row for row in sequences}

    def seq(device, policy, field):
        return by_key[(device, policy)][field]

    def tex_scientific(value):
        mantissa, exponent = f"{value:.2e}".split("e")
        return rf"{mantissa}\times 10^{{{int(exponent)}}}"

    successful = [row for row in topology if row.get("exit_code") == 0]
    peaks = [row["peak_gpu_gib"] for row in successful]
    trained_four = next(
        row for row in successful if row["layer_count"] == 4 and row["trained"]
    )
    random_four = next(
        row for row in successful if row["layer_count"] == 4 and not row["trained"]
    )
    barrier_comparison = next(
        row for row in comparisons if row["policy"] == "primal_barrier"
    )
    target_parameters = trained_four["parameter_tensor_elements"]
    max_parameter_mismatch = max(
        abs(row["parameter_tensor_elements"] - target_parameters) / target_parameters
        for row in successful
    )
    cross_hardware_deltas = []
    for checksum in {row.get("checkpoint_sha256") for row in all_topology}:
        values = [
            row["peak_gpu_gib"]
            for row in all_topology
            if row.get("exit_code") == 0
            and row.get("checkpoint_sha256") == checksum
        ]
        if len(values) > 1:
            cross_hardware_deltas.append(max(values) - min(values))
    strict = {(row["kkt_device"], row["policy"]): row for row in tolerance}
    lines = [
        "% Generated by experiments/analysis/analyze_sequence_and_topology.py",
        rf"\newcommand{{\PFRSequenceSuccesses}}{{{sum(row['success_count'] for row in sequences)}}}",
        rf"\newcommand{{\PFRSequenceTotal}}{{{100 * len(sequences)}}}",
        rf"\newcommand{{\PFRWarmCPUColdIter}}{{{seq('CPU', 'cold', 'steady_iteration_median'):.0f}}}",
        rf"\newcommand{{\PFRWarmCPUPrimalIter}}{{{seq('CPU', 'primal', 'steady_iteration_median'):.0f}}}",
        rf"\newcommand{{\PFRWarmCPUBarrierIter}}{{{seq('CPU', 'primal_barrier', 'steady_iteration_median'):.0f}}}",
        rf"\newcommand{{\PFRWarmGPUColdIter}}{{{seq('GPU', 'cold', 'steady_iteration_median'):.0f}}}",
        rf"\newcommand{{\PFRWarmGPUPrimalIter}}{{{seq('GPU', 'primal', 'steady_iteration_median'):.0f}}}",
        rf"\newcommand{{\PFRWarmGPUBarrierIter}}{{{seq('GPU', 'primal_barrier', 'steady_iteration_median'):.0f}}}",
        rf"\newcommand{{\PFRWarmEqualBarrierSteps}}{{{barrier_comparison['equal_iteration_steps']}}}",
        rf"\newcommand{{\PFRWarmCPUBarrierTime}}{{{seq('CPU', 'primal_barrier', 'steady_time_median_ms'):.1f}}}",
        rf"\newcommand{{\PFRWarmGPUBarrierTime}}{{{seq('GPU', 'primal_barrier', 'steady_time_median_ms'):.1f}}}",
        rf"\newcommand{{\PFRPersistentMedianTime}}{{{seq('GPU reuse', 'persistent', 'steady_time_median_ms'):.1f}}}",
        rf"\newcommand{{\PFRPersistentMaxTime}}{{{seq('GPU reuse', 'persistent', 'steady_time_max_ms') / 1000:.2f}}}",
        rf"\newcommand{{\PFRWarmBarrierControlError}}{{{tex_scientific(max(seq('CPU', 'primal_barrier', 'max_control_inf_error'), seq('GPU', 'primal_barrier', 'max_control_inf_error')))}}}",
        rf"\newcommand{{\FNOShapeMinMemory}}{{{min(peaks):.2f}}}",
        rf"\newcommand{{\FNOShapeMaxMemory}}{{{max(peaks):.2f}}}",
        rf"\newcommand{{\FNOShapeMemoryDelta}}{{{max(peaks) - min(peaks):.2f}}}",
        rf"\newcommand{{\FNOShapeMemoryIncrease}}{{{100 * (max(peaks) / min(peaks) - 1):.1f}}}",
        rf"\newcommand{{\FNOShapeMaxParameterMismatch}}{{{100 * max_parameter_mismatch:.1f}}}",
        rf"\newcommand{{\FNOShapeTrainedMemory}}{{{trained_four['peak_gpu_gib']:.2f}}}",
        rf"\newcommand{{\FNOShapeRandomMemory}}{{{random_four['peak_gpu_gib']:.2f}}}",
        rf"\newcommand{{\FNOShapeFastHessian}}{{{1e3 * min(row['warm_hessian_median_s'] for row in successful):.0f}}}",
        rf"\newcommand{{\FNOShapeSlowHessian}}{{{1e3 * max(row['warm_hessian_median_s'] for row in successful):.0f}}}",
        rf"\newcommand{{\FNOShapeCrossHardwareDelta}}{{{max(cross_hardware_deltas):.2f}}}",
        rf"\newcommand{{\PFRStrictCPUPrimalIter}}{{{strict[('CPU', 'primal')]['steady_iteration_median']:.0f}}}",
        rf"\newcommand{{\PFRStrictCPUBarrierIter}}{{{strict[('CPU', 'primal_barrier')]['steady_iteration_median']:.0f}}}",
        rf"\newcommand{{\PFRStrictGPUPrimalIter}}{{{strict[('GPU', 'primal')]['steady_iteration_median']:.0f}}}",
        rf"\newcommand{{\PFRStrictGPUBarrierIter}}{{{strict[('GPU', 'primal_barrier')]['steady_iteration_median']:.0f}}}",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def plot(path: Path, traces, topology):
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 3.4))
    colors = {
        "cold": "#777777",
        "primal": "#2f6b9a",
        "primal_barrier": "#d17824",
        "primal_dual": "#5f3c8a",
        "primal_dual_barrier": "#bd3f3f",
        "persistent": "#2b8a66",
    }
    plot_labels = {
        "cold": "cold rollout",
        "primal": "shifted primal",
        "primal_barrier": "primal + low barrier",
        "primal_dual_barrier": "primal-dual + low barrier",
    }
    for (device, policy), values in traces.items():
        if device != "CPU" or policy not in (
            "cold",
            "primal",
            "primal_barrier",
            "primal_dual_barrier",
        ):
            continue
        axes[0].plot(
            np.arange(len(values)),
            values,
            label=plot_labels[policy],
            color=colors[policy],
            linewidth=1.35,
        )
    axes[0].axvline(51, color="black", linestyle=":", linewidth=0.9)
    axes[0].set_xlabel("sequence step")
    axes[0].set_ylabel("interior-point iterations")
    axes[0].set_title("Initialization changes iterations")
    axes[0].legend(frameon=False, fontsize=7)

    successful_all = sorted(
        [row for row in topology if row.get("exit_code") == 0],
        key=lambda row: row["layer_count"],
    )
    successful = []
    for row in successful_all:
        if row["layer_count"] == 4 and not row["trained"]:
            continue
        successful.append(row)
    if successful:
        axes[1].plot(
            [row["layer_count"] for row in successful],
            [row["peak_gpu_gib"] for row in successful],
            marker="o",
            color="#bd3f3f",
            linewidth=1.5,
        )
        for row in successful:
            axes[1].annotate(
                f"w={row['width']}",
                (row["layer_count"], row["peak_gpu_gib"]),
                xytext=(3, 4),
                textcoords="offset points",
            fontsize=7,
            )
        random_four = [
            row for row in successful_all
            if row["layer_count"] == 4 and not row["trained"]
        ]
        if random_four:
            axes[1].scatter(
                [4],
                [random_four[0]["peak_gpu_gib"]],
                facecolors="none",
                edgecolors="#bd3f3f",
                s=42,
                linewidths=1.1,
                label="random-weight control",
            )
    axes[1].set_xlabel("Fourier layers (parameter matched)")
    axes[1].set_ylabel("fresh-process peak GPU GiB")
    axes[1].set_title("Parameter count does not fix AD memory")
    axes[1].grid(axis="y", color="#dddddd", linewidth=0.6)
    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sequence-dir", type=Path, default=Path("output/runs/pfr_warm_start")
    )
    parser.add_argument(
        "--topology-dir",
        type=Path,
        default=Path("output/runs/fno_replay_shape_memory"),
    )
    parser.add_argument(
        "--primary-device-kind", default="NVIDIA H100 80GB HBM3"
    )
    parser.add_argument(
        "--tolerance-dir",
        type=Path,
        default=Path("output/runs/pfr_warm_start_tolerance"),
    )
    parser.add_argument(
        "--output", type=Path, default=Path("output/runs/sequence_topology_summary.json")
    )
    parser.add_argument(
        "--tex", type=Path, default=Path("paper/generated/sequence_topology_tables.tex")
    )
    parser.add_argument(
        "--macros", type=Path, default=Path("paper/generated/sequence_topology_macros.tex")
    )
    parser.add_argument(
        "--figure", type=Path, default=Path("paper/figures/sequence_topology.pdf")
    )
    args = parser.parse_args()

    sequences, traces, comparisons = sequence_records(args.sequence_dir)
    topology = topology_records(args.topology_dir) if args.topology_dir.exists() else []
    tolerance = tolerance_records(args.tolerance_dir)
    primary_topology = [
        row for row in topology
        if row.get("hardware") == args.primary_device_kind
    ]
    successful_memory = [
        row for row in primary_topology if row.get("exit_code") == 0
    ]
    summary = {
        "schema_version": 1,
        "complete": (
            len(sequences) == 11
            and len(successful_memory) == 6
            and len(tolerance) == 4
        ),
        "sequence_rows": sequences,
        "cpu_gpu_iteration_comparisons": comparisons,
        "topology_rows": topology,
        "tolerance_rows": tolerance,
        "primary_topology_device": args.primary_device_kind,
    }
    if successful_memory:
        peaks = [row["peak_gpu_gib"] for row in successful_memory]
        summary["topology_peak_ratio"] = max(peaks) / min(peaks)
        summary["topology_peak_range_gib"] = [min(peaks), max(peaks)]
    cross_hardware_deltas = []
    for checksum in {row.get("checkpoint_sha256") for row in topology}:
        values = [
            row["peak_gpu_gib"]
            for row in topology
            if row.get("exit_code") == 0
            and row.get("checkpoint_sha256") == checksum
        ]
        if len(values) > 1:
            cross_hardware_deltas.append(max(values) - min(values))
    if cross_hardware_deltas:
        summary["cross_hardware_max_peak_delta_gib"] = max(cross_hardware_deltas)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n")
    write_tex(args.tex, sequences, primary_topology, tolerance)
    write_macros(
        args.macros,
        sequences,
        comparisons,
        primary_topology,
        topology,
        tolerance,
    )
    plot(args.figure, traces, primary_topology)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
