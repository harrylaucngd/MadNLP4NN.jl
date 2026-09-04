#!/usr/bin/env python3
"""Fail closed if headline ICLR-paper claims drift from raw JSON artifacts."""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from build_iclr_artifacts import paired_instance_effect, replay_rows


SUCCESS_STATUSES = {"SOLVE_SUCCEEDED", "SOLVED_TO_ACCEPTABLE_LEVEL"}


def load(path):
    return json.loads(path.read_text())


def common_success(record, tolerance=1e-6):
    return (
        record.get("status") in SUCCESS_STATUSES
        and record.get("kkt_error", np.inf) <= tolerance
        and record.get("raw_constraint_inf", 0.0) <= 1e-6
        and record.get("max_raw_constraint_violation", 0.0) <= 1e-6
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    run_dir = root / "output" / "runs"
    paper_dir = root / "paper"
    checks = {}

    replay = load(run_dir / "darcy_fd_replay64_prospective12_curvature.json")
    pilot = set(replay["config"]["selection_excluded_pilot_samples"])
    samples = set(replay["config"]["samples"])
    checks["replay_sample_count"] = len(samples) == 12
    checks["replay_disjoint_from_pilot"] = not (samples & pilot)
    replay_records = [
        (entry, entry["warm"][0])
        for entry in replay["results"]
        if entry["warm"]
    ]
    checks["replay_attempt_count"] = len(replay_records) == 360
    checks["replay_common_success_360"] = all(
        common_success(record) for _, record in replay_records
    )
    groups = {}
    for entry, record in replay_records:
        key = (entry["sample"], entry["latent_resolution"], entry["start_id"])
        groups.setdefault(key, []).append((entry["method"], record))
    checks["replay_complete_groups_120"] = (
        len(groups) == 120 and all(len(group) == 3 for group in groups.values())
    )
    ranking_agreement = sum(
        min(group, key=lambda item: item[1]["metrics"]["surrogate_target_relative_l2"])[0]
        == min(group, key=lambda item: item[1]["metrics"]["simulator_target_relative_l2"])[0]
        for group in groups.values()
    )
    checks["replay_ranking_agreement_35"] = ranking_agreement == 35
    replay_flat = replay_rows(replay)
    paired_fno = paired_instance_effect(
        replay_flat,
        resolution=32,
        first_method="gauss_newton_dlpack_gpu",
        second_method="lbfgs_cpu",
        metric="fno",
    )
    paired_pde = paired_instance_effect(
        replay_flat,
        resolution=32,
        first_method="gauss_newton_dlpack_gpu",
        second_method="lbfgs_cpu",
        metric="pde",
    )
    checks["paired_fno_improves_all_twelve_instances"] = (
        paired_fno["negative_count"] == 12
    )
    checks["paired_fno_ci_excludes_zero"] = paired_fno["ci_high"] < 0
    checks["paired_pde_ci_contains_zero"] = (
        paired_pde["ci_low"] <= 0 <= paired_pde["ci_high"]
    )

    pfr = load(run_dir / "pfr_case3_first_step_exact_r5.json")
    pfr_methods = {entry["method"]: entry for entry in pfr["results"]}
    gpu = pfr_methods["exact_dlpack_gpu"]
    cpu = pfr_methods["exact_cpu"]
    checks["pfr_case3_five_gpu_successes"] = gpu["common_success_count"] == 5
    checks["pfr_case3_five_cpu_successes"] = cpu["common_success_count"] == 5
    speedup = cpu["median_elapsed_s"] / gpu["median_elapsed_s"]
    checks["pfr_case3_speedup_rounds_to_8_7"] = round(speedup, 1) == 8.7

    mnist = load(run_dir / "mnist_prior_1024_multistart5.json")
    mnist_records = [entry["warm"][0] for entry in mnist["results"]]
    checks["mnist_fifteen_common_successes"] = (
        len(mnist_records) == 15
        and all(common_success(record, 1e-7) for record in mnist_records)
    )

    sequence_dir = run_dir / "pfr_warm_start"
    sequence_payloads = [
        load(path) for path in sorted(sequence_dir.glob("h100_80gb_*.json"))
    ]
    checks["pfr_sequence_eleven_complete_traces"] = (
        len(sequence_payloads) == 11
        and all(payload.get("complete") for payload in sequence_payloads)
        and all(len(payload["records"]) == 100 for payload in sequence_payloads)
    )
    sequence_records = [
        record for payload in sequence_payloads for record in payload["records"]
    ]
    checks["pfr_sequence_common_success_1100"] = (
        len(sequence_records) == 1100
        and all(common_success(record) for record in sequence_records)
    )
    sequence_by_key = {
        (payload["method"], payload["warm_start_mode"]): payload
        for payload in sequence_payloads
    }

    def steady_iterations(payload):
        return [
            record["iterations"]
            for record in payload["records"]
            if record["step"] not in (0, 51)
        ]

    cpu_primal = sequence_by_key[("exact_cpu", "primal")]
    cpu_dual = sequence_by_key[("exact_cpu", "primal_dual")]
    gpu_primal = sequence_by_key[("exact_dlpack_gpu", "primal")]
    gpu_dual = sequence_by_key[("exact_dlpack_gpu", "primal_dual")]
    checks["pfr_shifted_equality_duals_change_no_iterations"] = (
        [row["iterations"] for row in cpu_primal["records"]]
        == [row["iterations"] for row in cpu_dual["records"]]
        and [row["iterations"] for row in gpu_primal["records"]]
        == [row["iterations"] for row in gpu_dual["records"]]
    )
    cpu_barrier = sequence_by_key[("exact_cpu", "primal_barrier")]
    gpu_barrier = sequence_by_key[("exact_dlpack_gpu", "primal_barrier")]
    barrier_cpu_iterations = np.asarray(steady_iterations(cpu_barrier))
    barrier_gpu_iterations = np.asarray(steady_iterations(gpu_barrier))
    checks["pfr_barrier_medians_are_three"] = (
        np.median(barrier_cpu_iterations) == 3
        and np.median(barrier_gpu_iterations) == 3
    )
    checks["pfr_barrier_cpu_gpu_equal_on_66_steps"] = sum(
        first["iterations"] == second["iterations"]
        for first, second in zip(cpu_barrier["records"], gpu_barrier["records"])
    ) == 66
    persistent = sequence_by_key[("exact_dlpack_gpu_reuse", "persistent")]
    checks["pfr_persistent_has_one_audited_rebuild"] = sum(
        record["fallback_rebuild"] for record in persistent["records"]
    ) == 1
    dual_check = load(sequence_dir / "dual_initialization_check.json")
    checks["pfr_dual_start_is_nonzero"] = (
        dual_check["records"][1]["dual_initialized"]
        and dual_check["records"][1]["initial_dual_inf"] > 0.1
    )

    tolerance_payloads = [
        load(path)
        for path in sorted((run_dir / "pfr_warm_start_tolerance").glob("*.json"))
    ]
    tolerance_records = [
        record for payload in tolerance_payloads for record in payload["records"]
    ]
    checks["pfr_strict_tolerance_common_success_400"] = (
        len(tolerance_payloads) == 4
        and len(tolerance_records) == 400
        and all(common_success(record, 1e-8) for record in tolerance_records)
    )
    tolerance_by_key = {
        (payload["method"], payload["warm_start_mode"]): payload
        for payload in tolerance_payloads
    }
    strict_medians = {
        key: np.median(steady_iterations(payload))
        for key, payload in tolerance_by_key.items()
    }
    checks["pfr_strict_tolerance_medians_6_4_4_4"] = strict_medians == {
        ("exact_cpu", "primal"): 6.0,
        ("exact_cpu", "primal_barrier"): 4.0,
        ("exact_dlpack_gpu", "primal"): 4.0,
        ("exact_dlpack_gpu", "primal_barrier"): 4.0,
    }

    topology_dir = run_dir / "fno_replay_shape_memory"
    topology_results = [
        load(path)
        for path in sorted(topology_dir.glob("h100_80gb_*_z32.json"))
        if not path.name.endswith("_monitor.json")
    ]
    topology_monitors = {
        path.name.replace("_monitor.json", ""): load(path)
        for path in topology_dir.glob("h100_80gb_*_z32_monitor.json")
    }
    topology_rows = []
    for path in sorted(topology_dir.glob("h100_80gb_*_z32.json")):
        if path.name.endswith("_monitor.json"):
            continue
        result = load(path)
        monitor = topology_monitors[path.name.replace(".json", "")]
        topology_rows.append((result, monitor["peak_gpu_memory_bytes"] / 2**30))
    checks["fno_topology_six_complete_h100_graphs"] = (
        len(topology_results) == 6
        and all(result.get("complete") for result in topology_results)
        and all(monitor["exit_code"] == 0 for monitor in topology_monitors.values())
    )
    target_parameters = 1188353
    checks["fno_topology_parameter_mismatch_at_most_1_1_percent"] = all(
        abs(result["parameter_tensor_elements"] - target_parameters)
        / target_parameters
        <= 0.011
        for result, _ in topology_rows
    )
    topology_peaks = [peak for _, peak in topology_rows]
    checks["fno_topology_memory_range_32_72_to_37_25_gib"] = (
        round(min(topology_peaks), 2) == 32.72
        and round(max(topology_peaks), 2) == 37.25
    )
    four_layer = [
        (result, peak) for result, peak in topology_rows if result["layer_count"] == 4
    ]
    checks["fno_trained_random_four_layer_memory_identical"] = (
        len(four_layer) == 2 and four_layer[0][1] == four_layer[1][1]
    )
    all_topology_rows = []
    for path in topology_dir.glob("*_z32.json"):
        if path.name.endswith("_monitor.json"):
            continue
        result = load(path)
        monitor = load(Path(str(path).replace(".json", "_monitor.json")))
        all_topology_rows.append(
            (result["checkpoint_sha256"], result["device_kind"], monitor["peak_gpu_memory_bytes"] / 2**30)
        )
    cross_hardware_deltas = []
    for checksum in {row[0] for row in all_topology_rows}:
        values = [row[2] for row in all_topology_rows if row[0] == checksum]
        if len(values) == 2:
            cross_hardware_deltas.append(max(values) - min(values))
    checks["fno_cross_hardware_peak_delta_at_most_0_22_gib"] = (
        len(cross_hardware_deltas) == 6
        and max(cross_hardware_deltas) <= 0.22
    )

    latex_sources = [paper_dir / "iclr_workshop.tex", *sorted((paper_dir / "sections").glob("*.tex"))]
    latex_text = "\n".join(path.read_text() for path in latex_sources)
    cited = set()
    for match in re.finditer(r"\\cite[pt]?\{([^}]+)\}", latex_text):
        cited.update(key.strip() for key in match.group(1).split(","))
    bib_keys = set(
        re.findall(r"^@\w+\{([^,]+),", (paper_dir / "references_iclr.bib").read_text(), re.MULTILINE)
    )
    checks["all_citations_resolve"] = cited <= bib_keys
    missing_citations = sorted(cited - bib_keys)

    log = (paper_dir / "iclr_workshop.log").read_text(errors="replace")
    checks["latex_no_undefined_references"] = (
        "undefined references" not in log.lower()
        and "undefined citations" not in log.lower()
    )
    checks["latex_no_overfull_boxes"] = "Overfull" not in log
    public_sources = [
        paper_dir / "iclr_workshop.tex",
        *sorted((paper_dir / "sections").glob("*.tex")),
        *sorted((paper_dir / "generated").glob("*.tex")),
    ]
    public_text = "\n".join(path.read_text() for path in public_sources)
    checks["public_paper_has_no_internal_host_or_paths"] = not re.search(
        r"node\d+|output/|/orcd/|/home/|squeue|slurm",
        public_text,
        re.IGNORECASE,
    )

    checks = {key: bool(value) for key, value in checks.items()}
    payload = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "checks": checks,
        "all_passed": all(checks.values()),
        "derived": {
            "replay_ranking_agreement": ranking_agreement,
            "replay_groups": len(groups),
            "pfr_case3_speedup": speedup,
            "paired_fno": paired_fno,
            "paired_pde": paired_pde,
            "pfr_sequence_solves": len(sequence_records),
            "pfr_strict_tolerance_medians": {
                f"{key[0]}:{key[1]}": float(value)
                for key, value in strict_medians.items()
            },
            "fno_topology_peak_range_gib": [min(topology_peaks), max(topology_peaks)],
            "fno_cross_hardware_max_peak_delta_gib": max(cross_hardware_deltas),
            "missing_citations": missing_citations,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))
    if not payload["all_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
