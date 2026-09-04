#!/usr/bin/env python3
"""Aggregate the current validated pilot/prospective artifacts into one JSON."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def load(run_dir, name):
    return json.loads((run_dir / name).read_text())


def common_success(record, tolerance=1e-6):
    return (
        record.get("status")
        in ("SOLVE_SUCCEEDED", "SOLVED_TO_ACCEPTABLE_LEVEL")
        and record.get("kkt_error", np.inf) <= tolerance
    )


def summarize_pfr_first_step(run_dir):
    madnlp = load(run_dir, "pfr_nmpc_first_step_structured_paths_r5_v2.json")
    rows = []
    for method in madnlp["results"]:
        successful = [record for record in method["warm"] if common_success(record)]
        rows.append(
            {
                "method": method["method"],
                "successes": len(successful),
                "attempts": len(method["warm"]),
                "median_elapsed_s": float(
                    np.median([record["elapsed_s"] for record in successful])
                ),
                "median_iterations": float(
                    np.median([record["iterations"] for record in successful])
                ),
            }
        )
    for filename, method_name in (
        ("pfr_nmpc_cyipopt_structured_exact_r3.json", "cyipopt_exact"),
        ("pfr_nmpc_cyipopt_structured_lbfgs_r3.json", "cyipopt_limited_memory"),
    ):
        payload = load(run_dir, filename)
        records = payload["records"][1:]
        rows.append(
            {
                "method": method_name,
                "successes": sum(record["status"] == 0 for record in records),
                "attempts": len(records),
                "median_elapsed_s": float(
                    np.median([record["elapsed_s"] for record in records])
                ),
            }
        )
    full = load(run_dir, "pfr_nmpc_omlt_full_p40_r3.json")
    rows.append(
        {
            "method": "omlt_full_space_ipopt",
            "successes": sum(
                record["termination_condition"] == "optimal"
                for record in full["records"]
            ),
            "attempts": len(full["records"]),
            "median_elapsed_s": float(
                np.median([record["elapsed_s"] for record in full["records"]])
            ),
            "variable_count": full["variable_count"],
            "constraint_count": full["constraint_count"],
        }
    )
    return rows


def summarize_pfr_closed_loop(run_dir):
    rows = []
    for filename, method_name in (
        ("pfr_nmpc_closed_loop_structured_exact_cpu.json", "madnlp_exact_cpu_kkt"),
        ("pfr_nmpc_closed_loop_structured_reuse_gpu.json", "madnlp_persistent_dlpack_gpu"),
    ):
        payload = load(run_dir, filename)
        warm = payload["records"][1:]
        times = np.asarray([record["elapsed_s"] for record in warm])
        rows.append(
            {
                "method": method_name,
                "successes": payload["common_success_count"],
                "attempts": payload["steps"],
                "median_elapsed_s": float(np.median(times)),
                "mean_elapsed_s": float(np.mean(times)),
                "q95_elapsed_s": float(np.quantile(times, 0.95)),
                "max_elapsed_s": float(np.max(times)),
                "fallback_count": sum(
                    record.get("fallback_rebuild", False)
                    for record in payload["records"]
                ),
                "max_control_inf_error": payload["max_control_inf_error"],
                "max_state_inf_error": payload["max_state_inf_error"],
            }
        )
    return rows


def summarize_pfr_scale(run_dir):
    """Collect the published CNN cases without assuming every method succeeded."""
    rows = []
    julia_runs = (
        (2, "pfr_case2_first_step_exact_r5.json"),
        (3, "pfr_case3_first_step_exact_r5.json"),
        (3, "pfr_case3_first_step_lbfgs_cpu.json"),
    )
    for case, filename in julia_runs:
        path = run_dir / filename
        if not path.exists():
            continue
        payload = json.loads(path.read_text())
        for entry in payload["results"]:
            warm = entry["warm"]
            successful = [
                record
                for record in warm
                if common_success(record)
                and record.get("raw_constraint_inf", np.inf) <= 1e-6
            ]
            rows.append(
                {
                    "case": case,
                    "n": payload["n"],
                    "m": payload["m"],
                    "jacobian_nnz": payload["jacobian_nnz"],
                    "hessian_nnz": payload["hessian_nnz"],
                    "method": entry["method"],
                    "successes": len(successful),
                    "attempts": len(warm),
                    "median_elapsed_s": (
                        float(np.median([record["elapsed_s"] for record in successful]))
                        if successful
                        else None
                    ),
                    "median_iterations": (
                        float(np.median([record["iterations"] for record in successful]))
                        if successful
                        else None
                    ),
                    "objective": float(warm[0]["objective"]) if warm else None,
                    "max_raw_constraint_inf": (
                        float(max(record["raw_constraint_inf"] for record in warm))
                        if warm
                        else None
                    ),
                }
            )

    for case in (2, 3):
        for curvature in ("exact", "limited_memory"):
            preferred = (
                run_dir / "pfr_case3_cyipopt_exact_r3.json"
                if case == 3 and curvature == "exact"
                else run_dir / f"pfr_case{case}_cyipopt_{curvature}.json"
            )
            path = preferred
            if not path.exists():
                continue
            payload = json.loads(path.read_text())
            warm = [record for record in payload["records"] if not record["warmup"]]
            successful = [
                record
                for record in warm
                if record["status"] == 0
                and record["primal_feas"] <= 1e-6
                and record["dual_feas"] <= 1e-6
                and record["complementarity"] <= 1e-6
            ]
            rows.append(
                {
                    "case": case,
                    "n": payload["n"],
                    "m": payload["m"],
                    "jacobian_nnz": payload["jacobian_nnz"],
                    "hessian_nnz": payload["hessian_nnz"],
                    "method": f"cyipopt_{curvature}",
                    "successes": len(successful),
                    "attempts": len(warm),
                    "median_elapsed_s": (
                        float(np.median([record["elapsed_s"] for record in successful]))
                        if successful
                        else None
                    ),
                    "median_iterations": None,
                    "objective": float(warm[0]["objective"]) if warm else None,
                    "max_raw_constraint_inf": (
                        float(max(record["raw_constraint_inf"] for record in warm))
                        if warm
                        else None
                    ),
                }
            )
    return sorted(rows, key=lambda row: (row["case"], row["method"]))


def summarize_darcy(run_dir):
    lbfgs = load(run_dir, "neuraloperator_darcy64_prospective12_lbfgs.json")
    exact = load(run_dir, "neuraloperator_darcy64_prospective_hard3_exact_dlpack.json")
    rows = []
    failures = []
    lbfgs_by_key = {}
    for entry in lbfgs["results"]:
        record = entry["warm"][0]
        key = (entry["sample"], entry["latent_resolution"], entry["start_id"])
        lbfgs_by_key[key] = record
        if not common_success(record):
            failures.append(key)
        rows.append(
            {
                "key": key,
                "success": common_success(record),
                "elapsed_s": record["elapsed_s"],
                "surrogate_residual": record["metrics"][
                    "surrogate_target_relative_l2"
                ],
                "coefficient_error": record["metrics"]["coefficient_relative_l2"],
                "representability_floor": record["instance_baselines"][
                    "coefficient_relative_l2_floor"
                ],
                "true_coefficient_forward_error": record["instance_baselines"][
                    "true_coefficient_surrogate_target_relative_l2"
                ],
            }
        )
    exact_by_key = {
        (entry["sample"], entry["latent_resolution"], entry["start_id"]): entry[
            "warm"
        ][0]
        for entry in exact["results"]
    }
    lbfgs_total = sum(record["elapsed_s"] for record in lbfgs_by_key.values())
    fallback_time = sum(exact_by_key[key]["elapsed_s"] for key in failures)
    by_resolution = {}
    for resolution in (16, 32):
        successful = [
            row for row in rows if row["key"][1] == resolution and row["success"]
        ]
        by_resolution[str(resolution)] = {
            "successes": len(successful),
            "attempts": sum(row["key"][1] == resolution for row in rows),
            "median_coefficient_error": float(
                np.median([row["coefficient_error"] for row in successful])
            ),
            "median_coefficient_gap_over_floor": float(
                np.median(
                    [
                        row["coefficient_error"] - row["representability_floor"]
                        for row in successful
                    ]
                )
            ),
            "median_surrogate_to_true_forward_ratio": float(
                np.median(
                    [
                        row["surrogate_residual"]
                        / row["true_coefficient_forward_error"]
                        for row in successful
                    ]
                )
            ),
        }
    return {
        "lbfgs_successes": len(rows) - len(failures),
        "attempts": len(rows),
        "failed_keys": failures,
        "exact_successes_on_hard_subset": sum(
            common_success(record) for record in exact_by_key.values()
        ),
        "exact_attempts_on_hard_subset": len(exact_by_key),
        "routed_successes": len(rows)
        - len(failures)
        + sum(common_success(exact_by_key[key]) for key in failures),
        "routing_time_overhead_fraction": fallback_time / lbfgs_total,
        "by_resolution": by_resolution,
    }


def summarize_darcy_replay(run_dir):
    path = run_dir / "darcy_fd_replay64_prospective12_curvature.json"
    if not path.exists():
        return {}
    payload = json.loads(path.read_text())
    rows = []
    for entry in payload["results"]:
        if not entry["warm"]:
            continue
        record = entry["warm"][0]
        baseline = record["instance_baselines"]
        metrics = record["metrics"]
        rows.append(
            {
                "sample": entry["sample"],
                "latent_resolution": entry["latent_resolution"],
                "start_id": entry["start_id"],
                "method": entry["method"],
                "success": common_success(record),
                "elapsed_s": record["elapsed_s"],
                "iterations": record["iterations"],
                "coefficient_error": metrics["coefficient_relative_l2"],
                "representability_floor": baseline[
                    "coefficient_relative_l2_floor"
                ],
                "surrogate_residual": metrics[
                    "surrogate_target_relative_l2"
                ],
                "simulator_residual": metrics[
                    "simulator_target_relative_l2"
                ],
                "true_coefficient_surrogate_residual": baseline[
                    "true_coefficient_surrogate_target_relative_l2"
                ],
            }
        )
    by_method_resolution = {}
    for method in sorted({row["method"] for row in rows}):
        for resolution in sorted({row["latent_resolution"] for row in rows}):
            selected = [
                row
                for row in rows
                if row["method"] == method
                and row["latent_resolution"] == resolution
                and row["success"]
            ]
            if not selected:
                continue
            by_method_resolution[f"{method}:latent{resolution}"] = {
                "successes": len(selected),
                "attempts": sum(
                    row["method"] == method
                    and row["latent_resolution"] == resolution
                    for row in rows
                ),
                "median_elapsed_s": float(
                    np.median([row["elapsed_s"] for row in selected])
                ),
                "median_iterations": float(
                    np.median([row["iterations"] for row in selected])
                ),
                "median_coefficient_error": float(
                    np.median([row["coefficient_error"] for row in selected])
                ),
                "median_gap_over_representability_floor": float(
                    np.median(
                        [
                            row["coefficient_error"]
                            - row["representability_floor"]
                            for row in selected
                        ]
                    )
                ),
                "median_surrogate_residual": float(
                    np.median([row["surrogate_residual"] for row in selected])
                ),
                "median_simulator_residual": float(
                    np.median([row["simulator_residual"] for row in selected])
                ),
                "median_simulator_to_surrogate_ratio": float(
                    np.median(
                        [
                            row["simulator_residual"] / row["surrogate_residual"]
                            for row in selected
                        ]
                    )
                ),
                "outfits_true_coefficient_count": sum(
                    row["surrogate_residual"]
                    < row["true_coefficient_surrogate_residual"]
                    for row in selected
                ),
            }

    groups = {}
    for row in rows:
        if row["success"]:
            key = (row["sample"], row["latent_resolution"], row["start_id"])
            groups.setdefault(key, []).append(row)
    complete = [group for group in groups.values() if len(group) == 3]
    ranking_agreement = sum(
        min(group, key=lambda row: row["surrogate_residual"])["method"]
        == min(group, key=lambda row: row["simulator_residual"])["method"]
        for group in complete
    )
    return {
        "rows": rows,
        "by_method_resolution": by_method_resolution,
        "surrogate_simulator_ranking_agreement": ranking_agreement,
        "complete_method_groups": len(complete),
    }


def summarize_mnist(run_dir):
    rows = []
    for filename in (
        "mnist_prior_128_paths_tol1e7.json",
        "mnist_prior_large_paths_r3.json",
        "mnist_prior_large_exact_cpu_r1.json",
    ):
        payload = load(run_dir, filename)
        for entry in payload["results"]:
            successful = [
                record
                for record in entry["warm"]
                if common_success(record, 1e-7)
                and record["max_raw_constraint_violation"] <= 1e-6
            ]
            rows.append(
                {
                    "parameter_count": entry["parameter_count"],
                    "method": entry["method"],
                    "successes": len(successful),
                    "attempts": len(entry["warm"]),
                    "median_elapsed_s": float(
                        np.median([record["elapsed_s"] for record in successful])
                    ),
                    "objective": float(successful[0]["objective"]),
                    "max_raw_constraint_violation": float(
                        max(
                            record["max_raw_constraint_violation"]
                            for record in successful
                        )
                    ),
                }
            )
    for filename, method in (
        ("mnist_prior_cyipopt_exact_r1.json", "cyipopt_exact"),
        ("mnist_prior_cyipopt_lbfgs_r1.json", "cyipopt_limited_memory"),
    ):
        payload = load(run_dir, filename)
        for record in payload["records"]:
            if record["warmup"]:
                continue
            rows.append(
                {
                    "parameter_count": record["parameter_count"],
                    "method": method,
                    "successes": int(
                        record["status"] == 0
                        and record["max_raw_constraint_violation"] <= 1e-6
                    ),
                    "attempts": 1,
                    "median_elapsed_s": record["elapsed_s"],
                    "objective": record["objective"],
                    "max_raw_constraint_violation": record[
                        "max_raw_constraint_violation"
                    ],
                }
            )
    return sorted(rows, key=lambda row: (row["parameter_count"], row["method"]))


def summarize_mnist_multistart(run_dir):
    path = run_dir / "mnist_prior_1024_multistart5.json"
    if not path.exists():
        return {}
    payload = json.loads(path.read_text())
    rows = []
    for entry in payload["results"]:
        record = entry["warm"][0]
        success = (
            common_success(record, 1e-7)
            and record["max_raw_constraint_violation"] <= 1e-6
        )
        rows.append(
            {
                "parameter_count": entry["parameter_count"],
                "start_id": entry["start_id"],
                "method": entry["method"],
                "success": success,
                "elapsed_s": record["elapsed_s"],
                "objective": record["objective"],
                "iterations": record["iterations"],
                "max_raw_constraint_violation": record[
                    "max_raw_constraint_violation"
                ],
            }
        )
    by_method = {}
    for method in sorted({row["method"] for row in rows}):
        successful = [
            row for row in rows if row["method"] == method and row["success"]
        ]
        by_method[method] = {
            "successes": len(successful),
            "attempts": sum(row["method"] == method for row in rows),
            "median_elapsed_s": float(
                np.median([row["elapsed_s"] for row in successful])
            ),
            "median_objective": float(
                np.median([row["objective"] for row in successful])
            ),
            "min_objective": float(min(row["objective"] for row in successful)),
            "max_objective": float(max(row["objective"] for row in successful)),
        }
    return {"rows": rows, "by_method": by_method}


def summarize_memory(run_dir):
    cases = (
        ("pfr_exact_cpu", "resource_pfr_exact_cpu.json"),
        ("pfr_exact_dlpack", "resource_pfr_exact_dlpack.json"),
        ("mnist_5m_exact_dlpack", "resource_mnist1024_exact_dlpack.json"),
        ("mnist_5m_lbfgs", "resource_mnist1024_lbfgs.json"),
        ("darcy_latent32_exact_dlpack", "resource_darcy32_exact_dlpack.json"),
        ("darcy_latent32_lbfgs", "resource_darcy32_lbfgs.json"),
        ("pfr_case3_exact_cpu", "resource_pfr_case3_exact_cpu.json"),
        ("pfr_case3_exact_dlpack", "resource_pfr_case3_exact_dlpack.json"),
        ("pfr_case3_lbfgs", "resource_pfr_case3_lbfgs_cpu.json"),
        (
            "darcy_replay_latent32_exact_dlpack",
            "resource_darcy_fd_replay32_exact_dlpack_noprealloc.json",
        ),
        (
            "darcy_replay_latent32_gauss_newton_dlpack",
            "resource_darcy_fd_replay32_gauss_newton_dlpack_noprealloc.json",
        ),
    )
    rows = []
    for name, filename in cases:
        if not (run_dir / filename).exists():
            continue
        payload = load(run_dir, filename)
        rows.append(
            {
                "case": name,
                "exit_code": payload["exit_code"],
                "peak_host_gib": payload["peak_process_tree_rss_bytes"] / 2**30,
                "peak_gpu_gib": payload["peak_gpu_memory_bytes"] / 2**30,
                "monitored_process_elapsed_s": payload["elapsed_s"],
            }
        )
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, default=Path("output/runs"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = {
        "pfr_first_step": summarize_pfr_first_step(args.run_dir),
        "pfr_closed_loop": summarize_pfr_closed_loop(args.run_dir),
        "pfr_scale": summarize_pfr_scale(args.run_dir),
        "darcy_prospective": summarize_darcy(args.run_dir),
        "darcy_replay": summarize_darcy_replay(args.run_dir),
        "mnist_direct_prior": summarize_mnist(args.run_dir),
        "mnist_multistart": summarize_mnist_multistart(args.run_dir),
        "resource_peaks": summarize_memory(args.run_dir),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
