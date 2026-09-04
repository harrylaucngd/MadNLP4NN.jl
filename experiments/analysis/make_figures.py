#!/usr/bin/env python3
"""Generate paper figures from validated JSON artifacts only."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


COLORS = {
    "cpu": "#3B6FB6",
    "staged": "#E18D2F",
    "dlpack": "#2A9D6F",
    "lbfgs": "#8E5AA9",
    "ipopt": "#555555",
}


def load(run_dir, filename):
    return json.loads((run_dir / filename).read_text())


def save(fig, output_dir, stem):
    fig.tight_layout()
    for suffix in ("pdf", "png"):
        fig.savefig(output_dir / f"{stem}.{suffix}", dpi=220, bbox_inches="tight")
    plt.close(fig)


def controlled_figure(run_dir, output_dir):
    exact = load(run_dir, "madnlp_paths_exact_pinned_r10.json")
    lbfgs = load(run_dir, "madnlp_paths_quick_lbfgs_cpu.json")
    labels = {
        "gpu_ad_cpu_kkt": ("Exact: CPU KKT", COLORS["cpu"], "o"),
        "host_staged_gpu_kkt": ("Exact: staged GPU KKT", COLORS["staged"], "s"),
        "dlpack_gpu_kkt": ("Exact: DLPack GPU KKT", COLORS["dlpack"], "^"),
    }
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    for path, (label, color, marker) in labels.items():
        rows = sorted(
            (entry for entry in exact["results"] if entry["path"] == path),
            key=lambda entry: entry["config"]["n"],
        )
        ax.plot(
            [entry["config"]["n"] for entry in rows],
            [1000 * entry["summary"]["median_elapsed_s"] for entry in rows],
            marker=marker,
            color=color,
            label=label,
        )
    rows = sorted(lbfgs["results"], key=lambda entry: entry["config"]["n"])
    ax.plot(
        [entry["config"]["n"] for entry in rows],
        [1000 * entry["summary"]["median_elapsed_s"] for entry in rows],
        marker="D",
        color=COLORS["lbfgs"],
        label="L-BFGS: CPU KKT",
    )
    ax.set(xlabel="Decision dimension $n$", ylabel="Warm solve time (ms)")
    ax.set_yscale("log")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    save(fig, output_dir, "controlled_regime")


def pfr_figure(run_dir, output_dir):
    dense = load(run_dir, "pfr_nmpc_first_step_paths_r3.json")
    sparse = load(run_dir, "pfr_nmpc_first_step_structured_paths_r5_v2.json")
    methods = ["exact_cpu", "exact_dlpack_gpu", "lbfgs_cpu"]
    label = {
        "exact_cpu": "Exact CPU",
        "exact_dlpack_gpu": "Exact DLPack GPU",
        "lbfgs_cpu": "L-BFGS CPU",
    }
    color = {
        "exact_cpu": COLORS["cpu"],
        "exact_dlpack_gpu": COLORS["dlpack"],
        "lbfgs_cpu": COLORS["lbfgs"],
    }
    dense_values = {
        entry["method"]: entry["median_elapsed_s"] for entry in dense["results"]
    }
    sparse_values = {
        entry["method"]: entry["median_elapsed_s"] for entry in sparse["results"]
    }
    fig, axes = plt.subplots(1, 3, figsize=(12.8, 3.4))
    x = np.arange(len(methods))
    width = 0.35
    axes[0].bar(
        x - width / 2,
        [1000 * dense_values[method] for method in methods],
        width,
        color=[color[method] for method in methods],
        alpha=0.45,
        label="Dense declaration",
    )
    axes[0].bar(
        x + width / 2,
        [1000 * sparse_values[method] for method in methods],
        width,
        color=[color[method] for method in methods],
        label="Correct sparsity",
    )
    axes[0].set_xticks(x, [label[method] for method in methods], rotation=18)
    axes[0].set_ylabel("First-step warm time (ms)")
    axes[0].set_yscale("log")
    axes[0].set_title("Structure changes the ranking")
    axes[0].legend(fontsize=8)
    axes[0].grid(axis="y", alpha=0.25)

    cpu = load(run_dir, "pfr_nmpc_closed_loop_structured_exact_cpu.json")
    gpu = load(run_dir, "pfr_nmpc_closed_loop_structured_reuse_gpu.json")
    values = [
        [1000 * row["elapsed_s"] for row in cpu["records"][1:]],
        [1000 * row["elapsed_s"] for row in gpu["records"][1:]],
    ]
    axes[1].boxplot(values, tick_labels=["CPU exact", "Persistent GPU"], showfliers=True)
    axes[1].set_yscale("log")
    axes[1].set_ylabel("Closed-loop solve time (ms)")
    axes[1].set_title("GPU median vs. fallback tail")
    axes[1].grid(axis="y", alpha=0.25)

    case2 = load(run_dir, "pfr_case2_first_step_exact_tol1e8.json")
    case3_cpu = load(run_dir, "pfr_case3_first_step_exact_cpu.json")
    case3_gpu = load(run_dir, "pfr_case3_first_step_exact_dlpack.json")

    def method_time(payload, method):
        entry = next(row for row in payload["results"] if row["method"] == method)
        return entry["median_elapsed_s"]

    case_labels = ["1\n(400 eq.)", "2\n(800 eq.)", "3\n(7,500 eq.)"]
    cpu_times = [
        sparse_values["exact_cpu"],
        method_time(case2, "exact_cpu"),
        method_time(case3_cpu, "exact_cpu"),
    ]
    gpu_times = [
        sparse_values["exact_dlpack_gpu"],
        method_time(case2, "exact_dlpack_gpu"),
        method_time(case3_gpu, "exact_dlpack_gpu"),
    ]
    x = np.arange(len(case_labels))
    axes[2].bar(
        x - width / 2,
        cpu_times,
        width,
        color=COLORS["cpu"],
        label="Exact CPU KKT",
    )
    axes[2].bar(
        x + width / 2,
        gpu_times,
        width,
        color=COLORS["dlpack"],
        label="Exact DLPack GPU",
    )
    axes[2].set_xticks(x, case_labels)
    axes[2].set_yscale("log")
    axes[2].set_ylabel("First-step warm time (s)")
    axes[2].set_title("Published cases cross regimes")
    axes[2].legend(fontsize=8)
    axes[2].grid(axis="y", alpha=0.25)
    save(fig, output_dir, "pfr_structure_sequence")


def darcy_figure(run_dir, output_dir):
    data = load(run_dir, "neuraloperator_darcy64_prospective12_lbfgs.json")
    grouped = defaultdict(list)
    failures = defaultdict(int)
    for entry in data["results"]:
        record = entry["warm"][0]
        resolution = entry["latent_resolution"]
        success = record["status"] == "SOLVE_SUCCEEDED" and record["kkt_error"] <= 1e-6
        if success:
            grouped[resolution].append(record)
        else:
            failures[resolution] += 1
    resolutions = [16, 32]
    coefficient = [
        np.median([record["metrics"]["coefficient_relative_l2"] for record in grouped[r]])
        for r in resolutions
    ]
    floors = [
        np.median(
            [
                record["instance_baselines"]["coefficient_relative_l2_floor"]
                for record in grouped[r]
            ]
        )
        for r in resolutions
    ]
    residual_ratios = [
        np.median(
            [
                record["metrics"]["surrogate_target_relative_l2"]
                / record["instance_baselines"][
                    "true_coefficient_surrogate_target_relative_l2"
                ]
                for record in grouped[r]
            ]
        )
        for r in resolutions
    ]
    fig, axes = plt.subplots(1, 3, figsize=(12.8, 3.4))
    axes[0].plot(resolutions, coefficient, "o-", label="Recovered coefficient")
    axes[0].plot(resolutions, floors, "s--", label="Best representable field")
    axes[0].set(
        xlabel="Latent side length",
        ylabel="Median coefficient relative error",
        xticks=resolutions,
        title="Capacity widens the recovery gap",
    )
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.25)
    axes[1].bar(
        [str(value) for value in resolutions],
        residual_ratios,
        color=[COLORS["cpu"], COLORS["dlpack"]],
    )
    axes[1].axhline(1.0, color="black", linestyle="--", linewidth=1)
    axes[1].set(
        xlabel="Latent side length",
        ylabel="Optimized / true-coefficient FNO residual",
        title="Optimization out-fits the true coefficient",
    )
    axes[1].grid(axis="y", alpha=0.25)
    for index, resolution in enumerate(resolutions):
        axes[1].text(
            index,
            residual_ratios[index] + 0.03,
            f"{60 - failures[resolution]}/60 KKT",
            ha="center",
            fontsize=8,
        )

    replay = load(run_dir, "darcy_fd_replay64_multistart_curvature.json")
    replay_styles = {
        "exact_dlpack_gpu": ("Exact", COLORS["dlpack"]),
        "gauss_newton_dlpack_gpu": ("Gauss--Newton", COLORS["staged"]),
        "lbfgs_cpu": ("L-BFGS", COLORS["lbfgs"]),
    }
    for method, (label, color) in replay_styles.items():
        for resolution, marker in ((16, "o"), (32, "^")):
            records = [
                entry["warm"][0]
                for entry in replay["results"]
                if entry["method"] == method
                and entry["latent_resolution"] == resolution
                and entry["warm"][0]["status"] == "SOLVE_SUCCEEDED"
            ]
            axes[2].scatter(
                [row["metrics"]["surrogate_target_relative_l2"] for row in records],
                [row["metrics"]["simulator_target_relative_l2"] for row in records],
                color=color,
                marker=marker,
                alpha=0.65,
                s=22,
                label=f"{label}, {resolution}$^2$",
            )
    axes[2].set(
        xlabel="Surrogate relative residual",
        ylabel="Independent PDE relative residual",
        title="Surrogate rankings rarely transfer",
    )
    axes[2].grid(alpha=0.25)
    axes[2].legend(fontsize=6.5, ncols=2)
    axes[2].text(
        0.98,
        0.96,
        "same winner: 2/30",
        transform=axes[2].transAxes,
        ha="right",
        va="top",
        fontsize=8,
    )
    save(fig, output_dir, "darcy_validity")


def mnist_figure(run_dir, output_dir):
    sources = [
        load(run_dir, "mnist_prior_128_paths_tol1e7.json"),
        load(run_dir, "mnist_prior_large_paths_r3.json"),
        load(run_dir, "mnist_prior_large_exact_cpu_r1.json"),
    ]
    rows = []
    for payload in sources:
        for entry in payload["results"]:
            successful = [
                row
                for row in entry["warm"]
                if row["status"] == "SOLVE_SUCCEEDED"
                and row["kkt_error"] <= 1e-7
                and row["max_raw_constraint_violation"] <= 1e-6
            ]
            rows.append(
                (
                    entry["parameter_count"],
                    entry["method"],
                    np.median([row["elapsed_s"] for row in successful]),
                    successful[0]["objective"],
                )
            )
    styles = {
        "exact_cpu": ("Exact CPU KKT", COLORS["cpu"], "o"),
        "exact_dlpack_gpu": ("Exact DLPack GPU", COLORS["dlpack"], "^"),
        "lbfgs_cpu": ("L-BFGS CPU KKT", COLORS["lbfgs"], "D"),
    }
    fig, axes = plt.subplots(1, 3, figsize=(12.8, 3.4))
    for method, (label, color, marker) in styles.items():
        selected = sorted(row for row in rows if row[1] == method)
        axes[0].plot(
            [row[0] for row in selected],
            [row[2] for row in selected],
            marker=marker,
            color=color,
            label=label,
        )
        axes[1].plot(
            [row[0] for row in selected],
            [row[3] for row in selected],
            marker=marker,
            color=color,
            label=label,
        )
    axes[0].set_xscale("log")
    axes[0].set_yscale("log")
    axes[0].set(xlabel="Network parameters", ylabel="Warm solve time (s)")
    axes[0].grid(alpha=0.25)
    axes[0].legend(fontsize=8)
    axes[1].set_xscale("log")
    axes[1].set(xlabel="Network parameters", ylabel="Local L1 objective")
    axes[1].grid(alpha=0.25)
    axes[1].set_title("Feasible methods reach different KKT points")

    multistart = load(run_dir, "mnist_prior_1024_multistart5.json")
    distributions = []
    distribution_labels = []
    box_colors = []
    for method in ("exact_cpu", "exact_dlpack_gpu", "lbfgs_cpu"):
        values = [
            entry["warm"][0]["objective"]
            for entry in multistart["results"]
            if entry["method"] == method
            and entry["warm"][0]["status"] == "SOLVE_SUCCEEDED"
            and entry["warm"][0]["max_raw_constraint_violation"] <= 1e-6
        ]
        distributions.append(values)
        distribution_labels.append(styles[method][0].replace(" KKT", ""))
        box_colors.append(styles[method][1])
    boxes = axes[2].boxplot(
        distributions,
        tick_labels=distribution_labels,
        patch_artist=True,
        showmeans=True,
    )
    for patch, color in zip(boxes["boxes"], box_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.55)
    axes[2].tick_params(axis="x", labelrotation=18)
    axes[2].set_ylabel("Local L1 objective")
    axes[2].set_title("Five shared starts at 5.01M parameters")
    axes[2].grid(axis="y", alpha=0.25)
    save(fig, output_dir, "mnist_scaling_quality")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, default=Path("output/runs"))
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.facecolor": "white",
        }
    )
    controlled_figure(args.run_dir, args.output_dir)
    pfr_figure(args.run_dir, args.output_dir)
    darcy_figure(args.run_dir, args.output_dir)
    mnist_figure(args.run_dir, args.output_dir)


if __name__ == "__main__":
    main()
