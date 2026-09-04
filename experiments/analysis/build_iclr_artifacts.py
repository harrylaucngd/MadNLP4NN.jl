#!/usr/bin/env python3
"""Generate ICLR paper macros and compact figures from registered JSON runs."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import wilcoxon


COLORS = {
    "cpu": "#3B6FB6",
    "staged": "#E18D2F",
    "dlpack": "#2A9D6F",
    "lbfgs": "#8E5AA9",
    "ipopt": "#555555",
}


def load(run_dir, name):
    return json.loads((run_dir / name).read_text())


def q(values, probability):
    return float(np.quantile(np.asarray(values, dtype=float), probability))


def fmt(value, digits=2):
    return f"{value:.{digits}f}"


def common_success(record, tolerance=1e-6):
    return (
        record.get("status") in ("SOLVE_SUCCEEDED", "SOLVED_TO_ACCEPTABLE_LEVEL")
        and record.get("kkt_error", np.inf) <= tolerance
        and record.get("raw_constraint_inf", 0.0) <= 1e-6
        and record.get("max_raw_constraint_violation", 0.0) <= 1e-6
    )


def replay_rows(payload):
    rows = []
    for entry in payload["results"]:
        if not entry["warm"]:
            continue
        record = entry["warm"][0]
        metrics = record["metrics"]
        baseline = record["instance_baselines"]
        rows.append(
            {
                "sample": entry["sample"],
                "resolution": entry["latent_resolution"],
                "start": entry["start_id"],
                "method": entry["method"],
                "success": common_success(record),
                "time": record["elapsed_s"],
                "iterations": record["iterations"],
                "fno": metrics["surrogate_target_relative_l2"],
                "pde": metrics["simulator_target_relative_l2"],
                "coefficient": metrics["coefficient_relative_l2"],
                "floor": baseline["coefficient_relative_l2_floor"],
                "truth_fno": baseline[
                    "true_coefficient_surrogate_target_relative_l2"
                ],
            }
        )
    return rows


def summarize_replay(rows):
    summary = {}
    for method in sorted({row["method"] for row in rows}):
        for resolution in (16, 32):
            selected = [
                row
                for row in rows
                if row["method"] == method
                and row["resolution"] == resolution
                and row["success"]
            ]
            summary[(method, resolution)] = {
                "successes": len(selected),
                "attempts": sum(
                    row["method"] == method and row["resolution"] == resolution
                    for row in rows
                ),
                **{
                    f"{name}_{suffix}": q([row[name] for row in selected], probability)
                    for name in ("time", "iterations", "fno", "pde", "coefficient")
                    for suffix, probability in (("q1", 0.25), ("median", 0.5), ("q3", 0.75))
                },
                "gap_median": q(
                    [row["coefficient"] - row["floor"] for row in selected], 0.5
                ),
                "outfits_truth": sum(row["fno"] < row["truth_fno"] for row in selected),
            }
    grouped = defaultdict(list)
    for row in rows:
        if row["success"]:
            grouped[(row["sample"], row["resolution"], row["start"])].append(row)
    complete = [group for group in grouped.values() if len(group) == 3]
    agreement = sum(
        min(group, key=lambda row: row["fno"])["method"]
        == min(group, key=lambda row: row["pde"])["method"]
        for group in complete
    )
    return summary, complete, agreement


def paired_instance_effect(
    rows,
    *,
    resolution,
    first_method,
    second_method,
    metric,
    seed=20260828,
    bootstrap_samples=200_000,
):
    """Paired median effect using physical instances as the sampling unit."""
    samples = sorted({row["sample"] for row in rows})
    values = {}
    for method in (first_method, second_method):
        values[method] = np.asarray(
            [
                np.median(
                    [
                        row[metric]
                        for row in rows
                        if row["sample"] == sample
                        and row["resolution"] == resolution
                        and row["method"] == method
                        and row["success"]
                    ]
                )
                for sample in samples
            ]
        )
    differences = values[first_method] - values[second_method]
    rng = np.random.default_rng(seed)
    indices = rng.integers(
        0, len(differences), size=(bootstrap_samples, len(differences))
    )
    bootstrap = np.median(differences[indices], axis=1)
    _statistic, pvalue = wilcoxon(differences, alternative="two-sided")
    return {
        "instances": len(differences),
        "median": float(np.median(differences)),
        "ci_low": float(np.quantile(bootstrap, 0.025)),
        "ci_high": float(np.quantile(bootstrap, 0.975)),
        "negative_count": int(np.sum(differences < 0)),
        "positive_count": int(np.sum(differences > 0)),
        "wilcoxon_p": float(pvalue),
    }


def write_macros(run_dir, output):
    pfr = load(run_dir, "pfr_case3_first_step_exact_r5.json")
    pfr_by_method = {entry["method"]: entry for entry in pfr["results"]}
    gpu_times = [row["elapsed_s"] for row in pfr_by_method["exact_dlpack_gpu"]["warm"]]
    cpu_times = [row["elapsed_s"] for row in pfr_by_method["exact_cpu"]["warm"]]
    ipopt = load(run_dir, "pfr_case3_cyipopt_exact_r3.json")
    ipopt_times = [row["elapsed_s"] for row in ipopt["records"] if not row["warmup"]]
    replay = replay_rows(load(run_dir, "darcy_fd_replay64_prospective12_curvature.json"))
    replay_summary, groups, agreement = summarize_replay(replay)
    gn = replay_summary[("gauss_newton_dlpack_gpu", 32)]
    lbfgs = replay_summary[("lbfgs_cpu", 32)]
    exact = replay_summary[("exact_dlpack_gpu", 32)]
    paired_fno = paired_instance_effect(
        replay,
        resolution=32,
        first_method="gauss_newton_dlpack_gpu",
        second_method="lbfgs_cpu",
        metric="fno",
    )
    paired_pde = paired_instance_effect(
        replay,
        resolution=32,
        first_method="gauss_newton_dlpack_gpu",
        second_method="lbfgs_cpu",
        metric="pde",
    )
    groups32 = [group for group in groups if group[0]["resolution"] == 32]
    fno_winners32 = {
        method: sum(min(group, key=lambda row: row["fno"])["method"] == method for group in groups32)
        for method in ("exact_dlpack_gpu", "gauss_newton_dlpack_gpu", "lbfgs_cpu")
    }
    pde_winners32 = {
        method: sum(min(group, key=lambda row: row["pde"])["method"] == method for group in groups32)
        for method in ("exact_dlpack_gpu", "gauss_newton_dlpack_gpu", "lbfgs_cpu")
    }
    macros = {
        "PFRCaseOneSparseCPU": "43.3",
        "PFRCaseOneDenseCPU": "0.59",
        "PFRCaseThreeGPU": fmt(q(gpu_times, 0.5), 2),
        "PFRCaseThreeGPULo": fmt(q(gpu_times, 0.25), 2),
        "PFRCaseThreeGPUHi": fmt(q(gpu_times, 0.75), 2),
        "PFRCaseThreeCPU": fmt(q(cpu_times, 0.5), 2),
        "PFRCaseThreeCPULo": fmt(q(cpu_times, 0.25), 2),
        "PFRCaseThreeCPUHi": fmt(q(cpu_times, 0.75), 2),
        "PFRCaseThreeSpeedup": fmt(q(cpu_times, 0.5) / q(gpu_times, 0.5), 1),
        "PFRCaseThreeIpopt": fmt(q(ipopt_times, 0.5), 2),
        "PFRCaseThreeLBFGS": "530.36",
        "DarcyRouteBase": "115",
        "DarcyRouteTotal": "120",
        "DarcyRouteOverhead": "7.7",
        "ReplayGroups": str(len(groups)),
        "ReplayAgreement": str(agreement),
        "ReplaySolves": str(sum(row["success"] for row in replay)),
        "ReplayOutfitsTruth": str(sum(row["fno"] < row["truth_fno"] for row in replay)),
        "ReplayGNFNOWinsThirtyTwo": str(fno_winners32["gauss_newton_dlpack_gpu"]),
        "ReplayExactPDEWinsThirtyTwo": str(pde_winners32["exact_dlpack_gpu"]),
        "ReplayGNPDEWinsThirtyTwo": str(pde_winners32["gauss_newton_dlpack_gpu"]),
        "ReplayLBFGSPDEWinsThirtyTwo": str(pde_winners32["lbfgs_cpu"]),
        "ReplayPairedFNOReduction": fmt(-100 * paired_fno["median"], 3),
        "ReplayPairedFNOReductionLow": fmt(-100 * paired_fno["ci_high"], 3),
        "ReplayPairedFNOReductionHigh": fmt(-100 * paired_fno["ci_low"], 3),
        "ReplayPairedFNOPValue": fmt(paired_fno["wilcoxon_p"], 5),
        "ReplayPairedPDEDelta": fmt(100 * paired_pde["median"], 2),
        "ReplayPairedPDEDeltaLow": fmt(100 * paired_pde["ci_low"], 2),
        "ReplayPairedPDEDeltaHigh": fmt(100 * paired_pde["ci_high"], 2),
        "ReplayPairedPDEPValue": fmt(paired_pde["wilcoxon_p"], 3),
        "ReplayGNTime": fmt(gn["time_median"], 3),
        "ReplayGNSurrogate": fmt(100 * gn["fno_median"], 2),
        "ReplayGNPDE": fmt(100 * gn["pde_median"], 2),
        "ReplayLBFGSTime": fmt(lbfgs["time_median"], 2),
        "ReplayLBFGSSurrogate": fmt(100 * lbfgs["fno_median"], 2),
        "ReplayLBFGSPDE": fmt(100 * lbfgs["pde_median"], 2),
        "ReplayExactTime": fmt(exact["time_median"], 2),
        "ReplayExactSurrogate": fmt(100 * exact["fno_median"], 2),
        "ReplayExactPDE": fmt(100 * exact["pde_median"], 2),
        "MNISTStarts": "5",
        "MNISTSolved": "15",
        "MNISTTotal": "15",
        "MNISTExactMedianObjective": "4.736",
        "MNISTLBFGSMedianObjective": "5.590",
        "DarcyExactMemory": "33.54",
        "DarcyLBFGSMemory": "0.99",
    }
    text = "% Auto-generated by experiments/analysis/build_iclr_artifacts.py.\n"
    text += "".join(f"\\newcommand{{\\{key}}}{{{value}}}\n" for key, value in macros.items())
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text)
    table_path = output.parent / "replay_prospective_table.tex"
    method_labels = {
        "exact_dlpack_gpu": "Exact DLPack",
        "gauss_newton_dlpack_gpu": "Gauss--Newton",
        "lbfgs_cpu": "L-BFGS CPU",
    }
    table_lines = [
        "% Auto-generated prospective replay table.",
        "\\begin{tabular}{llrrrr}",
        "\\toprule",
        "Latent & Curvature & time (s) & FNO res. & PDE res. & coeff. err.\\\\",
        "\\midrule",
    ]
    for resolution in (16, 32):
        for method in ("exact_dlpack_gpu", "gauss_newton_dlpack_gpu", "lbfgs_cpu"):
            item = replay_summary[(method, resolution)]
            table_lines.append(
                f"${resolution}^2$ & {method_labels[method]} & "
                f"{item['time_median']:.3f} & {100 * item['fno_median']:.2f}\\% & "
                f"{100 * item['pde_median']:.2f}\\% & {100 * item['coefficient_median']:.1f}\\%\\\\"
            )
        if resolution == 16:
            table_lines.append("\\midrule")
    table_lines.extend(("\\bottomrule", "\\end{tabular}"))
    table_path.write_text("\n".join(table_lines) + "\n")
    paired_path = output.parent / "replay_paired_effects.tex"
    paired_path.write_text(
        "% Auto-generated paired effects; unit is one physical instance.\n"
        "\\begin{tabular}{lrrr}\n"
        "\\toprule\n"
        "Metric (GN $-$ L-BFGS) & median (pp) & bootstrap 95\\% CI & Wilcoxon $p$\\\\\n"
        "\\midrule\n"
        f"FNO residual & {100 * paired_fno['median']:.3f} & "
        f"[{100 * paired_fno['ci_low']:.3f}, {100 * paired_fno['ci_high']:.3f}] & "
        f"{paired_fno['wilcoxon_p']:.5f}\\\\\n"
        f"PDE residual & {100 * paired_pde['median']:.2f} & "
        f"[{100 * paired_pde['ci_low']:.2f}, {100 * paired_pde['ci_high']:.2f}] & "
        f"{paired_pde['wilcoxon_p']:.3f}\\\\\n"
        "\\bottomrule\n"
        "\\end{tabular}\n"
    )
    return replay, replay_summary, groups, agreement


def save(fig, output_dir, stem):
    fig.tight_layout()
    for suffix in ("pdf", "png"):
        fig.savefig(output_dir / f"{stem}.{suffix}", dpi=240, bbox_inches="tight")
    plt.close(fig)


def darcy_figure(run_dir, output_dir, replay_rows_, agreement, groups):
    public = load(run_dir, "neuraloperator_darcy64_prospective12_lbfgs.json")
    grouped = defaultdict(list)
    for entry in public["results"]:
        record = entry["warm"][0]
        if common_success(record):
            grouped[entry["latent_resolution"]].append(record)
    resolutions = [16, 32]
    coefficient = [
        q([row["metrics"]["coefficient_relative_l2"] for row in grouped[r]], 0.5)
        for r in resolutions
    ]
    floors = [
        q(
            [row["instance_baselines"]["coefficient_relative_l2_floor"] for row in grouped[r]],
            0.5,
        )
        for r in resolutions
    ]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.75))
    axes[0].plot(resolutions, coefficient, "o-", label="Recovered coefficient")
    axes[0].plot(resolutions, floors, "s--", label="Best latent representation")
    axes[0].set(
        xlabel="Latent side length",
        ylabel="Median relative error",
        xticks=resolutions,
        title="More capacity widens the gap",
    )
    axes[0].legend(fontsize=7)
    axes[0].grid(alpha=0.25)

    styles = {
        "exact_dlpack_gpu": ("Exact", COLORS["dlpack"]),
        "gauss_newton_dlpack_gpu": ("Gauss--Newton", COLORS["staged"]),
        "lbfgs_cpu": ("L-BFGS", COLORS["lbfgs"]),
    }
    for method, (label, color) in styles.items():
        for resolution, marker in ((16, "o"), (32, "^")):
            rows = [
                row
                for row in replay_rows_
                if row["method"] == method
                and row["resolution"] == resolution
                and row["success"]
            ]
            axes[1].scatter(
                [row["fno"] for row in rows],
                [row["pde"] for row in rows],
                s=10,
                alpha=0.28,
                color=color,
                marker=marker,
            )
            axes[1].scatter(
                [q([row["fno"] for row in rows], 0.5)],
                [q([row["pde"] for row in rows], 0.5)],
                s=55,
                edgecolor="black",
                linewidth=0.5,
                color=color,
                marker=marker,
                label=f"{label}, {resolution}$^2$",
            )
    axes[1].set(
        xlabel="FNO relative residual",
        ylabel="PDE relative residual",
        title="Surrogate rankings rarely transfer",
    )
    axes[1].grid(alpha=0.25)
    axes[1].legend(fontsize=5.8, ncols=2)
    save(fig, output_dir, "darcy_iclr")


def pfr_figure(run_dir, output_dir):
    dense = load(run_dir, "pfr_nmpc_first_step_paths_r3.json")
    sparse = load(run_dir, "pfr_nmpc_first_step_structured_paths_r5_v2.json")
    case2 = load(run_dir, "pfr_case2_first_step_exact_r5.json")
    case3 = load(run_dir, "pfr_case3_first_step_exact_r5.json")
    dense_values = {row["method"]: row["median_elapsed_s"] for row in dense["results"]}
    sparse_values = {row["method"]: row["median_elapsed_s"] for row in sparse["results"]}

    def method_time(payload, method):
        return next(row["median_elapsed_s"] for row in payload["results"] if row["method"] == method)

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.65))
    methods = ["exact_cpu", "exact_dlpack_gpu", "lbfgs_cpu"]
    labels = ["Exact CPU", "Exact GPU", "L-BFGS"]
    colors = [COLORS["cpu"], COLORS["dlpack"], COLORS["lbfgs"]]
    x = np.arange(3)
    width = 0.35
    axes[0].bar(
        x - width / 2,
        [dense_values[m] for m in methods],
        width,
        alpha=0.45,
        color=colors,
        label="Dense declaration",
    )
    axes[0].bar(
        x + width / 2,
        [sparse_values[m] for m in methods],
        width,
        color=colors,
        label="True sparsity",
    )
    axes[0].set_xticks(x, labels, rotation=15)
    axes[0].set_yscale("log")
    axes[0].set_ylabel("Warm time (s)")
    axes[0].set_title("Structure reverses case-1 ranking")
    axes[0].legend(fontsize=7)
    axes[0].grid(axis="y", alpha=0.25)

    case_labels = ["1\n400 eq.", "2\n800 eq.", "3\n7,500 eq."]
    cpu = [sparse_values["exact_cpu"], method_time(case2, "exact_cpu"), method_time(case3, "exact_cpu")]
    gpu = [sparse_values["exact_dlpack_gpu"], method_time(case2, "exact_dlpack_gpu"), method_time(case3, "exact_dlpack_gpu")]
    axes[1].bar(x - width / 2, cpu, width, color=COLORS["cpu"], label="CPU KKT")
    axes[1].bar(x + width / 2, gpu, width, color=COLORS["dlpack"], label="GPU KKT")
    axes[1].set_xticks(x, case_labels)
    axes[1].set_yscale("log")
    axes[1].set_ylabel("Warm time (s)")
    axes[1].set_title("Published cases cross regimes")
    axes[1].legend(fontsize=7)
    axes[1].grid(axis="y", alpha=0.25)
    save(fig, output_dir, "pfr_iclr")


def mnist_figure(run_dir, output_dir):
    sources = [
        load(run_dir, "mnist_prior_128_paths_tol1e7.json"),
        load(run_dir, "mnist_prior_large_paths_r3.json"),
        load(run_dir, "mnist_prior_large_exact_cpu_r1.json"),
    ]
    rows = []
    for payload in sources:
        for entry in payload["results"]:
            good = [row for row in entry["warm"] if common_success(row, 1e-7)]
            rows.append((entry["parameter_count"], entry["method"], q([row["elapsed_s"] for row in good], 0.5)))
    styles = {
        "exact_cpu": ("Exact CPU", COLORS["cpu"], "o"),
        "exact_dlpack_gpu": ("Exact GPU", COLORS["dlpack"], "^"),
        "lbfgs_cpu": ("L-BFGS", COLORS["lbfgs"], "D"),
    }
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.65))
    for method, (label, color, marker) in styles.items():
        selected = sorted(row for row in rows if row[1] == method)
        axes[0].plot([row[0] for row in selected], [row[2] for row in selected], marker=marker, color=color, label=label)
    axes[0].set_xscale("log")
    axes[0].set_yscale("log")
    axes[0].set(xlabel="Network parameters", ylabel="Warm time (s)", title="Parameter-count scaling")
    axes[0].legend(fontsize=7)
    axes[0].grid(alpha=0.25)

    multistart = load(run_dir, "mnist_prior_1024_multistart5.json")
    distributions = []
    labels = []
    colors = []
    for method in styles:
        values = [
            entry["warm"][0]["objective"]
            for entry in multistart["results"]
            if entry["method"] == method and common_success(entry["warm"][0], 1e-7)
        ]
        distributions.append(values)
        labels.append(styles[method][0])
        colors.append(styles[method][1])
    boxes = axes[1].boxplot(distributions, tick_labels=labels, patch_artist=True, showmeans=True)
    for box, color in zip(boxes["boxes"], colors):
        box.set_facecolor(color)
        box.set_alpha(0.55)
    axes[1].tick_params(axis="x", labelrotation=15)
    axes[1].set(
        ylabel=r"Local $\ell_1$ objective",
        title="Five starts, 5.01M parameters",
    )
    axes[1].grid(axis="y", alpha=0.25)
    save(fig, output_dir, "mnist_iclr")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, default=Path("output/runs"))
    parser.add_argument("--paper-dir", type=Path, default=Path("paper"))
    args = parser.parse_args()
    output_dir = args.paper_dir / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.size": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.facecolor": "white",
        }
    )
    rows, _summary, groups, agreement = write_macros(
        args.run_dir, args.paper_dir / "generated" / "result_macros.tex"
    )
    darcy_figure(args.run_dir, output_dir, rows, agreement, groups)
    pfr_figure(args.run_dir, output_dir)
    mnist_figure(args.run_dir, output_dir)
    print(
        json.dumps(
            {
                "replay_successes": sum(row["success"] for row in rows),
                "replay_groups": len(groups),
                "ranking_agreement": agreement,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
