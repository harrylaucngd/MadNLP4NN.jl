#!/usr/bin/env python3
"""SLSQP baseline for the exact PDEBench latent inversion problem."""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np
from scipy.optimize import minimize


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--max-iter", type=int, default=300)
    parser.add_argument("--ftol", type=float, default=1e-9)
    return parser.parse_args()


def truth_metrics(evaluator, solution, coefficient, pressure):
    field = evaluator.evaluate_field(solution)
    prediction = evaluator.evaluate_nn(solution).reshape(pressure.shape)
    constraints = evaluator.evaluate_cons(solution)
    return {
        "coefficient_relative_l2": float(
            np.linalg.norm(field - coefficient) / np.linalg.norm(coefficient)
        ),
        "coefficient_rmse": float(np.sqrt(np.mean((field - coefficient) ** 2))),
        "binary_accuracy": float(np.mean((field >= 0.55) == (coefficient >= 0.55))),
        "binaryness": float(np.mean(((field - 0.1) * (field - 1.0)) ** 2)),
        "pressure_relative_l2": float(
            np.linalg.norm(prediction - pressure) / np.linalg.norm(pressure)
        ),
        "pressure_rmse": float(np.sqrt(np.mean((prediction - pressure) ** 2))),
        "field_mean": float(np.mean(field)),
        "field_min": float(np.min(field)),
        "field_max": float(np.max(field)),
        "max_constraint_violation": float(
            max(0.0, np.max(constraints)) if constraints.size else 0.0
        ),
    }


def residual_audit(evaluator, result, lower=0.1, upper=1.0):
    x = result.x
    constraints = evaluator.evaluate_cons(x)
    jacobian = evaluator.evaluate_jac(x)
    gradient = evaluator.evaluate_grad(x)
    multipliers = np.asarray(getattr(result, "multipliers", np.zeros(len(constraints))))
    if multipliers.size >= len(constraints):
        multipliers = multipliers[-len(constraints):]
    else:
        multipliers = np.zeros(len(constraints))
    lagrangian_gradient = gradient + jacobian.T @ multipliers
    projected = lagrangian_gradient.copy()
    bound_tolerance = 1e-7
    projected[(x <= lower + bound_tolerance) & (projected > 0)] = 0.0
    projected[(x >= upper - bound_tolerance) & (projected < 0)] = 0.0
    primal = max(
        float(max(0.0, np.max(constraints))) if constraints.size else 0.0,
        float(max(0.0, lower - np.min(x))),
        float(max(0.0, np.max(x) - upper)),
    )
    return {
        "primal_feas": primal,
        "projected_stationarity": float(np.linalg.norm(projected, ord=np.inf)),
        "nonlinear_complementarity": float(
            np.max(np.abs(multipliers * constraints)) if constraints.size else 0.0
        ),
        "multipliers": multipliers.tolist(),
    }


def main():
    args = parse_args()
    project_python = Path(__file__).resolve().parents[2] / "python"
    sys.path.insert(0, str(project_python))
    from jax_nn_evaluator import create_pdebench_darcy_evaluator

    config = json.loads(args.config.read_text())
    records = []
    with h5py.File(config["data"], "r") as handle:
        for sample in config["samples"]:
            coefficient = np.asarray(handle["nu"][sample], dtype=np.float64)
            pressure = np.asarray(handle["tensor"][sample, 0], dtype=np.float64)
            for latent_resolution in config["latent_resolutions"]:
                x0 = np.full(latent_resolution**2, 0.541621900172992)
                evaluator = create_pdebench_darcy_evaluator(
                    config["model"],
                    pressure.reshape(-1),
                    x0,
                    full_resolution=128,
                    regularization_weight=config["regularization_weight"],
                    binary_weight=config.get("binary_weight", 0.0),
                    mean_lower=config["mean_lower"],
                    mean_upper=config["mean_upper"],
                    device="gpu",
                )

                # Compile every callback outside the timed repetitions.
                evaluator.evaluate_obj(x0)
                evaluator.evaluate_grad(x0)
                evaluator.evaluate_cons(x0)
                evaluator.evaluate_jac(x0)

                def objective(x):
                    return evaluator.evaluate_obj(x)

                def gradient(x):
                    return evaluator.evaluate_grad(x)

                constraints = {
                    "type": "ineq",
                    "fun": lambda x: -evaluator.evaluate_cons(x),
                    "jac": lambda x: -evaluator.evaluate_jac(x),
                }
                for repetition in range(args.repetitions + 1):
                    start = time.perf_counter()
                    result = minimize(
                        objective,
                        x0,
                        method="SLSQP",
                        jac=gradient,
                        bounds=[(0.1, 1.0)] * x0.size,
                        constraints=constraints,
                        options={
                            "maxiter": args.max_iter,
                            "ftol": args.ftol,
                            "disp": False,
                        },
                    )
                    elapsed = time.perf_counter() - start
                    record = {
                        "sample": sample,
                        "latent_resolution": latent_resolution,
                        "n": x0.size,
                        "repetition": repetition,
                        "cold": repetition == 0,
                        "elapsed_s": elapsed,
                        "success": bool(result.success),
                        "status": int(result.status),
                        "message": str(result.message),
                        "objective": float(result.fun),
                        "iterations": int(result.nit),
                        "function_evaluations": int(result.nfev),
                        "gradient_evaluations": int(result.njev),
                        **residual_audit(evaluator, result),
                        "metrics": truth_metrics(
                            evaluator, result.x, coefficient, pressure
                        ),
                    }
                    records.append(record)
                    print(record, flush=True)

    payload = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config": config,
        "repetitions": args.repetitions,
        "max_iter": args.max_iter,
        "ftol": args.ftol,
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
