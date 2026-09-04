#!/usr/bin/env python3
"""Same-oracle cyipopt baseline for published PFR-NMPC cases 2 and 3."""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import cyipopt
import jax
import numpy as np


class CyIpoptPFRCNNProblem:
    def __init__(self, evaluator):
        self.evaluator = evaluator
        self.jac_rows, self.jac_cols = evaluator.jacobian_structure()
        self.hess_rows, self.hess_cols = evaluator.hessian_structure()

    def objective(self, decision):
        return self.evaluator.evaluate_obj(decision)

    def gradient(self, decision):
        return self.evaluator.evaluate_grad(decision)

    def constraints(self, decision):
        return self.evaluator.evaluate_cons(decision)

    def jacobian(self, decision):
        return self.evaluator.evaluate_jac_coord(decision)

    def jacobianstructure(self):
        return self.jac_rows, self.jac_cols

    def hessian(self, decision, multipliers, objective_weight):
        return self.evaluator.evaluate_hess_coord(
            decision, multipliers, objective_weight
        )

    def hessianstructure(self):
        return self.hess_rows, self.hess_cols


def residual_audit(evaluator, solution, info, lower, upper):
    constraints = evaluator.evaluate_cons(solution)
    gradient = evaluator.evaluate_grad(solution)
    multipliers = np.asarray(info["mult_g"])
    lower_multipliers = np.asarray(info["mult_x_L"])
    upper_multipliers = np.asarray(info["mult_x_U"])
    stationarity = (
        gradient
        + evaluator.evaluate_jtprod(solution, multipliers)
        - lower_multipliers
        + upper_multipliers
    )
    nonfixed = lower < upper
    lower_slack = solution - lower
    upper_slack = upper - solution
    return {
        "raw_constraint_inf": float(np.linalg.norm(constraints, ord=np.inf)),
        "primal_feas": float(
            max(
                np.linalg.norm(constraints, ord=np.inf),
                max(0.0, float(np.max(lower - solution))),
                max(0.0, float(np.max(solution - upper))),
            )
        ),
        "dual_feas": float(np.linalg.norm(stationarity[nonfixed], ord=np.inf)),
        "complementarity": float(
            max(
                np.max(np.abs(lower_multipliers[nonfixed] * lower_slack[nonfixed])),
                np.max(np.abs(upper_multipliers[nonfixed] * upper_slack[nonfixed])),
            )
        ),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repetitions", type=int, default=1)
    parser.add_argument("--max-iter", type=int, default=500)
    parser.add_argument("--tol", type=float, default=1e-8)
    parser.add_argument(
        "--hessian", choices=("exact", "limited-memory"), default="exact"
    )
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))
    from jax_nn_evaluator import (
        create_pfr_cnn_nmpc_evaluator,
        load_pfr_cnn_published_case,
    )

    case_data = load_pfr_cnn_published_case(
        config["published_output"], config["case"]
    )
    evaluator = create_pfr_cnn_nmpc_evaluator(
        config["model"],
        case_data["current_state"],
        case_data["previous_control"],
        config["tracking_channels"],
        config["tracking_indices"],
        config["tracking_targets"],
        config["tracking_scales"],
        config["control_scales"],
        prediction_horizon=config["prediction_horizon"],
        control_horizon=config["control_horizon"],
        device="gpu",
    )
    x0 = evaluator.initial_guess()
    lower = np.asarray(evaluator.variable_lower)
    upper = np.asarray(evaluator.variable_upper)
    published_control = np.asarray(case_data["published_first_control"])
    adapter = CyIpoptPFRCNNProblem(evaluator)
    problem = cyipopt.Problem(
        n=evaluator.n,
        m=evaluator.m,
        problem_obj=adapter,
        lb=lower,
        ub=upper,
        cl=np.zeros(evaluator.m),
        cu=np.zeros(evaluator.m),
    )
    problem.add_option("tol", args.tol)
    problem.add_option("max_iter", args.max_iter)
    problem.add_option("print_level", 0)
    problem.add_option("sb", "yes")
    problem.add_option("linear_solver", "mumps")
    if args.hessian == "limited-memory":
        problem.add_option("hessian_approximation", "limited-memory")

    # Compile only the callbacks used by the chosen curvature path before timing.
    evaluator.evaluate_obj(x0)
    evaluator.evaluate_grad(x0)
    evaluator.evaluate_cons(x0)
    evaluator.evaluate_jac_coord(x0)
    if args.hessian == "exact":
        evaluator.evaluate_hess_coord(x0, np.zeros(evaluator.m), 1.0)
    jax.effects_barrier()

    records = []
    for repetition in range(args.repetitions + 1):
        jax.effects_barrier()
        started = time.perf_counter()
        solution, info = problem.solve(x0)
        jax.effects_barrier()
        elapsed = time.perf_counter() - started
        _, controls, _ = evaluator.unpack(solution)
        first_control = controls[0]
        residuals = residual_audit(evaluator, solution, info, lower, upper)
        record = {
            "repetition": repetition,
            "warmup": repetition == 0,
            "elapsed_s": elapsed,
            "status": int(info["status"]),
            "status_msg": str(info["status_msg"]),
            "iterations": int(info.get("iter_count", -1)),
            "objective": float(info["obj_val"]),
            "first_control": first_control.tolist(),
            "published_first_control": published_control.tolist(),
            "first_control_absolute_error": np.abs(
                first_control - published_control
            ).tolist(),
            **residuals,
        }
        records.append(record)
        print(record, flush=True)

    payload = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config": config,
        "n": evaluator.n,
        "m": evaluator.m,
        "jacobian_nnz": evaluator.jacobian_nnz,
        "hessian_nnz": evaluator.hessian_nnz,
        "hessian": args.hessian,
        "repetitions": args.repetitions,
        "max_iter": args.max_iter,
        "tol": args.tol,
        "cyipopt": cyipopt.__version__,
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
