#!/usr/bin/env python3
"""Same-oracle cyipopt baseline for the closest-prior MNIST NLP."""

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
import torchvision
from torchvision.transforms import ToTensor


class Adapter:
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repetitions", type=int, default=1)
    parser.add_argument("--tol", type=float, default=1e-7)
    parser.add_argument("--hessian", choices=("exact", "limited-memory"), default="exact")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))
    from jax_nn_evaluator import create_mnist_adversarial_evaluator

    data = torchvision.datasets.MNIST(
        root=config["data_dir"], train=False, transform=ToTensor(), download=False
    )
    image, label = data[config["sample"]]
    reference = image.reshape(-1).numpy()
    records = []
    for model_path in config["models"]:
        evaluator = create_mnist_adversarial_evaluator(
            model_path,
            reference,
            config["adversarial_label"],
            threshold=config["threshold"],
            device="gpu",
        )
        x0 = evaluator.initial_guess()
        lower = np.concatenate(
            (np.zeros(784), np.full(10, -np.inf), np.zeros(1568))
        )
        upper = np.concatenate(
            (np.ones(784), np.full(10, np.inf), np.full(1568, np.inf))
        )
        constraint_lower = np.concatenate(
            (np.zeros(794), [config["threshold"]], np.zeros(10))
        )
        constraint_upper = np.concatenate(
            (np.zeros(794), [np.inf], np.ones(10))
        )
        problem = cyipopt.Problem(
            n=evaluator.n,
            m=evaluator.m,
            problem_obj=Adapter(evaluator),
            lb=lower,
            ub=upper,
            cl=constraint_lower,
            cu=constraint_upper,
        )
        problem.add_option("tol", args.tol)
        problem.add_option("max_iter", 1000)
        problem.add_option("print_level", 0)
        problem.add_option("sb", "yes")
        problem.add_option("linear_solver", "mumps")
        if args.hessian == "limited-memory":
            problem.add_option("hessian_approximation", "limited-memory")
        evaluator.evaluate_obj(x0)
        evaluator.evaluate_grad(x0)
        evaluator.evaluate_cons(x0)
        evaluator.evaluate_jac_coord(x0)
        if args.hessian == "exact":
            evaluator.evaluate_hess_coord(x0, np.zeros(evaluator.m), 1.0)
        for repetition in range(args.repetitions + 1):
            jax.effects_barrier()
            started = time.perf_counter()
            solution, info = problem.solve(x0)
            jax.effects_barrier()
            elapsed = time.perf_counter() - started
            pixels, outputs, _, _ = evaluator.unpack(solution)
            constraints = evaluator.evaluate_cons(solution)
            raw_violation = max(
                np.max(np.abs(constraints[:794])),
                max(0.0, config["threshold"] - constraints[794]),
                np.max(np.maximum(0.0, np.maximum(-constraints[795:], constraints[795:] - 1))),
            )
            record = {
                "model": model_path,
                "parameter_count": evaluator.parameter_count,
                "repetition": repetition,
                "warmup": repetition == 0,
                "elapsed_s": elapsed,
                "status": int(info["status"]),
                "status_msg": str(info["status_msg"]),
                "objective": float(info["obj_val"]),
                "actual_l1_perturbation": float(np.linalg.norm(pixels - reference, ord=1)),
                "target_probability": float(outputs[config["adversarial_label"]]),
                "predicted_label": int(np.argmax(outputs)),
                "max_raw_constraint_violation": float(raw_violation),
            }
            records.append(record)
            print(record, flush=True)
    payload = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config": config,
        "true_label": int(label),
        "hessian": args.hessian,
        "tol": args.tol,
        "cyipopt": cyipopt.__version__,
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
