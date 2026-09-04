#!/usr/bin/env python3
"""OMLT full/reduced-space PFR-NMPC first-step formulation baseline."""

from __future__ import annotations

import argparse
import json
import pickle
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import onnx
import pyomo.environ as pyo
import torch
from omlt import OffsetScaling, OmltBlock
from omlt.io import load_onnx_neural_network
from omlt.neuralnet import FullSpaceSmoothNNFormulation, ReducedSpaceSmoothNNFormulation


def raw_network_from_checkpoint(checkpoint, activation="tanh"):
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    layer_indices = sorted(
        int(key.split(".")[2])
        for key in state
        if key.startswith("dnn.dense_layers.") and key.endswith(".weight")
    )
    layers = []
    for position, index in enumerate(layer_indices):
        weight = state[f"dnn.dense_layers.{index}.weight"]
        bias = state[f"dnn.dense_layers.{index}.bias"]
        layer = torch.nn.Linear(weight.shape[1], weight.shape[0], dtype=torch.float64)
        layer.weight.data.copy_(weight.double())
        layer.bias.data.copy_(bias.double())
        layers.append(layer)
        if position + 1 < len(layer_indices):
            layers.append(torch.nn.Tanh() if activation == "tanh" else torch.nn.Sigmoid())
    return (
        torch.nn.Sequential(*layers).eval(),
        state["lb"].double().numpy(),
        state["ub"].double().numpy(),
    )


def build_network_definition(checkpoint):
    network, lower, upper = raw_network_from_checkpoint(checkpoint, activation="tanh")
    export_network, _, _ = raw_network_from_checkpoint(
        checkpoint, activation="sigmoid"
    )
    with tempfile.NamedTemporaryFile(suffix=".onnx") as handle:
        torch.onnx.export(
            export_network,
            torch.zeros(1, lower.size, dtype=torch.float64),
            handle.name,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
            opset_version=17,
            dynamo=False,
        )
        model = onnx.load(handle.name)
    scaling = OffsetScaling(
        offset_inputs=(upper + lower) / 2,
        factor_inputs=(upper - lower) / 2,
        offset_outputs=np.zeros(10),
        factor_outputs=np.ones(10),
    )
    definition = load_onnx_neural_network(
            model,
            scaling_object=scaling,
            input_bounds={index: (-1.0, 1.0) for index in range(lower.size)},
    )
    for layer in definition.layers:
        if layer.activation == "sigmoid":
            layer.activation = "tanh"
    return definition, network, lower, upper


def rollout(network, lower, upper, horizon):
    state = torch.full((1, 10), 0.5, dtype=torch.float64)
    control = torch.full((1, 2), 0.5, dtype=torch.float64)
    states = [state.numpy()[0]]
    with torch.no_grad():
        for _ in range(horizon):
            inputs = torch.cat((state, control), dim=1)
            normalized = 2 * (inputs - torch.as_tensor(lower)) / torch.as_tensor(
                upper - lower
            ) - 1
            state = network(normalized)
            states.append(state.numpy()[0])
    return np.asarray(states)


def build_model(config, formulation_name):
    definition, network, lower, upper = build_network_definition(config["model"])
    formulation = (
        FullSpaceSmoothNNFormulation(definition)
        if formulation_name == "full-space"
        else ReducedSpaceSmoothNNFormulation(definition)
    )
    prediction_horizon = config["prediction_horizon"]
    control_horizon = config["control_horizon"]
    initial_states = rollout(network, lower, upper, prediction_horizon)
    model = pyo.ConcreteModel()
    model.steps = pyo.RangeSet(0, prediction_horizon)
    model.control_steps = pyo.RangeSet(-1, prediction_horizon - 1)
    model.transitions = pyo.RangeSet(0, prediction_horizon - 1)
    model.states = pyo.RangeSet(1, 10)
    model.controls = pyo.RangeSet(1, 2)
    for step in range(1, prediction_horizon + 1):
        block = OmltBlock()
        setattr(model, f"nn{step}", block)
        block.build_formulation(formulation)

    model.x = pyo.Var(
        model.steps,
        model.states,
        bounds=(config["state_lower"], config["state_upper"]),
        initialize=lambda _model, step, state: initial_states[step, state - 1],
    )
    model.u = pyo.Var(
        model.control_steps,
        model.controls,
        bounds=(config["control_lower"], config["control_upper"]),
        initialize=0.5,
    )
    model.delta_u = pyo.Var(model.transitions, model.controls, initialize=0.0)
    model.setpoint = pyo.Var(initialize=config["setpoint"])
    model.delta_constraint = pyo.Constraint(
        model.transitions,
        model.controls,
        rule=lambda m, step, control: m.delta_u[step, control]
        == m.u[step, control] - m.u[step - 1, control],
    )
    model.state_inputs = pyo.Constraint(
        model.transitions,
        model.states,
        rule=lambda m, step, state: m.x[step, state]
        == getattr(m, f"nn{step + 1}").inputs[state - 1],
    )
    model.state_outputs = pyo.Constraint(
        model.transitions,
        model.states,
        rule=lambda m, step, state: m.x[step + 1, state]
        == getattr(m, f"nn{step + 1}").outputs[state - 1],
    )
    model.control_inputs = pyo.Constraint(
        model.transitions,
        model.controls,
        rule=lambda m, step, control: m.u[step, control]
        == getattr(m, f"nn{step + 1}").inputs[10 + control - 1],
    )

    def control_horizon_rule(m, step, control):
        if step < control_horizon:
            return pyo.Constraint.Skip
        return m.u[step, control] == m.u[control_horizon - 1, control]

    model.control_horizon = pyo.Constraint(
        model.control_steps, model.controls, rule=control_horizon_rule
    )
    model.objective = pyo.Objective(
        expr=sum(
            (model.x[step, 10] - model.setpoint) ** 2
            for step in model.steps
        )
        + sum(
            model.delta_u[step, control] ** 2
            for step in model.transitions
            for control in model.controls
        )
    )
    for state, value in enumerate(config["current_state"], start=1):
        model.x[0, state].fix(value)
    for control, value in enumerate(config["previous_control"], start=1):
        model.u[-1, control].fix(value)
    model.setpoint.fix(config["setpoint"])
    return model


def max_constraint_violation(model):
    violation = 0.0
    for constraint in model.component_data_objects(pyo.Constraint, active=True):
        body = pyo.value(constraint.body)
        if constraint.lower is not None:
            violation = max(violation, pyo.value(constraint.lower) - body)
        if constraint.upper is not None:
            violation = max(violation, body - pyo.value(constraint.upper))
    return float(max(0.0, violation))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--formulation", choices=("full-space", "reduced-space"), required=True)
    parser.add_argument("--ipopt", type=Path, required=True)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--tol", type=float, default=1e-6)
    parser.add_argument("--prediction-horizon", type=int)
    parser.add_argument("--control-horizon", type=int)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    if args.prediction_horizon is not None:
        config["prediction_horizon"] = args.prediction_horizon
    if args.control_horizon is not None:
        config["control_horizon"] = args.control_horizon
    started = time.perf_counter()
    model = build_model(config, args.formulation)
    build_time = time.perf_counter() - started
    variable_count = sum(1 for _ in model.component_data_objects(pyo.Var))
    constraint_count = sum(1 for _ in model.component_data_objects(pyo.Constraint, active=True))
    initial_values = {
        id(variable): variable.value
        for variable in model.component_data_objects(pyo.Var)
    }
    solver = pyo.SolverFactory("ipopt", executable=str(args.ipopt))
    solver.options["tol"] = args.tol
    solver.options["max_iter"] = 500
    solver.options["print_level"] = 0
    solver.options["sb"] = "yes"
    solver.options["linear_solver"] = "mumps"
    with open(config["published_output"], "rb") as handle:
        published = pickle.load(handle)
    published_control = np.asarray([published["F"][1], published["C_in"][1]])
    records = []
    for repetition in range(args.repetitions):
        for variable in model.component_data_objects(pyo.Var):
            variable.set_value(initial_values[id(variable)], skip_validation=True)
        start = time.perf_counter()
        result = solver.solve(model, tee=False)
        elapsed = time.perf_counter() - start
        control = np.asarray([pyo.value(model.u[0, 1]), pyo.value(model.u[0, 2])])
        record = {
            "repetition": repetition,
            "elapsed_s": elapsed,
            "solver_time_s": float(getattr(result.solver, "time", np.nan)),
            "termination_condition": str(result.solver.termination_condition),
            "objective": float(pyo.value(model.objective)),
            "first_control": control.tolist(),
            "published_first_control": published_control.tolist(),
            "first_control_absolute_error": np.abs(control - published_control).tolist(),
            "max_constraint_violation": max_constraint_violation(model),
        }
        records.append(record)
        print(record, flush=True)
    payload = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config": config,
        "formulation": args.formulation,
        "build_time_s": build_time,
        "variable_count": variable_count,
        "constraint_count": constraint_count,
        "repetitions": args.repetitions,
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
