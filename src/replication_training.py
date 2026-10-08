"""Phase 3A: sequential continuation primitives; no oracle, M0 or campaign runner.

Reuse the immutable 49-parameter representation and the sourced M1/M2 losses.
The phase-2A Adam is intentionally not imported: its state rejects k > 300.
All labels come from the caller. Importing this module performs no learning.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict, replace
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys
import tempfile

from src import neural_surrogate as n
from src import surrogate_losses as losses

SCHEMA_VERSION = 1
MAX_UPDATES = 3000
CONDITIONS = ("M1_16", "M2_16", "M1_32")


def integer(value, name, maximum=MAX_UPDATES):
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer, not a boolean")
    if not 0 <= value <= maximum:
        raise ValueError(f"{name} outside [0, {maximum}]")
    return value


def canonical_bytes(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False).encode("utf-8")


def identity(value):
    return hashlib.sha256(canonical_bytes(value)).hexdigest().upper()


def _copy_json(value):
    return json.loads(canonical_bytes(value))


def _text(value, name):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")
    return value


def _digest(value, name):
    _text(value, name)
    if len(value) != 64 or any(c not in "0123456789ABCDEF" for c in value):
        raise ValueError(f"{name} must be an uppercase SHA-256")
    return value


def runtime_identity():
    return {"implementation": sys.implementation.name,
            "python_version": ".".join(map(str, sys.version_info[:3])),
            "executable": sys.executable,
            "isolated": bool(sys.flags.isolated),
            "no_site": bool(sys.flags.no_site),
            "no_bytecode": bool(sys.dont_write_bytecode)}


@dataclass(frozen=True)
class AdamConfig:
    eta: float = 0.001
    beta1: float = 0.9
    beta2: float = 0.999
    epsilon: float = 1e-8

    def __post_init__(self):
        for key in ("eta", "beta1", "beta2", "epsilon"):
            object.__setattr__(self, key, n._finite_scalar(getattr(self, key), key))
        if self.eta <= 0 or self.epsilon <= 0:
            raise ValueError("eta and epsilon must be positive")
        if not (0 <= self.beta1 < 1 and 0 <= self.beta2 < 1):
            raise ValueError("betas must be in [0,1)")


@dataclass(frozen=True)
class AdamState:
    parameters: n.NetworkParameters
    first_moment: tuple[float, ...]
    second_moment: tuple[float, ...]
    updates: int = 0
    config: AdamConfig = AdamConfig()

    def __post_init__(self):
        if not isinstance(self.parameters, n.NetworkParameters):
            raise TypeError("use the preserved NetworkParameters")
        if not isinstance(self.config, AdamConfig):
            raise TypeError("config must be AdamConfig")
        for key in ("first_moment", "second_moment"):
            object.__setattr__(self, key, n._finite_tuple(
                getattr(self, key), 49, key))
        integer(self.updates, "updates")
        if any(x < 0 for x in self.second_moment):
            raise ValueError("negative second moment")
        if self.updates == 0 and any(
                x != 0 for x in self.first_moment + self.second_moment):
            raise ValueError("a zero-step state must have zero moments")

    @classmethod
    def zero(cls, parameters, config=AdamConfig()):
        # Independent objects; all numerical storage is immutable tuples.
        return cls(n.NetworkParameters.from_vector(parameters.to_vector()),
                   tuple(0.0 for _ in range(49)),
                   tuple(0.0 for _ in range(49)), 0, config)


def adam_step(state, gradient):
    """One candidate update; never mutate the last valid state."""
    if not isinstance(state, AdamState):
        raise TypeError("state must be AdamState")
    grad = n._finite_tuple(gradient, 49, "gradient")
    if state.updates == MAX_UPDATES:
        raise ValueError("3000 completed updates already reached")
    t, cfg = state.updates + 1, state.config
    correction1, correction2 = 1 - cfg.beta1 ** t, 1 - cfg.beta2 ** t
    first, second, values = [], [], []
    for i, (theta, g, m, v) in enumerate(zip(
            state.parameters.to_vector(), grad,
            state.first_moment, state.second_moment)):
        square = n._finite_result(g * g, f"gradient square[{i}]")
        mt = n._finite_result(cfg.beta1 * m + (1 - cfg.beta1) * g, "first moment")
        vt = n._finite_result(cfg.beta2 * v + (1 - cfg.beta2) * square,
                              "second moment")
        mc = n._finite_result(mt / correction1, "corrected first moment")
        vc = n._finite_result(vt / correction2, "corrected second moment")
        denominator = n._finite_result(math.sqrt(vc) + cfg.epsilon, "denominator")
        candidate = n._finite_result(theta - cfg.eta * mc / denominator, "parameter")
        first.append(mt)
        second.append(vt)
        values.append(candidate)
    return AdamState(n.NetworkParameters.from_vector(values),
                     first, second, t, cfg)


def initialize_parameters(seed):
    integer(seed, "seed", 2**63 - 1)
    rng = random.Random(seed)
    bound = math.sqrt(6.0 / 17.0)
    w = tuple(rng.uniform(-bound, bound) for _ in range(16))
    v = tuple(rng.uniform(-bound, bound) for _ in range(16))
    return n.NetworkParameters(w, (0.0,) * 16, v, 0.0)


def paired_initial_states(seed, config=AdamConfig()):
    parameters = initialize_parameters(seed)
    return {condition: AdamState.zero(parameters, config) for condition in CONDITIONS}


def midpoint_sites(count):
    integer(count, "site count", 32)
    if count not in (16, 32):
        raise ValueError("only the frozen 16/32 training sites")
    points = tuple(n.A + (j + 0.5) * (n.B - n.A) / count for j in range(count))
    if not all(n.A < p < n.B for p in points):
        raise ValueError("invalid midpoint domain")
    if not all(left < right for left, right in zip(points, points[1:])):
        raise ValueError("non-increasing sites")
    return points


def _equal(actual, wanted, name):
    if canonical_bytes(actual) != canonical_bytes(wanted):
        raise ValueError(f"incompatible frozen {name}")


def validate_plan(plan):
    """Read the authority; do not redefine its statistical or decision choices."""
    if not isinstance(plan, dict):
        raise TypeError("plan must be a dictionary")
    _equal(plan["schema_version"], 1, "schema")
    _equal(plan["phase"], "2C", "source phase")
    _equal(plan["replication_status"], "PLANNED_NOT_EXECUTED", "campaign status")
    _equal(plan["physical_model"]["domain"], [n.A, n.B], "domain")
    net = plan["network"]
    for key, wanted in (
        ("architecture", [1, 16, 1]), ("activation", "tanh"),
        ("parameter_count", 49), ("order", list(n.PARAMETER_ORDER)),
        ("normalization", {"rule": "2*(p-a)/(b-a)-1",
                          "alpha": n.INPUT_ALPHA, "beta": n.INPUT_BETA,
                          "c_out": 0, "s_out": 1}),
    ):
        _equal(net[key], wanted, key)
    loss = plan["losses"]
    for key, wanted in (
        ("M1", "mean(((f-t)/S0)^2)"), ("M2", "M1+mean(((g-r)/S1)^2)"),
        ("sensitivity_coefficient", 1), ("half_factor", False),
        ("point_weights", "uniform means"), ("regularization", "none"),
    ):
        _equal(loss[key], wanted, key)
    losses._positive_scale(loss["S0"], "S0")
    losses._positive_scale(loss["S1"], "S1")
    _text(loss["scales_origin"], "scale provenance")
    optimizer = plan["optimizer"]
    for key, wanted in (
        ("name", "Adam"), ("eta", 0.001), ("beta1", 0.9), ("beta2", 0.999),
        ("epsilon", 1e-8), ("epsilon_position", "sqrt(v_corr)+epsilon"),
        ("bias_correction", True), ("initial_first_moment", 0),
        ("initial_second_moment", 0), ("initial_updates", 0),
        ("first_update_k", 1),
    ):
        _equal(optimizer[key], wanted, key)
    rep = plan["replication"]
    _equal(rep["seeds"], list(range(20262001, 20262021)), "20 seeds")
    wanted_conditions = [
        {"id": "M1_16", "method": "M1", "training_sites": 16,
         "value_labels": 16, "derivative_labels": 0, "C_label_q1_1": 16},
        {"id": "M2_16", "method": "M2", "training_sites": 16,
         "value_labels": 16, "derivative_labels": 16, "C_label_q1_1": 32},
        {"id": "M1_32", "method": "M1", "training_sites": 32,
         "value_labels": 32, "derivative_labels": 0, "C_label_q1_1": 32},
    ]
    _equal(rep["conditions"], wanted_conditions, "conditions")
    for key, wanted in (
        ("trajectories", 60), ("updates_per_trajectory", 3000),
        ("maximum_updates", 180000), ("snapshots", [300, 3000]),
        ("primary_snapshot", 3000), ("secondary_snapshot", 300),
        ("full_batch", True), ("continue_state_without_reset", True),
        ("logging_controls", [0, 300, 3000]), ("checkpoint_selection", False),
        ("early_stopping_for_validation", False), ("automatic_budget_tuning", False),
    ):
        _equal(rep[key], wanted, key)
    _equal(rep["training_sites"]["rule"], "a+(j+1/2)*(b-a)/n", "midpoint rule")
    _equal(rep["training_sites"]["n"], [16, 32], "site counts")
    init = rep["initialization"]
    for key, wanted in (
        ("generator", "local random.Random"), ("bound_rule", "sqrt(6/17)"),
        ("draw_order", "16 w draws, then 16 v draws; no draws for biases"),
        ("hidden_biases", 0), ("output_bias", 0),
        ("shared_initial_parameters_across_conditions", True),
        ("separate_optimizer_states", True),
    ):
        _equal(init[key], wanted, key)
    if set(midpoint_sites(16)) & set(midpoint_sites(32)):
        raise ValueError("16/32 sites overlap")
    return plan


def protocol_signature(plan):
    validate_plan(plan)
    # Exclude dates, previous output hashes, execution metadata and stale
    # implementation-status prose. The original config byte hash is separate.
    scientific = {key: plan[key] for key in (
        "physical_model", "network", "losses", "optimizer", "common_diagnostics",
        "decisions", "hypothesis", "future_statistics", "failure_rules")}
    scientific["replication"] = {
        key: value for key, value in plan["replication"].items()
        if key != "future_orchestrator_required"}
    return identity(scientific)


def build_jobs(plan):
    validate_plan(plan)
    signature = protocol_signature(plan)
    jobs = []
    for seed in plan["replication"]["seeds"]:
        initial_id = identity(initialize_parameters(seed).to_vector())
        for condition in plan["replication"]["conditions"]:
            name = condition["id"]
            directory = f"results/replication/seed-{seed}/{name}"
            jobs.append({
                "seed": seed, "condition": name, "method": condition["method"],
                "points": midpoint_sites(condition["training_sites"]),
                "job_id": identity([signature, seed, name]),
                "protocol_signature": signature, "initial_parameter_id": initial_id,
                "label_identity": None, "label_identity_status": "NOT_YET_ACQUIRED",
                "target_updates": 3000, "snapshot_updates": [300, 3000],
                "scales": {"S0": float(plan["losses"]["S0"]),
                           "S1": float(plan["losses"]["S1"]),
                           "origin": plan["losses"]["scales_origin"]},
                "normalization": asdict(n.AffineNormalization(
                    **{key: plan["network"]["normalization"][key]
                       for key in ("alpha", "beta", "c_out", "s_out")})),
                "adam": asdict(AdamConfig(
                    **{key: plan["optimizer"][key]
                       for key in ("eta", "beta1", "beta2", "epsilon")})),
                "checkpoint_path": directory + "/checkpoint.json",
                "snapshot_paths": {str(k): directory + f"/snapshot_{k:04d}.json"
                                   for k in (300, 3000)},
            })
    if len({job["job_id"] for job in jobs}) != 60:
        raise ValueError("duplicate jobs")
    return jobs


@dataclass(frozen=True)
class SuppliedLabels:
    points: tuple[float, ...]
    values: tuple[float, ...]
    derivatives: tuple[float, ...] | None = None

    def __post_init__(self):
        points = tuple(n._validate_p(p) for p in self.points)
        if not points or len(set(points)) != len(points):
            raise ValueError("nonempty distinct supplied points required")
        object.__setattr__(self, "points", points)
        object.__setattr__(self, "values", n._finite_tuple(
            self.values, len(points), "values"))
        if self.derivatives is not None:
            object.__setattr__(self, "derivatives", n._finite_tuple(
                self.derivatives, len(points), "derivatives"))

    def data_identity(self, method):
        if method not in ("M1", "M2"):
            raise ValueError("unknown method")
        payload = {"points": self.points, "values": self.values}
        if method == "M2":
            if self.derivatives is None:
                raise ValueError("M2 needs supplied derivatives")
            payload["derivatives"] = self.derivatives
        # Unused derivative labels cannot affect M1's scientific data identity.
        return identity(payload)


@dataclass(frozen=True)
class LossAdapter:
    method: str
    labels: SuppliedLabels
    S0: float
    S1: float
    scales_origin: str
    normalization: n.AffineNormalization = n.DEFAULT_NORMALIZATION

    def __post_init__(self):
        if not isinstance(self.labels, SuppliedLabels):
            raise TypeError("supply labels, never an oracle")
        self.labels.data_identity(self.method)
        object.__setattr__(self, "S0", losses._positive_scale(self.S0, "S0"))
        object.__setattr__(self, "S1", losses._positive_scale(self.S1, "S1"))
        _text(self.scales_origin, "scale provenance")
        if not isinstance(self.normalization, n.AffineNormalization):
            raise TypeError("normalization must be the preserved affine type")

    def __call__(self, parameters):
        network = n.NeuralSurrogate(parameters, self.normalization)
        if self.method == "M1":
            return losses.m1_loss_and_gradient(
                network, self.labels.points, self.labels.values, S0=self.S0)
        return losses.m2_loss_and_gradient(
            network, self.labels.points, self.labels.values, self.labels.derivatives,
            S0=self.S0, S1=self.S1)


def make_context(*, seed, condition, method, protocol_id, data_id, points,
                 S0, S1, scales_origin, critical_sources, initial_parameter_id,
                 environment=None, config=AdamConfig(), target_updates=3000,
                 snapshot_updates=(300, 3000), data_origin,
                 normalization=n.DEFAULT_NORMALIZATION):
    context = {
        "schema_version": SCHEMA_VERSION, "seed": seed, "condition": condition,
        "method": method, "architecture": [1, 16, 1],
        "parameter_order": list(n.PARAMETER_ORDER),
        "normalization": asdict(normalization),
        "scales": {"S0": S0, "S1": S1, "origin": scales_origin},
        "points": list(points), "data_identity": data_id, "data_origin": data_origin,
        "protocol_signature": protocol_id, "initial_parameter_id": initial_parameter_id,
        "adam": asdict(config), "target_updates": target_updates,
        "snapshot_updates": list(snapshot_updates),
        "critical_sources": critical_sources,
        "environment": environment if environment is not None else runtime_identity(),
    }
    return validate_context(context)


def context_from_labels(job, adapter, *, critical_sources, environment=None):
    if adapter.method != job["method"] or adapter.labels.points != tuple(job["points"]):
        raise ValueError("provided labels do not belong to this planned job")
    _equal({"S0": adapter.S0, "S1": adapter.S1, "origin": adapter.scales_origin},
           job["scales"], "supplied scales against the frozen job")
    _equal(asdict(adapter.normalization), job["normalization"],
           "normalization against the frozen job")
    return make_context(
        seed=job["seed"], condition=job["condition"], method=adapter.method,
        protocol_id=job["protocol_signature"],
        data_id=adapter.labels.data_identity(adapter.method), points=adapter.labels.points,
        S0=adapter.S0, S1=adapter.S1, scales_origin=adapter.scales_origin,
        critical_sources=critical_sources, initial_parameter_id=job["initial_parameter_id"],
        environment=environment, data_origin="caller supplied, provenance checked upstream",
        normalization=adapter.normalization, config=AdamConfig(**job["adam"]),
        target_updates=job["target_updates"],
        snapshot_updates=tuple(job["snapshot_updates"]))


def validate_context(context):
    context = _copy_json(context)
    integer(context["schema_version"], "context schema", 1)
    _equal(context["schema_version"], SCHEMA_VERSION, "context schema")
    integer(context["seed"], "seed", 2**63 - 1)
    for key in ("condition", "method", "data_origin"):
        _text(context[key], key)
    if context["method"] not in ("M1", "M2", "QUADRATIC_FIXTURE"):
        raise ValueError("unknown context method")
    _equal(context["architecture"], [1, 16, 1], "architecture")
    _equal(context["parameter_order"], list(n.PARAMETER_ORDER), "parameter order")
    n.AffineNormalization(**context["normalization"])
    AdamConfig(**context["adam"])
    for key in ("S0", "S1"):
        losses._positive_scale(context["scales"][key], key)
    _text(context["scales"]["origin"], "scale origin")
    for key in ("data_identity", "protocol_signature", "initial_parameter_id"):
        _digest(context[key], key)
    if not isinstance(context["critical_sources"], dict) or not context["critical_sources"]:
        raise ValueError("critical source identities required")
    for path, digest in context["critical_sources"].items():
        _text(path, "critical source path")
        _digest(digest, "critical source hash")
    env = context["environment"]
    for key in ("python_version", "implementation", "executable"):
        _text(env[key], key)
    for key in ("isolated", "no_site", "no_bytecode"):
        if type(env[key]) is not bool or not env[key]:
            raise ValueError("require -I -S -B")
    points = tuple(n._validate_p(p) for p in context["points"])
    if len(points) != len(set(points)):
        raise ValueError("duplicate context points")
    if not points and context["method"] != "QUADRATIC_FIXTURE":
        raise ValueError("training sites required")
    target = integer(context["target_updates"], "target updates")
    if target == 0:
        raise ValueError("positive final target required")
    snapshots = context["snapshot_updates"]
    if not isinstance(snapshots, list):
        raise TypeError("snapshot steps must be a list")
    for k in snapshots:
        integer(k, "snapshot step", target)
    if snapshots != sorted(set(snapshots)) or any(k == 0 for k in snapshots):
        raise ValueError("snapshot steps must be positive, unique, increasing")
    return context


@dataclass(frozen=True)
class EvaluationCounts:
    update_attempts: int = 0
    update_completed: int = 0
    logging_attempts: int = 0
    logging_completed: int = 0
    known_recomputed_loss_gradient_calls: int = 0

    def __post_init__(self):
        for key, value in asdict(self).items():
            integer(value, key, 10**12)
        if self.update_completed > self.update_attempts:
            raise ValueError("completed update evaluations exceed attempts")
        if self.logging_completed > self.logging_attempts:
            raise ValueError("completed logging evaluations exceed attempts")


def state_payload(state):
    return {"parameters": state.parameters.to_vector(),
            "first_moment": state.first_moment, "second_moment": state.second_moment,
            "updates": state.updates, "adam": asdict(state.config)}


def state_from_payload(payload):
    return AdamState(n.NetworkParameters.from_vector(payload["parameters"]),
                     payload["first_moment"], payload["second_moment"],
                     payload["updates"], AdamConfig(**payload["adam"]))


@dataclass(frozen=True)
class TrainingCheckpoint:
    state: AdamState
    context: dict
    snapshots: dict
    counts: EvaluationCounts = EvaluationCounts()
    logs: tuple = ()
    status: str = "READY"
    incident: dict | None = None

    def __post_init__(self):
        context = validate_context(self.context)
        object.__setattr__(self, "context", context)
        if not isinstance(self.state, AdamState) or not isinstance(self.counts, EvaluationCounts):
            raise TypeError("invalid state/count types")
        _equal(asdict(self.state.config), context["adam"], "state Adam config")
        if self.state.updates > context["target_updates"]:
            raise ValueError("state exceeds context target")
        if self.state.updates == 0:
            _equal(identity(self.state.parameters.to_vector()),
                   context["initial_parameter_id"], "initial parameters")
        if self.counts.update_completed < self.state.updates:
            raise ValueError("lost cost of completed logical updates")
        snapshots = dict(self.snapshots)
        for key, state in snapshots.items():
            if not isinstance(key, str) or key != str(state.updates):
                raise ValueError("snapshot key must be its exact global step")
            if state.updates not in context["snapshot_updates"] or state.updates > self.state.updates:
                raise ValueError("unexpected snapshot")
            _equal(asdict(state.config), context["adam"], "snapshot Adam config")
        for step in context["snapshot_updates"]:
            if step <= self.state.updates and str(step) not in snapshots:
                raise ValueError("missing already-acquired snapshot")
        if str(self.state.updates) in snapshots and snapshots[str(self.state.updates)] != self.state:
            raise ValueError("snapshot differs from current state")
        object.__setattr__(self, "snapshots", snapshots)
        logs = tuple(_copy_json(item) for item in self.logs)
        steps = []
        for log in logs:
            step = integer(log["updates"], "logged step", self.state.updates)
            n._finite_scalar(log["loss"], "logged loss")
            n._finite_scalar(log["gradient_norm"], "logged gradient norm")
            steps.append(step)
        if steps != sorted(set(steps)) or self.counts.logging_completed < len(logs):
            raise ValueError("duplicate logs or lost logging cost")
        object.__setattr__(self, "logs", logs)
        if self.status not in ("READY", "COMPLETED", "STOPPED_ERROR"):
            raise ValueError("invalid job status")
        if (self.status == "COMPLETED") != (self.state.updates == context["target_updates"]):
            # A logging failure at the final step is still a recorded failure.
            if self.status != "STOPPED_ERROR":
                raise ValueError("completion status disagrees with global step")
        if (self.status == "STOPPED_ERROR") != (self.incident is not None):
            raise ValueError("failure status must carry its incident")
        if self.incident is not None:
            object.__setattr__(self, "incident", _copy_json(self.incident))

    @classmethod
    def start(cls, state, context):
        if state.updates != 0:
            raise ValueError("start accepts only a zero state; restore for continuation")
        return cls(state, context, {})


def _validated(checkpoint, expected_context):
    if not isinstance(checkpoint, TrainingCheckpoint):
        raise TypeError("checkpoint must be TrainingCheckpoint")
    _equal(validate_context(checkpoint.context), validate_context(expected_context),
           "checkpoint context (including exact Python version)")
    _equal(checkpoint.context["environment"], runtime_identity(),
           "current deterministic runtime")
    return TrainingCheckpoint(checkpoint.state, checkpoint.context, checkpoint.snapshots,
                              checkpoint.counts, checkpoint.logs,
                              checkpoint.status, checkpoint.incident)


def _evaluate(callback, parameters):
    result = callback(parameters)
    value = n._finite_scalar(result.loss, "loss")
    gradient = n._finite_tuple(result.gradient, 49, "loss gradient")
    return value, gradient


def _validate_evaluator(callback, context):
    if not callable(callback):
        raise TypeError("loss/gradient callback required")
    if context["method"] == "QUADRATIC_FIXTURE":
        return  # Explicit algorithmic fixture, never an oracle objective.
    if not isinstance(callback, LossAdapter):
        raise TypeError("M1/M2 continuation requires the supplied-label adapter")
    for actual, wanted, name in (
        (callback.method, context["method"], "adapter method"),
        (list(callback.labels.points), context["points"], "adapter sites"),
        (callback.labels.data_identity(callback.method),
         context["data_identity"], "actual adapter labels"),
        ({"S0": callback.S0, "S1": callback.S1, "origin": callback.scales_origin},
         context["scales"], "adapter scales"),
        (asdict(callback.normalization), context["normalization"], "adapter normalization"),
    ):
        _equal(actual, wanted, name)


def continue_to(checkpoint, target, evaluate, *, expected_context):
    """Advance to an ABSOLUTE step; snapshots contain post-update states.

    No logging recomputation is hidden here. Call log_loss_gradient explicitly
    at the frozen controls, before saving the final completed checkpoint.
    """
    cp = _validated(checkpoint, expected_context)
    target = integer(target, "absolute target", cp.context["target_updates"])
    if target < cp.state.updates:
        raise ValueError("cannot continue backwards")
    if target == cp.state.updates:
        return cp  # No evaluation, RNG use, logging, mutation or write.
    if cp.status != "READY":
        raise ValueError("completed/failed job cannot be restarted automatically")
    _validate_evaluator(evaluate, cp.context)
    state, snapshots, counts = cp.state, dict(cp.snapshots), cp.counts
    incident = None
    stage = "loss_gradient"
    while state.updates < target:
        try:
            stage = "loss_gradient"
            counts = replace(counts, update_attempts=counts.update_attempts + 1)
            _, gradient = _evaluate(evaluate, state.parameters)
            counts = replace(counts, update_completed=counts.update_completed + 1)
            stage = "Adam_candidate"
            candidate = adam_step(state, gradient)
            state = candidate  # Only after all 49 components/state validated.
            if state.updates in cp.context["snapshot_updates"]:
                snapshots[str(state.updates)] = state
        except (ArithmeticError, ValueError, TypeError) as exc:
            incident = {"stage": stage, "attempted_update": state.updates + 1,
                        "completed_updates": state.updates,
                        "error_type": type(exc).__name__, "message": str(exc),
                        "automatic_retry": False}
            break
    status = ("STOPPED_ERROR" if incident else
              "COMPLETED" if state.updates == cp.context["target_updates"] else "READY")
    return TrainingCheckpoint(state, cp.context, snapshots, counts, cp.logs, status, incident)


def log_loss_gradient(checkpoint, evaluate, *, expected_context):
    cp = _validated(checkpoint, expected_context)
    if cp.status == "STOPPED_ERROR":
        raise ValueError("do not continue a failed job")
    if any(log["updates"] == cp.state.updates for log in cp.logs):
        return cp  # An already acquired output is not a second result.
    _validate_evaluator(evaluate, cp.context)
    counts = replace(cp.counts, logging_attempts=cp.counts.logging_attempts + 1)
    try:
        loss, gradient = _evaluate(evaluate, cp.state.parameters)
        norm = n._finite_result(math.hypot(*gradient), "gradient norm")
        counts = replace(counts, logging_completed=counts.logging_completed + 1)
        logs = cp.logs + ({"updates": cp.state.updates, "loss": loss,
                          "gradient_norm": norm},)
        return replace(cp, counts=counts, logs=logs)
    except (ArithmeticError, ValueError, TypeError) as exc:
        return replace(cp, counts=counts, status="STOPPED_ERROR",
                       incident={"stage": "logging", "completed_updates": cp.state.updates,
                                 "error_type": type(exc).__name__, "message": str(exc),
                                 "automatic_retry": False})


def retain_known_recomputed_cost(checkpoint, calls, *, expected_context):
    """Explicit known work after an interrupted save; never estimate unknown cost."""
    cp = _validated(checkpoint, expected_context)
    integer(calls, "known recomputed calls", 10**12)
    return replace(cp, counts=replace(
        cp.counts, known_recomputed_loss_gradient_calls=
        cp.counts.known_recomputed_loss_gradient_calls + calls))


def checkpoint_payload(checkpoint, *, expected_context):
    cp = _validated(checkpoint, expected_context)
    body = {
        "schema_version": SCHEMA_VERSION, "context": cp.context,
        "context_signature": identity(cp.context), "state": state_payload(cp.state),
        "snapshots": {key: state_payload(value) for key, value in cp.snapshots.items()},
        "counts": asdict(cp.counts), "logs": cp.logs,
        "status": cp.status, "incident": cp.incident,
    }
    return dict(body, payload_sha256=identity(body))


def restore_payload(payload, *, expected_context):
    payload = _copy_json(payload)
    integer(payload["schema_version"], "checkpoint schema", 1)
    _equal(payload["schema_version"], SCHEMA_VERSION, "checkpoint schema")
    wanted_hash = payload.pop("payload_sha256")
    _equal(identity(payload), wanted_hash, "checkpoint integrity")
    context = validate_context(payload["context"])
    _equal(identity(context), payload["context_signature"], "context integrity")
    _equal(context, validate_context(expected_context), "restoration context")
    _equal(context["environment"], runtime_identity(), "restoration Python runtime")
    return TrainingCheckpoint(
        state_from_payload(payload["state"]), context,
        {key: state_from_payload(value) for key, value in payload["snapshots"].items()},
        EvaluationCounts(**payload["counts"]), tuple(payload["logs"]),
        payload["status"], payload["incident"])


def _object_pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def read_json(path):
    def reject(value):
        raise ValueError(f"non-finite JSON literal: {value}")
    return json.loads(Path(path).read_text(encoding="utf-8"),
                      parse_constant=reject, object_pairs_hook=_object_pairs)


def load_checkpoint(path, *, expected_context):
    return restore_payload(read_json(path), expected_context=expected_context)


def _check_existing(path, candidate, expected_context):
    if not path.exists():
        return None, False
    existing = load_checkpoint(path, expected_context=expected_context)
    old = checkpoint_payload(existing, expected_context=expected_context)
    new = checkpoint_payload(candidate, expected_context=expected_context)
    if canonical_bytes(old) == canonical_bytes(new):
        return identity(read_json(path)), True
    if existing.status in ("COMPLETED", "STOPPED_ERROR"):
        raise ValueError("already completed/failed output; preserve it")
    if candidate.state.updates < existing.state.updates:
        raise ValueError("refuse checkpoint rollback")
    if candidate.state.updates == existing.state.updates and candidate.state != existing.state:
        raise ValueError("different numerical result at the same global step")
    for key, value in existing.snapshots.items():
        if candidate.snapshots.get(key) != value:
            raise ValueError("lost/changed already-acquired snapshot")
    if candidate.logs[:len(existing.logs)] != existing.logs:
        raise ValueError("lost/changed already-acquired logging")
    for key, value in asdict(existing.counts).items():
        if asdict(candidate.counts)[key] < value:
            raise ValueError("lost known executed cost")
    return identity(read_json(path)), False


def save_checkpoint(path, checkpoint, *, expected_context):
    """Sequential single-writer atomic replace; not a universal crash guarantee."""
    path = Path(path)
    cp = _validated(checkpoint, expected_context)
    payload = checkpoint_payload(cp, expected_context=expected_context)
    if not path.parent.is_dir():
        raise ValueError("destination directory must already exist")
    prior_id, identical = _check_existing(path, cp, expected_context)
    if identical:
        return {"status": "REUSED_IDENTICAL", "written": False}
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
                mode="w", encoding="utf-8", newline="\n", delete=False,
                dir=path.parent, prefix=path.name + ".", suffix=".tmp") as handle:
            temporary = Path(handle.name)
            handle.write(json.dumps(payload, sort_keys=True, indent=2,
                                    allow_nan=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        restored = load_checkpoint(temporary, expected_context=expected_context)
        _equal(checkpoint_payload(restored, expected_context=expected_context),
               payload, "written checkpoint round trip")
        current_id = identity(read_json(path)) if path.exists() else None
        if current_id != prior_id:
            raise ValueError("destination changed during write; sequential collision")
        os.replace(temporary, path)
        temporary = None
        return {"status": "WRITTEN", "written": True}
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
