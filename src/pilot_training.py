"""Bounded phase 2A pilot primitives; standard library, preserved losses.

No oracle import: labels must be acquired before calling train_method.
Adam's constants are fixed; second_moment is not the network's output block v.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
import hashlib
import json
import math
import random

from src import neural_surrogate as n
from src import surrogate_losses as losses

SEED = 20261006
MAX_UPDATES = 300
CONTROL_UPDATES = (0, 1, 10, 50, 100, 200, 300)
ETA, BETA1, BETA2, EPSILON = 0.001, 0.9, 0.999, 1e-8
NEAR_SATURATION = 0.999


def canonical_id(value) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"),
                     ensure_ascii=True, allow_nan=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest().upper()


def _count(value, name, maximum):
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if not 0 <= value <= maximum:
        raise ValueError(f"{name} must be between 0 and {maximum}")
    return value


def _positive(value, name):
    result = n._finite_scalar(value, name)
    if result <= 0:
        raise ValueError(f"{name} must be positive")
    return result


@dataclass(frozen=True)
class AdamState:
    first_moment: tuple[float, ...]
    second_moment: tuple[float, ...]
    updates: int = 0

    def __post_init__(self):
        for name in ("first_moment", "second_moment"):
            object.__setattr__(self, name, n._finite_tuple(
                getattr(self, name), n.PARAMETER_COUNT, name))
        if any(value < 0 for value in self.second_moment):
            raise ValueError("second_moment must be nonnegative")
        _count(self.updates, "updates", MAX_UPDATES)

    @classmethod
    def zero(cls):
        return cls((0.0,) * n.PARAMETER_COUNT, (0.0,) * n.PARAMETER_COUNT)


def adam_step(parameters, gradient, state):
    """Return new immutable parameters/state; reject the 301st update."""
    if not isinstance(parameters, n.NetworkParameters):
        raise TypeError("parameters must be existing NetworkParameters")
    if not isinstance(state, AdamState):
        raise TypeError("state must be AdamState")
    grad = n._finite_tuple(gradient, n.PARAMETER_COUNT, "gradient")
    if state.updates == MAX_UPDATES:
        raise ValueError("maximum of 300 updates already reached")
    k = state.updates + 1
    correction1, correction2 = 1.0 - BETA1 ** k, 1.0 - BETA2 ** k
    first, second, updated = [], [], []
    for i, (theta, gi, mi, si) in enumerate(zip(
            parameters.to_vector(), grad, state.first_moment, state.second_moment)):
        square = n._finite_result(gi * gi, f"gradient square[{i}]")
        mk = n._finite_result(BETA1 * mi + (1.0 - BETA1) * gi, f"first moment[{i}]")
        sk = n._finite_result(BETA2 * si + (1.0 - BETA2) * square, f"second moment[{i}]")
        mc = n._finite_result(mk / correction1, f"corrected first moment[{i}]")
        sc = n._finite_result(sk / correction2, f"corrected second moment[{i}]")
        denominator = n._finite_result(math.sqrt(sc) + EPSILON, f"Adam denominator[{i}]")
        value = n._finite_result(theta - ETA * mc / denominator, f"parameter[{i}]")
        first.append(mk)
        second.append(sk)
        updated.append(value)
    return n.NetworkParameters.from_vector(updated), AdamState(first, second, k)


def initialize_parameters():
    """Exactly 16 w draws, then 16 v draws; no global RNG mutation."""
    rng = random.Random(SEED)
    bound = math.sqrt(6.0 / 17.0)
    w = tuple(rng.uniform(-bound, bound) for _ in range(n.HIDDEN_UNITS))
    v = tuple(rng.uniform(-bound, bound) for _ in range(n.HIDDEN_UNITS))
    return n.NetworkParameters(w, (0.0,) * 16, v, 0.0)


def make_sites():
    width = n.B - n.A
    training = tuple(n.A + (j + 0.5) * width / 16 for j in range(16))
    validation = tuple(n.A + j * width / 16 for j in range(17))
    _partition(training, "training")
    _partition(validation, "validation")
    if set(training) & set(validation) or len(set(training + validation)) != 33:
        raise ValueError("pilot sites must be 33 distinct floats")
    return training, validation


def _partition(points, name):
    if isinstance(points, (str, bytes)):
        raise TypeError(f"{name} must contain real points")
    sites = tuple(points)
    if not sites:
        raise ValueError(f"{name} cannot be empty")
    sites = tuple(n._validate_p(p) for p in sites)
    if len(set(sites)) != len(sites):
        raise ValueError(f"{name} must have distinct points")
    return sites


@dataclass(frozen=True)
class SiteLabels:
    points: tuple[float, ...]
    values: tuple[float, ...]
    derivatives: tuple[float, ...]

    def __post_init__(self):
        object.__setattr__(self, "points", _partition(self.points, "label sites"))
        for name in ("values", "derivatives"):
            object.__setattr__(self, name, n._finite_tuple(
                getattr(self, name), len(self.points), name))

    def payload(self):
        return asdict(self)


def acquire_labels(training_points, validation_points, value_call, derivative_call,
                   *, endpoint_cache=None):
    """Evaluate uncached sites once; immutable objects out, caller cache unchanged."""
    training = _partition(training_points, "training")
    validation = _partition(validation_points, "validation")
    if set(training) & set(validation):
        raise ValueError("training and validation overlap")
    if not callable(value_call) or not callable(derivative_call):
        raise TypeError("label evaluators must be callable")
    supplied = dict(endpoint_cache or {})
    union = training + validation
    if any(p not in union for p in supplied):
        raise ValueError("cache contains a site outside these partitions")
    cache, reused = {}, []
    for p, pair in supplied.items():
        cache[n._validate_p(p)] = n._finite_tuple(pair, 2, "cached value/derivative")
    counts = {"value_calls": 0, "derivative_calls": 0, "cached_sites": 0}
    for p in union:
        if p in cache:
            counts["cached_sites"] += 1
            reused.append(p)
            continue
        value = n._finite_result(n._finite_scalar(value_call(p), "value label"), "value label")
        counts["value_calls"] += 1
        derivative = n._finite_result(n._finite_scalar(
            derivative_call(p), "derivative label"), "derivative label")
        counts["derivative_calls"] += 1
        cache[p] = (value, derivative)
    def labelled(points):
        return SiteLabels(points, tuple(cache[p][0] for p in points),
                          tuple(cache[p][1] for p in points))
    return labelled(training), labelled(validation), dict(
        counts, unique_sites=len(union), total_value_labels=len(union),
        total_derivative_labels=len(union), reused_sites=reused)


def saturation_diagnostic(network, points):
    """One activation-only pass/site; no new value/sensitivity evaluation."""
    rounded = near = 0
    maximum = 0.0
    for p in points:
        z, h = network._activations(p)
        rounded += sum(abs(x) == 1.0 for x in h)
        near += sum(abs(x) >= NEAR_SATURATION for x in h)
        for w, b in zip(network.parameters.w, network.parameters.b):
            maximum = max(maximum, abs(n._finite_result(w * z + b, "preactivation")))
    total = len(points) * n.HIDDEN_UNITS
    return dict(activation_count=total, rounded_count=rounded,
                near_count=near, near_threshold=NEAR_SATURATION,
                rounded_fraction=rounded / total, near_fraction=near / total,
                max_abs_preactivation=maximum, certified=False)


def _training_loss(method, network, labels, S0, S1):
    if method == "M1":
        return losses.m1_loss_and_gradient(
            network, labels.points, labels.values, S0=S0)
    return losses.m2_loss_and_gradient(
        network, labels.points, labels.values, labels.derivatives, S0=S0, S1=S1)


def train_method(method, initial_parameters, training, *, S0, S1,
                 updates=MAX_UPDATES):
    """Only training labels enter. Logging is extra; no validation argument."""
    if method not in ("M1", "M2"):
        raise ValueError("method must be M1 or M2")
    if not isinstance(initial_parameters, n.NetworkParameters):
        raise TypeError("initial_parameters must be NetworkParameters")
    if not isinstance(training, SiteLabels):
        raise TypeError("training must be SiteLabels")
    S0, S1 = _positive(S0, "S0"), _positive(S1, "S1")
    _count(updates, "requested updates", MAX_UPDATES)
    if updates == 0:
        raise ValueError("request at least one update")
    parameters = n.NetworkParameters.from_vector(initial_parameters.to_vector())
    network = n.NeuralSurrogate(parameters)
    state = AdamState.zero()
    controls, incident = [], None
    update_loss_calls = logging_loss_calls = 0
    stage, attempted_update = "initial_control", 0
    checkpoints = set(k for k in CONTROL_UPDATES if k <= updates) | {updates}

    def log_control():
        nonlocal logging_loss_calls
        result = _training_loss(method, network, training, S0, S1)
        logging_loss_calls += 1
        norm = n._finite_result(math.hypot(*result.gradient), "gradient norm")
        controls.append(dict(
            updates=state.updates, training_loss=result.loss,
            value_component=result.value_component,
            sensitivity_component=result.sensitivity_component,
            gradient_norm=norm,
            saturation=saturation_diagnostic(network, training.points)))

    try:
        log_control()
        for k in range(1, updates + 1):
            stage, attempted_update = "update_loss_and_gradient", k
            result = _training_loss(method, network, training, S0, S1)
            update_loss_calls += 1
            stage = "Adam_update"
            parameters, new_state = adam_step(network.parameters, result.gradient, state)
            network, state = n.NeuralSurrogate(parameters), new_state
            if k in checkpoints:
                stage = "control_logging"
                log_control()
    except (FloatingPointError, OverflowError) as exc:
        incident = dict(stage=stage, attempted_update=attempted_update,
                        completed_updates=state.updates, error_type=type(exc).__name__,
                        detail=str(exc), automatic_retry=False)
    vector = network.parameters.to_vector()
    return dict(method=method, status=("COMPLETED" if incident is None else "STOPPED_NUMERICAL_ERROR"),
                requested_updates=updates, completed_updates=state.updates,
                initial_parameter_id=canonical_id(initial_parameters.to_vector()),
                final_parameter_id=canonical_id(vector), final_parameters=vector,
                final_Adam_state=asdict(state), controls=controls,
                call_counts_from_control_flow={
                    "update_loss_and_gradient_completed": update_loss_calls,
                    "logging_loss_and_gradient_completed": logging_loss_calls,
                    "counts_are_control_flow_observations_not_kernel_instrumentation": True},
                incident=incident, validation_used_in_updates=False,
                oracle_queried_in_updates=False, checkpoint_selection=False)


def validation_diagnostics(parameters, labels, *, S0, S1):
    if not isinstance(labels, SiteLabels):
        raise TypeError("validation labels must be SiteLabels")
    S0, S1 = _positive(S0, "S0"), _positive(S1, "S1")
    network = n.NeuralSurrogate(parameters)
    predicted_values, predicted_derivatives, errors0, errors1 = [], [], [], []
    for p, target, derivative in zip(labels.points, labels.values, labels.derivatives):
        evaluation = network.evaluate(p)
        predicted_values.append(evaluation.value)
        predicted_derivatives.append(evaluation.input_derivative)
        errors0.append(n._finite_result(evaluation.value - target, "validation value error"))
        errors1.append(n._finite_result(
            evaluation.input_derivative - derivative, "validation derivative error"))
    root_count = math.sqrt(len(labels.points))
    rmse0 = n._finite_result(math.hypot(*errors0) / root_count, "value RMSE")
    rmse1 = n._finite_result(math.hypot(*errors1) / root_count, "derivative RMSE")
    return dict(points=labels.points, predicted_values=predicted_values,
                predicted_derivatives=predicted_derivatives,
                value_RMSE=rmse0, derivative_RMSE=rmse1,
                normalized_value_RMSE=n._finite_result(rmse0 / S0, "normalized value RMSE"),
                normalized_derivative_RMSE=n._finite_result(rmse1 / S1, "normalized derivative RMSE"),
                discrete_max_abs_value_error=max(map(abs, errors0)),
                discrete_max_abs_derivative_error=max(map(abs, errors1)),
                maxima_are_uniform_bounds=False, used_for_training_or_selection=False)
