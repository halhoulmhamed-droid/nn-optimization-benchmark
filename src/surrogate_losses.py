"""Exact sourced M1/M2 mean-square losses, with targets supplied by the caller.

M1/M2 are methods; L1/L2 are their training objectives. M2 uses common
value/sensitivity sites, two fixed positive scales and coefficient 1.
No half factor, point weighting, lambda, weight decay or oracle query.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from src import neural_surrogate as n
from src import neural_gradients as ng


@dataclass(frozen=True)
class LossEvaluation:
    loss: float
    gradient: tuple[float, ...]
    value_component: float
    sensitivity_component: float
    n_values: int
    n_sensitivities: int
    method: str


@dataclass(frozen=True)
class QuadraticEvaluation:
    loss: float
    gradient: tuple[float, ...]


def _positive_scale(value: float, name: str) -> float:
    scale = n._finite_scalar(value, name)
    if scale <= 0.0:
        raise ValueError(f"{name} must be strictly positive")
    return scale


def _fixed_tuple(values: Iterable[float], name: str) -> tuple[float, ...]:
    if isinstance(values, (str, bytes)):
        raise TypeError(f"{name} must be an iterable of real scalars")
    try:
        items = tuple(values)
    except TypeError as exc:
        raise TypeError(f"{name} must be an iterable of real scalars") from exc
    return tuple(n._finite_scalar(item, f"{name}[{i}]")
                 for i, item in enumerate(items))


def _targets(
    network: n.NeuralSurrogate, points: Iterable[float],
    value_targets: Iterable[float],
    sensitivity_targets: Iterable[float] | None = None,
) -> tuple[tuple[float, ...], tuple[float, ...], tuple[float, ...] | None]:
    ng._require_network(network)
    sites = _fixed_tuple(points, "points")
    values = _fixed_tuple(value_targets, "value_targets")
    if not sites or len(sites) != len(values):
        raise ValueError("require nonempty points and equally sized value targets")
    for p in sites:
        n._validate_p(p)  # Validate the whole control tuple before evaluating.
    sensitivities = None
    if sensitivity_targets is not None:
        sensitivities = _fixed_tuple(sensitivity_targets, "sensitivity_targets")
        if len(sensitivities) != len(sites):
            raise ValueError("M2 requires one physical sensitivity target per common site")
    return sites, values, sensitivities


def _normalised_square(value: float, target: float, scale: float) -> tuple[float, float]:
    residual = n._finite_result(value - target, "residual")
    normalised = n._finite_result(residual / scale, "normalised residual")
    square = n._finite_result(normalised * normalised, "squared residual")
    return square, normalised


def _normalised_term(
    value: float, target: float, gradient: tuple[float, ...], scale: float,
) -> tuple[float, tuple[float, ...]]:
    square, normalised = _normalised_square(value, target, scale)
    factor = n._finite_result((2.0 * normalised) / scale, "loss gradient factor")
    vector = tuple(n._finite_result(factor * component, f"loss term gradient[{i}]")
                   for i, component in enumerate(gradient))
    return square, vector


def _mean(values: Iterable[float], count: int, name: str) -> float:
    return n._finite_sum((value / count for value in values), name)


def _mean_gradient(vectors: list[tuple[float, ...]], count: int) -> tuple[float, ...]:
    return tuple(_mean((vector[i] for vector in vectors), count,
                       f"mean loss gradient[{i}]")
                 for i in range(n.PARAMETER_COUNT))


def m1_loss_value(network, points, value_targets, *, S0: float) -> float:
    S0 = _positive_scale(S0, "S0")
    sites, targets, _ = _targets(network, points, value_targets)
    return _mean((_normalised_square(network.forward(p), target, S0)[0]
                  for p, target in zip(sites, targets)), len(sites), "L1")


def m2_loss_value(network, points, value_targets, sensitivity_targets, *,
                  S0: float, S1: float) -> float:
    S0, S1 = _positive_scale(S0, "S0"), _positive_scale(S1, "S1")
    if sensitivity_targets is None:
        raise ValueError("M2 requires physical sensitivity targets")
    sites, targets, sensitivities = _targets(
        network, points, value_targets, sensitivity_targets)
    terms = []
    for p, target, sensitivity_target in zip(sites, targets, sensitivities):
        forward = network.evaluate(p)
        value = _normalised_square(forward.value, target, S0)[0]
        sensitivity = _normalised_square(forward.input_derivative,
                                          sensitivity_target, S1)[0]
        terms.append(n._finite_sum((value, sensitivity), "M2 point loss"))
    return _mean(terms, len(sites), "L2")


def m1_loss_and_gradient(network, points, value_targets, *, S0: float) -> LossEvaluation:
    S0 = _positive_scale(S0, "S0")
    sites, targets, _ = _targets(network, points, value_targets)
    losses, vectors = [], []
    for p, target in zip(sites, targets):
        result = ng.value_and_parameter_gradient(network, p)
        loss, gradient = _normalised_term(result.value, target, result.gradient, S0)
        losses.append(loss)
        vectors.append(gradient)
    component = _mean(losses, len(sites), "L1")
    return LossEvaluation(component, _mean_gradient(vectors, len(sites)),
                          component, 0.0, len(sites), 0, "M1")


def m2_loss_and_gradient(network, points, value_targets, sensitivity_targets, *,
                         S0: float, S1: float) -> LossEvaluation:
    S0, S1 = _positive_scale(S0, "S0"), _positive_scale(S1, "S1")
    if sensitivity_targets is None:
        raise ValueError("M2 requires physical sensitivity targets")
    sites, targets, sensitivities = _targets(
        network, points, value_targets, sensitivity_targets)
    values, sensitivities_losses, vectors = [], [], []
    for p, target, sensitivity_target in zip(sites, targets, sensitivities):
        result = ng.evaluate_parameter_gradients(network, p)
        value, grad_value = _normalised_term(
            result.forward.value, target, result.value_gradient, S0)
        sensitivity, grad_sensitivity = _normalised_term(
            result.forward.input_derivative, sensitivity_target,
            result.sensitivity_gradient, S1)
        values.append(value)
        sensitivities_losses.append(sensitivity)
        vectors.append(tuple(n._finite_sum((left, right), f"M2 point gradient[{i}]")
                             for i, (left, right) in
                             enumerate(zip(grad_value, grad_sensitivity))))
    value_component = _mean(values, len(sites), "M2 value component")
    sensitivity_component = _mean(sensitivities_losses, len(sites),
                                  "M2 sensitivity component")
    return LossEvaluation(
        n._finite_sum((value_component, sensitivity_component), "L2"),
        _mean_gradient(vectors, len(sites)), value_component,
        sensitivity_component, len(sites), len(sites), "M2")


def quadratic_error_and_gradient(
    f: float, g: float, target: float, sensitivity_target: float,
    grad_f: Iterable[float], grad_g: Iterable[float], *, A: float, B: float,
) -> QuadraticEvaluation:
    """Algebraic control primitive; A/B are not configurable M1/M2 weights.

    Nonnegative coefficients are allowed to be zero. Even inactive inputs
    and both 49-vectors must be finite; inactive residual arithmetic is skipped.
    """
    f, g = n._finite_scalar(f, "f"), n._finite_scalar(g, "g")
    target = n._finite_scalar(target, "target")
    sensitivity_target = n._finite_scalar(sensitivity_target, "sensitivity_target")
    A, B = n._finite_scalar(A, "A"), n._finite_scalar(B, "B")
    if min(A, B) < 0.0:
        raise ValueError("quadratic coefficients must be nonnegative")
    vf = n._finite_tuple(grad_f, n.PARAMETER_COUNT, "grad_f")
    vg = n._finite_tuple(grad_g, n.PARAMETER_COUNT, "grad_g")
    losses, vectors = [], []
    for coefficient, value, wanted, vector in (
        (A, f, target, vf), (B, g, sensitivity_target, vg),
    ):
        if coefficient == 0.0:
            continue
        residual = n._finite_result(value - wanted, "quadratic residual")
        losses.append(n._finite_result(coefficient * residual * residual,
                                       "quadratic loss term"))
        factor = n._finite_result(2.0 * coefficient * residual, "quadratic gradient factor")
        vectors.append(tuple(n._finite_result(factor * entry, f"quadratic gradient[{i}]")
                             for i, entry in enumerate(vector)))
    return QuadraticEvaluation(
        n._finite_sum(losses, "quadratic loss"),
        tuple(n._finite_sum((vector[i] for vector in vectors), f"quadratic gradient[{i}]")
              for i in range(n.PARAMETER_COUNT)),
    )
