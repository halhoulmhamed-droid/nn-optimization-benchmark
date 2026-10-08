"""Analytic 49-parameter gradients of the preserved scalar surrogate.

Order and parameter representation are those of src.neural_surrogate.
p and all four affine constants are fixed during parameter differentiation.
No optimiser, training, full Hessian or autonomous second p derivative.
"""
from __future__ import annotations

from dataclasses import dataclass

from src import neural_surrogate as n


@dataclass(frozen=True)
class ScalarParameterGradient:
    value: float
    gradient: tuple[float, ...]


@dataclass(frozen=True)
class ParameterEvaluation:
    forward: n.ForwardEvaluation
    value_gradient: tuple[float, ...]
    sensitivity_gradient: tuple[float, ...]


def _require_network(network: n.NeuralSurrogate) -> None:
    if not isinstance(network, n.NeuralSurrogate):
        raise TypeError("network must be the existing NeuralSurrogate")


def _value_gradient(
    network: n.NeuralSurrogate, z: float, h: tuple[float, ...],
) -> tuple[float, ...]:
    params, s = network.parameters, network.normalization.s_out
    q = tuple(1.0 - hi * hi for hi in h)
    result = (
        *(s * (vi * (z * qi)) for vi, qi in zip(params.v, q)),
        *(s * (vi * qi) for vi, qi in zip(params.v, q)),
        *(s * hi for hi in h),
        s,
    )
    return tuple(n._finite_result(value, f"value gradient[{i}]")
                 for i, value in enumerate(result))


def _sensitivity_gradient(
    network: n.NeuralSurrogate, z: float, h: tuple[float, ...],
) -> tuple[float, ...]:
    params, norm = network.parameters, network.normalization
    scale = n._finite_result(norm.s_out * norm.alpha, "s_out*alpha")
    q = tuple(1.0 - hi * hi for hi in h)
    factors = tuple(n._finite_result(1.0 - 2.0 * wi * z * hi, "mixed dw factor")
                    for wi, hi in zip(params.w, h))
    result = (
        *(scale * (vi * qi) * factor
          for vi, qi, factor in zip(params.v, q, factors)),
        *(-2.0 * scale * (vi * (wi * qi)) * hi
          for vi, wi, hi, qi in zip(params.v, params.w, h, q)),
        *(scale * (wi * qi) for wi, qi in zip(params.w, q)),
        0.0,
    )
    return tuple(n._finite_result(value, f"sensitivity gradient[{i}]")
                 for i, value in enumerate(result))


def value_and_parameter_gradient(
    network: n.NeuralSurrogate, p: float,
) -> ScalarParameterGradient:
    """Return physical temperature f and grad_theta f; no sensitivity computed."""
    _require_network(network)
    # Reuse the exact phase 1C scalar kernels and their guards.
    z, h = network._activations(p)
    _, value = network._value(h)
    return ScalarParameterGradient(value, _value_gradient(network, z, h))


def sensitivity_and_parameter_gradient(
    network: n.NeuralSurrogate, p: float,
) -> ScalarParameterGradient:
    """Return physical g=df/dp and grad_theta g (the authorised mixed derivative)."""
    _require_network(network)
    z, h = network._activations(p)
    _, sensitivity = network._derivative(h)
    return ScalarParameterGradient(
        sensitivity, _sensitivity_gradient(network, z, h))


def evaluate_parameter_gradients(
    network: n.NeuralSurrogate, p: float,
) -> ParameterEvaluation:
    """Share the 16 activations from public phase 1C evaluate, then two vectors."""
    _require_network(network)
    forward = network.evaluate(p)
    return ParameterEvaluation(
        forward,
        _value_gradient(network, forward.normalized_input, forward.activations),
        _sensitivity_gradient(network, forward.normalized_input, forward.activations),
    )
