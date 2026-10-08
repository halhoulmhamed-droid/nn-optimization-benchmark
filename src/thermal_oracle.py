"""Scalar, dimensionless thermal oracle from the preserved phase 1A documents.

Only p=s in [0.02, 0.20] and the protocol's closed lambda interval are
accepted. Floating evaluation is not a rigorous numerical certificate.
No third-party dependency and no neural-network code.
"""
from __future__ import annotations

import math
from numbers import Real

A = 0.02
B = 0.20
FIRST_AMPLITUDE = 1.0 / math.sqrt(2.0)
SECOND_AMPLITUDE = 0.2
FIRST_RATE = math.pi ** 2
SECOND_RATE = 4.0 * math.pi ** 2


def finite_scalar(value: Real, name: str) -> float:
    """Reject booleans, non-real scalars and non-finite representations."""
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real scalar, not a boolean")
    try:
        result = float(value)
    except (OverflowError, ValueError) as exc:
        raise ValueError(f"{name} must have a finite float representation") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def validate_time(p: Real) -> float:
    result = finite_scalar(p, "p")
    if not A <= result <= B:
        raise ValueError(f"p must be in [{A}, {B}]")
    return result


def _value_unchecked(p: float) -> float:
    return math.fsum((
        FIRST_AMPLITUDE * math.exp(-FIRST_RATE * p),
        SECOND_AMPLITUDE * math.exp(-SECOND_RATE * p),
    ))


def _derivative_unchecked(p: float) -> float:
    return -math.fsum((
        FIRST_AMPLITUDE * FIRST_RATE * math.exp(-FIRST_RATE * p),
        SECOND_AMPLITUDE * SECOND_RATE * math.exp(-SECOND_RATE * p),
    ))


def _curvature_unchecked(p: float) -> float:
    return math.fsum((
        FIRST_AMPLITUDE * FIRST_RATE ** 2 * math.exp(-FIRST_RATE * p),
        SECOND_AMPLITUDE * SECOND_RATE ** 2 * math.exp(-SECOND_RATE * p),
    ))


# Six analytic evaluations per module import, shared preparation information.
VALUE_A = _value_unchecked(A)
VALUE_B = _value_unchecked(B)
DERIVATIVE_A = _derivative_unchecked(A)
DERIVATIVE_B = _derivative_unchecked(B)
CURVATURE_A = _curvature_unchecked(A)
CURVATURE_B = _curvature_unchecked(B)
CRITICAL_LAMBDA_A = -DERIVATIVE_A
CRITICAL_LAMBDA_B = -DERIVATIVE_B
LAMBDA_MIN = CRITICAL_LAMBDA_B / 2.0
LAMBDA_MAX = 3.0 * CRITICAL_LAMBDA_A / 2.0
S0 = VALUE_A - VALUE_B
S1 = -DERIVATIVE_A
MIN_CURVATURE = CURVATURE_B
MAX_CURVATURE = CURVATURE_A


def validate_lambda(lambda_: Real) -> float:
    result = finite_scalar(lambda_, "lambda")
    if not LAMBDA_MIN <= result <= LAMBDA_MAX:
        raise ValueError(f"lambda must be in [{LAMBDA_MIN}, {LAMBDA_MAX}]")
    return result


def temperature(p: Real) -> float:
    """T(p)=F(p)=exp(-pi^2 p)/sqrt(2)+0.2 exp(-4 pi^2 p)."""
    return _value_unchecked(validate_time(p))


def temperature_derivative(p: Real) -> float:
    """Derivative in physical, dimensionless p; no input normalisation."""
    return _derivative_unchecked(validate_time(p))


def temperature_second_derivative(p: Real) -> float:
    """Positive curvature of T; its global bound is proved in local L01."""
    return _curvature_unchecked(validate_time(p))


# Actual aliases, so F and T do not denote different models.
F = T = temperature
F_prime = T_prime = temperature_derivative
F_second = T_second = temperature_second_derivative


def objective(p: Real, lambda_: Real) -> float:
    p = validate_time(p)
    lambda_ = validate_lambda(lambda_)
    return temperature(p) + lambda_ * p


def objective_gradient(p: Real, lambda_: Real) -> float:
    p = validate_time(p)
    lambda_ = validate_lambda(lambda_)
    return temperature_derivative(p) + lambda_


def objective_hessian(p: Real, lambda_: Real) -> float:
    p = validate_time(p)
    validate_lambda(lambda_)  # Contract applies even though lambda cancels.
    return temperature_second_derivative(p)


def model_constants() -> dict:
    """Return observed floats, with analytic provenance, not certified bounds."""
    return {
        "units": "dimensionless",
        "time_domain": [A, B],
        "sensor_x": 0.25,
        "diffusivity": 1.0,
        "amplitudes": [FIRST_AMPLITUDE, SECOND_AMPLITUDE],
        "rates": [FIRST_RATE, SECOND_RATE],
        "critical_lambda_a": CRITICAL_LAMBDA_A,
        "critical_lambda_b": CRITICAL_LAMBDA_B,
        "lambda_domain": [LAMBDA_MIN, LAMBDA_MAX],
        "S0": S0, "S1": S1,
        "m_float": MIN_CURVATURE,
        "L_phys_float": MAX_CURVATURE,
        "curvature_proof": "docs/math/lemmas_and_counterexamples.md#l01",
        "certified": False,
        "preparation_evaluations_per_import": {
            "value": 2, "derivative": 2, "curvature": 2,
            "executed_total": 6,
            "scope": "shared module preparation; not included in each M0 result",
        },
    }

