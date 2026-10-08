"""Physical reference M0, with conditional error estimates and explicit cost.

The private scalar bisection helper is reused by independent unit controls.
It does not validate global monotonicity by a numerical grid. For the actual
thermal objective that property is proved in local phase 1A L01.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Callable

from src import thermal_oracle as thermal

DEFAULT_WIDTH_TOL = 1e-10
DEFAULT_MAX_ITERATIONS = 80


def kkt_residual_interval(p: float, gradient: float, a: float, b: float) -> float:
    """Exact boundary membership, with no tolerance-based snapping."""
    p = thermal.finite_scalar(p, "p")
    gradient = thermal.finite_scalar(gradient, "gradient")
    a = thermal.finite_scalar(a, "a")
    b = thermal.finite_scalar(b, "b")
    if not a < b or not a <= p <= b:
        raise ValueError("require a < b and p in [a, b]")
    if p == a:
        return max(0.0, -gradient)
    if p == b:
        return max(0.0, gradient)
    return abs(gradient)


@dataclass(frozen=True)
class _SearchState:
    p_ref: float
    lower: float
    upper: float
    g_lower: float
    g_upper: float
    branch: str
    iterations: int
    stopping_reason: str


def _bisect_monotone_gradient(
    gradient: Callable[[float], float],
    a: float,
    b: float,
    *,
    width_tol: float,
    max_iterations: int,
) -> _SearchState:
    """Assume a continuous strictly increasing gradient on [a,b].

    All sign-based statements are conditional on correct evaluations.
    A floating zero retains the pre-existing bracket rather than setting
    its width to zero. Iterations count midpoint gradient evaluations.
    """
    a = thermal.finite_scalar(a, "a")
    b = thermal.finite_scalar(b, "b")
    width_tol = thermal.finite_scalar(width_tol, "width_tol")
    if not a < b or not math.isfinite(b - a):
        raise ValueError("require finite positive interval width")
    if width_tol <= 0.0:
        raise ValueError("width_tol must be strictly positive")
    if isinstance(max_iterations, bool) or not isinstance(max_iterations, int):
        raise TypeError("max_iterations must be an integer")
    if max_iterations < 0:
        raise ValueError("max_iterations must be nonnegative")

    g_a = thermal.finite_scalar(gradient(a), "gradient(a)")
    g_b = thermal.finite_scalar(gradient(b), "gradient(b)")
    if g_a > g_b:
        raise ValueError("endpoint gradients violate the monotonicity assumption")
    if g_a >= 0.0:
        return _SearchState(a, a, a, g_a, g_a, "lower_boundary", 0,
                            "lower_boundary_sign")
    if g_b <= 0.0:
        return _SearchState(b, b, b, g_b, g_b, "upper_boundary", 0,
                            "upper_boundary_sign")

    lo, hi, g_lo, g_hi = a, b, g_a, g_b
    iterations = 0
    while True:
        mid = lo + (hi - lo) / 2.0
        if hi - lo <= width_tol:
            reason = "width_tolerance"
            break
        if iterations >= max_iterations:
            reason = "max_iterations"
            break
        if mid == lo or mid == hi:
            reason = "floating_midpoint_stagnation"
            break
        g_mid = thermal.finite_scalar(gradient(mid), "gradient(mid)")
        iterations += 1
        if g_mid == 0.0:
            reason = "floating_zero_gradient"
            break  # Keep lo,hi: float zero is not an exact root proof.
        if g_mid < 0.0:
            lo, g_lo = mid, g_mid
        else:
            hi, g_hi = mid, g_mid
    return _SearchState(mid, lo, hi, g_lo, g_hi, "interior",
                        iterations, reason)


class _PhysicalEvaluations:
    """Small per-solve cache of the fixed thermal F/F'/F''; no global cache."""
    def __init__(self) -> None:
        self._cache = {kind: {} for kind in ("value", "derivative", "curvature")}
        self._requests = {kind: 0 for kind in self._cache}

    def evaluate(self, kind: str, p: float) -> float:
        self._requests[kind] += 1
        cache = self._cache[kind]
        if p not in cache:
            function = {
                "value": thermal.temperature,
                "derivative": thermal.temperature_derivative,
                "curvature": thermal.temperature_second_derivative,
            }[kind]
            cache[p] = function(p)
        return cache[p]

    def counts(self) -> dict:
        by_kind = {}
        for kind, cache in self._cache.items():
            requested = self._requests[kind]
            executed = len(cache)
            by_kind[kind] = {
                "requested": requested,
                "executed": executed,
                "unique": executed,
                "reused": requested - executed,
            }
        return {
            "scope": "one M0 solve; unique keys=(kind,p); no inter-solve cache",
            "by_kind": by_kind,
            "total": {key: sum(item[key] for item in by_kind.values())
                      for key in ("requested", "executed", "unique", "reused")},
            "shared_preparation_included": False,
        }


@dataclass(frozen=True)
class ReferenceResult:
    lambda_value: float
    p_ref: float
    j_ref: float
    branch: str
    bracket: tuple[float, float]
    bracket_gradients: tuple[float, float]
    width: float
    iterations: int
    stopping_reason: str
    width_tolerance: float
    width_tolerance_met: bool
    max_iterations: int
    gradient_at_reference: float
    kkt_residual: float
    exact_midpoint_distance_bound_conditional: float
    rounded_midpoint_radius_estimate: float
    reference_value_error_estimate_conditional: float
    evaluations: dict
    numerical_status: str = "FLOATING_REFERENCE_NOT_CERTIFIED"
    bound_condition: str = (
        "correct exact-model branch/signs, enclosing bracket, valid curvature; "
        "floating arithmetic and libm errors are not rigorously bounded"
    )

    def to_dict(self) -> dict:
        return asdict(self)


def solve_thermal_reference(
    lambda_: float,
    *,
    width_tol: float = DEFAULT_WIDTH_TOL,
    max_iterations: int = DEFAULT_MAX_ITERATIONS,
) -> ReferenceResult:
    """Return M0; the protocol lambda domain is checked, not invented."""
    lambda_ = thermal.validate_lambda(lambda_)
    evaluations = _PhysicalEvaluations()

    def gradient(p: float) -> float:
        return evaluations.evaluate("derivative", p) + lambda_

    state = _bisect_monotone_gradient(
        gradient, thermal.A, thermal.B,
        width_tol=width_tol, max_iterations=max_iterations,
    )
    width = state.upper - state.lower
    radius = max(state.p_ref - state.lower, state.upper - state.p_ref)
    g_ref = gradient(state.p_ref)  # Cache reuse is counted when applicable.
    j_ref = evaluations.evaluate("value", state.p_ref) + lambda_ * state.p_ref
    beta = 0.5 * thermal.MAX_CURVATURE * radius ** 2
    return ReferenceResult(
        lambda_value=lambda_, p_ref=state.p_ref, j_ref=j_ref,
        branch=state.branch, bracket=(state.lower, state.upper),
        bracket_gradients=(state.g_lower, state.g_upper), width=width,
        iterations=state.iterations, stopping_reason=state.stopping_reason,
        width_tolerance=float(width_tol),
        width_tolerance_met=width <= float(width_tol),
        max_iterations=max_iterations, gradient_at_reference=g_ref,
        kkt_residual=kkt_residual_interval(
            state.p_ref, g_ref, thermal.A, thermal.B),
        exact_midpoint_distance_bound_conditional=width / 2.0,
        rounded_midpoint_radius_estimate=radius,
        reference_value_error_estimate_conditional=beta,
        evaluations=evaluations.counts(),
    )


def regret_interval(
    raw_difference: float,
    reference_error_bound: float,
    *,
    decision_value_error_bound: float = 0.0,
    reference_value_error_bound: float = 0.0,
) -> dict:
    """Conditional interval; caller must justify every supplied error bound.

    The raw difference is preserved even if negative. Zero defaults mean an
    explicit assumption of exact value evaluations, not measured roundoff.
    Inconsistent assumptions produce an empty intersection, reported as such.
    """
    raw = thermal.finite_scalar(raw_difference, "raw_difference")
    beta = thermal.finite_scalar(reference_error_bound, "reference_error_bound")
    err_p = thermal.finite_scalar(decision_value_error_bound, "decision error")
    err_ref = thermal.finite_scalar(reference_value_error_bound, "reference error")
    if min(beta, err_p, err_ref) < 0.0:
        raise ValueError("error bounds must be nonnegative")
    error = err_p + err_ref
    upper = raw + error + beta
    lower = max(0.0, raw - error)
    compatible = lower <= upper
    return {
        "raw_difference": raw,
        "raw_difference_negative": raw < 0.0,
        "reference_suboptimality_bound": beta,
        "value_evaluation_error_sum": error,
        "lower": lower,
        "upper": upper,
        "compatible": compatible,
        "interval": [lower, upper] if compatible else None,
        "status": "CONDITIONAL_ON_SUPPLIED_BOUNDS_NOT_CERTIFIED",
        "anomaly": None if compatible else "empty interval: assumptions inconsistent",
    }

