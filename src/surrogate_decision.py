"""Bounded global grid selection from cached surrogate values, without an oracle.

All numerical bounds here are float estimates of real-arithmetic formulae.
No training, local refinement, root search or numerical certification.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import math
from numbers import Real
from typing import Callable, Iterable

from src.neural_surrogate import NeuralSurrogate

MAX_INTERVALS = 1024
TANH_SECOND_MAX_FLOAT = 4.0 / (3.0 * math.sqrt(3.0))


def _finite(value: Real, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real scalar, not bool")
    try:
        result = float(value)
    except (OverflowError, ValueError) as exc:
        raise ValueError(f"{name} has no finite float representation") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _computed(value: float, name: str) -> float:
    if not math.isfinite(value):
        raise FloatingPointError(f"nonfinite computed {name}")
    return value


def _scalars(values: Iterable[Real], name: str) -> tuple[float, ...]:
    if isinstance(values, (str, bytes)):
        raise TypeError(f"{name} must be an iterable of real scalars")
    return tuple(_finite(v, f"{name}[{i}]") for i, v in enumerate(values))


@dataclass(frozen=True)
class DecisionGrid:
    a: float
    b: float
    points: tuple[float, ...]
    h_max: float = field(init=False)

    def __post_init__(self) -> None:
        a, b = _finite(self.a, "a"), _finite(self.b, "b")
        points = _scalars(self.points, "points")
        if not a < b or not math.isfinite(b - a):
            raise ValueError("require finite positive interval width")
        if not 2 <= len(points) <= MAX_INTERVALS + 1:
            raise ValueError("grid requires 1..1024 intervals")
        if points[0] != a or points[-1] != b:
            raise ValueError("grid endpoints must equal a,b exactly")
        if any(not a <= p <= b for p in points):
            raise ValueError("grid point outside [a,b]")
        widths = tuple(y - x for x, y in zip(points, points[1:]))
        if any(not math.isfinite(w) or w <= 0.0 for w in widths):
            raise ValueError("grid must be strictly increasing")
        object.__setattr__(self, "a", a)
        object.__setattr__(self, "b", b)
        object.__setattr__(self, "points", points)
        object.__setattr__(self, "h_max", max(widths))

    @property
    def intervals(self) -> int:
        return len(self.points) - 1


def make_grid(a: Real, b: Real, intervals: int = MAX_INTERVALS) -> DecisionGrid:
    a, b = _finite(a, "a"), _finite(b, "b")
    if isinstance(intervals, bool) or not isinstance(intervals, int):
        raise TypeError("intervals must be an integer")
    if not 1 <= intervals <= MAX_INTERVALS:
        raise ValueError("intervals must lie in 1..1024")
    if not a < b or not math.isfinite(b - a):
        raise ValueError("require finite positive interval width")
    # No clipping: a repeated/out-of-domain interior float is an invalid grid.
    points = (a,) + tuple(a + (b - a) * j / intervals
                         for j in range(1, intervals)) + (b,)
    return DecisionGrid(a, b, points)


@dataclass(frozen=True)
class PredictionCache:
    grid: DecisionGrid
    values: tuple[float, ...]
    source: str = "surrogate_forward"
    prediction_evaluations: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.grid, DecisionGrid):
            raise TypeError("grid must be DecisionGrid")
        values = _scalars(self.values, "predictions")
        if len(values) != len(self.grid.points):
            raise ValueError("one prediction is required per grid point")
        if not isinstance(self.source, str) or not self.source:
            raise ValueError("source must be a nonempty string")
        if (isinstance(self.prediction_evaluations, bool)
                or not isinstance(self.prediction_evaluations, int)):
            raise TypeError("prediction_evaluations must be an integer")
        if self.prediction_evaluations not in (0, len(values)):
            raise ValueError("cache count must be 0 (supplied) or the grid size")
        object.__setattr__(self, "values", values)


def cache_predictions(
    predictor: Callable[[float], float],
    grid: DecisionGrid,
    *,
    source: str = "surrogate_forward",
) -> PredictionCache:
    if not callable(predictor):
        raise TypeError("predictor must be callable")
    if not isinstance(grid, DecisionGrid):
        raise TypeError("grid must be DecisionGrid")
    if not isinstance(source, str) or not source:
        raise ValueError("source must be a nonempty string")
    values = tuple(_finite(predictor(p), "prediction") for p in grid.points)
    return PredictionCache(grid, values, source, len(values))


def grid_value_error(curvature_bound: Real, h_max: Real) -> float:
    """B*h_max^2/8 in floats; NOT a rigorously rounded upper bound."""
    curvature_bound = _finite(curvature_bound, "curvature_bound")
    h_max = _finite(h_max, "h_max")
    if curvature_bound < 0.0 or h_max <= 0.0:
        raise ValueError("require B >= 0 and h_max > 0")
    if curvature_bound == 0.0:
        return 0.0
    # Detect overflow rather than silently declaring an infinite bound.
    square = _computed(h_max * h_max, "grid-width square")
    return _computed(curvature_bound * square / 8.0, "grid value error")


def estimate_network_grid_error(
    model: NeuralSurrogate, grid: DecisionGrid,
) -> dict:
    """Real global curvature formula specialised to the saved network.

    alpha_norm is a slope; alpha_grid is an objective-suboptimality estimate.
    Rounded zero/underflow is not a certificate of exactly zero curvature.
    """
    if not isinstance(model, NeuralSurrogate):
        raise TypeError("model must be NeuralSurrogate")
    if not isinstance(grid, DecisionGrid):
        raise TypeError("grid must be DecisionGrid")
    n = model.normalization
    terms = tuple(_computed(abs(v) * (w * w), "curvature term")
                  for v, w in zip(model.parameters.v, model.parameters.w))
    try:
        total = _computed(math.fsum(terms), "curvature sum")
    except (OverflowError, ValueError) as exc:
        raise FloatingPointError("nonfinite curvature sum") from exc
    slope_square = _computed(n.alpha * n.alpha, "normalisation slope square")
    bound = _computed(
        TANH_SECOND_MAX_FLOAT * abs(n.s_out) * slope_square * total,
        "network curvature bound")
    affine_by_exact_zero_factors = (
        n.alpha == 0.0 or n.s_out == 0.0
        or all(v == 0.0 or w == 0.0
               for v, w in zip(model.parameters.v, model.parameters.w)))
    return {
        "B_hat_float": bound,
        "alpha_norm": n.alpha,
        "s_out": n.s_out,
        "sum_abs_v_w_squared_float": total,
        "tanh_second_max_float": TANH_SECOND_MAX_FLOAT,
        "h_max_float": grid.h_max,
        "alpha_grid_float": grid_value_error(bound, grid.h_max),
        "mathematical_curvature": (
            "4/(3*sqrt(3))*abs(s_out)*alpha_norm^2*sum(abs(v_i)*w_i^2)"),
        "mathematical_grid_error": "B_hat*h_max^2/8",
        "exact_affine_from_zero_factors": affine_by_exact_zero_factors,
        "rounded_zero_without_affine_proof": (
            bound == 0.0 and not affine_by_exact_zero_factors),
        "e_hat_rigorous_bound": None,
        "grid_plus_comparison_error": "alpha_grid+2*e_hat if e_hat known",
        "position_error_bound": None,
        "certified": False,
        "status": "CONDITIONAL_REAL_BOUND_WITH_UNVALIDATED_FLOAT_ESTIMATE",
        "conditions": [
            "finite fixed real network parameters and affine constants",
            "true maximal spacing of the candidate grid",
            "rigorous rounding bound for B_hat and alpha_grid not supplied",
            "objective evaluation/comparison errors not rigorously bounded",
        ],
    }


@dataclass(frozen=True)
class GridDecision:
    p_hat: float
    index: int
    predicted_temperature: float
    predicted_objective: float
    lambda_value: float
    candidate_origin: str
    prediction_source: str
    grid_size: int
    grid_intervals: int
    h_max: float
    objective_evaluations: int
    network_evaluations_in_selection: int = 0
    physical_evaluations_in_selection: int = 0
    floating_grid_optimum: bool = True
    continuous_global_minimum_claimed: bool = False
    optimization_error_status: str = (
        "grid error conditional on separate curvature bound and unknown "
        "floating comparison errors"
    )

    def to_dict(self) -> dict:
        return asdict(self)


def select_grid_minimum(cache: PredictionCache, lambda_value: Real) -> GridDecision:
    """Only cached network values, p and lambda; ties use the smallest p."""
    if not isinstance(cache, PredictionCache):
        raise TypeError("cache must be PredictionCache")
    lambda_value = _finite(lambda_value, "lambda_value")
    best_index, best_value = 0, None
    for i, (p, prediction) in enumerate(zip(cache.grid.points, cache.values)):
        candidate = _computed(prediction + lambda_value * p,
                              "discrete objective")
        if best_value is None or candidate < best_value:  # Strict: keep ties.
            best_index, best_value = i, candidate
    origin = ("lower_boundary" if best_index == 0 else
              "upper_boundary" if best_index == len(cache.values) - 1 else
              "interior_grid")
    return GridDecision(
        cache.grid.points[best_index], best_index, cache.values[best_index],
        best_value, lambda_value, origin, cache.source, len(cache.values),
        cache.grid.intervals, cache.grid.h_max, len(cache.values),
    )


def kkt_residual_interval(p: Real, gradient: Real, a: Real, b: Real) -> float:
    """Generic exact-membership KKT residual, matching phase 1A/physical M0."""
    p, gradient = _finite(p, "p"), _finite(gradient, "gradient")
    a, b = _finite(a, "a"), _finite(b, "b")
    if not a < b or not a <= p <= b:
        raise ValueError("require a < b and p in [a,b]")
    if p == a:
        return max(0.0, -gradient)
    if p == b:
        return max(0.0, gradient)
    return abs(gradient)

