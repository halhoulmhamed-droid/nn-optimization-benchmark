"""Fixed scalar 1-16-1 tanh surrogate: forward value and physical input derivative.

The 49 trainable numbers have order w[0:16], b[0:16], v[0:16], d.
Normalisation constants are fixed, separate information, not parameters.
Default output is the dimensionless temperature, without centring/scaling.
No oracle import, training, parameter gradients, second derivative or solver.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Real
from typing import Iterable

A = 0.02
B = 0.20
HIDDEN_UNITS = 16
PARAMETER_COUNT = 49
INPUT_ALPHA = 2.0 / (B - A)
INPUT_BETA = -1.0 - INPUT_ALPHA * A
PARAMETER_ORDER = ("w[0:16]", "b[0:16]", "v[0:16]", "d")


def _finite_scalar(value: Real, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real scalar, not a boolean")
    try:
        result = float(value)
    except (OverflowError, ValueError) as exc:
        raise ValueError(f"{name} must have a finite float representation") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _finite_tuple(values: Iterable[Real], size: int, name: str) -> tuple[float, ...]:
    if isinstance(values, (str, bytes)):
        raise TypeError(f"{name} must be an iterable of real scalars")
    try:
        items = tuple(values)
    except TypeError as exc:
        raise TypeError(f"{name} must be an iterable of real scalars") from exc
    if len(items) != size:
        raise ValueError(f"{name} must contain exactly {size} scalars")
    return tuple(_finite_scalar(item, f"{name}[{i}]")
                 for i, item in enumerate(items))


def _finite_result(value: float, name: str) -> float:
    if not math.isfinite(value):
        raise FloatingPointError(f"non-finite computed {name}")
    return value


def _finite_sum(terms: Iterable[float], name: str) -> float:
    try:
        result = math.fsum(terms)
    except (OverflowError, ValueError) as exc:
        raise FloatingPointError(f"non-finite computed {name}") from exc
    return _finite_result(result, name)


def _validate_p(p: Real) -> float:
    value = _finite_scalar(p, "p")
    if not A <= value <= B:
        raise ValueError(f"p must be in [{A}, {B}]")
    return value


@dataclass(frozen=True)
class NetworkParameters:
    """Immutable finite parameters; d is c0 in phase 1A L02."""
    w: tuple[float, ...]
    b: tuple[float, ...]
    v: tuple[float, ...]
    d: float

    def __post_init__(self) -> None:
        for name in ("w", "b", "v"):
            object.__setattr__(self, name, _finite_tuple(
                getattr(self, name), HIDDEN_UNITS, name))
        object.__setattr__(self, "d", _finite_scalar(self.d, "d"))

    def to_vector(self) -> tuple[float, ...]:
        return self.w + self.b + self.v + (self.d,)

    @classmethod
    def from_vector(cls, vector: Iterable[Real]) -> NetworkParameters:
        values = _finite_tuple(vector, PARAMETER_COUNT, "parameter vector")
        return cls(values[:16], values[16:32], values[32:48], values[48])


@dataclass(frozen=True)
class AffineNormalization:
    """Fixed N(p)=alpha*p+beta, T_hat=c_out+s_out*y, on the same physical P.

    Non-default affine constants support explicit test fixtures. They are
    never estimated from data or included in the 49-parameter vector.
    The default is algebraically 2*(p-a)/(b-a)-1; float operation order is
    alpha*p+beta, so endpoint/midpoint rounding is not snapped away.
    """
    alpha: float = INPUT_ALPHA
    beta: float = INPUT_BETA
    c_out: float = 0.0
    s_out: float = 1.0

    def __post_init__(self) -> None:
        for name in ("alpha", "beta", "c_out", "s_out"):
            object.__setattr__(self, name, _finite_scalar(getattr(self, name), name))

    def normalize_input(self, p: Real) -> float:
        p = _validate_p(p)  # Domain checked before affine arithmetic.
        return _finite_result(self.alpha * p + self.beta, "normalised input")


DEFAULT_NORMALIZATION = AffineNormalization()


@dataclass(frozen=True)
class ForwardEvaluation:
    """One activation pass, with value and both input-derivative conventions."""
    normalized_input: float
    network_value: float
    value: float
    normalized_derivative: float  # dy/dz, before output scaling.
    input_derivative: float       # dT_hat/dp = s_out*alpha*dy/dz.
    activations: tuple[float, ...]


@dataclass(frozen=True)
class NeuralSurrogate:
    parameters: NetworkParameters
    normalization: AffineNormalization = DEFAULT_NORMALIZATION

    def __post_init__(self) -> None:
        if not isinstance(self.parameters, NetworkParameters):
            raise TypeError("parameters must be NetworkParameters")
        if not isinstance(self.normalization, AffineNormalization):
            raise TypeError("normalization must be AffineNormalization")

    def _activations(self, p: Real) -> tuple[float, tuple[float, ...]]:
        z = self.normalization.normalize_input(p)
        h = tuple(math.tanh(_finite_result(w * z + b, "preactivation"))
                  for w, b in zip(self.parameters.w, self.parameters.b))
        return z, h

    def _value(self, h: tuple[float, ...]) -> tuple[float, float]:
        y = _finite_sum((self.parameters.d,
                         *(v * hi for v, hi in zip(self.parameters.v, h))),
                        "linear network value")
        value = _finite_result(self.normalization.c_out
                               + self.normalization.s_out * y, "output")
        return y, value

    def _derivative(self, h: tuple[float, ...]) -> tuple[float, float]:
        # Multiply w by the finite tanh factor first. No clipping of h or p.
        terms = (_finite_result(v * (w * (1.0 - hi * hi)), "derivative term")
                 for v, w, hi in zip(self.parameters.v, self.parameters.w, h))
        dy_dz = _finite_sum(terms, "normalised input derivative")
        physical = _finite_result(
            self.normalization.s_out * (self.normalization.alpha * dy_dz),
            "physical input derivative")
        return dy_dz, physical

    def forward(self, p: Real) -> float:
        """Return T_hat(p); validate p before all network arithmetic."""
        _, h = self._activations(p)
        return self._value(h)[1]

    def input_derivative(self, p: Real) -> float:
        """Return dT_hat/dp, not dy/dz; no gradient with respect to weights."""
        _, h = self._activations(p)
        return self._derivative(h)[1]

    def evaluate(self, p: Real) -> ForwardEvaluation:
        """Compute value and derivative together, sharing 16 tanh evaluations."""
        z, h = self._activations(p)
        y, value = self._value(h)
        dy_dz, physical = self._derivative(h)
        return ForwardEvaluation(z, y, value, dy_dz, physical, h)
