"""Forgetting curves beyond a single exponential, written as sums of exponentials.

Generative Agents decay recency exponentially from the last access. The rational analysis of
memory (Anderson and Schooler 1991), on which ACT-R's base-level learning rests, finds instead
that the odds that a memory will be needed fall as a power of the time since each use and add
up over uses:

    D_i(T) = sum_j phi(T - t_ij)            phi(age) = (1 + age / h0) ** -d    (d = 0.5 in ACT-R)

where t_i1 < t_i2 < ... are the times at which memory i was created or retrieved. phi is
completely monotone, so by Bernstein's theorem it is a mixture of exponentials, and a few of
them approximate it closely on any bounded range of ages (Beylkin and Monzon 2005). With

    phi(age) ~= sum_r w_r exp(-lam_r age),      w_r >= 0,

every term is separable in (T, t), so for any anchor time tau

    D_i(T) ~= sum_r exp(-lam_r (T - tau)) * w_r sum_j exp(-lam_r (tau - t_ij)) = <g(T), z_i>,

an inner product between a query-side vector g (R entries) and a memory-side vector z_i. Each
use of a memory adds one term to z_i, so frequency costs nothing extra. A memory needs only

    y_ir = sum_j exp(-lam_r (t_last,i - t_ij))   (at most the number of uses)   and   t_last,i,

because z_ir(tau) = w_r y_ir exp(-lam_r (tau - t_last,i)), and a use at time t updates
y_ir <- y_ir exp(-lam_r (t - t_last,i)) + 1 in O(R). The single exponential of the paper is the
case R = 1 without accumulation.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np

YEAR_H = 365 * 24.0


@dataclass(frozen=True)
class Curve:
    """A forgetting curve phi(age), age in hours, phi(0) = 1, non-increasing."""

    kind: str          # "exp", "power", "hyperbolic" or "stretched"
    a: float           # exp: decay per hour; power: exponent d; hyperbolic: scale h (hours);
                       # stretched: rate per day**beta
    b: float = 0.0     # power: offset h0 (hours); stretched: exponent beta

    def __call__(self, age) -> np.ndarray:
        age = np.maximum(np.asarray(age, dtype=np.float64), 0.0)
        if self.kind == "exp":
            return np.power(self.a, age)
        if self.kind == "power":
            return np.power(1.0 + age / self.b, -self.a)
        if self.kind == "hyperbolic":
            return 1.0 / (1.0 + age / self.a)
        if self.kind == "stretched":   # FadeMem's form, v(t) = exp(-lam * t**beta) with t in days
            return np.exp(-self.a * np.power(age / 24.0, self.b))
        raise ValueError(self.kind)

    @property
    def timescale(self) -> float:
        """Hours over which the curve first drops noticeably (used to place the fastest rate)."""
        if self.kind == "exp":
            return 1.0 / -np.log(self.a)
        if self.kind == "power":
            return self.b
        if self.kind == "hyperbolic":
            return self.a
        return 24.0 * (0.105 / self.a) ** (1.0 / self.b)   # stretched: time to fall to 0.9

    @property
    def label(self) -> str:
        if self.kind == "exp":
            return f"exponential ({self.a:g} per hour)"
        if self.kind == "power":
            return f"power law, d = {self.a:g}, h0 = {self.b:g} h"
        if self.kind == "hyperbolic":
            return f"hyperbolic, h = {self.a:g} h"
        half_life = (np.log(2) / self.a) ** (1.0 / self.b)
        return f"stretched exponential, beta = {self.b:g}, half-life {half_life:.1f} days"


def exponential(decay: float = 0.995) -> Curve:
    return Curve("exp", decay)


def power_law(d: float = 0.5, h0: float = 1.0) -> Curve:
    return Curve("power", d, h0)


def hyperbolic(h: float = 24.0) -> Curve:
    return Curve("hyperbolic", h)


def stretched_exp(rate_per_day: float, beta: float) -> Curve:
    return Curve("stretched", rate_per_day, beta)


ACT_R = power_law(0.5, 1.0)                        # ACT-R's default decay, one-hour offset
FADEMEM = stretched_exp(0.1, 0.8)                  # FadeMem's long-term layer (L. Wei et al. 2026)


class ExpSum:
    """phi(age) ~= sum_r w[r] * exp(-lam[r] * age), rates in 1/hour, ascending."""

    def __init__(self, lam, w, curve: Curve | None = None, age_max: float = YEAR_H):
        self.lam = np.asarray(lam, dtype=np.float64)
        self.w = np.asarray(w, dtype=np.float64)
        self.curve = curve
        self.age_max = age_max

    @property
    def R(self) -> int:
        return len(self.lam)

    def __call__(self, age) -> np.ndarray:
        age = np.maximum(np.asarray(age, dtype=np.float64), 0.0)
        return np.exp(-np.multiply.outer(age, self.lam)) @ self.w

    def errors(self, curve: Curve | None = None, age_max: float | None = None,
               n: int = 20000) -> tuple[float, float]:
        """Errors on [0, age_max]: (largest relative error where the curve is above 1% of its
        start, largest error as a share of the curve's range). The second is what changes a
        min-max normalized score."""
        curve = curve or self.curve
        age_max = age_max or self.age_max
        ages = np.concatenate([[0.0], np.geomspace(1e-3, age_max, n)])
        y, f = curve(ages), self(ages)
        big = y >= 0.01
        return (float(np.max(np.abs(f[big] - y[big]) / y[big])),
                float(np.max(np.abs(f - y)) / (y.max() - y.min())))


def _nnls(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    from scipy.optimize import nnls
    return nnls(A, b)[0]


@lru_cache(maxsize=64)
def fit_expsum(curve: Curve, R: int = 8, age_max: float = YEAR_H, n_grid: int = 4000) -> ExpSum:
    """Non-negative sum of R exponentials that minimises the largest weighted error on [0, age_max].

    The rates are log-spaced between a slowest rate tied to age_max and a fastest rate tied to
    the curve's own time scale; both ends are chosen from a small grid, the weights by
    non-negative least squares on relative errors. An exponential curve is returned exactly (R = 1).
    """
    if curve.kind == "exp":
        return ExpSum([-np.log(curve.a)], [1.0], curve, age_max)
    ages = np.concatenate([[0.0], np.geomspace(1e-2, age_max, n_grid)])
    y = curve(ages)
    # relative error where the curve is above 1% of its start, absolute error below: what
    # matters for ranking is the error against the score's range, and a curve that falls
    # to 1e-5 would otherwise spend every exponential on its far tail
    W = 1.0 / np.maximum(y, 0.01)
    best = None
    for lo in np.geomspace(0.01, 1.0, 9):
        for hi in np.geomspace(0.1, 30.0, 11):
            lam = np.geomspace(lo / age_max, hi / curve.timescale, R)
            A = np.exp(-np.outer(ages, lam))
            w = _nnls(A * W[:, None], y * W)
            rel = float(np.max(np.abs(A @ w - y) * W))
            if best is None or rel < best[0]:
                best = (rel, lam, w)
    return ExpSum(best[1], best[2], curve, age_max)
