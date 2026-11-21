"""Signal detection theory on accept/reject verdicts.

d' = z(TTR) - z(FTR), where the true-target rate is the accept rate on the cell
whose correct verdict is accept (C) and the false-target rate is the accept rate
on the cell whose correct verdict is reject (D_u).

The log-linear correction is applied *only* when a rate is exactly 0 or 1, which
is the pre-registered rule. Applying it everywhere would shrink every estimate
toward zero, and applying it nowhere would make a floored cell produce an
infinite d' -- which is precisely the failure this probe is looking for, so it
has to be visible rather than silently patched.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

#: Inverse normal CDF: Acklam's rational approximation, relative error ~1e-9.
#: Checked against scipy.stats.norm.ppf -- max absolute error 1.6e-9 over
#: p in [0.001, 0.999]. Kept stdlib-only so analysis never pulls in a heavy
#: dependency tree while a model is resident.
_A = (-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
      1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00)
_B = (-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
      6.680131188771972e+01, -1.328068155288572e+01)
_C = (-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
      -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00)
_D = (7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
      3.754408661907416e+00)


def z(p: float) -> float:
    if not 0.0 < p < 1.0:
        raise ValueError(f"z() needs 0 < p < 1, got {p}; correct the rate first")
    if p < 0.02425:
        q = math.sqrt(-2 * math.log(p))
        return (((((_C[0]*q+_C[1])*q+_C[2])*q+_C[3])*q+_C[4])*q+_C[5]) / \
               ((((_D[0]*q+_D[1])*q+_D[2])*q+_D[3])*q+1)
    if p > 1 - 0.02425:
        q = math.sqrt(-2 * math.log(1 - p))
        return -((((((_C[0]*q+_C[1])*q+_C[2])*q+_C[3])*q+_C[4])*q+_C[5]) /
                 ((((_D[0]*q+_D[1])*q+_D[2])*q+_D[3])*q+1))
    q = p - 0.5
    r = q * q
    return (((((_A[0]*r+_A[1])*r+_A[2])*r+_A[3])*r+_A[4])*r+_A[5])*q / \
           (((((_B[0]*r+_B[1])*r+_B[2])*r+_B[3])*r+_B[4])*r+1)


@dataclass(frozen=True)
class Rates:
    hits: int
    n: int
    corrected: bool

    @property
    def rate(self) -> float:
        if self.corrected:
            return (self.hits + 0.5) / (self.n + 1)
        return self.hits / self.n


def rate(hits: int, n: int) -> Rates:
    """Correct only at the boundary, as pre-registered."""
    return Rates(hits, n, corrected=(hits == 0 or hits == n))


def d_prime(ttr: Rates, ftr: Rates) -> float:
    return z(ttr.rate) - z(ftr.rate)


def criterion(ttr: Rates, ftr: Rates) -> float:
    return -0.5 * (z(ttr.rate) + z(ftr.rate))
