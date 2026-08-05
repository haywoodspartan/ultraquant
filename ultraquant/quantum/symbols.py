"""The Greek letters, implemented: special functions and chaos *measurement*.

The request behind this module asked for psi, phi, theta, xi, chi, lambda, mu,
nu, pi, rho, sigma, tau, omega, gamma, eta and zeta "as variable functions for
entropy and chaos". It arrives after the chaos-injection experiment was built,
measured, and reverted — every exploration rate priced correctly measured
harm, and a deterministic clock beat the random pulse in both regimes. So
these letters are implemented on the side of that verdict that survives:
**instruments that measure entropy and chaos, and the genuine mathematics the
letters name** — from scratch, pure stdlib, each with a reference gate its
implementation must pass.

What each letter is here (and honest notes where a letter names a *context*
rather than a computable function):

============ ==============================================================
``psi``      wave function: amplitude ``i`` of a statevector
``theta``    phase of a complex amplitude, in radians
``phi``      the golden ratio, computed as ``(1 + sqrt(5)) / 2``
``xi``       Riemann's xi, composed from ``gamma_fn`` and ``zeta``; its
             functional equation ``xi(s) == xi(1 - s)`` is a built-in
             self-test of both
``chi``      Euler characteristic ``V - E + F`` (faces default 0: graphs)
``lam``      eigenvalues of a real symmetric matrix (Jacobi; ``lambda`` is
             a Python keyword, hence ``lam``)
``mu``       mean
``nu``       frequency from a period, ``1 / T``
``pi_``      the circle constant (stdlib's, re-exported for completeness)
``rho``      Pearson correlation
``sigma``    standard deviation (population)
``tau``      the turn constant ``2 * pi``
``omega``    angular frequency ``2 * pi * nu``
``gamma_fn`` the Gamma function, Lanczos approximation with reflection
``euler_gamma`` the Euler-Mascheroni constant, computed (not pasted) by
             Euler-Maclaurin correction of the harmonic series
``eta``      Dirichlet eta (the alternating zeta), the engine behind
             ``zeta``'s continuation; also plain efficiency ``out / in``
             via ``efficiency``
``zeta``     the Riemann zeta function for real ``s > 0``, ``s != 1``, via
             eta with binomial acceleration
``shannon``  Shannon entropy of a distribution - the chaos *measurement*:
             this is what quantifies QPU counts, timing wobble, or any
             claimed randomness, instead of injecting any
``min_entropy`` the conservative entropy floor, ``-log2(max p)``
============ ==============================================================

Every function is gated by reference values in ``tests/test_symbols.py`` —
``gamma_fn`` against ``math.gamma`` to 1e-12, ``zeta(2)`` against
``pi^2 / 6``, ``euler_gamma`` against its known digits, ``xi`` against its own
functional equation — because a special function that is almost right is
worse than none.

Pure Python standard library.
"""

from __future__ import annotations

import cmath
import math
from typing import Sequence

__all__ = [
    "psi", "theta", "phi", "xi", "chi", "lam", "mu", "nu", "pi_", "rho",
    "sigma", "tau", "omega", "gamma_fn", "euler_gamma", "eta", "efficiency",
    "zeta", "shannon", "min_entropy",
]

# --------------------------------------------------------------------------- #
# constants
# --------------------------------------------------------------------------- #

#: The circle constant, re-exported so the sandbox namespace is complete.
pi_ = math.pi

#: The turn constant.
tau = 2.0 * math.pi

#: The golden ratio, from its defining equation x^2 = x + 1.
phi = (1.0 + math.sqrt(5.0)) / 2.0


def _compute_euler_gamma(terms: int = 50) -> float:
    """Euler-Mascheroni via Euler-Maclaurin correction of ``H_n - ln n``.

    The naive limit converges like 1/n; the correction terms
    ``1/(2n) - 1/(12n^2) + 1/(120n^4)`` push it to ~1e-13 by n = 50.
    Computed at import so the constant is *earned*, then verified against the
    known digits in the tests.
    """
    n = terms
    harmonic = sum(1.0 / k for k in range(1, n + 1))
    return (harmonic - math.log(n) - 1.0 / (2 * n)
            + 1.0 / (12 * n ** 2) - 1.0 / (120 * n ** 4))


#: The Euler-Mascheroni constant, computed rather than pasted.
euler_gamma = _compute_euler_gamma()


# --------------------------------------------------------------------------- #
# special functions
# --------------------------------------------------------------------------- #

#: Lanczos coefficients (g = 7, n = 9) - the standard set.
_LANCZOS_G = 7.0
_LANCZOS = (
    0.99999999999980993,
    676.5203681218851,
    -1259.1392167224028,
    771.32342877765313,
    -176.61502916214059,
    12.507343278686905,
    -0.13857109526572012,
    9.9843695780195716e-6,
    1.5056327351493116e-7,
)


def gamma_fn(x: float) -> float:
    """The Gamma function on the reals, by Lanczos approximation.

    Reflection handles ``x < 0.5``; poles at non-positive integers raise.
    Gate: agrees with ``math.gamma`` (independent implementation) to 1e-12
    relative across the tested range - stdlib is used as the *referee*, not
    the implementation.
    """
    if x < 0.5:
        if x == math.floor(x):
            raise ValueError(f"gamma has a pole at {x}")
        # Reflection: Gamma(x) Gamma(1-x) = pi / sin(pi x)
        return math.pi / (math.sin(math.pi * x) * gamma_fn(1.0 - x))
    x -= 1.0
    series = _LANCZOS[0]
    for index, coefficient in enumerate(_LANCZOS[1:], start=1):
        series += coefficient / (x + index)
    t = x + _LANCZOS_G + 0.5
    return math.sqrt(2.0 * math.pi) * t ** (x + 0.5) * math.exp(-t) * series


def eta(s: float, terms: int = 60) -> float:
    """Dirichlet eta - the alternating zeta - with binomial acceleration.

    Converges for every real ``s > 0``, which is exactly what lets
    :func:`zeta` reach into ``0 < s < 1`` where its own series diverges.
    """
    if s <= 0.0:
        raise ValueError("eta implemented for s > 0")
    # Euler transform of the alternating series (Borwein-style weights).
    total = 0.0
    binom = 1.0
    outer = 0.5
    for k in range(terms):
        inner = 0.0
        coefficient = 1.0
        for j in range(k + 1):
            inner += coefficient * (j + 1) ** (-s)
            coefficient *= -(k - j) / (j + 1.0)
        total += outer * inner
        outer *= 0.5
        binom += 1.0
    return total


def zeta(s: float) -> float:
    """The Riemann zeta function for real ``s > 0``, ``s != 1``.

    Via ``zeta(s) = eta(s) / (1 - 2^(1-s))``. Gates: ``zeta(2) = pi^2/6``,
    ``zeta(4) = pi^4/90`` to 1e-10, and Apery's ``zeta(3)`` to its known
    digits.
    """
    if s == 1.0:
        raise ValueError("zeta has its pole at s = 1")
    if s <= 0.0:
        raise ValueError("real-axis implementation covers s > 0")
    return eta(s) / (1.0 - 2.0 ** (1.0 - s))


def xi(s: float) -> float:
    """Riemann's xi: ``1/2 s (s-1) pi^(-s/2) Gamma(s/2) zeta(s)``.

    The showpiece composition - it exercises :func:`gamma_fn` and
    :func:`zeta` together, and its functional equation ``xi(s) = xi(1 - s)``
    is a self-test no coincidence passes: get either component slightly wrong
    and the symmetry breaks measurably.
    """
    return (0.5 * s * (s - 1.0) * math.pi ** (-s / 2.0)
            * gamma_fn(s / 2.0) * zeta(s))


# --------------------------------------------------------------------------- #
# statistics and entropy - the chaos measurements
# --------------------------------------------------------------------------- #

def mu(values: Sequence[float]) -> float:
    """Mean."""
    if not values:
        raise ValueError("mean of nothing")
    return sum(values) / len(values)


def sigma(values: Sequence[float]) -> float:
    """Population standard deviation."""
    m = mu(values)
    return math.sqrt(sum((v - m) ** 2 for v in values) / len(values))


def rho(xs: Sequence[float], ys: Sequence[float]) -> float:
    """Pearson correlation coefficient."""
    if len(xs) != len(ys) or len(xs) < 2:
        raise ValueError("correlation needs two equal-length series")
    mx, my = mu(xs), mu(ys)
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    sx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    sy = math.sqrt(sum((y - my) ** 2 for y in ys))
    if sx == 0.0 or sy == 0.0:
        return 0.0
    return cov / (sx * sy)


def shannon(counts: Sequence[float]) -> float:
    """Shannon entropy in bits of a count/probability vector.

    The honest end of the chaos idea: this *measures* unpredictability -
    of QPU measurement counts, timing wobble, or anything claiming to be
    random - instead of injecting any. Uniform over n gives ``log2 n``; a
    delta gives 0. It was exactly this kind of measurement (raw distinct
    values, not whitened prettiness) that caught the all-zero jitter trap.
    """
    total = float(sum(counts))
    if total <= 0.0:
        raise ValueError("entropy of an empty distribution")
    h = 0.0
    for count in counts:
        if count > 0:
            p = count / total
            h -= p * math.log2(p)
    return h


def min_entropy(counts: Sequence[float]) -> float:
    """Min-entropy ``-log2(max p)`` - the conservative floor.

    The number an entropy *source* must be judged by: it credits only the
    hardest-to-guess outcome, which is why it is the right gate for anything
    feeding randomness to anything else.
    """
    total = float(sum(counts))
    if total <= 0.0:
        raise ValueError("entropy of an empty distribution")
    return -math.log2(max(counts) / total)


# --------------------------------------------------------------------------- #
# physics-shaped helpers and bridges
# --------------------------------------------------------------------------- #

def psi(state: Sequence[complex], index: int) -> complex:
    """The wave function: amplitude ``index`` of a statevector."""
    return complex(state[index])


def theta(amplitude: complex) -> float:
    """Phase of a complex amplitude, in radians."""
    return cmath.phase(amplitude)


def nu(period: float) -> float:
    """Frequency from a period."""
    if period == 0.0:
        raise ValueError("zero period")
    return 1.0 / period


def omega(frequency: float) -> float:
    """Angular frequency ``2 pi nu``."""
    return tau * frequency


def efficiency(out_value: float, in_value: float) -> float:
    """Plain efficiency ``out / in`` - the eta of engineering."""
    if in_value == 0.0:
        raise ValueError("efficiency with zero input")
    return out_value / in_value


def chi(vertices: int, edges: int, faces: int = 0) -> int:
    """Euler characteristic ``V - E + F`` (graphs default to ``F = 0``)."""
    return int(vertices) - int(edges) + int(faces)


def lam(matrix: Sequence[Sequence[float]]) -> list[float]:
    """Eigenvalues of a real symmetric matrix, ascending.

    Bridges to the Jacobi eigensolver the entanglement-entropy analysis
    already trusts, rather than duplicating one.
    """
    from ultraquant.quantum.analysis import eigenvalues_hermitian

    values = eigenvalues_hermitian(
        [[complex(v) for v in row] for row in matrix]
    )
    return sorted(values)
