"""The Greek letters, each held to a reference value.

A special function that is almost right is worse than none, so every function
in ``quantum/symbols.py`` is gated against independent references: stdlib's
``math.gamma`` as referee for the Lanczos implementation, closed forms for
``zeta`` at even integers, Apery's constant, the known digits of
Euler-Mascheroni, and — the showpiece — Riemann's xi held to its own
functional equation ``xi(s) == xi(1 - s)``, which fails measurably if either
``gamma_fn`` or ``zeta`` drifts.

The entropy functions are the honest end of the chaos idea: they *measure*
unpredictability instead of injecting it, and their gates are the exact
distributions (uniform -> log2 n, delta -> 0).
"""

from __future__ import annotations

import math
import unittest

from ultraquant.quantum import symbols as S


class SpecialFunctionTests(unittest.TestCase):
    """Gamma, zeta, eta, xi against independent references."""

    def test_gamma_matches_the_stdlib_referee(self) -> None:
        """math.gamma is an independent implementation - the referee, not
        the engine. Lanczos must agree to 1e-12 relative."""
        for x in (0.1, 0.5, 1.0, 1.5, 2.0, 3.7, 5.0, 7.2, 10.0, 15.5):
            with self.subTest(x=x):
                self.assertAlmostEqual(
                    S.gamma_fn(x) / math.gamma(x), 1.0, places=12
                )

    def test_gamma_half_is_root_pi(self) -> None:
        self.assertAlmostEqual(S.gamma_fn(0.5), math.sqrt(math.pi), places=13)

    def test_gamma_factorial_identity(self) -> None:
        for n in range(1, 10):
            with self.subTest(n=n):
                self.assertAlmostEqual(S.gamma_fn(n + 1),
                                       math.factorial(n), places=6)

    def test_gamma_reflection_covers_negatives(self) -> None:
        self.assertAlmostEqual(S.gamma_fn(-0.5) / math.gamma(-0.5), 1.0,
                               places=12)

    def test_gamma_poles_raise(self) -> None:
        for x in (0.0, -1.0, -2.0):
            with self.subTest(x=x), self.assertRaises(ValueError):
                S.gamma_fn(x)

    def test_zeta_closed_forms(self) -> None:
        self.assertAlmostEqual(S.zeta(2.0), math.pi ** 2 / 6.0, places=10)
        self.assertAlmostEqual(S.zeta(4.0), math.pi ** 4 / 90.0, places=10)

    def test_zeta_apery(self) -> None:
        self.assertAlmostEqual(S.zeta(3.0), 1.2020569031595942854, places=12)

    def test_zeta_pole_raises(self) -> None:
        with self.assertRaises(ValueError):
            S.zeta(1.0)

    def test_eta_alternating_identity(self) -> None:
        """eta(1) = ln 2 - the classic alternating harmonic sum."""
        self.assertAlmostEqual(S.eta(1.0), math.log(2.0), places=10)

    def test_xi_functional_equation(self) -> None:
        """The self-test no coincidence passes: xi(s) == xi(1 - s).

        Both gamma_fn and zeta feed xi; perturb either and the symmetry
        breaks at the sixth decimal, not the sixteenth.
        """
        for s in (0.2, 0.3, 0.4, 0.45, 0.6, 0.75, 0.9):
            with self.subTest(s=s):
                self.assertAlmostEqual(S.xi(s), S.xi(1.0 - s), places=12)

    def test_euler_gamma_known_digits(self) -> None:
        """Computed by Euler-Maclaurin, verified against the constant."""
        self.assertAlmostEqual(S.euler_gamma, 0.5772156649015329, places=10)

    def test_the_constants(self) -> None:
        self.assertAlmostEqual(S.phi ** 2, S.phi + 1.0, places=12)
        self.assertEqual(S.tau, 2.0 * math.pi)
        self.assertEqual(S.pi_, math.pi)


class EntropyMeasurementTests(unittest.TestCase):
    """The chaos measurements - quantifying randomness, not injecting it."""

    def test_uniform_distribution_is_log2_n(self) -> None:
        for n in (2, 4, 8, 16):
            with self.subTest(n=n):
                self.assertAlmostEqual(S.shannon([1] * n), math.log2(n),
                                       places=12)

    def test_a_delta_has_no_entropy(self) -> None:
        self.assertEqual(S.shannon([7, 0, 0, 0]), 0.0)

    def test_shannon_accepts_counts_or_probabilities(self) -> None:
        self.assertAlmostEqual(S.shannon([3, 1]), S.shannon([0.75, 0.25]),
                               places=12)

    def test_min_entropy_is_the_conservative_floor(self) -> None:
        counts = [3, 1, 1, 1]
        self.assertLessEqual(S.min_entropy(counts), S.shannon(counts))
        self.assertAlmostEqual(S.min_entropy([1, 1]), 1.0, places=12)

    def test_empty_distributions_raise(self) -> None:
        for fn in (S.shannon, S.min_entropy):
            with self.subTest(fn=fn.__name__), self.assertRaises(ValueError):
                fn([])


class StatisticsTests(unittest.TestCase):
    """mu, sigma, rho."""

    def test_mean_and_deviation(self) -> None:
        self.assertEqual(S.mu([1.0, 2.0, 3.0]), 2.0)
        self.assertAlmostEqual(S.sigma([1.0, 2.0, 3.0]),
                               math.sqrt(2.0 / 3.0), places=12)

    def test_correlation_extremes(self) -> None:
        xs = [1.0, 2.0, 3.0, 4.0]
        self.assertAlmostEqual(S.rho(xs, xs), 1.0, places=12)
        self.assertAlmostEqual(S.rho(xs, [-x for x in xs]), -1.0, places=12)
        self.assertEqual(S.rho(xs, [5.0] * 4), 0.0)


class BridgeTests(unittest.TestCase):
    """psi, theta, lam, chi and the physics helpers."""

    def test_psi_and_theta_read_a_real_statevector(self) -> None:
        from ultraquant.quantum.gates import H
        from ultraquant.quantum.state import QuantumState

        state = QuantumState(1)
        state.apply_gate(H, 0)
        amplitude = S.psi(state.amplitudes, 0)
        self.assertAlmostEqual(abs(amplitude), 1 / math.sqrt(2), places=12)
        self.assertAlmostEqual(S.theta(amplitude), 0.0, places=12)

    def test_lam_agrees_with_known_eigenvalues(self) -> None:
        self.assertEqual([round(v, 9) for v in S.lam([[2.0, 0.0], [0.0, 3.0]])],
                         [2.0, 3.0])
        values = S.lam([[0.0, 1.0], [1.0, 0.0]])
        self.assertAlmostEqual(values[0], -1.0, places=9)
        self.assertAlmostEqual(values[1], 1.0, places=9)

    def test_chi_of_the_platonic_cube(self) -> None:
        self.assertEqual(S.chi(8, 12, 6), 2)
        self.assertEqual(S.chi(vertices=3, edges=3), 0)  # a triangle cycle

    def test_frequency_helpers(self) -> None:
        self.assertEqual(S.nu(0.5), 2.0)
        self.assertAlmostEqual(S.omega(1.0), S.tau, places=12)
        self.assertEqual(S.efficiency(3.0, 4.0), 0.75)


class SandboxIntegrationTests(unittest.TestCase):
    """The letters are usable from chat's calc:, and escapes stay blocked."""

    def setUp(self) -> None:
        from ultraquant.interpreter.codefunc import SafeCodeRunner

        self.runner = SafeCodeRunner()

    def test_zeta_from_the_sandbox(self) -> None:
        self.assertAlmostEqual(self.runner.run("zeta(2)")["result"],
                               math.pi ** 2 / 6.0, places=10)

    def test_the_functional_equation_is_visible_from_chat(self) -> None:
        result = self.runner.run("xi(0.3) - xi(0.7)")["result"]
        self.assertAlmostEqual(result, 0.0, places=12)

    def test_entropy_measurement_from_the_sandbox(self) -> None:
        self.assertEqual(self.runner.run("shannon([1,1,1,1])")["result"], 2.0)

    def test_symbols_do_not_open_attribute_escapes(self) -> None:
        from ultraquant.interpreter.codefunc import CodeError

        for source in ("zeta.__globals__", "gamma_fn.__self__",
                       "shannon.__code__"):
            with self.subTest(source=source), self.assertRaises(CodeError):
                self.runner.run(source)


if __name__ == "__main__":
    unittest.main()
