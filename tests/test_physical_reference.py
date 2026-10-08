"""Algorithm controls include independent single-exponential/logarithm roots."""
import math
import unittest

from src import thermal_oracle as t
from src.physical_reference import (
    _bisect_monotone_gradient, kkt_residual_interval,
    regret_interval, solve_thermal_reference,
)

LOG_ROOT_ULPS = 12.0
REFERENCE_WIDTHS = (1e-5, 1e-8, 1e-11)


class PhysicalReferenceTests(unittest.TestCase):
    def test_lower_boundary(self):
        result = solve_thermal_reference(t.LAMBDA_MAX)
        self.assertEqual(result.branch, "lower_boundary")
        self.assertEqual(result.p_ref, t.A)
        self.assertEqual(result.bracket, (t.A, t.A))
        self.assertEqual(result.kkt_residual, 0.0)
        self.assertEqual(result.stopping_reason, "lower_boundary_sign")
        self.assertGreater(result.gradient_at_reference, 0.0)

    def test_upper_boundary(self):
        result = solve_thermal_reference(t.LAMBDA_MIN)
        self.assertEqual(result.branch, "upper_boundary")
        self.assertEqual(result.p_ref, t.B)
        self.assertEqual(result.bracket, (t.B, t.B))
        self.assertEqual(result.kkt_residual, 0.0)
        self.assertEqual(result.stopping_reason, "upper_boundary_sign")
        self.assertLess(result.gradient_at_reference, 0.0)

    def test_float_threshold_equalities_and_adjacent_lambdas(self):
        # Equalities refer to evaluated float thresholds, not exact pi/exp.
        self.assertEqual(t.objective_gradient(t.A, t.CRITICAL_LAMBDA_A), 0.0)
        self.assertEqual(t.objective_gradient(t.B, t.CRITICAL_LAMBDA_B), 0.0)
        for lambda_, expected in (
            (t.CRITICAL_LAMBDA_A, "lower_boundary"),
            (math.nextafter(t.CRITICAL_LAMBDA_A, math.inf), "lower_boundary"),
            (math.nextafter(t.CRITICAL_LAMBDA_A, -math.inf), "interior"),
            (t.CRITICAL_LAMBDA_B, "upper_boundary"),
            (math.nextafter(t.CRITICAL_LAMBDA_B, -math.inf), "upper_boundary"),
            (math.nextafter(t.CRITICAL_LAMBDA_B, math.inf), "interior"),
        ):
            with self.subTest(lambda_=lambda_):
                self.assertEqual(solve_thermal_reference(lambda_).branch, expected)

    def test_interior_bracket_tolerance_and_value_error_formula(self):
        lambda_ = (t.CRITICAL_LAMBDA_A + t.CRITICAL_LAMBDA_B) / 2.0
        result = solve_thermal_reference(lambda_)
        lo, hi = result.bracket
        self.assertEqual(result.branch, "interior")
        self.assertLessEqual(lo, result.p_ref)
        self.assertLessEqual(result.p_ref, hi)
        self.assertLess(t.objective_gradient(lo, lambda_), 0.0)
        self.assertGreater(t.objective_gradient(hi, lambda_), 0.0)
        self.assertTrue(result.width_tolerance_met)
        self.assertEqual(result.stopping_reason, "width_tolerance")
        radius = max(result.p_ref - lo, hi - result.p_ref)
        self.assertEqual(result.reference_value_error_estimate_conditional,
                         t.MAX_CURVATURE * radius ** 2 / 2.0)
        self.assertEqual(result.exact_midpoint_distance_bound_conditional,
                         result.width / 2.0)
        self.assertIn("NOT_CERTIFIED", result.numerical_status)

    def test_tighter_references_keep_nested_brackets(self):
        lambda_ = (t.CRITICAL_LAMBDA_A + t.CRITICAL_LAMBDA_B) / 2.0
        references = [solve_thermal_reference(lambda_, width_tol=tol)
                      for tol in REFERENCE_WIDTHS]
        for previous, current in zip(references, references[1:]):
            self.assertGreaterEqual(current.bracket[0], previous.bracket[0])
            self.assertLessEqual(current.bracket[1], previous.bracket[1])
            self.assertLess(current.width, previous.width)
            self.assertGreater(current.iterations, previous.iterations)

    def test_independent_single_exponential_logarithmic_root(self):
        amplitude, rate, lambda_ = 1.3, 2.1, 0.7
        def gradient(p):
            return -amplitude * rate * math.exp(-rate * p) + lambda_
        state = _bisect_monotone_gradient(
            gradient, 0.0, 2.0, width_tol=1e-10, max_iterations=80)
        logarithmic_root = math.log(amplitude * rate / lambda_) / rate
        log_allowance = LOG_ROOT_ULPS * math.ulp(logarithmic_root)
        self.assertLessEqual(state.lower - log_allowance, logarithmic_root)
        self.assertLessEqual(logarithmic_root, state.upper + log_allowance)
        radius = max(state.p_ref - state.lower, state.upper - state.p_ref)
        self.assertLessEqual(abs(state.p_ref - logarithmic_root),
                             radius + log_allowance)

    def test_floating_zero_keeps_uncertainty(self):
        mid = t.A + (t.B - t.A) / 2.0
        lambda_ = -t.temperature_derivative(mid)
        result = solve_thermal_reference(lambda_, width_tol=1e-12)
        self.assertEqual(result.stopping_reason, "floating_zero_gradient")
        self.assertEqual(result.bracket, (t.A, t.B))
        self.assertEqual(result.p_ref, mid)
        self.assertEqual(result.iterations, 1)
        self.assertGreater(result.width, 0.0)
        self.assertFalse(result.width_tolerance_met)
        self.assertEqual(result.kkt_residual, 0.0)  # Float diagnostic only.

    def test_adjacent_float_midpoint_stagnation(self):
        a = 1.0
        b = math.nextafter(a, math.inf)
        def gradient(p):
            return 2.0 * ((p - a) / (b - a)) - 1.0
        state = _bisect_monotone_gradient(
            gradient, a, b, width_tol=(b - a) / 4.0, max_iterations=80)
        self.assertEqual(state.stopping_reason, "floating_midpoint_stagnation")
        self.assertEqual(state.iterations, 0)
        self.assertEqual((state.lower, state.upper), (a, b))
        self.assertIn(state.p_ref, (a, b))

    def test_zero_and_one_iteration_budgets(self):
        lambda_ = (t.CRITICAL_LAMBDA_A + t.CRITICAL_LAMBDA_B) / 2.0
        for budget in (0, 1):
            with self.subTest(budget=budget):
                result = solve_thermal_reference(
                    lambda_, width_tol=1e-15, max_iterations=budget)
                self.assertEqual(result.iterations, budget)
                self.assertEqual(result.stopping_reason, "max_iterations")
                self.assertFalse(result.width_tolerance_met)
                self.assertGreater(result.width, 0.0)

    def test_cache_and_evaluation_accounts(self):
        result = solve_thermal_reference(t.LAMBDA_MAX)
        counts = result.evaluations["by_kind"]
        self.assertEqual(counts["value"],
                         dict(requested=1, executed=1, unique=1, reused=0))
        self.assertEqual(counts["derivative"],
                         dict(requested=3, executed=2, unique=2, reused=1))
        self.assertEqual(counts["curvature"]["executed"], 0)
        total = result.evaluations["total"]
        self.assertEqual(total["requested"],
                         total["executed"] + total["reused"])

    def test_kkt_boundaries_and_no_tolerance_snapping(self):
        self.assertEqual(kkt_residual_interval(t.A, 3.0, t.A, t.B), 0.0)
        self.assertEqual(kkt_residual_interval(t.A, -3.0, t.A, t.B), 3.0)
        self.assertEqual(kkt_residual_interval(t.B, -3.0, t.A, t.B), 0.0)
        self.assertEqual(kkt_residual_interval(t.B, 3.0, t.A, t.B), 3.0)
        near_a = math.nextafter(t.A, t.B)
        self.assertEqual(kkt_residual_interval(near_a, 3.0, t.A, t.B), 3.0)

    def test_invalid_reference_inputs_and_options(self):
        for lambda_ in (0.0, math.nan, math.inf):
            with self.assertRaises(ValueError):
                solve_thermal_reference(lambda_)
        for tolerance in (0.0, -1.0, math.nan, math.inf):
            with self.assertRaises(ValueError):
                solve_thermal_reference(t.LAMBDA_MIN, width_tol=tolerance)
        for budget in (-1,):
            with self.assertRaises(ValueError):
                solve_thermal_reference(t.LAMBDA_MIN, max_iterations=budget)
        for budget in (True, 1.5, "4"):
            with self.assertRaises(TypeError):
                solve_thermal_reference(t.LAMBDA_MIN, max_iterations=budget)
        with self.assertRaises(ValueError):
            kkt_residual_interval(t.A - 0.001, 0.0, t.A, t.B)
        with self.assertRaises(ValueError):
            kkt_residual_interval(t.A, math.nan, t.A, t.B)
        with self.assertRaises(ValueError):
            _bisect_monotone_gradient(
                lambda p: -p, 0.0, 1.0, width_tol=1e-5, max_iterations=5)

    def test_negative_raw_difference_is_preserved(self):
        result = regret_interval(-1e-6, 2e-6)
        self.assertEqual(result["raw_difference"], -1e-6)
        self.assertTrue(result["raw_difference_negative"])
        self.assertTrue(result["compatible"])
        self.assertEqual(result["interval"], [0.0, 1e-6])
        inconsistent = regret_interval(-1e-6, 0.0)
        self.assertFalse(inconsistent["compatible"])
        self.assertIsNone(inconsistent["interval"])
        self.assertEqual(inconsistent["raw_difference"], -1e-6)

    def test_regret_interval_value_errors_and_invalid_bounds(self):
        result = regret_interval(
            1.0, 0.3, decision_value_error_bound=0.1,
            reference_value_error_bound=0.2)
        self.assertAlmostEqual(result["lower"], 0.7)
        self.assertAlmostEqual(result["upper"], 1.6)
        for raw, beta in ((math.nan, 0.0), (0.0, -1.0), (0.0, math.inf)):
            with self.assertRaises(ValueError):
                regret_interval(raw, beta)


if __name__ == "__main__":
    unittest.main()

