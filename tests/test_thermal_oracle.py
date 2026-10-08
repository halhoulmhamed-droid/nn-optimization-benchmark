"""Independent expressions, finite differences and public-domain contracts."""
import math
import sys
import unittest

from src import thermal_oracle as t

VALUE_REL_TOL = 2e-14
VALUE_ABS_TOL = 5e-15
DERIVATIVE_ABS_TOL = 5e-12
ROUNDING_ALLOWANCE_FACTOR = 12.0
STABLE_STEPS = (1e-3, 1e-4, 1e-5)
SMALL_STEPS = (1e-8, 1e-10, 1e-12)
POINTS = (0.04, 0.10, 0.18)


def analytic_sensor(p, order):
    # Independent spatial expression u(1/4,p) and its time derivatives.
    return sum(amplitude * (-(mode * math.pi) ** 2) ** order
               * math.exp(-(mode * math.pi) ** 2 * p)
               * math.sin(mode * math.pi / 4.0)
               for mode, amplitude in ((1, 1.0), (2, 0.2)))


def higher_derivative_magnitude(p, order):
    return (math.pi ** (2 * order) / math.sqrt(2.0)
            * math.exp(-math.pi ** 2 * p)
            + 0.2 * (4.0 * math.pi ** 2) ** order
            * math.exp(-4.0 * math.pi ** 2 * p))


def finite_difference_record(p, h, order):
    function = t.temperature if order == 1 else t.temperature_derivative
    left, right = function(p - h), function(p + h)
    approximation = (right - left) / (2.0 * h)
    exact_float = (t.temperature_derivative(p) if order == 1
                   else t.temperature_second_derivative(p))
    truncation = h * h * higher_derivative_magnitude(p - h, order + 2) / 6.0
    rounding = (ROUNDING_ALLOWANCE_FACTOR * sys.float_info.epsilon
                * (abs(left) + abs(right)) / (2.0 * h))
    return {
        "p": p, "h": h, "derivative_order": order,
        "approximation": approximation, "analytic_float": exact_float,
        "absolute_error": abs(approximation - exact_float),
        "truncation_bound_exact_formula": truncation,
        "rounding_allowance_heuristic": rounding,
        "tolerance": truncation + rounding,
        "certified": False,
    }


class ThermalOracleTests(unittest.TestCase):
    def assert_close(self, actual, expected, abs_tol=VALUE_ABS_TOL):
        self.assertTrue(math.isclose(actual, expected,
                                    rel_tol=VALUE_REL_TOL, abs_tol=abs_tol),
                        (actual, expected))

    def test_values_against_original_spatial_modes(self):
        for p in (t.A, *POINTS, t.B):
            with self.subTest(p=p):
                self.assert_close(t.temperature(p), analytic_sensor(p, 0))

    def test_derivatives_against_original_spatial_modes(self):
        for p in (t.A, *POINTS, t.B):
            for order, function in ((1, t.temperature_derivative),
                                    (2, t.temperature_second_derivative)):
                with self.subTest(p=p, order=order):
                    self.assert_close(function(p), analytic_sensor(p, order),
                                      DERIVATIVE_ABS_TOL)

    def test_F_and_T_are_the_same_oracle(self):
        self.assertIs(t.F, t.T)
        self.assertIs(t.F_prime, t.T_prime)
        self.assertIs(t.F_second, t.T_second)
        self.assert_close(t.F(0.10), t.T(0.10))

    def test_objective_and_its_derivatives(self):
        lambda_ = (t.CRITICAL_LAMBDA_A + t.CRITICAL_LAMBDA_B) / 2.0
        for p in (t.A, 0.10, t.B):
            with self.subTest(p=p):
                self.assert_close(t.objective(p, lambda_),
                                  analytic_sensor(p, 0) + lambda_ * p)
                self.assert_close(t.objective_gradient(p, lambda_),
                                  analytic_sensor(p, 1) + lambda_,
                                  DERIVATIVE_ABS_TOL)
                self.assert_close(t.objective_hessian(p, lambda_),
                                  analytic_sensor(p, 2), DERIVATIVE_ABS_TOL)

    def test_constants_match_proven_endpoint_formulas(self):
        self.assert_close(t.S0, analytic_sensor(t.A, 0) - analytic_sensor(t.B, 0))
        self.assert_close(t.S1, -analytic_sensor(t.A, 1), DERIVATIVE_ABS_TOL)
        self.assert_close(t.MIN_CURVATURE, analytic_sensor(t.B, 2),
                          DERIVATIVE_ABS_TOL)
        self.assert_close(t.MAX_CURVATURE, analytic_sensor(t.A, 2),
                          DERIVATIVE_ABS_TOL)
        self.assertGreater(t.MIN_CURVATURE, 0.0)
        self.assertGreater(t.MAX_CURVATURE, t.MIN_CURVATURE)
        self.assertEqual(t.LAMBDA_MIN, -t.DERIVATIVE_B / 2.0)
        self.assertEqual(t.LAMBDA_MAX, -3.0 * t.DERIVATIVE_A / 2.0)

    def test_finite_differences_in_truncation_region(self):
        for p in POINTS:
            for order in (1, 2):
                records = [finite_difference_record(p, h, order)
                           for h in STABLE_STEPS]
                for item in records:
                    with self.subTest(p=p, order=order, h=item["h"]):
                        self.assertLessEqual(item["absolute_error"],
                                             item["tolerance"])
                # Only this truncation-dominated region is tested for decrease.
                self.assertLess(records[1]["absolute_error"],
                                records[0]["absolute_error"])
                self.assertLess(records[2]["absolute_error"],
                                records[1]["absolute_error"])

    def test_small_difference_steps_do_not_require_monotone_improvement(self):
        for p in POINTS:
            for order in (1, 2):
                for h in SMALL_STEPS:
                    item = finite_difference_record(p, h, order)
                    with self.subTest(p=p, order=order, h=h):
                        self.assertLessEqual(item["absolute_error"],
                                             item["tolerance"])

    def test_time_rejects_nonfinite_out_of_domain_and_invalid_types(self):
        for function in (t.temperature, t.temperature_derivative,
                         t.temperature_second_derivative):
            for p in (math.nan, math.inf, -math.inf, 0.0,
                      math.nextafter(t.A, -math.inf),
                      math.nextafter(t.B, math.inf)):
                with self.subTest(function=function.__name__, p=p):
                    with self.assertRaises(ValueError):
                        function(p)
            for p in (True, "0.1", None, complex(0.1, 0.0)):
                with self.assertRaises(TypeError):
                    function(p)

    def test_lambda_contract_even_when_hessian_does_not_depend_on_it(self):
        for function in (t.objective, t.objective_gradient, t.objective_hessian):
            for lambda_ in (0.0, math.nan, math.inf,
                            math.nextafter(t.LAMBDA_MIN, -math.inf),
                            math.nextafter(t.LAMBDA_MAX, math.inf)):
                with self.assertRaises(ValueError):
                    function(0.10, lambda_)
            with self.assertRaises(TypeError):
                function(0.10, True)
        self.assertEqual(t.validate_lambda(t.LAMBDA_MIN), t.LAMBDA_MIN)
        self.assertEqual(t.validate_lambda(t.LAMBDA_MAX), t.LAMBDA_MAX)


if __name__ == "__main__":
    unittest.main()

