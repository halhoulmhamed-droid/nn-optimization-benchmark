"""Sourced means and gradients, with fixed, explicit algebraic target tuples."""
import math
import unittest
from unittest.mock import patch

from src import neural_surrogate as n
from src import neural_gradients as ng
from src import surrogate_losses as sl
import test_neural_gradients as checks

# Existing physical scales read from the preserved phase 1B report.
# No oracle import or new physical evaluation; not estimated from these targets.
S0 = 0.5729505106050928
S1 = 9.313693283147819
LOSS_REL_TOL = 5e-14
LOSS_ABS_TOL = 3e-13
LOSS_GRAD_ABS_TOL = 4e-12
CONTROL_TARGETS = (
    ("one_point", (0.08,), (0.4,), (-1.0,)),
    ("three_points", (n.A, 0.11, n.B), (0.5, 0.3, 0.2), (-2.0, -1.5, -0.8)),
)
LOSS_FD_CONTROLS = []
AGGREGATION_CONTROLS = []


def loss_bounds(network, points, targets, sensitivity_targets, scale0, scale1,
                index, delta, method):
    seconds, thirds = [], []
    for p, target, wanted_g in zip(points, targets, sensitivity_targets):
        fb, gb = checks.parameter_bounds(network, p, index, delta)
        second = 2.0 / scale0 ** 2 * (fb[1] ** 2 + (fb[0] + abs(target)) * fb[2])
        third = 2.0 / scale0 ** 2 * (
            3 * fb[1] * fb[2] + (fb[0] + abs(target)) * fb[3])
        if method == "M2":
            second += 2.0 / scale1 ** 2 * (
                gb[1] ** 2 + (gb[0] + abs(wanted_g)) * gb[2])
            third += 2.0 / scale1 ** 2 * (
                3 * gb[1] * gb[2] + (gb[0] + abs(wanted_g)) * gb[3])
        seconds.append(second)
        thirds.append(third)
    return math.fsum(seconds) / len(points), math.fsum(thirds) / len(points)


def independent_loss_from_public_network(network, points, targets, sensitivities, method):
    value = sum(((network.forward(p) - target) / S0) ** 2
                for p, target in zip(points, targets)) / len(points)
    if method == "M2":
        value += sum(((network.input_derivative(p) - target) / S1) ** 2
                     for p, target in zip(points, sensitivities)) / len(points)
    return value


class SurrogateLossTests(unittest.TestCase):
    def assert_close(self, actual, expected, abs_tol=LOSS_ABS_TOL):
        self.assertTrue(math.isclose(actual, expected, rel_tol=LOSS_REL_TOL,
                                    abs_tol=abs_tol), (actual, expected))

    def test_one_point_factors_no_half_and_fixed_sensitivity_coefficient(self):
        network = checks.one_neuron(w=0.0, b=0.0, v=0.0, d=1.25)
        first = sl.m1_loss_and_gradient(network, (0.08,), (0.25,), S0=2.0)
        second = sl.m2_loss_and_gradient(network, (0.08,), (0.25,), (-3.0,),
                                          S0=2.0, S1=3.0)
        self.assertEqual(first.loss, 0.25)
        self.assertEqual(first.gradient, (0.0,) * 48 + (0.5,))
        self.assertEqual(second.loss, 1.25)
        self.assertEqual(second.value_component, 0.25)
        self.assertEqual(second.sensitivity_component, 1.0)
        self.assertEqual(second.gradient, (0.0,) * 48 + (0.5,))
        self.assertEqual((first.method, second.method), ("M1", "M2"))

    def test_three_points_are_means_not_sums_or_half_means(self):
        network = checks.one_neuron(w=0.0, b=0.0, v=0.0, d=1.25)
        sites, targets, sensitivities = (n.A, 0.11, n.B), (0.25, 0.75, -0.25), (-3.0, 0.0, 3.0)
        first = sl.m1_loss_and_gradient(network, sites, targets, S0=2.0)
        second = sl.m2_loss_and_gradient(network, sites, targets, sensitivities,
                                          S0=2.0, S1=3.0)
        expected_value = (1.0 ** 2 + 0.5 ** 2 + 1.5 ** 2) / (3.0 * 4.0)
        self.assert_close(first.loss, expected_value)
        self.assert_close(second.loss, expected_value + 2.0 / 3.0)
        self.assert_close(first.gradient[48], 0.5)
        self.assert_close(second.gradient[48], 0.5)
        self.assertEqual((second.n_values, second.n_sensitivities), (3, 3))
        AGGREGATION_CONTROLS.append(dict(
            control_only_scales=[2.0, 3.0], n=3, observed_M1=first.loss,
            expected_M1=expected_value, observed_M2=second.loss,
            expected_M2=expected_value + 2.0 / 3.0, passed=True))

    def test_scalar_values_against_independent_public_network_aggregation(self):
        for network in (checks.moderate_network(), checks.affine_control_network()):
            for _, points, targets, sensitivities in CONTROL_TARGETS:
                first = sl.m1_loss_and_gradient(network, points, targets, S0=S0)
                second = sl.m2_loss_and_gradient(network, points, targets, sensitivities,
                                                  S0=S0, S1=S1)
                self.assert_close(first.loss, independent_loss_from_public_network(
                    network, points, targets, sensitivities, "M1"))
                self.assert_close(second.loss, independent_loss_from_public_network(
                    network, points, targets, sensitivities, "M2"))
                self.assert_close(first.loss, sl.m1_loss_value(network, points, targets, S0=S0))
                self.assert_close(second.loss, sl.m2_loss_value(
                    network, points, targets, sensitivities, S0=S0, S1=S1))

    def test_all_49_components_of_each_defined_loss(self):
        for fixture, network in (("default", checks.moderate_network()),
                                  ("affine_output_scaled", checks.affine_control_network())):
            before = network.parameters.to_vector()
            for targets_name, points, targets, sensitivities in CONTROL_TARGETS:
                for method in ("M1", "M2"):
                    if method == "M1":
                        analytic = sl.m1_loss_and_gradient(network, points, targets, S0=S0)
                        evaluator = lambda net: sl.m1_loss_value(net, points, targets, S0=S0)
                    else:
                        analytic = sl.m2_loss_and_gradient(
                            network, points, targets, sensitivities, S0=S0, S1=S1)
                        evaluator = lambda net: sl.m2_loss_value(
                            net, points, targets, sensitivities, S0=S0, S1=S1)
                    for step in checks.PARAMETER_STEPS:
                        records = []
                        for i in range(49):
                            delta = step * max(1.0, abs(before[i]))
                            bound2, bound3 = loss_bounds(
                                network, points, targets, sensitivities, S0, S1, i, delta, method)
                            records.append(checks.fd_record(
                                network, i, step, evaluator, analytic.gradient[i], bound2, bound3))
                        row = dict(fixture=fixture, target_tuple=targets_name,
                                   method=method, step=step, sites=points,
                                   loss=analytic.loss, **checks.summary(records))
                        LOSS_FD_CONTROLS.append(row)
                        self.assertTrue(row["successful"], row)
            self.assertEqual(before, network.parameters.to_vector())

    def test_zero_residuals_give_zero_losses_and_gradients(self):
        network = checks.moderate_network()
        points = (0.04, 0.11, 0.18)
        targets = tuple(network.forward(p) for p in points)
        sensitivities = tuple(network.input_derivative(p) for p in points)
        first = sl.m1_loss_and_gradient(network, points, targets, S0=S0)
        second = sl.m2_loss_and_gradient(network, points, targets, sensitivities, S0=S0, S1=S1)
        self.assertEqual(first.loss, 0.0)
        self.assertEqual(second.loss, 0.0)
        self.assertEqual(first.gradient, (0.0,) * 49)
        self.assertEqual(second.gradient, (0.0,) * 49)

    def test_M2_equals_M1_at_fixed_zero_sensitivity_residuals(self):
        network = checks.moderate_network()
        points, targets = (0.04, 0.11, 0.18), (0.5, 0.3, 0.2)
        frozen_sensitivities = tuple(network.input_derivative(p) for p in points)
        first = sl.m1_loss_and_gradient(network, points, targets, S0=S0)
        second = sl.m2_loss_and_gradient(
            network, points, targets, frozen_sensitivities, S0=S0, S1=S1)
        self.assertEqual(second.sensitivity_component, 0.0)
        self.assertEqual(first.loss, second.loss)
        self.assertEqual(first.gradient, second.gradient)
        # This is a pointwise equality, not a zero-weight M2 variant.

    def test_sensitivity_term_does_not_depend_on_output_bias(self):
        network = checks.one_neuron(w=0.0, b=0.3, v=-0.7)
        p = 0.08
        target = network.forward(p)
        result = sl.m2_loss_and_gradient(
            network, (p,), (target,), (1.0,), S0=S0, S1=S1)
        self.assertEqual(result.value_component, 0.0)
        self.assertGreater(result.sensitivity_component, 0.0)
        self.assertEqual(result.gradient[48], 0.0)
        self.assertNotEqual(result.gradient[3], 0.0)

    def test_quadratic_control_primitive_against_a_scalar_polynomial(self):
        vf = tuple((i-24)/31.0 for i in range(49))
        vg = tuple((24-i)/47.0 for i in range(49))
        theta = tuple((i-20)/57.0 for i in range(49))
        f = 0.3 + math.fsum(x*y for x, y in zip(theta, vf))
        g = -0.2 + math.fsum(x*y for x, y in zip(theta, vg))
        A, B, target, sensitivity_target = 0.7, 1.2, 0.4, -0.6
        result = sl.quadratic_error_and_gradient(
            f, g, target, sensitivity_target, vf, vg, A=A, B=B)
        self.assert_close(result.loss, A*(f-target)**2+B*(g-sensitivity_target)**2)
        # Direct quadratic polynomial along each coordinate has exact centred slope.
        h = 1e-4
        for i in range(49):
            right = A*(f+h*vf[i]-target)**2+B*(g+h*vg[i]-sensitivity_target)**2
            left = A*(f-h*vf[i]-target)**2+B*(g-h*vg[i]-sensitivity_target)**2
            approximation = (right-left)/(2*h)
            tolerance = 32*math.ulp(1.0)*(1+abs(left)+abs(right))/(2*h)+checks.FD_ABS_FLOOR
            self.assertLessEqual(abs(result.gradient[i]-approximation), tolerance)

    def test_zero_quadratic_weights_are_allowed_only_for_generic_primitive(self):
        vf, vg = (1.0,) * 49, (2.0,) * 49
        zero = sl.quadratic_error_and_gradient(
            1e308, 1.0, -1e308, -2.0, vf, vg, A=0.0, B=0.0)
        self.assertEqual(zero.loss, 0.0)
        self.assertEqual(zero.gradient, (0.0,) * 49)
        value_only = sl.quadratic_error_and_gradient(2, 3, 1, -1, vf, vg, A=0.5, B=0)
        self.assertEqual(value_only.loss, 0.5)
        self.assertEqual(value_only.gradient, vf)
        with self.assertRaises(TypeError):
            sl.m2_loss_and_gradient(checks.moderate_network(), (0.08,), (0.4,), (-1.0,),
                                      S0=S0, S1=S1, sensitivity_weight=0.0)

    def test_parameters_affine_constants_and_target_lists_are_preserved(self):
        network = checks.affine_control_network()
        points, targets, sensitivities = [0.04, 0.11, 0.18], [0.5, 0.3, 0.2], [-2, -1.5, -0.8]
        snapshot = (network.parameters.to_vector(), network.normalization,
                    points.copy(), targets.copy(), sensitivities.copy())
        sl.m1_loss_and_gradient(network, points, targets, S0=S0)
        sl.m2_loss_and_gradient(network, points, targets, sensitivities, S0=S0, S1=S1)
        self.assertEqual(snapshot, (network.parameters.to_vector(), network.normalization,
                                    points, targets, sensitivities))

    def test_empty_mismatched_or_missing_targets_are_rejected(self):
        network = checks.moderate_network()
        for function in (sl.m1_loss_value, sl.m1_loss_and_gradient):
            for points, targets in (((), ()), ((0.08,), ()), ((0.08,), (1, 2))):
                with self.assertRaises(ValueError):
                    function(network, points, targets, S0=S0)
        for function in (sl.m2_loss_value, sl.m2_loss_and_gradient):
            for sensitivities in (None, (), (1, 2)):
                with self.assertRaises(ValueError):
                    function(network, (0.08,), (0.4,), sensitivities, S0=S0, S1=S1)
        with self.assertRaises(TypeError):
            sl.m1_loss_value(network, None, (0.4,), S0=S0)
        with self.assertRaises(TypeError):
            sl.m1_loss_value(network, (0.08,), "0.4", S0=S0)

    def test_invalid_targets_domains_and_scales_are_rejected_before_evaluation(self):
        network = checks.moderate_network()
        for bad in (math.nan, math.inf, -math.inf):
            with self.assertRaises(ValueError):
                sl.m1_loss_value(network, (0.08,), (bad,), S0=S0)
            with self.assertRaises(ValueError):
                sl.m2_loss_and_gradient(network, (0.08,), (0.4,), (bad,), S0=S0, S1=S1)
        for bad in (0.0, -1.0, math.nan, math.inf):
            with self.assertRaises(ValueError):
                sl.m1_loss_and_gradient(network, (0.08,), (0.4,), S0=bad)
            with self.assertRaises(ValueError):
                sl.m2_loss_and_gradient(network, (0.08,), (0.4,), (-1.0,), S0=S0, S1=bad)
        for bad in (True, "1", None):
            with self.assertRaises(TypeError):
                sl.m1_loss_value(network, (0.08,), (0.4,), S0=bad)
        for p in (0.0, math.nan, math.nextafter(n.B, math.inf)):
            with patch.object(ng, "value_and_parameter_gradient",
                              side_effect=AssertionError("evaluation before validation")):
                with self.assertRaises(ValueError):
                    sl.m1_loss_and_gradient(network, (0.08, p), (0.4, 0.2), S0=S0)

    def test_invalid_generic_coefficients_vector_sizes_and_values(self):
        vf, vg = (1.0,) * 49, (2.0,) * 49
        for A, B in ((-1, 1), (1, -1), (math.nan, 1), (1, math.inf)):
            with self.assertRaises(ValueError):
                sl.quadratic_error_and_gradient(1, 2, 0, 0, vf, vg, A=A, B=B)
        for size in (48, 50):
            with self.assertRaises(ValueError):
                sl.quadratic_error_and_gradient(1, 2, 0, 0, [0]*size, vg, A=1, B=1)
        invalid = list(vf)
        invalid[32] = math.nan
        with self.assertRaises(ValueError):
            sl.quadratic_error_and_gradient(1, 2, 0, 0, invalid, vg, A=0, B=0)
        with self.assertRaises(TypeError):
            sl.quadratic_error_and_gradient(1, 2, 0, 0, vf, vg, A=True, B=1)
        with self.assertRaises(ValueError):
            sl.quadratic_error_and_gradient(math.inf, 2, 0, 0, vf, vg, A=0, B=0)

    def test_nonfinite_residual_loss_or_gradient_is_an_explicit_error(self):
        network = checks.moderate_network()
        for function in (sl.m1_loss_value, sl.m1_loss_and_gradient):
            with self.assertRaises(FloatingPointError):
                function(network, (0.08,), (1e308,), S0=S0)
            with self.assertRaises(FloatingPointError):
                function(network, (0.08,), (0.4,), S0=1e-300)
        with self.assertRaises(FloatingPointError):
            sl.quadratic_error_and_gradient(
                1e308, 0, 0, 0, (1.0,)*49, (0.0,)*49, A=1, B=0)


if __name__ == "__main__":
    unittest.main()
