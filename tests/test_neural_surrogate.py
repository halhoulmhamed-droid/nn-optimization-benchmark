"""Deterministic control fixtures only: none is trained or fitted to oracle data."""
from dataclasses import FrozenInstanceError
import math
import sys
import unittest

from src import neural_surrogate as n

VALUE_REL_TOL = 3e-14
VALUE_ABS_TOL = 2e-14
DERIVATIVE_ABS_TOL = 2e-12
NORMALIZATION_ABS_TOL = 5e-15
ROUNDING_ALLOWANCE_FACTOR = 32.0
INTERIOR_POINTS = (0.05, 0.11, 0.17)
STABLE_STEPS = (1e-3, 1e-4, 1e-5)
SMALL_STEPS = (1e-8, 1e-10, 1e-12)
BOUNDARY_STEPS = (1e-4, 1e-5)


def single_active(index=3, w=0.8, b=-0.3, v=-1.25, d=0.15,
                  normalization=n.DEFAULT_NORMALIZATION):
    weights, biases, outputs = [0.0] * 16, [0.0] * 16, [0.0] * 16
    weights[index], biases[index], outputs[index] = w, b, v
    return n.NeuralSurrogate(n.NetworkParameters(weights, biases, outputs, d),
                             normalization)


def multi_active():
    return n.NeuralSurrogate(n.NetworkParameters(
        [0.2 + 0.03 * i for i in range(16)],
        [-0.25 + 0.025 * i for i in range(16)],
        [0.05 + 0.003 * i for i in range(16)], -0.1))


def independent_expression(network, p):
    # sinh/cosh and 1/cosh^2 are distinct expressions from tanh and 1-h^2.
    # Used only for moderate arguments; large saturation has its own control.
    norm, params = network.normalization, network.parameters
    z = norm.alpha * p + norm.beta
    arguments = [w * z + b for w, b in zip(params.w, params.b)]
    y = params.d + sum(v * math.sinh(q) / math.cosh(q)
                       for v, q in zip(params.v, arguments))
    dy_dz = sum(v * w / math.cosh(q) ** 2
                for v, w, q in zip(params.v, params.w, arguments))
    return norm.c_out + norm.s_out * y, norm.s_out * norm.alpha * dy_dz


def third_derivative_bound(network):
    # Analysis for FD truncation only; no second/third network derivative API.
    # tanh'''(q)=2*(1-t^2)*(3*t^2-1), hence |tanh'''| <= 2.
    norm, params = network.normalization, network.parameters
    return (2.0 * abs(norm.s_out) * abs(norm.alpha) ** 3
            * math.fsum(abs(v) * abs(w) ** 3
                        for v, w in zip(params.v, params.w)))


def finite_difference_record(network, p, h, stencil="central"):
    if stencil == "central":
        sites = (p - h, p + h)
        coefficients = (-1.0, 1.0)
        truncation_factor = 1.0 / 6.0
    elif stencil == "forward_second_order":
        sites = (p, p + h, p + 2.0 * h)
        coefficients = (-3.0, 4.0, -1.0)
        truncation_factor = 1.0 / 3.0
    elif stencil == "backward_second_order":
        sites = (p, p - h, p - 2.0 * h)
        coefficients = (3.0, -4.0, 1.0)
        truncation_factor = 1.0 / 3.0
    else:
        raise ValueError("unknown finite-difference stencil")
    values = [network.forward(site) for site in sites]
    approximation = math.fsum(c * value for c, value in
                              zip(coefficients, values)) / (2.0 * h)
    analytic = network.input_derivative(p)
    truncation = truncation_factor * third_derivative_bound(network) * h * h
    rounding = (ROUNDING_ALLOWANCE_FACTOR * sys.float_info.epsilon
                * (1.0 + math.fsum(abs(c * value) for c, value in
                                  zip(coefficients, values))) / (2.0 * h))
    tolerance = truncation + rounding + DERIVATIVE_ABS_TOL
    return {
        "p": p, "h": h, "stencil": stencil, "sampled_sites": sites,
        "approximation": approximation, "analytic_float": analytic,
        "absolute_error": abs(approximation - analytic),
        "truncation_upper_expression": truncation,
        "rounding_allowance_heuristic": rounding, "tolerance": tolerance,
        "passed": abs(approximation - analytic) <= tolerance,
        "certified": False,
    }


class NeuralSurrogateTests(unittest.TestCase):
    def assert_close(self, actual, expected, abs_tol=VALUE_ABS_TOL):
        self.assertTrue(math.isclose(actual, expected, rel_tol=VALUE_REL_TOL,
                                    abs_tol=abs_tol), (actual, expected))

    def test_exact_parameter_count_order_and_round_trip(self):
        vector = tuple((i - 24) / 37.0 for i in range(49))
        parameters = n.NetworkParameters.from_vector(vector)
        self.assertEqual(n.HIDDEN_UNITS, 16)
        self.assertEqual(n.PARAMETER_COUNT, 49)
        self.assertEqual(parameters.w, vector[:16])
        self.assertEqual(parameters.b, vector[16:32])
        self.assertEqual(parameters.v, vector[32:48])
        self.assertEqual(parameters.d, vector[48])
        self.assertEqual(parameters.to_vector(), vector)
        self.assertEqual(n.NetworkParameters.from_vector(parameters.to_vector()),
                         parameters)

    def test_parameter_order_controls_last_neuron_and_output_bias(self):
        vector = [0.0] * 49
        vector[15], vector[31], vector[47], vector[48] = 0.7, -0.4, 1.3, -0.2
        network = n.NeuralSurrogate(n.NetworkParameters.from_vector(vector))
        # Independent point arithmetic catches exchanging w, b, v or d.
        for p in (n.A, 0.13, n.B):
            q = 0.7 * (2.0 * (p - 0.02) / (0.20 - 0.02) - 1.0) - 0.4
            self.assert_close(network.forward(p),
                              -0.2 + 1.3 * math.sinh(q) / math.cosh(q))
            self.assert_close(network.input_derivative(p),
                              (2.0 / (0.20 - 0.02)) * 1.3 * 0.7
                              / math.cosh(q) ** 2, DERIVATIVE_ABS_TOL)

    def test_parameters_copy_inputs_and_are_immutable(self):
        weights = [0.0] * 16
        parameters = n.NetworkParameters(weights, weights, weights, 0.5)
        weights[0] = 7.0
        self.assertEqual(parameters.to_vector(), (0.0,) * 48 + (0.5,))
        with self.assertRaises(FrozenInstanceError):
            parameters.d = 1.0

    def test_zero_input_weights_make_a_constant(self):
        network = n.NeuralSurrogate(n.NetworkParameters(
            [0.0] * 16, [(i - 8) / 16.0 for i in range(16)],
            [(i + 1) / 31.0 for i in range(16)], -0.7))
        expected, _ = independent_expression(network, 0.10)
        for p in (n.A, 0.11, n.B):
            self.assert_close(network.forward(p), expected)
            self.assertEqual(network.input_derivative(p), 0.0)

    def test_zero_output_weights_leave_only_output_bias(self):
        network = n.NeuralSurrogate(n.NetworkParameters(
            [0.3 + i / 20.0 for i in range(16)], [0.2] * 16,
            [0.0] * 16, -0.375))
        for p in (n.A, 0.11, n.B):
            self.assertEqual(network.forward(p), -0.375)
            self.assertEqual(network.input_derivative(p), 0.0)

    def test_single_active_neuron_independent_value_and_derivative(self):
        network = single_active()
        for p in (n.A, 0.04, 0.10, 0.18, n.B):
            with self.subTest(p=p):
                expected_value, expected_derivative = independent_expression(network, p)
                self.assert_close(network.forward(p), expected_value)
                self.assert_close(network.input_derivative(p), expected_derivative,
                                  DERIVATIVE_ABS_TOL)

    def test_active_tanh_argument_zero_has_nonzero_physical_derivative(self):
        p, w, v, d = 0.125, 1.5, -0.8, 0.3
        norm = n.DEFAULT_NORMALIZATION
        b = -w * (norm.alpha * p + norm.beta)
        network = single_active(w=w, b=b, v=v, d=d)
        result = network.evaluate(p)
        self.assertEqual(result.activations[3], 0.0)
        self.assert_close(result.value, d)
        self.assert_close(result.normalized_derivative, v * w,
                          DERIVATIVE_ABS_TOL)
        self.assert_close(result.input_derivative, (2.0 / (n.B - n.A)) * v * w,
                          DERIVATIVE_ABS_TOL)

    def test_default_normalization_boundaries_and_interior(self):
        norm = n.DEFAULT_NORMALIZATION
        self.assertEqual(norm.alpha, 2.0 / (n.B - n.A))
        self.assertEqual((norm.c_out, norm.s_out), (0.0, 1.0))
        for p, expected in ((n.A, -1.0), (0.11, 0.0), (n.B, 1.0)):
            self.assert_close(norm.normalize_input(p), expected,
                              NORMALIZATION_ABS_TOL)

    def test_output_scale_sign_and_offset_are_explicit(self):
        norm = n.AffineNormalization(c_out=0.7, s_out=-2.5)
        network = single_active(w=1.1, b=-0.2, v=-0.7, normalization=norm)
        for p in (n.A, 0.10, n.B):
            value, derivative = independent_expression(network, p)
            self.assert_close(network.forward(p), value)
            self.assert_close(network.input_derivative(p), derivative,
                              DERIVATIVE_ABS_TOL)
            self.assertGreater(derivative, 0.0)

    def test_distinct_affine_fixture_checks_the_full_chain_rule(self):
        norm = n.AffineNormalization(alpha=-3.0, beta=0.4,
                                     c_out=-0.9, s_out=2.75)
        network = single_active(w=-0.6, b=0.2, v=1.2, normalization=norm)
        for p in (n.A, 0.125, n.B):
            value, derivative = independent_expression(network, p)
            result = network.evaluate(p)
            self.assert_close(result.value, value)
            self.assert_close(result.input_derivative, derivative,
                              DERIVATIVE_ABS_TOL)
            self.assert_close(result.input_derivative,
                              norm.s_out * norm.alpha * result.normalized_derivative,
                              DERIVATIVE_ABS_TOL)
            self.assertGreater(result.input_derivative, 0.0)

    def test_zero_fixed_scales_have_the_expected_constant_effect(self):
        for norm in (n.AffineNormalization(alpha=0.0, beta=0.3),
                     n.AffineNormalization(c_out=-0.25, s_out=0.0)):
            network = single_active(normalization=norm)
            first = network.forward(n.A)
            for p in (0.11, n.B):
                self.assertEqual(network.forward(p), first)
                self.assertEqual(network.input_derivative(p), 0.0)

    def test_joint_evaluation_agrees_with_separate_calls(self):
        network = multi_active()
        for p in INTERIOR_POINTS:
            result = network.evaluate(p)
            self.assertEqual(len(result.activations), 16)
            self.assertEqual(result.value, network.forward(p))
            self.assertEqual(result.input_derivative, network.input_derivative(p))
            self.assert_close(result.input_derivative,
                              n.INPUT_ALPHA * result.normalized_derivative,
                              DERIVATIVE_ABS_TOL)

    def test_parameter_sizes_are_rejected(self):
        for size in (0, 48, 50):
            with self.assertRaises(ValueError):
                n.NetworkParameters.from_vector([0.0] * size)
        for name in ("w", "b", "v"):
            for size in (15, 17):
                groups = dict(w=[0.0] * 16, b=[0.0] * 16, v=[0.0] * 16, d=0.0)
                groups[name] = [0.0] * size
                with self.assertRaises(ValueError):
                    n.NetworkParameters(**groups)
        for bad in (None, 1.0, "0" * 49):
            with self.assertRaises(TypeError):
                n.NetworkParameters.from_vector(bad)

    def test_nonfinite_parameters_and_invalid_scalar_types_are_rejected(self):
        for index in (0, 16, 32, 48):
            for bad in (math.nan, math.inf, -math.inf, 10 ** 400):
                vector = [0.0] * 49
                vector[index] = bad
                with self.subTest(index=index, bad=str(bad)):
                    with self.assertRaises(ValueError):
                        n.NetworkParameters.from_vector(vector)
            for bad in (True, None, "1", complex(1, 0)):
                vector = [0.0] * 49
                vector[index] = bad
                with self.assertRaises(TypeError):
                    n.NetworkParameters.from_vector(vector)

    def test_affine_constants_must_be_finite_real_scalars(self):
        for name in ("alpha", "beta", "c_out", "s_out"):
            for bad in (math.nan, math.inf, -math.inf, 10 ** 400):
                with self.assertRaises(ValueError):
                    n.AffineNormalization(**{name: bad})
            for bad in (True, None, "1", complex(1, 0)):
                with self.assertRaises(TypeError):
                    n.AffineNormalization(**{name: bad})
        with self.assertRaises(TypeError):
            n.NeuralSurrogate([0.0] * 49)
        with self.assertRaises(TypeError):
            n.NeuralSurrogate(single_active().parameters, normalization=None)

    def test_input_domain_and_types_are_checked_by_every_public_evaluator(self):
        network = single_active()
        functions = (network.forward, network.input_derivative, network.evaluate,
                     network.normalization.normalize_input)
        for function in functions:
            for p in (math.nan, math.inf, -math.inf, 0.0,
                      math.nextafter(n.A, -math.inf),
                      math.nextafter(n.B, math.inf), 10 ** 400):
                with self.assertRaises(ValueError):
                    function(p)
            for p in (True, None, "0.1", complex(0.1, 0)):
                with self.assertRaises(TypeError):
                    function(p)

    def test_nonfinite_output_and_sums_raise_instead_of_clipping(self):
        network = n.NeuralSurrogate(n.NetworkParameters(
            [0.0] * 16, [1.0] * 16, [1e308] * 16, 0.0))
        for function in (network.forward, network.evaluate):
            with self.assertRaises(FloatingPointError):
                function(0.10)
        scaled = single_active(normalization=n.AffineNormalization(s_out=1e308),
                               d=10.0)
        with self.assertRaises(FloatingPointError):
            scaled.forward(0.10)

    def test_nonfinite_derivative_raises_with_finite_forward_value(self):
        norm = n.AffineNormalization(alpha=8.0, beta=-1.0)
        network = single_active(w=1e308, b=0.0, v=1e308, d=0.0,
                                normalization=norm)
        self.assertEqual(network.forward(0.125), 0.0)
        for function in (network.input_derivative, network.evaluate):
            with self.assertRaises(FloatingPointError):
                function(0.125)

    def test_nonfinite_affine_input_and_preactivation_are_explicit_errors(self):
        normal = n.AffineNormalization(alpha=1e308, beta=1.7e308)
        with self.assertRaises(FloatingPointError):
            single_active(normalization=normal).forward(n.B)
        network = single_active(w=1e308, normalization=n.AffineNormalization(
            alpha=1e308, beta=0.0))
        for function in (network.forward, network.input_derivative, network.evaluate):
            with self.assertRaises(FloatingPointError):
                function(n.B)

    def test_saturation_zero_is_a_floating_effect_not_an_exact_identity(self):
        network = single_active(w=20.0, b=0.0, v=1.0, d=0.0)
        for p, sign in ((n.A, -1.0), (n.B, 1.0)):
            result = network.evaluate(p)
            self.assertEqual(result.activations[3], sign)
            self.assertEqual(result.input_derivative, 0.0)
            q = 20.0 * result.normalized_input
            # This alternative expression still represents the positive factor.
            exp_small = math.exp(-2.0 * abs(q))
            positive_factor = 4.0 * exp_small / (1.0 + exp_small) ** 2
            self.assertGreater(positive_factor, 0.0)
        extreme = single_active(w=1000.0, b=0.0, v=1.0, d=0.0)
        self.assertEqual(extreme.input_derivative(n.B), 0.0)
        self.assertTrue(math.isfinite(extreme.forward(n.B)))

    def test_central_differences_on_nontrivial_interior_derivatives(self):
        network = multi_active()
        for p in INTERIOR_POINTS:
            self.assertGreater(abs(network.input_derivative(p)), 1.0)
            for h in STABLE_STEPS:
                record = finite_difference_record(network, p, h)
                with self.subTest(p=p, h=h):
                    self.assertTrue(record["passed"], record)

    def test_small_steps_do_not_require_monotone_error_decrease(self):
        network = multi_active()
        for p in INTERIOR_POINTS:
            for h in SMALL_STEPS:
                record = finite_difference_record(network, p, h)
                with self.subTest(p=p, h=h):
                    self.assertTrue(record["passed"], record)

    def test_boundary_second_order_stencils_stay_inside_domain(self):
        network = multi_active()
        for p, stencil in ((n.A, "forward_second_order"),
                           (n.B, "backward_second_order")):
            expected_value, expected_derivative = independent_expression(network, p)
            self.assert_close(network.forward(p), expected_value)
            self.assert_close(network.input_derivative(p), expected_derivative,
                              DERIVATIVE_ABS_TOL)
            for h in BOUNDARY_STEPS:
                record = finite_difference_record(network, p, h, stencil)
                self.assertTrue(all(n.A <= site <= n.B
                                    for site in record["sampled_sites"]))
                self.assertTrue(record["passed"], record)


if __name__ == "__main__":
    unittest.main()
