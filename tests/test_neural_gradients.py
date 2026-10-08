"""Independent expressions and complete component checks; no learned parameters."""
import math
import sys
import unittest

from src import neural_surrogate as n
from src import neural_gradients as ng

ANALYTIC_REL_TOL = 5e-14
VALUE_GRAD_ABS_TOL = 3e-13
SENSITIVITY_GRAD_ABS_TOL = 4e-12
FD_ABS_FLOOR = 4e-12
ROUNDING_FACTOR = 32.0
PARAMETER_STEPS = (1e-3, 1e-4, 1e-5)
SMALL_STEPS = (1e-8, 1e-10, 1e-12)
POINTS = (n.A, 0.08, 0.11, n.B)
BLOCKS = {"w": range(0, 16), "b": range(16, 32), "v": range(32, 48), "d": range(48, 49)}
GRADIENT_FD_CONTROLS = []
INDEPENDENT_CONTROLS = []
MIXED_CONTROLS = []
SMALL_STEP_CONTROLS = []
SATURATION_CONTROLS = []
NONREPRESENTABLE_CONTROLS = []


def moderate_network(normalization=n.DEFAULT_NORMALIZATION):
    return n.NeuralSurrogate(n.NetworkParameters(
        [0.12 + 0.04 * i for i in range(16)],
        [-0.3 + 0.035 * i for i in range(16)],
        [(-1.0) ** i * (0.15 + 0.006 * i) for i in range(16)], 0.2),
        normalization)


def affine_control_network():
    return moderate_network(n.AffineNormalization(
        alpha=-3.0, beta=0.4, c_out=0.7, s_out=-2.75))


def one_neuron(w=0.8, b=-0.3, v=-1.25, d=0.15, index=3,
               normalization=n.DEFAULT_NORMALIZATION):
    weights, biases, outputs = [0.0] * 16, [0.0] * 16, [0.0] * 16
    weights[index], biases[index], outputs[index] = w, b, v
    return n.NeuralSurrogate(n.NetworkParameters(weights, biases, outputs, d),
                             normalization)


def perturbed_pair(network, index, base_step):
    vector = network.parameters.to_vector()
    requested = base_step * max(1.0, abs(vector[index]))
    plus, minus = list(vector), list(vector)
    plus[index], minus[index] = vector[index] + requested, vector[index] - requested
    dp, dm = plus[index] - vector[index], vector[index] - minus[index]
    metadata = dict(index=index, base_step=base_step, requested_delta=requested,
                    actual_delta_plus=dp, actual_delta_minus=dm,
                    original_parameter=vector[index])
    if dp <= 0.0 or dm <= 0.0:
        return None, None, dict(metadata, status="PERTURBATION_NOT_REPRESENTABLE",
                               finite_difference_executed=False)
    return (
        n.NeuralSurrogate(n.NetworkParameters.from_vector(plus), network.normalization),
        n.NeuralSurrogate(n.NetworkParameters.from_vector(minus), network.normalization),
        dict(metadata, status="REPRESENTABLE", finite_difference_executed=True),
    )


def parameter_bounds(network, p, index, delta):
    """Coefficient envelopes only, for test truncation; no derivative API."""
    norm, params = network.normalization, network.parameters
    z = norm.alpha * p + norm.beta
    weights, outputs = list(map(abs, params.w)), list(map(abs, params.v))
    bias_bound = abs(params.d)
    if index < 16:
        weights[index] += delta
    elif 32 <= index < 48:
        outputs[index - 32] += delta
    elif index == 48:
        bias_bound += delta
    scale, mixed = abs(norm.s_out), abs(norm.s_out * norm.alpha)
    f0 = abs(norm.c_out) + scale * (bias_bound + math.fsum(outputs))
    g0 = mixed * math.fsum(v * w for v, w in zip(outputs, weights))
    if index < 16:
        w, v = weights[index], outputs[index]
        f = (f0, scale * v * abs(z), 2 * scale * v * z * z,
             2 * scale * v * abs(z) ** 3)
        g = (g0, mixed * v * (1 + 2 * w * abs(z)),
             mixed * v * (4 * abs(z) + 2 * w * z * z),
             mixed * v * (6 * z * z + 80 * w * abs(z) ** 3))
    elif index < 32:
        i = index - 16
        f = (f0, scale * outputs[i], 2 * scale * outputs[i], 2 * scale * outputs[i])
        g = (g0, 2 * mixed * outputs[i] * weights[i],
             2 * mixed * outputs[i] * weights[i],
             80 * mixed * outputs[i] * weights[i])
    elif index < 48:
        i = index - 32
        f = (f0, scale, 0.0, 0.0)
        g = (g0, mixed * weights[i], 0.0, 0.0)
    else:
        f, g = (f0, scale, 0.0, 0.0), (g0, 0.0, 0.0, 0.0)
    return f, g


def fd_record(network, index, step, evaluator, analytic, second_bound, third_bound):
    plus, minus, metadata = perturbed_pair(network, index, step)
    if plus is None:
        return metadata
    right, left = evaluator(plus), evaluator(minus)
    dp, dm = metadata["actual_delta_plus"], metadata["actual_delta_minus"]
    denominator = dp + dm
    approximation = (right - left) / denominator
    truncation = third_bound * max(dp, dm) ** 2 / 6.0
    asymmetry = second_bound * abs(dp - dm) / 2.0
    rounding = ROUNDING_FACTOR * sys.float_info.epsilon * (
        1.0 + abs(left) + abs(right)) / denominator
    tolerance = truncation + asymmetry + rounding + FD_ABS_FLOOR
    error = abs(approximation - analytic)
    relative_scale = max(abs(approximation), abs(analytic), 1.0)
    return dict(metadata, analytic=analytic, approximation=approximation,
                absolute_error=error, scaled_error=error / relative_scale,
                relative_scale=relative_scale, scaled_tolerance=tolerance / relative_scale,
                truncation_expression=truncation,
                asymmetry_expression=asymmetry, rounding_allowance_heuristic=rounding,
                tolerance=tolerance, passed=error <= tolerance, certified=False)


def summary(records):
    valid = [row for row in records if row["finite_difference_executed"]]
    failures = [row for row in valid if not row["passed"]]
    return dict(
        checked_components=len(valid),
        all_49_indices_checked=sorted(row["index"] for row in valid) == list(range(49)),
        successful=len(valid) == 49 and not failures,
        errors_by_block={
            block: dict(
                components=sum(row["index"] in indices for row in valid),
                max_absolute_error=max((row["absolute_error"] for row in valid
                                        if row["index"] in indices), default=0.0),
                max_scaled_error=max((row["scaled_error"] for row in valid
                                      if row["index"] in indices), default=0.0),
                max_error_over_tolerance=max((row["absolute_error"] / row["tolerance"]
                    for row in valid if row["index"] in indices), default=0.0))
            for block, indices in BLOCKS.items()
        },
        absolute_errors_by_component=[row.get("absolute_error") for row in records],
        failures=failures, nonrepresentable=[row for row in records
                                            if not row["finite_difference_executed"]],
    )


def independent_one_neuron(network, p, index=3):
    norm, pars = network.normalization, network.parameters
    z = norm.alpha * p + norm.beta
    w, b, v = pars.w[index], pars.b[index], pars.v[index]
    u = w * z + b
    sinh, cosh = math.sinh(u), math.cosh(u)
    sech2 = 1.0 / cosh ** 2
    df, dg = [0.0] * 49, [0.0] * 49
    df[index] = norm.s_out * v * z * sech2
    df[16 + index] = norm.s_out * v * sech2
    df[32 + index] = norm.s_out * sinh / cosh
    df[48] = norm.s_out
    dg[index] = norm.s_out * norm.alpha * v * (
        sech2 - 2.0 * w * z * sinh / cosh ** 3)
    dg[16 + index] = -2.0 * norm.s_out * norm.alpha * v * w * sinh / cosh ** 3
    dg[32 + index] = norm.s_out * norm.alpha * w * sech2
    return tuple(df), tuple(dg)


class NeuralGradientTests(unittest.TestCase):
    def assert_vector_close(self, actual, expected, abs_tol):
        self.assertEqual(len(actual), 49)
        for i, (left, right) in enumerate(zip(actual, expected)):
            with self.subTest(index=i):
                self.assertTrue(math.isclose(left, right, rel_tol=ANALYTIC_REL_TOL,
                                             abs_tol=abs_tol), (i, left, right))

    def test_independent_one_active_neuron_all_four_blocks(self):
        for norm in (n.DEFAULT_NORMALIZATION,
                     n.AffineNormalization(alpha=-3.0, beta=0.4,
                                           c_out=0.7, s_out=-2.75)):
            network = one_neuron(normalization=norm)
            for p in (n.A, 0.08, n.B):
                result = ng.evaluate_parameter_gradients(network, p)
                expected_f, expected_g = independent_one_neuron(network, p)
                self.assert_vector_close(result.value_gradient, expected_f, VALUE_GRAD_ABS_TOL)
                self.assert_vector_close(result.sensitivity_gradient, expected_g,
                                         SENSITIVITY_GRAD_ABS_TOL)
                INDEPENDENT_CONTROLS.append(dict(
                    p=p, normalization=(norm.alpha, norm.beta, norm.c_out, norm.s_out),
                    value_max_abs_error=max(abs(x-y) for x, y in
                                            zip(result.value_gradient, expected_f)),
                    sensitivity_max_abs_error=max(abs(x-y) for x, y in
                                                   zip(result.sensitivity_gradient, expected_g)),
                    checked_components_per_vector=49, passed=True))

    def test_argument_and_z_zero_with_nonunit_output_scale(self):
        norm = n.AffineNormalization(alpha=8.0, beta=-1.0, c_out=0.7, s_out=-2.5)
        network = one_neuron(w=0.8, b=0.0, v=-1.25, normalization=norm)
        result = ng.evaluate_parameter_gradients(network, 0.125)
        self.assertEqual(result.forward.normalized_input, 0.0)
        self.assertEqual(result.forward.activations[3], 0.0)
        self.assertEqual(result.value_gradient[3], 0.0)
        self.assertEqual(result.value_gradient[19], -2.5 * -1.25)
        self.assertEqual(result.value_gradient[35], 0.0)
        self.assertEqual(result.value_gradient[48], -2.5)
        self.assertEqual(result.sensitivity_gradient[3], -2.5 * 8.0 * -1.25)
        self.assertEqual(result.sensitivity_gradient[19], 0.0)
        self.assertEqual(result.sensitivity_gradient[35], -2.5 * 8.0 * 0.8)
        self.assertEqual(result.sensitivity_gradient[48], 0.0)

    def test_zero_w_does_not_annul_sensitivity_w_gradient(self):
        network = one_neuron(w=0.0, b=0.3, v=-0.7)
        result = ng.evaluate_parameter_gradients(network, 0.08)
        self.assertEqual(result.forward.input_derivative, 0.0)
        expected = n.INPUT_ALPHA * -0.7 / math.cosh(0.3) ** 2
        self.assertTrue(math.isclose(result.sensitivity_gradient[3], expected,
                                    rel_tol=ANALYTIC_REL_TOL, abs_tol=SENSITIVITY_GRAD_ABS_TOL))
        self.assertNotEqual(result.sensitivity_gradient[3], 0.0)

    def test_zero_v_does_not_annul_value_or_sensitivity_v_gradient(self):
        network = one_neuron(w=0.8, b=-0.3, v=0.0)
        result = ng.evaluate_parameter_gradients(network, 0.08)
        self.assertEqual(result.forward.value, 0.15)
        self.assertEqual(result.forward.input_derivative, 0.0)
        self.assertNotEqual(result.value_gradient[35], 0.0)
        self.assertNotEqual(result.sensitivity_gradient[35], 0.0)
        expected_f, expected_g = independent_one_neuron(network, 0.08)
        self.assert_vector_close(result.value_gradient, expected_f, VALUE_GRAD_ABS_TOL)
        self.assert_vector_close(result.sensitivity_gradient, expected_g, SENSITIVITY_GRAD_ABS_TOL)

    def test_separate_and_joint_calls_match_preserved_public_network(self):
        network = moderate_network()
        for p in POINTS:
            joint = ng.evaluate_parameter_gradients(network, p)
            f = ng.value_and_parameter_gradient(network, p)
            g = ng.sensitivity_and_parameter_gradient(network, p)
            self.assertEqual(joint.forward.value, network.forward(p))
            self.assertEqual(joint.forward.input_derivative, network.input_derivative(p))
            self.assertEqual((f.value, f.gradient),
                             (joint.forward.value, joint.value_gradient))
            self.assertEqual((g.value, g.gradient),
                             (joint.forward.input_derivative, joint.sensitivity_gradient))

    def test_all_49_components_against_public_parameter_perturbations(self):
        for fixture, network in (("default", moderate_network()),
                                  ("affine_output_scaled", affine_control_network())):
            before = network.parameters.to_vector()
            for p in POINTS:
                result = ng.evaluate_parameter_gradients(network, p)
                for kind, vector in (("value", result.value_gradient),
                                     ("physical_sensitivity", result.sensitivity_gradient)):
                    for step in PARAMETER_STEPS:
                        records = []
                        for i in range(49):
                            delta = step * max(1.0, abs(before[i]))
                            f_bounds, g_bounds = parameter_bounds(network, p, i, delta)
                            bounds = f_bounds if kind == "value" else g_bounds
                            callback = (lambda net: net.forward(p)) if kind == "value" else (
                                lambda net: net.input_derivative(p))
                            records.append(fd_record(network, i, step, callback, vector[i],
                                                     bounds[2], bounds[3]))
                        row = dict(fixture=fixture, p=p, quantity=kind, step=step,
                                   **summary(records))
                        GRADIENT_FD_CONTROLS.append(row)
                        self.assertTrue(row["successful"], row)
            self.assertEqual(before, network.parameters.to_vector())

    def test_mixed_derivative_by_differentiating_value_parameter_gradient_in_p(self):
        network, p = moderate_network(), 0.08
        exact = ng.sensitivity_and_parameter_gradient(network, p).gradient
        norm, pars = network.normalization, network.parameters
        for h in (1e-4, 1e-5):
            left = ng.value_and_parameter_gradient(network, p-h).gradient
            right = ng.value_and_parameter_gradient(network, p+h).gradient
            zmax = max(abs(norm.alpha*(p-h)+norm.beta), abs(norm.alpha*(p+h)+norm.beta))
            records = []
            for i in range(49):
                if i < 16:
                    w, v = abs(pars.w[i]), abs(pars.v[i])
                    bound = abs(norm.s_out)*v*abs(norm.alpha)**3*(6*w*w+80*zmax*w**3)
                elif i < 32:
                    j = i-16
                    bound = 80*abs(norm.s_out*pars.v[j])*abs(norm.alpha*pars.w[j])**3
                elif i < 48:
                    bound = 2*abs(norm.s_out)*abs(norm.alpha*pars.w[i-32])**3
                else:
                    bound = 0.0
                approximation = (right[i]-left[i])/(2*h)
                tolerance = bound*h*h/6 + ROUNDING_FACTOR*sys.float_info.epsilon*(
                    1+abs(left[i])+abs(right[i]))/(2*h) + FD_ABS_FLOOR
                records.append(dict(index=i, absolute_error=abs(approximation-exact[i]),
                                    tolerance=tolerance,
                                    passed=abs(approximation-exact[i]) <= tolerance))
            row = dict(p=p, h=h, components=49, records=records,
                       successful=all(x["passed"] for x in records))
            MIXED_CONTROLS.append(row)
            self.assertTrue(row["successful"], row)

    def test_small_parameter_steps_preserve_roundoff_observations(self):
        network, p = moderate_network(), 0.08
        result = ng.evaluate_parameter_gradients(network, p)
        for kind, vector in (("value", result.value_gradient),
                             ("physical_sensitivity", result.sensitivity_gradient)):
            for i in (0, 16, 32, 48):
                for step in SMALL_STEPS:
                    delta = step * max(1.0, abs(network.parameters.to_vector()[i]))
                    fb, gb = parameter_bounds(network, p, i, delta)
                    bounds = fb if kind == "value" else gb
                    callback = (lambda net: net.forward(p)) if kind == "value" else (
                        lambda net: net.input_derivative(p))
                    row = dict(quantity=kind, **fd_record(
                        network, i, step, callback, vector[i], bounds[2], bounds[3]))
                    SMALL_STEP_CONTROLS.append(row)
                    self.assertTrue(row.get("passed", False), row)

    def test_unrepresentable_parameter_perturbation_is_not_a_verified_gradient(self):
        network = moderate_network()
        i = 0
        step = math.ulp(network.parameters.w[i]) / 8.0
        plus, minus, row = perturbed_pair(network, i, step)
        NONREPRESENTABLE_CONTROLS.append(row)
        self.assertIsNone(plus)
        self.assertIsNone(minus)
        self.assertFalse(row["finite_difference_executed"])
        self.assertEqual(row["status"], "PERTURBATION_NOT_REPRESENTABLE")

    def test_saturation_limits_remain_visible_for_both_gradient_vectors(self):
        network = one_neuron(w=20.0, b=0.0, v=1.0, d=0.0)
        for p in (n.A, n.B):
            result = ng.evaluate_parameter_gradients(network, p)
            h = result.forward.activations[3]
            self.assertEqual(abs(h), 1.0)
            self.assertEqual(result.value_gradient[3], 0.0)
            self.assertEqual(result.value_gradient[19], 0.0)
            self.assertEqual(result.value_gradient[35], h)
            self.assertEqual(result.value_gradient[48], 1.0)
            self.assertEqual(result.sensitivity_gradient, (0.0,) * 49)
            z = result.forward.normalized_input
            tiny = math.exp(-2.0 * abs(20.0*z))
            positive_q = 4*tiny/(1+tiny)**2
            alternative_dg_dv = n.INPUT_ALPHA*20.0*positive_q
            self.assertGreater(alternative_dg_dv, 0.0)
            SATURATION_CONTROLS.append(dict(
                p=p, rounded_h=h, computed_df_dw=result.value_gradient[3],
                computed_dg_dv=result.sensitivity_gradient[35],
                positive_dg_dv_alternative_float=alternative_dg_dv,
                mathematical_accuracy_claimed=False, passed=True))

    def test_invalid_network_type_and_physical_inputs(self):
        for function in (ng.value_and_parameter_gradient,
                         ng.sensitivity_and_parameter_gradient,
                         ng.evaluate_parameter_gradients):
            with self.assertRaises(TypeError):
                function([0.0]*49, 0.1)
            for p in (math.nan, math.inf, -math.inf, 0.0,
                      math.nextafter(n.A, -math.inf), math.nextafter(n.B, math.inf)):
                with self.assertRaises(ValueError):
                    function(moderate_network(), p)
            for p in (True, "0.1", None, complex(0.1, 0)):
                with self.assertRaises(TypeError):
                    function(moderate_network(), p)

    def test_nonfinite_parameter_gradients_are_explicit_errors(self):
        network = one_neuron(w=0.0, b=0.0, v=1e308, d=0.0,
                             normalization=n.AffineNormalization(s_out=3.0))
        self.assertEqual(network.forward(0.1), 0.0)
        self.assertEqual(network.input_derivative(0.1), 0.0)
        for function in (ng.value_and_parameter_gradient,
                         ng.sensitivity_and_parameter_gradient,
                         ng.evaluate_parameter_gradients):
            with self.assertRaises(FloatingPointError):
                function(network, 0.1)


if __name__ == "__main__":
    unittest.main()
