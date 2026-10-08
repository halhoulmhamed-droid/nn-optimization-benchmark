"""New Adam/pipeline controls only; no thermal oracle and no old suites."""
from decimal import Decimal, localcontext
import inspect
import math
import random
import unittest
from unittest.mock import patch

from src import neural_surrogate as n
from src import pilot_training as pt

ADAM_ABS_TOL, ADAM_REL_TOL = 5e-15, 2e-13
SCALAR_ABS_TOL, SCALAR_REL_TOL = 2e-14, 2e-13
INDEPENDENT_ADAM_CONTROLS = []


def fixture_parameters():
    w, b, v = [0.0]*16, [0.0]*16, [0.0]*16
    w[3], b[3], v[3] = 0.4, 0.1, -0.7
    return n.NetworkParameters(w, b, v, 0.05)


def fixture_labels():
    return pt.SiteLabels((0.05, 0.15), (0.3, 0.2), (-1.0, -0.5))


def decimal_reference(initial, gradients):
    """Independent 50-digit component history, not float moment reuse."""
    with localcontext() as ctx:
        ctx.prec = 50
        first, second = [Decimal(0)]*49, [Decimal(0)]*49
        theta = [Decimal(str(x)) for x in initial.to_vector()]
        eta, beta1, beta2, epsilon = map(Decimal, ("0.001", "0.9", "0.999", "1e-8"))
        history = []
        for k, gradient in enumerate(gradients, 1):
            for i, g in enumerate(gradient):
                gi = Decimal(str(g))
                first[i] = beta1*first[i] + (1-beta1)*gi
                second[i] = beta2*second[i] + (1-beta2)*gi*gi
                mc = first[i] / (1-beta1**k)
                sc = second[i] / (1-beta2**k)
                theta[i] -= eta*mc/(sc.sqrt()+epsilon)
            history.append((tuple(map(float, theta)), tuple(map(float, first)),
                            tuple(map(float, second))))
        return history


class PilotTrainingTests(unittest.TestCase):
    def assert_close(self, actual, expected, *, adam=False):
        self.assertTrue(math.isclose(actual, expected,
            abs_tol=ADAM_ABS_TOL if adam else SCALAR_ABS_TOL,
            rel_tol=ADAM_REL_TOL if adam else SCALAR_REL_TOL), (actual, expected))

    def test_first_and_second_updates_against_independent_decimal_history(self):
        params = n.NetworkParameters.from_vector(tuple((i-24)/97.0 for i in range(49)))
        g1 = tuple((0.1, -0.2, 0.0, 1e-10)[i % 4] for i in range(49))
        g2 = tuple((-0.3, 0.4, 1e-12, -1e-10)[i % 4] for i in range(49))
        expected = decimal_reference(params, (g1, g2))
        state, current = pt.AdamState.zero(), params
        for k, gradient in enumerate((g1, g2), 1):
            current, state = pt.adam_step(current, gradient, state)
            observed = (current.to_vector(), state.first_moment, state.second_moment)
            maxima = []
            for actual_vector, wanted_vector in zip(observed, expected[k-1]):
                for actual, wanted in zip(actual_vector, wanted_vector):
                    self.assert_close(actual, wanted, adam=True)
                maxima.append(max(abs(x-y) for x, y in zip(actual_vector, wanted_vector)))
            self.assertEqual(state.updates, k)
            INDEPENDENT_ADAM_CONTROLS.append(dict(
                update=k, components=49, max_abs_parameter_error=maxima[0],
                max_abs_first_moment_error=maxima[1], max_abs_second_moment_error=maxima[2],
                reference_decimal_precision=50, certified=False, passed=True))

    def test_epsilon_is_outside_sqrt_and_bias_corrected_first_update(self):
        initial = n.NetworkParameters.from_vector((0.0,)*49)
        result, state = pt.adam_step(initial, (1e-10,)*49, pt.AdamState.zero())
        expected = -0.001*1e-10/(1e-10+1e-8)
        self.assert_close(result.w[0], expected, adam=True)
        self.assert_close(state.first_moment[0], 1e-11, adam=True)
        self.assert_close(state.second_moment[0], 1e-23, adam=True)

    def test_zero_gradient_from_zero_state_preserves_parameters(self):
        original = fixture_parameters()
        result, state = pt.adam_step(original, (0.0,)*49, pt.AdamState.zero())
        self.assertEqual(result.to_vector(), original.to_vector())
        self.assertEqual(state.first_moment, (0.0,)*49)
        self.assertEqual(state.second_moment, (0.0,)*49)
        self.assertEqual(state.updates, 1)

    def test_inputs_and_optimizer_states_are_immutable_and_distinct(self):
        initial = fixture_parameters()
        before = initial.to_vector()
        left, right = pt.AdamState.zero(), pt.AdamState.zero()
        self.assertIsNot(left, right)
        grad = [0.2]*49
        _, changed = pt.adam_step(initial, grad, left)
        self.assertEqual(left.updates, 0)
        self.assertEqual(right.updates, 0)
        self.assertEqual(changed.updates, 1)
        self.assertEqual(before, initial.to_vector())
        self.assertEqual(grad, [0.2]*49)
        first = pt.train_method("M1", initial, fixture_labels(), S0=2.0, S1=3.0, updates=2)
        second = pt.train_method("M2", initial, fixture_labels(), S0=2.0, S1=3.0, updates=2)
        self.assertEqual(first["initial_parameter_id"], second["initial_parameter_id"])
        self.assertEqual(first["final_Adam_state"]["updates"], 2)
        self.assertEqual(second["final_Adam_state"]["updates"], 2)
        self.assertNotEqual(first["final_Adam_state"]["first_moment"],
                            second["final_Adam_state"]["first_moment"])
        self.assertEqual(initial.to_vector(), before)

    def test_initialization_reproducible_draw_order_and_global_RNG_preserved(self):
        global_state = random.getstate()
        first, second = pt.initialize_parameters(), pt.initialize_parameters()
        self.assertEqual(first.to_vector(), second.to_vector())
        self.assertEqual(random.getstate(), global_state)
        rng, bound = random.Random(20261006), math.sqrt(6.0/17.0)
        expected_w = tuple(rng.uniform(-bound, bound) for _ in range(16))
        expected_v = tuple(rng.uniform(-bound, bound) for _ in range(16))
        self.assertEqual(first.to_vector(), expected_w + (0.0,)*16 + expected_v + (0.0,))
        self.assertTrue(all(-bound <= x <= bound for x in first.w + first.v))

    def test_float_sites_are_exactly_16_17_distinct_and_disjoint(self):
        training, validation = pt.make_sites()
        self.assertEqual((len(training), len(validation)), (16, 17))
        self.assertEqual(training, tuple(n.A+(j+0.5)*(n.B-n.A)/16 for j in range(16)))
        self.assertEqual(validation, tuple(n.A+j*(n.B-n.A)/16 for j in range(17)))
        self.assertEqual(len(set(training+validation)), 33)
        self.assertFalse(set(training) & set(validation))
        self.assertEqual((validation[0], validation[-1]), (n.A, n.B))

    def test_endpoint_cache_labels_acquired_once_and_reused_after_training(self):
        training, validation = pt.make_sites()
        called_values, called_derivatives = [], []
        cache = {n.A: (1+n.A, 1.0), n.B: (1+n.B, 1.0)}
        original_cache = cache.copy()
        def value(p):
            called_values.append(p)
            return 1+p
        def derivative(p):
            called_derivatives.append(p)
            return 1.0
        train, valid, counts = pt.acquire_labels(
            training, validation, value, derivative, endpoint_cache=cache)
        self.assertEqual(counts["value_calls"], 31)
        self.assertEqual(counts["derivative_calls"], 31)
        self.assertEqual(counts["cached_sites"], 2)
        self.assertEqual(len(set(called_values)), 31)
        self.assertEqual(called_values, called_derivatives)
        self.assertEqual(cache, original_cache)
        result = pt.train_method("M1", fixture_parameters(), train, S0=2.0, S1=3.0, updates=1)
        pt.validation_diagnostics(n.NetworkParameters.from_vector(result["final_parameters"]),
                                  valid, S0=2.0, S1=3.0)
        self.assertEqual((len(called_values), len(called_derivatives)), (31, 31))
        self.assertEqual(train.values, tuple(1+p for p in training))
        self.assertEqual(valid.derivatives, (1.0,)*17)

    def test_M1_never_passes_derivative_labels_to_its_loss(self):
        labels = fixture_labels()
        altered = pt.SiteLabels(labels.points, labels.values, (100.0, -200.0))
        first = pt.train_method("M1", fixture_parameters(), labels, S0=2.0, S1=3.0, updates=1)
        second = pt.train_method("M1", fixture_parameters(), altered, S0=2.0, S1=3.0, updates=1)
        self.assertEqual(first["final_parameters"], second["final_parameters"])
        self.assertEqual(first["controls"], second["controls"])

    def test_only_training_sites_reach_gradients_validation_is_diagnostic(self):
        training = fixture_labels()
        validation = pt.SiteLabels((n.A,n.B), (1000.0,-1000.0), (500.0,-500.0))
        observed = []
        original = pt.losses.m1_loss_and_gradient
        def guarded(network, points, targets, *, S0):
            observed.append(tuple(points))
            self.assertEqual(tuple(points), training.points)
            self.assertEqual(tuple(targets), training.values)
            return original(network, points, targets, S0=S0)
        with patch.object(pt.losses, "m1_loss_and_gradient", side_effect=guarded):
            result = pt.train_method("M1", fixture_parameters(), training,
                                     S0=2.0, S1=3.0, updates=2)
            pt.validation_diagnostics(n.NetworkParameters.from_vector(result["final_parameters"]),
                                      validation, S0=2.0, S1=3.0)
        self.assertEqual(len(observed), 5)  # 2 updates + controls 0/1/2.
        self.assertNotIn("validation", inspect.signature(pt.train_method).parameters)
        self.assertFalse(result["validation_used_in_updates"])
        self.assertEqual(result["completed_updates"], 2)

    def test_control_logging_is_separate_from_updates_and_budget_is_bounded(self):
        result = pt.train_method("M2", fixture_parameters(), fixture_labels(),
                                 S0=2.0, S1=3.0, updates=2)
        self.assertEqual(result["completed_updates"], 2)
        self.assertEqual([x["updates"] for x in result["controls"]], [0,1,2])
        self.assertEqual(result["call_counts_from_control_flow"]["update_loss_and_gradient_completed"], 2)
        self.assertEqual(result["call_counts_from_control_flow"]["logging_loss_and_gradient_completed"], 3)
        with patch.object(pt.losses, "m1_loss_and_gradient",
                          side_effect=AssertionError("must reject before loss")):
            with self.assertRaises(ValueError):
                pt.train_method("M1", fixture_parameters(), fixture_labels(),
                                S0=2.0, S1=3.0, updates=301)
        full = pt.AdamState((0.0,)*49, (0.0,)*49, 300)
        with self.assertRaises(ValueError):
            pt.adam_step(fixture_parameters(), (0.0,)*49, full)

    def test_numerical_error_stops_without_retry_and_preserves_last_parameters(self):
        initial = fixture_parameters()
        original = pt.losses.m1_loss_and_gradient
        calls = 0
        def fail_after_initial(*args, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise FloatingPointError("synthetic invalid update")
            return original(*args, **kwargs)
        with patch.object(pt.losses, "m1_loss_and_gradient", side_effect=fail_after_initial):
            result = pt.train_method("M1", initial, fixture_labels(), S0=2.0, S1=3.0, updates=2)
        self.assertEqual(calls, 2)
        self.assertEqual(result["status"], "STOPPED_NUMERICAL_ERROR")
        self.assertEqual(result["completed_updates"], 0)
        self.assertEqual(result["final_parameters"], initial.to_vector())
        self.assertEqual(result["incident"]["attempted_update"], 1)
        self.assertFalse(result["incident"]["automatic_retry"])

    def test_validation_metrics_against_constant_algebraic_reference(self):
        parameters = n.NetworkParameters((0.0,)*16, (0.0,)*16, (0.0,)*16, 0.5)
        labels = pt.SiteLabels((n.A,n.B), (0.4,0.6), (-1.0,-2.0))
        result = pt.validation_diagnostics(parameters, labels, S0=0.5, S1=2.0)
        self.assert_close(result["value_RMSE"], 0.1)
        self.assert_close(result["derivative_RMSE"], math.sqrt(2.5))
        self.assert_close(result["normalized_value_RMSE"], 0.2)
        self.assert_close(result["normalized_derivative_RMSE"], math.sqrt(2.5)/2)
        self.assert_close(result["discrete_max_abs_value_error"], 0.1)
        self.assertEqual(result["discrete_max_abs_derivative_error"], 2.0)
        self.assertFalse(result["maxima_are_uniform_bounds"])

    def test_saturation_control_keeps_rounding_limit_visible(self):
        w, b, v = [0.0]*16, [0.0]*16, [0.0]*16
        w[0], v[0] = 20.0, 1.0
        network = n.NeuralSurrogate(n.NetworkParameters(w,b,v,0))
        result = pt.saturation_diagnostic(network, (n.A,n.B))
        self.assertEqual(result["rounded_count"], 2)
        self.assertEqual(result["near_count"], 2)
        self.assertEqual(result["activation_count"], 32)
        self.assert_close(result["max_abs_preactivation"], 20.0)

    def test_invalid_sizes_types_domains_and_finite_contracts(self):
        params = fixture_parameters()
        for size in (48,50):
            with self.assertRaises(ValueError):
                pt.adam_step(params, [0.0]*size, pt.AdamState.zero())
        for bad in (math.nan, math.inf, -math.inf):
            with self.assertRaises(ValueError):
                pt.adam_step(params, [bad]+[0.0]*48, pt.AdamState.zero())
        for bad in (True, "1", None):
            with self.assertRaises(TypeError):
                pt.adam_step(params, [bad]+[0.0]*48, pt.AdamState.zero())
        for k in (-1,301):
            with self.assertRaises(ValueError):
                pt.AdamState((0.0,)*49, (0.0,)*49, k)
        for k in (True,1.5):
            with self.assertRaises(TypeError):
                pt.AdamState((0.0,)*49, (0.0,)*49, k)
        with self.assertRaises(ValueError):
            pt.AdamState((0.0,)*49, (-1.0,)+(0.0,)*48)
        with self.assertRaises(ValueError):
            pt.SiteLabels((n.A,), (), (0.0,))
        with self.assertRaises(ValueError):
            pt.SiteLabels((n.A,), (math.inf,), (0.0,))
        for points in ((), (n.A,n.A), (0.0,), (math.nextafter(n.B,math.inf),)):
            with self.assertRaises(ValueError):
                pt.SiteLabels(points, (0.0,)*len(points), (0.0,)*len(points))
        with self.assertRaises(ValueError):
            pt.acquire_labels((n.A,), (n.A,), lambda p:0, lambda p:0)
        for bad in (0.0,-1.0,math.nan,math.inf):
            with self.assertRaises(ValueError):
                pt.train_method("M1", params, fixture_labels(), S0=bad, S1=3.0, updates=1)
        for bad in (0,301):
            with self.assertRaises(ValueError):
                pt.train_method("M1", params, fixture_labels(), S0=2.0, S1=3.0, updates=bad)
        with self.assertRaises(TypeError):
            pt.adam_step(None, (0.0,)*49, pt.AdamState.zero())
        with self.assertRaises(TypeError):
            pt.adam_step(params, (0.0,)*49, None)
        with self.assertRaises(ValueError):
            pt.train_method("PSO", params, fixture_labels(), S0=2.0, S1=3.0, updates=1)

    def test_overflow_is_explicit_without_input_mutation(self):
        parameters, state = fixture_parameters(), pt.AdamState.zero()
        before = parameters.to_vector()
        with self.assertRaises(FloatingPointError):
            pt.adam_step(parameters, (1e308,)*49, state)
        self.assertEqual(parameters.to_vector(), before)
        self.assertEqual(state.updates, 0)


if __name__ == "__main__":
    unittest.main()
