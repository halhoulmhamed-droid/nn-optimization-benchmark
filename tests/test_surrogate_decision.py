"""New phase 2B fixtures only; no oracle, M0, training or old tests."""
from __future__ import annotations

import builtins
from fractions import Fraction
import importlib.util
import math
from pathlib import Path
import sys
import unittest
from unittest import mock

from src import surrogate_decision as decision
from src.neural_surrogate import (
    AffineNormalization, NetworkParameters, NeuralSurrogate,
)

ABS_TOL = 5e-14
REL_TOL = 2e-12
CURV_ABS = 2e-13
CURV_REL = 2e-13
TEST_COUNTERS = {
    "prediction_callbacks_executed": 0,
    "successful_selections": 0,
    "discrete_objective_evaluations_successful_selections": 0,
    "physical_callbacks": 0,
    "M0_solves": 0,
    "training_updates": 0,
}
GRID_ERROR_CONTROLS = []


def cache(function, grid):
    def counted(p):
        TEST_COUNTERS["prediction_callbacks_executed"] += 1
        return function(p)
    return decision.cache_predictions(counted, grid, source="unit_fixture")


def select(predictions, lambda_value=0.0):
    result = decision.select_grid_minimum(predictions, lambda_value)
    TEST_COUNTERS["successful_selections"] += 1
    TEST_COUNTERS["discrete_objective_evaluations_successful_selections"] += (
        result.objective_evaluations)
    return result


def single_network(w=0.7, v=-0.8, *, alpha=2.0, s_out=1.0):
    parameters = NetworkParameters(
        (w,) + (0.0,) * 15, (0.1,) + (0.0,) * 15,
        (v,) + (0.0,) * 15, 0.2)
    return NeuralSurrogate(
        parameters, AffineNormalization(alpha, -0.3, 0.4, s_out))


class SurrogateDecisionTests(unittest.TestCase):
    def assertClose(self, actual, expected, *, curvature=False):
        atol, rtol = (CURV_ABS, CURV_REL) if curvature else (ABS_TOL, REL_TOL)
        self.assertTrue(math.isclose(actual, expected, abs_tol=atol, rel_tol=rtol),
                        f"{actual!r} != {expected!r}")

    def test_constant_ties_choose_smallest_p(self):
        grid = decision.make_grid(0.0, 1.0, 8)
        result = select(cache(lambda p: 3.0, grid))
        self.assertEqual((result.index, result.p_hat), (0, 0.0))
        self.assertEqual(result.candidate_origin, "lower_boundary")
        self.assertEqual(result.network_evaluations_in_selection, 0)
        self.assertEqual(result.physical_evaluations_in_selection, 0)
        self.assertFalse(result.continuous_global_minimum_claimed)
        self.assertEqual(decision.grid_value_error(0.0, grid.h_max), 0.0)

    def test_affine_and_lambda_choose_both_boundaries(self):
        grid = decision.make_grid(0.0, 1.0, 8)
        predictions = cache(lambda p: 2.0 * p + 1.0, grid)
        self.assertEqual(select(predictions, 0.0).p_hat, 0.0)
        self.assertEqual(select(predictions, -3.0).p_hat, 1.0)
        self.assertEqual(select(predictions, -2.0).p_hat, 0.0)

    def test_quadratic_between_grid_points_exact_bound(self):
        grid = decision.make_grid(0.0, 1.0, 4)
        centre = Fraction(3, 8)
        result = select(cache(lambda p: (p - float(centre)) ** 2, grid))
        self.assertEqual((result.index, result.p_hat), (1, 0.25))
        exact_gap = (Fraction(result.p_hat) - centre) ** 2
        exact_bound = Fraction(2) * Fraction(grid.h_max) ** 2 / 8
        self.assertEqual(exact_gap, exact_bound)
        self.assertClose(result.predicted_objective, float(exact_gap))
        self.assertClose(decision.grid_value_error(2.0, grid.h_max),
                         float(exact_bound))
        GRID_ERROR_CONTROLS.append({
            "fixture": "quadratic_halfway", "exact_gap": str(exact_gap),
            "exact_bound": str(exact_bound), "gap_float": float(exact_gap),
            "estimate_float": decision.grid_value_error(2.0, grid.h_max),
        })

    def test_nonconvex_polynomial_global_from_known_extrema(self):
        def phi_exact(x):
            return x**4 / 4 - Fraction(4, 9) * x**3 + Fraction(25, 96) * x**2 - x/16
        candidates = [Fraction(0), Fraction(1, 4), Fraction(1, 3),
                      Fraction(3, 4), Fraction(1)]
        global_point = min(candidates, key=phi_exact)
        self.assertEqual(global_point, Fraction(3, 4))
        self.assertLess(phi_exact(global_point), phi_exact(Fraction(1, 4)))
        self.assertLess(phi_exact(global_point), phi_exact(Fraction(0)))
        self.assertLess(phi_exact(global_point), phi_exact(Fraction(1)))
        # Derivative roots and sign alternation establish all extrema analytically.
        for x in (Fraction(1, 4), Fraction(1, 3), Fraction(3, 4)):
            expanded = x**3 - Fraction(4, 3)*x**2 + Fraction(25, 48)*x - Fraction(1, 16)
            self.assertEqual(expanded, 0)
        for x, sign in [(Fraction(1, 8), -1), (Fraction(7, 24), 1),
                        (Fraction(1, 2), -1), (Fraction(7, 8), 1)]:
            derivative = (x-Fraction(1,4))*(x-Fraction(1,3))*(x-Fraction(3,4))
            self.assertGreater(sign * derivative, 0)
        phi = lambda x: x**4/4.0 - 4.0*x**3/9.0 + 25.0*x*x/96.0 - x/16.0
        grid = decision.make_grid(0.0, 1.0, 8)
        result = select(cache(phi, grid))
        self.assertEqual(result.p_hat, 0.75)
        self.assertClose(result.predicted_objective, float(phi_exact(global_point)))
        GRID_ERROR_CONTROLS.append({
            "fixture": "nonconvex_known_minima",
            "exact_gap": "0",
            "exact_bound": str(Fraction(41, 48)*Fraction(1,8)**2/8),
            "curvature_reason": "phi'' convex quadratic; endpoint max 41/48, vertex -31/432",
        })

    def test_nonuniform_grid_uses_actual_max_spacing(self):
        grid = decision.DecisionGrid(0.0, 1.0, (0.0, 0.125, 0.75, 1.0))
        self.assertEqual(grid.h_max, 0.625)
        centre = Fraction(1, 2)
        result = select(cache(lambda p: (p - 0.5)**2, grid))
        gap = (Fraction(result.p_hat) - centre)**2
        bound = Fraction(2)*Fraction(grid.h_max)**2/8
        self.assertLessEqual(gap, bound)
        self.assertClose(decision.grid_value_error(2.0, grid.h_max), float(bound))
        GRID_ERROR_CONTROLS.append({
            "fixture": "nonuniform_quadratic", "exact_gap": str(gap),
            "exact_bound": str(bound),
        })

    def test_predictions_reused_for_seven_tasks_without_parameter_mutation(self):
        network = single_network()
        original = network.parameters.to_vector()
        grid = decision.make_grid(0.02, 0.20, 1024)
        before = TEST_COUNTERS["prediction_callbacks_executed"]
        predictions = cache(network.forward, grid)
        for lam in (0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0):
            result = select(predictions, lam)
            self.assertEqual(result.objective_evaluations, 1025)
        self.assertEqual(TEST_COUNTERS["prediction_callbacks_executed"] - before, 1025)
        self.assertEqual(original, network.parameters.to_vector())
        self.assertEqual(predictions.prediction_evaluations, 1025)
        with self.assertRaises((AttributeError, TypeError)):
            predictions.values = ()

    def test_exact_grid_budget_and_endpoints(self):
        grid = decision.make_grid(0.02, 0.20)
        self.assertEqual(grid.intervals, 1024)
        self.assertEqual(len(grid.points), 1025)
        self.assertEqual(grid.points[0], 0.02)
        self.assertEqual(grid.points[-1], 0.20)
        self.assertEqual(grid.h_max, max(y-x for x,y in zip(grid.points,grid.points[1:])))
        self.assertTrue(all(0.02 <= p <= 0.20 for p in grid.points))
        self.assertTrue(all(x<y for x,y in zip(grid.points,grid.points[1:])))

    def test_selector_module_import_and_execution_without_physics(self):
        original_import = builtins.__import__
        def guard(name, *args, **kwargs):
            if name in ("src.thermal_oracle", "src.physical_reference", "src.pilot_training"):
                raise AssertionError("forbidden dependency in selector")
            return original_import(name, *args, **kwargs)
        module_name = "_phase2b_independent_selector_fixture"
        specification = importlib.util.spec_from_file_location(
            module_name, Path(decision.__file__))
        module = importlib.util.module_from_spec(specification)
        with mock.patch.dict(sys.modules, {module_name: module,
                                         "src.thermal_oracle": None,
                                         "src.physical_reference": None,
                                         "src.pilot_training": None}):
            with mock.patch("builtins.__import__", side_effect=guard):
                specification.loader.exec_module(module)
                grid = module.make_grid(0.0, 1.0, 4)
                predictions = module.PredictionCache(grid, (1.0, 0.5, -1.0, 0.2, 2.0))
                result = module.select_grid_minimum(predictions, 0.0)
        self.assertEqual(result.p_hat, 0.5)
        self.assertEqual(result.physical_evaluations_in_selection, 0)

    def test_generic_kkt_all_cases_and_no_snapping(self):
        cases = [(0.0, 2.0, 0.0), (0.0, -2.0, 2.0),
                 (1.0, -3.0, 0.0), (1.0, 3.0, 3.0),
                 (0.5, -4.0, 4.0), (0.5, 0.0, 0.0)]
        for p, gradient, expected in cases:
            self.assertEqual(decision.kkt_residual_interval(p, gradient, 0.0, 1.0), expected)
        tiny_inside = math.nextafter(0.0, 1.0)
        self.assertEqual(decision.kkt_residual_interval(tiny_inside, 2.0, 0.0, 1.0), 2.0)

    def test_curvature_single_neuron_independent_expression(self):
        network = single_network(w=0.7, v=-0.8, alpha=2.0, s_out=-3.0)
        grid = decision.make_grid(0.02, 0.20, 16)
        bound = decision.estimate_network_grid_error(network, grid)
        expected = 4.0*3.0*4.0*0.8*0.49/(3.0*math.sqrt(3.0))
        self.assertClose(bound["B_hat_float"], expected, curvature=True)
        self.assertClose(bound["alpha_grid_float"],
                         expected * grid.h_max**2/8.0, curvature=True)
        self.assertFalse(bound["certified"])
        self.assertIsNone(bound["e_hat_rigorous_bound"])
        self.assertIsNone(bound["position_error_bound"])

    def test_curvature_scaling_alpha_squared(self):
        grid = decision.make_grid(0.02, 0.20, 16)
        one = decision.estimate_network_grid_error(single_network(alpha=1.0), grid)
        three = decision.estimate_network_grid_error(single_network(alpha=3.0), grid)
        self.assertClose(three["B_hat_float"], 9.0*one["B_hat_float"], curvature=True)

    def test_curvature_scaling_output_absolute_value(self):
        grid = decision.make_grid(0.02, 0.20, 16)
        one = decision.estimate_network_grid_error(single_network(s_out=1.0), grid)
        negative = decision.estimate_network_grid_error(single_network(s_out=-2.5), grid)
        self.assertClose(negative["B_hat_float"], 2.5*one["B_hat_float"], curvature=True)

    def test_curvature_scaling_weight_squared(self):
        grid = decision.make_grid(0.02, 0.20, 16)
        one = decision.estimate_network_grid_error(single_network(w=0.3), grid)
        doubled = decision.estimate_network_grid_error(single_network(w=-0.6), grid)
        self.assertClose(doubled["B_hat_float"], 4.0*one["B_hat_float"], curvature=True)

    def test_zero_curvature_affine_cases(self):
        grid = decision.make_grid(0.02, 0.20, 16)
        for model in (single_network(w=0.0), single_network(v=0.0),
                      single_network(alpha=0.0), single_network(s_out=0.0)):
            estimate = decision.estimate_network_grid_error(model, grid)
            self.assertEqual(estimate["B_hat_float"], 0.0)
            self.assertTrue(estimate["exact_affine_from_zero_factors"])
            self.assertEqual(estimate["alpha_grid_float"], 0.0)

    def test_invalid_grids_and_stagnating_construction(self):
        for args in [(0.0, 0.0, 4), (1.0, 0.0, 4),
                     (0.0, 1.0, 0), (0.0, 1.0, 1025),
                     (0.0, 1.0, True), (0.0, 1.0, 3.0),
                     (float("nan"), 1.0, 4), (0.0, float("inf"), 4),
                     (-1e308, 1e308, 4)]:
            with self.assertRaises((TypeError, ValueError)):
                decision.make_grid(*args)
        for points in [(0.0,), (0.0, 0.0, 1.0), (0.0, 0.8, 0.3, 1.0),
                       (0.1, 1.0), (0.0, 0.5, 0.9), (0.0, 1.1, 1.0),
                       (0.0, float("nan"), 1.0)]:
            with self.assertRaises((TypeError, ValueError)):
                decision.DecisionGrid(0.0, 1.0, points)
        with self.assertRaises(ValueError):
            decision.make_grid(1.0, math.nextafter(1.0, math.inf), 2)

    def test_invalid_predictions_sizes_values_and_sources(self):
        grid = decision.make_grid(0.0, 1.0, 2)
        for values in [(0.0,), (0.0, math.nan, 1.0), (0.0, math.inf, 1.0), "abc"]:
            with self.assertRaises((TypeError, ValueError)):
                decision.PredictionCache(grid, values)
        with self.assertRaises(TypeError):
            decision.cache_predictions(None, grid)
        with self.assertRaises(ValueError):
            decision.cache_predictions(lambda p: p, grid, source="")
        with self.assertRaises(ValueError):
            cache(lambda p: math.nan, grid)

    def test_invalid_lambda_and_nonfinite_objective(self):
        grid = decision.make_grid(0.0, 1.0, 2)
        predictions = decision.PredictionCache(grid, (0.0, 0.0, 1e308))
        for value in (True, math.nan, math.inf, -math.inf, "0"):
            with self.assertRaises((TypeError, ValueError)):
                decision.select_grid_minimum(predictions, value)
        with self.assertRaises(FloatingPointError):
            decision.select_grid_minimum(predictions, 1e308)

    def test_invalid_bound_and_curvature_overflow(self):
        for curvature, width in [(-1.0, 0.1), (1.0, 0.0), (math.inf, 0.1),
                                 (1.0, math.nan), (True, 0.1)]:
            with self.assertRaises((TypeError, ValueError)):
                decision.grid_value_error(curvature, width)
        with self.assertRaises(FloatingPointError):
            decision.grid_value_error(1e308, 1e308)
        grid = decision.make_grid(0.02, 0.20, 16)
        with self.assertRaises(FloatingPointError):
            decision.estimate_network_grid_error(single_network(w=1e308), grid)

    def test_rounded_underflow_zero_is_not_affine_proof(self):
        grid = decision.make_grid(0.02, 0.20, 16)
        estimate = decision.estimate_network_grid_error(
            single_network(w=1e-200), grid)
        self.assertEqual(estimate["B_hat_float"], 0.0)
        self.assertFalse(estimate["exact_affine_from_zero_factors"])
        self.assertTrue(estimate["rounded_zero_without_affine_proof"])
        self.assertFalse(estimate["certified"])

    def test_invalid_kkt_inputs(self):
        for args in [(-0.1, 1.0, 0.0, 1.0), (1.1, 1.0, 0.0, 1.0),
                     (0.0, 1.0, 1.0, 0.0), (0.0, math.nan, 0.0, 1.0),
                     (True, 1.0, 0.0, 1.0)]:
            with self.assertRaises((TypeError, ValueError)):
                decision.kkt_residual_interval(*args)

