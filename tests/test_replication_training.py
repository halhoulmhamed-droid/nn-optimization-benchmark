"""New phase-3A tests only: synthetic objectives, no thermal imports or labels."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, replace
from decimal import Decimal, localcontext
import math
from pathlib import Path
import random
import struct
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from src import replication_training as r
from src import neural_surrogate as n
from src import surrogate_losses as losses

# Set by the explicit validation script; no test files are written in the project.
FIXTURE_ROOT = None
PLAN = None
CRITICAL_SOURCES = None
PILOT_INITIALIZATION = None
ABS_TOL = 5e-15
REL_TOL = 2e-13
LEDGER = {
    "synthetic_adam_updates_completed": 0,
    "direct_adam_attempts": 0,
    "continuation_Adam_candidates": 0,
    "continuation_loss_gradient_attempts": 0,
    "other_explicit_synthetic_loss_gradient_calls": 0,
    "quadratic_loss_gradient_calls": 0,
    "synthetic_neural_loss_gradient_calls": {"M1": 0, "M2": 0},
    "direct_existing_loss_API_calls": {"M1": 0, "M2": 0},
    "synthetic_neural_point_value_gradient_evaluations": 0,
    "synthetic_neural_point_sensitivity_gradient_evaluations": 0,
    "checkpoint_write_calls": 0,
    "checkpoint_writes_completed": 0,
    "checkpoint_identical_reuses": 0,
    "independent_Decimal_reference_steps": 0,
}
EVIDENCE = {}


def explicit_parameters():
    return n.NetworkParameters.from_vector((i - 24) * 0.004 for i in range(49))


def context(parameters, *, method="QUADRATIC_FIXTURE", data_id=None,
            points=(), target=3000, snapshots=(300, 3000), seed=73):
    return r.make_context(
        seed=seed, condition="SYNTHETIC_" + method, method=method,
        protocol_id=r.protocol_signature(PLAN),
        data_id=data_id or r.identity({"synthetic": "Q49", "c": coefficients(), "u": targets()}),
        points=points, S0=PLAN["losses"]["S0"], S1=PLAN["losses"]["S1"],
        scales_origin=PLAN["losses"]["scales_origin"],
        critical_sources=CRITICAL_SOURCES,
        initial_parameter_id=r.identity(parameters.to_vector()),
        target_updates=target, snapshot_updates=snapshots,
        data_origin="EXPLICIT_SYNTHETIC_FIXTURE_NOT_THERMAL")


def coefficients():
    return tuple(1.0 + (i % 5) / 8.0 for i in range(49))


def targets():
    return tuple((i % 7 - 3) * 0.02 for i in range(49))


class Quadratic:
    def __init__(self):
        self.calls = 0

    def __call__(self, parameters):
        self.calls += 1
        LEDGER["quadratic_loss_gradient_calls"] += 1
        residuals = tuple(x - u for x, u in zip(parameters.to_vector(), targets()))
        loss = 0.5 * math.fsum(c * e * e for c, e in zip(coefficients(), residuals))
        gradient = tuple(c * e for c, e in zip(coefficients(), residuals))
        return SimpleNamespace(loss=loss, gradient=gradient)


class CountedAdapter(r.LossAdapter):
    def __call__(self, parameters):
        LEDGER["synthetic_neural_loss_gradient_calls"][self.method] += 1
        record_neural_points(self.method, len(self.labels.points))
        return super().__call__(parameters)


def record_neural_points(method, count):
    LEDGER["synthetic_neural_point_value_gradient_evaluations"] += count
    if method == "M2":
        LEDGER["synthetic_neural_point_sensitivity_gradient_evaluations"] += count


def advance(cp, target, evaluate):
    result = r.continue_to(cp, target, evaluate, expected_context=cp.context)
    LEDGER["synthetic_adam_updates_completed"] += result.state.updates - cp.state.updates
    LEDGER["continuation_Adam_candidates"] += (
        result.counts.update_completed - cp.counts.update_completed)
    LEDGER["continuation_loss_gradient_attempts"] += (
        result.counts.update_attempts - cp.counts.update_attempts)
    return result


def step(state, gradient):
    LEDGER["direct_adam_attempts"] += 1
    result = r.adam_step(state, gradient)
    LEDGER["synthetic_adam_updates_completed"] += 1
    return result


def save(path, cp, expected=None):
    LEDGER["checkpoint_write_calls"] += 1
    result = r.save_checkpoint(path, cp, expected_context=expected or cp.context)
    LEDGER["checkpoint_writes_completed"] += int(result["written"])
    LEDGER["checkpoint_identical_reuses"] += int(not result["written"])
    return result


def binary_state(state):
    return struct.pack(">147dQ", *(state.parameters.to_vector() +
                                  state.first_moment + state.second_moment),
                       state.updates)


class ReplicationTrainingTests(unittest.TestCase):
    def setUp(self):
        if FIXTURE_ROOT is None or PLAN is None:
            raise RuntimeError("use scripts/check_replication_readiness.py --tests-only")
        self.directory = Path(FIXTURE_ROOT) / self._testMethodName
        self.directory.mkdir(parents=True, exist_ok=False)
        self.parameters = explicit_parameters()

    def start(self, *, target=8, snapshots=(3, 8)):
        ctx = context(self.parameters, target=target, snapshots=snapshots)
        return r.TrainingCheckpoint.start(r.AdamState.zero(self.parameters), ctx)

    def close(self, actual, expected):
        self.assertLessEqual(abs(actual - expected),
                             ABS_TOL + REL_TOL * max(abs(actual), abs(expected)))

    def test_adam_first_two_independent_decimal(self):
        state = r.AdamState.zero(self.parameters)
        gradients = [
            tuple(((-1) ** i) * (i + 1) * 1e-10 for i in range(49)),
            tuple(((-1) ** (i + 1)) * (i + 2) * 3e-10 for i in range(49)),
        ]
        maximum = 0.0
        with localcontext() as ctx:
            ctx.prec = 60
            theta = list(map(Decimal.from_float, self.parameters.to_vector()))
            m, v = [Decimal(0)] * 49, [Decimal(0)] * 49
            eta = Decimal.from_float(0.001)
            b1, b2 = Decimal.from_float(0.9), Decimal.from_float(0.999)
            eps = Decimal.from_float(1e-8)
            for k, gradient in enumerate(gradients, 1):
                state = step(state, gradient)
                LEDGER["independent_Decimal_reference_steps"] += 1
                for i, gi in enumerate(gradient):
                    g = Decimal.from_float(gi)
                    m[i] = b1 * m[i] + (1 - b1) * g
                    v[i] = b2 * v[i] + (1 - b2) * g * g
                    theta[i] -= eta * (m[i] / (1 - b1**k)) / (
                        (v[i] / (1 - b2**k)).sqrt() + eps)
                    for actual, expected in (
                        (state.parameters.to_vector()[i], float(theta[i])),
                        (state.first_moment[i], float(m[i])),
                        (state.second_moment[i], float(v[i])),
                    ):
                        maximum = max(maximum, abs(actual - expected))
                        self.close(actual, expected)
                self.assertEqual(state.updates, k)
        EVIDENCE["Adam_Decimal"] = {"components_per_step": 49, "steps": 2,
                                    "max_absolute_error": maximum,
                                    "small_gradients_exercise_epsilon_placement": True}

    def test_bias_correction_at_global_step_301(self):
        g = tuple((i - 23) * 0.02 for i in range(49))
        first = tuple(x * (1 - 0.9**300) for x in g)
        second = tuple(x * x * (1 - 0.999**300) for x in g)
        state = r.AdamState(self.parameters, first, second, 300)
        result = step(state, g)
        maximum = 0.0
        with localcontext() as ctx:
            ctx.prec = 60
            b1, b2 = Decimal.from_float(0.9), Decimal.from_float(0.999)
            eta, eps = Decimal.from_float(0.001), Decimal.from_float(1e-8)
            for i in range(49):
                dg = Decimal.from_float(g[i])
                m = b1 * Decimal.from_float(first[i]) + (1 - b1) * dg
                v = b2 * Decimal.from_float(second[i]) + (1 - b2) * dg * dg
                expected = Decimal.from_float(self.parameters.to_vector()[i]) - eta * (
                    m / (1 - b1**301)) / ((v / (1 - b2**301)).sqrt() + eps)
                maximum = max(maximum, abs(result.parameters.to_vector()[i] - float(expected)))
                self.close(result.parameters.to_vector()[i], float(expected))
        LEDGER["independent_Decimal_reference_steps"] += 1
        self.assertEqual(result.updates, 301)
        EVIDENCE["step_301"] = {"max_absolute_error": maximum, "global_counter": 301}

    def test_zero_gradient_preserves_zero_moments(self):
        state = r.AdamState.zero(self.parameters)
        result = step(state, (0.0,) * 49)
        self.assertEqual(result.parameters, state.parameters)
        self.assertEqual(result.first_moment, state.first_moment)
        self.assertEqual(result.second_moment, state.second_moment)
        self.assertEqual(result.updates, 1)

    def test_invalid_state_and_hyperparameters(self):
        for vector in ([0.0] * 48, [True] * 49, [math.nan] * 49, [math.inf] * 49):
            with self.subTest(vector_type=repr(vector[0])):
                with self.assertRaises((ValueError, TypeError)):
                    r.AdamState(self.parameters, vector, (0.0,) * 49)
        for count in (True, -1, 3001, 1.0):
            with self.subTest(count=count), self.assertRaises((ValueError, TypeError)):
                r.AdamState(self.parameters, (0.0,) * 49, (0.0,) * 49, count)
        with self.assertRaises(ValueError):
            r.AdamState(self.parameters, (0.0,) * 49, (-1.0,) * 49, 1)
        with self.assertRaises(ValueError):
            r.AdamState(self.parameters, (1.0,) * 49, (0.0,) * 49)
        for kwargs in ({"eta": True}, {"epsilon": 0}, {"beta1": 1},
                       {"beta2": math.nan}):
            with self.subTest(kwargs=kwargs), self.assertRaises((ValueError, TypeError)):
                r.AdamConfig(**kwargs)

    def test_invalid_candidate_never_replaces_state(self):
        state = r.AdamState.zero(self.parameters)
        original = binary_state(state)
        for gradient in ((1e200,) * 49, (False,) * 49, (math.nan,) * 49, (0.0,) * 48):
            with self.subTest(kind=repr(gradient[0])):
                with self.assertRaises((ValueError, TypeError, FloatingPointError)):
                    step(state, gradient)
                self.assertEqual(binary_state(state), original)

    def test_single_Q49_uninterrupted_vs_serialized_300_then_3000(self):
        ctx = context(self.parameters)
        initial = r.TrainingCheckpoint.start(r.AdamState.zero(self.parameters), ctx)
        uninterrupted_Q, resumed_Q = Quadratic(), Quadratic()
        full = advance(initial, 3000, uninterrupted_Q)
        prefix = advance(initial, 300, resumed_Q)
        self.assertEqual(binary_state(prefix.state), binary_state(full.snapshots["300"]))
        path = self.directory / "prefix_300.json"
        save(path, prefix)
        restored = r.load_checkpoint(path, expected_context=ctx)
        self.assertEqual(binary_state(restored.state), binary_state(prefix.state))
        self.assertEqual(restored, prefix)
        calls_before = resumed_Q.calls
        final = advance(restored, 3000, resumed_Q)
        self.assertEqual(resumed_Q.calls - calls_before, 2700)
        self.assertEqual(binary_state(full.state), binary_state(final.state))
        self.assertEqual(full, final)
        self.assertEqual(set(final.snapshots), {"300", "3000"})
        self.assertEqual(uninterrupted_Q.calls, 3000)
        self.assertEqual(resumed_Q.calls, 3000)
        self.assertEqual(final.counts.update_completed, 3000)
        EVIDENCE["continuation_Q49"] = {
            "definition": "Q=1/2*sum_i (1+(i%5)/8)*(theta_i-(i%7-3)*0.02)^2",
            "initialization": "theta_i=(i-24)*0.004; i=0..48",
            "uninterrupted_updates": 3000, "serialized_prefix_updates": 300,
            "continuation_updates": 2700, "fixture_total_updates": 6000,
            "state_at_300_binary_exact": True, "final_state_binary_exact": True,
            "final_parameters_moments_counter_exact": True,
            "snapshot_300_sha256": r.identity(r.state_payload(prefix.state)),
            "final_sha256": r.identity(r.state_payload(final.state)),
            "snapshots_preserved": [300, 3000],
            "not_a_thermal_training_or_research_replicate": True,
        }

    def test_short_resume_absolute_target_and_idempotence(self):
        initial = self.start()
        uninterrupted_Q, resumed_Q = Quadratic(), Quadratic()
        full = advance(initial, 8, uninterrupted_Q)
        prefix = advance(initial, 5, resumed_Q)
        path = self.directory / "prefix_5.json"
        save(path, prefix)
        loaded = r.load_checkpoint(path, expected_context=initial.context)
        final = advance(loaded, 8, resumed_Q)
        self.assertEqual(binary_state(final.state), binary_state(full.state))
        calls = resumed_Q.calls
        no_op = advance(final, 8, resumed_Q)
        self.assertEqual(no_op, final)
        self.assertEqual(resumed_Q.calls, calls)
        with self.assertRaises(ValueError):
            advance(final, 7, resumed_Q)
        with self.assertRaises((TypeError, ValueError)):
            advance(final, True, resumed_Q)
        self.assertEqual(prefix.state.updates, 5)
        EVIDENCE["short_resume"] = {"interruption_step": 5, "absolute_target": 8,
                                   "total_updates": 16, "exact": True,
                                   "idempotent_calls": 0, "backwards_rejected": True}

    def test_adapter_M1_matches_existing_API_and_two_updates(self):
        self.adapter_check("M1")

    def test_adapter_M2_matches_existing_API_and_two_updates(self):
        self.adapter_check("M2")

    def adapter_check(self, method):
        labels = r.SuppliedLabels((0.05, 0.14), (0.13, -0.07), (0.4, -0.3))
        scale = PLAN["losses"]
        adapter = CountedAdapter(method, labels, scale["S0"], scale["S1"],
                                 scale["scales_origin"])
        network = n.NeuralSurrogate(self.parameters)
        LEDGER["direct_existing_loss_API_calls"][method] += 1
        record_neural_points(method, 2)
        if method == "M1":
            direct = losses.m1_loss_and_gradient(network, labels.points, labels.values,
                                                S0=scale["S0"])
        else:
            direct = losses.m2_loss_and_gradient(
                network, labels.points, labels.values, labels.derivatives,
                S0=scale["S0"], S1=scale["S1"])
        self.assertEqual(adapter(self.parameters), direct)
        original = (self.parameters.to_vector(), asdict(labels))
        ctx = context(self.parameters, method=method,
                      data_id=labels.data_identity(method), points=labels.points,
                      target=2, snapshots=(2,))
        initial = r.TrainingCheckpoint.start(r.AdamState.zero(self.parameters), ctx)
        final = advance(initial, 2, adapter)
        self.assertEqual(final.status, "COMPLETED")
        self.assertEqual(final.counts.update_completed, 2)
        self.assertEqual((self.parameters.to_vector(), asdict(labels)), original)
        EVIDENCE["adapter_" + method] = {
            "points": 2, "origin": "explicit synthetic tuples, no oracle",
            "direct_API_gradient_49_exact": True, "fixture_updates": 2}

    def test_M1_ignores_supplied_derivatives_and_their_identity(self):
        base = r.SuppliedLabels((0.06, 0.15), (0.2, -0.1))
        other = r.SuppliedLabels(base.points, base.values, (999.0, -777.0))
        scale = PLAN["losses"]
        left = CountedAdapter("M1", base, scale["S0"], scale["S1"], scale["scales_origin"])
        right = CountedAdapter("M1", other, scale["S0"], scale["S1"], scale["scales_origin"])
        self.assertEqual(left(self.parameters), right(self.parameters))
        self.assertEqual(base.data_identity("M1"), other.data_identity("M1"))
        changed = r.SuppliedLabels(base.points, base.values, (1.0, 2.0))
        self.assertNotEqual(other.data_identity("M2"), changed.data_identity("M2"))

    def test_labels_and_adapter_contracts(self):
        invalid = [
            ((), (), None), ((0.1,), (), None), ((0.1, 0.1), (0.0, 0.0), None),
            ((True,), (0.0,), None), ((0.01,), (0.0,), None),
            ((0.1,), (math.nan,), None), ((0.1,), (0.0,), (math.inf,)),
            ((0.1,), (0.0,), ()),
        ]
        for args in invalid:
            with self.subTest(args=args), self.assertRaises((TypeError, ValueError)):
                r.SuppliedLabels(*args)
        labels = r.SuppliedLabels((0.1,), (0.0,))
        for args in (("M2", labels, 1.0, 1.0, "synthetic"),
                     ("M1", labels, 0.0, 1.0, "synthetic"),
                     ("M1", labels, 1.0, True, "synthetic"),
                     ("M1", labels, 1.0, 1.0, "")):
            with self.subTest(args=args), self.assertRaises((TypeError, ValueError)):
                r.LossAdapter(*args)

    def test_planning_sixty_jobs_and_one_twenty_snapshots(self):
        jobs = r.build_jobs(PLAN)
        self.assertEqual(len(jobs), 60)
        self.assertEqual(len({(j["seed"], j["condition"]) for j in jobs}), 60)
        self.assertEqual(sum(len(j["snapshot_paths"]) for j in jobs), 120)
        self.assertEqual([j["condition"] for j in jobs[:3]], list(r.CONDITIONS))
        self.assertEqual(jobs[0]["seed"], 20262001)
        self.assertEqual(jobs[-1]["seed"], 20262020)
        self.assertTrue(all(j["label_identity"] is None for j in jobs))
        self.assertTrue(all(j["target_updates"] == 3000 for j in jobs))
        for seed in PLAN["replication"]["seeds"]:
            grouped = [j for j in jobs if j["seed"] == seed]
            self.assertEqual(len({j["initial_parameter_id"] for j in grouped}), 1)
            self.assertEqual(grouped[0]["points"], grouped[1]["points"])
            self.assertFalse(set(grouped[0]["points"]) & set(grouped[2]["points"]))
        EVIDENCE["planning"] = {"jobs": 60, "output_snapshots": 120,
                                "jobs_executed": 0, "labels_identity_not_acquired": True}

    def test_initialization_pilot_exact_local_and_paired(self):
        global_before = random.getstate()
        seed = PILOT_INITIALIZATION["seed"]
        parameters = r.initialize_parameters(seed)
        self.assertEqual(parameters.to_vector(), tuple(PILOT_INITIALIZATION["parameters"]))
        self.assertEqual(r.identity(parameters.to_vector()), PILOT_INITIALIZATION["parameter_id"])
        independent = random.Random(19)
        bound = math.sqrt(6.0 / 17.0)
        expected_w = tuple(independent.uniform(-bound, bound) for _ in range(16))
        expected_v = tuple(independent.uniform(-bound, bound) for _ in range(16))
        actual = r.initialize_parameters(19)
        self.assertEqual(actual.w, expected_w)
        self.assertEqual(actual.v, expected_v)
        for seed in PLAN["replication"]["seeds"]:
            states = r.paired_initial_states(seed)
            self.assertEqual(len({r.identity(s.parameters.to_vector()) for s in states.values()}), 1)
            self.assertEqual(len({id(s) for s in states.values()}), 3)
            self.assertEqual(len({id(s.parameters) for s in states.values()}), 3)
            self.assertEqual(len({id(s.first_moment) for s in states.values()}), 3)
            self.assertEqual(len({id(s.second_moment) for s in states.values()}), 3)
            self.assertTrue(all(isinstance(s.first_moment, tuple) for s in states.values()))
        self.assertEqual(random.getstate(), global_before)

    def test_frozen_plan_rejects_contradictions_and_booleans(self):
        for section, key, value in (
            ("optimizer", "eta", 0.002), ("optimizer", "epsilon_position", "sqrt(v+epsilon)"),
            ("losses", "half_factor", True), ("losses", "sensitivity_coefficient", True),
            ("replication", "updates_per_trajectory", 300),
            ("replication", "snapshots", [3000]), ("replication", "full_batch", False),
        ):
            changed = deepcopy(PLAN)
            changed[section][key] = value
            with self.subTest(section=section, key=key), self.assertRaises(ValueError):
                r.validate_plan(changed)

    def test_signature_excludes_dates_and_duration(self):
        changed = deepcopy(PLAN)
        changed["mission_date"] = "2099-01-01"
        changed["execution_metadata"] = {"start_utc": "2099", "duration": 999}
        self.assertEqual(r.protocol_signature(changed), r.protocol_signature(PLAN))
        data = r.SuppliedLabels((0.1,), (0.2,))
        altered = r.SuppliedLabels((0.1,), (0.3,))
        self.assertNotEqual(data.data_identity("M1"), altered.data_identity("M1"))

    def test_checkpoint_float_roundtrip_and_identical_output_reuse(self):
        values = tuple(-0.0 if i == 0 else (i - 24) * 0.004 for i in range(49))
        parameters = n.NetworkParameters.from_vector(values)
        ctx = context(parameters, target=4, snapshots=(4,))
        cp = r.TrainingCheckpoint.start(r.AdamState.zero(parameters), ctx)
        path = self.directory / "checkpoint.json"
        self.assertTrue(save(path, cp)["written"])
        raw = path.read_bytes()
        restored = r.load_checkpoint(path, expected_context=ctx)
        self.assertEqual(binary_state(cp.state), binary_state(restored.state))
        self.assertFalse(save(path, restored)["written"])
        self.assertEqual(path.read_bytes(), raw)

    def test_restore_refuses_context_mismatches_before_callback(self):
        cp = self.start()
        payload = r.checkpoint_payload(cp, expected_context=cp.context)
        mutations = [
            ("seed", 74), ("condition", "OTHER"), ("data_identity", "A" * 64),
            ("protocol_signature", "B" * 64), ("schema_version", 2),
            ("initial_parameter_id", "C" * 64),
        ]
        contexts = []
        for key, value in mutations:
            changed = deepcopy(cp.context)
            changed[key] = value
            contexts.append((key, changed))
        for key, value in (("S0", 2.0), ("S1", 3.0)):
            changed = deepcopy(cp.context)
            changed["scales"][key] = value
            contexts.append((key, changed))
        for field, key, value in (
            ("adam", "eta", 0.002), ("normalization", "s_out", 2.0),
            ("environment", "python_version", "3.12.0"),
        ):
            changed = deepcopy(cp.context)
            changed[field][key] = value
            contexts.append((field, changed))
        changed = deepcopy(cp.context)
        changed["critical_sources"]["changed.py"] = "D" * 64
        contexts.append(("sources", changed))
        q = Quadratic()
        for label, changed in contexts:
            with self.subTest(label=label):
                with self.assertRaises((ValueError, TypeError)):
                    r.restore_payload(payload, expected_context=changed)
                with self.assertRaises((ValueError, TypeError)):
                    r.continue_to(cp, 1, q, expected_context=changed)
        self.assertEqual(q.calls, 0)
        EVIDENCE["provenance_refusals"] = {"contexts_tested": len(contexts),
                                         "callbacks_after_mismatch": 0}

    def test_checkpoint_integrity_nonfinite_and_duplicate_keys(self):
        cp = self.start()
        path = self.directory / "copy_corrupted.json"
        payload = r._copy_json(r.checkpoint_payload(cp, expected_context=cp.context))
        corrupted = deepcopy(payload)
        corrupted["state"]["parameters"][0] = 999.0
        with self.assertRaises(ValueError):
            r.restore_payload(corrupted, expected_context=cp.context)
        payload["state"]["parameters"][0] = math.nan
        with self.assertRaises(ValueError):
            r.restore_payload(payload, expected_context=cp.context)
        for text in ('{"x":NaN}', '{"x":Infinity}', '{"x":1,"x":2}'):
            path.write_text(text, encoding="utf-8")
            with self.assertRaises(ValueError):
                r.read_json(path)

    def test_atomic_failure_preserves_last_valid_checkpoint(self):
        base = self.start(target=4, snapshots=(2, 4))
        path = self.directory / "checkpoint.json"
        save(path, base)
        original = path.read_bytes()
        candidate = advance(base, 2, Quadratic())
        with patch.object(r.os, "replace", side_effect=OSError("simulated replacement failure")):
            with self.assertRaises(OSError):
                save(path, candidate)
        self.assertEqual(path.read_bytes(), original)
        self.assertEqual(r.load_checkpoint(path, expected_context=base.context), base)
        self.assertFalse(list(self.directory.glob("*.tmp")))
        EVIDENCE["atomic_write_failure"] = {"simulated": True,
                                            "last_valid_bytes_preserved": True}

    def test_collisions_rollback_and_completed_output_policy(self):
        initial = self.start(target=4, snapshots=(2, 4))
        path = self.directory / "checkpoint.json"
        prefix = advance(initial, 2, Quadratic())
        save(path, prefix)
        original = path.read_bytes()
        with self.assertRaises(ValueError):
            save(path, initial)
        incompatible = deepcopy(prefix.context)
        incompatible["seed"] += 1
        with self.assertRaises(ValueError):
            save(path, prefix, incompatible)
        changed = list(prefix.state.parameters.to_vector())
        changed[0] += 0.01
        state = replace(prefix.state, parameters=n.NetworkParameters.from_vector(changed))
        bad_same_step = replace(prefix, state=state, snapshots={"2": state})
        with self.assertRaises(ValueError):
            save(path, bad_same_step)
        self.assertEqual(path.read_bytes(), original)
        completed = advance(prefix, 4, Quadratic())
        save(path, completed)
        completed_raw = path.read_bytes()
        self.assertFalse(save(path, completed)["written"])
        different_cost = r.retain_known_recomputed_cost(completed, 1,
                                                       expected_context=completed.context)
        with self.assertRaises(ValueError):
            save(path, different_cost)
        self.assertEqual(path.read_bytes(), completed_raw)

    def test_missing_acquired_snapshot_is_rejected(self):
        prefix = advance(self.start(), 3, Quadratic())
        with self.assertRaises(ValueError):
            replace(prefix, snapshots={})

    def test_logging_is_separate_idempotent_and_known_cost_retained(self):
        initial = self.start(target=2, snapshots=(2,))
        q = Quadratic()
        logged = r.log_loss_gradient(initial, q, expected_context=initial.context)
        self.assertEqual(logged.state, initial.state)
        self.assertEqual(logged.counts.logging_completed, 1)
        self.assertEqual(r.log_loss_gradient(logged, q, expected_context=initial.context), logged)
        final = advance(logged, 2, q)
        final = r.log_loss_gradient(final, q, expected_context=initial.context)
        self.assertEqual(final.state.updates, 2)
        self.assertEqual(final.counts.update_completed, 2)
        self.assertEqual(final.counts.logging_completed, 2)
        retained = r.retain_known_recomputed_cost(final, 3, expected_context=initial.context)
        self.assertEqual(retained.counts.known_recomputed_loss_gradient_calls, 3)
        self.assertEqual(retained.state, final.state)
        self.assertEqual(q.calls, 4)
        EVIDENCE["cost_bookkeeping"] = {
            "fixture_updates": 2, "logging_calls_executed": 2,
            "known_recomputed_calls_bookkeeping_fixture": 3,
            "bookkeeping_example_is_not_three_extra_executions": True}

    def test_failed_candidate_records_cost_without_restart(self):
        initial = self.start()
        def huge_gradient(parameters):
            LEDGER["other_explicit_synthetic_loss_gradient_calls"] += 1
            return SimpleNamespace(loss=1.0, gradient=(1e200,) * 49)
        failed = advance(initial, 1, huge_gradient)
        self.assertEqual(failed.status, "STOPPED_ERROR")
        self.assertEqual(binary_state(failed.state), binary_state(initial.state))
        self.assertEqual(failed.counts.update_attempts, 1)
        self.assertEqual(failed.counts.update_completed, 1)
        self.assertEqual(failed.incident["stage"], "Adam_candidate")
        q = Quadratic()
        with self.assertRaises(ValueError):
            advance(failed, 1, q)
        self.assertEqual(q.calls, 0)

    def test_context_from_real_job_requires_actual_matching_labels(self):
        job = r.build_jobs(PLAN)[0]
        labels = r.SuppliedLabels(job["points"], tuple((i - 8) * 0.01 for i in range(16)))
        scale = PLAN["losses"]
        adapter = r.LossAdapter("M1", labels, scale["S0"], scale["S1"], scale["scales_origin"])
        ctx = r.context_from_labels(job, adapter, critical_sources=CRITICAL_SOURCES)
        self.assertEqual(ctx["data_identity"], labels.data_identity("M1"))
        self.assertEqual(ctx["points"], list(labels.points))
        wrong = r.LossAdapter("M1", r.SuppliedLabels((0.1,), (0.1,)),
                              scale["S0"], scale["S1"], scale["scales_origin"])
        with self.assertRaises(ValueError):
            r.context_from_labels(job, wrong, critical_sources=CRITICAL_SOURCES)
        for changed in (
            r.LossAdapter("M1", labels, 2 * scale["S0"], scale["S1"], scale["scales_origin"]),
            r.LossAdapter("M1", labels, scale["S0"], scale["S1"], "wrong scale origin"),
            r.LossAdapter("M1", labels, scale["S0"], scale["S1"], scale["scales_origin"],
                          n.AffineNormalization(s_out=2.0)),
        ):
            with self.subTest(changed=changed), self.assertRaises(ValueError):
                r.context_from_labels(job, changed, critical_sources=CRITICAL_SOURCES)
        # Building a context on synthetic labels is not executing the planned job.
        EVIDENCE["frozen_job_context_binding"] = {
            "wrong_sites_scales_origin_normalization_refused": True,
            "job_Adam_and_targets_used_without_defaults": True}

    def test_evaluator_context_mismatch_precedes_any_updates(self):
        scale = PLAN["losses"]
        labels = r.SuppliedLabels((0.05, 0.14), (0.13, -0.07), (0.4, -0.3))
        ctx = context(self.parameters, method="M1", data_id=labels.data_identity("M1"),
                      points=labels.points, target=2, snapshots=(2,))
        initial = r.TrainingCheckpoint.start(r.AdamState.zero(self.parameters), ctx)
        bad = CountedAdapter("M1", r.SuppliedLabels(labels.points, (0.0, 0.0)),
                             scale["S0"], scale["S1"], scale["scales_origin"])
        before = LEDGER["synthetic_neural_loss_gradient_calls"]["M1"]
        with self.assertRaises(ValueError):
            advance(initial, 1, bad)
        self.assertEqual(LEDGER["synthetic_neural_loss_gradient_calls"]["M1"], before)
