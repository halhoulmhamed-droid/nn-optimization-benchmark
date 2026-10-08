"""New phase3B integration controls: explicit synthetic data, no thermal import.

Tolerances are declared before execution. Short paths are operational fixtures,
not scientific jobs or repeats of the long continuation controls from phase3A.
"""
from __future__ import annotations

import csv
from dataclasses import replace
import json
import math
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

from src import replication_campaign as c
from src import replication_training as r
from src import neural_surrogate as n
from src import surrogate_losses as losses

FIXTURE_ROOT = None
PLAN = None
SOURCES = None
TOLERANCES = {"direct_adapter_absolute": 0.0, "direct_adapter_relative": 0.0,
              "same_runtime_resume": "exact states, counts, logs and metrics",
              "CSV_float_round_trip": "exact float equality"}
LEDGER = {"synthetic_updates": 0, "synthetic_loss_gradient_attempts": 0,
          "synthetic_logging_attempts": 0, "direct_loss_api_controls": 0,
          "synthetic_value_callbacks": 0, "synthetic_derivative_callbacks": 0,
          "snapshot_measurements": 0, "snapshot_network_joint_calls": 0,
          "snapshot_grid_predictions": 0, "snapshot_discrete_objectives": 0,
          "real_thermal_acquisitions": 0, "real_M0_calls": 0}
EVIDENCE = {}


def synthetic_value(p):
    LEDGER["synthetic_value_callbacks"] += 1
    return 0.5 - 0.2*p + 2.0*p*p


def synthetic_derivative(p):
    LEDGER["synthetic_derivative_callbacks"] += 1
    return -0.2 + 4.0*p


def cache_fixture():
    return c.PhysicalCache(r.identity({"scope": "TEST_FIXTURE", "formula": "0.5-0.2p+2p^2"}),
                           value_call=synthetic_value, derivative_call=synthetic_derivative,
                           scope="TEST_FIXTURE")


def short_fixture(method="M1", target=6):
    cache = cache_fixture()
    points = (0.055, 0.145)
    values = tuple(cache.get("T", p, role="training") for p in points)
    derivatives = (tuple(cache.get("T_prime", p, role="training") for p in points)
                   if method == "M2" else None)
    labels = r.SuppliedLabels(points, values, derivatives)
    diagnostic_points = (0.041, 0.093, 0.167)
    diag = r.SuppliedLabels(
        diagnostic_points, tuple(cache.get("T", p, role="diagnostics") for p in diagnostic_points),
        tuple(cache.get("T_prime", p, role="diagnostics") for p in diagnostic_points))
    boundaries = r.SuppliedLabels(
        (n.A, n.B), tuple(cache.get("T", p, role="boundaries") for p in (n.A,n.B)),
        tuple(cache.get("T_prime", p, role="boundaries") for p in (n.A,n.B)))
    job = r._copy_json(r.build_jobs(PLAN)[0])
    job.update(seed=73, condition="TEST_FIXTURE_"+method, method=method, points=list(points),
               target_updates=target, snapshot_updates=[2,target],
               initial_parameter_id=r.identity(r.initialize_parameters(73).to_vector()),
               protocol_signature=r.identity({"scope": "TEST_FIXTURE", "target": target}),
               job_id=r.identity(["TEST_FIXTURE", method, target]))
    primary = PLAN["decisions"]["primary_tasks"]
    tasks = [dict(t, task_class=("primary_interior" if t["id"] in primary
                               else "secondary_boundary_or_transition"))
             for t in PLAN["decisions"]["tasks"]]
    # Independent synthetic references. Their labels are not used for selection.
    refs = {t["id"]: dict(p_ref=n.A, j_ref=boundaries.values[0]+t["lambda_value"]*n.A,
                          branch="lower_boundary",
                          reference_value_error_estimate_conditional=0.0,
                          scope="TEST_FIXTURE") for t in tasks}
    return job, labels, cache, diag, boundaries, tasks, refs


def account(status, before=None):
    before = before or {"logical_updates": 0, "counts": {"update_attempts": 0,
                                                       "logging_attempts": 0}, "measures": {}}
    LEDGER["synthetic_updates"] += status["logical_updates"]-before["logical_updates"]
    LEDGER["synthetic_loss_gradient_attempts"] += (
        status["counts"]["update_attempts"]-before["counts"]["update_attempts"])
    LEDGER["synthetic_logging_attempts"] += (
        status["counts"]["logging_attempts"]-before["counts"]["logging_attempts"])
    for step, measure in status["measures"].items():
        if step not in before["measures"]:
            LEDGER["snapshot_measurements"] += 1
            counts = measure["counters"]
            LEDGER["snapshot_network_joint_calls"] += (
                counts["diagnostic_joint_network_evaluations"]+
                counts["boundary_joint_network_evaluations"])
            LEDGER["snapshot_grid_predictions"] += counts["grid_forward_network_evaluations"]
            LEDGER["snapshot_discrete_objectives"] += counts["discrete_objective_evaluations"]


class CampaignIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.directory = Path(FIXTURE_ROOT)/self._testMethodName
        self.directory.mkdir(exist_ok=False)

    def run_short(self, fixture, name, **options):
        job, labels, cache, diag, boundaries, tasks, refs = fixture
        status = c.run_job(job, labels, cache, diag, boundaries, tasks, refs, SOURCES,
                           self.directory/name, save_every=1, grid_intervals=8,
                           cache_path=self.directory/(name+"_cache.json"), **options)
        return status

    def test_cache_shared_calls_and_exact_site_identity(self):
        cache = cache_fixture()
        p, adjacent = 0.08, math.nextafter(0.08, n.B)
        cache.get("T", p, role="training")
        cache.get("T", p, role="diagnostics")
        cache.get("T_prime", p, role="diagnostics")
        cache.get("T", adjacent, role="decisions")
        self.assertEqual(cache.counts["T"]["new_calls"], 2)
        self.assertEqual(cache.counts["T"]["hits"], 1)
        self.assertEqual(cache.counts["T_prime"]["new_calls"], 1)
        self.assertNotEqual(cache.key("T", p), cache.key("T", adjacent))
        path = self.directory/"cache.json"
        cache.save(path)
        restored = c.PhysicalCache.load(path, cache.oracle_id)
        self.assertEqual(cache.payload(), restored.payload())
        with self.assertRaises(c.IntegrityError):
            cache.seed("T", p, 1.0, origin="TEST_FIXTURE", provenance={"fixture": "changed"})
        for invalid in (True, float("nan"), float("inf"), n.A-0.001):
            with self.assertRaises((TypeError,ValueError)):
                cache.get("T", invalid, role="invalid")
        with self.assertRaises(ValueError):
            cache.get("temperature_rounded", p, role="invalid")

    def test_full_geometry_shared_acquisition_and_training_separation(self):
        cache = cache_fixture()
        training, diag, boundaries = c.acquire_datasets(PLAN, cache)
        self.assertEqual(cache.counts["T"]["new_calls"], 307)
        self.assertEqual(cache.counts["T_prime"]["new_calls"], 275)
        self.assertEqual(cache.counts["T"]["requests"], 323)
        self.assertEqual(cache.counts["T"]["hits"], 16)
        self.assertEqual(len(diag.points), 257)
        self.assertFalse(set(diag.points) & (set(training["M1_16"].points) |
                                             set(training["M1_32"].points)))
        self.assertIsNone(training["M1_16"].derivatives)
        self.assertIsNone(training["M1_32"].derivatives)
        self.assertEqual(len(training["M2_16"].derivatives),16)
        cache.callbacks = {"T": None, "T_prime": None}
        for job in r.build_jobs(PLAN)[:2]:
            adapter, ctx = c.make_job_context(job, training[job["condition"]], cache, SOURCES)
            cp = r.TrainingCheckpoint.start(r.AdamState.zero(r.initialize_parameters(job["seed"])),
                                            ctx)
            cp = r.continue_to(cp,2,adapter,expected_context=ctx)
            self.assertEqual(cp.state.updates,2)
            LEDGER["synthetic_updates"] += cp.state.updates
            LEDGER["synthetic_loss_gradient_attempts"] += cp.counts.update_attempts
            self.assertFalse(set(adapter.labels.points) & set(diag.points))
        EVIDENCE["synthetic_dataset_geometry"] = dict(values=307, derivatives=275,
            diagnostic_sites=257, disjoint=True, physical_callbacks_disabled_during_updates=True)

    def test_label_provenance_subset_and_used_label_rejection(self):
        job, labels, cache, diag, boundaries, tasks, refs = short_fixture()
        adapter, context = c.make_job_context(job, labels, cache, SOURCES)
        cp = r.TrainingCheckpoint.start(r.AdamState.zero(r.initialize_parameters(73)),context)
        path = self.directory/"checkpoint.json"
        r.save_checkpoint(path,cp,expected_context=context)
        cache.get("T",0.077,role="decisions")
        self.assertEqual(context,c.make_job_context(job,labels,cache,SOURCES)[1])
        cache.get("T_prime",labels.points[0],role="unused_derivative")
        cache.entries[cache.key("T_prime",labels.points[0])]["value"] += 100.0
        self.assertEqual(context,c.make_job_context(job,labels,cache,SOURCES)[1])
        cache.entries[cache.key("T",labels.points[0])]["value"] += 0.1
        changed = c.make_job_context(job,labels,cache,SOURCES)[1]
        with self.assertRaises(ValueError):
            r.load_checkpoint(path,expected_context=changed)
        self.assertEqual(r.load_checkpoint(path,expected_context=context).state, cp.state)

    def test_short_continuous_and_persisted_resume_exact_no_duplicate(self):
        continuous_fixture = short_fixture("M2")
        continuous = self.run_short(continuous_fixture,"continuous")
        account(continuous)
        resumed_fixture = short_fixture("M2")
        paused = self.run_short(resumed_fixture,"resumed",stop_after=3)
        account(paused)
        cache_path = self.directory/"resumed_cache.json"
        job, labels, cache, diag, boundaries, tasks, refs = resumed_fixture
        cache = c.PhysicalCache.load(cache_path,cache.oracle_id,
                                    value_call=synthetic_value,derivative_call=synthetic_derivative)
        resumed_fixture = (job,labels,cache,diag,boundaries,tasks,refs)
        resumed = self.run_short(resumed_fixture,"resumed",resume=True)
        account(resumed,paused)
        context = c.make_job_context(job,labels,cache,SOURCES)[1]
        left = r.load_checkpoint(self.directory/"continuous"/"training_state.json",
                                 expected_context=context)
        right = r.load_checkpoint(self.directory/"resumed"/"training_state.json",
                                  expected_context=context)
        self.assertEqual(left.state,right.state)
        self.assertEqual(left.snapshots,right.snapshots)
        self.assertEqual(left.counts,right.counts)
        self.assertEqual(left.logs,right.logs)
        self.assertEqual(continuous["measures"],resumed["measures"])
        self.assertEqual(resumed["counts"]["update_attempts"],6)
        self.assertEqual(resumed["counts"]["logging_attempts"],3)
        checkpoint_hash = c.file_hash(self.directory/"resumed"/"training_state.json")
        again = self.run_short(resumed_fixture,"resumed",resume=True)
        account(again,resumed)
        self.assertEqual(again,resumed)
        self.assertEqual(checkpoint_hash,c.file_hash(self.directory/"resumed"/"training_state.json"))
        self.assertEqual(len(list((self.directory/"resumed").glob("snapshot_*.json"))),2)
        EVIDENCE["short_resume"] = dict(updates=6,paused_at=3,exact_state=True,
            exact_moments=True,exact_snapshots=True,exact_logs=True,
            exact_measurements=True,second_resume_no_update=True,
            decision_labels_persisted_before_status=True)

    def test_changed_training_label_rejected_before_resume_write(self):
        fixture = short_fixture(target=4)
        paused = self.run_short(fixture,"job",stop_after=1)
        account(paused)
        job, labels, cache, diag, boundaries, tasks, refs = fixture
        state_path = self.directory/"job"/"training_state.json"
        digest = c.file_hash(state_path)
        new_values = list(labels.values)
        new_values[0] += 0.01
        changed = r.SuppliedLabels(labels.points,new_values)
        cache.entries[cache.key("T",labels.points[0])]["value"] = new_values[0]
        with self.assertRaises(ValueError):
            self.run_short((job,changed,cache,diag,boundaries,tasks,refs),"job",resume=True)
        self.assertEqual(digest,c.file_hash(state_path))

    def test_adapter_direct_apis_and_m1_ignores_derivatives(self):
        for method in ("M1","M2"):
            job,labels,cache,*unused = short_fixture(method)
            adapter,context = c.make_job_context(job,labels,cache,SOURCES)
            parameters = r.initialize_parameters(73)
            model = n.NeuralSurrogate(parameters,adapter.normalization)
            if method == "M1":
                direct = losses.m1_loss_and_gradient(model,labels.points,labels.values,S0=adapter.S0)
                other = r.LossAdapter("M1",r.SuppliedLabels(labels.points,labels.values,(1e6,-1e6)),
                                      adapter.S0,adapter.S1,adapter.scales_origin,adapter.normalization)
                self.assertEqual(adapter(parameters),other(parameters))
                LEDGER["direct_loss_api_controls"] += 2
            else:
                direct = losses.m2_loss_and_gradient(model,labels.points,labels.values,
                                                    labels.derivatives,S0=adapter.S0,S1=adapter.S1)
            self.assertEqual(adapter(parameters),direct)
            LEDGER["direct_loss_api_controls"] += 2

    def test_selection_before_physical_evaluation(self):
        job,labels,cache,diag,boundaries,tasks,refs = short_fixture()
        adapter,context = c.make_job_context(job,labels,cache,SOURCES)
        state = r.AdamState.zero(r.initialize_parameters(73))
        grid = c.sd.make_grid(n.A,n.B,8)
        model = n.NeuralSurrogate(state.parameters,adapter.normalization)
        first,bound = c.select_model(model,tasks,grid)
        # Change evaluation labels only after all candidates have been selected.
        for entry in cache.entries.values():
            if entry["quantity"] == "T":
                entry["value"] += 100.0
        second,_ = c.select_model(model,tasks,grid)
        self.assertEqual([s.p_hat for s in first],[s.p_hat for s in second])
        measured = c.measure_snapshot(state,context,diag,boundaries,tasks,refs,cache,
                                      grid_intervals=8)
        self.assertTrue(measured["all_selections_completed_before_physical_decision_requests"])
        self.assertEqual([s.p_hat for s in first],[d["p_hat"] for d in measured["decisions"]])
        LEDGER["snapshot_measurements"] += 1
        LEDGER["snapshot_grid_predictions"] += 27 # two selections plus measured selection
        LEDGER["snapshot_discrete_objectives"] += 27*7
        LEDGER["snapshot_network_joint_calls"] += 5

    def test_failure_preserved_and_csv_missing_not_zero(self):
        fixture = short_fixture(target=4)
        with patch.object(r.LossAdapter,"__call__",side_effect=ArithmeticError("TEST_FIXTURE failure")):
            failed = self.run_short(fixture,"failed")
        account(failed)
        self.assertEqual(failed["status"],"FAILED")
        self.assertEqual(failed["logical_updates"],0)
        self.assertEqual(failed["counts"]["logging_attempts"],1)
        self.assertEqual(failed["missing_snapshot_steps"],[2,4])
        job,*unused,tasks,refs = fixture
        metrics,decisions = c.aggregate_rows([job],{(job["seed"],job["condition"]):failed},tasks)
        self.assertEqual((len(metrics),len(decisions)),(2,14))
        self.assertTrue(all(row["status"] == "MISSING" for row in metrics+decisions))
        path = self.directory/"metrics.csv"
        c.write_csv(path,c.METRIC_FIELDS,metrics)
        with path.open(encoding="utf-8",newline="") as handle:
            rows = list(csv.DictReader(handle))
        self.assertTrue(all(row["rmse_value"] == "" for row in rows))
        self.assertTrue(all(row["missing_reason"] for row in rows))
        self.assertFalse(list((self.directory/"failed").glob("snapshot_*.json")))
        EVIDENCE["failure_policy"] = dict(logging_failure_kept=True,
            numerical_fields_missing_not_zero=True, reserved_metric_rows=2,reserved_decision_rows=14)

    def test_measurement_failure_keeps_completed_training_state(self):
        fixture = short_fixture(target=4)
        original = c.measure_snapshot
        def failed_at_final(state,*args,**kwargs):
            if state.updates == 4:
                raise ArithmeticError("TEST_FIXTURE final measurement failed")
            return original(state,*args,**kwargs)
        with patch.object(c,"measure_snapshot",side_effect=failed_at_final):
            status = self.run_short(fixture,"job")
        account(status)
        self.assertEqual(status["status"],"FAILED")
        self.assertEqual(status["logical_updates"],4)
        self.assertIn("2",status["measures"])
        self.assertNotIn("4",status["measures"])
        job,labels,cache,*unused = fixture
        context = c.make_job_context(job,labels,cache,SOURCES)[1]
        cp = r.load_checkpoint(self.directory/"job"/"training_state.json",expected_context=context)
        self.assertEqual(cp.status,"COMPLETED")
        self.assertEqual(cp.state.updates,4)
        self.assertEqual(len(list((self.directory/"job").glob("snapshot_*.json"))),2)

    def test_atomic_cache_collision_failure_and_invalid_value(self):
        cache = cache_fixture()
        cache.get("T",0.05,role="training")
        path = self.directory/"cache.json"
        cache.save(path)
        digest = c.file_hash(path)
        cache.get("T",0.06,role="training")
        with patch.object(c.os,"replace",side_effect=OSError("TEST_FIXTURE disk failure")):
            with self.assertRaises(OSError):
                cache.save(path)
        self.assertEqual(c.file_hash(path),digest)
        self.assertFalse(list(self.directory.glob("*.tmp")))
        invalid = cache.payload()
        next(iter(invalid["entries"].values()))["value"] = float("nan")
        with self.assertRaises(ValueError):
            c.atomic_record(path,invalid)
        self.assertEqual(c.file_hash(path),digest)
        body = cache.payload()
        body["context_id"] = r.identity({"different":"TEST_FIXTURE"})
        with self.assertRaises(c.IntegrityError):
            c.atomic_record(path,body)
        self.assertEqual(c.file_hash(path),digest)

    def test_csv_exact_floats_unique_and_negative_D_preserved(self):
        metrics = [dict(seed=73,condition="TEST_FIXTURE",step=2,status="AVAILABLE",
                        rmse_value=0.012345678901234567,mean_D_interior=-1.2e-18)]
        path = self.directory/"metrics.csv"
        c.write_csv(path,c.METRIC_FIELDS,metrics)
        with path.open(encoding="utf-8",newline="") as handle:
            row = next(csv.DictReader(handle))
        self.assertEqual(float(row["rmse_value"]),metrics[0]["rmse_value"])
        self.assertEqual(float(row["mean_D_interior"]),-1.2e-18)
        self.assertEqual(len(row),len(c.METRIC_FIELDS))

    def test_real_plan_without_acquisition_or_M0(self):
        jobs = r.build_jobs(PLAN)
        keys = {(j["seed"],j["condition"]) for j in jobs}
        self.assertEqual(len(keys),60)
        self.assertEqual(sum(len(j["snapshot_updates"]) for j in jobs),120)
        self.assertEqual(sum(j["target_updates"] for j in jobs),180000)
        self.assertEqual(len(PLAN["decisions"]["tasks"]),7)
        self.assertEqual(len(PLAN["decisions"]["primary_tasks"]),3)
        for i in range(0,60,3):
            self.assertEqual(len({j["initial_parameter_id"] for j in jobs[i:i+3]}),1)
        self.assertNotIn("src.thermal_oracle",sys.modules)
        self.assertNotIn("src.physical_reference",sys.modules)
        EVIDENCE["plan"] = dict(jobs=60, snapshots=120, updates=180000,
            seeds=20, conditions=3, thermal_oracle_not_imported=True)

