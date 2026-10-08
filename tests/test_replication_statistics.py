"""Independent small statistical fixtures. No historical/model tests executed."""
from __future__ import annotations

import copy
import csv
import json
import math
from pathlib import Path
import random
import unittest

from src import replication_statistics as s

FIXTURE_ROOT = None
PLAN = None
TOLERANCES = {"absolute":s.STAT_TEST_ABS,"relative":s.STAT_TEST_REL,
              "data_coherence_absolute":s.COHERENCE_ABS,"data_coherence_relative":s.COHERENCE_REL,
              "ranks_indices_CSV_round_trip":"exact"}
EVIDENCE = {}
COUNTERS = {"synthetic_bootstrap_rows":0,"real_campaign_bootstraps":0,
            "physical_calls":0,"neural_calls":0,"training_updates":0}


def small_blocks():
    blocks = []
    base_scores = [3.0,1.0,4.0,2.0]
    for i in range(4):
        models = {}
        for step in (300,3000):
            for condition,offset in (("M1_16",0.0),("M2_16",-0.5),("M1_32",0.5)):
                models[s.model_key(condition,step)] = dict(
                    rmse_value=float(i+1),rmse_derivative=float(i+1),
                    S=base_scores[i]+offset+(0.25 if step==300 else 0.0))
        blocks.append(dict(seed=i+1,models=models))
    return tuple(blocks)


def small_specs():
    return [
        dict(id="rho_value",kind="rho_value",condition="M1_16",step=3000),
        dict(id="rho_sensitivity",kind="rho_sensitivity",condition="M1_16",step=3000),
        dict(id="Delta",kind="Delta_rho",condition="M1_16",step=3000),
        dict(id="mean_A",kind="paired_mean_S",condition="M2_16",baseline="M1_16",step=3000),
        dict(id="mean_A_300",kind="paired_mean_S",condition="M2_16",baseline="M1_16",step=300)]


def fixture_tables():
    recipe = s.recipe_from_plan(PLAN)
    recipe.update(scope="TEST_FIXTURE",seeds=[1,2])
    refs = {t["id"]:dict(p_ref=0.02,j_ref=1.0,branch="lower_boundary",
                        reference_value_error_estimate_conditional=0.0)
            for t in recipe["tasks"]}
    metrics,decisions = [],[]
    for seed in recipe["seeds"]:
        for condition in recipe["conditions"]:
            for step in recipe["steps"]:
                row = {h:"0.0" for h in s.METRIC_HEADERS}
                row.update(seed=str(seed),condition=condition,step=str(step),status="AVAILABLE",
                           missing_reason="",parameter_id="A"*64,rmse_value=repr(seed*0.1),
                           rmse_derivative=repr(seed*0.2),mean_D_interior=repr(seed*0.01),
                           n_value_labels="16",n_derivative_labels="0",C_label_q1_1="16")
                metrics.append(row)
                for task in recipe["tasks"]:
                    decision = {h:"0.0" for h in s.DECISION_HEADERS}
                    decision.update(seed=str(seed),condition=condition,step=str(step),task=task["id"],
                        task_class="primary_interior" if task["id"] in recipe["primary_tasks"]
                                   else "secondary_boundary_or_transition",
                        lambda_value=repr(task["lambda_value"]),status="AVAILABLE",missing_reason="",
                        p_hat="0.02",p_ref="0.02",physical_branch="lower_boundary",
                        candidate_origin="lower_boundary",physical_objective=repr(1.0+seed*0.01),
                        reference_objective="1.0",D=repr(seed*0.01),branch_mismatch="False",
                        D_negative="False",numerical_status="FLOATING_D_NOT_CERTIFIED_REGRET")
                    decisions.append(decision)
    return recipe,refs,metrics,decisions


class StatisticalTests(unittest.TestCase):
    def setUp(self):
        self.directory = Path(FIXTURE_ROOT)/self._testMethodName
        self.directory.mkdir(exist_ok=False)

    def close(self,a,b):
        self.assertTrue(math.isclose(a,b,abs_tol=s.STAT_TEST_ABS,rel_tol=s.STAT_TEST_REL),
                        f"{a!r} versus {b!r}")

    def test_average_ranks_known_ties(self):
        self.assertEqual(s.average_ranks([3,1,1,2]),(4.0,1.5,1.5,3.0))
        profile = s.tie_profile([3,1,1,2])
        self.assertEqual((profile["distinct"],profile["tie_groups"],profile["tied_observations"]),(3,1,2))

    def test_exact_ties_no_near_equal_grouping(self):
        adjacent = math.nextafter(1.0,2.0)
        self.assertEqual(s.average_ranks([adjacent,1.0]),(2.0,1.0))
        self.assertEqual(s.tie_profile([adjacent,1.0])["distinct"],2)

    def test_concordant_and_reversed_signed_correlations(self):
        self.close(s.spearman([3,4,8,10],[30,40,80,100])["value"],1.0)
        self.close(s.spearman([3,4,8,10],[100,80,40,30])["value"],-1.0)

    def test_permutation_formula_independent(self):
        x,y = [1,2,3,4,5],[3,1,5,2,4]
        expected = 1-6*(4+1+4+4+1)/(5*(25-1))
        self.close(s.spearman(x,y)["value"],expected)

    def test_tied_ranks_independent_covariance(self):
        # Expected ranks (4,1.5,1.5,3),(1,2,3,4), centered at 2.5.
        # Covariance sum -1.5, squared sums 4.5 and 5 => -1/sqrt(10).
        self.close(s.spearman([3,1,1,2],[1,2,3,4])["value"],-1/math.sqrt(10))

    def test_constant_undefined_and_delta_propagation(self):
        undefined = s.spearman([1,1,1],[2,3,4])
        self.assertIsNone(undefined["value"])
        self.assertEqual(undefined["reason"],"CONSTANT_X")
        self.assertIsNone(s.delta_rho(undefined,dict(value=0.7,reason=None))["value"])
        self.assertEqual(s.spearman([1,1],[2,2])["reason"],"CONSTANT_X_AND_Y")

    def test_invalid_statistical_values_rejected(self):
        for invalid in (True,float("nan"),float("inf")):
            with self.assertRaises((TypeError,ValueError)):
                s.average_ranks([1,invalid])
        with self.assertRaises(ValueError):
            s.spearman([1],[1])
        with self.assertRaises(ValueError):
            s.spearman([1,2],[1,2,3])

    def test_quantile_known_interpolation_bounds(self):
        self.assertEqual(s.quantile([0,10,20,30],0.25),7.5)
        self.assertEqual(s.quantile([30,0,20,10],0.75),22.5)
        self.assertEqual(s.quantile([4],0.025),4.0)
        self.assertEqual(s.quantile([0,30],0.975),29.25)
        self.assertEqual(s.quantile([0,30],0),0.0)
        self.assertEqual(s.quantile([0,30],1),30.0)
        for q in (-0.1,1.1,True):
            with self.assertRaises((TypeError,ValueError)):
                s.quantile([0,1],q)

    def test_paired_distribution_manual(self):
        result = s.paired_summary([-2,-1,0,4])
        self.assertEqual(result["mean"],0.25)
        self.assertEqual(result["median"],-0.5)
        self.assertEqual((result["q1"],result["q3"],result["IQR"]),(-1.25,1.0,2.25))
        self.close(result["sample_sd"],math.sqrt(20.75/3))
        self.assertEqual((result["negative"],result["zero"],result["positive"]),(2,1,1))

    def test_local_generator_deterministic_matrix_global_RNG_preserved(self):
        before = random.getstate()
        first,second = s.block_indices(4,8,17),s.block_indices(4,8,17)
        self.assertEqual(first,second)
        self.assertEqual(before,random.getstate())
        self.assertTrue(all(len(row)==4 and all(0<=i<4 for i in row) for row in first))
        self.assertTrue(any(len(set(row))<4 for row in first))
        self.assertEqual(s.identity(first),s.identity(second))

    def test_joint_resampling_recomputed_ranks_and_duplicate_blocks(self):
        matrix = ((0,0,1,3),(3,2,1,0),(0,0,0,0))
        rows,intervals = s.coupled_bootstrap(small_blocks(),matrix,small_specs())
        COUNTERS["synthetic_bootstrap_rows"] += len(rows)
        # Ranks of sampled x: (1.5,1.5,3,4); y: (3.5,3.5,1,2).
        self.close(rows[0]["estimates"]["rho_value"]["value"],-7/9)
        self.assertEqual(rows[0]["indices"],[0,0,1,3])
        self.assertEqual(rows[0]["distinct_seeds"],3)
        self.assertIsNone(rows[2]["estimates"]["rho_value"]["value"])
        self.assertIsNone(rows[2]["estimates"]["Delta"]["value"])
        self.assertEqual(rows[0]["estimates"]["mean_A"]["value"],-0.5)
        self.assertEqual(rows[0]["estimates"]["mean_A_300"]["value"],-0.5)
        self.assertEqual(intervals["mean_A"]["lower"],-0.5)
        self.assertEqual(intervals["mean_A"]["upper"],-0.5)
        self.assertEqual(intervals["rho_value"]["undefined"],1)
        EVIDENCE["coupled_bootstrap"] = dict(recomputed_ranks=True,
            repeated_indices_retained=True,paired_stages_preserved=True,undefined_reason_retained=True)

    def test_defined_threshold_1799_1800(self):
        known = dict(value=0.5,reason=None)
        missing = dict(value=None,reason="CONSTANT_SAMPLE")
        rejected = s.bootstrap_interval([known]*1799+[missing]*201,B=2000)
        accepted = s.bootstrap_interval([known]*1800+[missing]*200,B=2000)
        self.assertEqual(rejected["status"],"NON_IDENTIFIABLE_BOOTSTRAP")
        self.assertIsNone(rejected["lower"])
        self.assertEqual((accepted["lower"],accepted["upper"]),(0.5,0.5))
        self.assertEqual(accepted["undefined_reasons"],{"CONSTANT_SAMPLE":200})

    def test_interval_inference_signed_rho_caution(self):
        positive = dict(lower=0.1,upper=0.4)
        result = s.inference_delta(dict(value=0.2,reason=None),positive,
                                  dict(value=-0.8,reason=None),dict(value=-0.6,reason=None))
        self.assertEqual(result["status"],"SUPPORT_EXPLORATOIRE_DIRECTION_ATTENDUE")
        self.assertFalse(result["positive_error_associations"])
        self.assertEqual(s.inference_contrast(dict(lower=-0.2,upper=-0.1)),
                         "FAVORABLE_M2_EXPLORATOIRE")
        self.assertEqual(s.inference_contrast(dict(lower=-0.2,upper=0.0)),"INCONCLUSIVE")

    def test_csv_json_round_trip_and_missing_bootstrap_cells(self):
        rows,intervals = s.coupled_bootstrap(small_blocks(),((0,0,0,0),(0,1,2,3)),small_specs())
        COUNTERS["synthetic_bootstrap_rows"] += len(rows)
        headers,cooked = s.bootstrap_csv_rows(rows,small_specs())
        path = self.directory/"bootstrap.csv"
        s.write_csv(path,headers,cooked)
        read = s.read_csv(path,headers)
        self.assertEqual(json.loads(read[0]["seed_indices_json"]),[0,0,0,0])
        self.assertEqual(read[0]["rho_value"],"")
        self.assertIn("CONSTANT",read[0]["rho_value__status_reason"])
        self.assertEqual(float(read[1]["mean_A"]),-0.5)
        duplicated = self.directory/"duplicate.json"
        duplicated.write_text('{"x":1,"x":2}',encoding="utf-8")
        with self.assertRaises(s.DataIntegrityError):
            s.read_json(duplicated)
        invalid = self.directory/"nonfinite.json"
        for text in ('{"x":NaN}', '{"x":1e999}'):
            invalid.write_text(text,encoding="utf-8")
            with self.assertRaises(s.DataIntegrityError):
                s.read_json(invalid)

    def test_valid_table_coherence_and_seed_blocks(self):
        recipe,refs,metrics,decisions = fixture_tables()
        originals = copy.deepcopy((metrics,decisions))
        blocks,coherence = s.tables_to_blocks(metrics,decisions,recipe,refs)
        self.assertEqual(len(blocks),2)
        self.assertEqual(len(blocks[0]["models"]),6)
        self.assertEqual((metrics,decisions),originals)
        self.assertTrue(coherence["metric_keys_unique"])
        self.assertEqual(coherence["seed_blocks"],2)
        self.assertEqual(len(s.seed_rows(blocks,recipe)),12)

    def test_duplicate_missing_nonfinite_and_incoherent_tables(self):
        recipe,refs,metrics,decisions = fixture_tables()
        cases = []
        cases.append((metrics+[metrics[0]],decisions))
        cases.append((metrics[:-1],decisions))
        cases.append((metrics,decisions+[decisions[0]]))
        bad = copy.deepcopy(metrics);bad[0]["rmse_value"]="nan";cases.append((bad,decisions))
        bad = copy.deepcopy(metrics);bad[0]["mean_D_interior"]="0.9";cases.append((bad,decisions))
        bad = copy.deepcopy(decisions);bad[0]["D"]="0.9";cases.append((metrics,bad))
        bad = copy.deepcopy(decisions);bad[0]["lambda_value"]="0.48619";cases.append((metrics,bad))
        for m,d in cases:
            with self.assertRaises(s.DataIntegrityError):
                s.tables_to_blocks(m,d,recipe,refs)

    def test_saved_record_and_byte_provenance_mismatch_rejected(self):
        body = dict(scope="TEST_FIXTURE",values=[1.0,2.0])
        record = dict(body,record_sha256=s.identity(body))
        self.assertEqual(s.checked_record(record),body)
        corrupt = copy.deepcopy(record);corrupt["values"][0]=3.0
        with self.assertRaises(s.DataIntegrityError):
            s.checked_record(corrupt)
        path = self.directory/"data.json"
        path.write_text(json.dumps(body),encoding="utf-8")
        self.assertEqual(s.read_json(path,s.file_hash(path)),body)
        with self.assertRaises(s.DataIntegrityError):
            s.read_json(path,"A"*64)
