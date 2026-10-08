"""Phase3C: saved-data statistics only. No model, oracle or training imports."""
from __future__ import annotations

from collections import Counter
import csv
import hashlib
import json
import math
from numbers import Real
from pathlib import Path
import random
import sys

COHERENCE_ABS = 1e-12
COHERENCE_REL = 1e-10
STAT_TEST_ABS = 2e-14
STAT_TEST_REL = 2e-14
METRIC_HEADERS = (
    "seed","condition","step","status","missing_reason","parameter_id",
    "rmse_value","rmse_derivative","rmse_value_normalized","rmse_derivative_normalized",
    "max_abs_value_error_discrete","max_abs_derivative_error_discrete",
    "mean_D_interior","rounded_saturation_fraction","near_saturation_fraction",
    "boundary_abs_value_error_a","boundary_abs_value_error_b",
    "boundary_abs_derivative_error_a","boundary_abs_derivative_error_b",
    "training_loss","n_value_labels","n_derivative_labels","C_label_q1_1")
DECISION_HEADERS = (
    "seed","condition","step","task","task_class","lambda_value","status","missing_reason",
    "p_hat","p_ref","physical_branch","candidate_origin","predicted_objective",
    "physical_objective","reference_objective","D","physical_gradient","neural_gradient",
    "physical_kkt","neural_kkt","alpha_grid_float","beta_ref_conditional",
    "branch_mismatch","D_negative","numerical_status")
SEED_FIELDS = (
    "seed","condition","step","panel_role","rmse_value","rmse_derivative","S",
    "D_interior_quarter","D_interior_half","D_interior_three_quarters",
    "principal_rank_value","principal_rank_sensitivity","principal_rank_S",
    "d_same_sites_S","d_equal_label_budget_S")


class DataIntegrityError(ValueError):
    """Unexpected source/status/schema/provenance: block inference."""


def finite(value, name="value"):
    if isinstance(value,bool) or not isinstance(value,Real):
        raise TypeError(f"{name}: finite real, not bool, required")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name}: non-finite")
    return value


def integer(value, name="integer", minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name}: integer >= {minimum} required")
    return value


def canonical(value):
    return json.dumps(value,sort_keys=True,separators=(",",":"),
                      ensure_ascii=True,allow_nan=False).encode("utf-8")


def identity(value):
    return hashlib.sha256(canonical(value)).hexdigest().upper()


def byte_hash(data):
    return hashlib.sha256(data).hexdigest().upper()


def file_hash(path):
    return byte_hash(Path(path).read_bytes())


def runtime_identity():
    return dict(python_version=sys.version.split()[0],implementation=sys.implementation.name,
                executable=sys.executable,isolated=bool(sys.flags.isolated),
                no_site=bool(sys.flags.no_site),no_bytecode=bool(sys.dont_write_bytecode))


def _pairs(items):
    result = {}
    for key,value in items:
        if key in result:
            raise DataIntegrityError(f"duplicate JSON key {key}")
        result[key] = value
    return result


def read_json(path, expected_hash=None):
    data = Path(path).read_bytes()
    if expected_hash is not None and byte_hash(data) != expected_hash:
        raise DataIntegrityError(f"source bytes changed: {path}")
    def nonfinite(text):
        raise DataIntegrityError(f"invalid JSON numeric constant {text}")
    def finite_float(text):
        value = float(text)
        if not math.isfinite(value):
            raise DataIntegrityError(f"non-finite JSON float {text}")
        return value
    return json.loads(data.decode("utf-8"),parse_constant=nonfinite,
                      parse_float=finite_float,object_pairs_hook=_pairs)


def checked_record(value, digest_key="record_sha256"):
    body = dict(value)
    digest = body.pop(digest_key)
    if identity(body) != digest:
        raise DataIntegrityError(f"record integrity mismatch ({digest_key})")
    return body


def average_ranks(values):
    values = tuple(finite(v,"rank value") for v in values)
    if not values:
        raise ValueError("ranks need a nonempty sample")
    order = sorted(range(len(values)),key=values.__getitem__)
    ranks = [0.0]*len(values)
    left = 0
    while left < len(order):
        right = left+1
        while right < len(order) and values[order[right]] == values[order[left]]:
            right += 1
        rank = (left+1+right)/2.0
        for i in order[left:right]:
            ranks[i] = rank
        left = right
    return tuple(ranks)


def tie_profile(values):
    values = tuple(finite(v) for v in values)
    counts = Counter(values)
    unique = sorted(counts)
    gaps = [b-a for a,b in zip(unique,unique[1:])]
    return dict(n=len(values),distinct=len(unique),
                tie_groups=sum(c>1 for c in counts.values()),
                tied_observations=sum(c for c in counts.values() if c>1),
                duplicate_observations=len(values)-len(unique),
                smallest_distinct_gap=min(gaps) if gaps else None,
                ties_rule="exact stored floating values; no rounding/isclose")


def spearman(x,y):
    x,y = tuple(finite(v,"x") for v in x),tuple(finite(v,"y") for v in y)
    if len(x) != len(y) or len(x)<2:
        raise ValueError("Spearman needs equal samples of size >=2")
    constant_x,constant_y = len(set(x))==1,len(set(y))==1
    if constant_x or constant_y:
        reason = ("CONSTANT_X_AND_Y" if constant_x and constant_y else
                  "CONSTANT_X" if constant_x else "CONSTANT_Y")
        return dict(value=None,reason=reason)
    rx,ry = average_ranks(x),average_ranks(y)
    center = (len(x)+1)/2.0
    dx,dy = tuple(v-center for v in rx),tuple(v-center for v in ry)
    sx,sy = math.fsum(v*v for v in dx),math.fsum(v*v for v in dy)
    covariance = math.fsum(a*b for a,b in zip(dx,dy))
    result = finite(covariance/math.sqrt(sx*sy),"rho")
    if abs(result)>1.0+STAT_TEST_ABS:
        raise ArithmeticError("rank correlation outside admissible numerical range")
    return dict(value=result,reason=None)


def delta_rho(value_rho,sensitivity_rho):
    if value_rho["value"] is None or sensitivity_rho["value"] is None:
        return dict(value=None,reason="UNDEFINED_COMPONENT:"+str(value_rho["reason"])+
                    ";"+str(sensitivity_rho["reason"]))
    return dict(value=finite(sensitivity_rho["value"]-value_rho["value"],"Delta"),reason=None)


def quantile(values,q):
    values = sorted(finite(v,"quantile sample") for v in values)
    q = finite(q,"quantile")
    if not values or not 0<=q<=1:
        raise ValueError("nonempty sample and q in [0,1] required")
    h = (len(values)-1)*q
    lo,hi = math.floor(h),math.ceil(h)
    if lo==hi:
        return values[lo]
    fraction = h-lo
    return finite(math.fsum(((1-fraction)*values[lo],fraction*values[hi])),"quantile")


def paired_summary(differences):
    values = tuple(finite(v,"paired difference") for v in differences)
    if len(values)<2:
        raise ValueError("paired summary requires at least two pairs")
    mean = finite(math.fsum(values)/len(values),"paired mean")
    variance = finite(math.fsum((v-mean)**2 for v in values)/(len(values)-1),"sample variance")
    q1,q3 = quantile(values,0.25),quantile(values,0.75)
    return dict(n=len(values),mean=mean,median=quantile(values,0.5),q1=q1,q3=q3,
                IQR=q3-q1,sample_sd=math.sqrt(variance),dispersion="sample sd (n-1) and IQR",
                minimum=min(values),maximum=max(values),
                negative=sum(v<0 for v in values),zero=sum(v==0 for v in values),
                positive=sum(v>0 for v in values),differences=list(values))


def block_indices(n,B,seed):
    integer(n,"block count",2);integer(B,"replications",1);integer(seed,"bootstrap seed")
    generator = random.Random(seed)
    return tuple(tuple(generator.randrange(n) for _ in range(n)) for _ in range(B))


def bootstrap_interval(estimates,*,B,min_defined_fraction=0.9):
    integer(B,"B",1)
    if len(estimates)!=B:
        raise ValueError("one estimate or undefined record required per replication")
    fraction = finite(min_defined_fraction,"minimum fraction")
    if not 0<fraction<=1:
        raise ValueError("invalid defined fraction")
    defined, reasons = [],Counter()
    for estimate in estimates:
        if estimate["value"] is None:
            if not isinstance(estimate["reason"],str) or not estimate["reason"]:
                raise ValueError("undefined estimate requires explicit reason")
            reasons[estimate["reason"]] += 1
        else:
            if estimate["reason"] is not None:
                raise ValueError("defined estimate cannot have an undefined reason")
            defined.append(finite(estimate["value"],"bootstrap estimate"))
    enough = len(defined)>=math.ceil(B*fraction)
    return dict(status=("DEFINED_ON_DEFINED_REPLICATES" if enough
                        else "NON_IDENTIFIABLE_BOOTSTRAP"),
                lower=quantile(defined,0.025) if enough else None,
                upper=quantile(defined,0.975) if enough else None,
                defined=len(defined),undefined=B-len(defined),B=B,
                defined_fraction=len(defined)/B,required_defined=math.ceil(B*fraction),
                undefined_reasons=dict(reasons),
                interpretation="percentiles conditional on defined empirical replicates")


def inference_delta(estimate,interval,rho_value,rho_sensitivity):
    if estimate["value"] is None or interval["lower"] is None:
        label = "NON_IDENTIFIABLE"
    elif interval["lower"]>0:
        label = "SUPPORT_EXPLORATOIRE_DIRECTION_ATTENDUE"
    elif interval["upper"]<0:
        label = "DIRECTION_OPPOSEE"
    else:
        label = "INCONCLUSIVE"
    return dict(status=label,point_sign=("undefined" if estimate["value"] is None
                       else "positive" if estimate["value"]>0 else "nonpositive"),
                positive_error_associations=(rho_value["value"] is not None and
                    rho_sensitivity["value"] is not None and rho_value["value"]>0 and
                    rho_sensitivity["value"]>0),
                caution="positive Delta alone does not make two negative associations useful")


def inference_contrast(interval):
    if interval["lower"] is None:
        return "NON_IDENTIFIABLE"
    if interval["upper"]<0:
        return "FAVORABLE_M2_EXPLORATOIRE"
    if interval["lower"]>0:
        return "FAVORABLE_BASELINE_EXPLORATOIRE"
    return "INCONCLUSIVE"


def recipe_from_plan(plan):
    rep,stats = plan["replication"],plan["future_statistics"]
    if (rep["seeds"]!=list(range(20262001,20262021)) or
        [c["id"] for c in rep["conditions"]]!=["M1_16","M2_16","M1_32"] or
        rep["snapshots"]!=[300,3000] or rep["primary_snapshot"]!=3000 or
        plan["common_diagnostics"]["sites"]!=257):
        raise DataIntegrityError("unexpected replication panels/sites")
    bootstrap = stats["bootstrap"]
    wanted = dict(replications=2000,seed=20262000,min_defined_fraction=0.9,
                  percentile_rule="linear interpolation at (n-1)*q on sorted defined replicates (type 7)")
    for key,value in wanted.items():
        if bootstrap[key]!=value:
            raise DataIntegrityError(f"changed bootstrap rule {key}")
    if (stats["comparisons"]["A"]!="per-seed S_M2_16-S_M1_16" or
        stats["comparisons"]["B"]!="per-seed S_M2_16-S_M1_32"):
        raise DataIntegrityError("unexpected central contrasts")
    tasks = plan["decisions"]["tasks"]
    primary = plan["decisions"]["primary_tasks"]
    secondary = plan["decisions"]["secondary_tasks"]
    if (primary!=["interior_quarter","interior_half","interior_three_quarters"] or
        secondary!=["B_half","B_transition","A_transition","A_three_halves"] or
        len(tasks)!=7 or len({t["id"] for t in tasks})!=7):
        raise DataIntegrityError("unexpected task partition")
    return dict(scope="REAL_SAVED_CAMPAIGN",seeds=rep["seeds"],
                conditions=[c["id"] for c in rep["conditions"]],steps=rep["snapshots"],
                primary_condition="M1_16",primary_step=3000,primary_tasks=primary,
                secondary_tasks=secondary,tasks=tasks,domain=plan["physical_model"]["domain"],
                bootstrap=dict(B=2000,seed=20262000,min_defined_fraction=0.9),
                contrasts=[dict(id="same_sites",baseline="M1_16"),
                           dict(id="equal_label_budget_q1_1",baseline="M1_32")],
                paired_RMSE_contrasts="not explicitly prescribed in 2C; not added",
                interpretation_rule=dict(origin="fixed interpretation rule recorded before the saved analysis",
                    reason="2C fixes positive Delta hypothesis but no interval decision rule",
                    positive="support if Delta interval strictly positive",
                    zero="inconclusive if interval includes zero",
                    negative="opposite direction if interval strictly negative",
                    undefined="non identifiable"))


def read_csv(path,headers,expected_hash=None):
    data = Path(path).read_bytes()
    if expected_hash is not None and byte_hash(data)!=expected_hash:
        raise DataIntegrityError(f"CSV source bytes changed: {path}")
    import io
    reader = csv.DictReader(io.StringIO(data.decode("utf-8"),newline=""))
    if tuple(reader.fieldnames or ())!=tuple(headers):
        raise DataIntegrityError("unexpected or duplicate CSV headers")
    rows = list(reader)
    if any(None in row or None in row.values() for row in rows):
        raise DataIntegrityError("malformed CSV row")
    return rows


def _csv_int(text,name):
    if not isinstance(text,str) or not text.isdecimal():
        raise DataIntegrityError(f"invalid integer lexeme {name}")
    return int(text)


def _csv_float(text,name):
    try:
        return finite(float(text),name)
    except (TypeError,ValueError,OverflowError) as exc:
        raise DataIntegrityError(f"invalid finite numeric lexeme {name}: {text}") from exc


def _csv_bool(text,name):
    if text not in ("True","False"):
        raise DataIntegrityError(f"invalid boolean lexeme {name}")
    return text=="True"


def _close(a,b):
    return math.isclose(a,b,abs_tol=COHERENCE_ABS,rel_tol=COHERENCE_REL)


def model_key(condition,step):
    return condition+"@"+str(step)


def tables_to_blocks(metric_rows,decision_rows,recipe,references):
    expected = {(s,c,k) for s in recipe["seeds"] for c in recipe["conditions"]
                for k in recipe["steps"]}
    metrics,decisions = {},{}
    for source in metric_rows:
        row = dict(source)
        key = (_csv_int(row["seed"],"seed"),row["condition"],_csv_int(row["step"],"step"))
        if key in metrics:
            raise DataIntegrityError("duplicate metric key")
        if row["status"]!="AVAILABLE" or row["missing_reason"]:
            raise DataIntegrityError("unexpected missing metric")
        for field in METRIC_HEADERS[6:]:
            row[field] = (_csv_int(row[field],field) if field in
                          ("n_value_labels","n_derivative_labels","C_label_q1_1")
                          else _csv_float(row[field],field))
            if row[field]<0 and field!="mean_D_interior":
                raise DataIntegrityError(f"invalid negative metric {field}")
        row["lexemes"] = dict(source)
        metrics[key] = row
    if set(metrics)!=expected:
        raise DataIntegrityError("incomplete or unexpected metric panel")
    task_map = {t["id"]:t for t in recipe["tasks"]}
    expected_decisions = {key+(task,) for key in expected for task in task_map}
    max_D_gap,max_S_gap = 0.0,0.0
    for source in decision_rows:
        row = dict(source)
        key = (_csv_int(row["seed"],"seed"),row["condition"],_csv_int(row["step"],"step"),row["task"])
        if key in decisions:
            raise DataIntegrityError("duplicate decision key")
        if key not in expected_decisions:
            raise DataIntegrityError("unexpected decision identity")
        if row["status"]!="AVAILABLE" or row["missing_reason"]:
            raise DataIntegrityError("unexpected missing decision")
        for field in ("lambda_value","p_hat","p_ref","predicted_objective","physical_objective",
                      "reference_objective","D","physical_gradient","neural_gradient",
                      "physical_kkt","neural_kkt","alpha_grid_float","beta_ref_conditional"):
            row[field] = _csv_float(row[field],field)
        for field in ("branch_mismatch","D_negative"):
            row[field] = _csv_bool(row[field],field)
        task,ref = task_map[row["task"]],references[row["task"]]
        wanted_class = ("primary_interior" if row["task"] in recipe["primary_tasks"]
                        else "secondary_boundary_or_transition")
        if (row["lambda_value"]!=task["lambda_value"] or row["task_class"]!=wanted_class or
            row["p_ref"]!=ref["p_ref"] or row["reference_objective"]!=ref["j_ref"] or
            row["physical_branch"]!=ref["branch"] or
            row["beta_ref_conditional"]!=ref["reference_value_error_estimate_conditional"]):
            raise DataIntegrityError("decision/task/reference provenance mismatch")
        a,b = recipe["domain"]
        if not (a<=row["p_hat"]<=b and a<=row["p_ref"]<=b):
            raise DataIntegrityError("decision outside physical domain")
        if any(row[f]<0 for f in ("physical_kkt","neural_kkt","alpha_grid_float","beta_ref_conditional")):
            raise DataIntegrityError("negative residual or error estimate")
        if row["numerical_status"]!="FLOATING_D_NOT_CERTIFIED_REGRET":
            raise DataIntegrityError("changed numerical status")
        computed = row["physical_objective"]-row["reference_objective"]
        gap = abs(computed-row["D"])
        max_D_gap = max(max_D_gap,gap)
        if not _close(computed,row["D"]) or row["D_negative"]!=(row["D"]<0):
            raise DataIntegrityError("raw D arithmetic/sign inconsistent")
        row["lexemes"] = dict(source)
        decisions[key] = row
    if set(decisions)!=expected_decisions:
        raise DataIntegrityError("incomplete decision panel")
    blocks = []
    for seed in recipe["seeds"]:
        models = {}
        for condition in recipe["conditions"]:
            for step in recipe["steps"]:
                key = (seed,condition,step)
                metric = metrics[key]
                by_task = {task:decisions[key+(task,)] for task in task_map}
                score = finite(math.fsum(by_task[t]["D"] for t in recipe["primary_tasks"])/3,"S")
                max_S_gap = max(max_S_gap,abs(score-metric["mean_D_interior"]))
                if not _close(score,metric["mean_D_interior"]):
                    raise DataIntegrityError("archived mean_D_interior inconsistent")
                models[model_key(condition,step)] = dict(condition=condition,step=step,
                    rmse_value=metric["rmse_value"],rmse_derivative=metric["rmse_derivative"],
                    S=score,metric=metric,decisions=by_task)
        blocks.append(dict(seed=seed,models=models))
    return tuple(blocks),dict(metric_rows=len(metrics),decision_rows=len(decisions),
        seed_blocks=len(blocks),max_abs_D_identity_gap=max_D_gap,max_abs_S_identity_gap=max_S_gap,
        abs_tolerance=COHERENCE_ABS,rel_tolerance=COHERENCE_REL,
        tolerance_scope="arithmetic consistency only, not physical or total floating error",
        negative_D_preserved=sum(d["D"]<0 for d in decisions.values()),
        metric_keys_unique=True,decision_keys_unique=True)


def statistic_specs(recipe):
    specs = []
    for step in recipe["steps"]:
        for condition in recipe["conditions"]:
            panel = model_key(condition,step)
            role = ("primary" if condition==recipe["primary_condition"] and
                    step==recipe["primary_step"] else "secondary")
            for name in ("rho_value","rho_sensitivity","Delta_rho"):
                specs.append(dict(id=name+"__"+panel,kind=name,condition=condition,
                                  step=step,role=role))
        for contrast in recipe["contrasts"]:
            specs.append(dict(id="mean_d_"+contrast["id"]+"__"+str(step),
                kind="paired_mean_S",baseline=contrast["baseline"],condition="M2_16",
                step=step,role=("central_comparison" if step==recipe["primary_step"] else "secondary")))
    return specs


def sample_statistics(blocks,specs):
    result,panel_cache = {},{}
    for spec in specs:
        step,condition = spec["step"],spec["condition"]
        panel = model_key(condition,step)
        if spec["kind"]=="paired_mean_S":
            differences = [b["models"][panel]["S"]-b["models"][model_key(spec["baseline"],step)]["S"]
                           for b in blocks]
            result[spec["id"]] = dict(value=finite(math.fsum(differences)/len(differences),"mean d"),
                                      reason=None)
            continue
        if panel not in panel_cache:
            models = [b["models"][panel] for b in blocks]
            score = [m["S"] for m in models]
            rv = spearman([m["rmse_value"] for m in models],score)
            rs = spearman([m["rmse_derivative"] for m in models],score)
            panel_cache[panel] = dict(rho_value=rv,rho_sensitivity=rs,Delta_rho=delta_rho(rv,rs))
        result[spec["id"]] = panel_cache[panel][spec["kind"]]
    return result


def coupled_bootstrap(blocks,matrix,specs,*,min_defined_fraction=0.9):
    if len({b["seed"] for b in blocks})!=len(blocks):
        raise DataIntegrityError("duplicate seed blocks")
    rows,columns = [],{s["id"]:[] for s in specs}
    for replicate,indices in enumerate(matrix,1):
        if len(indices)!=len(blocks) or any(type(i) is not int or not 0<=i<len(blocks) for i in indices):
            raise ValueError("invalid block index row")
        # The same objects carry every condition, stage and task; duplicates retained.
        sampled = [blocks[i] for i in indices]
        estimates = sample_statistics(sampled,specs)
        rows.append(dict(replicate=replicate,indices=list(indices),
                         distinct_seeds=len(set(indices)),estimates=estimates))
        for name,value in estimates.items():
            columns[name].append(value)
    intervals = {name:bootstrap_interval(values,B=len(matrix),
                     min_defined_fraction=min_defined_fraction) for name,values in columns.items()}
    return rows,intervals


def seed_rows(blocks,recipe):
    principal_key = model_key(recipe["primary_condition"],recipe["primary_step"])
    rank0 = average_ranks([b["models"][principal_key]["rmse_value"] for b in blocks])
    rank1 = average_ranks([b["models"][principal_key]["rmse_derivative"] for b in blocks])
    rankS = average_ranks([b["models"][principal_key]["S"] for b in blocks])
    rows = []
    for i,block in enumerate(blocks):
        for step in recipe["steps"]:
            for condition in recipe["conditions"]:
                model = block["models"][model_key(condition,step)]
                principal = model_key(condition,step)==principal_key
                row = dict(seed=block["seed"],condition=condition,step=step,
                    panel_role="primary" if principal else "secondary",
                    rmse_value=model["rmse_value"],rmse_derivative=model["rmse_derivative"],S=model["S"],
                    principal_rank_value=rank0[i] if principal else None,
                    principal_rank_sensitivity=rank1[i] if principal else None,
                    principal_rank_S=rankS[i] if principal else None)
                row.update({"D_"+task:model["decisions"][task]["D"] for task in recipe["primary_tasks"]})
                for contrast in recipe["contrasts"]:
                    key = "d_same_sites_S" if contrast["id"]=="same_sites" else "d_equal_label_budget_S"
                    row[key] = (model["S"]-block["models"][model_key(contrast["baseline"],step)]["S"]
                                if condition=="M2_16" else None)
                rows.append(row)
    return rows


def boundary_diagnostics(blocks,recipe):
    result = []
    for step in recipe["steps"]:
        for condition in recipe["conditions"]:
            for task in recipe["secondary_tasks"]:
                saved = [b["models"][model_key(condition,step)]["decisions"][task] for b in blocks]
                values = [d["D"] for d in saved]
                result.append(dict(condition=condition,step=step,task=task,n=len(saved),
                    mean_D=math.fsum(values)/len(values),minimum_D=min(values),maximum_D=max(values),
                    negative_D=sum(v<0 for v in values),zero_D=sum(v==0 for v in values),
                    positive_D=sum(v>0 for v in values),
                    branch_mismatches=sum(d["branch_mismatch"] for d in saved),
                    mean_physical_kkt=math.fsum(d["physical_kkt"] for d in saved)/len(saved),
                    mean_neural_kkt=math.fsum(d["neural_kkt"] for d in saved)/len(saved)))
    return result


def load_saved_campaign(root,input_hashes,recipe,plan):
    """Load explicit public scientific projections; verify every supplied byte identity."""
    root=Path(root)
    def read(relative):
        if relative not in input_hashes:
            raise DataIntegrityError("missing public input identity "+relative)
        return read_json(root/relative,input_hashes[relative])
    manifest=read("results/replication/campaign.public.json")
    data=read("results/models/campaign_models.public.json")
    if (manifest["status"]!="CAMPAIGN_COMPLETED" or manifest["recipe"]!=plan
        or data["export_kind"]!="scientific_projection_not_historical_checkpoint"
        or len(data["jobs"])!=60):
        raise DataIntegrityError("public campaign completion/recipe differs")
    references=manifest["references"]
    required=("results/models/campaign_models.public.json","results/replication/phase3b_metrics.csv",
              "results/replication/phase3b_decisions.csv")
    if set(manifest["public_output_sha256"])!=set(required):
        raise DataIntegrityError("public output identity set differs")
    for relative in required:
        if manifest["public_output_sha256"][relative]!=input_hashes.get(relative):
            raise DataIntegrityError("public output bytes differ from the campaign export identity")
    metrics=read_csv(root/"results/replication/phase3b_metrics.csv",METRIC_HEADERS,
                     input_hashes["results/replication/phase3b_metrics.csv"])
    decisions=read_csv(root/"results/replication/phase3b_decisions.csv",DECISION_HEADERS,
                       input_hashes["results/replication/phase3b_decisions.csv"])
    blocks,coherence=tables_to_blocks(metrics,decisions,recipe,references)
    by_seed={block["seed"]:block for block in blocks}
    jobs={(job["seed"],job["condition"]):job for job in data["jobs"]}
    expected={(seed,condition) for seed in recipe["seeds"] for condition in recipe["conditions"]}
    if len(jobs)!=60 or set(jobs)!=expected:
        raise DataIntegrityError("public jobs missing/duplicated")
    snapshots=0
    for (seed,condition),job in jobs.items():
        if job["status"]!="COMPLETED" or job["logical_updates"]!=3000:
            raise DataIntegrityError("incomplete public job")
        models={model["step"]:model for model in job["models"]}
        if len(job["models"])!=2 or set(models)!=set(recipe["steps"]):
            raise DataIntegrityError("public snapshots missing/duplicated")
        for step,model in models.items():
            state=model["state"]
            if state["updates"]!=step:
                raise DataIntegrityError("public state/snapshot count differs")
            if state["adam"]!={key:plan["optimizer"][key] for key in ("eta","beta1","beta2","epsilon")}:
                raise DataIntegrityError("public Adam configuration differs")
            for field in ("parameters","first_moment","second_moment"):
                if len(state[field])!=49:
                    raise DataIntegrityError("public state dimensions differ")
                for value in state[field]:
                    finite(value,"public state component")
            if any(value<0 for value in state["second_moment"]):
                raise DataIntegrityError("negative public second moment")
            archived=by_seed[seed]["models"][model_key(condition,step)]
            if identity(state["parameters"])!=archived["metric"]["parameter_id"]:
                raise DataIntegrityError("public model/CSV parameter identity differs")
            for field,value in model["metrics"].items():
                if field in METRIC_HEADERS and field not in ("missing_reason","status") and archived["metric"][field]!=value:
                    raise DataIntegrityError("public metric/CSV differs "+field)
            if len(model["decisions"])!=7 or len({d["task"] for d in model["decisions"]})!=7:
                raise DataIntegrityError("public task records missing/duplicated")
            for saved in model["decisions"]:
                row=archived["decisions"][saved["task"]]
                for field,value in saved.items():
                    if field in DECISION_HEADERS and field!="missing_reason" and row[field]!=value:
                        raise DataIntegrityError("public decision/CSV differs "+field)
            snapshots+=1
    for seed in recipe["seeds"]:
        if len({jobs[seed,condition]["initial_parameter_id"] for condition in recipe["conditions"]})!=1:
            raise DataIntegrityError("initialization no longer paired")
    coherence.update(jobs_verified=60,snapshots_verified=snapshots,source_modules_imported=False,
                     physical_evaluations=0,neural_evaluations=0,
                     source_identity_basis="public projection byte identities and per-model CSV consistency")
    return blocks,coherence,manifest


def run_saved_analysis(blocks,recipe):
    specs = statistic_specs(recipe)
    estimates = sample_statistics(blocks,specs)
    cfg = recipe["bootstrap"]
    matrix = block_indices(len(blocks),cfg["B"],cfg["seed"])
    rows,intervals = coupled_bootstrap(blocks,matrix,specs,
                                      min_defined_fraction=cfg["min_defined_fraction"])
    panels,contrasts = [],[]
    for step in recipe["steps"]:
        for condition in recipe["conditions"]:
            key = model_key(condition,step)
            ids = {k:k+"__"+key for k in ("rho_value","rho_sensitivity","Delta_rho")}
            models = [b["models"][key] for b in blocks]
            record = dict(condition=condition,step=step,n=len(blocks),
                role="primary" if key==model_key(recipe["primary_condition"],recipe["primary_step"])
                               else "secondary",
                tie_profiles={field:tie_profile([m[field] for m in models])
                              for field in ("rmse_value","rmse_derivative","S")},
                statistics={k:dict(estimate=estimates[v],interval=intervals[v]) for k,v in ids.items()})
            record["inference"] = inference_delta(estimates[ids["Delta_rho"]],
                intervals[ids["Delta_rho"]],estimates[ids["rho_value"]],estimates[ids["rho_sensitivity"]])
            panels.append(record)
        for contrast in recipe["contrasts"]:
            name = "mean_d_"+contrast["id"]+"__"+str(step)
            differences = [b["models"][model_key("M2_16",step)]["S"]-
                           b["models"][model_key(contrast["baseline"],step)]["S"] for b in blocks]
            contrasts.append(dict(id=contrast["id"],step=step,metric="S",units="raw physical objective difference",
                role="central_comparison" if step==recipe["primary_step"] else "secondary",
                seed_order=[b["seed"] for b in blocks],summary=paired_summary(differences),
                interval=intervals[name],inference=inference_contrast(intervals[name]),
                pairing="M2_16 minus "+contrast["baseline"]))
    return (dict(panels=panels,contrasts=contrasts,boundary_diagnostics=boundary_diagnostics(blocks,recipe),
                statistic_specs=specs,bootstrap=dict(B=cfg["B"],seed=cfg["seed"],n=len(blocks),
                    seed_order=[b["seed"] for b in blocks],matrix_sha256=identity(matrix),
                    matrix_encoding="compact ASCII JSON array of 2000 lists of 20 zero-based indices",
                    generation="random.Random(20262000); randrange(20) in row-major order",
                    distinct_seed_count_min=min(r["distinct_seeds"] for r in rows),
                    distinct_seed_count_max=max(r["distinct_seeds"] for r in rows),
                    shared_matrix_all_statistics=True,duplicate_draws_retained=True)),
           rows,seed_rows(blocks,recipe))


def write_csv(path,headers,rows):
    path = Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("w",encoding="utf-8",newline="") as handle:
        writer = csv.DictWriter(handle,fieldnames=headers,extrasaction="raise")
        writer.writeheader()
        for row in rows:
            cooked = {}
            for field in headers:
                value = row.get(field)
                cooked[field] = "" if value is None else repr(finite(value)) if isinstance(value,float) else value
            writer.writerow(cooked)


def bootstrap_csv_rows(rows,specs):
    headers = ["replicate","seed_indices_json","distinct_seeds"]
    for spec in specs:
        headers.extend((spec["id"],spec["id"]+"__status_reason"))
    cooked = []
    for row in rows:
        result = dict(replicate=row["replicate"],
            seed_indices_json=json.dumps(row["indices"],separators=(",",":"),allow_nan=False),
            distinct_seeds=row["distinct_seeds"])
        for name,estimate in row["estimates"].items():
            result[name] = estimate["value"]
            result[name+"__status_reason"] = "DEFINED" if estimate["value"] is not None else estimate["reason"]
        cooked.append(result)
    return headers,cooked
