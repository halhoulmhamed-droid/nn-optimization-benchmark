"""Explicit portable campaign entry point; all numerical work stays in preserved modules."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
OUT=MANIFEST=CACHE=SUMMARY=METRICS=DECISIONS=None
CHECKS_ARGUMENT=None

def utc():
    return datetime.now(timezone.utc).isoformat()

def tasks_from_plan(plan):
    return [dict(task,task_class=("primary_interior" if task["id"] in
            plan["decisions"]["primary_tasks"] else "secondary_boundary_or_transition"))
            for task in plan["decisions"]["tasks"]]

def oracle_identity(plan,sources,r):
    return r.identity({"formula":plan["physical_model"]["formula"],
                       "domain":plan["physical_model"]["domain"],
                       "units":plan["physical_model"]["units"],
                       "oracle_source_sha256":sources["src/thermal_oracle.py"]})

def archived_labels(plan,sources,r):
    saved=r.read_json(ROOT/"results/data/pilot_labels.public.json")
    data={key:saved["labels"][key] for key in ("training","validation")}
    if r.identity(data)!=saved["labels"]["canonical_id"]:
        raise ValueError("archived scientific label payload differs")
    r._equal(saved["physical_model"]["formula"],plan["physical_model"]["formula"],"archive physical formula")
    r._equal(saved["physical_model"]["domain"],plan["physical_model"]["domain"],"archive physical domain")
    if tuple(data["training"]["points"])!=r.midpoint_sites(16):
        raise ValueError("archived training coordinates differ")
    labels={role:r.SuppliedLabels(item["points"],item["values"],item["derivatives"])
            for role,item in data.items()}
    return labels,{"source_file":"results/data/pilot_labels.public.json",
                   "source_sha256":sources["results/data/pilot_labels.public.json"],
                   "label_payload_id":saved["labels"]["canonical_id"],
                   "oracle_source_sha256":sources["src/thermal_oracle.py"]}

def references_from_archive(plan,sources,r,c):
    data=r.read_json(ROOT/"results/data/physical_references.public.json")
    r._equal(data["physical_model"],plan["physical_model"],"reference physical model")
    r._equal(data["S0"],plan["losses"]["S0"],"reference S0")
    r._equal(data["S1"],plan["losses"]["S1"],"reference S1")
    expected_tasks={task["id"]:task["lambda_value"] for task in plan["decisions"]["tasks"]}
    refs=data["references"]
    if set(refs)!=set(expected_tasks):
        raise ValueError("seven exact references required")
    for task,ref in refs.items():
        if ref["lambda_value"]!=expected_tasks[task] or ref["task"]!=task:
            raise ValueError("reference task/lambda differs")
        a,b=ref["bracket"]
        if not 0.02<=a<=ref["p_ref"]<=b<=0.20 or ref["width"]!=b-a:
            raise ValueError("reference bracket/domain differs")
        if ref["width_tolerance"]!=1e-10 or ref["max_iterations"]!=80:
            raise ValueError("reference tolerance/budget differs")
        if c.sd.kkt_residual_interval(ref["p_ref"],ref["gradient_at_reference"],0.02,0.20)!=ref["kkt_residual"]:
            raise ValueError("saved reference residual differs")
    return refs

def snapshot_totals(statuses):
    totals={}
    for status in statuses.values():
        for measure in status["measures"].values():
            for name,value in measure["counters"].items():
                totals[name]=totals.get(name,0)+value
    return totals

def run_campaign(checks, plan, sources, backup, r, c, *, resume):
    existing_paths = (MANIFEST, CACHE, SUMMARY, METRICS, DECISIONS, OUT/"phase3b_runs")
    if not resume and any(p.exists() for p in existing_paths):
        raise c.IntegrityError("new campaign refuses existing phase3B destination; use --resume")
    identity_sources = dict(sources)
    identity_sources["results/data/pilot_labels.public.json"] = sources["results/data/pilot_labels.public.json"]
    identity_sources["results/data/physical_references.public.json"] = sources["results/data/physical_references.public.json"]
    oid = oracle_identity(plan, sources, r)
    campaign_id = r.identity(dict(plan_sha256=sources["configs/replication_plan.json"],
                                  protocol_signature=r.protocol_signature(plan),
                                  sources=identity_sources, environment=r.runtime_identity(),
                                  oracle_id=oid))
    jobs = r.build_jobs(plan)
    tasks = tasks_from_plan(plan)
    references = references_from_archive(plan, sources, r, c)
    if resume:
        manifest = c.load_record(MANIFEST, campaign_id)
        r._equal(manifest["sources"], identity_sources, "resume source identities")
    else:
        manifest = dict(schema_version=1, kind="campaign_manifest", context_id=campaign_id,
            status="PREPARING", created_utc=utc(), recipe=plan,
            sources=identity_sources, environment=r.runtime_identity(),
            oracle_id=oid, references=references,
            reference_accounting=dict(archived_references_reused=7, new_M0_calls=0,
                                      legacy_solve_cost_not_current_cost=True),
            jobs=[dict(seed=j["seed"], condition=j["condition"], job_id=j["job_id"],
                       initial_parameter_id=j["initial_parameter_id"]) for j in jobs],
            dataset_signatures=None, physics_preparation_history=[])
        c.atomic_record(MANIFEST, manifest)
    started = time.perf_counter()
    import_was_new = "src.thermal_oracle" not in sys.modules
    from src import thermal_oracle as thermal
    prep = dict(invocation_utc=utc(), source_sha256=sources["src/thermal_oracle.py"],
                value=2 if import_was_new else 0, derivative=2 if import_was_new else 0,
                curvature=2 if import_was_new else 0,
                directly_instrumented=False, basis="source constants and new module import")
    manifest["physics_preparation_history"].append(prep)
    c.atomic_record(MANIFEST, manifest)
    r._equal(thermal.S0, plan["losses"]["S0"], "physical preparation S0")
    r._equal(thermal.S1, plan["losses"]["S1"], "physical preparation S1")
    r._equal(thermal.LAMBDA_MIN, plan["physical_model"]["lambda_domain"][0], "lambda min")
    r._equal(thermal.LAMBDA_MAX, plan["physical_model"]["lambda_domain"][1], "lambda max")
    if CACHE.exists():
        cache = c.PhysicalCache.load(CACHE, oid, value_call=thermal.temperature,
                                     derivative_call=thermal.temperature_derivative)
    else:
        cache = c.PhysicalCache(oid, value_call=thermal.temperature,
                                derivative_call=thermal.temperature_derivative)
    archives, info = archived_labels(plan, identity_sources, r)
    for labels in archives.values():
        for p, v, g in zip(labels.points, labels.values, labels.derivatives):
            cache.seed("T", p, v, origin="ARCHIVED", provenance=info)
            cache.seed("T_prime", p, g, origin="ARCHIVED", provenance=info)
    # Import endpoints are already available; retain archive provenance on collision.
    for q, pairs in (
        ("T", ((thermal.A, thermal.VALUE_A), (thermal.B, thermal.VALUE_B))),
        ("T_prime", ((thermal.A, thermal.DERIVATIVE_A), (thermal.B, thermal.DERIVATIVE_B))),
    ):
        for p, value in pairs:
            cache.seed(q, p, value, origin="IMPORT_PREPARATION", provenance={"source_sha256": sources["src/thermal_oracle.py"]})
    acquisition_start = time.perf_counter()
    training, diagnostic, boundaries = c.acquire_datasets(plan, cache)
    cache.save(CACHE)
    acquisition_seconds = time.perf_counter()-acquisition_start
    data_signatures = {
        "training": {name: labels.data_identity("M2" if name == "M2_16" else "M1")
                     for name, labels in training.items()},
        "diagnostics": diagnostic.data_identity("M2"),
        "boundaries": boundaries.data_identity("M2")}
    if manifest["dataset_signatures"] is not None:
        r._equal(manifest["dataset_signatures"], data_signatures, "resume actual datasets")
    else:
        manifest["dataset_signatures"] = data_signatures
        for record, job in zip(manifest["jobs"], jobs):
            _, context = c.make_job_context(job, training[job["condition"]], cache, identity_sources)
            record["context_signature"] = r.identity(context)
        manifest["status"] = "READY_FROZEN"
        c.atomic_record(MANIFEST, manifest)
    statuses = {}
    completed_before = 0
    progress_start = time.perf_counter()
    def progress(job, step, status):
        print(json.dumps(dict(event="milestone", seed=job["seed"], condition=job["condition"],
                              step=step, jobs_processed=len(statuses),
                              elapsed_seconds=round(time.perf_counter()-progress_start, 2))),
              flush=True)
    try:
        for index, job in enumerate(jobs, 1):
            directory = OUT/"phase3b_runs"/f"seed_{job['seed']}"/job["condition"]
            job_resume = directory.exists() and resume
            if job_resume:
                prior = c.load_record(directory/"job_status.json")
                if prior["status"] == "COMPLETED":
                    completed_before += 1
            result = c.run_job(
                job, training[job["condition"]], cache, diagnostic, boundaries,
                tasks, references, identity_sources, directory, resume=job_resume,
                progress=progress, cache_path=CACHE)
            statuses[(job["seed"], job["condition"])] = result
            cache.save(CACHE)
            print(json.dumps(dict(event="job_finished", job=index, total=60,
                                  seed=job["seed"], condition=job["condition"],
                                  status=result["status"], updates=result["logical_updates"]),
                             ensure_ascii=False), flush=True)
        metric_rows, decision_rows = c.aggregate_rows(jobs, statuses, tasks)
        c.write_csv(METRICS, c.METRIC_FIELDS, metric_rows)
        c.write_csv(DECISIONS, c.DECISION_FIELDS, decision_rows)
        failures = [dict(seed=k[0], condition=k[1], reason=v["missing_reason"])
                    for k,v in statuses.items() if v["status"] != "COMPLETED"]
        valid_metrics = sum(row["status"] == "AVAILABLE" for row in metric_rows)
        verdict = ("CAMPAIGN_COMPLETED" if not failures and valid_metrics == 120
                   else "CAMPAIGN_COMPLETED_WITH_FAILURES")
        counts = {name: sum(s["counts"][name] for s in statuses.values()) for name in (
            "update_attempts", "update_completed", "logging_attempts", "logging_completed",
            "known_recomputed_loss_gradient_calls")}
        logical = sum(s["logical_updates"] for s in statuses.values())
        timings = {name: sum(s["timings"][name] for s in statuses.values())
                   for name in ("training", "logging", "metrics", "persistence")}
        timings.update(acquisition=acquisition_seconds, total_this_invocation=time.perf_counter()-started)
        network_counts = snapshot_totals(statuses)
        network_counts["logging_activation_only_sites"] = sum(s["logging_activation_only_sites"] for s in statuses.values())
        network_counts["training_value_and_parameter_gradient_sites_from_code"] = sum(
            (s["counts"]["update_completed"]+s["counts"]["logging_completed"])*s["n_value_labels"]
            for s in statuses.values())
        network_counts["training_sensitivity_and_mixed_gradient_sites_from_code"] = sum(
            (s["counts"]["update_completed"]+s["counts"]["logging_completed"])*s["n_derivative_labels"]
            for s in statuses.values())
        uncertainty = [dict(seed=k[0], condition=k[1], windows=s["uncertain_execution_windows"])
                       for k,s in statuses.items() if s["uncertain_execution_windows"]]
        output_files = [MANIFEST, CACHE, METRICS, DECISIONS]
        output_files.extend(sorted((OUT/"phase3b_runs").rglob("*.json")))
        summary = dict(schema_version=1, kind="campaign_summary", context_id=campaign_id,
            status=verdict, next_phase=("GO_FOR_PHASE3C" if verdict == "CAMPAIGN_COMPLETED"
                                      else "HOLD_PRIMARY_STATISTICS_UNTIL_VALID_COMPLETE_OUTPUTS"),
            jobs_processed=len(statuses), jobs_completed=len(statuses)-len(failures),
            jobs_skipped_complete_on_resume=completed_before,
            logical_updates=logical, executed_loss_gradient_counts=counts,
            executed_updates_exact_if_no_uncertain_window=logical if not uncertainty else None,
            execution_uncertainty_windows=uncertainty,
            snapshots_available=sum(len(s["snapshot_steps_available"]) for s in statuses.values()),
            metrics_rows=len(metric_rows), metrics_valid=valid_metrics,
            decisions_rows=len(decision_rows), decisions_valid=sum(x["status"] == "AVAILABLE" for x in decision_rows),
            physical_cache=cache.report(), physics_preparation_history=manifest["physics_preparation_history"],
            reference_accounting=manifest["reference_accounting"],
            network_counts=network_counts, timings_observed_seconds=timings,
            descriptive_3000=c.descriptive_table(metric_rows),
            failures=failures, incidents=[],
            file_sha256={str(p.relative_to(ROOT)).replace("\\","/"):c.file_hash(p) for p in output_files},
            scientific_limits=["exploratory after pilot; seed is replication unit",
                "raw D against floating M0, not certified exact regret",
                "no Spearman, paired inference, bootstrap or hypothesis conclusion",
                "timings are this invocation/PC, not a universal speed ranking"],
            initialized_at=manifest["created_utc"], finished_utc=utc())
        c.atomic_record(SUMMARY, summary)
        return {"status": verdict, "jobs": len(statuses), "updates": logical,
                "snapshots": summary["snapshots_available"], "metrics": valid_metrics,
                "decisions": summary["decisions_valid"]}
    except BaseException as exc:
        cache.save(CACHE)
        partial = dict(schema_version=1, kind="campaign_summary", context_id=campaign_id,
            status=("CAMPAIGN_PAUSED" if isinstance(exc, KeyboardInterrupt) else "HOLD_INTEGRITY"),
            error_type=type(exc).__name__, message=str(exc), jobs_processed=len(statuses),
            physical_cache=cache.report(), physics_preparation_history=manifest["physics_preparation_history"],
            resume_command="python -I -S -B scripts/run_replication_campaign.py --run-campaign --resume --output " + str(OUT.parent.parent.relative_to(ROOT)).replace("\\","/") + " --checks " + CHECKS_ARGUMENT,
            no_algorithm_correction_or_retry_performed=True, finished_utc=utc())
        c.atomic_record(SUMMARY, partial)
        raise

def main():
    global OUT,MANIFEST,CACHE,SUMMARY,METRICS,DECISIONS,CHECKS_ARGUMENT
    parser=argparse.ArgumentParser(description=__doc__)
    mode=parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run",action="store_true")
    mode.add_argument("--run-campaign",action="store_true")
    parser.add_argument("--output")
    parser.add_argument("--checks")
    parser.add_argument("--resume",action="store_true")
    args=parser.parse_args()
    if not (args.dry_run or args.run_campaign):
        parser.print_help()
        return 0
    if not args.output or (args.resume and not args.run_campaign):
        parser.error("Use --output; --resume requires --run-campaign")
    sys.path.insert(0,str(ROOT))
    from src import public_io as contract
    contract.validate_runtime()
    contract.verify_archive()
    from src import replication_training as r
    from src import replication_campaign as c
    plan=r.validate_plan(r.read_json(ROOT/"configs/replication_plan.json"))
    destination=contract.output_path(args.output,resume=args.resume)
    jobs=r.build_jobs(plan)
    if args.dry_run:
        print(json.dumps(dict(status="PLANNED_NOT_EXECUTED",jobs=len(jobs),snapshots=120,
                         maximum_logical_updates=180000,output=args.output,
                         labels_acquired=0,training_updates=0)))
        return 0
    if not args.checks:
        parser.error("--run-campaign requires --checks from a successful public test run")
    contract.test_gate(args.checks)
    CHECKS_ARGUMENT=args.checks
    destination.mkdir(parents=True,exist_ok=args.resume)
    OUT=destination/"results/replication"
    MANIFEST=OUT/"phase3b_manifest.json"
    CACHE=OUT/"phase3b_physical_cache.json"
    SUMMARY=OUT/"phase3b_summary.json"
    METRICS=OUT/"phase3b_metrics.csv"
    DECISIONS=OUT/"phase3b_decisions.csv"
    sources=contract.source_hashes()
    sources["results/data/pilot_labels.public.json"]=contract.digest(ROOT/"results/data/pilot_labels.public.json")
    sources["results/data/physical_references.public.json"]=contract.digest(ROOT/"results/data/physical_references.public.json")
    answer=run_campaign(None,plan,sources,None,r,c,resume=args.resume)
    if answer["status"]=="CAMPAIGN_COMPLETED":
        contract.export_completed_campaign(destination)
    print(json.dumps(answer,ensure_ascii=True,allow_nan=False),flush=True)
    return 0 if answer["status"]=="CAMPAIGN_COMPLETED" else 1

if __name__=="__main__":
    raise SystemExit(main())

