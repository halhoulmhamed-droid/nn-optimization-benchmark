"""Portable data and output contracts; no scientific imports."""
from __future__ import annotations
import argparse
import csv
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
PUBLIC_SEAL = ROOT / "scientific_files.json"

def require(condition, message):
    if not condition:
        raise ValueError(message)

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest().upper()

def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
                      allow_nan=False).encode("utf-8")

def identity(value):
    return hashlib.sha256(canonical(value)).hexdigest().upper()

def pairs(items):
    result = {}
    for key, value in items:
        require(key not in result, "Duplicate JSON field: " + key)
        result[key] = value
    return result

def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"), object_pairs_hook=pairs,
                      parse_constant=lambda v: (_ for _ in ()).throw(ValueError(v)))

def write_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n")

def source_hashes():
    names = sorted(p for folder in ("src", "tests", "scripts") for p in (ROOT / folder).glob("*.py"))
    names.append(ROOT / "configs/replication_plan.json")
    return {p.relative_to(ROOT).as_posix(): digest(p) for p in names}

def verify_archive():
    seal = read(PUBLIC_SEAL)
    require(seal["schema_version"] == 1 and seal["kind"] == "public_scientific_copy_integrity",
            "Public file-identity contract differs")
    for relative, entry in seal["files"].items():
        path = ROOT / relative
        require(not Path(relative).is_absolute() and ".." not in Path(relative).parts
                and path.resolve().is_relative_to(ROOT), "Unsafe public file path")
        require(path.is_file() and path.stat().st_size == entry["bytes"]
                and digest(path) == entry["sha256"], "Public file bytes differ: " + relative)
    return seal

def output_path(relative, *, resume=False, create=False):
    path = Path(relative)
    require(not path.is_absolute() and ".." not in path.parts, "Use a relative reproduction output")
    path = ROOT / path
    scope = ROOT / "reproductions"
    require(path.resolve().is_relative_to(scope.resolve()) and path.resolve() != scope.resolve(),
            "Outputs must be in a named directory under reproductions/")
    for p in [*path.parents, path]:
        if p.exists():
            require(not p.is_symlink() and not (getattr(p.lstat(), "st_file_attributes", 0) & 1024),
                    "Refuse redirected output directory")
    if resume:
        require(path.is_dir(), "Resume output does not exist")
    else:
        require(not path.exists(), "New execution refuses an existing output directory")
    if create:
        path.mkdir(parents=True, exist_ok=resume)
    return path

def project_input(relative):
    path = Path(relative)
    require(not path.is_absolute() and ".." not in path.parts, "Use a project-relative input path")
    result = ROOT / path
    require(result.resolve().is_relative_to(ROOT), "Input leaves project")
    require(result.is_file(), "Input file missing: " + relative)
    return result

def validate_runtime():
    require(sys.implementation.name == "cpython" and sys.version_info[:3] == (3, 11, 9),
            "Reproduction recipe targets CPython 3.11.9; other runtimes are unverified")
    require(sys.flags.isolated and sys.flags.no_site and sys.dont_write_bytecode, "Use -I -S -B")

def test_gate(relative):
    proof_path = project_input(relative)
    proof = read(proof_path)
    require(proof["status"] == "PASSED" and proof["skipped"] == 0 and proof["failures"] == 0
            and proof["errors"] == 0, "Public numerical checks are not successful")
    sources = source_hashes()
    version = ".".join(map(str, sys.version_info[:3]))
    require(proof["source_sha256"] == sources, "Code/recipe changed since numerical checks")
    require(proof["python_version"] == version,
            "Numerical checks use another Python version")
    expected = sorted(path.stem for path in (ROOT / "tests").glob("test_*.py"))
    require(bool(expected) and proof.get("test_modules") == expected
            and "test_module" not in proof, "Require the complete numerical-check summary")
    require(proof.get("commands") == [dict(test_module=name, exit_code=0) for name in expected],
            "Every test-module subprocess must have succeeded")
    require(set(proof.get("ledgers", {})) == set(expected), "Test-module ledgers are incomplete")
    totals = {key: 0 for key in ("tests_run", "failures", "errors", "skipped")}
    for name in expected:
        child = read(proof_path.parent / name / "test_results.json")
        require(child.get("test_module") == name and child.get("status") == "PASSED"
                and child.get("source_sha256") == sources and child.get("python_version") == version,
                "Test-module report differs from the complete summary")
        for key in totals:
            value = child.get(key)
            require(type(value) is int and (value > 0 if key == "tests_run" else value == 0),
                    "Test-module counts must record executed, successful, unskipped tests")
            totals[key] += value
        require(child.get("ledger") == proof["ledgers"][name], "Test-module ledger differs")
    require(all(type(proof.get(key)) is int and proof[key] == value for key, value in totals.items()),
            "Complete test counts differ from individual reports")
    return proof

def read_only_summary():
    seal = verify_archive()
    analysis = read(ROOT / "results/analysis/phase3c_analysis.public.json")
    manifest = read(ROOT / "results/replication/campaign.public.json")
    models = read(ROOT / "results/models/campaign_models.public.json")
    require(manifest["status"] == "CAMPAIGN_COMPLETED" and len(models["jobs"]) == 60,
            "Saved campaign completion differs")
    require(sum(len(job["models"]) for job in models["jobs"]) == 120,
            "Saved public model count differs")
    sizes = {}
    for relative, expected in (
        ("report/data/learned_curves.csv",257), ("report/data/primary_panel.csv",20),
        ("report/data/paired_scores.csv",20), ("report/data/paired_effects.csv",2),
        ("results/replication/phase3b_metrics.csv",120),
        ("results/replication/phase3b_decisions.csv",840),
        ("results/analysis/phase3c_seed_statistics.csv",120),
        ("results/analysis/phase3c_bootstrap_samples.csv",2000)):
        with (ROOT/relative).open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        require(len(rows) == expected, "Saved CSV row count differs: " + relative)
        sizes[relative] = len(rows)
    primary = next(panel for panel in analysis["results"]["panels"] if panel["role"] == "primary")
    return {"status":"SAVED_FILES_CHECKED_WITHOUT_SCIENTIFIC_REEVALUATION",
            "files_verified":len(seal["files"]), "CSV_rows":sizes,
            "primary_Delta_rho":primary["statistics"]["Delta_rho"],
            "publication_state":"PUBLIC_SCIENTIFIC_COPY",
            "new_training_updates":0,"new_physical_calls":0,"new_network_calls":0,"new_bootstraps":0}

def checked_record(path, digest_key="record_sha256"):
    record=read(path)
    recorded=record.pop(digest_key)
    require(identity(record)==recorded,"Record payload integrity differs: "+str(path))
    return record

def project_models(run_root, manifest):
    jobs=[]
    for entry in manifest["jobs"]:
        seed,condition=entry["seed"],entry["condition"]
        directory=Path(run_root)/"results/replication/phase3b_runs"/f"seed_{seed}"/condition
        status=checked_record(directory/"job_status.json")
        final=checked_record(directory/"training_state.json","payload_sha256")
        require(status["status"]=="COMPLETED" and final["status"]=="COMPLETED"
                and status["logical_updates"]==3000 and final["state"]["updates"]==3000,
                "Incomplete model cannot enter the scientific projection")
        require(identity(final["context"])==entry["context_signature"]
                and final["context_signature"]==entry["context_signature"],
                "Job context differs from manifest")
        context=final["context"]
        model_records=[]
        for step in (300,3000):
            snapshot=checked_record(directory/f"snapshot_{step:04d}.json","payload_sha256")
            require(snapshot["context"]==context and snapshot["state"]==final["snapshots"][str(step)],
                    "Snapshot continuity differs")
            measure=status["measures"][str(step)]
            # Saved arrays/results are projected, never recomputed.
            model_records.append(dict(step=step,state=snapshot["state"],metrics=measure["metrics"],
                decisions=[{k:v for k,v in d.items() if k!="decision_label_provenance"}
                           for d in measure["decisions"]],boundaries=measure["boundaries"],
                saturation=measure["saturation"],grid_error_estimate=measure["grid_error_estimate"]))
        jobs.append(dict(seed=seed,condition=condition,method=context["method"],status=status["status"],
            logical_updates=status["logical_updates"],initial_parameter_id=context["initial_parameter_id"],
            points=context["points"],normalization=context["normalization"],
            scales={key:context["scales"][key] for key in ("S0","S1")},
            data_identity=context["data_identity"],logs=final["logs"],counts=final["counts"],
            models=model_records))
    require(len(jobs)==60 and len({(j["seed"],j["condition"]) for j in jobs})==60,"Job identities differ")
    return {"export_kind":"scientific_projection_not_historical_checkpoint",
            "jobs":jobs,"schema_version":1,
            "checkpoint_use":"Archived projections are for inspection; resume uses complete future generated checkpoints"}

def export_completed_campaign(output_root):
    output_root=Path(output_root)
    manifest=checked_record(output_root/"results/replication/phase3b_manifest.json")
    summary=checked_record(output_root/"results/replication/phase3b_summary.json")
    require(summary["status"]=="CAMPAIGN_COMPLETED" and summary["jobs_completed"]==60
            and summary["snapshots_available"]==120 and summary["metrics_valid"]==120
            and summary["decisions_valid"]==840,"Incomplete campaign: primary analysis is withheld")
    model_data=project_models(output_root,manifest)
    campaign={"schema_version":1,"export_kind":"scientific_campaign_projection",
              "status":summary["status"],"recipe":manifest["recipe"],"references":manifest["references"],
              "jobs_completed":summary["jobs_completed"],"logical_updates":summary["logical_updates"],
              "snapshots_available":summary["snapshots_available"],"metrics_valid":summary["metrics_valid"],
              "decisions_valid":summary["decisions_valid"]}
    model_relative="results/models/campaign_models.public.json"
    model_path=output_root/model_relative
    if model_path.exists():
        require(read(model_path)==model_data,"Completed model projection collision")
    else:
        write_new(model_path,model_data)
    campaign["public_output_sha256"]={name:digest(output_root/name) for name in (
        model_relative,"results/replication/phase3b_metrics.csv",
        "results/replication/phase3b_decisions.csv")}
    for relative,value in {"results/replication/campaign.public.json":campaign}.items():
        path=output_root/relative
        if path.exists():
            require(read(path)==value,"Completed public projection collision")
        else:
            write_new(path,value)
