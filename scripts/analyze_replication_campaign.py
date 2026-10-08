"""Explicit future saved-data analysis in a new output directory; no model imports."""
import argparse
from pathlib import Path
import sys
import json
ROOT=Path(__file__).resolve().parents[1]
def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analyze",action="store_true")
    parser.add_argument("--campaign-dir",default=".")
    parser.add_argument("--output")
    parser.add_argument("--checks")
    args=parser.parse_args()
    if not args.analyze:
        parser.print_help()
        return 0
    if not args.output or not args.checks:
        parser.error("--analyze requires a new --output and --checks from public numerical checks")
    sys.path.insert(0,str(ROOT))
    from src import public_io as contract
    contract.validate_runtime()
    contract.verify_archive()
    contract.test_gate(args.checks)
    campaign=Path(args.campaign_dir)
    if campaign.is_absolute() or ".." in campaign.parts:
        parser.error("--campaign-dir is project-relative")
    campaign=(ROOT/campaign).resolve()
    if campaign!=ROOT and not campaign.is_relative_to(ROOT/"reproductions"):
        parser.error("--campaign-dir must be this project or a reproduction output")
    from src import replication_statistics as s
    plan=contract.read(ROOT/"configs/replication_plan.json")
    recipe=s.recipe_from_plan(plan)
    names=("results/replication/campaign.public.json","results/models/campaign_models.public.json",
           "results/replication/phase3b_metrics.csv","results/replication/phase3b_decisions.csv")
    inputs={name:s.file_hash(campaign/name) for name in names}
    blocks,coherence,manifest=s.load_saved_campaign(campaign,inputs,recipe,plan)
    out=contract.output_path(args.output,create=True)
    # Only an explicitly requested future analysis reaches this preserved numerical call.
    results,bootstrap_rows,model_rows=s.run_saved_analysis(blocks,recipe)
    headers,cooked=s.bootstrap_csv_rows(bootstrap_rows,results["statistic_specs"])
    s.write_csv(out/"seed_statistics.csv",s.SEED_FIELDS,model_rows)
    s.write_csv(out/"bootstrap_samples.csv",headers,cooked)
    contract.write_new(out/"analysis.json",dict(status="ANALYSIS_COMPLETED",
        export_kind="new_saved_data_analysis_not_historical_archive",recipe=recipe,results=results,
        data_checks=coherence,input_sha256=inputs,code_sha256=contract.source_hashes(),
        export_sha256={"seed_statistics.csv":s.file_hash(out/"seed_statistics.csv"),
                       "bootstrap_samples.csv":s.file_hash(out/"bootstrap_samples.csv")},
        limits=contract.read(ROOT/"results/analysis/phase3c_analysis.public.json")["limits"],
        new_training_updates=0,new_physical_calls=0,new_neural_calls=0))
    print(json.dumps(dict(status="ANALYSIS_COMPLETED",n=20,B=2000,output=args.output)))
    return 0
if __name__=="__main__":
    raise SystemExit(main())

