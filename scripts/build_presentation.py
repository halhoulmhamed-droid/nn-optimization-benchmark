"""Document-only inspection or explicit future pdfLaTeX build in a new directory."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[1]
FIGURES=("learned_curves","error_decision","paired_scores","paired_effects")
INPUTS=("report/main.tex","report/references.tex","slides/main.tex",*("report/figures/"+n+".tex" for n in FIGURES),
        *("report/data/"+n+".csv" for n in ("learned_curves","primary_panel","paired_scores","paired_effects")))
def main():
    parser=argparse.ArgumentParser(description=__doc__)
    modes=parser.add_mutually_exclusive_group()
    modes.add_argument("--check",action="store_true")
    modes.add_argument("--build",action="store_true")
    parser.add_argument("--output")
    args=parser.parse_args()
    if not (args.check or args.build):
        parser.print_help()
        return 0
    sys.path.insert(0,str(ROOT))
    from src import public_io as contract
    seal=contract.verify_archive()
    if args.check:
        print(json.dumps(dict(status="DOCUMENT_BYTES_CHECKED_ONLY",PDFs=6,
                              new_network_calls=0,new_physical_calls=0,new_compilations=0,
                              visual_review="Reuse delivered review; no new local visual inspection")))
        return 0
    if not args.output:
        parser.error("--build requires a new --output directory")
    compiler=shutil.which("pdflatex")
    if compiler is None:
        raise RuntimeError("Existing compatible TeX Live/pdfLaTeX is required; nothing is installed")
    out=contract.output_path(args.output,create=True)
    for relative in INPUTS:
        target=out/relative
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(ROOT/relative,target)
    records=[]
    for relative,passes in [(f"report/figures/{n}.tex",1) for n in FIGURES]+[("report/main.tex",2),("slides/main.tex",2)]:
        source=out/relative
        for step in range(passes):
            command=[compiler,"-interaction=nonstopmode","-halt-on-error","-no-shell-escape",source.name]
            process=subprocess.run(command,cwd=source.parent,capture_output=True,text=True,encoding="utf-8",errors="replace")
            log=out/f"build_{source.parent.name}_{source.stem}_{step+1}.txt"
            log.write_text(process.stdout+process.stderr,encoding="utf-8")
            records.append(dict(source=relative,pass_number=step+1,command=command,
                                exit_code=process.returncode,log=log.name))
            contract.write_new(out/f"build_step_{len(records):02d}.json",records[-1])
            if process.returncode:
                raise RuntimeError("Document build failed; preserve logs; do not rerun scientific exports")
    contract.write_new(out/"build_summary.json",dict(status="BUILT_VISUAL_REVIEW_REQUIRED",commands=records,
                       review_required="citations/labels/logs; four figures; all report pages and slides",
                       scientific_evaluations=0))
    print("BUILT_VISUAL_REVIEW_REQUIRED",out)
    return 0
if __name__=="__main__":
    raise SystemExit(main())

