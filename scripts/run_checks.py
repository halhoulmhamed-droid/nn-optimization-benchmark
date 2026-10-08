"""Explicit, process-isolated future numerical checks. Not run during preparation."""
import argparse
import importlib.util
import io
from pathlib import Path
import subprocess
import sys
import unittest
ROOT=Path(__file__).resolve().parents[1]

def one_module(name,relative_output,contract):
    path=ROOT/"tests"/(name+".py")
    contract.require(path.is_file() and name in {p.stem for p in (ROOT/"tests").glob("test_*.py")},
                     "Unknown copied test module")
    out=contract.output_path(relative_output,create=True)
    plan=contract.read(ROOT/"configs/replication_plan.json")
    initialization=contract.read(ROOT/"results/data/pilot_initialization.public.json")["initialization"]
    sources=contract.source_hashes()
    # Historical tests import neighboring fixture helpers by their module name.
    sys.path.insert(0,str(ROOT/"tests"))
    spec=importlib.util.spec_from_file_location("public_"+name,path)
    module=importlib.util.module_from_spec(spec)
    sys.modules[spec.name]=module
    spec.loader.exec_module(module)
    # Short fixture roots leave room for unchanged test method and temporary-file names.
    module_names=sorted(p.stem for p in (ROOT/"tests").glob("test_*.py"))
    fixture=out.parent/f"{module_names.index(name):02d}"
    fixture.mkdir()
    for key,value in (("FIXTURE_ROOT",fixture),("PLAN",plan),("CRITICAL_SOURCES",sources),
                      ("SOURCES",sources),("PILOT_INITIALIZATION",initialization)):
        if hasattr(module,key):
            setattr(module,key,value)
    stream=io.StringIO()
    result=unittest.TextTestRunner(stream=stream,verbosity=2).run(
        unittest.defaultTestLoader.loadTestsFromModule(module))
    (out/"test_output.txt").write_text(stream.getvalue(),encoding="utf-8")
    report=dict(status="PASSED" if result.wasSuccessful() and not result.skipped else "FAILED",
        tests_run=result.testsRun,failures=len(result.failures),errors=len(result.errors),
        skipped=len(result.skipped),source_sha256=sources,python_version=".".join(map(str,sys.version_info[:3])),
        test_module=name,ledger=getattr(module,"LEDGER",getattr(module,"COUNTERS",{})),
        fixture_directory=fixture.relative_to(ROOT).as_posix(),
        tolerances_policy="unchanged in byte-identical tests")
    contract.write_new(out/"test_results.json",report)
    return 0 if report["status"]=="PASSED" else 1

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    mode=parser.add_mutually_exclusive_group()
    mode.add_argument("--tests-only",action="store_true")
    mode.add_argument("--test-module")
    parser.add_argument("--output")
    args=parser.parse_args()
    if not (args.tests_only or args.test_module):
        parser.print_help()
        return 0
    if not args.output:
        parser.error("Tests require a new --output directory under reproductions/")
    sys.path.insert(0,str(ROOT))
    from src import public_io as contract
    contract.validate_runtime()
    contract.verify_archive()
    if args.test_module:
        return one_module(args.test_module,args.output,contract)
    out=contract.output_path(args.output,create=True)
    reports=[]
    commands=[]
    for path in sorted((ROOT/"tests").glob("test_*.py")):
        child_relative=(out/path.stem).relative_to(ROOT).as_posix()
        command=[sys.executable,"-I","-S","-B",str(Path(__file__).resolve()),
                 "--test-module",path.stem,"--output",child_relative]
        completed=subprocess.run(command,cwd=ROOT)
        record_path=out/path.stem/"test_results.json"
        if not record_path.exists():
            raise RuntimeError("Test subprocess failed without a report; preserve outputs")
        record=contract.read(record_path)
        contract.require(completed.returncode==(0 if record["status"]=="PASSED" else 1),
                         "Test subprocess/result status differs")
        reports.append(record)
        commands.append(dict(test_module=path.stem,exit_code=completed.returncode))
    totals={name:sum(r[name] for r in reports) for name in ("tests_run","failures","errors","skipped")}
    result=dict(status="PASSED" if all(r["status"]=="PASSED" for r in reports) else "FAILED",
                **totals,source_sha256=contract.source_hashes(),
                python_version=".".join(map(str,sys.version_info[:3])),
                test_modules=[r["test_module"] for r in reports],commands=commands,
                isolation="one interpreter process per historical test module",
                scope="mathematical and synthetic integration checks; not a research campaign",
                ledgers={r["test_module"]:r["ledger"] for r in reports})
    contract.write_new(out/"test_results.json",result)
    print(result["status"],result["tests_run"])
    return 0 if result["status"]=="PASSED" else 1

if __name__=="__main__":
    raise SystemExit(main())
