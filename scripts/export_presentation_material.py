"""Read sealed presentation material without re-exporting or model evaluation."""
import argparse
from pathlib import Path
import json
import sys
ROOT=Path(__file__).resolve().parents[1]
def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check",action="store_true")
    args=parser.parse_args()
    if not args.check:
        parser.print_help()
        return 0
    sys.path.insert(0,str(ROOT))
    from src.public_io import read_only_summary
    print(json.dumps(read_only_summary(),ensure_ascii=True,indent=2,allow_nan=False))
    return 0
if __name__=="__main__":
    raise SystemExit(main())

