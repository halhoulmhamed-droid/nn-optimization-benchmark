"""Compatibility entry for explicit public numerical checks."""
from pathlib import Path
import runpy
if __name__=="__main__":
    runpy.run_path(str(Path(__file__).with_name("run_checks.py")),run_name="__main__")

