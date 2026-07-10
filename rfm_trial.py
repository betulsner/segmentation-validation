"""Compatibility wrapper for the renamed baseline script."""

from pathlib import Path
import runpy

if __name__ == "__main__":
    runpy.run_path(str(Path(__file__).with_name("rfm_baseline_mvp.py")), run_name="__main__")
