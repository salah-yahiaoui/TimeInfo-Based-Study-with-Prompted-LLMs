from __future__ import annotations

import subprocess
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent


def run_script(script_name: str) -> None:
    script_path = SCRIPT_DIR / script_name
    subprocess.run([sys.executable, str(script_path)], check=True)


def main() -> None:
    run_script("run_gold_restricted_evaluation.py")
    run_script("generate_evaluation_figures.py")


if __name__ == "__main__":
    main()
