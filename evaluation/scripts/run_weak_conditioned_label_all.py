from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    commands = [
        [sys.executable, str(script_dir / "run_weak_conditioned_label_evaluation.py")],
        [sys.executable, str(script_dir / "generate_weak_conditioned_label_latex.py")],
    ]

    for command in commands:
        print(f"[WEAK LABEL PIPELINE] running={' '.join(command)}", flush=True)
        subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
