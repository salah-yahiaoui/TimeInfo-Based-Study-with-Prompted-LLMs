from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


ROOT_DIR = Path(__file__).resolve().parents[2]
EVAL_DIR = ROOT_DIR / "evaluation"
RESULTS_DIR = EVAL_DIR / "results"
TABLES_DIR = EVAL_DIR / "tables"
FIGURES_DIR = EVAL_DIR / "figures"

GOLD_PATH = ROOT_DIR / "data" / "TimeTankSample.csv"

EXPERIMENTS = [
    {
        "experiment_name": "single_pass",
        "experiment_label": "Single-pass",
        "prediction_root": ROOT_DIR / "generation_results" / "single_pass",
    },
    {
        "experiment_name": "multi_pass",
        "experiment_label": "Multi-pass",
        "prediction_root": ROOT_DIR / "generation_results" / "multi_pass",
    },
]


def ensure_output_dirs() -> None:
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def to_float(value: Any) -> float:
    if isinstance(value, bool):
        return float(int(value))
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        value = value.strip()
        if not value:
            return 0.0
        return float(value)
    return 0.0


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = collect_fieldnames(rows)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: serialize_csv_value(value) for key, value in row.items()})


def collect_fieldnames(rows: list[dict[str, Any]]) -> list[str]:
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    return fieldnames


def serialize_csv_value(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, (str, int, float, bool)):
        return value
    return json.dumps(value, ensure_ascii=False)
