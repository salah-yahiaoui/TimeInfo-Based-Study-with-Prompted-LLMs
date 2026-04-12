from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable

from timetank_annotation.config import REQUIRED_COLUMNS
from timetank_annotation.schemas import TemporalAnnotation


def read_corpus_rows(input_path: Path, max_rows: int | None = None) -> tuple[list[dict[str, str]], list[str]]:
    with input_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError("Input CSV has no header row.")

        missing = [column for column in REQUIRED_COLUMNS if column not in reader.fieldnames]
        if missing:
            raise ValueError(f"Input CSV is missing required columns: {missing}")

        rows: list[dict[str, str]] = []
        for idx, row in enumerate(reader):
            rows.append(row)
            if max_rows is not None and idx + 1 >= max_rows:
                break

    return rows, list(reader.fieldnames)


class AnnotationCsvWriter:
    def __init__(self, path: Path, fieldnames: list[str]) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = self.path.open("w", encoding="utf-8", newline="")
        self._writer = csv.DictWriter(self._handle, fieldnames=fieldnames)
        self._writer.writeheader()
        self._handle.flush()

    def write_rows(self, rows: Iterable[dict[str, str]]) -> None:
        for row in rows:
            self._writer.writerow(row)
        self._handle.flush()

    def close(self) -> None:
        self._handle.close()

    def __enter__(self) -> "AnnotationCsvWriter":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


def deduplicate_annotations(annotations: list[TemporalAnnotation]) -> list[TemporalAnnotation]:
    seen: set[tuple[str, str]] = set()
    unique: list[TemporalAnnotation] = []
    for annotation in annotations:
        key = (annotation.text, annotation.label)
        if key in seen:
            continue
        seen.add(key)
        unique.append(annotation)
    return unique


def rebuild_output_rows(
    source_row: dict[str, str],
    annotations: list[TemporalAnnotation],
) -> list[dict[str, str]]:
    annotations = deduplicate_annotations(annotations)

    if not annotations:
        row = dict(source_row)
        row["temporal_expression"] = ""
        row["interval"] = ""
        return [row]

    output_rows: list[dict[str, str]] = []
    for annotation in annotations:
        row = dict(source_row)
        row["temporal_expression"] = annotation.text
        row["interval"] = annotation.label
        output_rows.append(row)

    return output_rows
