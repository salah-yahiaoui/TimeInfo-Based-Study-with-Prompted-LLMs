from __future__ import annotations

import dataclasses
import json
from datetime import date, datetime
from pathlib import Path
from typing import Any


def slugify_model_name(model_name: str) -> str:
    return model_name.replace("/", "__")


def to_jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if dataclasses.is_dataclass(value):
        return {k: to_jsonable(v) for k, v in dataclasses.asdict(value).items()}
    if hasattr(value, "model_dump"):
        return to_jsonable(value.model_dump())
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [to_jsonable(item) for item in value]
    if hasattr(value, "__dict__"):
        return {k: to_jsonable(v) for k, v in vars(value).items()}
    return str(value)


class JsonlLogger:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = self.path.open("w", encoding="utf-8")

    def write(self, payload: dict[str, Any]) -> None:
        self._handle.write(json.dumps(to_jsonable(payload), ensure_ascii=False) + "\n")
        self._handle.flush()

    def close(self) -> None:
        self._handle.close()

    def __enter__(self) -> "JsonlLogger":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
