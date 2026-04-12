from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

DEFAULT_MODELS: list[str] = [
    "minimax/minimax-m2.7",
    "qwen/qwen3.5-9b",
    "z-ai/glm-5",
]

REQUIRED_COLUMNS: list[str] = [
    "article_id",
    "article_title",
    "id_phrase",
    "phrase",
    "temporal_expression",
    "interval",
]

ALLOWED_LABELS: tuple[str, ...] = (
    "closed",
    "closed_duration",
    "left_open",
    "right_open",
)


@dataclass(slots=True, frozen=True)
class RuntimeConfig:
    input_path: Path
    output_dir: Path
    models: list[str]
    max_rows: int | None = None
    max_retries_per_row: int = 3
    retry_base_delay_seconds: float = 2.0
    timeout_seconds: float | None = None
    reasoning_effort: str | None = None
    openrouter_api_key: str | None = None
    openrouter_app_url: str | None = None
    openrouter_app_title: str | None = None

    @classmethod
    def from_args(
        cls,
        *,
        input_path: str,
        output_dir: str,
        models: list[str] | None = None,
        max_rows: int | None = None,
        max_retries_per_row: int = 3,
        retry_base_delay_seconds: float = 2.0,
        timeout_seconds: float | None = None,
        reasoning_effort: str | None = None,
    ) -> "RuntimeConfig":
        return cls(
            input_path=Path(input_path),
            output_dir=Path(output_dir),
            models=models or DEFAULT_MODELS,
            max_rows=max_rows,
            max_retries_per_row=max_retries_per_row,
            retry_base_delay_seconds=retry_base_delay_seconds,
            timeout_seconds=timeout_seconds,
            reasoning_effort=reasoning_effort,
            openrouter_api_key=os.getenv("OPENROUTER_API_KEY"),
            openrouter_app_url=os.getenv("OPENROUTER_APP_URL"),
            openrouter_app_title=os.getenv("OPENROUTER_APP_TITLE", "Temporal Corpus Annotator"),
        )

    def validate(self) -> None:
        if not self.input_path.exists():
            raise FileNotFoundError(f"Input CSV not found: {self.input_path}")
        if not self.openrouter_api_key:
            raise EnvironmentError(
                "OPENROUTER_API_KEY is missing. Export it or define it in a .env file created from .env.example."
            )
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "annotations").mkdir(parents=True, exist_ok=True)
        (self.output_dir / "logs").mkdir(parents=True, exist_ok=True)
        (self.output_dir / "messages").mkdir(parents=True, exist_ok=True)
        (self.output_dir / "summaries").mkdir(parents=True, exist_ok=True)
