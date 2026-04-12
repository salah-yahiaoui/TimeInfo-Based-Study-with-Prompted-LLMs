from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

TemporalLabel = Literal["closed", "closed_duration", "left_open", "right_open"]


class TemporalAnnotation(BaseModel):
    """One temporal expression extracted from a sentence."""

    text: str = Field(
        ...,
        description="Exact temporal span copied from the sentence without paraphrase.",
        min_length=1,
    )
    label: TemporalLabel = Field(
        ...,
        description="Interval type for the extracted temporal expression.",
    )


class AnnotationLogRecord(BaseModel):
    row_index: int
    model_name: str
    model_slug: str
    article_id: str | None = None
    id_phrase: str | None = None
    status: Literal["success", "error"]
    attempt_count: int
    duration_seconds: float
    phrase: str
    annotations: list[TemporalAnnotation] = Field(default_factory=list)
    error: str | None = None
    usage: dict | None = None
    usage_input_tokens: int = 0
    usage_output_tokens: int = 0
    usage_total_tokens: int = 0
    usage_requests: int = 0
    usage_tool_calls: int = 0
    usage_details: dict[str, int] = Field(default_factory=dict)
    run_id: str | None = None
    messages_file: str | None = None
