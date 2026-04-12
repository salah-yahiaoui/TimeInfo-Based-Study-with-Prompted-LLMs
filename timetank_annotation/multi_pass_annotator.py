from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass
from typing import Any, Literal

from pydantic import BaseModel, Field
from pydantic_ai import Agent, PromptedOutput
from pydantic_ai.models.openrouter import OpenRouterModel, OpenRouterModelSettings
from pydantic_ai.providers.openrouter import OpenRouterProvider

from timetank_annotation.single_pass_annotator import format_annotation_error
from timetank_annotation.multi_pass_prompts import (
    ANNOTATOR_SYSTEM_PROMPT,
    VERIFIER_SYSTEM_PROMPT,
    render_annotation_request,
    render_verification_request,
)
from timetank_annotation.schemas import AnnotationLogRecord, TemporalAnnotation


class ReviewIssue(BaseModel):
    kind: Literal[
        "missing_annotation",
        "wrong_label",
        "invalid_span",
        "extra_annotation",
        "duplicate_annotation",
        "format_problem",
        "other",
    ]
    message: str = Field(..., min_length=1)


class ReviewDecision(BaseModel):
    verdict: Literal["accept", "revise"]
    feedback: str = Field(..., min_length=1)
    issues: list[ReviewIssue] = Field(default_factory=list)
    suggested_annotations: list[TemporalAnnotation] = Field(default_factory=list)


@dataclass(slots=True)
class ReviewRoundRecord:
    round_number: int
    annotator_prompt: str
    annotator_annotations: list[TemporalAnnotation]
    annotator_usage: dict[str, Any] | None
    reviewer_prompt: str
    reviewer_decision: ReviewDecision
    reviewer_usage: dict[str, Any] | None
    controller_feedback: list[str]
    accepted: bool


@dataclass(slots=True)
class AnnotationOutcome:
    annotations: list[TemporalAnnotation]
    duration_seconds: float
    attempt_count: int
    usage: dict[str, Any] | None
    run_id: str | None
    messages_json: str | None
    review_round_count: int
    accepted_by_reviewer: bool

    @property
    def step_count(self) -> int:
        return self.review_round_count


class TemporalAnnotatorReAct:
    def __init__(
        self,
        *,
        model_name: str,
        api_key: str,
        app_url: str | None,
        app_title: str | None,
        reasoning_effort: str | None = None,
        max_retries_per_row: int = 3,
        retry_base_delay_seconds: float = 2.0,
        max_review_rounds: int = 3,
        reviewer_model_name: str | None = None,
    ) -> None:
        self.model_name = model_name
        self.reviewer_model_name = reviewer_model_name or model_name
        self.max_retries_per_row = max_retries_per_row
        self.retry_base_delay_seconds = retry_base_delay_seconds
        self.max_review_rounds = max_review_rounds

        provider = OpenRouterProvider(
            api_key=api_key,
            app_url=app_url,
            app_title=app_title,
        )

        annotator_model = OpenRouterModel(model_name, provider=provider)
        reviewer_model = OpenRouterModel(self.reviewer_model_name, provider=provider)

        settings_kwargs: dict[str, Any] = {
            "temperature": 1.0,
            "openrouter_usage": {"include": True},
            "openrouter_provider": {
                "allow_fallbacks": True,
                "require_parameters": True,
                "ignore": ["atlas-cloud"],
            },
        }
        if reasoning_effort:
            settings_kwargs["openrouter_reasoning"] = {
                "effort": reasoning_effort,
                "exclude": True,
            }

        model_settings = OpenRouterModelSettings(**settings_kwargs)

        self.annotation_agent = Agent(
            annotator_model,
            system_prompt=ANNOTATOR_SYSTEM_PROMPT,
            output_type=PromptedOutput(
                list[TemporalAnnotation],
                name="temporal_annotations",
                description="Full temporal annotation list for the sentence under the operational TimeInfo interval scheme.",
            ),
            model_settings=model_settings,
        )
        self.verification_agent = Agent(
            reviewer_model,
            system_prompt=VERIFIER_SYSTEM_PROMPT,
            output_type=PromptedOutput(
                ReviewDecision,
                name="annotation_review",
                description="Structured review decision for the candidate annotation list.",
            ),
            model_settings=model_settings,
        )

    async def annotate(self, sentence: str) -> AnnotationOutcome:
        last_error: Exception | None = None

        for attempt in range(1, self.max_retries_per_row + 1):
            started = time.perf_counter()
            try:
                return await self._run_review_loop(
                    sentence=sentence,
                    attempt_count=attempt,
                    started=started,
                )
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                if attempt >= self.max_retries_per_row:
                    break
                await asyncio.sleep(self.retry_base_delay_seconds * (2 ** (attempt - 1)))

        assert last_error is not None
        raise last_error

    async def _run_review_loop(
        self,
        *,
        sentence: str,
        attempt_count: int,
        started: float,
    ) -> AnnotationOutcome:
        review_records: list[ReviewRoundRecord] = []
        usage_total = _empty_usage()
        last_run_id: str | None = None

        previous_annotations: list[TemporalAnnotation] | None = None
        reviewer_feedback: str | None = None
        reviewer_suggested_annotations: list[TemporalAnnotation] | None = None
        controller_feedback: list[str] | None = None
        final_annotations: list[TemporalAnnotation] | None = None
        accepted_by_reviewer = False

        for review_round in range(1, self.max_review_rounds + 1):
            annotator_prompt = render_annotation_request(
                sentence=sentence,
                review_round=review_round,
                max_review_rounds=self.max_review_rounds,
                previous_annotations=previous_annotations,
                reviewer_feedback=reviewer_feedback,
                reviewer_suggested_annotations=reviewer_suggested_annotations,
                controller_feedback=controller_feedback,
            )
            annotation_result = await self.annotation_agent.run(annotator_prompt)
            annotator_annotations = list(annotation_result.output)
            annotator_usage = _safe_to_dict(annotation_result.usage())
            _merge_usage(usage_total, annotator_usage)
            last_run_id = getattr(annotation_result, "run_id", None) or last_run_id

            reviewer_prompt = render_verification_request(
                sentence=sentence,
                candidate_annotations=annotator_annotations,
                review_round=review_round,
                max_review_rounds=self.max_review_rounds,
            )
            review_result = await self.verification_agent.run(reviewer_prompt)
            reviewer_decision = review_result.output
            reviewer_usage = _safe_to_dict(review_result.usage())
            _merge_usage(usage_total, reviewer_usage)
            last_run_id = getattr(review_result, "run_id", None) or last_run_id

            if reviewer_decision.verdict == "accept" and not reviewer_decision.suggested_annotations:
                reviewer_decision = reviewer_decision.model_copy(
                    update={"suggested_annotations": annotator_annotations}
                )

            controller_feedback = _collect_controller_feedback(sentence, annotator_annotations)
            accepted = reviewer_decision.verdict == "accept" and not controller_feedback

            review_records.append(
                ReviewRoundRecord(
                    round_number=review_round,
                    annotator_prompt=annotator_prompt,
                    annotator_annotations=annotator_annotations,
                    annotator_usage=annotator_usage,
                    reviewer_prompt=reviewer_prompt,
                    reviewer_decision=reviewer_decision,
                    reviewer_usage=reviewer_usage,
                    controller_feedback=controller_feedback,
                    accepted=accepted,
                )
            )

            final_annotations = annotator_annotations
            previous_annotations = annotator_annotations
            reviewer_feedback = _combine_feedback(reviewer_decision, controller_feedback)
            reviewer_suggested_annotations = reviewer_decision.suggested_annotations

            if accepted:
                accepted_by_reviewer = True
                break

        assert final_annotations is not None
        duration = time.perf_counter() - started
        messages_json = _serialize_review_trace(
            sentence=sentence,
            model_name=self.model_name,
            reviewer_model_name=self.reviewer_model_name,
            max_review_rounds=self.max_review_rounds,
            accepted_by_reviewer=accepted_by_reviewer,
            final_annotations=final_annotations,
            review_records=review_records,
        )
        return AnnotationOutcome(
            annotations=final_annotations,
            duration_seconds=duration,
            attempt_count=attempt_count,
            usage=usage_total,
            run_id=last_run_id,
            messages_json=messages_json,
            review_round_count=len(review_records),
            accepted_by_reviewer=accepted_by_reviewer,
        )

    @staticmethod
    def make_error_log(
        *,
        row_index: int,
        model_name: str,
        model_slug: str,
        phrase: str,
        article_id: str | None,
        id_phrase: str | None,
        attempt_count: int,
        duration_seconds: float,
        error: Exception,
    ) -> AnnotationLogRecord:
        return AnnotationLogRecord(
            row_index=row_index,
            model_name=model_name,
            model_slug=model_slug,
            article_id=article_id,
            id_phrase=id_phrase,
            status="error",
            attempt_count=attempt_count,
            duration_seconds=duration_seconds,
            phrase=phrase,
            annotations=[],
            error=format_annotation_error(error),
            usage=None,
            usage_input_tokens=0,
            usage_output_tokens=0,
            usage_total_tokens=0,
            usage_requests=0,
            usage_tool_calls=0,
            usage_details={},
            run_id=None,
            messages_file=None,
        )


def _collect_controller_feedback(sentence: str, annotations: list[TemporalAnnotation]) -> list[str]:
    issues: list[str] = []
    seen: set[tuple[str, str]] = set()

    for annotation in annotations:
        key = (annotation.text, annotation.label)
        if key in seen:
            issues.append(
                f"Duplicate annotation detected for span {annotation.text!r} with label {annotation.label!r}."
            )
        else:
            seen.add(key)

        if annotation.text not in sentence:
            issues.append(
                f"Span {annotation.text!r} does not occur exactly in the sentence and must be copied verbatim."
            )

    return issues


def _combine_feedback(reviewer_decision: ReviewDecision, controller_feedback: list[str]) -> str:
    segments: list[str] = []
    reviewer_feedback = " ".join(reviewer_decision.feedback.split()).strip()
    if reviewer_feedback:
        segments.append(f"Reviewer feedback: {reviewer_feedback}")
    if reviewer_decision.issues:
        issue_lines = [f"- {issue.kind}: {issue.message}" for issue in reviewer_decision.issues]
        segments.append("Reviewer issues:\n" + "\n".join(issue_lines))
    if controller_feedback:
        controller_lines = [f"- {issue}" for issue in controller_feedback]
        segments.append("Controller technical feedback:\n" + "\n".join(controller_lines))
    return "\n\n".join(segments) if segments else "Revise the previous annotation list."


def _serialize_review_trace(
    *,
    sentence: str,
    model_name: str,
    reviewer_model_name: str,
    max_review_rounds: int,
    accepted_by_reviewer: bool,
    final_annotations: list[TemporalAnnotation],
    review_records: list[ReviewRoundRecord],
) -> str:
    payload = {
        "mode": "multi_agent_review_loop",
        "sentence": sentence,
        "annotation_model": model_name,
        "reviewer_model": reviewer_model_name,
        "max_review_rounds": max_review_rounds,
        "review_round_count": len(review_records),
        "accepted_by_reviewer": accepted_by_reviewer,
        "final_annotations": [annotation.model_dump() for annotation in final_annotations],
        "rounds": [
            {
                "round_number": record.round_number,
                "annotator_prompt": record.annotator_prompt,
                "annotator_annotations": [item.model_dump() for item in record.annotator_annotations],
                "annotator_usage": record.annotator_usage,
                "reviewer_prompt": record.reviewer_prompt,
                "reviewer_decision": record.reviewer_decision.model_dump(),
                "reviewer_usage": record.reviewer_usage,
                "controller_feedback": record.controller_feedback,
                "accepted": record.accepted,
            }
            for record in review_records
        ],
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def _empty_usage() -> dict[str, Any]:
    return {
        "input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
        "requests": 0,
        "tool_calls": 0,
        "details": {},
    }


def _merge_usage(target: dict[str, Any], extra: dict[str, Any] | None) -> None:
    if not extra:
        return

    step_input = _coerce_int(extra.get("input_tokens", extra.get("request_tokens", 0)))
    step_output = _coerce_int(extra.get("output_tokens", extra.get("response_tokens", 0)))
    step_total = _coerce_int(extra.get("total_tokens", 0))
    if step_total == 0:
        step_total = step_input + step_output

    target["input_tokens"] += step_input
    target["output_tokens"] += step_output
    target["total_tokens"] += step_total
    target["requests"] += _coerce_int(extra.get("requests", 0))
    target["tool_calls"] += _coerce_int(extra.get("tool_calls", 0))

    details = extra.get("details", {})
    if isinstance(details, dict):
        merged_details = target.setdefault("details", {})
        for key, value in details.items():
            merged_details[str(key)] = merged_details.get(str(key), 0) + _coerce_int(value)


def _safe_to_dict(value: Any) -> dict[str, Any] | None:
    if value is None:
        return None
    if hasattr(value, "model_dump"):
        dumped = value.model_dump()
        return dumped if isinstance(dumped, dict) else {"value": dumped}
    if isinstance(value, dict):
        return value
    if hasattr(value, "__dict__"):
        return dict(vars(value))
    try:
        dumped = json.loads(json.dumps(value))
        return dumped if isinstance(dumped, dict) else {"value": dumped}
    except TypeError:
        return {"value": str(value)}


def _coerce_int(value: Any) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    return 0
