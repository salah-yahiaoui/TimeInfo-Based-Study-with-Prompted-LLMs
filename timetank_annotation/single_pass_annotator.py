from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import ValidationError
from pydantic_ai import Agent, PromptedOutput
from pydantic_ai.models.openrouter import OpenRouterModel, OpenRouterModelSettings
from pydantic_ai.providers.openrouter import OpenRouterProvider

from timetank_annotation.single_pass_prompts import render_prompt
from timetank_annotation.schemas import AnnotationLogRecord, TemporalAnnotation


@dataclass(slots=True)
class AnnotationOutcome:
    annotations: list[TemporalAnnotation]
    duration_seconds: float
    attempt_count: int
    usage: dict[str, Any] | None
    run_id: str | None
    messages_json: str | None


class TemporalAnnotator:
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
    ) -> None:
        self.model_name = model_name
        self.max_retries_per_row = max_retries_per_row
        self.retry_base_delay_seconds = retry_base_delay_seconds

        provider = OpenRouterProvider(
            api_key=api_key,
            app_url=app_url,
            app_title=app_title,
        )
        model = OpenRouterModel(model_name, provider=provider)

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

        self.agent = Agent(
            model,
            output_type=PromptedOutput(
                list[TemporalAnnotation],
                name="temporal_annotations",
                description="Extract temporal expressions and assign one of the four interval labels.",
            ),
            model_settings=model_settings,
        )

    async def annotate(self, sentence: str) -> AnnotationOutcome:
        last_error: Exception | None = None

        for attempt in range(1, self.max_retries_per_row + 1):
            started = time.perf_counter()
            try:
                result = await self.agent.run(render_prompt(sentence))
                duration = time.perf_counter() - started
                annotations = list(result.output)
                usage = _safe_to_dict(result.usage())
                messages_json = _safe_bytes_to_str(result.new_messages_json())
                return AnnotationOutcome(
                    annotations=annotations,
                    duration_seconds=duration,
                    attempt_count=attempt,
                    usage=usage,
                    run_id=getattr(result, "run_id", None),
                    messages_json=messages_json,
                )
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                if attempt >= self.max_retries_per_row:
                    break
                await asyncio.sleep(self.retry_base_delay_seconds * (2 ** (attempt - 1)))

        assert last_error is not None
        raise last_error

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


def persist_messages(messages_dir: Path, basename: str, messages_json: str | None) -> str | None:
    if not messages_json:
        return None
    messages_dir.mkdir(parents=True, exist_ok=True)
    path = messages_dir / f"{basename}.json"
    path.write_text(messages_json, encoding="utf-8")
    return str(path)


def make_success_log(
    *,
    row_index: int,
    model_name: str,
    model_slug: str,
    phrase: str,
    article_id: str | None,
    id_phrase: str | None,
    outcome: AnnotationOutcome,
    messages_file: str | None,
) -> AnnotationLogRecord:
    usage_stats = summarize_usage(outcome.usage)
    return AnnotationLogRecord(
        row_index=row_index,
        model_name=model_name,
        model_slug=model_slug,
        article_id=article_id,
        id_phrase=id_phrase,
        status="success",
        attempt_count=outcome.attempt_count,
        duration_seconds=outcome.duration_seconds,
        phrase=phrase,
        annotations=outcome.annotations,
        error=None,
        usage=outcome.usage,
        usage_input_tokens=usage_stats["input_tokens"],
        usage_output_tokens=usage_stats["output_tokens"],
        usage_total_tokens=usage_stats["total_tokens"],
        usage_requests=usage_stats["requests"],
        usage_tool_calls=usage_stats["tool_calls"],
        usage_details=usage_stats["details"],
        run_id=outcome.run_id,
        messages_file=messages_file,
    )


def summarize_usage(usage: dict[str, Any] | None) -> dict[str, Any]:
    if not usage:
        return {
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
            "requests": 0,
            "tool_calls": 0,
            "details": {},
        }

    input_tokens = _coerce_int(usage.get("input_tokens", usage.get("request_tokens", 0)))
    output_tokens = _coerce_int(usage.get("output_tokens", usage.get("response_tokens", 0)))
    total_tokens = _coerce_int(usage.get("total_tokens", input_tokens + output_tokens))
    requests = _coerce_int(usage.get("requests", 0))
    tool_calls = _coerce_int(usage.get("tool_calls", 0))

    raw_details = usage.get("details", {})
    details: dict[str, int] = {}
    if isinstance(raw_details, dict):
        for key, value in raw_details.items():
            coerced = _coerce_optional_int(value)
            if coerced is not None:
                details[str(key)] = coerced

    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
        "requests": requests,
        "tool_calls": tool_calls,
        "details": details,
    }


def format_annotation_error(error: Exception) -> str:
    base_message = f"{type(error).__name__}: {error}"
    diagnosis = _extract_openrouter_diagnosis(error)
    if diagnosis:
        return f"{base_message}\nOpenRouter diagnosis: {diagnosis}"
    return base_message


def _extract_openrouter_diagnosis(error: Exception) -> str | None:
    for candidate in _iter_exception_chain(error):
        status_code = getattr(candidate, "status_code", None)
        body = getattr(candidate, "body", None)
        if status_code is not None and body is not None:
            message = _stringify_error_body(body)
            if message:
                return _build_openrouter_diagnosis(status_code, message)

        if isinstance(candidate, ValidationError):
            for issue in candidate.errors(include_url=False):
                input_value = issue.get("input")
                if not isinstance(input_value, dict):
                    continue
                error_payload = input_value.get("error")
                if isinstance(error_payload, dict):
                    message = error_payload.get("message")
                    code = error_payload.get("code")
                    if isinstance(message, str):
                        return _build_openrouter_diagnosis(code, message)

    text = str(error)
    if "Invalid response from openrouter chat completions endpoint" in text and "provider" in text:
        return (
            "OpenRouter returned an error payload instead of a valid chat completion. "
            "In PydanticAI 1.74.0, the OpenRouter adapter expects a completion object with a top-level "
            "`provider` field, so upstream 502 responses surface as `UnexpectedModelBehavior`. "
            "The underlying cause is usually that no downstream endpoint matched the requested features "
            "(most often tool-based structured output and/or reasoning support for the selected model)."
        )
    return None


def _build_openrouter_diagnosis(code: Any, message: str) -> str:
    code_text = f"code={code}" if code is not None else "code=unknown"
    lowered = message.lower()
    hints: list[str] = []
    if "tool" in lowered or "structured" in lowered:
        hints.append("tool/structured-output support")
    if "reason" in lowered:
        hints.append("reasoning support")
    if "provider" in lowered or "endpoint" in lowered:
        hints.append("provider routing")

    if hints:
        return f"{code_text}; {message} Likely blocked by: {', '.join(hints)}."
    return f"{code_text}; {message}"


def _iter_exception_chain(error: Exception) -> list[BaseException]:
    chain: list[BaseException] = []
    seen: set[int] = set()
    current: BaseException | None = error

    while current is not None and id(current) not in seen:
        chain.append(current)
        seen.add(id(current))
        current = current.__cause__ or current.__context__

    return chain


def _stringify_error_body(body: Any) -> str | None:
    if isinstance(body, str):
        return body
    if isinstance(body, bytes):
        return body.decode("utf-8", errors="replace")
    if isinstance(body, dict):
        error_payload = body.get("error")
        if isinstance(error_payload, dict):
            message = error_payload.get("message")
            if isinstance(message, str):
                return message
        return json.dumps(body, ensure_ascii=False)
    return None


def _safe_bytes_to_str(payload: bytes | bytearray | str | None) -> str | None:
    if payload is None:
        return None
    if isinstance(payload, str):
        return payload
    return payload.decode("utf-8")


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
    coerced = _coerce_optional_int(value)
    return coerced if coerced is not None else 0


def _coerce_optional_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    return None
