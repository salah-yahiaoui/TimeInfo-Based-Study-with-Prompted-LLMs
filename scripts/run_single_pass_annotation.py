from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from timetank_annotation.single_pass_annotator import (
    TemporalAnnotator,
    format_annotation_error,
    make_success_log,
    persist_messages,
    summarize_usage,
)
from timetank_annotation.config import RuntimeConfig
from timetank_annotation.csv_utils import AnnotationCsvWriter, read_corpus_rows, rebuild_output_rows
from timetank_annotation.logging_utils import JsonlLogger, slugify_model_name, to_jsonable


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Annotate a temporal corpus with multiple OpenRouter models via PydanticAI."
    )
    parser.add_argument("--input", default="data/TimeTankSample.csv", help="Path to the input CSV corpus.")
    parser.add_argument(
        "--output-dir",
        default="generation_results/single_pass",
        help="Directory where outputs will be saved.",
    )
    parser.add_argument(
        "--models",
        nargs="*",
        default=None,
        help="Optional list of model names. Defaults to the 3 configured models.",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Optional limit for debugging on a subset of rows.",
    )
    parser.add_argument(
        "--max-retries-per-row",
        type=int,
        default=3,
        help="How many attempts per sentence before logging a failure.",
    )
    parser.add_argument(
        "--retry-base-delay-seconds",
        type=float,
        default=2.0,
        help="Base delay for exponential backoff between attempts.",
    )
    parser.add_argument(
        "--reasoning-effort",
        default=None,
        choices=["low", "medium", "high"],
        help="Optional OpenRouter reasoning effort to enable for compatible models.",
    )
    return parser


async def annotate_one_model(
    *,
    model_name: str,
    rows: list[dict[str, str]],
    fieldnames: list[str],
    config: RuntimeConfig,
) -> None:
    model_slug = slugify_model_name(model_name)
    annotation_path = config.output_dir / "annotations" / f"{model_slug}.csv"
    log_path = config.output_dir / "logs" / f"{model_slug}.jsonl"
    summary_path = config.output_dir / "summaries" / f"{model_slug}.summary.json"
    messages_dir = config.output_dir / "messages" / model_slug

    annotator = TemporalAnnotator(
        model_name=model_name,
        api_key=config.openrouter_api_key or "",
        app_url=config.openrouter_app_url,
        app_title=config.openrouter_app_title,
        reasoning_effort=config.reasoning_effort,
        max_retries_per_row=config.max_retries_per_row,
        retry_base_delay_seconds=config.retry_base_delay_seconds,
    )

    summary = {
        "model_name": model_name,
        "model_slug": model_slug,
        "input_csv": str(config.input_path),
        "annotation_csv": str(annotation_path),
        "log_file": str(log_path),
        "messages_dir": str(messages_dir),
        "total_input_rows": len(rows),
        "successful_rows": 0,
        "failed_rows": 0,
        "total_output_rows": 0,
        "usage_requests": 0,
        "usage_tool_calls": 0,
        "usage_input_tokens": 0,
        "usage_output_tokens": 0,
        "usage_total_tokens": 0,
        "usage_details": {},
        "reasoning_effort": config.reasoning_effort,
        "temperature": 1.0,
        "started_at_unix": time.time(),
        "last_processed_row_index": 0,
    }

    print(
        f"[MODEL START] {model_name} rows={len(rows)} temperature=1.0 reasoning={config.reasoning_effort or 'off'}",
        flush=True,
    )
    _write_summary(summary_path, summary)

    with AnnotationCsvWriter(annotation_path, fieldnames) as csv_writer, JsonlLogger(log_path) as logger:
        for row_index, row in enumerate(rows, start=1):
            phrase = row.get("phrase", "")
            article_id = row.get("article_id")
            id_phrase = row.get("id_phrase")
            basename = f"{row_index:06d}_{article_id or 'na'}_{id_phrase or 'na'}"

            started = time.perf_counter()
            try:
                outcome = await annotator.annotate(phrase)
                usage_stats = summarize_usage(outcome.usage)
                messages_file = persist_messages(messages_dir, basename, outcome.messages_json)
                success_log = make_success_log(
                    row_index=row_index,
                    model_name=model_name,
                    model_slug=model_slug,
                    phrase=phrase,
                    article_id=article_id,
                    id_phrase=id_phrase,
                    outcome=outcome,
                    messages_file=messages_file,
                )
                logger.write(success_log.model_dump())

                output_rows = rebuild_output_rows(row, outcome.annotations)
                csv_writer.write_rows(output_rows)

                summary["successful_rows"] += 1
                summary["total_output_rows"] += len(output_rows)
                summary["usage_requests"] += usage_stats["requests"]
                summary["usage_tool_calls"] += usage_stats["tool_calls"]
                summary["usage_input_tokens"] += usage_stats["input_tokens"]
                summary["usage_output_tokens"] += usage_stats["output_tokens"]
                summary["usage_total_tokens"] += usage_stats["total_tokens"]
                _merge_usage_details(summary["usage_details"], usage_stats["details"])

                print(
                    f"[{model_slug}] {row_index}/{len(rows)} success "
                    f"attempts={outcome.attempt_count} annotations={len(outcome.annotations)} "
                    f"row_tokens={usage_stats['total_tokens']} "
                    f"tot_tokens={summary['usage_total_tokens']} "
                    f"ok={summary['successful_rows']} err={summary['failed_rows']}",
                    flush=True,
                )
            except Exception as exc:  # noqa: BLE001
                duration = time.perf_counter() - started
                error_log = annotator.make_error_log(
                    row_index=row_index,
                    model_name=model_name,
                    model_slug=model_slug,
                    phrase=phrase,
                    article_id=article_id,
                    id_phrase=id_phrase,
                    attempt_count=config.max_retries_per_row,
                    duration_seconds=duration,
                    error=exc,
                )
                logger.write(error_log.model_dump())

                output_rows = rebuild_output_rows(row, [])
                csv_writer.write_rows(output_rows)

                summary["failed_rows"] += 1
                summary["total_output_rows"] += len(output_rows)

                print(
                    f"[{model_slug}] {row_index}/{len(rows)} error "
                    f"attempts={config.max_retries_per_row} "
                    f"ok={summary['successful_rows']} err={summary['failed_rows']} "
                    f"detail={_single_line(format_annotation_error(exc), limit=260)}",
                    flush=True,
                )

            summary["last_processed_row_index"] = row_index
            _write_summary(summary_path, summary)

    summary["finished_at_unix"] = time.time()
    _write_summary(summary_path, summary)
    print(
        f"[MODEL DONE] {model_name} ok={summary['successful_rows']} err={summary['failed_rows']} "
        f"out_rows={summary['total_output_rows']} tokens={summary['usage_total_tokens']}",
        flush=True,
    )


async def main() -> None:
    load_dotenv()
    parser = build_parser()
    args = parser.parse_args()

    config = RuntimeConfig.from_args(
        input_path=args.input,
        output_dir=args.output_dir,
        models=args.models,
        max_rows=args.max_rows,
        max_retries_per_row=args.max_retries_per_row,
        retry_base_delay_seconds=args.retry_base_delay_seconds,
        reasoning_effort=args.reasoning_effort,
    )
    config.validate()

    rows, fieldnames = read_corpus_rows(config.input_path, max_rows=config.max_rows)

    for model_name in config.models:
        await annotate_one_model(
            model_name=model_name,
            rows=rows,
            fieldnames=fieldnames,
            config=config,
        )

    print(f"All outputs written to: {config.output_dir}")


def _merge_usage_details(target: dict[str, int], extra: dict[str, int]) -> None:
    for key, value in extra.items():
        target[key] = target.get(key, 0) + value


def _single_line(value: str, *, limit: int) -> str:
    flattened = " ".join(value.split())
    if len(flattened) <= limit:
        return flattened
    return flattened[: limit - 3] + "..."


def _write_summary(path: Path, summary: dict[str, object]) -> None:
    path.write_text(json.dumps(to_jsonable(summary), ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    asyncio.run(main())
