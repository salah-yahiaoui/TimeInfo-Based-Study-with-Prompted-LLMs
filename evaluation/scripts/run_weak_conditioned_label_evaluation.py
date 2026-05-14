from __future__ import annotations

from collections import Counter
from pathlib import Path

import run_professional_evaluation as base_eval
from analysis_utils import (
    DOCS_DIR,
    EVAL_DIR,
    EXPERIMENTS,
    LABELS,
    TABLES_DIR,
    ensure_output_dirs,
    model_display_name,
    read_csv_rows,
    slugify_model_name,
    write_csv,
    write_text,
)

SUB_EVAL_DIR = EVAL_DIR / "weak_conditioned_label_eval"
SUB_RESULTS_DIR = SUB_EVAL_DIR / "results"
SUB_TABLES_DIR = SUB_EVAL_DIR / "tables"
SUB_LATEX_DIR = SUB_EVAL_DIR / "latex"
SUB_DOCS_DIR = SUB_EVAL_DIR / "docs"


def ensure_sub_eval_dirs() -> None:
    ensure_output_dirs()
    SUB_EVAL_DIR.mkdir(parents=True, exist_ok=True)
    SUB_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    SUB_TABLES_DIR.mkdir(parents=True, exist_ok=True)
    SUB_LATEX_DIR.mkdir(parents=True, exist_ok=True)
    SUB_DOCS_DIR.mkdir(parents=True, exist_ok=True)


def evaluate_label_prediction_on_weak_matches(
    *,
    experiment_name: str,
    experiment_label: str,
    prediction_file: Path,
    gold_by_sentence: dict[base_eval.SentenceKey, list[base_eval.GoldAnnotation]],
    gold_total: int,
) -> dict[str, object]:
    prediction_records = base_eval.load_prediction_records(prediction_file)
    model_name = base_eval.infer_model_name(prediction_file)
    model_slug = slugify_model_name(model_name)

    predicted_buckets_counter: Counter[str] = Counter()
    tp_by_label: Counter[str] = Counter()
    support_by_label: Counter[str] = Counter()
    details_rows: list[dict[str, object]] = []
    confusion_rows: list[dict[str, object]] = []
    confusion_counter: Counter[tuple[str, str]] = Counter()
    weak_matched_total = 0

    for sentence_key, gold_items in gold_by_sentence.items():
        pred_sentence = prediction_records.get(sentence_key)
        pred_annotations = pred_sentence.annotations if pred_sentence is not None else []

        matches, _unused_unmatched_pred, unmatched_gold_indices = base_eval.maximum_matching(
            pred_annotations,
            gold_items,
            base_eval.weak_match,
        )
        matched_gold_to_pred = {gold_idx: (pred_idx, score) for pred_idx, gold_idx, score in matches}
        unmatched_gold_set = set(unmatched_gold_indices)

        for gold_idx, gold in enumerate(gold_items):
            if gold_idx in unmatched_gold_set:
                confusion_counter[(gold.label, "__NO_WEAK_MATCH__")] += 1
                details_rows.append(
                    {
                        "experiment_name": experiment_name,
                        "experiment_label": experiment_label,
                        "evaluation_kind": "label_prediction_on_weak",
                        "model_name": model_name,
                        "model_display_name": model_display_name(model_name),
                        "model_slug": model_slug,
                        "gold_row_index": gold.row_index,
                        "article_id": gold.article_id,
                        "id_phrase": gold.id_phrase,
                        "article_title": gold.article_title,
                        "phrase": gold.phrase,
                        "gold_temporal_expression": gold.text,
                        "gold_label": gold.label,
                        "used_for_label_eval": 0,
                        "weak_match_found": 0,
                        "weak_match_score": 0.0,
                        "matched_predicted_expression": "",
                        "predicted_label_bucket": "__NO_WEAK_MATCH__",
                        "predicted_label_raw": "",
                        "status": "not_weak_matched",
                        "correct": 0,
                    }
                )
                continue

            pred_idx, score = matched_gold_to_pred[gold_idx]
            pred = pred_annotations[pred_idx]
            weak_matched_total += 1
            support_by_label[gold.label] += 1

            if pred.label in LABELS:
                predicted_bucket = pred.label
                predicted_buckets_counter[predicted_bucket] += 1
                status = "correct" if predicted_bucket == gold.label else "wrong_label"
                if predicted_bucket == gold.label:
                    tp_by_label[gold.label] += 1
            else:
                predicted_bucket = "__INVALID__"
                status = "invalid_label"

            confusion_counter[(gold.label, predicted_bucket)] += 1
            details_rows.append(
                {
                    "experiment_name": experiment_name,
                    "experiment_label": experiment_label,
                    "evaluation_kind": "label_prediction_on_weak",
                    "model_name": model_name,
                    "model_display_name": model_display_name(model_name),
                    "model_slug": model_slug,
                    "gold_row_index": gold.row_index,
                    "article_id": gold.article_id,
                    "id_phrase": gold.id_phrase,
                    "article_title": gold.article_title,
                    "phrase": gold.phrase,
                    "gold_temporal_expression": gold.text,
                    "gold_label": gold.label,
                    "used_for_label_eval": 1,
                    "weak_match_found": 1,
                    "weak_match_score": score,
                    "matched_predicted_expression": pred.text,
                    "predicted_label_bucket": predicted_bucket,
                    "predicted_label_raw": pred.label,
                    "status": status,
                    "correct": int(predicted_bucket == gold.label),
                }
            )

    per_label_rows: list[dict[str, object]] = []
    micro_tp = micro_fp = micro_fn = 0
    macro_precision_sum = 0.0
    macro_recall_sum = 0.0
    macro_f1_sum = 0.0

    for label in LABELS:
        support = support_by_label[label]
        tp = tp_by_label[label]
        predicted = predicted_buckets_counter[label]
        fp = predicted - tp
        fn = support - tp
        precision = base_eval.safe_ratio(tp, tp + fp)
        recall = base_eval.safe_ratio(tp, tp + fn)
        f1 = base_eval.safe_f1(precision, recall)
        micro_tp += tp
        micro_fp += fp
        micro_fn += fn
        macro_precision_sum += precision
        macro_recall_sum += recall
        macro_f1_sum += f1
        per_label_rows.append(
            {
                "experiment_name": experiment_name,
                "experiment_label": experiment_label,
                "evaluation_kind": "label_prediction_on_weak",
                "model_name": model_name,
                "model_display_name": model_display_name(model_name),
                "model_slug": model_slug,
                "label": label,
                "support": support,
                "predicted": predicted,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "weak_matched_total": weak_matched_total,
                "weak_match_coverage_over_gold": base_eval.safe_ratio(weak_matched_total, gold_total),
            }
        )

    micro_precision = base_eval.safe_ratio(micro_tp, micro_tp + micro_fp)
    micro_recall = base_eval.safe_ratio(micro_tp, micro_tp + micro_fn)
    micro_f1 = base_eval.safe_f1(micro_precision, micro_recall)
    macro_precision = macro_precision_sum / len(LABELS)
    macro_recall = macro_recall_sum / len(LABELS)
    macro_f1 = macro_f1_sum / len(LABELS)

    for (gold_label, predicted_bucket), count in sorted(confusion_counter.items()):
        confusion_rows.append(
            {
                "experiment_name": experiment_name,
                "experiment_label": experiment_label,
                "evaluation_kind": "label_prediction_on_weak",
                "model_name": model_name,
                "model_display_name": model_display_name(model_name),
                "model_slug": model_slug,
                "gold_label": gold_label,
                "predicted_bucket": predicted_bucket,
                "count": count,
            }
        )

    summary_row = {
        "experiment_name": experiment_name,
        "experiment_label": experiment_label,
        "evaluation_kind": "label_prediction_on_weak",
        "model_name": model_name,
        "model_display_name": model_display_name(model_name),
        "model_slug": model_slug,
        "micro_precision": micro_precision,
        "micro_recall": micro_recall,
        "micro_f1": micro_f1,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "micro_tp": micro_tp,
        "micro_fp": micro_fp,
        "micro_fn": micro_fn,
        "weak_matched_total": weak_matched_total,
        "gold_total": gold_total,
        "weak_match_coverage_over_gold": base_eval.safe_ratio(weak_matched_total, gold_total),
    }

    return {
        "summary_row": summary_row,
        "per_label_rows": per_label_rows,
        "details_rows": details_rows,
        "confusion_rows": confusion_rows,
    }


def build_combined_tables(
    label_per_label_rows: list[dict[str, object]],
    label_summary_rows: list[dict[str, object]],
) -> None:
    base_per_label_rows = read_csv_rows(TABLES_DIR / "per_label_metrics.csv")
    base_summary_rows = read_csv_rows(TABLES_DIR / "summary_metrics.csv")

    combined_per_label = [
        row
        for row in base_per_label_rows
        if row["evaluation_kind"] in {"span_weak", "span_strict"}
    ]
    combined_summary = [
        row
        for row in base_summary_rows
        if row["evaluation_kind"] in {"span_weak", "span_strict"}
    ]

    combined_per_label.extend(label_per_label_rows)
    combined_summary.extend(label_summary_rows)

    write_csv(SUB_TABLES_DIR / "per_label_metrics.csv", combined_per_label)
    write_csv(SUB_TABLES_DIR / "summary_metrics.csv", combined_summary)
    write_csv(SUB_TABLES_DIR / "label_prediction_on_weak_per_label_metrics.csv", label_per_label_rows)
    write_csv(SUB_TABLES_DIR / "label_prediction_on_weak_summary_metrics.csv", label_summary_rows)


def main() -> None:
    ensure_sub_eval_dirs()
    gold_rows = base_eval.load_gold_annotations(base_eval.GOLD_PATH)
    gold_by_sentence = base_eval.build_gold_by_sentence(gold_rows)
    gold_total = len(gold_rows)

    print(
        f"[WEAK LABEL EVAL START] gold_rows={gold_total} gold_sentences={len(gold_by_sentence)} output={SUB_EVAL_DIR}",
        flush=True,
    )

    label_per_label_rows: list[dict[str, object]] = []
    label_summary_rows: list[dict[str, object]] = []
    label_confusion_rows: list[dict[str, object]] = []

    for experiment in EXPERIMENTS:
        experiment_name = experiment["experiment_name"]
        experiment_label = experiment["experiment_label"]
        prediction_dir = base_eval.resolve_prediction_dir(Path(experiment["prediction_root"]))
        prediction_files = base_eval.discover_prediction_files(prediction_dir)

        if not prediction_files:
            raise FileNotFoundError(f"No prediction files found in {prediction_dir}")

        print(
            f"[WEAK LABEL EVAL EXPERIMENT] name={experiment_name} models={len(prediction_files)}",
            flush=True,
        )

        for prediction_file in prediction_files:
            model_name = base_eval.infer_model_name(prediction_file)
            model_slug = slugify_model_name(model_name)
            print(
                f"[WEAK LABEL EVAL MODEL START] experiment={experiment_name} file={prediction_file.name}",
                flush=True,
            )

            label_result = evaluate_label_prediction_on_weak_matches(
                experiment_name=experiment_name,
                experiment_label=experiment_label,
                prediction_file=prediction_file,
                gold_by_sentence=gold_by_sentence,
                gold_total=gold_total,
            )

            label_per_label_rows.extend(label_result["per_label_rows"])
            label_summary_rows.append(label_result["summary_row"])
            label_confusion_rows.extend(label_result["confusion_rows"])

            model_dir = SUB_RESULTS_DIR / experiment_name / model_slug
            model_dir.mkdir(parents=True, exist_ok=True)
            write_csv(model_dir / "label_prediction_on_weak_rows.csv", label_result["details_rows"])
            write_csv(model_dir / "label_prediction_on_weak_confusion.csv", label_result["confusion_rows"])

            summary_row = label_result["summary_row"]
            print(
                f"[WEAK LABEL EVAL MODEL DONE] model={model_name} "
                f"coverage={summary_row['weak_match_coverage_over_gold']:.4f} "
                f"label_f1={summary_row['micro_f1']:.4f}",
                flush=True,
            )

    build_combined_tables(label_per_label_rows, label_summary_rows)
    write_csv(SUB_TABLES_DIR / "label_prediction_on_weak_confusion.csv", label_confusion_rows)
    write_text(
        SUB_DOCS_DIR / "00_execution_note.md",
        (
            "# Execution Note\n\n"
            "This folder contains the revised label evaluation conditioned on weak span matches.\n\n"
            "Weak span matching is used only to decide which gold annotations enter the label evaluation subset.\n"
        ),
    )
    print(f"[WEAK LABEL EVAL DONE] tables={SUB_TABLES_DIR} results={SUB_RESULTS_DIR}", flush=True)


if __name__ == "__main__":
    main()
