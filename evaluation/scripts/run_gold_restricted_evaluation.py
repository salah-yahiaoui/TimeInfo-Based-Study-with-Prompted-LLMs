from __future__ import annotations

import csv
import json
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from analysis_utils import EXPERIMENTS, GOLD_PATH, RESULTS_DIR, TABLES_DIR, ensure_output_dirs, write_csv
from timetank_annotation.config import ALLOWED_LABELS
from timetank_annotation.logging_utils import slugify_model_name, to_jsonable

MISSING_BUCKET = "__MISSING__"
MULTIPLE_BUCKET = "__MULTIPLE__"
INVALID_BUCKET = "__INVALID__"

SentenceKey = tuple[str, str]


@dataclass(frozen=True, slots=True)
class EvalAnnotation:
    text: str
    label: str


@dataclass(slots=True)
class SentenceRecord:
    article_id: str
    id_phrase: str
    article_title: str
    phrase: str
    annotations: list[EvalAnnotation]


def normalize_text(value: str | None) -> str:
    if value is None:
        return ""
    return " ".join(value.split())


def normalize_label(value: str | None) -> str:
    if value is None:
        return ""
    return value.strip()


def resolve_prediction_dir(path: Path) -> Path:
    if path.is_dir() and (path / "annotations").is_dir():
        return path / "annotations"
    return path


def discover_prediction_files(predictions_dir: Path) -> list[Path]:
    return sorted(
        path for path in predictions_dir.glob("*.csv")
        if path.suffix.lower() == ".csv" and path.name != "TimeTankSample.csv"
    )


def infer_model_name(prediction_file: Path) -> str:
    summary_path = prediction_file.parent.parent / "summaries" / f"{prediction_file.stem}.summary.json"
    if summary_path.exists():
        try:
            data = json.loads(summary_path.read_text(encoding="utf-8"))
            model_name = data.get("model_name")
            if isinstance(model_name, str) and model_name:
                return model_name
        except json.JSONDecodeError:
            pass
    return prediction_file.stem


def load_prediction_records(prediction_path: Path) -> dict[SentenceKey, SentenceRecord]:
    with prediction_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        grouped: dict[SentenceKey, SentenceRecord] = {}
        for row in reader:
            article_id = row.get("article_id", "")
            id_phrase = row.get("id_phrase", "")
            key = (article_id, id_phrase)
            record = grouped.get(key)
            if record is None:
                record = SentenceRecord(
                    article_id=article_id,
                    id_phrase=id_phrase,
                    article_title=row.get("article_title", ""),
                    phrase=row.get("phrase", ""),
                    annotations=[],
                )
                grouped[key] = record

            annotation = _row_to_annotation(row)
            if annotation is not None and annotation not in record.annotations:
                record.annotations.append(annotation)

    return grouped


def _row_to_annotation(row: dict[str, str]) -> EvalAnnotation | None:
    text = normalize_text(row.get("temporal_expression"))
    label = normalize_label(row.get("interval"))
    if not text or not label:
        return None
    return EvalAnnotation(text=text, label=label)


@dataclass(frozen=True, slots=True)
class GoldInstance:
    row_index: int
    article_id: str
    article_title: str
    id_phrase: str
    phrase: str
    temporal_expression: str
    interval: str

    @property
    def key(self) -> SentenceKey:
        return (self.article_id, self.id_phrase)


@dataclass(slots=True)
class GoldRestrictedResult:
    experiment_name: str
    experiment_label: str
    model_name: str
    model_slug: str
    prediction_file: Path
    total_gold_instances: int
    total_gold_sentences: int
    unknown_prediction_sentence_count: int
    ignored_extra_predictions_total: int
    missing_prediction_count: int
    multiple_prediction_count: int
    invalid_prediction_bucket_count: int
    wrong_allowed_label_count: int
    correct_prediction_count: int
    sentence_exact_count: int
    micro_tp: int
    micro_fp: int
    micro_fn: int
    micro_precision: float
    micro_recall: float
    micro_f1: float
    macro_precision: float
    macro_recall: float
    macro_f1: float
    weighted_precision: float
    weighted_recall: float
    weighted_f1: float
    accuracy: float
    sentence_exact_rate: float
    details_rows: list[dict[str, object]]
    sentence_rows: list[dict[str, object]]
    per_label_rows: list[dict[str, object]]
    confusion_rows: list[dict[str, object]]
    extra_rows: list[dict[str, object]]

    def summary_dict(self) -> dict[str, object]:
        return {
            "experiment_name": self.experiment_name,
            "experiment_label": self.experiment_label,
            "model_name": self.model_name,
            "model_slug": self.model_slug,
            "prediction_file": str(self.prediction_file),
            "total_gold_instances": self.total_gold_instances,
            "total_gold_sentences": self.total_gold_sentences,
            "unknown_prediction_sentence_count": self.unknown_prediction_sentence_count,
            "ignored_extra_predictions_total": self.ignored_extra_predictions_total,
            "missing_prediction_count": self.missing_prediction_count,
            "multiple_prediction_count": self.multiple_prediction_count,
            "invalid_prediction_bucket_count": self.invalid_prediction_bucket_count,
            "wrong_allowed_label_count": self.wrong_allowed_label_count,
            "correct_prediction_count": self.correct_prediction_count,
            "sentence_exact_count": self.sentence_exact_count,
            "micro_tp": self.micro_tp,
            "micro_fp": self.micro_fp,
            "micro_fn": self.micro_fn,
            "micro_precision": self.micro_precision,
            "micro_recall": self.micro_recall,
            "micro_f1": self.micro_f1,
            "macro_precision": self.macro_precision,
            "macro_recall": self.macro_recall,
            "macro_f1": self.macro_f1,
            "weighted_precision": self.weighted_precision,
            "weighted_recall": self.weighted_recall,
            "weighted_f1": self.weighted_f1,
            "accuracy": self.accuracy,
            "sentence_exact_rate": self.sentence_exact_rate,
            "per_label": self.per_label_rows,
        }


def load_gold_instances(gold_path: Path) -> list[GoldInstance]:
    with gold_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows: list[GoldInstance] = []
        for idx, row in enumerate(reader, start=1):
            rows.append(
                GoldInstance(
                    row_index=idx,
                    article_id=row.get("article_id", ""),
                    article_title=row.get("article_title", ""),
                    id_phrase=row.get("id_phrase", ""),
                    phrase=row.get("phrase", ""),
                    temporal_expression=normalize_text(row.get("temporal_expression")),
                    interval=normalize_label(row.get("interval")),
                )
            )
    return rows


def build_gold_span_lookup(gold_instances: list[GoldInstance]) -> dict[SentenceKey, set[str]]:
    lookup: dict[SentenceKey, set[str]] = defaultdict(set)
    for item in gold_instances:
        lookup[item.key].add(item.temporal_expression)
    return lookup


def build_gold_sentence_groups(gold_instances: list[GoldInstance]) -> dict[SentenceKey, list[GoldInstance]]:
    grouped: dict[SentenceKey, list[GoldInstance]] = defaultdict(list)
    for item in gold_instances:
        grouped[item.key].append(item)
    return grouped


def evaluate_prediction_file(
    *,
    experiment_name: str,
    experiment_label: str,
    gold_instances: list[GoldInstance],
    gold_spans_by_sentence: dict[SentenceKey, set[str]],
    gold_groups: dict[SentenceKey, list[GoldInstance]],
    prediction_file: Path,
) -> GoldRestrictedResult:
    prediction_records = load_prediction_records(prediction_file)
    model_name = infer_model_name(prediction_file)
    model_slug = slugify_model_name(model_name)

    extra_rows: list[dict[str, object]] = []
    unknown_prediction_sentence_count = 0
    invalid_predicted_label_total = 0

    for key, pred_sentence in prediction_records.items():
        gold_spans = gold_spans_by_sentence.get(key)
        if gold_spans is None:
            unknown_prediction_sentence_count += 1
            for ann in pred_sentence.annotations:
                extra_rows.append(
                    {
                        "experiment_name": experiment_name,
                        "experiment_label": experiment_label,
                        "model_name": model_name,
                        "model_slug": model_slug,
                        "article_id": pred_sentence.article_id,
                        "id_phrase": pred_sentence.id_phrase,
                        "article_title": pred_sentence.article_title,
                        "phrase": pred_sentence.phrase,
                        "predicted_temporal_expression": ann.text,
                        "predicted_interval": ann.label,
                        "extra_type": "unknown_sentence",
                    }
                )
                if ann.label not in ALLOWED_LABELS:
                    invalid_predicted_label_total += 1
            continue

        for ann in pred_sentence.annotations:
            if ann.label not in ALLOWED_LABELS:
                invalid_predicted_label_total += 1
            if ann.text not in gold_spans:
                extra_rows.append(
                    {
                        "experiment_name": experiment_name,
                        "experiment_label": experiment_label,
                        "model_name": model_name,
                        "model_slug": model_slug,
                        "article_id": pred_sentence.article_id,
                        "id_phrase": pred_sentence.id_phrase,
                        "article_title": pred_sentence.article_title,
                        "phrase": pred_sentence.phrase,
                        "predicted_temporal_expression": ann.text,
                        "predicted_interval": ann.label,
                        "extra_type": "span_not_in_gold",
                    }
                )

    predicted_buckets_counter: Counter[str] = Counter()
    tp_by_label: Counter[str] = Counter()
    support_by_label: Counter[str] = Counter()
    confusion_counter: Counter[tuple[str, str]] = Counter()

    details_rows: list[dict[str, object]] = []
    sentence_correct_counter: Counter[SentenceKey] = Counter()
    sentence_total_counter: Counter[SentenceKey] = Counter()

    missing_prediction_count = 0
    multiple_prediction_count = 0
    invalid_prediction_bucket_count = 0
    wrong_allowed_label_count = 0
    correct_prediction_count = 0

    for item in gold_instances:
        support_by_label[item.interval] += 1
        sentence_total_counter[item.key] += 1

        pred_sentence = prediction_records.get(item.key)
        all_labels_for_same_span: list[str] = []
        valid_labels_for_same_span: list[str] = []
        invalid_labels_for_same_span: list[str] = []

        if pred_sentence is not None:
            for ann in pred_sentence.annotations:
                if ann.text != item.temporal_expression:
                    continue
                all_labels_for_same_span.append(ann.label)
                if ann.label in ALLOWED_LABELS:
                    valid_labels_for_same_span.append(ann.label)
                else:
                    invalid_labels_for_same_span.append(ann.label)

        unique_valid_labels = sorted(set(valid_labels_for_same_span))
        unique_invalid_labels = sorted(set(invalid_labels_for_same_span))

        if not unique_valid_labels and not unique_invalid_labels:
            predicted_bucket = MISSING_BUCKET
            row_status = "missing_span"
            missing_prediction_count += 1
        elif len(unique_valid_labels) == 1 and not unique_invalid_labels:
            predicted_bucket = unique_valid_labels[0]
            row_status = "correct" if predicted_bucket == item.interval else "wrong_label"
            if predicted_bucket == item.interval:
                correct_prediction_count += 1
                sentence_correct_counter[item.key] += 1
            else:
                wrong_allowed_label_count += 1
        elif not unique_valid_labels and unique_invalid_labels:
            predicted_bucket = INVALID_BUCKET
            row_status = "invalid_label_only"
            invalid_prediction_bucket_count += 1
        else:
            predicted_bucket = MULTIPLE_BUCKET
            row_status = "multiple_labels"
            multiple_prediction_count += 1

        if predicted_bucket in ALLOWED_LABELS:
            predicted_buckets_counter[predicted_bucket] += 1
            if predicted_bucket == item.interval:
                tp_by_label[item.interval] += 1

        confusion_counter[(item.interval, predicted_bucket)] += 1

        details_rows.append(
            {
                "experiment_name": experiment_name,
                "experiment_label": experiment_label,
                "model_name": model_name,
                "model_slug": model_slug,
                "gold_row_index": item.row_index,
                "article_id": item.article_id,
                "id_phrase": item.id_phrase,
                "article_title": item.article_title,
                "phrase": item.phrase,
                "gold_temporal_expression": item.temporal_expression,
                "gold_interval": item.interval,
                "predicted_interval_bucket": predicted_bucket,
                "predicted_intervals_for_same_gold_span": "|".join(sorted(set(all_labels_for_same_span))),
                "row_status": row_status,
                "counted_as_correct": int(predicted_bucket == item.interval),
            }
        )

    per_label_rows: list[dict[str, object]] = []
    micro_tp = 0
    micro_fp = 0
    micro_fn = 0
    precision_values: list[float] = []
    recall_values: list[float] = []
    f1_values: list[float] = []
    weighted_precision_sum = 0.0
    weighted_recall_sum = 0.0
    weighted_f1_sum = 0.0

    for label in ALLOWED_LABELS:
        support = support_by_label[label]
        tp = tp_by_label[label]
        predicted_count = predicted_buckets_counter[label]
        fp = predicted_count - tp
        fn = support - tp
        precision = _safe_ratio(tp, tp + fp)
        recall = _safe_ratio(tp, tp + fn)
        f1 = _safe_f1(precision, recall)

        micro_tp += tp
        micro_fp += fp
        micro_fn += fn
        precision_values.append(precision)
        recall_values.append(recall)
        f1_values.append(f1)
        weighted_precision_sum += precision * support
        weighted_recall_sum += recall * support
        weighted_f1_sum += f1 * support

        per_label_rows.append(
            {
                "experiment_name": experiment_name,
                "experiment_label": experiment_label,
                "model_name": model_name,
                "model_slug": model_slug,
                "label": label,
                "support": support,
                "predicted_count": predicted_count,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }
        )

    total_gold_instances = len(gold_instances)
    total_gold_sentences = len(gold_groups)
    sentence_exact_count = sum(
        1 for key, items in gold_groups.items() if sentence_correct_counter[key] == len(items)
    )
    micro_precision = _safe_ratio(micro_tp, micro_tp + micro_fp)
    micro_recall = _safe_ratio(micro_tp, micro_tp + micro_fn)
    micro_f1 = _safe_f1(micro_precision, micro_recall)
    macro_precision = sum(precision_values) / len(ALLOWED_LABELS)
    macro_recall = sum(recall_values) / len(ALLOWED_LABELS)
    macro_f1 = sum(f1_values) / len(ALLOWED_LABELS)
    weighted_precision = weighted_precision_sum / total_gold_instances
    weighted_recall = weighted_recall_sum / total_gold_instances
    weighted_f1 = weighted_f1_sum / total_gold_instances
    accuracy = _safe_ratio(correct_prediction_count, total_gold_instances)
    sentence_exact_rate = _safe_ratio(sentence_exact_count, total_gold_sentences)

    sentence_rows: list[dict[str, object]] = []
    for key, items in gold_groups.items():
        gold_sentence = items[0]
        sentence_rows.append(
            {
                "experiment_name": experiment_name,
                "experiment_label": experiment_label,
                "model_name": model_name,
                "model_slug": model_slug,
                "article_id": gold_sentence.article_id,
                "id_phrase": gold_sentence.id_phrase,
                "article_title": gold_sentence.article_title,
                "phrase": gold_sentence.phrase,
                "gold_instance_count": len(items),
                "correct_instance_count": sentence_correct_counter[key],
                "sentence_exact_on_gold": int(sentence_correct_counter[key] == len(items)),
            }
        )

    confusion_rows = [
        {
            "experiment_name": experiment_name,
            "experiment_label": experiment_label,
            "model_name": model_name,
            "model_slug": model_slug,
            "gold_interval": gold_label,
            "predicted_bucket": predicted_bucket,
            "count": count,
        }
        for (gold_label, predicted_bucket), count in sorted(confusion_counter.items())
    ]

    return GoldRestrictedResult(
        experiment_name=experiment_name,
        experiment_label=experiment_label,
        model_name=model_name,
        model_slug=model_slug,
        prediction_file=prediction_file,
        total_gold_instances=total_gold_instances,
        total_gold_sentences=total_gold_sentences,
        unknown_prediction_sentence_count=unknown_prediction_sentence_count,
        ignored_extra_predictions_total=len(extra_rows),
        missing_prediction_count=missing_prediction_count,
        multiple_prediction_count=multiple_prediction_count,
        invalid_prediction_bucket_count=invalid_prediction_bucket_count,
        wrong_allowed_label_count=wrong_allowed_label_count,
        correct_prediction_count=correct_prediction_count,
        sentence_exact_count=sentence_exact_count,
        micro_tp=micro_tp,
        micro_fp=micro_fp,
        micro_fn=micro_fn,
        micro_precision=micro_precision,
        micro_recall=micro_recall,
        micro_f1=micro_f1,
        macro_precision=macro_precision,
        macro_recall=macro_recall,
        macro_f1=macro_f1,
        weighted_precision=weighted_precision,
        weighted_recall=weighted_recall,
        weighted_f1=weighted_f1,
        accuracy=accuracy,
        sentence_exact_rate=sentence_exact_rate,
        details_rows=details_rows,
        sentence_rows=sentence_rows,
        per_label_rows=per_label_rows,
        confusion_rows=confusion_rows,
        extra_rows=extra_rows,
    )


def write_model_outputs(base_dir: Path, result: GoldRestrictedResult) -> None:
    base_dir.mkdir(parents=True, exist_ok=True)
    model_dir = base_dir / result.model_slug
    model_dir.mkdir(parents=True, exist_ok=True)

    (model_dir / f"{result.model_slug}.summary.json").write_text(
        json.dumps(to_jsonable(result.summary_dict()), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_csv(model_dir / f"{result.model_slug}.row_level.csv", result.details_rows)
    write_csv(model_dir / f"{result.model_slug}.sentence_level.csv", result.sentence_rows)
    write_csv(model_dir / f"{result.model_slug}.per_label.csv", result.per_label_rows)
    write_csv(model_dir / f"{result.model_slug}.confusion.csv", result.confusion_rows)
    write_csv(model_dir / f"{result.model_slug}.ignored_extra_predictions.csv", result.extra_rows)


def write_aggregate_outputs(results: list[GoldRestrictedResult]) -> None:
    model_rows = []
    per_label_rows = []
    confusion_rows = []
    extra_summary_rows = []
    all_extra_rows = []
    error_rows = []

    for result in results:
        model_rows.append(
            {
                "experiment_name": result.experiment_name,
                "experiment_label": result.experiment_label,
                "model_name": result.model_name,
                "model_slug": result.model_slug,
                "prediction_file": str(result.prediction_file),
                "total_gold_instances": result.total_gold_instances,
                "total_gold_sentences": result.total_gold_sentences,
                "unknown_prediction_sentence_count": result.unknown_prediction_sentence_count,
                "ignored_extra_predictions_total": result.ignored_extra_predictions_total,
                "missing_prediction_count": result.missing_prediction_count,
                "multiple_prediction_count": result.multiple_prediction_count,
                "invalid_prediction_bucket_count": result.invalid_prediction_bucket_count,
                "wrong_allowed_label_count": result.wrong_allowed_label_count,
                "correct_prediction_count": result.correct_prediction_count,
                "sentence_exact_count": result.sentence_exact_count,
                "micro_tp": result.micro_tp,
                "micro_fp": result.micro_fp,
                "micro_fn": result.micro_fn,
                "micro_precision": result.micro_precision,
                "micro_recall": result.micro_recall,
                "micro_f1": result.micro_f1,
                "macro_precision": result.macro_precision,
                "macro_recall": result.macro_recall,
                "macro_f1": result.macro_f1,
                "weighted_precision": result.weighted_precision,
                "weighted_recall": result.weighted_recall,
                "weighted_f1": result.weighted_f1,
                "accuracy": result.accuracy,
                "sentence_exact_rate": result.sentence_exact_rate,
            }
        )
        per_label_rows.extend(result.per_label_rows)
        confusion_rows.extend(result.confusion_rows)
        all_extra_rows.extend(result.extra_rows)
        error_rows.extend(row for row in result.details_rows if not row["counted_as_correct"])

        extra_type_counter = Counter(str(row["extra_type"]) for row in result.extra_rows)
        extra_summary_rows.append(
            {
                "experiment_name": result.experiment_name,
                "experiment_label": result.experiment_label,
                "model_name": result.model_name,
                "model_slug": result.model_slug,
                "ignored_extra_predictions_total": result.ignored_extra_predictions_total,
                "extra_span_not_in_gold": extra_type_counter.get("span_not_in_gold", 0),
                "unknown_sentence_predictions": extra_type_counter.get("unknown_sentence", 0),
            }
        )

    delta_rows = build_delta_rows(model_rows)

    write_csv(TABLES_DIR / "model_metrics.csv", model_rows)
    write_csv(TABLES_DIR / "per_label_metrics.csv", per_label_rows)
    write_csv(TABLES_DIR / "confusion_matrix.csv", confusion_rows)
    write_csv(TABLES_DIR / "ignored_extra_predictions_summary.csv", extra_summary_rows)
    write_csv(TABLES_DIR / "experiment_deltas.csv", delta_rows)
    write_csv(RESULTS_DIR / "ignored_extra_predictions_all.csv", all_extra_rows)
    write_csv(RESULTS_DIR / "row_level_errors_all.csv", error_rows)


def build_delta_rows(model_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    by_model: dict[str, dict[str, dict[str, object]]] = defaultdict(dict)
    for row in model_rows:
        by_model[str(row["model_name"])][str(row["experiment_label"])] = row

    deltas: list[dict[str, object]] = []
    for model_name, rows in by_model.items():
        single_pass = rows.get("Single-pass")
        multi_pass = rows.get("Multi-pass")
        if single_pass is None or multi_pass is None:
            continue
        deltas.append(
            {
                "model_name": model_name,
                "delta_micro_precision": _to_float(multi_pass["micro_precision"]) - _to_float(single_pass["micro_precision"]),
                "delta_micro_recall": _to_float(multi_pass["micro_recall"]) - _to_float(single_pass["micro_recall"]),
                "delta_micro_f1": _to_float(multi_pass["micro_f1"]) - _to_float(single_pass["micro_f1"]),
                "delta_macro_f1": _to_float(multi_pass["macro_f1"]) - _to_float(single_pass["macro_f1"]),
                "delta_weighted_f1": _to_float(multi_pass["weighted_f1"]) - _to_float(single_pass["weighted_f1"]),
                "delta_accuracy": _to_float(multi_pass["accuracy"]) - _to_float(single_pass["accuracy"]),
                "delta_sentence_exact_rate": _to_float(multi_pass["sentence_exact_rate"]) - _to_float(single_pass["sentence_exact_rate"]),
                "delta_ignored_extra_predictions_total": int(multi_pass["ignored_extra_predictions_total"]) - int(single_pass["ignored_extra_predictions_total"]),
                "delta_missing_prediction_count": int(multi_pass["missing_prediction_count"]) - int(single_pass["missing_prediction_count"]),
                "delta_wrong_allowed_label_count": int(multi_pass["wrong_allowed_label_count"]) - int(single_pass["wrong_allowed_label_count"]),
            }
        )
    return deltas


def _safe_ratio(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator


def _safe_f1(precision: float, recall: float) -> float:
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def _to_float(value: object) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        return float(value)
    return 0.0


def main() -> None:
    ensure_output_dirs()
    gold_instances = load_gold_instances(GOLD_PATH)
    gold_spans_by_sentence = build_gold_span_lookup(gold_instances)
    gold_groups = build_gold_sentence_groups(gold_instances)

    print(
        f"[NEW EVAL START] gold_instances={len(gold_instances)} gold_sentences={len(gold_groups)} output={RESULTS_DIR}",
        flush=True,
    )

    all_results: list[GoldRestrictedResult] = []

    for experiment in EXPERIMENTS:
        experiment_name = experiment["experiment_name"]
        experiment_label = experiment["experiment_label"]
        prediction_dir = resolve_prediction_dir(Path(experiment["prediction_root"]))
        prediction_files = discover_prediction_files(prediction_dir)

        if not prediction_files:
            raise FileNotFoundError(f"No prediction CSV files found in {prediction_dir}")

        experiment_output_dir = RESULTS_DIR / experiment_name
        print(
            f"[NEW EVAL EXPERIMENT] name={experiment_name} models={len(prediction_files)}",
            flush=True,
        )

        for prediction_file in prediction_files:
            print(f"[NEW EVAL MODEL START] experiment={experiment_name} file={prediction_file.name}", flush=True)
            result = evaluate_prediction_file(
                experiment_name=experiment_name,
                experiment_label=experiment_label,
                gold_instances=gold_instances,
                gold_spans_by_sentence=gold_spans_by_sentence,
                gold_groups=gold_groups,
                prediction_file=prediction_file,
            )
            write_model_outputs(experiment_output_dir, result)
            all_results.append(result)
            print(
                f"[NEW EVAL MODEL DONE] model={result.model_name} "
                f"precision={result.micro_precision:.4f} "
                f"recall={result.micro_recall:.4f} "
                f"f1={result.micro_f1:.4f} "
                f"accuracy={result.accuracy:.4f}",
                flush=True,
            )

    write_aggregate_outputs(all_results)
    print(f"[NEW EVAL DONE] tables={TABLES_DIR} results={RESULTS_DIR}", flush=True)


if __name__ == "__main__":
    main()
