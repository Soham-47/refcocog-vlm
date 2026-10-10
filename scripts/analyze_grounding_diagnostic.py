"""Analyze grounding predictions from a paired model/split diagnostic."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_prediction_arg(value: str) -> tuple[str, Path]:
    label, separator, path = value.partition("=")
    if not separator or not label or not path:
        raise argparse.ArgumentTypeError("prediction must be LABEL=PATH")
    return label, Path(path)


def load_predictions(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            row["_line_number"] = line_number
            rows.append(row)
    return rows


def row_key(row: dict[str, Any]) -> str:
    ref_id = row.get("ref_id")
    return f"ref:{ref_id}" if ref_id not in (None, "") else f"index:{row.get('index')}"


def box_area(box: list[float] | None) -> float:
    if not box:
        return 0.0
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def box_center(box: list[float] | None) -> tuple[float, float] | None:
    if not box:
        return None
    return ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)


def metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    count = len(rows)
    ious = [float(row.get("iou", 0.0)) for row in rows]
    parsed = [bool(row.get("parse_ok")) for row in rows]
    area_ratios = []
    center_distances = []
    for row in rows:
        predicted = row.get("pred_bbox_pixels")
        target = row.get("gt_bbox_pixels")
        target_area = box_area(target)
        if predicted and target_area > 0:
            area_ratios.append(box_area(predicted) / target_area)
            pred_center = box_center(predicted)
            target_center = box_center(target)
            if pred_center and target_center:
                center_distances.append(
                    ((pred_center[0] - target_center[0]) ** 2
                     + (pred_center[1] - target_center[1]) ** 2) ** 0.5
                )
    return {
        "num_examples": count,
        "parse_failures": sum(not value for value in parsed),
        "mean_iou": sum(ious) / count if count else 0.0,
        "accuracy_iou_0.5": sum(value >= 0.5 for value in ious) / count if count else 0.0,
        "accuracy_iou_0.75": sum(value >= 0.75 for value in ious) / count if count else 0.0,
        "mean_pred_to_gt_area_ratio": sum(area_ratios) / len(area_ratios) if area_ratios else None,
        "mean_center_distance_pixels": sum(center_distances) / len(center_distances) if center_distances else None,
    }


def comparable_rows(base: list[dict[str, Any]], variant: list[dict[str, Any]]) -> tuple[list[tuple[dict[str, Any], dict[str, Any]]], list[str]]:
    base_by_key = {row_key(row): row for row in base}
    variant_by_key = {row_key(row): row for row in variant}
    reasons = []
    if len(base_by_key) != len(base) or len(variant_by_key) != len(variant):
        reasons.append("duplicate row keys")
    missing = sorted(set(base_by_key) - set(variant_by_key))
    extra = sorted(set(variant_by_key) - set(base_by_key))
    if missing:
        reasons.append(f"missing keys: {len(missing)}")
    if extra:
        reasons.append(f"extra keys: {len(extra)}")
    pairs = []
    for key in sorted(set(base_by_key) & set(variant_by_key)):
        base_row, variant_row = base_by_key[key], variant_by_key[key]
        if base_row.get("phrase") != variant_row.get("phrase"):
            reasons.append(f"phrase mismatch for {key}")
        if base_row.get("gt_bbox_pixels") != variant_row.get("gt_bbox_pixels"):
            reasons.append(f"ground-truth mismatch for {key}")
        pairs.append((base_row, variant_row))
    return pairs, reasons


def comparison(base: list[dict[str, Any]], variant: list[dict[str, Any]]) -> dict[str, Any]:
    pairs, reasons = comparable_rows(base, variant)
    if reasons:
        return {"comparable": False, "reasons": reasons[:20], "paired_examples": len(pairs)}
    deltas = [float(variant_row.get("iou", 0.0)) - float(base_row.get("iou", 0.0)) for base_row, variant_row in pairs]
    return {
        "comparable": True,
        "paired_examples": len(pairs),
        "mean_iou_delta": sum(deltas) / len(deltas) if deltas else 0.0,
        "improved": sum(delta > 1e-9 for delta in deltas),
        "degraded": sum(delta < -1e-9 for delta in deltas),
        "unchanged": sum(abs(delta) <= 1e-9 for delta in deltas),
        "parse_recovered": sum(not base_row.get("parse_ok") and variant_row.get("parse_ok") for base_row, variant_row in pairs),
        "parse_regressed": sum(base_row.get("parse_ok") and not variant_row.get("parse_ok") for base_row, variant_row in pairs),
    }


def model_and_split(label: str) -> tuple[str, str]:
    model, separator, split = label.partition("/")
    return model, split if separator else "default"


def build_report(predictions: list[tuple[str, Path]], base_model: str) -> dict[str, Any]:
    loaded = {label: load_predictions(path) for label, path in predictions}
    report = {"runs": {}, "comparisons": {}}
    for label, rows in loaded.items():
        report["runs"][label] = {"prediction_file": str(dict(predictions)[label]), **metrics(rows)}
    base_by_split = {
        split: label for label in loaded
        for model, split in [model_and_split(label)] if model == base_model
    }
    for label, rows in loaded.items():
        model, split = model_and_split(label)
        if model == base_model:
            continue
        base_label = base_by_split.get(split)
        report["comparisons"][f"{base_label or base_model + '/' + split}_vs_{label}"] = (
            comparison(loaded[base_label], rows)
            if base_label
            else {"comparable": False, "reasons": [f"no base run for split {split!r}"]}
        )
    return report


def self_check() -> None:
    base = [{"ref_id": "a", "phrase": "x", "gt_bbox_pixels": [0, 0, 10, 10], "pred_bbox_pixels": [0, 0, 5, 5], "iou": 0.25, "parse_ok": True}]
    improved = [{"ref_id": "a", "phrase": "x", "gt_bbox_pixels": [0, 0, 10, 10], "pred_bbox_pixels": [0, 0, 10, 10], "iou": 1.0, "parse_ok": True}]
    result = comparison(base, improved)
    assert result["comparable"] and result["improved"] == 1 and result["mean_iou_delta"] == 0.75
    mismatched = [{**improved[0], "gt_bbox_pixels": [1, 1, 10, 10]}]
    assert comparison(base, mismatched)["comparable"] is False
    print("Diagnostic analyzer checks passed")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prediction", action="append", type=parse_prediction_arg, help="LABEL=PATH; use labels such as base/train and sft/val")
    parser.add_argument("--base-model", default="base", help="Model prefix used for paired comparisons")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args()
    if args.self_check:
        self_check()
        return
    if not args.prediction or not args.output:
        parser.error("--prediction and --output are required unless --self-check is used")
    labels = [label for label, _ in args.prediction]
    if len(labels) != len(set(labels)):
        parser.error("prediction labels must be unique")
    if not any(model_and_split(label)[0] == args.base_model for label in labels):
        parser.error(f"base model prefix {args.base_model!r} is missing")
    report = build_report(args.prediction, args.base_model)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Diagnostic report: {args.output}")


if __name__ == "__main__":
    main()
