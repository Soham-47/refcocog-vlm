"""Run a zero-shot Qwen3-VL referring-expression grounding baseline."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from refcocog.grounding import iou, normalized_to_pixels, parse_box, source_bbox_to_pixels


def phrase_from(row: dict) -> str:
    sentences = row.get("sentences", [])
    if sentences:
        sentence = sentences[0]
        return sentence.get("sent", "") if isinstance(sentence, dict) else str(sentence)
    return str(row.get("phrase", row.get("sent", "")))


def run_self_check() -> None:
    assert parse_box("[[10, 20, 500, 600]]") == [10.0, 20.0, 500.0, 600.0]
    assert parse_box("[447, 64, 668, 338]") == [447.0, 64.0, 668.0, 338.0]
    assert parse_box("<|box_start|>(10,20),(500,600)<|box_end|>") == [10.0, 20.0, 500.0, 600.0]
    assert parse_box("no box") is None
    assert iou([0, 0, 10, 10], [0, 0, 10, 10]) == 1.0
    assert iou([0, 0, 10, 10], [10, 0, 20, 10]) == 0.0
    print("Parser and IoU checks passed")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--model", default="Qwen/Qwen3-VL-8B-Instruct")
    parser.add_argument("--adapter", type=Path, help="Optional ms-swift/PEFT LoRA checkpoint to evaluate.")
    parser.add_argument("--predictions", type=Path, default=Path("outputs/baseline_val50_predictions.jsonl"))
    parser.add_argument("--summary", type=Path, default=Path("outputs/baseline_val50_metrics.json"))
    parser.add_argument("--self-check", action="store_true", help="Check box parsing and IoU without loading a model.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.self_check:
        run_self_check()
        return
    if not args.manifest:
        raise SystemExit("--manifest is required unless --self-check is used")

    import torch
    from PIL import Image
    from transformers import AutoModelForImageTextToText, AutoProcessor

    model = AutoModelForImageTextToText.from_pretrained(
        args.model, dtype="auto", device_map="auto"
    )
    if args.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.adapter)
    processor = AutoProcessor.from_pretrained(args.model)
    args.predictions.parent.mkdir(parents=True, exist_ok=True)
    args.summary.parent.mkdir(parents=True, exist_ok=True)

    results = []
    with args.manifest.open(encoding="utf-8") as manifest, args.predictions.open("w", encoding="utf-8") as output:
        for index, line in enumerate(manifest, start=1):
            row = json.loads(line)
            phrase = phrase_from(row)
            raw = ""
            pred_1000 = pred_pixels = None
            error = None
            width = height = None
            try:
                with Image.open(row["image_path"]) as image_file:
                    image = image_file.convert("RGB")
                width, height = image.size
                prompt = (
                    "Locate the single object described by the phrase in this image. "
                    "Return only its bounding box as [[x1, y1, x2, y2]], using "
                    "coordinates on a 0-to-1000 scale relative to the image width and height. "
                    f"Phrase: {phrase}"
                )
                messages = [{"role": "user", "content": [
                    {"type": "image", "image": image}, {"type": "text", "text": prompt}
                ]}]
                inputs = processor.apply_chat_template(
                    messages, tokenize=True, add_generation_prompt=True,
                    return_dict=True, return_tensors="pt"
                ).to(model.device)
                with torch.inference_mode():
                    generated = model.generate(**inputs, max_new_tokens=64, do_sample=False)
                generated = generated[:, inputs.input_ids.shape[1]:]
                raw = processor.batch_decode(
                    generated, skip_special_tokens=True, clean_up_tokenization_spaces=False
                )[0].strip()
                pred_1000 = parse_box(raw)
                if pred_1000 is not None:
                    pred_pixels = normalized_to_pixels(pred_1000, width, height)
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"

            gt = [float(value) for value in row["bbox"]]
            if width is not None and height is not None:
                gt = source_bbox_to_pixels(
                    gt, int(row.get("source_width", width)), int(row.get("source_height", height)),
                    width, height,
                )
            score = iou(pred_pixels, gt) if pred_pixels is not None else 0.0
            result = {
                "index": index, "ref_id": row.get("ref_id"), "phrase": phrase,
                "image_path": row.get("image_path"), "gt_bbox_pixels": gt,
                "raw_prediction": raw, "pred_bbox_1000": pred_1000,
                "pred_bbox_pixels": pred_pixels, "iou": score,
                "parse_ok": pred_pixels is not None, "error": error,
            }
            output.write(json.dumps(result, ensure_ascii=False) + "\n")
            output.flush()
            results.append(result)
            print(f"[{index}] IoU={score:.3f} parse_ok={pred_pixels is not None}")

    count = len(results)
    summary = {
        "model": args.model,
        "manifest": str(args.manifest),
        "num_examples": count,
        "parse_failures": sum(not item["parse_ok"] for item in results),
        "mean_iou": sum(item["iou"] for item in results) / count if count else 0.0,
        "accuracy_iou_0.5": sum(item["iou"] >= 0.5 for item in results) / count if count else 0.0,
        "accuracy_iou_0.75": sum(item["iou"] >= 0.75 for item in results) / count if count else 0.0,
    }
    args.summary.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Per-example predictions: {args.predictions}")
    print(f"Metrics summary: {args.summary}")


if __name__ == "__main__":
    main()
