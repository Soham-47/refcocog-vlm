"""Convert RefCOCOg image manifests into ms-swift GRPO and SFT JSONL."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from refcocog.grounding import source_bbox_to_pixels


PROMPT = (
    "Locate the single object described by the phrase in this image. "
    "Return only its bounding box as [[x1, y1, x2, y2]], using coordinates "
    "on a 0-to-1000 scale relative to the image width and height. Phrase: "
)


def phrase_from(row: dict) -> str:
    sentence = row["sentences"][0]
    return sentence.get("sent", "") if isinstance(sentence, dict) else str(sentence)


def convert(row: dict) -> tuple[dict, dict]:
    image_path = Path(row["image_path"])
    with Image.open(image_path) as image:
        width, height = image.size
    source_width = int(row["source_width"])
    source_height = int(row["source_height"])
    bbox = [float(value) for value in row["bbox"]]
    if len(bbox) != 4 or not (0 <= bbox[0] < bbox[2] <= source_width and 0 <= bbox[1] < bbox[3] <= source_height):
        raise ValueError(f"Invalid pixel box for ref_id={row.get('ref_id')}: {bbox}")
    target_bbox = source_bbox_to_pixels(bbox, source_width, source_height, width, height)

    prompt = PROMPT + phrase_from(row)
    user = {"role": "user", "content": "<image>\n" + prompt}
    metadata = {
        "images": [str(image_path)],
        "target_bbox": target_bbox,
        "image_width": width,
        "image_height": height,
        "ref_id": row.get("ref_id"),
    }
    grpo = {"messages": [user], **metadata}
    normalized = [round(bbox[0] * 1000 / source_width, 1), round(bbox[1] * 1000 / source_height, 1),
                  round(bbox[2] * 1000 / source_width, 1), round(bbox[3] * 1000 / source_height, 1)]
    sft = {"messages": [user, {"role": "assistant", "content": json.dumps([normalized])}], **metadata}
    return grpo, sft


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--grpo-output", type=Path, required=True)
    parser.add_argument("--sft-output", type=Path, required=True)
    args = parser.parse_args()

    for output in (args.grpo_output, args.sft_output):
        output.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    with args.manifest.open(encoding="utf-8") as source, \
         args.grpo_output.open("w", encoding="utf-8") as grpo_file, \
         args.sft_output.open("w", encoding="utf-8") as sft_file:
        for count, line in enumerate(source, start=1):
            grpo, sft = convert(json.loads(line))
            grpo_file.write(json.dumps(grpo, ensure_ascii=False) + "\n")
            sft_file.write(json.dumps(sft, ensure_ascii=False) + "\n")
    print(f"Wrote {count} GRPO rows to {args.grpo_output}")
    print(f"Wrote {count} SFT rows to {args.sft_output}")


if __name__ == "__main__":
    main()
