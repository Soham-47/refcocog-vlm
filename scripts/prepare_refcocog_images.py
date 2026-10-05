"""Download only the images referenced by a RefCOCOg split."""

from __future__ import annotations

import argparse
import json
import os
from io import BytesIO
from pathlib import Path

import requests
from datasets import load_dataset
from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="train[:50]")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/scratch/ambpdc/soham/vlm/coco"),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("/scratch/ambpdc/soham/vlm/data/refcocog_manifest.jsonl"),
    )
    parser.add_argument("--timeout", type=float, default=10.0)
    parser.add_argument(
        "--insecure",
        action="store_true",
        help="Disable TLS verification for public image URLs if the HPC proxy requires it.",
    )
    return parser.parse_args()


def raw_info(example: dict) -> dict:
    value = example["raw_image_info"]
    return json.loads(value) if isinstance(value, str) else value


def source_file_name(example: dict) -> str:
    return Path(raw_info(example).get("file_name", example["file_name"])).name


def image_urls(example: dict) -> list[str]:
    info = raw_info(example)
    file_name = source_file_name(example)
    if "val2014" in file_name:
        coco_split = "val2014"
    else:
        coco_split = "train2014"

    urls = [
        info.get("flickr_url"),
        info.get("coco_url"),
        f"https://images.cocodataset.org/{coco_split}/{file_name}",
    ]
    # The dataset may contain the obsolete http://mscoco.org/images/... URL.
    urls = [url for url in urls if url and "mscoco.org" not in url]
    urls = [url for url in urls if url]
    if not urls:
        raise KeyError(f"No image URL in raw_image_info: {sorted(info)}")
    return urls


def destination(output_dir: Path, file_name: str) -> Path:
    if "val2014" in file_name:
        split_dir = "val2014"
    elif "train2014" in file_name:
        split_dir = "train2014"
    else:
        split_dir = "other"
    return output_dir / split_dir / Path(file_name).name


def download_image(url: str, path: Path, timeout: float, verify: bool) -> None:
    if path.exists():
        with Image.open(path) as image:
            image.verify()
        return

    response = requests.get(
        url,
        headers={"User-Agent": "refcocog-vlm/0.1"},
        timeout=timeout,
        verify=verify,
    )
    response.raise_for_status()

    with Image.open(BytesIO(response.content)) as image:
        image.verify()

    path.parent.mkdir(parents=True, exist_ok=True)
    partial_path = path.with_suffix(path.suffix + ".part")
    partial_path.write_bytes(response.content)
    partial_path.replace(path)


def main() -> None:
    args = parse_args()
    args.manifest.parent.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset("jxu124/refcocog", split=args.split)
    print(f"Loaded {len(dataset)} RefCOCOg examples from {args.split}")

    # ponytail: sequential downloads keep the first version simple; add bounded
    # concurrency only when full-split download time becomes the bottleneck.
    downloaded_sources: set[str] = set()
    manifest_rows: list[dict] = []
    failures = 0

    with args.manifest.open("w", encoding="utf-8") as manifest_file:
        for index, example in enumerate(dataset):
            example_file_name = Path(example["file_name"]).name
            source_name = source_file_name(example)
            path = destination(args.output_dir, source_name)
            try:
                last_error = None
                url = None
                if source_name not in downloaded_sources:
                    for candidate_url in image_urls(example):
                        try:
                            download_image(
                                candidate_url,
                                path,
                                args.timeout,
                                verify=not args.insecure,
                            )
                            url = candidate_url
                            downloaded_sources.add(source_name)
                            break
                        except Exception as exc:
                            last_error = exc
                    if url is None:
                        raise RuntimeError(last_error) from last_error
                else:
                    url = image_urls(example)[0]
            except Exception as exc:  # keep usable examples when one URL fails
                failures += 1
                print(f"[{index}] skipped {example_file_name}: {exc}")
                continue

            row = {
                "file_name": example_file_name,
                "source_file_name": source_name,
                "image_path": str(path),
                "image_url": url,
                "bbox": example["bbox"],
                "sentences": example["sentences"],
                "ref_id": example["ref_id"],
            }
            manifest_file.write(json.dumps(row) + "\n")
            manifest_file.flush()
            print(f"[{len(manifest_rows) + 1}] cached {example_file_name}")
            manifest_rows.append(row)

    print(f"Cached {len(manifest_rows)} unique images")
    print(f"Failed downloads: {failures}")
    print(f"Manifest: {args.manifest}")
    if not manifest_rows:
        raise RuntimeError("No images were downloaded successfully.")


if __name__ == "__main__":
    main()
