from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image

from common import PROJECT_ROOT, abs_path, load_config, read_jsonl


REQUIRED_KEYS = {"image", "context", "question", "answer"}
CONTEXT_KEYS = {"weather", "traffic_flow", "average_speed", "event_info"}


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate and preview SFT samples.")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs/default.yaml"))
    parser.add_argument("--input", default=None)
    parser.add_argument("--num_show", type=int, default=3)
    parser.add_argument("--check_images", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    rows = read_jsonl(abs_path(args.input or cfg["data"]["sft_jsonl"]))
    missing_images = 0
    bad_rows = 0
    for idx, row in enumerate(rows):
        if not REQUIRED_KEYS.issubset(row):
            bad_rows += 1
            print(f"[BAD] row {idx}: missing keys {REQUIRED_KEYS - set(row)}")
            continue
        if not CONTEXT_KEYS.issubset(row["context"]):
            bad_rows += 1
            print(f"[BAD] row {idx}: missing context keys {CONTEXT_KEYS - set(row['context'])}")
        image_path = Path(row["image"])
        if not image_path.exists():
            missing_images += 1
            print(f"[MISS] image not found: {image_path}")
        elif args.check_images:
            try:
                with Image.open(image_path) as im:
                    im.verify()
            except Exception as exc:
                bad_rows += 1
                print(f"[BAD] image failed to open: {image_path}, {exc}")

    print(f"Samples: {len(rows)}, bad_rows: {bad_rows}, missing_images: {missing_images}")
    for row in rows[: args.num_show]:
        print("=" * 80)
        print("image:", row["image"])
        print("context:", row["context"])
        print("question:", row["question"])
        print("answer:", row["answer"])


if __name__ == "__main__":
    main()
