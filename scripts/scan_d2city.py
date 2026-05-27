from __future__ import annotations

import argparse
from pathlib import Path

from common import PROJECT_ROOT, abs_path, load_config, write_json


VIDEO_SUFFIXES = {".mp4", ".avi", ".mov", ".mkv"}
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def infer_scene_id(file_path: Path, dataset_root: Path) -> str:
    rel = file_path.relative_to(dataset_root)
    if len(rel.parts) > 1:
        return rel.parts[0]
    return file_path.stem


def scan_dataset(dataset_root: Path) -> list[dict[str, str]]:
    records: list[dict[str, str]] = []
    for file_path in sorted(dataset_root.rglob("*")):
        if not file_path.is_file():
            continue
        suffix = file_path.suffix.lower()
        if suffix in VIDEO_SUFFIXES:
            file_type = "video"
        elif suffix in IMAGE_SUFFIXES:
            file_type = "image"
        else:
            continue
        records.append(
            {
                "file_path": str(file_path.resolve()),
                "file_type": file_type,
                "scene_id": infer_scene_id(file_path, dataset_root),
                "source_name": file_path.name,
            }
        )
    return records


def parse_args() -> argparse.Namespace:
    cfg = load_config()
    return argparse.ArgumentParser(description="Scan D2-City videos and images.").parse_args()


def main() -> None:
    parser = argparse.ArgumentParser(description="Scan D2-City videos and images.")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs/default.yaml"))
    parser.add_argument("--dataset_root", default=None)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    dataset_root = Path(args.dataset_root or cfg["d2city_root"]).resolve()
    output = abs_path(args.output or cfg["data"]["raw_index"])
    if not dataset_root.exists():
        raise FileNotFoundError(f"D2-City path does not exist: {dataset_root}")

    records = scan_dataset(dataset_root)
    write_json(records, output)
    print(f"Scanned {len(records)} files. Saved to {output}")


if __name__ == "__main__":
    main()
