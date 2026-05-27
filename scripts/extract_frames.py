from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

from tqdm import tqdm

from common import PROJECT_ROOT, abs_path, load_config, read_json, sanitize_name, write_json


def link_or_copy_image(src: Path, dst: Path, mode: str) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    if mode == "copy":
        shutil.copy2(src, dst)
    else:
        try:
            os.symlink(src, dst)
        except FileExistsError:
            pass
        except OSError:
            shutil.copy2(src, dst)


def extract_video_frames(video_path: Path, out_dir: Path, scene_id: str, interval_sec: float) -> list[dict]:
    try:
        import cv2
    except ImportError as exc:
        raise ImportError(
            "extract_frames.py requires opencv-python. Install dependencies with: "
            "pip install -r requirements.txt"
        ) from exc

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[WARN] Failed to open video: {video_path}")
        return []

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration_sec = total_frames / fps if total_frames > 0 else 0
    stem = sanitize_name(video_path.stem)
    scene_dir = out_dir / sanitize_name(scene_id)
    scene_dir.mkdir(parents=True, exist_ok=True)

    records = []
    timestamps = list(range(0, max(int(duration_sec), 1) + 1, int(interval_sec)))
    if not timestamps:
        timestamps = [0]
    for ts in timestamps:
        frame_id = int(ts * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
        ok, frame = cap.read()
        if not ok:
            continue
        frame_path = scene_dir / f"{stem}_t{ts:06d}.jpg"
        cv2.imwrite(str(frame_path), frame)
        records.append(
            {
                "frame_path": str(frame_path.resolve()),
                "video_path": str(video_path.resolve()),
                "timestamp_sec": int(ts),
                "scene_id": scene_id,
            }
        )
    cap.release()
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract frames from D2-City videos and import images.")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs/default.yaml"))
    parser.add_argument("--raw_index", default=None)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--output_index", default=None)
    parser.add_argument("--interval_sec", type=float, default=None)
    parser.add_argument("--image_mode", choices=["symlink", "copy"], default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    raw_index = abs_path(args.raw_index or cfg["data"]["raw_index"])
    output_dir = abs_path(args.output_dir or cfg["data"]["frames_dir"])
    output_index = abs_path(args.output_index or cfg["data"]["frame_index"])
    interval_sec = args.interval_sec or cfg["frame_extraction"]["interval_sec"]
    image_mode = args.image_mode or cfg["frame_extraction"]["image_mode"]

    records = read_json(raw_index)
    frame_records = []
    for item in tqdm(records, desc="Extract/import frames"):
        src = Path(item["file_path"])
        scene_id = item["scene_id"]
        if item["file_type"] == "video":
            frame_records.extend(extract_video_frames(src, output_dir, scene_id, interval_sec))
        else:
            scene_dir = output_dir / sanitize_name(scene_id)
            dst = scene_dir / f"{sanitize_name(src.stem)}{src.suffix.lower()}"
            link_or_copy_image(src, dst, image_mode)
            frame_records.append(
                {
                    "frame_path": str(dst.resolve()),
                    "video_path": "",
                    "timestamp_sec": 0,
                    "scene_id": scene_id,
                }
            )

    write_json(frame_records, output_index)
    print(f"Saved {len(frame_records)} frame records to {output_index}")


if __name__ == "__main__":
    main()
