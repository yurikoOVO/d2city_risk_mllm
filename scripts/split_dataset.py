from __future__ import annotations

import argparse
import random

from common import PROJECT_ROOT, abs_path, load_config, read_jsonl, write_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description="Split SFT jsonl into train/val/test.")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs/default.yaml"))
    parser.add_argument("--input", default=None)
    parser.add_argument("--train", default=None)
    parser.add_argument("--val", default=None)
    parser.add_argument("--test", default=None)
    parser.add_argument("--train_ratio", type=float, default=None)
    parser.add_argument("--val_ratio", type=float, default=None)
    parser.add_argument("--test_ratio", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    split_cfg = cfg["split"]
    train_ratio = args.train_ratio if args.train_ratio is not None else split_cfg["train_ratio"]
    val_ratio = args.val_ratio if args.val_ratio is not None else split_cfg["val_ratio"]
    test_ratio = args.test_ratio if args.test_ratio is not None else split_cfg["test_ratio"]
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError("train_ratio + val_ratio + test_ratio must equal 1.0")

    rows = read_jsonl(abs_path(args.input or cfg["data"]["sft_jsonl"]))
    rng = random.Random(args.seed if args.seed is not None else split_cfg["seed"])
    rng.shuffle(rows)
    n = len(rows)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    train_rows = rows[:n_train]
    val_rows = rows[n_train : n_train + n_val]
    test_rows = rows[n_train + n_val :]

    write_jsonl(train_rows, abs_path(args.train or cfg["data"]["train_jsonl"]))
    write_jsonl(val_rows, abs_path(args.val or cfg["data"]["val_jsonl"]))
    write_jsonl(test_rows, abs_path(args.test or cfg["data"]["test_jsonl"]))
    print(f"Split {n} samples -> train={len(train_rows)}, val={len(val_rows)}, test={len(test_rows)}")


if __name__ == "__main__":
    main()
