from __future__ import annotations

import argparse
from typing import Any

from common import abs_path, read_jsonl, write_jsonl


BASIC_CONTEXT_KEYS = ["weather", "traffic_flow", "average_speed", "vehicle_count"]


def keep_base_fields(row: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
    output = {
        "image": row["image"],
        "context": context,
        "question": row["question"],
        "answer": row["answer"],
    }
    if "risk_level" in row:
        output["risk_level"] = row["risk_level"]
    if "abnormal_event" in row:
        output["abnormal_event"] = row["abnormal_event"]
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Build test ablation sets from data/test.jsonl.")
    parser.add_argument("--input", default=str(abs_path("data/test.jsonl")))
    parser.add_argument("--image_only_output", default=str(abs_path("data/test_image_only.jsonl")))
    parser.add_argument("--basic_context_output", default=str(abs_path("data/test_basic_context.jsonl")))
    parser.add_argument("--full_context_output", default=str(abs_path("data/test_full_context.jsonl")))
    args = parser.parse_args()

    rows = read_jsonl(args.input)
    image_only_rows = []
    basic_context_rows = []
    full_context_rows = []

    for row in rows:
        context = row.get("context", {})
        image_only_rows.append(keep_base_fields(row, {}))
        basic_context = {key: context[key] for key in BASIC_CONTEXT_KEYS if key in context}
        basic_context_rows.append(keep_base_fields(row, basic_context))
        full_context_rows.append(keep_base_fields(row, dict(context)))

    write_jsonl(image_only_rows, args.image_only_output)
    write_jsonl(basic_context_rows, args.basic_context_output)
    write_jsonl(full_context_rows, args.full_context_output)
    print(f"Saved {len(image_only_rows)} image-only samples to {args.image_only_output}")
    print(f"Saved {len(basic_context_rows)} basic-context samples to {args.basic_context_output}")
    print(f"Saved {len(full_context_rows)} full-context samples to {args.full_context_output}")


if __name__ == "__main__":
    main()
