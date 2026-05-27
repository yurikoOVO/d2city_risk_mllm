from __future__ import annotations

import argparse
import json
from pathlib import Path

from common import PROJECT_ROOT, abs_path, read_jsonl


def clean_cell(value) -> str:
    if isinstance(value, dict):
        value = json.dumps(value, ensure_ascii=False)
    text = str(value).replace("\n", "<br>")
    return text.replace("|", "\\|")


def brief_comment(row: dict) -> str:
    truth = row.get("ground_truth", "")
    lora = row.get("finetuned_output", row.get("lora_output", ""))
    base = row.get("base_output", "")
    score = 0
    for key in ["风险等级", "异常事件", "原因分析", "管理建议"]:
        if key in lora:
            score += 1
    if score >= 3:
        return "LoRA输出结构较完整"
    if len(lora) > len(base):
        return "LoRA输出更充分，需人工复核标签一致性"
    if truth and truth[:8] in lora:
        return "LoRA与弱标注开头较一致"
    return "需人工复核"


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert compare jsonl results to a markdown table.")
    parser.add_argument("--input", default=str(abs_path("outputs/infer_results.jsonl")))
    parser.add_argument("--output", default=str(abs_path("outputs/compare_results.md")))
    args = parser.parse_args()

    rows = read_jsonl(args.input)
    lines = ["| image | context | ground_truth | base_output | finetuned_output | 简评 |", "|---|---|---|---|---|---|"]
    for row in rows:
        finetuned_output = row.get("finetuned_output", row.get("lora_output", ""))
        lines.append(
            "| "
            + " | ".join(
                [
                    clean_cell(row.get("image", "")),
                    clean_cell(row.get("context", {})),
                    clean_cell(row.get("ground_truth", "")),
                    clean_cell(row.get("base_output", "")),
                    clean_cell(finetuned_output),
                    clean_cell(brief_comment(row)),
                ]
            )
            + " |"
        )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Saved markdown table to {output}")


if __name__ == "__main__":
    main()
