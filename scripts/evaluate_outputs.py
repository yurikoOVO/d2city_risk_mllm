from __future__ import annotations

import argparse
import re
from collections import Counter
from pathlib import Path
from typing import Any

from common import abs_path, read_jsonl


RISK_PATTERNS = {
    "high": [r"高风险", r"risk[_\s-]*level[:：]?\s*high", r"\bhigh\b"],
    "medium": [r"中风险", r"中等风险", r"risk[_\s-]*level[:：]?\s*medium", r"\bmedium\b"],
    "low": [r"低风险", r"risk[_\s-]*level[:：]?\s*low", r"\blow\b"],
}

ABNORMAL_PATTERNS = {
    "accident_related_congestion": [r"事故相关拥堵", r"事故.*拥堵", r"accident_related_congestion"],
    "construction_impact": [r"施工影响", r"道路施工", r"construction_impact"],
    "event_related_congestion": [r"活动散场拥堵", r"大型活动.*拥堵", r"event_related_congestion"],
    "vehicle_breakdown": [r"车辆故障", r"vehicle_breakdown"],
    "severe_congestion": [r"严重拥堵", r"severe_congestion"],
    "congestion": [r"交通拥堵", r"拥堵", r"\bcongestion\b"],
    "normal": [r"正常通行", r"未识别到明确异常", r"无明显异常", r"\bnormal\b"],
}


def parse_label(text: str, patterns: dict[str, list[str]]) -> str:
    text = text or ""
    for label, regexes in patterns.items():
        for pattern in regexes:
            if re.search(pattern, text, flags=re.IGNORECASE):
                return label
    return "unknown"


def parse_risk_level(text: str) -> str:
    return parse_label(text, RISK_PATTERNS)


def parse_abnormal_event(text: str) -> str:
    return parse_label(text, ABNORMAL_PATTERNS)


def ground_truth_labels(row: dict[str, Any]) -> tuple[str, str]:
    truth = row.get("ground_truth", row.get("answer", ""))
    risk = row.get("risk_level") or parse_risk_level(truth)
    abnormal = row.get("abnormal_event") or parse_abnormal_event(truth)
    return risk, abnormal


def evaluate_field(rows: list[dict[str, Any]], output_field: str) -> dict[str, Any]:
    risk_correct = 0
    abnormal_correct = 0
    evaluated = 0
    parsed_risks = Counter()
    parsed_abnormals = Counter()

    for row in rows:
        output = row.get(output_field, "")
        if not output:
            continue
        gt_risk, gt_abnormal = ground_truth_labels(row)
        pred_risk = parse_risk_level(output)
        pred_abnormal = parse_abnormal_event(output)
        parsed_risks[pred_risk] += 1
        parsed_abnormals[pred_abnormal] += 1
        risk_correct += int(pred_risk == gt_risk)
        abnormal_correct += int(pred_abnormal == gt_abnormal)
        evaluated += 1

    return {
        "output_field": output_field,
        "evaluated": evaluated,
        "risk_accuracy": risk_correct / evaluated if evaluated else 0.0,
        "abnormal_accuracy": abnormal_correct / evaluated if evaluated else 0.0,
        "parsed_risks": parsed_risks,
        "parsed_abnormals": parsed_abnormals,
    }


def print_result(result: dict[str, Any]) -> None:
    print(f"## {result['output_field']}")
    print(f"evaluated: {result['evaluated']}")
    print(f"risk_level accuracy: {result['risk_accuracy']:.4f}")
    print(f"abnormal_event accuracy: {result['abnormal_accuracy']:.4f}")
    print("parsed risk_level:", dict(result["parsed_risks"]))
    print("parsed abnormal_event:", dict(result["parsed_abnormals"]))
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate parsed risk_level and abnormal_event from model outputs.")
    parser.add_argument("--input", default=str(abs_path("outputs/infer_results.jsonl")))
    parser.add_argument(
        "--output_fields",
        nargs="+",
        default=["base_output", "finetuned_output"],
        help="Model output fields to evaluate.",
    )
    args = parser.parse_args()

    rows = read_jsonl(args.input)
    print(f"# Evaluation: {Path(args.input)}")
    print(f"total rows: {len(rows)}")
    print()
    for field in args.output_fields:
        print_result(evaluate_field(rows, field))


if __name__ == "__main__":
    main()
