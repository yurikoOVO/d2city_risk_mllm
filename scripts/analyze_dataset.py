from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from statistics import mean

from common import PROJECT_ROOT, abs_path, read_jsonl


def count_table(title: str, counter: Counter, total: int) -> list[str]:
    lines = [f"## {title}", "", "| value | count | ratio |", "|---|---:|---:|"]
    for key, count in counter.most_common():
        ratio = count / total if total else 0
        lines.append(f"| {key} | {count} | {ratio:.2%} |")
    if not counter:
        lines.append("| N/A | 0 | 0.00% |")
    lines.append("")
    return lines


def cross_table(cross: dict[str, Counter]) -> list[str]:
    risk_order = ["low", "medium", "high"]
    event_keys = sorted({event for counts in cross.values() for event in counts})
    lines = ["## Risk Level x Abnormal Event", ""]
    if not event_keys:
        return lines + ["No samples.", ""]

    lines.append("| risk_level | " + " | ".join(event_keys) + " | total |")
    lines.append("|---|" + "|".join(["---:"] * (len(event_keys) + 1)) + "|")
    for risk in risk_order:
        counts = cross.get(risk, Counter())
        row_values = [str(counts.get(event, 0)) for event in event_keys]
        lines.append(f"| {risk} | " + " | ".join(row_values) + f" | {sum(counts.values())} |")
    lines.append("")
    return lines


def conditional_event_abnormal_table(rows: list[dict]) -> list[str]:
    event_to_abnormal: dict[str, Counter] = defaultdict(Counter)
    for row in rows:
        event_info = row.get("context", {}).get("event_info", "unknown")
        abnormal_event = row.get("abnormal_event", "unknown")
        event_to_abnormal[event_info][abnormal_event] += 1

    abnormal_keys = sorted({key for counts in event_to_abnormal.values() for key in counts})
    lines = ["## Conditional Probability: Event Info -> Abnormal Event", ""]
    if not abnormal_keys:
        return lines + ["No samples.", ""]

    lines.append("| event_info | total | " + " | ".join(abnormal_keys) + " |")
    lines.append("|---|---:|" + "|".join(["---:"] * len(abnormal_keys)) + "|")
    for event_info in sorted(event_to_abnormal):
        counts = event_to_abnormal[event_info]
        total = sum(counts.values())
        probs = [f"{counts.get(key, 0) / total:.2%}" if total else "0.00%" for key in abnormal_keys]
        lines.append(f"| {event_info} | {total} | " + " | ".join(probs) + " |")
    lines.append("")
    return lines


def build_markdown(rows: list[dict]) -> str:
    total = len(rows)
    risk_counter = Counter(row.get("risk_level", "unknown") for row in rows)
    abnormal_counter = Counter(row.get("abnormal_event", "unknown") for row in rows)
    weather_counter = Counter(row.get("context", {}).get("weather", "unknown") for row in rows)
    flow_counter = Counter(row.get("context", {}).get("traffic_flow", "unknown") for row in rows)
    event_counter = Counter(row.get("context", {}).get("event_info", "unknown") for row in rows)
    speeds = [row.get("context", {}).get("average_speed") for row in rows]
    speeds = [s for s in speeds if isinstance(s, (int, float))]
    vehicle_counts = [row.get("context", {}).get("vehicle_count") for row in rows]
    vehicle_counts = [v for v in vehicle_counts if isinstance(v, (int, float))]

    cross: dict[str, Counter] = defaultdict(Counter)
    for row in rows:
        cross[row.get("risk_level", "unknown")][row.get("abnormal_event", "unknown")] += 1

    lines = ["# Dataset Statistics", "", f"- Total samples: {total}", ""]
    if speeds:
        lines.extend(
            [
                "## Average Speed",
                "",
                "| min | max | mean |",
                "|---:|---:|---:|",
                f"| {min(speeds):.2f} | {max(speeds):.2f} | {mean(speeds):.2f} |",
                "",
            ]
        )
    else:
        lines.extend(["## Average Speed", "", "No valid speed values.", ""])

    if vehicle_counts:
        lines.extend(
            [
                "## Vehicle Count",
                "",
                "| min | max | mean |",
                "|---:|---:|---:|",
                f"| {min(vehicle_counts):.2f} | {max(vehicle_counts):.2f} | {mean(vehicle_counts):.2f} |",
                "",
            ]
        )
    else:
        lines.extend(["## Vehicle Count", "", "No valid vehicle_count values.", ""])

    lines.extend(count_table("Risk Level Distribution", risk_counter, total))
    lines.extend(count_table("Abnormal Event Distribution", abnormal_counter, total))
    lines.extend(count_table("Weather Distribution", weather_counter, total))
    lines.extend(count_table("Traffic Flow Distribution", flow_counter, total))
    lines.extend(count_table("Event Info Distribution", event_counter, total))
    lines.extend(cross_table(cross))
    lines.extend(conditional_event_abnormal_table(rows))
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze D2-City risk SFT dataset statistics.")
    parser.add_argument("--input", default=str(abs_path("data/risk_sft.jsonl")))
    parser.add_argument("--output", default=str(abs_path("outputs/dataset_stats.md")))
    args = parser.parse_args()

    rows = read_jsonl(args.input)
    markdown = build_markdown(rows)
    output = abs_path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(markdown + "\n", encoding="utf-8")
    print(markdown)
    print(f"\nSaved dataset statistics to {output}")


if __name__ == "__main__":
    main()
