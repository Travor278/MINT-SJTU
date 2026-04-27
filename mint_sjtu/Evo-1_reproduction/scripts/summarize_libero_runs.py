#!/usr/bin/env python3
"""Summarize Evo-1 LIBERO reproduction logs across seeds."""

from __future__ import annotations

import argparse
import csv
import re
import statistics
from pathlib import Path


SUITE_RE = re.compile(r"Start task suite\s+([A-Za-z0-9_]+)")
TOTAL_RE = re.compile(r"Total Successful Episodes:\s*(\d+)\s*/\s*(\d+)")
SEED_RE = re.compile(r"seed(\d+)", re.IGNORECASE)


def infer_seed(path: Path) -> str:
    match = SEED_RE.search(path.stem)
    if match:
        return match.group(1)
    if path.stem == "Evo1_libero_all":
        return "42"
    return path.stem


def parse_log(path: Path) -> dict:
    suites = []
    current_suite = None
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        suite_match = SUITE_RE.search(line)
        if suite_match:
            current_suite = suite_match.group(1)
            continue

        total_match = TOTAL_RE.search(line)
        if total_match and current_suite:
            success = int(total_match.group(1))
            total = int(total_match.group(2))
            suites.append(
                {
                    "suite": current_suite,
                    "success": success,
                    "total": total,
                    "rate": success / total if total else 0.0,
                }
            )
            current_suite = None

    return {
        "path": path,
        "seed": infer_seed(path),
        "suites": suites,
        "success": sum(item["success"] for item in suites),
        "total": sum(item["total"] for item in suites),
    }


def format_pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("logs", nargs="+", type=Path)
    parser.add_argument("--csv", type=Path, help="Optional CSV output path.")
    args = parser.parse_args()

    rows = []
    for log in args.logs:
        parsed = parse_log(log)
        if not parsed["suites"]:
            print(f"[warn] no completed suite summary found in {log}")
            continue

        overall_rate = parsed["success"] / parsed["total"] if parsed["total"] else 0.0
        rows.append(
            {
                "seed": parsed["seed"],
                "log": str(log),
                "suite": "overall",
                "success": parsed["success"],
                "total": parsed["total"],
                "rate": overall_rate,
            }
        )
        for suite in parsed["suites"]:
            rows.append(
                {
                    "seed": parsed["seed"],
                    "log": str(log),
                    "suite": suite["suite"],
                    "success": suite["success"],
                    "total": suite["total"],
                    "rate": suite["rate"],
                }
            )

    if not rows:
        return 1

    print("| seed | suite | success | total | rate |")
    print("|---:|---|---:|---:|---:|")
    for row in rows:
        print(
            f"| {row['seed']} | {row['suite']} | {row['success']} | "
            f"{row['total']} | {format_pct(row['rate'])} |"
        )

    complete_overall = [
        row["rate"]
        for row in rows
        if row["suite"] == "overall" and row["total"] == 400
    ]
    if complete_overall:
        mean = statistics.mean(complete_overall)
        if len(complete_overall) > 1:
            std = statistics.stdev(complete_overall)
            print(f"\nComplete-run overall: mean={format_pct(mean)}, std={format_pct(std)}")
        else:
            print(f"\nComplete-run overall: mean={format_pct(mean)}")

    if args.csv:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with args.csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f, fieldnames=["seed", "suite", "success", "total", "rate", "log"]
            )
            writer.writeheader()
            for row in rows:
                out = dict(row)
                out["rate"] = f"{row['rate']:.6f}"
                writer.writerow(out)
        print(f"\nCSV written to {args.csv}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
