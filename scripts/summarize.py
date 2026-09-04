#!/usr/bin/env python3
"""Summarize detailed trial matrix JSON files."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", nargs="+", type=Path)
    parser.add_argument("--json", action="store_true")
    return parser


def main() -> int:
    args = _parser().parse_args()
    rows = []
    for path in args.results:
        document = json.loads(path.read_text(encoding="utf-8"))
        if document.get("schema_version") != "sglang.fa4_fp8_kv_baseline.v2":
            continue
        results = document["results"]
        fusion_speedups = [
            result["measurement"]["protocols"]["cuda_graph_replay"][
                "fusion_speedup_over_unfused_end_to_end_p50"
            ]
            for result in results
        ]
        rows.append(
            {
                "kv_length": results[0]["shape"]["kv_length_per_sequence"],
                "cases": len(results),
                "passed_cases": sum(
                    result.get("status") == "passed" for result in results
                ),
                "split_counts": sorted(
                    {result["launch"]["actual_num_splits"] for result in results}
                ),
                "fusion_speedup_min": min(fusion_speedups),
                "fusion_speedup_median": statistics.median(fusion_speedups),
                "fusion_speedup_max": max(fusion_speedups),
            }
        )
    rows.sort(key=lambda row: row["kv_length"])
    passed = bool(rows) and all(
        row["passed_cases"] == row["cases"] for row in rows
    )
    summary = {"passed": passed, "rows": rows}
    if args.json:
        print(json.dumps(summary, indent=2, sort_keys=True))
    else:
        print("| KV | Passed | Splits | Fused speedup min/median/max |")
        print("|---:|---:|:---|:---|")
        for row in rows:
            splits = ",".join(str(value) for value in row["split_counts"])
            speedup = (
                f"{row['fusion_speedup_min']:.3f} / "
                f"{row['fusion_speedup_median']:.3f} / "
                f"{row['fusion_speedup_max']:.3f}"
            )
            print(
                f"| {row['kv_length']} | {row['passed_cases']}/{row['cases']} "
                f"| {splits} | {speedup} |"
            )
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
