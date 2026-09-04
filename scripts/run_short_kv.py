#!/usr/bin/env python3
"""Run the complete customer short-KV correctness and latency matrix."""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path


def _parse_int_list(value: str) -> tuple[int, ...]:
    values = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    if not values or any(item <= 0 for item in values):
        raise argparse.ArgumentTypeError(
            "expected a comma-separated list of positive integers"
        )
    return values


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--kv-lengths",
        type=_parse_int_list,
        default=(128, 256, 512, 768, 1024),
    )
    parser.add_argument("--gqa-groups", default="6,8,16")
    parser.add_argument("--mtp-tokens", default="2,3,4,5,6,7,8")
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--trials", type=int, default=30)
    parser.add_argument(
        "--no-refine-ambiguous",
        action="store_true",
        help="Disable generic startup refinement of model-ambiguous routes.",
    )
    return parser


def _validate_cache(path: Path, device_name: str) -> None:
    import torch

    from sglang.kernels.ops.attention.fa4_sm120.splitkv_calibration import (
        SplitKvCalibrationCache,
        SplitKvCalibrationKey,
        splitkv_device_identity,
        splitkv_implementation_identity,
    )

    device = torch.device(device_name)
    device_key, _ = splitkv_device_identity(device)
    implementation = splitkv_implementation_identity()
    cache = SplitKvCalibrationCache(path)
    families = (
        "bf16-to-bf16-hd128-m64n64-paged64-gather-varlen-singleqk",
        "fp8e4m3-to-bf16-hd128-m64n64-paged64-gather-varlen-singleqk",
        "bf16-to-bf16-hd128-m64n128-paged64-gather-varlen-singleqk",
        "fp8e4m3-to-bf16-hd128-m64n128-paged64-gather-varlen-singleqk",
    )
    entries = {}
    for family in families:
        key = SplitKvCalibrationKey(device_key, implementation, family)
        entries[family] = cache.get(key)
    missing = [family for family, entry in entries.items() if entry is None]
    if missing:
        raise RuntimeError(
            "calibration cache is stale or incomplete; rerun "
            f"prepare_calibration.py (missing={missing})"
        )
def main() -> int:
    args = _parser().parse_args()
    if not args.cache.is_file():
        raise FileNotFoundError(f"calibration cache does not exist: {args.cache}")
    calibration_mode = "load" if args.no_refine_ambiguous else "tune"
    os.environ["SGLANG_FA4_SPLITKV_CALIBRATION"] = calibration_mode
    os.environ["SGLANG_FA4_SPLITKV_CALIBRATION_CACHE"] = str(args.cache)
    _validate_cache(args.cache, args.device)

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from bench_fp8_kv import main as run_one_matrix
    from sglang.kernels.ops.attention.fa4_sm120.splitkv_router import (
        splitkv_calibration_session,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    reports = []
    with splitkv_calibration_session(
        calibration_mode,
        allow_tuning=not args.no_refine_ambiguous,
    ):
        for kv_length in args.kv_lengths:
            output = args.output_dir / f"kv-{kv_length}.json"
            status = run_one_matrix(
                [
                    "--gqa-groups",
                    args.gqa_groups,
                    "--mtp-tokens",
                    args.mtp_tokens,
                    "--modes",
                    "splitkv",
                    "--splitkv-splits",
                    "0",
                    "--kv-length",
                    str(kv_length),
                    "--device",
                    args.device,
                    "--warmup",
                    str(args.warmup),
                    "--trials",
                    str(args.trials),
                    "--run-fused",
                    "--output",
                    str(output),
                    "--quiet",
                ]
            )
            document = json.loads(output.read_text(encoding="utf-8"))
            passed_cases = sum(
                result.get("status") == "passed"
                for result in document["results"]
            )
            reports.append(
                {
                    "kv_length": kv_length,
                    "result": output.name,
                    "cases": len(document["results"]),
                    "passed_cases": passed_cases,
                    "passed": status == 0
                    and passed_cases == len(document["results"]),
                }
            )
    manifest = {
        "schema_version": "sglang.fa4_fp8_kv_trial_matrix.v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "kv_lengths": list(args.kv_lengths),
        "refine_ambiguous": not args.no_refine_ambiguous,
        "reports": reports,
        "passed": all(report["passed"] for report in reports),
    }
    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0 if manifest["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
