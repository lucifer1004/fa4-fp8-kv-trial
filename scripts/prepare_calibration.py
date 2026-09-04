#!/usr/bin/env python3
"""Prepare the four calibrated route families used by the trial matrix."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path


ROUTES = (
    ("bf16", 6, 2, "bf16-m64n64"),
    ("fp8e4m3", 6, 2, "fp8-m64n64"),
    ("bf16", 6, 3, "bf16-m64n128"),
    ("fp8e4m3", 6, 3, "fp8-m64n128"),
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--pool-mib", type=int, default=512)
    parser.add_argument("--attempts", type=int, default=2)
    return parser


def main() -> int:
    args = _parser().parse_args()
    script = Path(__file__).with_name("calibrate_splitkv.py")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.cache.parent.mkdir(parents=True, exist_ok=True)
    if args.attempts <= 0:
        raise ValueError("attempts must be positive")
    descriptor, staging_name = tempfile.mkstemp(
        prefix=f".{args.cache.name}.",
        suffix=".staging",
        dir=args.cache.parent,
    )
    os.close(descriptor)
    staging_cache = Path(staging_name)
    results = []
    published = False
    try:
        for kv_dtype, gqa_group, mtp_tokens, name in ROUTES:
            output = args.output_dir / f"{name}.json"
            document = None
            last_error = None
            for attempt in range(1, args.attempts + 1):
                attempt_output = args.output_dir / f".{name}.attempt-{attempt}.json"
                command = [
                    sys.executable,
                    str(script),
                    "--device",
                    args.device,
                    "--kv-dtype",
                    kv_dtype,
                    "--gqa-group",
                    str(gqa_group),
                    "--mtp-tokens",
                    str(mtp_tokens),
                    "--kv-length",
                    "8192",
                    "--pool-mib",
                    str(args.pool_mib),
                    "--cache-path",
                    str(staging_cache),
                    "--output",
                    str(attempt_output),
                ]
                completed = subprocess.run(
                    command,
                    check=False,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.PIPE,
                    text=True,
                )
                if completed.returncode == 0:
                    document = json.loads(
                        attempt_output.read_text(encoding="utf-8")
                    )
                    document["cache_path"] = str(args.cache)
                    output.write_text(
                        json.dumps(document, indent=2, sort_keys=True) + "\n",
                        encoding="utf-8",
                    )
                    attempt_output.unlink()
                    break
                last_error = completed.stderr.strip()
                if attempt_output.exists():
                    attempt_output.unlink()
            if document is None:
                raise RuntimeError(
                    f"calibration failed for {name} after {args.attempts} "
                    f"attempts:\n{last_error}"
                )
            results.append(
                {
                    "name": name,
                    "route_family": document["calibration_key"]["route_family"],
                    "selected_kv_tiles_per_cta": document["selection"][
                        "kv_tiles_per_cta"
                    ],
                    "selected_num_splits": document["selection"]["num_splits"],
                    "result": output.name,
                }
            )
        os.replace(staging_cache, args.cache)
        published = True
    finally:
        if not published and staging_cache.exists():
            staging_cache.unlink()
    manifest = {
        "schema_version": "sglang.fa4_fp8_kv_trial_calibration.v2",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "cache": str(args.cache),
        "routes": results,
        "passed": len(results) == len(ROUTES),
    }
    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
