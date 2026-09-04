#!/usr/bin/env python3
"""Check that the active environment can run the SM120 FP8-KV trial."""

from __future__ import annotations

import inspect
import json
import os
from pathlib import Path


_COMPATIBILITY_PATH = Path(__file__).resolve().parents[1] / "compatibility.json"


def main() -> int:
    import torch

    errors = []
    compatibility = json.loads(_COMPATIBILITY_PATH.read_text(encoding="utf-8"))
    expected_implementation = compatibility["sglang"][
        "splitkv_implementation_identity"
    ]
    if not torch.cuda.is_available():
        errors.append("CUDA is not available")
        capability = None
        device_name = None
    else:
        capability = torch.cuda.get_device_capability(0)
        device_name = torch.cuda.get_device_name(0)
        if capability[0] != 12:
            errors.append(
                f"device 0 has compute capability {capability}, expected 12.x"
            )
    if os.getenv("SGLANG_INKLING_FA4_USE_PIP") == "1":
        errors.append(
            "SGLANG_INKLING_FA4_USE_PIP=1 bypasses the supplied SM120 kernel"
        )

    implementation = None
    try:
        from sglang.kernels.ops.attention.fa4_sm120.splitkv_calibration import (
            splitkv_implementation_identity,
        )
        from sglang.kernels.ops.attention.flash_attention_v4_sm120 import (
            flash_attn_with_kvcache,
        )

        parameters = inspect.signature(flash_attn_with_kvcache).parameters
        required = {"k_descale", "v_descale", "max_seqlen_k"}
        missing = sorted(required - parameters.keys())
        if missing:
            errors.append(f"SGLang wrapper is missing parameters: {missing}")
        implementation = splitkv_implementation_identity()
        if implementation != expected_implementation:
            errors.append(
                "SGLang implementation identity mismatch: "
                f"found {implementation}, expected {expected_implementation}"
            )
    except (ImportError, AttributeError, OSError) as error:
        errors.append(f"supplied SGLang implementation is unavailable: {error}")

    report = {
        "passed": not errors,
        "errors": errors,
        "device": {
            "name": device_name,
            "compute_capability": capability,
        },
        "software": {
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "splitkv_implementation": implementation,
            "expected_splitkv_implementation": expected_implementation,
            "sglang_commit": compatibility["sglang"]["commit"],
        },
        "calibration": {
            "mode": os.getenv("SGLANG_FA4_SPLITKV_CALIBRATION", "load"),
            "cache": os.getenv("SGLANG_FA4_SPLITKV_CALIBRATION_CACHE"),
        },
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
