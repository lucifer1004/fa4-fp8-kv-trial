#!/usr/bin/env python3
"""Calibrate one production SM120 FA4 SplitKV route family."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--kv-dtype", choices=("bf16", "fp8e4m3"), default="bf16")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--kv-heads", type=int, default=1)
    parser.add_argument("--gqa-group", type=int, default=6)
    parser.add_argument("--mtp-tokens", type=int, default=2)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--kv-length", type=int, default=8192)
    parser.add_argument("--page-size", type=int, default=64)
    parser.add_argument("--pool-mib", type=int, default=512)
    parser.add_argument("--cache-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> int:
    args = _parser().parse_args()
    positive = (
        args.batch_size,
        args.kv_heads,
        args.gqa_group,
        args.mtp_tokens,
        args.head_dim,
        args.kv_length,
        args.page_size,
        args.pool_mib,
    )
    if any(value <= 0 for value in positive):
        raise ValueError("calibration dimensions and pool size must be positive")
    if args.kv_length % args.page_size:
        raise ValueError("kv-length must be divisible by page-size")
    os.environ["SGLANG_FA4_SPLITKV_CALIBRATION_POOL_MIB"] = str(args.pool_mib)

    import torch

    from sglang.kernels.ops.attention.fa4_sm120.runtime import sm120_forward_host
    from sglang.kernels.ops.attention.fa4_sm120.splitkv_calibration import (
        SplitKvCalibrationCache,
        SplitKvCalibrationKey,
        splitkv_device_identity,
        splitkv_implementation_identity,
    )
    from sglang.kernels.ops.attention.fa4_sm120.splitkv_model import (
        SplitKvWorkload,
        ceil_div,
        select_partition,
    )
    from sglang.kernels.ops.attention.fa4_sm120.splitkv_router import (
        SplitKvProbeSpec,
        SplitKvRouteSpec,
    )
    from sglang.kernels.ops.attention.fa4_sm120.splitkv_tuner import (
        calibrate_route_family,
    )

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device(args.device)
    capability = torch.cuda.get_device_capability(device)
    if capability[0] != 12:
        raise RuntimeError(f"SM120 is required, got capability {capability}")

    packed_q_rows = args.gqa_group * args.mtp_tokens
    properties = torch.cuda.get_device_properties(device)
    config = sm120_forward_host.select_config(
        head_dim=args.head_dim,
        head_dim_v=args.head_dim,
        tile_mn=None,
        has_bias=False,
        total_q_rows=args.batch_size * args.kv_heads * packed_q_rows,
        num_sms=properties.multi_processor_count,
        num_batch=args.batch_size,
        seqlen_q=args.mtp_tokens,
        seqlen_k=args.kv_length,
        num_head_kv=args.kv_heads,
        qhead_per_kvhead=args.gqa_group,
        is_causal=True,
        is_local=False,
        window_size_left=None,
        window_size_right=None,
        pack_gqa=True,
        paged_kv=True,
    )
    num_m_blocks = ceil_div(packed_q_rows, config.tile_m)
    num_n_blocks = ceil_div(args.kv_length, config.tile_n)
    direct_uniform_batch = sm120_forward_host.use_direct_uniform_batch(
        batch_size=args.batch_size,
        paged_kv=True,
        has_cu_seqlens_q=True,
        has_seqused_q=False,
        total_q=args.batch_size * args.mtp_tokens,
        max_seqlen_q=args.mtp_tokens,
        num_m_blocks=num_m_blocks,
    )
    split_qk_n = sm120_forward_host.use_paged_decode_split_qk_n(
        head_dim=args.head_dim,
        head_dim_v=args.head_dim,
        paged_kv=True,
        max_seqlen_q=args.mtp_tokens,
        has_compact_q_groups=True,
        packed_q_rows=packed_q_rows,
        tile_m=config.tile_m,
        tile_n=config.tile_n,
        num_n_blocks=num_n_blocks,
        is_causal=True,
        is_local=False,
        has_score_or_mask_mod=False,
    )
    kv_element_size = 2 if args.kv_dtype == "bf16" else 1
    route = SplitKvRouteSpec(
        kv_storage=args.kv_dtype,
        compute="bf16",
        head_dim=args.head_dim,
        head_dim_v=args.head_dim,
        tile_m=config.tile_m,
        tile_n=config.tile_n,
        page_size=args.page_size,
        direct_uniform_batch=direct_uniform_batch,
        split_qk_n=split_qk_n,
    )
    workload = SplitKvWorkload(
        total_mblocks=args.batch_size * args.kv_heads * num_m_blocks,
        num_n_blocks=num_n_blocks,
        main_bytes_per_kv_tile=(
            config.tile_n * args.head_dim * 2 * kv_element_size
        ),
        output_rows=args.batch_size * args.kv_heads * packed_q_rows,
        head_dim_v=args.head_dim,
        max_workspace_bytes=512 << 20,
    )
    probe = SplitKvProbeSpec(
        batch_size=args.batch_size,
        num_head_kv=args.kv_heads,
        qhead_per_kvhead=args.gqa_group,
        max_seqlen_q=args.mtp_tokens,
        max_seqlen_k=args.kv_length,
        page_size=args.page_size,
        causal=True,
    )
    constants = calibrate_route_family(
        route=route,
        workload=workload,
        probe=probe,
        device=device,
    )
    device_key, device_facts = splitkv_device_identity(device)
    key = SplitKvCalibrationKey(
        device=device_key,
        implementation=splitkv_implementation_identity(),
        route_family=route.family,
    )
    cache = SplitKvCalibrationCache(args.cache_path)
    cache.save_constants(key, constants)
    selection = select_partition(workload, constants)
    document = {
        "schema_version": "sglang.fa4_sm120_splitkv_calibration.v2",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "device": device_facts,
        "calibration_key": asdict(key),
        "cache_path": str(cache.path),
        "route": asdict(route),
        "requested_workload": asdict(workload),
        "selection": asdict(selection),
        "constants": asdict(constants),
        "protocol": {
            "kv_tile_grid": [2, 4, 8, 16, 64, 128],
            "graph_ring": "one_graph_contains_complete_page_table_ring",
            "page_tables": "randomized_reuse_distance_gt_L2",
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(document, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(document, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
