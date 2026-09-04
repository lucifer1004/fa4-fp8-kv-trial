# Copyright (c) 2026, SGLang Team.
"""Validate TP leader-only SM120 FA4 SplitKV startup calibration.

Run this script with ``torchrun`` on at least two homogeneous SM120 GPUs.  It
uses a CPU process group, lets rank zero execute real DRAM-faithful tuning, and
turns every tuner entry point on non-leader ranks into a hard failure.  The
test also captures and replays the resulting production FA4 route on every
rank before writing one machine-readable report from rank zero.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.distributed as dist


SCHEMA_VERSION = "sglang.fa4_sm120_splitkv_tp_consensus.v1"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--kv-length", type=int, default=8192)
    parser.add_argument("--page-size", type=int, default=64)
    parser.add_argument("--gqa-group", type=int, default=6)
    parser.add_argument("--mtp-tokens", type=int, default=2)
    return parser.parse_args()


def _count_route_families(payload: dict) -> int:
    return sum(
        len(implementation.get("route_families", {}))
        for device in payload.get("devices", {}).values()
        for implementation in device.get("implementations", {}).values()
    )


def main() -> None:
    args = _parse_args()
    if args.kv_length % args.page_size:
        raise ValueError("kv-length must be divisible by page-size")
    dist.init_process_group("gloo")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    if world_size < 2:
        raise RuntimeError("TP consensus validation requires at least two ranks")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    if torch.cuda.get_device_capability(device)[0] != 12:
        raise RuntimeError("TP consensus validation requires SM120 GPUs")

    from sglang.kernels.ops.attention.fa4_sm120 import splitkv_tuner
    from sglang.kernels.ops.attention.fa4_sm120.runtime import sm120_forward_host
    from sglang.kernels.ops.attention.fa4_sm120.splitkv_calibration import (
        SplitKvCalibrationCache,
    )
    from sglang.kernels.ops.attention.fa4_sm120.splitkv_router import (
        splitkv_calibration_registry,
        splitkv_calibration_session,
    )
    from sglang.kernels.ops.attention.flash_attention_v4_sm120 import (
        flash_attn_with_kvcache,
    )
    from sglang.kernels.ops.attention.flash_attn.cute.interface import (
        num_splits_heuristic,
    )

    splitkv_calibration_registry.cache = SplitKvCalibrationCache(args.cache)
    splitkv_calibration_registry.reset()

    counters = {"calibrate_route_family": 0, "refine_route_workload": 0}
    original_calibrate = splitkv_tuner.calibrate_route_family
    original_refine = splitkv_tuner.refine_route_workload

    def leader_calibrate(**kwargs):
        counters["calibrate_route_family"] += 1
        if rank != 0:
            raise AssertionError("non-leader rank attempted family calibration")
        return original_calibrate(**kwargs)

    def leader_refine(**kwargs):
        counters["refine_route_workload"] += 1
        if rank != 0:
            raise AssertionError("non-leader rank attempted workload refinement")
        return original_refine(**kwargs)

    splitkv_tuner.calibrate_route_family = leader_calibrate
    splitkv_tuner.refine_route_workload = leader_refine

    batch_size = 1
    kv_heads = 1
    head_dim = 128
    total_q = batch_size * args.mtp_tokens
    q_heads = kv_heads * args.gqa_group
    pages_per_sequence = args.kv_length // args.page_size
    generator = torch.Generator(device=device).manual_seed(20260903)
    q = torch.randn(
        total_q,
        q_heads,
        head_dim,
        generator=generator,
        device=device,
        dtype=torch.bfloat16,
    )
    k_cache = torch.randn(
        pages_per_sequence,
        args.page_size,
        kv_heads,
        head_dim,
        generator=generator,
        device=device,
        dtype=torch.bfloat16,
    )
    v_cache = torch.randn_like(k_cache)
    page_table = torch.arange(
        pages_per_sequence, device=device, dtype=torch.int32
    ).view(1, -1)
    cache_seqlens = torch.tensor(
        [args.kv_length], device=device, dtype=torch.int32
    )
    cu_seqlens_q = torch.tensor(
        [0, args.mtp_tokens], device=device, dtype=torch.int32
    )
    reference_out = torch.empty_like(q)
    auto_out = torch.empty_like(q)

    def launch(*, num_splits: int, out: torch.Tensor) -> torch.Tensor:
        return flash_attn_with_kvcache(
            q=q,
            k_cache=k_cache,
            v_cache=v_cache,
            page_table=page_table,
            cache_seqlens=cache_seqlens,
            cu_seqlens_q=cu_seqlens_q,
            max_seqlen_q=args.mtp_tokens,
            max_seqlen_k=args.kv_length,
            softmax_scale=head_dim**-0.5,
            causal=True,
            num_splits=num_splits,
            pack_gqa=True,
            out=out,
        )

    launch(num_splits=1, out=reference_out)
    reference = reference_out.clone()

    with splitkv_calibration_session(
        "tune", process_group=dist.group.WORLD, allow_tuning=True
    ):
        launch(num_splits=0, out=auto_out)
        eager = auto_out.clone()

        config = sm120_forward_host.select_config(
            head_dim=head_dim,
            head_dim_v=head_dim,
            tile_mn=None,
            has_bias=False,
            total_q_rows=q.numel() // head_dim,
            num_sms=torch.cuda.get_device_properties(device).multi_processor_count,
            num_batch=batch_size,
            seqlen_q=args.mtp_tokens,
            seqlen_k=args.kv_length,
            num_head_kv=kv_heads,
            qhead_per_kvhead=args.gqa_group,
            is_causal=True,
            is_local=False,
            window_size_left=None,
            window_size_right=None,
            pack_gqa=True,
            paged_kv=True,
        )
        packed_q_rows = args.mtp_tokens * args.gqa_group
        num_m_blocks = math.ceil(packed_q_rows / config.tile_m)
        num_n_blocks = math.ceil(args.kv_length / config.tile_n)
        plan = sm120_forward_host.resolve_plan(
            requested_num_splits=0,
            generic_num_n_blocks=num_n_blocks,
            head_dim=head_dim,
            head_dim_v=head_dim,
            batch_size=batch_size,
            num_head_kv=kv_heads,
            paged_kv=True,
            page_size=args.page_size,
            k=k_cache,
            v=v_cache,
            max_seqlen_q=args.mtp_tokens,
            max_seqlen_k=args.kv_length,
            pack_gqa=True,
            compute_dtype=q.dtype,
            element_size=k_cache.element_size(),
            packed_q_rows=packed_q_rows,
            tile_m=config.tile_m,
            tile_n=config.tile_n,
            num_m_blocks=num_m_blocks,
            total_mblocks=batch_size * kv_heads * num_m_blocks,
            num_sms=torch.cuda.get_device_properties(device).multi_processor_count,
            total_q=total_q,
            has_cu_seqlens_q=True,
            has_seqused_q=False,
            has_seqused_k=True,
            is_causal=True,
            is_local=False,
            window_size_left=None,
            window_size_right=None,
            has_score_or_mask_mod=False,
            is_stream_capturing=False,
            device=device,
            fake_mode=False,
            generic_heuristic=num_splits_heuristic,
        )
        launch(num_splits=0, out=auto_out)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            launch(num_splits=0, out=auto_out)
        graph.replay()
        torch.cuda.synchronize()
        replay = auto_out.clone()

    eager_close = torch.allclose(eager.float(), reference.float(), atol=2e-2, rtol=2e-2)
    replay_close = torch.allclose(
        replay.float(), reference.float(), atol=2e-2, rtol=2e-2
    )
    local_result = {
        "rank": rank,
        "device": torch.cuda.get_device_name(device),
        "plan": {
            "num_splits": plan.num_splits,
            "kv_tiles_per_cta": plan.split_kv_blocks_per_cta,
            "tile_m": config.tile_m,
            "tile_n": config.tile_n,
        },
        "tuner_calls": counters,
        "eager_correct": bool(eager_close),
        "graph_replay_correct": bool(replay_close),
    }
    gathered = [None] * world_size
    dist.all_gather_object(gathered, local_result)
    dist.barrier()

    if rank == 0:
        payload = json.loads(args.cache.read_text(encoding="utf-8"))
        plans = {
            (item["plan"]["num_splits"], item["plan"]["kv_tiles_per_cta"])
            for item in gathered
        }
        passed = (
            len(plans) == 1
            and gathered[0]["tuner_calls"]["calibrate_route_family"] == 1
            and all(
                item["tuner_calls"]["calibrate_route_family"] == 0
                and item["tuner_calls"]["refine_route_workload"] == 0
                for item in gathered[1:]
            )
            and all(
                item["eager_correct"] and item["graph_replay_correct"]
                for item in gathered
            )
            and _count_route_families(payload) == 1
        )
        report = {
            "schema_version": SCHEMA_VERSION,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "world_size": world_size,
            "process_group_backend": dist.get_backend(),
            "cache": str(args.cache),
            "cache_route_family_count": _count_route_families(payload),
            "ranks": gathered,
            "passed": passed,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        if not passed:
            raise AssertionError(
                "TP leader/broadcast/cache consensus validation failed"
            )
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
