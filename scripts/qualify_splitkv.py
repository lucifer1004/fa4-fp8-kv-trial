# Copyright (c) 2026, SGLang Team.
"""Exhaustively qualify calibrated SM120 FA4 SplitKV customer routes.

Each legal KV-tiles-per-CTA partition is timed against a rotating page-table
ring whose reuse distance exceeds L2.  ``torchrun`` may distribute independent
customer/dtype cases over homogeneous GPUs; rank zero writes the merged report.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Sequence

import torch
import torch.distributed as dist


SCHEMA_VERSION = "sglang.fa4_sm120_splitkv_exhaustive_oracle.v1"
GQA_GROUPS = (6, 8, 16)
MTP_TOKENS = (2, 3, 4, 5, 6, 7, 8)
KV_STORAGES = ("bf16", "fp8e4m3")


def _parse_int_list(value: str) -> tuple[int, ...]:
    values = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    if not values or any(item <= 0 for item in values):
        raise argparse.ArgumentTypeError(
            "expected a comma-separated list of positive integers"
        )
    return values


def _parse_kv_storages(value: str) -> tuple[str, ...]:
    values = tuple(part.strip() for part in value.split(",") if part.strip())
    invalid = sorted(set(values) - set(KV_STORAGES))
    if not values or invalid:
        raise argparse.ArgumentTypeError(
            f"expected a subset of {KV_STORAGES}; invalid={invalid}"
        )
    return values


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument(
        "--gqa-groups", type=_parse_int_list, default=GQA_GROUPS
    )
    parser.add_argument(
        "--mtp-tokens", type=_parse_int_list, default=MTP_TOKENS
    )
    parser.add_argument(
        "--kv-storages", type=_parse_kv_storages, default=KV_STORAGES
    )
    lengths = parser.add_mutually_exclusive_group()
    lengths.add_argument("--kv-length", type=int)
    lengths.add_argument(
        "--kv-lengths", type=_parse_int_list, default=(8192,)
    )
    parser.add_argument("--page-size", type=int, default=64)
    parser.add_argument("--warmup-batches", type=int, default=2)
    parser.add_argument("--timed-batches", type=int, default=5)
    parser.add_argument(
        "--write-refinements",
        action="store_true",
        help="Persist each measured oracle grain as an exact-workload override.",
    )
    parser.add_argument(
        "--limit-tasks",
        type=int,
        default=0,
        help="development-only prefix of the generated case/dtype tasks",
    )
    return parser.parse_args()


def _summary_us(samples: Sequence[float]) -> dict[str, float | int]:
    ordered = sorted(samples)
    return {
        "count": len(ordered),
        "min": ordered[0],
        "p50": statistics.median(ordered),
        "mean": statistics.fmean(ordered),
        "max": ordered[-1],
    }


def _measure_eager_event_us(
    calls: Sequence[Callable[[], None]], *, warmup: int, timed: int
) -> dict[str, float | int]:
    for _ in range(warmup):
        for call in calls:
            call()
    torch.cuda.synchronize()
    samples = []
    for _ in range(timed):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for call in calls:
            call()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) * 1000.0 / len(calls))
    return _summary_us(samples)


def _measure_wall_us(
    calls: Sequence[Callable[[], None]], *, warmup: int, timed: int
) -> dict[str, float | int]:
    for _ in range(warmup):
        for call in calls:
            call()
    torch.cuda.synchronize()
    samples = []
    for _ in range(timed):
        started = time.perf_counter_ns()
        for call in calls:
            call()
        torch.cuda.synchronize()
        samples.append((time.perf_counter_ns() - started) / 1000.0 / len(calls))
    return _summary_us(samples)


def _measure_graph_replay_us(
    calls: Sequence[Callable[[], None]], *, warmup: int, timed: int
) -> dict[str, float | int]:
    """Measure one graph containing a complete L2-cold page-table ring."""
    for call in calls:
        call()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for call in calls:
            call()
    torch.cuda.synchronize()
    for _ in range(warmup):
        graph.replay()
    torch.cuda.synchronize()
    samples = []
    for _ in range(timed):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        graph.replay()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) * 1000.0 / len(calls))
    result = _summary_us(samples)
    del graph
    return result


def _route_and_workload(
    *,
    gqa_group: int,
    mtp_tokens: int,
    kv_storage: str,
    kv_length: int,
    page_size: int,
    device: torch.device,
):
    from sglang.kernels.ops.attention.fa4_sm120.runtime import sm120_forward_host
    from sglang.kernels.ops.attention.fa4_sm120.splitkv_model import (
        SplitKvWorkload,
        ceil_div,
    )
    from sglang.kernels.ops.attention.fa4_sm120.splitkv_router import (
        SplitKvProbeSpec,
        SplitKvRouteSpec,
    )

    head_dim = 128
    batch_size = 1
    kv_heads = 1
    packed_q_rows = gqa_group * mtp_tokens
    props = torch.cuda.get_device_properties(device)
    config = sm120_forward_host.select_config(
        head_dim=head_dim,
        head_dim_v=head_dim,
        tile_mn=None,
        has_bias=False,
        total_q_rows=packed_q_rows,
        num_sms=props.multi_processor_count,
        num_batch=batch_size,
        seqlen_q=mtp_tokens,
        seqlen_k=kv_length,
        num_head_kv=kv_heads,
        qhead_per_kvhead=gqa_group,
        is_causal=True,
        is_local=False,
        window_size_left=None,
        window_size_right=None,
        pack_gqa=True,
        paged_kv=True,
    )
    num_m_blocks = ceil_div(packed_q_rows, config.tile_m)
    num_n_blocks = ceil_div(kv_length, config.tile_n)
    direct_uniform_batch = sm120_forward_host.use_direct_uniform_batch(
        batch_size=batch_size,
        paged_kv=True,
        has_cu_seqlens_q=True,
        has_seqused_q=False,
        total_q=mtp_tokens,
        max_seqlen_q=mtp_tokens,
        num_m_blocks=num_m_blocks,
    )
    split_qk_n = sm120_forward_host.use_paged_decode_split_qk_n(
        head_dim=head_dim,
        head_dim_v=head_dim,
        paged_kv=True,
        max_seqlen_q=mtp_tokens,
        has_compact_q_groups=True,
        packed_q_rows=packed_q_rows,
        tile_m=config.tile_m,
        tile_n=config.tile_n,
        num_n_blocks=num_n_blocks,
        is_causal=True,
        is_local=False,
        has_score_or_mask_mod=False,
    )
    element_size = 2 if kv_storage == "bf16" else 1
    route = SplitKvRouteSpec(
        kv_storage=kv_storage,
        compute="bf16",
        head_dim=head_dim,
        head_dim_v=head_dim,
        tile_m=config.tile_m,
        tile_n=config.tile_n,
        page_size=page_size,
        direct_uniform_batch=direct_uniform_batch,
        split_qk_n=split_qk_n,
    )
    workload = SplitKvWorkload(
        total_mblocks=num_m_blocks,
        num_n_blocks=num_n_blocks,
        main_bytes_per_kv_tile=config.tile_n * head_dim * 2 * element_size,
        output_rows=packed_q_rows,
        head_dim_v=head_dim,
        max_workspace_bytes=512 << 20,
    )
    probe = SplitKvProbeSpec(
        batch_size=batch_size,
        num_head_kv=kv_heads,
        qhead_per_kvhead=gqa_group,
        max_seqlen_q=mtp_tokens,
        max_seqlen_k=kv_length,
        page_size=page_size,
        causal=True,
    )
    return route, workload, probe


def _run_task(
    *,
    gqa_group: int,
    mtp_tokens: int,
    kv_storage: str,
    cache_path: Path,
    kv_length: int,
    page_size: int,
    warmup_batches: int,
    timed_batches: int,
    device: torch.device,
    write_refinements: bool,
) -> dict[str, object]:
    from sglang.kernels.ops.attention.fa4_sm120.splitkv_calibration import (
        SplitKvCalibrationCache,
        SplitKvCalibrationKey,
        splitkv_device_identity,
        splitkv_implementation_identity,
    )
    from sglang.kernels.ops.attention.fa4_sm120.splitkv_model import (
        near_optimal_partitions,
        predict_partition,
        predict_partitions,
        select_partition,
    )
    from sglang.kernels.ops.attention.fa4_sm120.splitkv_router import (
        splitkv_workload_key,
    )
    from sglang.kernels.ops.attention.fa4_sm120.splitkv_tuner import (
        _allocate_inputs,
        _make_call,
        _validate_probe_route,
    )

    route, workload, probe = _route_and_workload(
        gqa_group=gqa_group,
        mtp_tokens=mtp_tokens,
        kv_storage=kv_storage,
        kv_length=kv_length,
        page_size=page_size,
        device=device,
    )
    device_key, device_facts = splitkv_device_identity(device)
    key = SplitKvCalibrationKey(
        device=device_key,
        implementation=splitkv_implementation_identity(),
        route_family=route.family,
    )
    cache = SplitKvCalibrationCache(cache_path)
    entry = cache.get(key)
    if entry is None:
        raise RuntimeError(f"missing current calibration entry for {route.family}")
    workload_key = splitkv_workload_key(workload)
    refined_grain = entry.refinements.get(workload_key)
    policy = (
        select_partition(workload, entry.constants)
        if refined_grain is None
        else predict_partition(
            workload, entry.constants, kv_tiles_per_cta=refined_grain
        )
    )
    candidates = predict_partitions(workload, entry.constants)
    _validate_probe_route(
        route=route, probe=probe, kv_length=kv_length, device=device
    )
    inputs = _allocate_inputs(
        route=route, probe=probe, kv_length=kv_length, device=device
    )
    measured_us = {}
    for index, prediction in enumerate(candidates, start=1):
        grain = prediction.kv_tiles_per_cta
        candidate_calls = [
            _make_call(
                route=route,
                probe=probe,
                inputs=inputs,
                page_table=page_table,
                kv_length=kv_length,
                grain=grain,
            )
            for page_table in inputs.page_tables
        ]
        measured_us[grain] = _measure_graph_replay_us(
            candidate_calls, warmup=warmup_batches, timed=timed_batches
        )["p50"]
        if index % 16 == 0 or index == len(candidates):
            print(
                f"rank={dist.get_rank() if dist.is_initialized() else 0} "
                f"gqa={gqa_group} mtp={mtp_tokens} storage={kv_storage} "
                f"partitions={index}/{len(candidates)}",
                flush=True,
            )
    oracle = min(
        candidates,
        key=lambda item: (measured_us[item.kv_tiles_per_cta], item.num_splits),
    )
    unsplit = next(item for item in candidates if item.num_splits == 1)
    near_optimal = near_optimal_partitions(
        workload,
        entry.constants,
        relative_tolerance=0.10,
    )
    near_optimal_grains = {
        prediction.kv_tiles_per_cta for prediction in near_optimal
    }
    ambiguous = len(near_optimal) > 1
    oracle_covered = oracle.kv_tiles_per_cta in near_optimal_grains
    policy_us = measured_us[policy.kv_tiles_per_cta]
    oracle_us = measured_us[oracle.kv_tiles_per_cta]
    unsplit_us = measured_us[unsplit.kv_tiles_per_cta]

    selected_calls = [
        _make_call(
            route=route,
            probe=probe,
            inputs=inputs,
            page_table=page_table,
            kv_length=kv_length,
            grain=policy.kv_tiles_per_cta,
        )
        for page_table in inputs.page_tables
    ]
    unsplit_calls = [
        _make_call(
            route=route,
            probe=probe,
            inputs=inputs,
            page_table=page_table,
            kv_length=kv_length,
            grain=unsplit.kv_tiles_per_cta,
        )
        for page_table in inputs.page_tables
    ]
    protocols = {
        "cuda_graph_replay": {
            "policy_p50_us": policy_us,
            "unsplit_p50_us": unsplit_us,
            "oracle_p50_us": oracle_us,
        },
        "eager_cuda_event": {
            "policy": _measure_eager_event_us(
                selected_calls, warmup=warmup_batches, timed=timed_batches
            ),
            "unsplit": _measure_eager_event_us(
                unsplit_calls, warmup=warmup_batches, timed=timed_batches
            ),
        },
        "synchronized_wall": {
            "policy": _measure_wall_us(
                selected_calls, warmup=warmup_batches, timed=timed_batches
            ),
            "unsplit": _measure_wall_us(
                unsplit_calls, warmup=warmup_batches, timed=timed_batches
            ),
        },
    }
    model_to_oracle = policy_us / oracle_us
    policy_to_unsplit = policy_us / unsplit_us
    passed = (
        policy_to_unsplit <= 1.0
        and oracle_covered
        and (ambiguous or model_to_oracle <= 1.10)
    )
    if write_refinements:
        cache.save_refinement(key, workload_key, oracle.kv_tiles_per_cta)
    candidate_rows = [
        {
            **asdict(prediction),
            "measured_cuda_graph_p50_us": measured_us[
                prediction.kv_tiles_per_cta
            ],
        }
        for prediction in candidates
    ]
    result = {
        "case": {
            "gqa_group": gqa_group,
            "mtp_tokens": mtp_tokens,
            "kv_storage": kv_storage,
            "kv_length": kv_length,
            "page_size": page_size,
        },
        "device": device_facts,
        "route_family": route.family,
        "workload": asdict(workload),
        "workload_key": workload_key,
        "cache_eligible": True,
        "refined": refined_grain is not None,
        "ambiguous": ambiguous,
        "near_optimal_grains": sorted(near_optimal_grains),
        "oracle_covered_by_model": oracle_covered,
        "policy": asdict(policy),
        "oracle": asdict(oracle),
        "unsplit": asdict(unsplit),
        "legal_partition_count": len(candidates),
        "candidates": candidate_rows,
        "protocols": protocols,
        "model_to_oracle_ratio": model_to_oracle,
        "policy_to_unsplit_ratio": policy_to_unsplit,
        "passed": passed,
        "refinement_written": write_refinements,
    }
    del inputs
    torch.cuda.empty_cache()
    return result


def main() -> None:
    args = _parse_args()
    kv_lengths = (
        (args.kv_length,) if args.kv_length is not None else args.kv_lengths
    )
    if any(kv_length % args.page_size for kv_length in kv_lengths):
        raise ValueError("every KV length must be divisible by page-size")
    distributed = "RANK" in os.environ and "WORLD_SIZE" in os.environ
    if distributed:
        dist.init_process_group("gloo")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        local_rank = int(os.environ["LOCAL_RANK"])
    else:
        rank = 0
        world_size = 1
        local_rank = 0
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    if torch.cuda.get_device_capability(device)[0] != 12:
        raise RuntimeError("exhaustive qualification requires SM120 GPUs")

    tasks = [
        (kv_length, gqa_group, mtp_tokens, kv_storage)
        for kv_length in kv_lengths
        for kv_storage in args.kv_storages
        for gqa_group in args.gqa_groups
        for mtp_tokens in args.mtp_tokens
    ]
    if args.limit_tasks:
        tasks = tasks[: args.limit_tasks]
    local_results = []
    for task_index, (kv_length, gqa_group, mtp_tokens, kv_storage) in enumerate(
        tasks
    ):
        if task_index % world_size != rank:
            continue
        print(
            f"rank={rank} starting kv={kv_length} gqa={gqa_group} "
            f"mtp={mtp_tokens} storage={kv_storage}",
            flush=True,
        )
        local_results.append(
            _run_task(
                gqa_group=gqa_group,
                mtp_tokens=mtp_tokens,
                kv_storage=kv_storage,
                cache_path=args.cache,
                kv_length=kv_length,
                page_size=args.page_size,
                warmup_batches=args.warmup_batches,
                timed_batches=args.timed_batches,
                device=device,
                write_refinements=args.write_refinements,
            )
        )

    if distributed:
        gathered = [None] * world_size
        dist.all_gather_object(gathered, local_results)
        results = [item for rank_items in gathered for item in rank_items]
    else:
        results = local_results
    results.sort(
        key=lambda item: (
            item["case"]["kv_length"],
            item["case"]["gqa_group"],
            item["case"]["mtp_tokens"],
            item["case"]["kv_storage"],
        )
    )
    if rank == 0:
        expected = len(tasks)
        report = {
            "schema_version": SCHEMA_VERSION,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "protocol": {
                "primary": "cuda_graph_replay",
                "kv_pool": "larger_than_L2",
                "page_tables": "randomized_rotating_reuse_distance_gt_L2",
                "submission": "queued_batches",
                "cuda_graph_ring": "one_graph_contains_complete_page_table_ring",
                "warmup_batches": args.warmup_batches,
                "timed_batches": args.timed_batches,
            },
            "kv_lengths": list(kv_lengths),
            "world_size": world_size,
            "write_refinements": args.write_refinements,
            "expected_case_dtype_tasks": expected,
            "completed_case_dtype_tasks": len(results),
            "all_legal_partitions_measured": all(
                item["legal_partition_count"] == item["workload"]["num_n_blocks"]
                for item in results
            ),
            "results": results,
            "summary": {
                "passed_tasks": sum(bool(item["passed"]) for item in results),
                "failed_tasks": sum(not item["passed"] for item in results),
                "ambiguous_tasks": sum(
                    bool(item["ambiguous"]) for item in results
                ),
                "max_model_to_oracle_ratio": max(
                    item["model_to_oracle_ratio"] for item in results
                ),
                "max_policy_to_unsplit_ratio": max(
                    item["policy_to_unsplit_ratio"] for item in results
                ),
            },
        }
        report["passed"] = (
            len(results) == expected
            and report["all_legal_partitions_measured"]
            and report["summary"]["failed_tasks"] == 0
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        if not report["passed"] and not args.write_refinements:
            raise AssertionError("exhaustive SplitKV oracle qualification failed")
    if distributed:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
