# Copyright (c) 2026, SGLang Team.
"""Compare fused FP8-KV FA4 with dequantization followed by BF16 FA4.

The unfused baseline materializes BF16 K/V outside FlashAttention.  The fused
path consumes the same FP8 E4M3 tensors and per-tensor scales through SGLang's
SM120 wrapper, keeping BF16 QK and PV MMA.

Run from the customer kit inside the patched SGLang environment::

    python scripts/bench_fp8_kv.py --modes splitkv --splitkv-splits 0 \
        --kv-length 1024 --run-fused --output results/kv-1024.json

``mtp_tokens`` is the total number of target-verify query tokens in one
sequence (the accepted token plus draft tokens), not the number of draft-model
layers.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import statistics
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable, Sequence

import torch
import triton
import triton.language as tl

from sglang.kernels.ops.attention.fa4_sm120.runtime import (
    sm120_forward_host,
    splitkv_calibration_partition,
)
from sglang.kernels.ops.attention.flash_attention_v4_sm120 import (
    flash_attn_with_kvcache,
)
from sglang.kernels.ops.attention.flash_attn.cute.interface import (
    num_splits_heuristic,
)


SCHEMA_VERSION = "sglang.fa4_fp8_kv_baseline.v2"
FP8_DTYPE = torch.float8_e4m3fn
FP8_MAX = torch.finfo(FP8_DTYPE).max
VALID_MODES = ("decode", "splitkv")


@triton.jit
def _dequantize_kv_kernel(
    k_fp8,
    v_fp8,
    k_scale,
    v_scale,
    k_bf16,
    v_bf16,
    num_elements: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < num_elements
    resolved_k_scale = tl.load(k_scale)
    resolved_v_scale = tl.load(v_scale)
    k = tl.load(k_fp8 + offsets, mask=mask).to(tl.float32) * resolved_k_scale
    v = tl.load(v_fp8 + offsets, mask=mask).to(tl.float32) * resolved_v_scale
    tl.store(k_bf16 + offsets, k, mask=mask)
    tl.store(v_bf16 + offsets, v, mask=mask)


@dataclass(frozen=True)
class Case:
    gqa_group: int
    mtp_tokens: int
    mode: str


@dataclass
class Inputs:
    q: torch.Tensor
    k_fp8: torch.Tensor
    v_fp8: torch.Tensor
    k_scale: torch.Tensor
    v_scale: torch.Tensor
    k_bf16: torch.Tensor
    v_bf16: torch.Tensor
    page_table: torch.Tensor
    cache_seqlens: torch.Tensor
    cu_seqlens_q: torch.Tensor
    out: torch.Tensor


def _parse_int_list(value: str) -> list[int]:
    values = [int(part.strip()) for part in value.split(",") if part.strip()]
    if not values or any(item <= 0 for item in values):
        raise argparse.ArgumentTypeError(
            "expected a comma-separated list of positive integers"
        )
    return values


def _parse_modes(value: str) -> list[str]:
    modes = [part.strip().lower() for part in value.split(",") if part.strip()]
    invalid = sorted(set(modes) - set(VALID_MODES))
    if not modes or invalid:
        expected = ",".join(VALID_MODES)
        raise argparse.ArgumentTypeError(
            f"expected a comma-separated subset of {expected}; invalid={invalid}"
        )
    return modes


def _percentile(sorted_values: Sequence[float], fraction: float) -> float:
    if not sorted_values:
        raise ValueError("cannot summarize an empty sample")
    position = fraction * (len(sorted_values) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(sorted_values[lower])
    weight = position - lower
    return float(sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight)


def _summarize_ms(samples: Sequence[float]) -> dict[str, float | int]:
    ordered = sorted(float(sample) for sample in samples)
    return {
        "count": len(ordered),
        "mean": statistics.fmean(ordered),
        "min": ordered[0],
        "p05": _percentile(ordered, 0.05),
        "p50": _percentile(ordered, 0.50),
        "p95": _percentile(ordered, 0.95),
        "max": ordered[-1],
    }


def _time_cuda_ms(
    fn: Callable[[], object], *, warmup: int, trials: int
) -> dict[str, float | int]:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    starts = [torch.cuda.Event(enable_timing=True) for _ in range(trials)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(trials)]
    for start, end in zip(starts, ends):
        start.record()
        fn()
        end.record()
    ends[-1].synchronize()
    return _summarize_ms([start.elapsed_time(end) for start, end in zip(starts, ends)])


def _time_wall_ms(
    fn: Callable[[], object], *, warmup: int, trials: int
) -> dict[str, float | int]:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    samples = []
    for _ in range(trials):
        started = time.perf_counter_ns()
        fn()
        torch.cuda.synchronize()
        samples.append((time.perf_counter_ns() - started) / 1e6)
    return _summarize_ms(samples)


def _capture_cuda_graph(
    fn: Callable[[], object], *, warmup: int
) -> torch.cuda.CUDAGraph:
    # Compile and populate allocator / launch-plan caches before capture. The
    # capture itself still resolves the graph-specific SM120 PDL plan once;
    # replay measures only the recorded GPU work and graph launch.
    for _ in range(max(warmup, 1)):
        fn()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        fn()
    torch.cuda.synchronize()
    return graph


def _measure_component(
    fn: Callable[[], object], *, warmup: int, trials: int
) -> dict[str, dict[str, float | int]]:
    graph = _capture_cuda_graph(fn, warmup=warmup)
    return {
        "cuda_graph_replay": _time_cuda_ms(
            graph.replay,
            warmup=warmup,
            trials=trials,
        ),
        "eager_cuda_event": _time_cuda_ms(fn, warmup=warmup, trials=trials),
        "synchronized_wall": _time_wall_ms(fn, warmup=warmup, trials=trials),
    }


def _quantize_per_tensor(value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    scale = (value.float().abs().amax() / FP8_MAX).clamp_min(1e-12)
    quantized = (value.float() / scale).clamp(-FP8_MAX, FP8_MAX).to(FP8_DTYPE)
    return quantized, scale


def _make_inputs(
    case: Case,
    *,
    batch_size: int,
    kv_heads: int,
    head_dim: int,
    kv_length: int,
    page_size: int,
    seed: int,
    device: torch.device,
) -> Inputs:
    if kv_length % page_size != 0:
        raise ValueError("kv_length must be divisible by page_size")
    q_heads = kv_heads * case.gqa_group
    pages_per_sequence = kv_length // page_size
    num_pages = batch_size * pages_per_sequence
    generator = torch.Generator(device=device).manual_seed(seed)

    q = torch.randn(
        batch_size * case.mtp_tokens,
        q_heads,
        head_dim,
        generator=generator,
        device=device,
        dtype=torch.bfloat16,
    )
    k_source = torch.randn(
        num_pages,
        page_size,
        kv_heads,
        head_dim,
        generator=generator,
        device=device,
        dtype=torch.bfloat16,
    )
    v_source = torch.randn(
        k_source.shape,
        generator=generator,
        device=device,
        dtype=torch.bfloat16,
    )
    k_fp8, k_scale = _quantize_per_tensor(k_source)
    v_fp8, v_scale = _quantize_per_tensor(v_source)
    page_table = torch.randperm(
        num_pages,
        generator=generator,
        device=device,
        dtype=torch.int64,
    ).to(torch.int32).view(batch_size, pages_per_sequence)

    return Inputs(
        q=q,
        k_fp8=k_fp8,
        v_fp8=v_fp8,
        k_scale=k_scale,
        v_scale=v_scale,
        k_bf16=torch.empty_like(k_fp8, dtype=torch.bfloat16),
        v_bf16=torch.empty_like(v_fp8, dtype=torch.bfloat16),
        page_table=page_table,
        cache_seqlens=torch.full(
            (batch_size,), kv_length, device=device, dtype=torch.int32
        ),
        cu_seqlens_q=(
            torch.arange(batch_size + 1, device=device, dtype=torch.int32)
            * case.mtp_tokens
        ),
        out=torch.empty_like(q),
    )


def _dequantize(inputs: Inputs, *, block_size: int, num_warps: int) -> None:
    num_elements = inputs.k_fp8.numel()
    _dequantize_kv_kernel[(triton.cdiv(num_elements, block_size),)](
        inputs.k_fp8,
        inputs.v_fp8,
        inputs.k_scale,
        inputs.v_scale,
        inputs.k_bf16,
        inputs.v_bf16,
        num_elements=num_elements,
        BLOCK_SIZE=block_size,
        num_warps=num_warps,
    )


def _attention_reference(
    inputs: Inputs,
    k_bf16: torch.Tensor,
    v_bf16: torch.Tensor,
    *,
    batch_size: int,
    mtp_tokens: int,
) -> torch.Tensor:
    """FP32 math reference over independently reconstructed BF16 K/V."""
    outputs = []
    for batch_idx in range(batch_size):
        q = inputs.q[
            batch_idx * mtp_tokens : (batch_idx + 1) * mtp_tokens
        ].float()
        pages = inputs.page_table[batch_idx].long()
        k = k_bf16.index_select(0, pages).flatten(0, 1).float()
        v = v_bf16.index_select(0, pages).flatten(0, 1).float()
        group = q.shape[1] // k.shape[1]
        k = k.repeat_interleave(group, dim=1)
        v = v.repeat_interleave(group, dim=1)
        scores = torch.einsum("qhd,khd->hqk", q, k) * (q.shape[-1] ** -0.5)

        q_idx = torch.arange(mtp_tokens, device=q.device)
        kv_idx = torch.arange(k.shape[0], device=q.device)
        relative_position = q_idx[:, None] + k.shape[0] - mtp_tokens - kv_idx[None]
        scores.masked_fill_(relative_position[None] < 0, -torch.inf)
        probabilities = torch.softmax(scores, dim=-1)
        outputs.append(torch.einsum("hqk,khd->qhd", probabilities, v))
    return torch.cat(outputs, dim=0)


def _tensor_correctness(
    actual: torch.Tensor,
    expected: torch.Tensor,
    *,
    atol: float,
    rtol: float,
) -> dict[str, float | bool]:
    error = (actual.float() - expected.float()).abs()
    close = torch.isclose(actual.float(), expected.float(), atol=atol, rtol=rtol)
    return {
        "passed": bool(close.all().item()),
        "atol": atol,
        "rtol": rtol,
        "max_abs_error": float(error.max().item()),
        "mean_abs_error": float(error.mean().item()),
    }


def _run_case(
    case: Case,
    *,
    batch_size: int,
    kv_heads: int,
    head_dim: int,
    kv_length: int,
    page_size: int,
    splitkv_splits: int,
    splitkv_kv_tiles_per_cta: int | None,
    dequant_block_size: int,
    dequant_num_warps: int,
    warmup: int,
    trials: int,
    seed: int,
    atol: float,
    rtol: float,
    device: torch.device,
    run_fused: bool,
) -> dict[str, object]:
    inputs = _make_inputs(
        case,
        batch_size=batch_size,
        kv_heads=kv_heads,
        head_dim=head_dim,
        kv_length=kv_length,
        page_size=page_size,
        seed=seed,
        device=device,
    )
    num_splits = 1 if case.mode == "decode" else splitkv_splits
    packed_q_rows = case.mtp_tokens * case.gqa_group
    num_sms = torch.cuda.get_device_properties(device).multi_processor_count
    selected_config = sm120_forward_host.select_config(
        head_dim=head_dim,
        head_dim_v=head_dim,
        tile_mn=None,
        has_bias=False,
        total_q_rows=inputs.q.shape[0] * inputs.q.shape[1],
        num_sms=num_sms,
        num_batch=batch_size,
        seqlen_q=case.mtp_tokens,
        seqlen_k=kv_length,
        num_head_kv=kv_heads,
        qhead_per_kvhead=case.gqa_group,
        is_causal=True,
        is_local=False,
        window_size_left=None,
        window_size_right=None,
        pack_gqa=True,
        paged_kv=True,
    )
    num_m_blocks = math.ceil(packed_q_rows / selected_config.tile_m)
    total_m_blocks = batch_size * kv_heads * num_m_blocks
    num_n_blocks = math.ceil(kv_length / selected_config.tile_n)
    calibration_grain = (
        splitkv_kv_tiles_per_cta if case.mode == "splitkv" else None
    )
    def resolve_plan(k_cache: torch.Tensor, v_cache: torch.Tensor):
        with (
            splitkv_calibration_partition(calibration_grain)
            if calibration_grain is not None
            else contextlib.nullcontext()
        ):
            return sm120_forward_host.resolve_plan(
                requested_num_splits=num_splits,
                generic_num_n_blocks=num_n_blocks,
                head_dim=head_dim,
                head_dim_v=head_dim,
                batch_size=batch_size,
                num_head_kv=kv_heads,
                paged_kv=True,
                page_size=page_size,
                k=k_cache,
                v=v_cache,
                max_seqlen_q=case.mtp_tokens,
                max_seqlen_k=kv_length,
                pack_gqa=True,
                compute_dtype=inputs.q.dtype,
                element_size=k_cache.element_size(),
                packed_q_rows=packed_q_rows,
                tile_m=selected_config.tile_m,
                tile_n=selected_config.tile_n,
                num_m_blocks=num_m_blocks,
                total_mblocks=total_m_blocks,
                num_sms=num_sms,
                total_q=inputs.q.shape[0],
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

    bf16_plan = resolve_plan(inputs.k_bf16, inputs.v_bf16)
    fp8_plan = resolve_plan(inputs.k_fp8, inputs.v_fp8) if run_fused else None
    actual_num_splits = bf16_plan.num_splits

    def partition_context():
        return (
            splitkv_calibration_partition(calibration_grain)
            if calibration_grain is not None
            else contextlib.nullcontext()
        )

    def run_fa4() -> torch.Tensor:
        with partition_context():
            return flash_attn_with_kvcache(
                q=inputs.q,
                k_cache=inputs.k_bf16,
                v_cache=inputs.v_bf16,
                page_table=inputs.page_table,
                cache_seqlens=inputs.cache_seqlens,
                cu_seqlens_q=inputs.cu_seqlens_q,
                max_seqlen_q=case.mtp_tokens,
                max_seqlen_k=kv_length,
                softmax_scale=head_dim**-0.5,
                causal=True,
                num_splits=num_splits,
                pack_gqa=True,
                out=inputs.out,
            )

    def run_combined() -> torch.Tensor:
        _dequantize(
            inputs,
            block_size=dequant_block_size,
            num_warps=dequant_num_warps,
        )
        return run_fa4()

    def run_fused_fa4() -> torch.Tensor:
        with partition_context():
            return flash_attn_with_kvcache(
                q=inputs.q,
                k_cache=inputs.k_fp8,
                v_cache=inputs.v_fp8,
                page_table=inputs.page_table,
                cache_seqlens=inputs.cache_seqlens,
                cu_seqlens_q=inputs.cu_seqlens_q,
                max_seqlen_q=case.mtp_tokens,
                max_seqlen_k=kv_length,
                k_descale=inputs.k_scale.expand(batch_size, kv_heads),
                v_descale=inputs.v_scale.expand(batch_size, kv_heads),
                softmax_scale=head_dim**-0.5,
                causal=True,
                num_splits=num_splits,
                pack_gqa=True,
                out=inputs.out,
            )

    _dequantize(
        inputs,
        block_size=dequant_block_size,
        num_warps=dequant_num_warps,
    )
    output = run_fa4().clone()
    expected_k_bf16 = (inputs.k_fp8.float() * inputs.k_scale).to(torch.bfloat16)
    expected_v_bf16 = (inputs.v_fp8.float() * inputs.v_scale).to(torch.bfloat16)
    reference = _attention_reference(
        inputs,
        expected_k_bf16,
        expected_v_bf16,
        batch_size=batch_size,
        mtp_tokens=case.mtp_tokens,
    )
    k_correctness = _tensor_correctness(
        inputs.k_bf16, expected_k_bf16, atol=0.0, rtol=0.0
    )
    v_correctness = _tensor_correctness(
        inputs.v_bf16, expected_v_bf16, atol=0.0, rtol=0.0
    )
    attention_correctness = _tensor_correctness(
        output, reference, atol=atol, rtol=rtol
    )
    fused_correctness = None
    fused_output = None
    if run_fused:
        fused_output = run_fused_fa4().clone()
        fused_correctness = _tensor_correctness(
            fused_output, reference, atol=atol, rtol=rtol
        )
    correctness = {
        "passed": (
            k_correctness["passed"]
            and v_correctness["passed"]
            and attention_correctness["passed"]
            and (fused_correctness is None or fused_correctness["passed"])
        ),
        "materialization": {
            "reference": "fp8_to_fp32_times_resolved_scale_then_bf16_round",
            "k": k_correctness,
            "v": v_correctness,
        },
        "attention": {
            "reference": "fp32_sdpa_over_expected_bf16_dequantized_kv",
            **attention_correctness,
        },
    }
    if fused_correctness is not None:
        correctness["fused_attention"] = {
            "reference": "fp32_sdpa_over_expected_bf16_dequantized_kv",
            **fused_correctness,
        }

    component_protocols = {}
    component_protocols["kv_dequant"] = _measure_component(
        lambda: _dequantize(
            inputs,
            block_size=dequant_block_size,
            num_warps=dequant_num_warps,
        ),
        warmup=warmup,
        trials=trials,
    )
    component_protocols["bf16_fa4"] = _measure_component(
        run_fa4,
        warmup=warmup,
        trials=trials,
    )
    component_protocols["end_to_end"] = _measure_component(
        run_combined,
        warmup=warmup,
        trials=trials,
    )
    if run_fused:
        component_protocols["fused_fp8_kv_fa4"] = _measure_component(
            run_fused_fa4,
            warmup=warmup,
            trials=trials,
        )
    protocols = {
        protocol: {
            "components": {
                component: measurements[protocol]
                for component, measurements in component_protocols.items()
            }
        }
        for protocol in (
            "cuda_graph_replay",
            "eager_cuda_event",
            "synchronized_wall",
        )
    }
    for protocol in protocols.values():
        protocol_components = protocol["components"]
        protocol["standalone_dequant_to_end_to_end_p50_ratio"] = float(
            protocol_components["kv_dequant"]["p50"]
        ) / float(protocol_components["end_to_end"]["p50"])
        protocol["end_to_end_increment_over_fa4_p50_ms"] = float(
            protocol_components["end_to_end"]["p50"]
        ) - float(protocol_components["bf16_fa4"]["p50"])
        protocol["end_to_end_increment_over_fa4_p50_ratio"] = float(
            protocol["end_to_end_increment_over_fa4_p50_ms"]
        ) / float(protocol_components["bf16_fa4"]["p50"])
        if run_fused:
            protocol["fusion_speedup_over_unfused_end_to_end_p50"] = float(
                protocol_components["end_to_end"]["p50"]
            ) / float(protocol_components["fused_fp8_kv_fa4"]["p50"])

    # Preserve the v1 component location as the eager CUDA-event view while
    # making graph replay the v2 comparison authority.
    components = protocols["eager_cuda_event"]["components"]
    graph_components = protocols["cuda_graph_replay"]["components"]
    dequant_bytes_read = 2 * inputs.k_fp8.numel() * inputs.k_fp8.element_size()
    dequant_bytes_written = (
        inputs.k_bf16.numel() * inputs.k_bf16.element_size()
        + inputs.v_bf16.numel() * inputs.v_bf16.element_size()
    )
    dequant_p50_seconds = float(graph_components["kv_dequant"]["p50"]) / 1e3

    result = {
        "status": "passed" if correctness["passed"] else "failed",
        "implementation": "unfused_fp8_kv_dequant_bf16_fa4",
        "case": {
            "gqa_group": case.gqa_group,
            "mtp_tokens": case.mtp_tokens,
            "mtp_definition": "total_target_verify_query_tokens",
            "mode": case.mode,
            "num_splits": num_splits,
            "actual_num_splits": actual_num_splits,
            "fused_actual_num_splits": (
                None if fp8_plan is None else fp8_plan.num_splits
            ),
            "kv_tiles_per_split_cta_override": calibration_grain,
            "input_seed": seed,
        },
        "shape": {
            "batch_size": batch_size,
            "q_heads": kv_heads * case.gqa_group,
            "kv_heads": kv_heads,
            "head_dim": head_dim,
            "query_length_per_sequence": case.mtp_tokens,
            "kv_length_per_sequence": kv_length,
            "page_size": page_size,
            "num_physical_pages": inputs.k_fp8.shape[0],
        },
        "launch": {
            "packed_q_rows_per_kv_head": packed_q_rows,
            "tile_m": selected_config.tile_m,
            "tile_n": selected_config.tile_n,
            "num_threads": selected_config.num_threads,
            "num_m_blocks_per_batch_kv_head": num_m_blocks,
            "num_n_blocks": num_n_blocks,
            "requested_num_splits": num_splits,
            "actual_num_splits": actual_num_splits,
            "fused_actual_num_splits": (
                None if fp8_plan is None else fp8_plan.num_splits
            ),
            "selected_kv_tiles_per_split_cta": (
                bf16_plan.split_kv_blocks_per_cta
                or math.ceil(num_n_blocks / bf16_plan.num_splits)
            ),
            "fused_selected_kv_tiles_per_split_cta": (
                None
                if fp8_plan is None
                else fp8_plan.split_kv_blocks_per_cta
                or math.ceil(num_n_blocks / fp8_plan.num_splits)
            ),
            "kv_tiles_per_split_cta_override": calibration_grain,
            "main_ctas": total_m_blocks * actual_num_splits,
            "kv_blocks_per_split_cta": (
                calibration_grain
                if calibration_grain is not None
                else math.ceil(num_n_blocks / actual_num_splits)
            ),
            "cached_paged_decode_eligible": (
                sm120_forward_host._supports_cached_paged_decode(
                    head_dim=head_dim,
                    head_dim_v=head_dim,
                    effective_q_rows=packed_q_rows,
                )
            ),
        },
        "storage": {
            "k_dtype": str(inputs.k_fp8.dtype),
            "v_dtype": str(inputs.v_fp8.dtype),
            "scale_dtype": str(inputs.k_scale.dtype),
            "scale_granularity": "per_tensor",
            "scale_policy": "max_abs_divided_by_fp8_max",
            "k_scale": float(inputs.k_scale.item()),
            "v_scale": float(inputs.v_scale.item()),
            "materialized_dtype": str(inputs.k_bf16.dtype),
            "q_dtype": str(inputs.q.dtype),
            "qk_compute_dtype": "torch.bfloat16",
            "pv_compute_dtype": "torch.bfloat16",
        },
        "measurement": {
            "warmup_iterations_per_component": warmup,
            "trial_iterations_per_component": trials,
            "unit": "ms",
            "dequant_kernel": {
                "provider": "triton",
                "block_size": dequant_block_size,
                "num_warps": dequant_num_warps,
                "bytes_read": dequant_bytes_read,
                "bytes_written": dequant_bytes_written,
                "effective_gbps_p50": (
                    (dequant_bytes_read + dequant_bytes_written)
                    / dequant_p50_seconds
                    / 1e9
                ),
            },
            "interpretation": (
                "CUDA Graph replay is the kernel-focused comparison authority; "
                "eager CUDA events may include uncovered host dispatch gaps, and "
                "synchronized wall time includes host plus GPU latency. Component "
                "distributions are independently warmed and measured; end-to-end "
                "includes cache-locality effects and is not their sum."
            ),
            "primary_protocol": "cuda_graph_replay",
            "components": components,
            "protocols": protocols,
            "standalone_dequant_to_end_to_end_p50_ratio": (
                protocols["eager_cuda_event"][
                    "standalone_dequant_to_end_to_end_p50_ratio"
                ]
            ),
            "end_to_end_increment_over_fa4_p50_ms": (
                protocols["eager_cuda_event"][
                    "end_to_end_increment_over_fa4_p50_ms"
                ]
            ),
            "end_to_end_increment_over_fa4_p50_ratio": (
                protocols["eager_cuda_event"][
                    "end_to_end_increment_over_fa4_p50_ratio"
                ]
            ),
        },
        "correctness": correctness,
    }
    if run_fused:
        result["measurement"]["fusion_speedup_over_unfused_end_to_end_p50"] = (
            protocols["eager_cuda_event"][
                "fusion_speedup_over_unfused_end_to_end_p50"
            ]
        )
    return result


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gqa-groups", type=_parse_int_list, default=[6, 8, 16])
    parser.add_argument(
        "--mtp-tokens", type=_parse_int_list, default=[2, 3, 4, 5, 6, 7, 8]
    )
    parser.add_argument("--modes", type=_parse_modes, default=list(VALID_MODES))
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--kv-heads", type=int, default=1)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--kv-length", type=int, default=8192)
    parser.add_argument("--page-size", type=int, default=64)
    parser.add_argument("--splitkv-splits", type=int, default=8)
    parser.add_argument(
        "--splitkv-kv-tiles-per-cta",
        type=int,
        help="Internal calibration override; derives splits from this KV-tile grain.",
    )
    parser.add_argument("--dequant-block-size", type=int, default=512)
    parser.add_argument("--dequant-num-warps", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--trials", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260903)
    parser.add_argument("--atol", type=float, default=0.02)
    parser.add_argument("--rtol", type=float, default=0.02)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--run-fused",
        action="store_true",
        help="Also run the SM120 fused FP8-KV implementation.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional JSON output path. The complete document is always printed.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Write the JSON output without also printing the complete document.",
    )
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    positive = {
        "batch_size": args.batch_size,
        "kv_heads": args.kv_heads,
        "head_dim": args.head_dim,
        "kv_length": args.kv_length,
        "page_size": args.page_size,
        "dequant_block_size": args.dequant_block_size,
        "dequant_num_warps": args.dequant_num_warps,
        "trials": args.trials,
    }
    invalid = {name: value for name, value in positive.items() if value <= 0}
    if invalid:
        raise ValueError(f"arguments must be positive: {invalid}")
    if args.warmup < 0:
        raise ValueError("warmup must be non-negative")
    if args.splitkv_splits < 0:
        raise ValueError("splitkv_splits must be non-negative")
    if args.splitkv_splits == 1 and "splitkv" in args.modes:
        raise ValueError(
            "splitkv_splits must be zero (automatic) or greater than one"
        )
    if (
        args.splitkv_kv_tiles_per_cta is not None
        and args.splitkv_kv_tiles_per_cta <= 0
    ):
        raise ValueError("splitkv_kv_tiles_per_cta must be positive")
    if args.dequant_block_size & (args.dequant_block_size - 1):
        raise ValueError("dequant_block_size must be a power of two")
    if args.dequant_num_warps not in (1, 2, 4, 8):
        raise ValueError("dequant_num_warps must be one of 1, 2, 4, or 8")
    if args.kv_length % args.page_size:
        raise ValueError("kv_length must be divisible by page_size")
    for name in ("gqa_groups", "mtp_tokens", "modes"):
        values = getattr(args, name)
        if len(values) != len(set(values)):
            raise ValueError(f"{name} must not contain duplicates: {values}")


def main(argv: Iterable[str] | None = None) -> int:
    args = _build_parser().parse_args(list(argv) if argv is not None else None)
    _validate_args(args)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device(args.device)
    capability = torch.cuda.get_device_capability(device)
    if capability[0] != 12:
        raise RuntimeError(
            f"This benchmark requires SM12x; device {device} reports "
            f"{capability[0]}.{capability[1]}."
        )

    cases_with_seeds = []
    pair_index = 0
    for gqa_group in args.gqa_groups:
        for mtp_tokens in args.mtp_tokens:
            pair_seed = args.seed + pair_index
            cases_with_seeds.extend(
                (
                    Case(gqa_group=gqa_group, mtp_tokens=mtp_tokens, mode=mode),
                    pair_seed,
                )
                for mode in args.modes
            )
            pair_index += 1
    results = []
    for case, case_seed in cases_with_seeds:
        try:
            result = _run_case(
                case,
                batch_size=args.batch_size,
                kv_heads=args.kv_heads,
                head_dim=args.head_dim,
                kv_length=args.kv_length,
                page_size=args.page_size,
                splitkv_splits=args.splitkv_splits,
                splitkv_kv_tiles_per_cta=args.splitkv_kv_tiles_per_cta,
                dequant_block_size=args.dequant_block_size,
                dequant_num_warps=args.dequant_num_warps,
                warmup=args.warmup,
                trials=args.trials,
                seed=case_seed,
                atol=args.atol,
                rtol=args.rtol,
                device=device,
                run_fused=args.run_fused,
            )
        except Exception as error:  # Preserve partial matrix evidence in JSON.
            result = {
                "status": "error",
                "implementation": "unfused_fp8_kv_dequant_bf16_fa4",
                "case": {
                    "gqa_group": case.gqa_group,
                    "mtp_tokens": case.mtp_tokens,
                    "mtp_definition": "total_target_verify_query_tokens",
                    "mode": case.mode,
                    "num_splits": (
                        1 if case.mode == "decode" else args.splitkv_splits
                    ),
                    "input_seed": case_seed,
                },
                "error": f"{type(error).__name__}: {error}",
            }
        results.append(result)

    document = {
        "schema_version": SCHEMA_VERSION,
        "benchmark": "sm120_fa4_fp8_kv_dequant_baseline",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "device": {
            "name": torch.cuda.get_device_name(device),
            "compute_capability": f"{capability[0]}.{capability[1]}",
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
        },
        "matrix": {
            "gqa_groups": args.gqa_groups,
            "mtp_tokens": args.mtp_tokens,
            "modes": args.modes,
            "expected_customer_pairs": len(args.gqa_groups) * len(args.mtp_tokens),
            "executed_cases": len(cases_with_seeds),
        },
        "results": results,
    }
    rendered = json.dumps(document, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    if not args.quiet:
        print(rendered)
    return 0 if all(result["status"] == "passed" for result in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
