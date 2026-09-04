# SGLang SM120 FP8-KV trial kit

This kit validates the SGLang SM120 FA4 path that reads an FP8 E4M3 paged
K/V cache, applies distinct per-tensor K and V scales inside attention, and
keeps QK and PV MMA in BF16.  It is intentionally separate from the SGLang
source tree so trial instructions and result files can be shared without
shipping internal governance or machine configuration.

## Scope

The default matrix represents the current customer workload:

- SM120 GPU;
- BF16 query and output, FP8 E4M3 paged K/V;
- head dimension 128 and page size 64;
- one KV head with GQA group 6, 8, or 16;
- target-verification length 2 through 8;
- KV lengths 128, 256, 512, 768, and 1024;
- automatic SplitKV routing and CUDA Graph replay measurements.

The fused production path is deliberately limited to causal global
target-verification attention with packed GQA and query length at most 8.  It
does not cover cascade attention, context parallelism, sliding/local attention,
sinks, score modification, relative bias, or KV-cache bypass.  SGLang raises a
clear error if a target-verification request reaches one of those combinations.

The scripts must run inside the Python environment of the supplied SGLang
build.  They do not install or replace PyTorch, Triton, CUDA, or SGLang.

## Quick start

From this directory, with the supplied SGLang environment active:

```bash
python scripts/check_environment.py

python scripts/prepare_calibration.py \
  --cache results/calibration.json \
  --output-dir results/calibration

python scripts/run_short_kv.py \
  --cache results/calibration.json \
  --output-dir results/matrix

python scripts/summarize.py results/matrix/*.json
```

`prepare_calibration.py` takes longer than the matrix itself because it fits
the BF16 and FP8 constants for both M64N64 and M64N128 route families over a
multi-scale, L2-cold KV-tile grid.  It does not encode workload-specific route
answers.  `run_short_kv.py` generically refines only actual workloads whose
best modeled candidates lie inside the 10-percent regret envelope.  Reuse the
resulting cache while the GPU model and supplied SGLang source revision remain
unchanged.  Source or hardware changes intentionally invalidate the cache.

All commands return non-zero on an unsupported environment, a failed case, or
an invalid result.  Generated results are ignored by Git; share the JSON files
through the agreed secure channel instead of committing machine information.

Run latency measurements on an exclusive GPU.  Check `nvidia-smi` immediately
before and during a run; another process issuing short kernels on the same GPU
can materially distort these microsecond-scale results.  Correctness-only
results are less sensitive, but should still use the same isolated procedure.

## Server trial

Use the validated cache with the supplied SGLang checkout:

```bash
export SGLANG_FA4_SPLITKV_CALIBRATION=tune
export SGLANG_FA4_SPLITKV_CALIBRATION_CACHE="$(pwd)/results/calibration.json"

python -m sglang.launch_server \
  --model-path <MODEL> \
  --attention-backend fa4 \
  --kv-cache-dtype fp8_e4m3
```

Keep the remaining model, tensor-parallel, and speculative/MTP arguments from
the customer's qualified launch configuration.  A cache generated on another
GPU model or SGLang source revision is rejected; rerun
`prepare_calibration.py` after either changes.

Use `tune` for the first model CUDA Graph warmup so model-ambiguous actual
workloads receive bounded refinement.  Once those decisions are cached,
subsequent launches may switch `SGLANG_FA4_SPLITKV_CALIBRATION` to `load`.

## Direct and advanced entry points

- `scripts/bench_fp8_kv.py` runs one KV length and emits detailed correctness,
  component latency, eager, wall, and CUDA Graph distributions.
- `scripts/calibrate_splitkv.py` calibrates one route family.
- `scripts/qualify_splitkv.py` exhaustively measures every legal partition and
  is intended for performance qualification rather than a first smoke test.
- `scripts/validate_tp.py` verifies leader-only calibration and broadcast with
  `torchrun` on a homogeneous multi-GPU node.

The customer-facing benchmark calls
`sglang.kernels.ops.attention.flash_attention_v4_sm120.flash_attn_with_kvcache`.
It does not bypass production dispatch by calling the fused kernel directly.
