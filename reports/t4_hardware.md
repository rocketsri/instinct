# T4 hardware findings

Measured 2026-08-16 on a Colab T4 via `colab run --gpu T4 benchmarks/t4_probe.py`.
Device: Tesla T4, sm_75, 14.6 GiB, torch 2.11.0+cu128.

These numbers exist because three plan assumptions were stated from API semantics
rather than measurement, and the obvious API calls turn out to be misleading on
sm_75. One assumption survived, one was confirmed the hard way, and one was
**wrong and has been reverted**.

## Summary

| Claim | Verdict | Evidence |
| --- | --- | --- |
| Avoid bf16, use fp16 | **confirmed** | fp16 is 9.54x faster than bf16 |
| FlashAttention-2 unavailable | **confirmed** | explicit `RuntimeError`, sm_75 out of range |
| Fast weights are launch-bound | **confirmed, strongly** | width 64→512 costs the same wall-clock |
| Batched forking pays off | **confirmed** | 14.2x at 128 forks |
| `torch.compile(reduce-overhead)` helps | **REFUTED** | 0.45x — it is *slower* |

## 1. bf16 is emulated, not accelerated

`torch.cuda.is_bf16_supported()` returns **True** on a T4, which is misleading:
it counts emulation. T4 tensor cores are fp16-only.

| dtype | TFLOP/s (4096³ matmul) |
| --- | --- |
| fp16 | ~20.6 |
| fp32 | ~4.0 |
| bf16 | **2.16** |

bf16 is 9.5x slower than fp16 and even ~2x slower than fp32. The plan's
"no bf16" instruction stands — it was right in substance despite the boolean.

**Action:** fp16 autocast with an fp32 master copy, as planned. Never bf16.

## 2. FlashAttention-2 is genuinely unavailable

`torch.backends.cuda.flash_sdp_enabled()` also returns True, but that reports
backend *preference*, not hardware capability. Forcing the backend raises:

> Flash attention only supports gpu architectures in the range [sm80, sm121].
> Attempting to run on a sm 7.5 gpu.

| backend | result |
| --- | --- |
| flash (FA2) | `RuntimeError` — unavailable |
| mem_efficient | 0.351 ms |
| math | 3.434 ms |

**Action:** do not import or configure FA2. Use the default SDPA dispatch, which
lands ~10x ahead of the math fallback.

*Caveat:* the run also emitted "Memory Efficient attention has been runtime
disabled" warnings whose attribution to a specific backend block is ambiguous,
so the 0.351 ms figure should be treated as "the default dispatch path is ~10x
faster than math" rather than as a confirmed identification of the kernel. The
decision does not depend on resolving it.

## 3. The fast-weight regime is launch-bound — emphatically

A SwiGLU update at P2 scale, batch 32:

| width | time | GFLOP/s |
| --- | --- | --- |
| 64 | 80.7 µs | 26 |
| 128 | 82.7 µs | 101 |
| 256 | 89.9 µs | 373 |
| 512 | 84.8 µs | 1583 |
| 1024 | 121.7 µs | 4413 |

Width grows 8x from 64 to 512 — a 64x FLOP increase — at **constant wall-clock**.
The GPU is doing nothing but waiting on launches. Real work only starts showing
at width 1024.

**Action:** two consequences. Keep the fast-weight state at width ≤512, where
capacity is free in wall-clock terms. And treat FLOP counts as near-worthless
for scheduling decisions in this regime — `compute/budget.py` must report
wall-clock as the primary matched-compute currency, with FLOPs secondary. The
plan already required logging both; this says which one to trust here.

## 4. Batched forking: 14x, as premised

The P2 oracle forks many candidate writes and evaluates them together. Batched
`bmm` against a sequential Python loop over the same forks:

| forks | batched | sequential | speedup |
| --- | --- | --- | --- |
| 1 | 87.9 µs | 114.0 µs | 1.3x |
| 8 | 92.0 µs | 761.2 µs | 8.3x |
| 32 | 205.7 µs | 2804.8 µs | 13.6x |
| 128 | 734.6 µs | 10455.3 µs | 14.2x |

Note the shape: 8 forks cost essentially the same as 1. The oracle should
therefore fork **at least 32 candidates at a time**, and there is no reason to
fork fewer than 8 ever.

## 5. `torch.compile(mode="reduce-overhead")` — refuted, reverted

Predicted a win on the grounds that CUDA graphs target exactly the
launch-overhead-bound regime section 3 confirms we are in. Measured:

| path | time |
| --- | --- |
| eager | 84.7 µs |
| compiled | **188.9 µs (0.45x)** |

Plus a 17.4 s one-off compile cost, and inductor warning
`Not enough SMs to use max_autotune_gemm mode` — a T4's 40 SMs are too few for
the autotuned GEMM path, and the CUDA-graph machinery costs more than the
launches it elides at this size.

**Action:** `torch.compile` is **off by default**. The config flag stays, since
the picture may differ for the larger LM arm, but it must re-earn its place with
a measurement there. Batched forking (section 4) is the real answer to launch
overhead, and it is 14x rather than 2x.

This is the intended behaviour of the rule that no optimization ships without a
profile: the prediction was reasonable, the measurement disagreed, and the
measurement wins.

## Reproducing

```bash
bash scripts/colab_setup.sh                       # auth + entitlement check
colab run --gpu T4 benchmarks/t4_probe.py         # this table
```

`colab run` provisions, executes and tears down in one command, self-cleaning
even when the script raises. Prefer it over `colab new` for one-shot jobs: an
unstopped session burns compute units until the 24 h keep-alive cap.
