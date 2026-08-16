"""Measure the T4 facts that Stage 2's design depends on.

Three plan assumptions need checking against hardware rather than against API
semantics, because the obvious API calls are misleading on sm_75:

- ``torch.cuda.is_bf16_supported()`` returns True on a T4, but T4 tensor cores
  are fp16-only; bf16 is supported by *emulation*. The question that matters is
  throughput, not the boolean.
- ``torch.backends.cuda.flash_sdp_enabled()`` reports whether the backend is
  *preferred*, not whether it can run here. FlashAttention-2 needs Ampere+, so
  the kernel should refuse at dispatch and fall back.
- The fast-weight states in P2 are small enough that kernel-launch overhead is
  expected to dominate FLOPs. That is the premise behind batched forking and
  ``torch.compile(mode="reduce-overhead")``, and it should be measured before
  being relied on.

Run:  colab run --gpu T4 benchmarks/t4_probe.py
"""

from __future__ import annotations

import time

import torch
import torch.nn.functional as F


def sync() -> None:
    torch.cuda.synchronize()


def timed(fn, iters: int = 50, warmup: int = 10) -> float:
    """Median seconds per call, warmed up and synchronized."""
    for _ in range(warmup):
        fn()
    sync()
    samples = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        sync()
        samples.append(time.perf_counter() - t0)
    samples.sort()
    return samples[len(samples) // 2]


def section(title: str) -> None:
    print(f"\n{'=' * 66}\n{title}\n{'=' * 66}")


def main() -> None:
    dev = torch.device("cuda")
    props = torch.cuda.get_device_properties(0)
    section("device")
    print(f"{props.name}  sm_{props.major}{props.minor}  "
          f"{props.total_memory / 2**30:.1f} GiB  torch {torch.__version__}")
    print("is_bf16_supported()                :", torch.cuda.is_bf16_supported())
    try:
        print("is_bf16_supported(emulation=False) :",
              torch.cuda.is_bf16_supported(including_emulation=False))
    except TypeError:
        print("is_bf16_supported(emulation=False) : <arg unavailable in this torch>")

    # -- 1. does bf16 actually reach tensor cores? -------------------------
    section("1. matmul throughput by dtype  (4096^3, TFLOP/s)")
    n = 4096
    flops = 2 * n**3
    results = {}
    for name, dtype in [("fp32", torch.float32), ("fp16", torch.float16),
                        ("bf16", torch.bfloat16)]:
        try:
            a = torch.randn(n, n, device=dev, dtype=dtype)
            b = torch.randn(n, n, device=dev, dtype=dtype)
            sec = timed(lambda a=a, b=b: torch.mm(a, b), iters=20)
            results[name] = flops / sec / 1e12
            print(f"  {name}: {results[name]:7.2f} TFLOP/s   ({sec * 1e3:.2f} ms)")
            del a, b
            torch.cuda.empty_cache()
        except Exception as exc:
            print(f"  {name}: FAILED -- {type(exc).__name__}: {exc}")
    if "fp16" in results and "bf16" in results:
        ratio = results["fp16"] / max(results["bf16"], 1e-9)
        print(f"\n  fp16/bf16 speed ratio: {ratio:.2f}x")
        print("  -> fp16 hits tensor cores; a large ratio means bf16 does not."
              if ratio > 1.5 else
              "  -> comparable; bf16 is not obviously penalised here.")

    # -- 2. which SDPA backend actually executes? --------------------------
    section("2. scaled_dot_product_attention backends")
    q, k, v = (torch.randn(2, 8, 1024, 64, device=dev, dtype=torch.float16) for _ in range(3))
    from torch.nn.attention import SDPBackend, sdpa_kernel
    for label, backend in [("flash (FA2)", SDPBackend.FLASH_ATTENTION),
                           ("mem_efficient", SDPBackend.EFFICIENT_ATTENTION),
                           ("math", SDPBackend.MATH)]:
        try:
            with sdpa_kernel(backend):
                sec = timed(lambda: F.scaled_dot_product_attention(q, k, v), iters=20)
            print(f"  {label:<14}: OK    {sec * 1e3:7.3f} ms")
        except Exception as exc:
            print(f"  {label:<14}: unavailable -- {type(exc).__name__}")

    # -- 3. is the small-kernel regime launch-bound? -----------------------
    section("3. fast-weight regime: launch overhead vs FLOPs")
    print("  A SwiGLU-ish update at P2 scale. If time barely moves with width,")
    print("  we are launch-bound, and batching forks is worth more than its FLOPs.")
    for width in (64, 128, 256, 512, 1024):
        w1 = torch.randn(width, 4 * width, device=dev, dtype=torch.float16)
        w2 = torch.randn(4 * width, width, device=dev, dtype=torch.float16)
        x = torch.randn(32, width, device=dev, dtype=torch.float16)
        sec = timed(lambda x=x, w1=w1, w2=w2: F.silu(x @ w1) @ w2, iters=200, warmup=50)
        gflops = 2 * (32 * width * 4 * width + 32 * 4 * width * width) / sec / 1e9
        print(f"  width {width:>4}: {sec * 1e6:8.1f} us   {gflops:8.1f} GFLOP/s")

    # -- 4. does batching forks amortize the launch cost? ------------------
    section("4. batched forking (the P2 oracle premise)")
    width = 256
    x = torch.randn(32, width, device=dev, dtype=torch.float16)
    for n_forks in (1, 8, 32, 128):
        w1 = torch.randn(n_forks, width, 4 * width, device=dev, dtype=torch.float16)
        w2 = torch.randn(n_forks, 4 * width, width, device=dev, dtype=torch.float16)
        xb = x.unsqueeze(0).expand(n_forks, -1, -1)
        sec_b = timed(
            lambda xb=xb, w1=w1, w2=w2: torch.bmm(F.silu(torch.bmm(xb, w1)), w2),
            iters=100,
            warmup=20,
        )
        sec_s = timed(
            lambda w1=w1, w2=w2, n=n_forks: [F.silu(x @ w1[i]) @ w2[i] for i in range(n)],
            iters=20,
            warmup=5,
        )
        print(f"  {n_forks:>3} forks: batched {sec_b * 1e6:8.1f} us | "
              f"sequential {sec_s * 1e6:9.1f} us | speedup {sec_s / sec_b:6.1f}x")

    # -- 5. does torch.compile pay off here? -------------------------------
    section("5. torch.compile(mode='reduce-overhead') on the update step")
    width = 256
    w1 = torch.randn(width, 4 * width, device=dev, dtype=torch.float16)
    w2 = torch.randn(4 * width, width, device=dev, dtype=torch.float16)
    x = torch.randn(32, width, device=dev, dtype=torch.float16)

    def update(x, w1, w2):
        return F.silu(x @ w1) @ w2

    eager = timed(lambda: update(x, w1, w2), iters=200, warmup=50)
    try:
        t0 = time.perf_counter()
        compiled = torch.compile(update, mode="reduce-overhead")
        compiled(x, w1, w2)
        sync()
        compile_s = time.perf_counter() - t0
        comp = timed(lambda: compiled(x, w1, w2), iters=200, warmup=50)
        print(f"  eager    : {eager * 1e6:8.1f} us")
        print(f"  compiled : {comp * 1e6:8.1f} us   ({eager / comp:.2f}x)")
        print(f"  one-off compile cost: {compile_s:.1f} s")
        print("  -> worth the flag" if eager / comp > 1.15 else
              "  -> not worth it at this size; keep eager as the default")
    except Exception as exc:
        print(f"  compile FAILED -- {type(exc).__name__}: {exc}")

    section("done")


if __name__ == "__main__":
    main()
