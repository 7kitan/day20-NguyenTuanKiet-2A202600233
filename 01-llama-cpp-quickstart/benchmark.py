#!/usr/bin/env python3
"""01 — Quickstart latency benchmark.

Loads the primary GGUF model, measures TTFT/TPOT/P50/P95/P99, then loads
the comparison quantization and reruns the same prompts. Writes a results
markdown to benchmarks/01-quickstart-results.md.
"""

from __future__ import annotations

import json
import os
import statistics
import sys
import time
from pathlib import Path

try:
    from llama_cpp import Llama
except ImportError:
    print(
        "ERROR: llama_cpp not installed. Run the platform setup script.",
        file=sys.stderr,
    )
    sys.exit(1)

from dotenv import load_dotenv

load_dotenv()

PROMPTS = [
    "Explain TTFT and TPOT in one sentence each.",
    "Write a haiku about KV cache fragmentation.",
    "What is PagedAttention and why was it a 24x improvement?",
    "List three reasons goodput@SLO matters more than peak throughput.",
    "Compare Q4_K_M vs Q2_K quantization in three bullets.",
    "Why does FlashAttention give O(N) memory instead of O(N^2)?",
    "What's the difference between continuous batching and static batching?",
    "Sketch the steps of speculative decoding.",
    "Explain MLA (Multi-head Latent Attention) in two sentences.",
    "When should you use disaggregated prefill/decode serving?",
]


def env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    return int(raw) if raw and raw.isdigit() else default


def env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def load_active() -> dict:
    p = Path("models/active.json")
    if not p.exists():
        print(
            "ERROR: models/active.json missing. Run 00-setup/download-model.py.",
            file=sys.stderr,
        )
        sys.exit(1)
    return json.loads(p.read_text())


def load_hardware() -> dict:
    p = Path("hardware.json")
    return json.loads(p.read_text()) if p.exists() else {}


def make_llm(
    model_path: str, n_threads: int, n_ctx: int, n_batch: int, n_gpu_layers: int
) -> Llama:
    return Llama(
        model_path=model_path,
        n_ctx=n_ctx,
        n_threads=n_threads,
        n_batch=n_batch,
        n_gpu_layers=n_gpu_layers,
        verbose=False,
    )


def measure_one(
    llm: Llama, prompt: str, max_tokens: int, temperature: float
) -> tuple[float, float, int]:
    """Returns (ttft_ms, tpot_ms, n_tokens)."""
    start = time.perf_counter()
    first_token_at = None
    n_tokens = 0
    for chunk in llm.create_completion(
        prompt=prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        stream=True,
    ):
        text = chunk["choices"][0].get("text", "")
        if text and first_token_at is None:
            first_token_at = time.perf_counter()
        if text:
            n_tokens += 1
    end = time.perf_counter()

    if first_token_at is None or n_tokens == 0:
        return 0.0, 0.0, 0
    ttft_ms = (first_token_at - start) * 1000.0
    decode_ms = (end - first_token_at) * 1000.0
    tpot_ms = decode_ms / max(n_tokens - 1, 1)
    return ttft_ms, tpot_ms, n_tokens


def quantile(data: list[float], q: float) -> float:
    if not data:
        return 0.0
    return statistics.quantiles(sorted(data), n=100, method="inclusive")[
        max(0, int(q) - 1)
    ]


def benchmark_model(label: str, path: str, hw: dict) -> dict:
    n_threads = env_int("LAB_N_THREADS", hw.get("cpu", {}).get("cores_physical") or 4)
    n_ctx = env_int("LAB_N_CTX", 2048)
    n_batch = env_int("LAB_N_BATCH", 512)
    n_gpu_layers = env_int("LAB_N_GPU_LAYERS", 99 if any_gpu(hw) else 0)
    temp = env_float("LAB_TEMPERATURE", 0.7)
    max_tok = env_int("LAB_MAX_TOKENS", 64)

    print(f"\n── Loading {label}: {Path(path).name}")
    print(
        f"   n_threads={n_threads}  n_ctx={n_ctx}  n_batch={n_batch}  n_gpu_layers={n_gpu_layers}"
    )

    t0 = time.perf_counter()
    llm = make_llm(path, n_threads, n_ctx, n_batch, n_gpu_layers)
    load_ms = (time.perf_counter() - t0) * 1000.0
    print(f"   model loaded in {load_ms:.0f} ms")

    # Warm-up (kills cold-start skew)
    _ = measure_one(llm, "Hello.", max_tokens=8, temperature=0.0)

    ttfts: list[float] = []
    tpots: list[float] = []
    e2es: list[float] = []
    for i, prompt in enumerate(PROMPTS):
        t_start = time.perf_counter()
        ttft, tpot, ntok = measure_one(llm, prompt, max_tok, temp)
        e2e = (time.perf_counter() - t_start) * 1000.0
        if ntok > 0:
            ttfts.append(ttft)
            tpots.append(tpot)
            e2es.append(e2e)
            print(
                f"   [{i + 1:2d}/{len(PROMPTS)}] ttft={ttft:6.1f}ms  tpot={tpot:5.1f}ms  e2e={e2e:7.1f}ms  tok={ntok}"
            )

    summary = {
        "label": label,
        "model_path": path,
        "n_threads": n_threads,
        "n_ctx": n_ctx,
        "n_batch": n_batch,
        "n_gpu_layers": n_gpu_layers,
        "load_ms": round(load_ms, 1),
        "ttft_ms_p50": round(quantile(ttfts, 50), 1),
        "ttft_ms_p95": round(quantile(ttfts, 95), 1),
        "tpot_ms_p50": round(quantile(tpots, 50), 2),
        "tpot_ms_p95": round(quantile(tpots, 95), 2),
        "e2e_ms_p50": round(quantile(e2es, 50), 1),
        "e2e_ms_p95": round(quantile(e2es, 95), 1),
        "e2e_ms_p99": round(quantile(e2es, 99), 1),
        "decode_rate_tok_s": round(1000.0 / max(statistics.median(tpots), 0.001), 1)
        if tpots
        else 0,
    }
    return summary


def any_gpu(hw: dict) -> bool:
    backends = hw.get("gpu", {}).get("backends", {})
    return any(v for k, v in backends.items() if k != "cpu_only")


def render_md(primary: dict, compare: dict) -> str:
    def row(s: dict) -> str:
        return (
            f"| {Path(s['model_path']).name} | {s['load_ms']:.0f} | "
            f"{s['ttft_ms_p50']:.0f} / {s['ttft_ms_p95']:.0f} | "
            f"{s['tpot_ms_p50']:.1f} / {s['tpot_ms_p95']:.1f} | "
            f"{s['e2e_ms_p50']:.0f} / {s['e2e_ms_p95']:.0f} / {s['e2e_ms_p99']:.0f} | "
            f"{s['decode_rate_tok_s']:.1f} |"
        )

    return f"""# 01 — Quickstart Results

Settings: `n_threads={primary["n_threads"]}`, `n_ctx={primary["n_ctx"]}`, `n_batch={primary["n_batch"]}`, `n_gpu_layers={primary["n_gpu_layers"]}`.

| Model | Load (ms) | TTFT P50/P95 (ms) | TPOT P50/P95 (ms) | E2E P50/P95/P99 (ms) | Decode rate (tok/s) |
|---|---:|---:|---:|---:|---:|
{row(primary)}
{row(compare)}

## Observations

- TTFT is the prefill cost. With short prompts this is small; with long prompts it dominates.
- TPOT is per-token decode latency. The decode rate is `1000 / TPOT_p50`.
- The bigger quantization (Q4_K_M) is usually only ~30–60% slower than Q2_K but produces noticeably better text. Q2_K is for *truly* tight RAM.
- `n_threads = physical_cores` is usually best on CPU. Hyperthreading (`logical_cores`) often hurts because the work is bandwidth-bound.

(Edit this file with your own observations before submitting.)
"""


def main() -> int:
    active = load_active()
    hw = load_hardware()

    primary = benchmark_model("primary (Q4_K_M)", active["primary_model"], hw)
    compare = benchmark_model("compare (Q2_K)", active["compare_model"], hw)

    out_dir = Path("benchmarks")
    out_dir.mkdir(exist_ok=True)
    md = render_md(primary, compare)
    (out_dir / "01-quickstart-results.md").write_text(md)
    (out_dir / "01-quickstart-results.json").write_text(
        json.dumps({"primary": primary, "compare": compare}, indent=2)
    )

    print("\n" + md)
    print(f"\n==> Wrote benchmarks/01-quickstart-results.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
