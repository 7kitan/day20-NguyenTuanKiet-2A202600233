# 01 — Quickstart Results

Settings: `n_threads=1`, `n_ctx=2048`, `n_batch=128`, `n_gpu_layers=0`.

| Model | Load (ms) | TTFT P50/P95 (ms) | TPOT P50/P95 (ms) | E2E P50/P95/P99 (ms) | Decode rate (tok/s) |
|---|---:|---:|---:|---:|---:|
| Llama-3.2-3B-Instruct-Q4_K_M.gguf | 5126 | 308 / 503 | 79.4 / 82.6 | 5314 / 5531 / 5546 | 12.6 |
| Llama-3.2-3B-Instruct-Q3_K_L.gguf | 2992 | 326 / 421 | 140.9 / 143.9 | 9216 / 9310 / 9332 | 7.1 |

## Observations

- TTFT is the prefill cost. With short prompts this is small; with long prompts it dominates.
- TPOT is per-token decode latency. The decode rate is `1000 / TPOT_p50`.
- The bigger quantization (Q4_K_M) is usually only ~30–60% slower than Q2_K but produces noticeably better text. Q2_K is for *truly* tight RAM.
- `n_threads = physical_cores` is usually best on CPU. Hyperthreading (`logical_cores`) often hurts because the work is bandwidth-bound.

(Edit this file with your own observations before submitting.)
