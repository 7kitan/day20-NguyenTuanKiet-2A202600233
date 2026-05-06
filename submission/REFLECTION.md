# Reflection — Lab 20 (Personal Report)

> **Đây là báo cáo cá nhân.** Mỗi học viên chạy lab trên laptop của mình, với spec của mình. Số liệu của bạn không so sánh được với bạn cùng lớp — chỉ so sánh **before vs after trên chính máy bạn**. Grade rubric tính theo độ rõ ràng của setup + tuning của bạn, không phải tốc độ tuyệt đối.

---

**Họ Tên:** Nguyễn Tuấn Kiệt
**Cohort:** A20-K1
**Ngày submit:** 2026-05-06

---

## 1. Hardware spec (từ `00-setup/detect-hardware.py`)

> Paste output của `python 00-setup/detect-hardware.py` vào đây, hoặc điền thủ công:

- **OS:** macOS
- **CPU:** Apple M1 Pro
- **Cores:** 10 physical / 10 logical
- **CPU extensions:** NEON (ARM64)
- **RAM:** 16 GB
- **Accelerator:** Apple Metal
- **llama.cpp backend đã chọn:** Metal
- **Recommended model tier:** Llama-3.2-3B-Instruct (Q4_K_M)

**Setup story** (≤ 80 chữ): những gì cần thay đổi để lab chạy được trên máy bạn (vd: dùng WSL2, install CUDA Toolkit, fall back sang Vulkan vì ROCm phiên bản kén, tắt antivirus để pip install nhanh hơn, v.v.):

Model Llama-3.2-3B-Instruct Q4_K_M được recommend nhưng không có sẵn, phải chuyển sang Q3_K_L. Metal backend hoạt động ngay với llama.cpp trên M1 Pro 16GB RAM, không cần config thêm.

---

## 2. Track 01 — Quickstart numbers (từ `benchmarks/01-quickstart-results.md`)

> Paste bảng từ `benchmarks/01-quickstart-results.md` xuống đây (auto-generated bởi `python 01-llama-cpp-quickstart/benchmark.py`).

| Model | Load (ms) | TTFT P50/P95 (ms) | TPOT P50/P95 (ms) | E2E P50/P95/P99 (ms) | Decode rate (tok/s) |
|---|--:|--:|--:|--:|--:|
| Q4_K_M | 5126 | 308 / 503 | 79.4 / 82.6 | 5314 / 5531 / 5546 | 12.6 |
| Q3_K_L | 2992 | 326 / 421 | 140.9 / 143.9 | 9216 / 9310 / 9332 | 7.1 |

**Một quan sát** (≤ 50 chữ): Q4_K_M vs Q3_K_L trên máy bạn — số liệu nói gì? Quality đáng đánh đổi không?

Q4_K_M nhanh gấp đôi (12.6 vs 7.1 tok/s) và E2E latency thấp hơn nhiều (5.3s vs 9.2s). Q3_K_L chỉ thắng ở load time. Nếu có sẵn Q4_K_M thì đáng dùng hơn vì performance gap quá lớn.

---

## 3. Track 02 — llama-server load test

> Chạy 2 lần locust ở concurrency 10 và 50, paste tóm tắt bên dưới.

| Concurrency | Total RPS | TTFB P50 (ms) | E2E P95 (ms) | E2E P99 (ms) | Failures |
|--:|--:|--:|--:|--:|--:|
| 10 | 0.73 | 11000 | 19000 | 20000 | 0 |
| 50 | 0.87 | 15000 | 29000 | 30000 | 0 |

**KV-cache observation** (từ `record-metrics.py`): Server có 4 parallel slots với utilization ~98.5% (`n_busy_slots_per_decode=3.94`). Requests deferred tăng từ 0 lên peak 46 khi concurrency 50 vượt quá 4 slots. Metrics `llamacpp:kv_cache_usage_ratio` và `llamacpp:kv_cache_tokens` không được expose bởi llama-server build này (version 9020 từ Homebrew), mặc dù server đã chạy với flags `--metrics` và `--cache-prompt`. KV cache đang hoạt động internally (cần thiết cho parallel slots), nhưng không thể đo được cache hit rate.

---

## 4. Track 03 — Milestone integration

- **N16 (Cloud/IaC):** stub: localhost only, no K8s/Docker
- **N17 (Data pipeline):** stub: static JSONL corpus (10 docs)
- **N18 (Lakehouse):** stub: local file (corpus_sample.jsonl), no Delta Lake
- **N19 (Vector + Feature Store):** Adapted from day19/app/search.py - hybrid search (BM25 + semantic + RRF) copied to local search.py. Falls back to keyword matching if dependencies unavailable.

**Pipeline run output** (3 example queries with retrieved-context provenance):

```
=== Why is goodput more useful than throughput? ===
Loading searcher from /Users/kitan/dev/day20/03-milestone-integration/data/corpus_sample.jsonl...
Warning: could not load Searcher: Missing dependencies: fastembed, qdrant-client, rank-bm25
Falling back to toy data (no real vector search)
  contexts: ['n20-paged', 'n20-radix', 'n20-disagg']
  timings : {'embed': 10.0, 'retrieve': 0.0, 'llm': 1876.6, 'total': 1879.1}
  answer  : I couldn't find any information in the provided documents about why goodput is more useful...

=== What problem does PagedAttention actually solve? ===
  contexts: ['n20-paged', 'n20-radix', 'n20-disagg']
  timings : {'embed': 10.0, 'retrieve': 0.0, 'llm': 1023.7, 'total': 1023.8}
  answer  : PagedAttention treats KV cache like virtual memory pages, eliminating 60-80% fragmentation...

=== When should I think about disaggregated serving? ===
  contexts: ['n20-disagg', 'n20-paged', 'n20-radix']
  timings : {'embed': 10.0, 'retrieve': 0.1, 'llm': 2367.8, 'total': 2368.0}
  answer  : Based on the provided documents, you should consider disaggregated serving when...
```

**Nơi tốn nhiều ms nhất** trong pipeline (đo bằng `time.perf_counter` trong `pipeline.py`):

- embed: ~0 ms (toy data fallback, no real embeddings)
- retrieve: 0.0-0.1 ms (toy keyword matching)
- llama-server: 1024-2368 ms (range across 3 queries)

**Reflection** (≤ 60 chữ): bottleneck nằm ở đâu? Có khớp với kỳ vọng không?

Bottleneck hoàn toàn nằm ở llama-server (1.0-2.4s), chiếm >99% total time. Retrieve <1ms với toy data. Khớp kỳ vọng: LLM generation là phần chậm nhất trong RAG pipeline. Tối ưu llama-server có impact 100× hơn tối ưu retrieval cho workload này.

---

## 5. Bonus — The single change that mattered most

> **Most important section.** Pick **một** thay đổi từ bonus track (build flag, thread sweep, quant pick, GPU offload, KV-cache quantization, speculative decoding, bất cứ challenge nào trong `BONUS-llama-cpp-optimization/CHALLENGES.md`) đã tạo ra speedup lớn nhất trên máy bạn.

**Change:** _<vd: rebuild llama.cpp với `-DGGML_NATIVE=ON -DGGML_BLAS=ON`; vd: hạ `-t` từ 12 xuống 6; vd: bật Metal trên M2>_

**Before vs after** (paste 2-3 dòng từ sweep output):

```
before: <số liệu>
after:  <số liệu>
speedup: ~<X.Y>×
```

**Tại sao nó work** (1–2 đoạn ngắn — đây là phần grader đọc kỹ nhất):

_Giải thích như đang nói với một bạn cùng lớp đang ngồi cạnh. Tránh "vibes-based" reasoning — bám vào mô hình mental của hardware (memory bandwidth? compute? cache?). Nếu kết quả khác kỳ vọng từ deck, nói rõ — đó là phần grader thưởng điểm._

---

## 6. (Optional) Điều ngạc nhiên nhất

_(1–2 câu — không bắt buộc, nhưng người grader đọc tất cả)_

Tăng concurrency từ 10 lên 50 users chỉ tăng RPS từ 0.73 lên 0.87 (+19%), nhưng P95 latency tăng từ 19s lên 29s (+53%). Server đã saturated ở 4 parallel slots - thêm users chỉ tạo queue dài hơn, không tăng throughput. Đây chính là lý do deck nhấn mạnh goodput@SLO thay vì peak throughput.

---

## 7. Self-graded checklist

- [ ] `hardware.json` đã commit
- [ ] `models/active.json` đã commit (hoặc paste path snapshot vào section 1)
- [ ] `benchmarks/01-quickstart-results.md` đã commit
- [ ] `benchmarks/02-server-results.md` (hoặc CSV từ `record-metrics.py`) đã commit
- [ ] `benchmarks/bonus-*.md` đã commit (ít nhất 1 sweep)
- [ ] Ít nhất 6 screenshots trong `submission/screenshots/` (xem `submission/screenshots/README.md`)
- [ ] `make verify` exit 0 (chạy ngay trước khi push)
- [ ] Repo trên GitHub ở chế độ **public**
- [ ] Đã paste public repo URL vào VinUni LMS

---

**Quan trọng:** repo phải **public** đến khi điểm được công bố. Nếu private, grader không xem được → 0 điểm.
