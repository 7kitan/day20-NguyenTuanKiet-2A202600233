# Track 03 — Milestone Integration Notes

## Integration Summary

Connected day20 llama-server to a RAG pipeline using patterns from day18 (lakehouse) and day19 (vector search).

### Components Connected

| Day | Component | Implementation |
|-----|-----------|----------------|
| **N16** (Cloud/IaC) | Infrastructure | Stub: localhost only, no K8s/Docker |
| **N17** (Data pipeline) | Data ingestion | Stub: static JSONL corpus |
| **N18** (Lakehouse) | Data storage | Stub: local file (corpus_sample.jsonl), no Delta Lake |
| **N19** (Vector + Feature) | Retrieval | Adapted from day19/app/search.py - hybrid search (BM25 + semantic + RRF) |
| **N20** (Serving) | LLM inference | llama-server on localhost:8080 |

### What Was Adapted from Prior Days

**From day19/app/search.py:**
- `Searcher` class with BM25 + Qdrant + fastembed
- Hybrid search using Reciprocal Rank Fusion (RRF with k=60)
- In-memory Qdrant mode (no Docker required)
- Copied to `03-milestone-integration/search.py`

**From day18/scripts/lakehouse.py:**
- Pattern: medallion architecture (bronze/silver/gold layers)
- Not directly used, but informed data organization approach
- In production: would query Delta Lake tables for context documents

### Fallback Strategy

Pipeline has graceful degradation:
1. **Preferred:** Load `search.py` with fastembed + qdrant-client + rank-bm25
2. **Fallback:** Use toy keyword matching if dependencies unavailable
3. Both paths measure timing separately (embed, retrieve, llm)

### Latency Breakdown (measured with time.perf_counter)

From 3 test queries:

| Stage | Time (ms) | % of Total |
|-------|-----------|------------|
| Embed | ~0 | <1% (toy data) |
| Retrieve | 0.0-0.1 | <1% (toy data) |
| LLM (llama-server) | 1024-2368 | >99% |
| **Total E2E** | 1024-2368 | 100% |

**Observation:** LLM generation dominates latency. Retrieval is negligible with toy data (would be 10-50ms with real embeddings). This matches expectations: local LLM inference on CPU/Metal is the bottleneck in RAG pipelines.

### Production Gaps

What's stubbed vs what production needs:

| Component | Current State | Production Needs |
|-----------|---------------|------------------|
| Corpus | 10 static docs | Live Delta Lake table from N18 pipeline |
| Embeddings | Fallback to keywords | Real fastembed or OpenAI embeddings |
| Vector index | In-memory Qdrant | Persistent Qdrant server or Pinecone |
| Feature store | Not used | Feast online store for user/doc features |
| Caching | System prompt only | Redis for retrieved contexts |
| Observability | Basic timing | Prometheus metrics + traces |

### Key Insight

**Bottleneck is LLM, not retrieval.** Even with toy data, the pattern is clear: 1-2 seconds for generation vs <1ms for retrieval. Optimizing llama-server (quantization, batch size, GPU offload) has 100× more impact than optimizing vector search for this workload.

This validates the day20 deck's focus on serving optimization (TTFT, TPOT, KV cache) over retrieval optimization.

---

**Files:**
- `pipeline.py` - main RAG flow
- `search.py` - adapted from day19/app/search.py
- `data/corpus_sample.jsonl` - 10 serving-related documents
