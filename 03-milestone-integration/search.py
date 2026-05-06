"""Simplified searcher for day20 integration (adapted from day19/app/search.py).

Provides keyword (BM25) + semantic (vector) + hybrid (RRF) search.
Designed to work with Qdrant in-memory mode - no Docker required.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

try:
    from fastembed import TextEmbedding
    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, PointStruct, VectorParams
    from rank_bm25 import BM25Okapi
    DEPS_AVAILABLE = True
except ImportError:
    DEPS_AVAILABLE = False

Mode = Literal["keyword", "semantic", "hybrid"]
EMBED_MODEL = "BAAI/bge-small-en-v1.5"   # 384-dim, CPU-friendly
EMBED_DIM = 384
COLLECTION = "day20_integration"


@dataclass
class SearchHit:
    doc_id: str
    title: str
    text: str
    score: float

    def dict(self) -> dict:
        return {"doc_id": self.doc_id, "title": self.title, "text": self.text, "score": self.score}


class Searcher:
    """Holds BM25 index, Qdrant client, and document metadata."""

    def __init__(self) -> None:
        self.docs: list[dict] = []
        self.doc_ids: list[str] = []
        self.bm25: BM25Okapi | None = None
        self.client: QdrantClient | None = None
        self.embedder: TextEmbedding | None = None

    @property
    def size(self) -> int:
        return len(self.docs)

    @classmethod
    def from_corpus(cls, corpus_path: Path) -> "Searcher":
        if not DEPS_AVAILABLE:
            raise ImportError("Missing dependencies: fastembed, qdrant-client, rank-bm25")
        
        s = cls()
        s._load_docs(corpus_path)
        s._build_bm25()
        s._build_vector_index()
        return s

    # ── ingestion ───────────────────────────────────────────────────────
    def _load_docs(self, corpus_path: Path) -> None:
        with corpus_path.open(encoding="utf-8") as f:
            for line in f:
                d = json.loads(line)
                self.docs.append(d)
                self.doc_ids.append(d["doc_id"])

    def _build_bm25(self) -> None:
        tokenized = [self._tokenize(d["title"] + " " + d["text"]) for d in self.docs]
        self.bm25 = BM25Okapi(tokenized)

    def _build_vector_index(self) -> None:
        self.embedder = TextEmbedding(model_name=EMBED_MODEL)
        self.client = QdrantClient(":memory:")

        self.client.create_collection(
            collection_name=COLLECTION,
            vectors_config=VectorParams(size=EMBED_DIM, distance=Distance.COSINE),
        )

        # Embed in batches
        BATCH = 64
        points: list[PointStruct] = []
        for start in range(0, len(self.docs), BATCH):
            batch = self.docs[start:start + BATCH]
            texts = [d["title"] + " " + d["text"] for d in batch]
            vectors = list(self.embedder.embed(texts))
            for i, (d, v) in enumerate(zip(batch, vectors)):
                points.append(PointStruct(
                    id=start + i,
                    vector=v.tolist(),
                    payload={"doc_id": d["doc_id"], "title": d["title"], "text": d["text"]},
                ))
        self.client.upsert(collection_name=COLLECTION, points=points)

    # ── retrieval ───────────────────────────────────────────────────────
    @staticmethod
    def _tokenize(text: str) -> list[str]:
        return text.lower().split()

    def search(
        self,
        query: str,
        mode: Mode = "hybrid",
        top_k: int = 10,
        rrf_k: int = 60,
    ) -> list[SearchHit]:
        if mode == "keyword":
            return self._search_keyword(query, top_k)
        if mode == "semantic":
            return self._search_semantic(query, top_k)
        if mode == "hybrid":
            return self._search_hybrid(query, top_k, rrf_k)
        raise ValueError(f"unknown mode {mode!r}")

    def _search_keyword(self, query: str, top_k: int) -> list[SearchHit]:
        assert self.bm25 is not None
        scores = self.bm25.get_scores(self._tokenize(query))
        ranked = sorted(range(len(scores)), key=lambda i: -scores[i])[:top_k]
        return [
            SearchHit(
                doc_id=self.docs[i]["doc_id"],
                title=self.docs[i]["title"],
                text=self.docs[i]["text"],
                score=float(scores[i]),
            )
            for i in ranked
        ]

    def _search_semantic(self, query: str, top_k: int) -> list[SearchHit]:
        assert self.client is not None and self.embedder is not None
        q_vec = next(self.embedder.embed([query])).tolist()
        result = self.client.query_points(
            collection_name=COLLECTION,
            query=q_vec,
            limit=top_k,
        )
        return [
            SearchHit(
                doc_id=p.payload["doc_id"],
                title=p.payload["title"],
                text=p.payload["text"],
                score=float(p.score),
            )
            for p in result.points
        ]

    def _search_hybrid(self, query: str, top_k: int, rrf_k: int) -> list[SearchHit]:
        # Pull deeper top-K from each retriever for RRF
        depth = max(top_k * 5, 50)
        kw_hits = self._search_keyword(query, depth)
        sem_hits = self._search_semantic(query, depth)

        # Reciprocal Rank Fusion
        rrf_scores: dict[str, float] = {}
        meta: dict[str, SearchHit] = {}
        for hits in (kw_hits, sem_hits):
            for rank, h in enumerate(hits, start=1):
                rrf_scores[h.doc_id] = rrf_scores.get(h.doc_id, 0.0) + 1.0 / (rrf_k + rank)
                meta.setdefault(h.doc_id, h)

        ordered = sorted(rrf_scores.items(), key=lambda kv: -kv[1])[:top_k]
        return [
            SearchHit(
                doc_id=doc_id,
                title=meta[doc_id].title,
                text=meta[doc_id].text,
                score=score,
            )
            for doc_id, score in ordered
        ]
