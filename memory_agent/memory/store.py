from __future__ import annotations

import hashlib
import math
import os
import re
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np


@dataclass
class MemoryRecord:
    """A derived memory unit, separated from the raw dialogue log."""

    id: str
    subject: str
    predicate: str
    object: str
    text: str
    evidence: list[str]
    session_id: int | str
    date_time: str
    turn_index: int
    importance: float = 0.5
    created_at: int = 0
    updated_at: int = 0
    access_count: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class HashingEmbedder:
    """Small deterministic fallback embedder used when sentence-transformers is unavailable."""

    def __init__(self, dim: int = 384):
        self.dim = dim

    def encode(self, texts: list[str] | str, normalize_embeddings: bool = True):
        single = isinstance(texts, str)
        items = [texts] if single else list(texts)
        vecs = np.zeros((len(items), self.dim), dtype=np.float32)
        for row, text in enumerate(items):
            tokens = re.findall(r"[A-Za-z0-9_']+|[\u4e00-\u9fff]", text.lower())
            for token in tokens:
                digest = hashlib.md5(token.encode("utf-8")).digest()
                idx = int.from_bytes(digest[:4], "little") % self.dim
                sign = 1.0 if digest[4] % 2 == 0 else -1.0
                vecs[row, idx] += sign
        if normalize_embeddings:
            norms = np.linalg.norm(vecs, axis=1, keepdims=True)
            vecs = vecs / np.maximum(norms, 1e-12)
        return vecs[0] if single else vecs


def load_embedder():
    model_name = os.getenv("EMBED_MODEL", "BAAI/bge-small-zh-v1.5")
    local_only = os.getenv("EMBED_LOCAL_ONLY", "1").lower() not in {"0", "false", "no"}
    try:
        from sentence_transformers import SentenceTransformer

        try:
            return SentenceTransformer(model_name, local_files_only=local_only)
        except TypeError:
            return SentenceTransformer(model_name)
    except Exception:
        return HashingEmbedder()


class MemoryStore:
    """In-memory vector index for derived memory records."""

    def __init__(self, embedder=None):
        self.embedder = embedder or load_embedder()
        self.records: list[MemoryRecord] = []
        self.embeddings: np.ndarray | None = None

    def __len__(self) -> int:
        return len(self.records)

    def add(self, record: MemoryRecord) -> None:
        self.records.append(record)
        self._rebuild_embeddings()

    def replace(self, index: int, record: MemoryRecord) -> None:
        self.records[index] = record
        self._rebuild_embeddings()

    def extend(self, records: list[MemoryRecord]) -> None:
        if not records:
            return
        self.records.extend(records)
        self._rebuild_embeddings()

    def _rebuild_embeddings(self) -> None:
        if not self.records:
            self.embeddings = None
            return
        texts = [record.text for record in self.records]
        vecs = self.embedder.encode(texts, normalize_embeddings=True)
        self.embeddings = np.array(vecs, dtype=np.float32)

    def search(self, query: str, top_k: int = 8) -> list[tuple[MemoryRecord, float]]:
        if self.embeddings is None or not self.records:
            return []
        qvec = self.embedder.encode([query], normalize_embeddings=True)[0]
        sims = self.embeddings @ qvec.astype(np.float32)
        idx = np.argsort(-sims)[: min(top_k, len(self.records))]
        return [(self.records[int(i)], float(sims[int(i)])) for i in idx]

    def lexical_overlap(self, query: str, record: MemoryRecord) -> float:
        q_terms = self._terms(query)
        r_terms = self._terms(record.text)
        if not q_terms or not r_terms:
            return 0.0
        return len(q_terms & r_terms) / math.sqrt(len(q_terms) * len(r_terms))

    def _terms(self, text: str) -> set[str]:
        raw_terms = re.findall(r"[A-Za-z0-9_']+", text.lower())
        stop = {"the", "a", "an", "and", "or", "to", "of", "in", "on", "is", "was", "are", "did"}
        terms = set()
        for term in raw_terms:
            if term in stop or len(term) <= 2:
                continue
            if term.endswith("ing") and len(term) > 5:
                term = term[:-3]
            elif term.endswith("ed") and len(term) > 4:
                base = term[:-2]
                terms.add(base + "e")
                term = base
            elif term.endswith("s") and len(term) > 4:
                term = term[:-1]
            terms.add(term)
        return terms
