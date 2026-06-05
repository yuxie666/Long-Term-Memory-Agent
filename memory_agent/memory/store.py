from __future__ import annotations

import hashlib
import json
import math
import os
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


_EMBEDDER_CACHE = None


@dataclass
class MemoryRecord:
    """A memory unit stored in the local vector index."""

    id: str
    subject: str
    predicate: str
    object: str
    text: str
    evidence: list[str]
    session_id: int | str
    date_time: str
    turn_index: int | None
    memory_level: str = "high"
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
    global _EMBEDDER_CACHE
    if _EMBEDDER_CACHE is not None:
        return _EMBEDDER_CACHE

    model_name = os.getenv("EMBED_MODEL", "BAAI/bge-small-zh-v1.5")
    local_only = os.getenv("EMBED_LOCAL_ONLY", "1").lower() not in {"0", "false", "no"}
    try:
        from sentence_transformers import SentenceTransformer

        try:
            _EMBEDDER_CACHE = SentenceTransformer(model_name, local_files_only=local_only)
        except TypeError:
            _EMBEDDER_CACHE = SentenceTransformer(model_name)
    except Exception:
        _EMBEDDER_CACHE = HashingEmbedder()
    return _EMBEDDER_CACHE


class MemoryStore:
    """Local persistent vector index for low-level and high-level memories."""

    def __init__(self, embedder=None, persist_dir: str | Path | None = None, auto_load: bool = False):
        self.embedder = embedder or load_embedder()
        self.records: list[MemoryRecord] = []
        self.embeddings: np.ndarray | None = None
        default_dir = Path(__file__).resolve().parents[1] / "experiments" / "memory_index"
        self.persist_dir = Path(persist_dir or os.getenv("MEMORY_INDEX_DIR", str(default_dir)))
        if auto_load:
            self.load()

    def __len__(self) -> int:
        return len(self.records)

    def count(self, memory_level: str | None = None) -> int:
        if memory_level is None:
            return len(self.records)
        return sum(1 for record in self.records if record.memory_level == memory_level)

    def add(self, record: MemoryRecord) -> None:
        self.records.append(record)
        vec = self._encode_records([record])
        self.embeddings = vec if self.embeddings is None else np.vstack([self.embeddings, vec])

    def replace(self, index: int, record: MemoryRecord) -> None:
        self.records[index] = record
        vec = self._encode_records([record])
        if self.embeddings is None:
            self.embeddings = self._encode_records(self.records)
        else:
            self.embeddings[index] = vec[0]

    def extend(self, records: list[MemoryRecord]) -> None:
        if not records:
            return
        self.records.extend(records)
        vecs = self._encode_records(records)
        self.embeddings = vecs if self.embeddings is None else np.vstack([self.embeddings, vecs])

    def _encode_records(self, records: list[MemoryRecord]) -> np.ndarray:
        texts = [record.text for record in records]
        vecs = self.embedder.encode(texts, normalize_embeddings=True)
        return np.atleast_2d(np.array(vecs, dtype=np.float32))

    def _rebuild_embeddings(self) -> None:
        if not self.records:
            self.embeddings = None
            return
        texts = [record.text for record in self.records]
        vecs = self.embedder.encode(texts, normalize_embeddings=True)
        self.embeddings = np.array(vecs, dtype=np.float32)

    def search(
        self,
        query: str,
        top_k: int = 8,
        memory_level: str | None = None,
    ) -> list[tuple[MemoryRecord, float]]:
        if self.embeddings is None or not self.records:
            return []
        qvec = self.embedder.encode([query], normalize_embeddings=True)[0]
        sims = self.embeddings @ qvec.astype(np.float32)
        allowed = [
            i for i, record in enumerate(self.records)
            if memory_level is None or record.memory_level == memory_level
        ]
        if not allowed:
            return []
        ranked = sorted(allowed, key=lambda i: float(sims[i]), reverse=True)
        idx = ranked[: min(top_k, len(ranked))]
        return [(self.records[int(i)], float(sims[int(i)])) for i in idx]

    def records_by_level(self, memory_level: str | None = None) -> list[MemoryRecord]:
        if memory_level is None:
            return list(self.records)
        return [record for record in self.records if record.memory_level == memory_level]

    def save(self) -> None:
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        records_path = self.persist_dir / "records.json"
        embeddings_path = self.persist_dir / "embeddings.npy"
        with records_path.open("w", encoding="utf-8") as f:
            json.dump([record.to_dict() for record in self.records], f, ensure_ascii=False, indent=2)
        if self.embeddings is None:
            np.save(embeddings_path, np.zeros((0, 0), dtype=np.float32))
        else:
            np.save(embeddings_path, self.embeddings)

    def load(self) -> None:
        records_path = self.persist_dir / "records.json"
        embeddings_path = self.persist_dir / "embeddings.npy"
        if not records_path.exists():
            return
        with records_path.open(encoding="utf-8") as f:
            raw_records = json.load(f)
        self.records = [MemoryRecord(**record) for record in raw_records]
        if embeddings_path.exists():
            loaded = np.load(embeddings_path)
            self.embeddings = None if loaded.size == 0 else np.array(loaded, dtype=np.float32)
        else:
            self._rebuild_embeddings()
        if self.embeddings is not None and len(self.embeddings) != len(self.records):
            self._rebuild_embeddings()

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
