from __future__ import annotations

from enum import Enum
import math
import os

from .store import MemoryRecord, MemoryStore


class RetrievalStrategy(str, Enum):
    VECTOR_ONLY = "vector_only"
    THREE_FACTOR_ONLY = "three_factor_only"
    HYBRID = "hybrid"


class MemoryRetriever:
    """Retrieves low-level and high-level memories with configurable scoring."""

    def __init__(
        self,
        store: MemoryStore,
        top_k: int = 8,
        recency_weight: float = 0.15,
        importance_weight: float = 0.2,
        relevance_weight: float = 0.65,
        vector_weight: float = 0.7,
        factor_weight: float = 0.3,
        half_life: float = 80.0,
        strategy: str | RetrievalStrategy | None = None,
        vector_threshold: float = 0.18,
        factor_threshold: float = 0.25,
        hybrid_threshold: float = 0.22,
    ):
        self.store = store
        self.top_k = top_k
        self.recency_weight = recency_weight
        self.importance_weight = importance_weight
        self.relevance_weight = relevance_weight
        self.vector_weight = vector_weight
        self.factor_weight = factor_weight
        self.half_life = half_life
        configured = strategy or os.getenv("MEMORY_RETRIEVAL_STRATEGY", RetrievalStrategy.HYBRID.value)
        self.strategy = RetrievalStrategy(configured)
        self.vector_threshold = vector_threshold
        self.factor_threshold = factor_threshold
        self.hybrid_threshold = hybrid_threshold

    def retrieve(self, query: str) -> list[tuple[MemoryRecord, dict]]:
        candidates = self._collect_candidates(query)
        if not candidates:
            return []

        now = max((record.updated_at for record in self.store.records), default=0) + 1
        ranked = []
        for record, vector_similarity in candidates:
            recency = math.exp(-max(now - record.updated_at, 0) / self.half_life)
            lexical = self.store.lexical_overlap(query, record)
            vector_score = max(float(vector_similarity), 0.0)
            factor_score = (
                self.relevance_weight * lexical
                + self.importance_weight * record.importance
                + self.recency_weight * recency
            )
            if self.strategy == RetrievalStrategy.VECTOR_ONLY:
                score = vector_score
                threshold = self.vector_threshold
            elif self.strategy == RetrievalStrategy.THREE_FACTOR_ONLY:
                score = factor_score
                threshold = self.factor_threshold
            else:
                score = self.vector_weight * vector_score + self.factor_weight * factor_score
                threshold = self.hybrid_threshold
            if score < threshold:
                continue
            details = {
                "score": round(float(score), 4),
                "strategy": self.strategy.value,
                "memory_level": record.memory_level,
                "vector_similarity": round(float(vector_similarity), 4),
                "vector_score": round(float(vector_score), 4),
                "factor_score": round(float(factor_score), 4),
                "lexical": round(float(lexical), 4),
                "importance": round(float(record.importance), 4),
                "recency": round(float(recency), 4),
            }
            ranked.append((record, details))
        ranked.sort(key=lambda item: item[1]["score"], reverse=True)
        for record, _ in ranked[: self.top_k]:
            record.access_count += 1
        return ranked[: self.top_k]

    def _collect_candidates(self, query: str) -> list[tuple[MemoryRecord, float]]:
        candidate_map: dict[str, tuple[MemoryRecord, float]] = {}
        per_level_k = max(self.top_k * 4, self.top_k)
        for level in ("high", "low"):
            for record, score in self.store.search(query, top_k=per_level_k, memory_level=level):
                candidate_map[record.id] = (record, max(candidate_map.get(record.id, (record, 0.0))[1], score))

        if self.strategy in {RetrievalStrategy.THREE_FACTOR_ONLY, RetrievalStrategy.HYBRID}:
            for record in self.store.records:
                if record.id not in candidate_map and self.store.lexical_overlap(query, record) > 0:
                    candidate_map[record.id] = (record, 0.0)
            if self.strategy == RetrievalStrategy.THREE_FACTOR_ONLY:
                for record in self.store.records:
                    candidate_map.setdefault(record.id, (record, 0.0))
        return list(candidate_map.values())
