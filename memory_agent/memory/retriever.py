from __future__ import annotations

import math

from .store import MemoryRecord, MemoryStore


class MemoryRetriever:
    """Three-factor retrieval: relevance, importance and recency."""

    def __init__(
        self,
        store: MemoryStore,
        top_k: int = 8,
        recency_weight: float = 0.15,
        importance_weight: float = 0.2,
        relevance_weight: float = 0.65,
        half_life: float = 80.0,
    ):
        self.store = store
        self.top_k = top_k
        self.recency_weight = recency_weight
        self.importance_weight = importance_weight
        self.relevance_weight = relevance_weight
        self.half_life = half_life

    def retrieve(self, query: str) -> list[tuple[MemoryRecord, dict]]:
        candidates = self.store.search(query, top_k=max(self.top_k * 4, self.top_k))
        if not candidates:
            return []
        seen = {record.id for record, _ in candidates}
        for record in self.store.records:
            if record.id not in seen and self.store.lexical_overlap(query, record) > 0:
                candidates.append((record, 0.0))
                seen.add(record.id)
        now = max((record.updated_at for record in self.store.records), default=0) + 1
        ranked = []
        for record, relevance in candidates:
            recency = math.exp(-max(now - record.updated_at, 0) / self.half_life)
            lexical = self.store.lexical_overlap(query, record)
            combined_relevance = max((0.45 * relevance) + (0.55 * lexical), lexical)
            score = (
                self.relevance_weight * combined_relevance
                + self.importance_weight * record.importance
                + self.recency_weight * recency
            )
            details = {
                "score": round(float(score), 4),
                "relevance": round(float(relevance), 4),
                "lexical": round(float(lexical), 4),
                "importance": round(float(record.importance), 4),
                "recency": round(float(recency), 4),
            }
            ranked.append((record, details))
        ranked.sort(key=lambda item: item[1]["score"], reverse=True)
        for record, _ in ranked[: self.top_k]:
            record.access_count += 1
        return ranked[: self.top_k]
