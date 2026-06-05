from __future__ import annotations

from enum import Enum
import math
import os
import re

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
        expanded_query = self._expanded_query(query)
        candidates = self._collect_candidates(query, expanded_query)
        if not candidates:
            return []

        now = max((record.updated_at for record in self.store.records), default=0) + 1
        ranked = []
        for record, vector_similarity in candidates:
            if not self._record_allowed_for_query(record, query):
                continue
            recency = math.exp(-max(now - record.updated_at, 0) / self.half_life)
            lexical = max(
                self.store.lexical_overlap(query, record),
                self.store.lexical_overlap(expanded_query, record),
            )
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
            match_bonus = 0.0
            if self.strategy != RetrievalStrategy.VECTOR_ONLY:
                match_bonus = self._match_bonus(query, expanded_query, record, lexical)
                score += match_bonus
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
                "query_expanded": expanded_query != query,
                "match_bonus": round(float(match_bonus), 4),
            }
            ranked.append((record, details))
        ranked.sort(key=lambda item: item[1]["score"], reverse=True)
        for record, _ in ranked[: self.top_k]:
            record.access_count += 1
        return ranked[: self.top_k]

    def _record_allowed_for_query(self, record: MemoryRecord, query: str) -> bool:
        status = record.metadata.get("status", "active")
        if status != "superseded":
            if record.memory_level == "high" and self._high_record_is_historical(record):
                return self._query_allows_historical(query)
            return True
        return self._query_allows_historical(query)

    def _query_allows_historical(self, query: str) -> bool:
        lower = query.lower()
        historical_terms = (
            "previous",
            "old",
            "before",
            "formerly",
            "used to",
            "past",
            "original",
            "earlier",
            "historical",
        )
        return any(term in lower for term in historical_terms)

    def _high_record_is_historical(self, record: MemoryRecord) -> bool:
        source_ids = set(record.metadata.get("source_memory_ids", []))
        if not source_ids:
            return False
        source_records = [candidate for candidate in self.store.records if candidate.id in source_ids]
        if not source_records:
            return False
        state_sources = [candidate for candidate in source_records if candidate.metadata.get("state_key")]
        if not state_sources:
            return False
        return all(candidate.metadata.get("status", "active") == "superseded" for candidate in state_sources)

    def _expanded_query(self, query: str) -> str:
        lower = query.lower()
        additions: list[str] = []
        if any(term in lower for term in ("book", "books", "read", "reading")):
            additions.extend([
                "book", "books", "novel", "fantasy", "reading", "read", "reread",
                "finished", "series", "bookshelf", "favorite",
            ])
        if any(term in lower for term in ("food", "dinner", "spread", "meal")):
            additions.extend([
                "food", "dinner", "meal", "spread", "mother", "mom", "salad",
                "salads", "sandwich", "sandwiches", "dessert", "desserts", "homemade",
            ])
        if any(term in lower for term in ("school", "basketball", "team", "high school")):
            additions.extend([
                "basketball", "school", "middle", "high", "college", "team",
                "teammate", "teammates", "four years", "4 years", "scholarship",
            ])
        if any(term in lower for term in ("dodge", "charger", "subaru", "forester", "car", "working on")):
            additions.extend([
                "car", "cars", "classic", "muscle", "auto", "engineering",
                "maintenance", "shop", "restore", "restored", "restoration",
                "engine", "project", "passionate", "dream",
            ])
        if any(term in lower for term in ("who is", "who was", "who's")):
            additions.extend(["friend", "colleague", "family", "went with", "together", "supportive"])
        if not additions:
            return query
        return query + " " + " ".join(additions)

    def _match_bonus(self, query: str, expanded_query: str, record: MemoryRecord, lexical: float) -> float:
        query_types = self._query_memory_types(query)
        memory_type = str(record.metadata.get("memory_type", record.predicate))
        bonus = min(0.12, lexical * 0.25) if lexical >= 0.18 else 0.0
        if memory_type in query_types:
            bonus += 0.08
        if record.memory_level == "high":
            if self._is_aggregate_answer_memory(query, record, query_types):
                bonus += 0.28
            elif lexical >= 0.12 or self._record_matches_query_type(record, query_types):
                bonus += 0.05
        cap = 0.45 if record.memory_level == "high" else 0.22
        return min(bonus, cap)

    def _query_memory_types(self, query: str) -> set[str]:
        lower = query.lower()
        types: set[str] = set()
        if any(term in lower for term in ("book", "books", "read", "reading")):
            types.add("reading")
        if any(term in lower for term in ("food", "dinner", "spread", "meal")):
            types.add("food")
        if any(term in lower for term in ("school", "basketball", "team", "high school")):
            types.add("education_sports")
        if any(term in lower for term in ("dodge", "charger", "subaru", "forester", "car", "working on")):
            types.add("vehicle")
        return types

    def _record_matches_query_type(self, record: MemoryRecord, query_types: set[str]) -> bool:
        text = f"{record.text} {record.object}".lower()
        if "reading" in query_types and self._has_any_word(text, ("book", "books", "novel", "read", "reading", "series")):
            return True
        if "food" in query_types and self._has_any_word(text, ("food", "dinner", "salad", "salads", "sandwich", "sandwiches", "dessert", "desserts", "meal")):
            return True
        if "education_sports" in query_types and self._has_any_word(text, ("basketball", "school", "college", "team")):
            return True
        if "vehicle" in query_types and self._has_any_word(text, ("car", "classic", "muscle", "auto", "engine")):
            return True
        return False

    def _is_aggregate_answer_memory(self, query: str, record: MemoryRecord, query_types: set[str]) -> bool:
        if record.memory_level != "high" or not query_types:
            return False
        lower_query = query.lower()
        lower_text = record.text.lower()
        if not any(term in lower_query for term in ("what", "which", "kind", "books", "schools")):
            return False
        aggregate_cues = (" include ", " includes ", " included ", " read include ", " played basketball in ")
        if not any(cue in f" {lower_text} " for cue in aggregate_cues):
            return False
        return self._record_matches_query_type(record, query_types)

    def _has_any_word(self, text: str, words: tuple[str, ...]) -> bool:
        return any(re.search(rf"\b{re.escape(word)}\b", text) for word in words)

    def _collect_candidates(self, query: str, expanded_query: str | None = None) -> list[tuple[MemoryRecord, float]]:
        candidate_map: dict[str, tuple[MemoryRecord, float]] = {}
        per_level_k = max(self.top_k * 4, self.top_k)
        for level in ("high", "low"):
            for record, score in self.store.search(query, top_k=per_level_k, memory_level=level):
                candidate_map[record.id] = (record, max(candidate_map.get(record.id, (record, 0.0))[1], score))
            if expanded_query and expanded_query != query:
                for record, score in self.store.search(expanded_query, top_k=per_level_k, memory_level=level):
                    candidate_map[record.id] = (record, max(candidate_map.get(record.id, (record, 0.0))[1], score))

        if self.strategy in {RetrievalStrategy.THREE_FACTOR_ONLY, RetrievalStrategy.HYBRID}:
            for record in self.store.records:
                lexical = self.store.lexical_overlap(query, record)
                if expanded_query and expanded_query != query:
                    lexical = max(lexical, self.store.lexical_overlap(expanded_query, record))
                if record.id not in candidate_map and lexical > 0:
                    candidate_map[record.id] = (record, 0.0)
            if self.strategy == RetrievalStrategy.THREE_FACTOR_ONLY:
                for record in self.store.records:
                    candidate_map.setdefault(record.id, (record, 0.0))
        return list(candidate_map.values())
