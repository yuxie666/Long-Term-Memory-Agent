from __future__ import annotations

import re

from .store import MemoryRecord, MemoryStore


class MemoryUpdater:
    """Deduplicate and softly update conflicting memories."""

    def __init__(self, duplicate_threshold: float = 0.92, conflict_threshold: float = 0.72):
        self.duplicate_threshold = duplicate_threshold
        self.conflict_threshold = conflict_threshold

    def merge_into_store(self, store: MemoryStore, records: list[MemoryRecord]) -> dict:
        stats = {"added": 0, "deduplicated": 0, "updated": 0}
        for record in records:
            if len(store) == 0:
                store.add(record)
                stats["added"] += 1
                continue

            candidates = store.search(record.text, top_k=5)
            best, best_score = candidates[0]
            if best_score >= self.duplicate_threshold:
                stats["deduplicated"] += 1
                continue

            idx = self._find_conflict_index(store, record, candidates)
            if idx is not None:
                merged = self._merge(store.records[idx], record)
                store.replace(idx, merged)
                stats["updated"] += 1
            else:
                store.add(record)
                stats["added"] += 1
        return stats

    def _find_conflict_index(
        self,
        store: MemoryStore,
        record: MemoryRecord,
        candidates: list[tuple[MemoryRecord, float]],
    ) -> int | None:
        new_terms = self._content_terms(record.object)
        if not new_terms:
            return None
        for candidate, score in candidates:
            if score < self.conflict_threshold:
                continue
            if candidate.subject != record.subject:
                continue
            old_terms = self._content_terms(candidate.object)
            overlap = len(old_terms & new_terms) / max(len(old_terms | new_terms), 1)
            if overlap >= 0.35:
                return store.records.index(candidate)
        return None

    def _merge(self, old: MemoryRecord, new: MemoryRecord) -> MemoryRecord:
        new.metadata = dict(old.metadata) | {"supersedes": old.id}
        new.evidence = old.evidence[-1:] + new.evidence
        new.importance = max(old.importance, new.importance)
        new.access_count = old.access_count
        return new

    def _content_terms(self, text: str) -> set[str]:
        terms = set(re.findall(r"[A-Za-z0-9_']+", text.lower()))
        stop = {"the", "a", "an", "and", "or", "to", "of", "in", "on", "is", "was", "are", "i", "you"}
        return {term for term in terms if term not in stop and len(term) > 2}

