from __future__ import annotations

import re

from .store import MemoryRecord, MemoryStore


class MemoryUpdater:
    """Deduplicate and softly update conflicting memories."""

    def __init__(self, duplicate_threshold: float = 0.92, conflict_threshold: float = 0.72):
        self.duplicate_threshold = duplicate_threshold
        self.conflict_threshold = conflict_threshold

    def merge_into_store(self, store: MemoryStore, records: list[MemoryRecord]) -> dict:
        stats = {"added": 0, "deduplicated": 0, "updated": 0, "superseded": 0}
        for record in records:
            if any(existing.id == record.id for existing in store.records):
                stats["deduplicated"] += 1
                continue
            state_key = record.metadata.get("state_key")
            if state_key:
                state_stats = self._merge_state_record(store, record, str(state_key))
                for key, value in state_stats.items():
                    stats[key] += value
                continue
            if record.memory_level == "low":
                store.add(record)
                stats["added"] += 1
                continue
            if len(store) == 0:
                store.add(record)
                stats["added"] += 1
                continue

            candidates = [
                item for item in store.search(record.text, top_k=5, memory_level=record.memory_level)
                if item[0].memory_level == record.memory_level
            ]
            if not candidates:
                store.add(record)
                stats["added"] += 1
                continue
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

    def _merge_state_record(self, store: MemoryStore, record: MemoryRecord, state_key: str) -> dict:
        stats = {"added": 0, "deduplicated": 0, "updated": 0, "superseded": 0}
        normalized_new = self._normalize_value(record.object or record.text)
        active_matches: list[tuple[int, MemoryRecord]] = []
        for index, existing in enumerate(store.records):
            if existing.memory_level != record.memory_level:
                continue
            if existing.subject != record.subject:
                continue
            if existing.metadata.get("state_key") != state_key:
                continue
            if existing.metadata.get("status", "active") != "active":
                continue
            normalized_existing = self._normalize_value(existing.object or existing.text)
            if normalized_existing == normalized_new:
                stats["deduplicated"] += 1
                return stats
            active_matches.append((index, existing))

        for index, existing in active_matches:
            updated = MemoryRecord(**existing.to_dict())
            updated.metadata = dict(updated.metadata)
            updated.metadata["status"] = "superseded"
            updated.metadata["superseded_by"] = record.id
            updated.metadata["superseded_at"] = record.date_time
            store.replace(index, updated)
            stats["updated"] += 1
            stats["superseded"] += 1

        record.metadata = dict(record.metadata)
        record.metadata.setdefault("status", "active")
        if active_matches:
            record.metadata["supersedes"] = [existing.id for _, existing in active_matches]
            record.evidence = [line for _, existing in active_matches for line in existing.evidence[-1:]] + record.evidence
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
            if candidate.memory_level != record.memory_level:
                continue
            if candidate.subject != record.subject:
                continue
            old_terms = self._content_terms(candidate.object)
            overlap = len(old_terms & new_terms) / max(len(old_terms | new_terms), 1)
            if overlap >= 0.35:
                return store.records.index(candidate)
        return None

    def _merge(self, old: MemoryRecord, new: MemoryRecord) -> MemoryRecord:
        if old.memory_level == "high" and new.memory_level == "high":
            return self._merge_high_memory(old, new)
        new.metadata = dict(old.metadata) | {"supersedes": old.id}
        new.evidence = old.evidence[-1:] + new.evidence
        new.importance = max(old.importance, new.importance)
        new.access_count = old.access_count
        return new

    def _merge_high_memory(self, old: MemoryRecord, new: MemoryRecord) -> MemoryRecord:
        base, other = (old, new)
        if self._specificity_score(new) > self._specificity_score(old):
            base, other = new, old
        merged = MemoryRecord(**base.to_dict())
        merged.metadata = dict(base.metadata)
        base_sources = list(base.metadata.get("source_memory_ids", []))
        other_sources = list(other.metadata.get("source_memory_ids", []))
        merged.metadata["source_memory_ids"] = self._stable_union(base_sources, other_sources)
        merged.metadata["merged_from"] = self._stable_union(
            list(base.metadata.get("merged_from", [])) + [base.id],
            list(other.metadata.get("merged_from", [])) + [other.id],
        )
        merged.evidence = self._stable_union(base.evidence, other.evidence)[-8:]
        merged.importance = max(old.importance, new.importance)
        merged.updated_at = max(old.updated_at, new.updated_at)
        merged.access_count = max(old.access_count, new.access_count)
        return merged

    def _specificity_score(self, record: MemoryRecord) -> float:
        source_count = len(record.metadata.get("source_memory_ids", []))
        concrete_items = record.text.count(",") + record.text.count(";")
        quoted_or_named = len(re.findall(r"\b[A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+)*\b", record.text))
        return len(record.text.split()) + concrete_items * 3 + quoted_or_named + source_count * 0.02

    def _stable_union(self, first: list, second: list) -> list:
        seen = set()
        values = []
        for item in first + second:
            key = str(item)
            if key in seen:
                continue
            seen.add(key)
            values.append(item)
        return values

    def _normalize_value(self, text: str) -> str:
        return re.sub(r"\s+", " ", text.lower()).strip(" .;")

    def _content_terms(self, text: str) -> set[str]:
        terms = set(re.findall(r"[A-Za-z0-9_']+", text.lower()))
        stop = {"the", "a", "an", "and", "or", "to", "of", "in", "on", "is", "was", "are", "i", "you"}
        return {term for term in terms if term not in stop and len(term) > 2}
