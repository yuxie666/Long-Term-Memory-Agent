from __future__ import annotations

import hashlib
import re

from .store import MemoryRecord


STOP_STARTS = (
    "hi",
    "hello",
    "hey",
    "thanks",
    "thank you",
    "bye",
    "goodbye",
    "okay",
    "ok",
    "yeah",
    "yes",
    "no",
)


class MemoryWriter:
    """Builds low-level raw memories and high-level session memories."""

    def extract(self, conversation: dict) -> tuple[list[MemoryRecord], list[MemoryRecord], list[str]]:
        low_records: list[MemoryRecord] = []
        high_records: list[MemoryRecord] = []
        raw_log: list[str] = []
        counter = 0
        for sess in conversation.get("sessions", []):
            session_id = sess.get("session_id", "")
            date_time = sess.get("date_time", "")
            session_lines: list[str] = []
            source_turn_ids: list[str] = []
            for turn_index, turn in enumerate(sess.get("turns", [])):
                speaker = turn.get("speaker", "")
                text = self._clean(turn.get("text", ""))
                if not text:
                    continue
                raw_line = f"[Session {session_id} @ {date_time}] {speaker}: {text}"
                raw_log.append(raw_line)
                session_lines.append(f"{speaker}: {text}")
                mem_id = self._make_id("low", session_id, str(turn_index), speaker, text)
                source_turn_ids.append(mem_id)
                low_records.append(
                    MemoryRecord(
                        id=mem_id,
                        subject=speaker,
                        predicate="said",
                        object=text,
                        text=raw_line,
                        evidence=[raw_line],
                        session_id=session_id,
                        date_time=date_time,
                        turn_index=turn_index,
                        memory_level="low",
                        importance=self._estimate_importance(text),
                        created_at=counter,
                        updated_at=counter,
                        metadata={"dia_id": turn.get("dia_id", "")},
                    )
                )
                counter += 1
            if session_lines:
                summary = self._summarize_session(session_lines)
                mem_id = self._make_id("high", session_id, date_time, summary)
                high_records.append(
                    MemoryRecord(
                        id=mem_id,
                        subject=f"session_{session_id}",
                        predicate="summary",
                        object=summary,
                        text=f"Session {session_id} summary on {date_time}: {summary}",
                        evidence=raw_log[-len(session_lines):],
                        session_id=session_id,
                        date_time=date_time,
                        turn_index=None,
                        memory_level="high",
                        importance=self._estimate_importance(summary),
                        created_at=counter,
                        updated_at=counter,
                        metadata={"source_turn_ids": source_turn_ids},
                    )
                )
                counter += 1
        return low_records, high_records, raw_log

    def _clean(self, text: str) -> str:
        return re.sub(r"\s+", " ", text or "").strip()

    def _split_memorable_units(self, text: str) -> list[str]:
        parts = re.split(r"(?<=[.!?])\s+", text)
        units: list[str] = []
        for part in parts:
            part = part.strip(" -")
            if len(part) < 12:
                continue
            if part.lower().startswith(STOP_STARTS) and len(part.split()) < 8:
                continue
            units.append(part)
        return units[:3]

    def _summarize_session(self, session_lines: list[str]) -> str:
        candidates: list[tuple[float, int, str]] = []
        for index, line in enumerate(session_lines):
            score = self._estimate_importance(line)
            lower = line.lower()
            if any(marker in lower for marker in (" i ", " my ", " me ", " i'm ", " i'", "我", "我的")):
                score += 0.08
            if "?" in line:
                score -= 0.05
            candidates.append((score, index, line))
        candidates.sort(key=lambda item: (-item[0], item[1]))
        selected = sorted(candidates[: min(5, len(candidates))], key=lambda item: item[1])
        summary_parts = [line for _, _, line in selected]
        return "Key session facts: " + " | ".join(summary_parts)

    def _estimate_importance(self, text: str) -> float:
        lower = text.lower()
        score = 0.45
        keywords = [
            "love",
            "hate",
            "prefer",
            "favorite",
            "birthday",
            "job",
            "work",
            "family",
            "moved",
            "live",
            "allergic",
            "doctor",
            "hospital",
            "school",
            "travel",
            "plan",
            "remember",
            "changed",
            "used to",
        ]
        score += 0.08 * sum(1 for word in keywords if word in lower)
        if re.search(r"\b\d{4}\b|\b\d{1,2}[:/]\d{1,2}\b", text):
            score += 0.1
        if len(text.split()) > 18:
            score += 0.08
        return min(score, 1.0)

    def _make_id(self, *parts: object) -> str:
        digest = hashlib.sha1("|".join(str(part) for part in parts).encode("utf-8")).hexdigest()
        return digest[:12]
