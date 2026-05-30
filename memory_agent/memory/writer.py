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
    """Extracts structured memory units from raw dialogue turns."""

    def extract(self, conversation: dict) -> tuple[list[MemoryRecord], list[str]]:
        records: list[MemoryRecord] = []
        raw_log: list[str] = []
        counter = 0
        for sess in conversation.get("sessions", []):
            session_id = sess.get("session_id", "")
            date_time = sess.get("date_time", "")
            for turn_index, turn in enumerate(sess.get("turns", [])):
                speaker = turn.get("speaker", "")
                text = self._clean(turn.get("text", ""))
                if not text:
                    continue
                raw_line = f"[Session {session_id} @ {date_time}] {speaker}: {text}"
                raw_log.append(raw_line)
                for unit in self._split_memorable_units(text):
                    importance = self._estimate_importance(unit)
                    mem_text = f"{speaker} said on {date_time}: {unit}"
                    mem_id = self._make_id(speaker, unit, date_time, counter)
                    records.append(
                        MemoryRecord(
                            id=mem_id,
                            subject=speaker,
                            predicate="said",
                            object=unit,
                            text=mem_text,
                            evidence=[raw_line],
                            session_id=session_id,
                            date_time=date_time,
                            turn_index=turn_index,
                            importance=importance,
                            created_at=counter,
                            updated_at=counter,
                        )
                    )
                    counter += 1
        return records, raw_log

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

    def _make_id(self, speaker: str, text: str, date_time: str, counter: int) -> str:
        digest = hashlib.sha1(f"{speaker}|{text}|{date_time}|{counter}".encode("utf-8")).hexdigest()
        return digest[:12]

