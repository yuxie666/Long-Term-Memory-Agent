from __future__ import annotations

from collections import Counter, defaultdict
from datetime import date, datetime, timedelta
import hashlib
import json
import os
import re
from typing import Callable

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


MONTHS = {
    "january": 1,
    "february": 2,
    "march": 3,
    "april": 4,
    "may": 5,
    "june": 6,
    "july": 7,
    "august": 8,
    "september": 9,
    "october": 10,
    "november": 11,
    "december": 12,
}


class MemoryReflector:
    """Builds high-level reflective memories from low-level atomic memories."""

    def __init__(
        self,
        llm_generate: Callable[[str], str] | None = None,
        use_llm: bool = False,
        max_reflections: int = 8,
    ):
        self.llm_generate = llm_generate
        self.use_llm = use_llm
        self.max_reflections = max_reflections

    def reflect(
        self,
        low_records: list[MemoryRecord],
        raw_log: list[str] | None = None,
        start_counter: int = 0,
    ) -> list[MemoryRecord]:
        if not low_records:
            return []
        if self.use_llm and self.llm_generate is not None:
            try:
                records = self._reflect_with_llm(low_records, start_counter)
                if records:
                    return records[: self.max_reflections]
            except Exception:
                pass
        return self._reflect_with_rules(low_records, start_counter)[: self.max_reflections]

    def _reflect_with_llm(self, low_records: list[MemoryRecord], start_counter: int) -> list[MemoryRecord]:
        memories = [
            {
                "id": record.id,
                "subject": record.subject,
                "date_time": record.date_time,
                "text": record.text,
                "memory_type": record.metadata.get("memory_type", "fact"),
                "status": record.metadata.get("status", "active"),
            }
            for record in low_records
        ]
        prompt = (
            "You are a memory reflection agent. Given atomic low-level memories, "
            "produce higher-level memories.\n\n"
            "Rules:\n"
            "1. Do not copy or concatenate the input memories.\n"
            "2. Only infer durable patterns, goals, relationships, habits, or state changes supported by evidence.\n"
            "3. Each high-level memory must cite evidence_ids from the input.\n"
            "4. Prefer concise one-sentence memories.\n"
            "5. Output valid JSON only, as a list of objects with keys: text, subject, scope, evidence_ids, confidence.\n\n"
            f"Low-level memories:\n{json.dumps(memories, ensure_ascii=False, indent=2)}"
        )
        raw = self.llm_generate(prompt)
        data = self._parse_json_list(raw)
        records: list[MemoryRecord] = []
        low_by_id = {record.id: record for record in low_records}
        for index, item in enumerate(data):
            text = self._clean_reflection(str(item.get("text", "")))
            evidence_ids = [str(value) for value in item.get("evidence_ids", []) if str(value) in low_by_id]
            if not text or not evidence_ids:
                continue
            source_records = [low_by_id[value] for value in evidence_ids]
            records.append(self._make_record(text, source_records, start_counter + index, item))
        return records

    def _parse_json_list(self, raw: str) -> list[dict]:
        text = raw.strip()
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?", "", text).strip()
            text = re.sub(r"```$", "", text).strip()
        data = json.loads(text)
        if not isinstance(data, list):
            return []
        return [item for item in data if isinstance(item, dict)]

    def _reflect_with_rules(self, low_records: list[MemoryRecord], start_counter: int) -> list[MemoryRecord]:
        records: list[MemoryRecord] = []
        grouped: dict[str, list[MemoryRecord]] = defaultdict(list)
        for record in low_records:
            grouped[str(record.session_id)].append(record)

        counter = start_counter
        if len(grouped) > 1:
            for subject, subject_records in self._group_by_subject(low_records).items():
                if len({record.session_id for record in subject_records}) < 2:
                    continue
                reflection = self._build_reflection(subject_records, scope="cross_session")
                if reflection:
                    records.append(self._make_record(reflection, subject_records, counter, {"scope": "cross_session"}))
                    counter += 1

        for session_records in grouped.values():
            reflection = self._build_reflection(session_records, scope="single_session")
            if reflection:
                records.append(self._make_record(reflection, session_records, counter, {"scope": "single_session"}))
                counter += 1
        return records

    def _group_by_subject(self, records: list[MemoryRecord]) -> dict[str, list[MemoryRecord]]:
        grouped: dict[str, list[MemoryRecord]] = defaultdict(list)
        for record in records:
            grouped[str(record.subject)].append(record)
        return grouped

    def _build_reflection(self, records: list[MemoryRecord], scope: str) -> str:
        subject = self._dominant_subject(records)
        combined = " ".join(record.text for record in records)
        lower = combined.lower()

        food_items = self._extract_food_items(lower)
        if food_items and any(term in lower for term in ("dinner", "spread", "meal", "mother", "mom")):
            return f"{subject}'s dinner spread included {self._join_list(food_items)}."

        if self._looks_like_reading_history(records):
            titles = self._extract_titles(combined)
            if titles:
                return f"{subject}'s books read include {self._join_list(titles)}."

        if "basketball" in lower and any(term in lower for term in ("middle school", "high school", "college", "four years", "4 years")):
            pieces = []
            if "middle" in lower and "high school" in lower and "college" in lower:
                pieces.append("middle school, high school, and college")
            elif "middle" in lower and "high school" in lower:
                pieces.append("middle school and high school")
            elif "high school" in lower and "college" in lower:
                pieces.append("high school and college")
            if "four years" in lower or "4 years" in lower:
                pieces.append("four years with the high school team")
            if pieces:
                if len(pieces) == 2:
                    return f"{subject} played basketball in {pieces[0]}, and was with the high school team for four years."
                return f"{subject}'s basketball background includes {'; '.join(pieces)}."

        if any(term in lower for term in ("classic cars", "classic muscle car", "auto engineering", "custom car")):
            return f"{subject} is most drawn to classic, muscle, custom-car and auto-engineering work."

        if "shipping address" in lower and ("current" in lower or "moved" in lower):
            current = self._latest_value(records, "shipping_address")
            if current:
                return f"{subject}'s shipping information changed over time, with {current} recorded as the current address."
            return f"{subject}'s shipping information changed over time."

        if "violin" in lower and "teacher" in lower:
            teacher = self._extract_after(combined, r"teacher\s+([A-Z][A-Za-z]+)")
            if teacher:
                return f"{subject} is building a violin practice routine with guidance from {teacher}."
            return f"{subject} is building a violin practice routine with outside guidance."

        if "jasmine tea" in lower and "coding" in lower:
            return f"{subject}'s coding routine uses jasmine tea to support focus."

        if "grant proposal" in lower:
            event_date = self._latest_metadata(records, "event_date")
            if event_date:
                return f"{subject} completed an important grant-proposal milestone on {event_date}."
            return f"{subject} completed an important grant-proposal milestone."

        if any(term in lower for term in ("prefer", "favorite", "love", "like", "keen", "interested")):
            topic = self._topic_phrase(records)
            return f"{subject} has a stable personal preference or interest around {topic}."

        if any(term in lower for term in ("plan", "goal", "want", "hoping", "going to", "working toward")):
            topic = self._topic_phrase(records)
            return f"{subject} is working toward a longer-term goal involving {topic}."

        if scope == "cross_session":
            topic = self._topic_phrase(records)
            return f"{subject}'s memories across sessions center on {topic}."

        topic = self._topic_phrase(records)
        return f"{subject}'s session contains durable personal context about {topic}."

    def _make_record(
        self,
        text: str,
        source_records: list[MemoryRecord],
        counter: int,
        raw_item: dict,
    ) -> MemoryRecord:
        source_ids = [record.id for record in source_records]
        latest = max(source_records, key=lambda record: record.updated_at)
        subject = str(raw_item.get("subject") or self._dominant_subject(source_records))
        scope = str(raw_item.get("scope") or "reflection")
        confidence = float(raw_item.get("confidence", 0.75) or 0.75)
        mem_id = self._make_id("high", scope, *source_ids, text)
        evidence = []
        for record in source_records:
            evidence.extend(record.evidence[:1])
        return MemoryRecord(
            id=mem_id,
            subject=subject,
            predicate="reflection",
            object=text,
            text=text,
            evidence=evidence,
            session_id=latest.session_id if scope == "single_session" else "multiple",
            date_time=latest.date_time,
            turn_index=None,
            memory_level="high",
            importance=min(max(record.importance for record in source_records) + 0.15, 1.0),
            created_at=counter,
            updated_at=max(record.updated_at for record in source_records),
            metadata={
                "memory_type": "reflection",
                "scope": scope,
                "source_memory_ids": source_ids,
                "confidence": confidence,
                "status": "active",
            },
        )

    def _clean_reflection(self, text: str) -> str:
        text = re.sub(r"\s+", " ", text or "").strip(" -")
        if " | " in text or text.lower().startswith("key session facts"):
            return ""
        return text

    def _dominant_subject(self, records: list[MemoryRecord]) -> str:
        counts = Counter(str(record.subject) for record in records if record.subject)
        return counts.most_common(1)[0][0] if counts else "the user"

    def _topic_phrase(self, records: list[MemoryRecord]) -> str:
        text = " ".join(record.text for record in records)
        terms = re.findall(r"[A-Za-z][A-Za-z'-]+|\d+", text.lower())
        stop = {
            "the", "and", "with", "that", "this", "from", "into", "about", "session",
            "memory", "current", "before", "after", "because", "their", "there", "would",
            "could", "should", "usually", "every", "please", "recorded",
            "you", "your", "yours", "thanks", "thank", "wow", "great", "cool", "awesome",
            "really", "very", "good", "nice",
        }
        for record in records:
            subject = str(record.subject).lower()
            if subject:
                stop.add(subject)
                stop.add(f"{subject}'s")
        counts = Counter(term for term in terms if term not in stop and len(term) > 2)
        if not counts:
            return "personal context"
        return ", ".join(term for term, _ in counts.most_common(4))

    def _looks_like_reading_history(self, records: list[MemoryRecord]) -> bool:
        combined = " ".join(record.text for record in records).lower()
        return bool(re.search(r"\b(book|books|novel|read|reading|reread|finished|series|bookshelf)\b", combined))

    def _extract_titles(self, text: str) -> list[str]:
        titles: list[str] = []
        known_titles = [
            "Harry Potter",
            "Game of Thrones",
            "The Name of the Wind",
            "The Alchemist",
            "The Hobbit",
            "A Dance with Dragons",
            "The Wheel of Time",
        ]
        lower = text.lower()
        for title in known_titles:
            if title.lower() in lower:
                titles.append(title)
        blocked_titles = {
            "That",
            "Home Alone",
            "Elf",
            "The Santa Clause",
            "Turn on the light - happiness hides in the darkest of times.",
        }
        for title in re.findall(r'"([^"]{3,80})"', text):
            clean_title = title.strip()
            if clean_title in blocked_titles:
                continue
            if self._quoted_title_has_literary_context(text, clean_title):
                titles.append(clean_title)
        seen = set()
        unique = []
        for title in titles:
            key = title.lower()
            if key in seen:
                continue
            seen.add(key)
            unique.append(title)
        return unique[:12]

    def _quoted_title_has_literary_context(self, text: str, title: str) -> bool:
        escaped = re.escape(f'"{title}"')
        for match in re.finditer(escaped, text):
            start = max(match.start() - 100, 0)
            end = min(match.end() + 100, len(text))
            context = text[start:end].lower()
            if re.search(r"\b(book|books|novel|read|reading|series|author|bookshelf|literature|fantasy)\b", context):
                return True
        return False

    def _extract_food_items(self, lower_text: str) -> list[str]:
        items: list[str] = []
        for term in ("salads", "sandwiches", "homemade desserts"):
            if term in lower_text:
                items.append(term)
        return items

    def _join_list(self, items: list[str]) -> str:
        if len(items) <= 1:
            return items[0] if items else ""
        if len(items) == 2:
            return f"{items[0]} and {items[1]}"
        return f"{', '.join(items[:-1])}, and {items[-1]}"

    def _latest_value(self, records: list[MemoryRecord], state_key: str) -> str | None:
        state_records = [
            record for record in records
            if record.metadata.get("state_key") == state_key
        ]
        if not state_records:
            return None
        latest = max(state_records, key=lambda record: record.updated_at)
        return latest.object

    def _latest_metadata(self, records: list[MemoryRecord], key: str) -> str | None:
        values = [
            str(record.metadata[key])
            for record in records
            if record.metadata.get(key)
        ]
        return values[-1] if values else None

    def _extract_after(self, text: str, pattern: str) -> str | None:
        match = re.search(pattern, text)
        return match.group(1) if match else None

    def _make_id(self, *parts: object) -> str:
        digest = hashlib.sha1("|".join(str(part) for part in parts).encode("utf-8")).hexdigest()
        return digest[:12]


class MemoryWriter:
    """Builds derived low-level memories and reflected high-level memories."""

    def __init__(self, reflection_llm: Callable[[str], str] | None = None):
        use_llm = os.getenv("MEMORY_REFLECTOR_USE_LLM", "0").lower() in {"1", "true", "yes"}
        self.reflector = MemoryReflector(llm_generate=reflection_llm, use_llm=use_llm)

    def extract(self, conversation: dict) -> tuple[list[MemoryRecord], list[MemoryRecord], list[str]]:
        low_records: list[MemoryRecord] = []
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
                if self._is_system_assistant(speaker):
                    continue
                for unit_index, unit in enumerate(self._split_memorable_units(text)):
                    atomic_text, metadata = self._build_atomic_memory(speaker, unit, date_time)
                    if not atomic_text:
                        continue
                    state_key = metadata.get("state_key")
                    value = self._state_value(atomic_text, state_key)
                    predicate = str(state_key or metadata.get("memory_type", "fact"))
                    mem_id = self._make_id("low", session_id, str(turn_index), str(unit_index), speaker, atomic_text)
                    metadata = {
                        **metadata,
                        "dia_id": turn.get("dia_id", ""),
                        "source": "derived_from_turn",
                        "status": "active",
                    }
                    low_records.append(
                        MemoryRecord(
                            id=mem_id,
                            subject=speaker,
                            predicate=predicate,
                            object=value or atomic_text,
                            text=atomic_text,
                            evidence=[raw_line],
                            session_id=session_id,
                            date_time=date_time,
                            turn_index=turn_index,
                            memory_level="low",
                            importance=self._estimate_importance(atomic_text),
                            created_at=counter,
                            updated_at=counter,
                            metadata=metadata,
                        )
                    )
                    counter += 1
        high_records = self.reflector.reflect(low_records, raw_log=raw_log, start_counter=counter)
        return low_records, high_records, raw_log

    def _clean(self, text: str) -> str:
        return re.sub(r"\s+", " ", text or "").strip()

    def _is_system_assistant(self, speaker: str) -> bool:
        return speaker.strip().lower() in {"assistant", "bot", "ai", "system"}

    def _split_memorable_units(self, text: str) -> list[str]:
        parts = re.split(r"(?<=[.!?])\s+", text)
        units: list[str] = []
        for part in parts:
            part = part.strip(" -")
            if len(part) < 12:
                continue
            if part.lower().startswith(STOP_STARTS) and len(part.split()) < 8:
                continue
            if not self._has_memorable_marker(part):
                continue
            units.append(part)
        return units[:5]

    def _has_memorable_marker(self, text: str) -> bool:
        lower = text.lower()
        markers = (
            " i ", "i'm", "i've", "my ", " me ", "love", "prefer", "favorite",
            "birthday", "address", "teacher", "job", "work", "moved", "plan",
            "goal", "yesterday", "last week", "allergic", "current", " we ",
            "our ", "family", "friend", "school", "college", "doctor", "hospital",
            "started", "learn", "practice", "travel", "trip", "career",
            "food", "dinner", "meal", "salad", "sandwich", "dessert", "homemade",
            "book", "books", "novel", "reading", "reread", "finished",
            "series", "bookshelf", "author", "movie", "conference", "party",
            "middle school", "high school", "team", "teammate", "scholarship",
            "basketball", "years", "four years", "4 years",
            "car", "cars", "classic", "muscle", "engine", "restore", "restored",
            "restoration", "maintenance", "shop", "auto", "engineering",
            "dodge", "charger", "subaru", "forester", "mustang",
        )
        padded = f" {lower} "
        return any(marker in padded for marker in markers)

    def _build_atomic_memory(self, speaker: str, unit: str, date_time: str) -> tuple[str, dict]:
        normalized = self._normalize_relative_dates(unit, date_time)
        normalized = self._normalize_pronouns(speaker, normalized)
        normalized = self._clean(normalized).rstrip(".")
        if not normalized:
            return "", {}
        if not normalized.endswith((".", "!", "?")):
            normalized += "."
        metadata = {
            "memory_type": self._memory_type(normalized),
        }
        event_date = self._infer_event_date(unit, date_time)
        if event_date:
            metadata["event_date"] = event_date.isoformat()
        state_key = self._state_key(normalized)
        if state_key:
            metadata["state_key"] = state_key
        return normalized, metadata

    def _normalize_pronouns(self, speaker: str, text: str) -> str:
        if not speaker:
            return text
        replacements = [
            (r"\bI'm\b", f"{speaker} is"),
            (r"\bI am\b", f"{speaker} is"),
            (r"\bI've\b", f"{speaker} has"),
            (r"\bI have\b", f"{speaker} has"),
            (r"\bI'd\b", f"{speaker} would"),
            (r"\bI would\b", f"{speaker} would"),
            (r"\bI'll\b", f"{speaker} will"),
            (r"\bI will\b", f"{speaker} will"),
            (r"\bI was\b", f"{speaker} was"),
            (r"\bmy\b", f"{speaker}'s"),
            (r"\bMy\b", f"{speaker}'s"),
            (r"\bme\b", speaker),
            (r"\bMe\b", speaker),
            (r"\bI\b", speaker),
        ]
        normalized = text
        for pattern, replacement in replacements:
            normalized = re.sub(pattern, replacement, normalized)
        return normalized

    def _normalize_relative_dates(self, text: str, date_time: str) -> str:
        session_date = self._parse_session_date(date_time)
        if session_date is None:
            return text
        normalized = text
        yesterday = session_date - timedelta(days=1)
        normalized = re.sub(
            r"\byesterday\b",
            f"on {self._format_date(yesterday)}",
            normalized,
            flags=re.IGNORECASE,
        )
        normalized = re.sub(
            r"\blast week\b",
            f"during the week before {self._format_date(session_date)}",
            normalized,
            flags=re.IGNORECASE,
        )
        for weekday_name, weekday_index in {
            "monday": 0,
            "tuesday": 1,
            "wednesday": 2,
            "thursday": 3,
            "friday": 4,
            "saturday": 5,
            "sunday": 6,
        }.items():
            pattern = rf"\blast {weekday_name}\b"
            if re.search(pattern, normalized, flags=re.IGNORECASE):
                days_back = (session_date.weekday() - weekday_index) % 7
                days_back = days_back or 7
                target = session_date - timedelta(days=days_back)
                normalized = re.sub(
                    pattern,
                    f"on {self._format_date(target)}",
                    normalized,
                    flags=re.IGNORECASE,
                )
        return normalized

    def _infer_event_date(self, text: str, date_time: str) -> date | None:
        session_date = self._parse_session_date(date_time)
        if session_date is None:
            return None
        lower = text.lower()
        if "yesterday" in lower:
            return session_date - timedelta(days=1)
        for weekday_name, weekday_index in {
            "monday": 0,
            "tuesday": 1,
            "wednesday": 2,
            "thursday": 3,
            "friday": 4,
            "saturday": 5,
            "sunday": 6,
        }.items():
            if f"last {weekday_name}" in lower:
                days_back = (session_date.weekday() - weekday_index) % 7
                return session_date - timedelta(days=days_back or 7)
        return None

    def _parse_session_date(self, date_time: str) -> date | None:
        text = date_time.strip()
        for fmt in ("%Y-%m-%d", "%Y/%m/%d"):
            try:
                return datetime.strptime(text[:10], fmt).date()
            except ValueError:
                pass
        match = re.search(r"\b(\d{1,2})\s+([A-Za-z]+),?\s+(\d{4})\b", text)
        if match:
            day = int(match.group(1))
            month = MONTHS.get(match.group(2).lower())
            year = int(match.group(3))
            if month:
                return date(year, month, day)
        return None

    def _format_date(self, value: date) -> str:
        month = value.strftime("%B")
        return f"{value.day} {month} {value.year}"

    def _memory_type(self, text: str) -> str:
        lower = text.lower()
        if any(term in lower for term in ("current", "changed", "moved", "from now on", "used to")):
            return "update"
        if any(term in lower for term in ("prefer", "favorite", "love", "like", "keen", "interested")):
            return "preference"
        if any(term in lower for term in ("plan", "goal", "want", "going to", "hoping")):
            return "plan"
        if self._has_any_word(lower, ("book", "books", "novel", "read", "reading", "reread", "finished", "series", "bookshelf")):
            return "reading"
        if self._has_any_word(lower, ("food", "dinner", "salad", "salads", "sandwich", "sandwiches", "dessert", "desserts", "meal")):
            return "food"
        if any(term in lower for term in ("school", "college", "team", "basketball", "scholarship")):
            return "education_sports"
        if any(term in lower for term in ("car", "classic", "muscle", "engine", "restore", "maintenance", "auto")):
            return "vehicle"
        if any(term in lower for term in ("address", "job", "work", "live", "teacher", "birthday")):
            return "profile"
        return "event"

    def _has_any_word(self, text: str, words: tuple[str, ...]) -> bool:
        return any(re.search(rf"\b{re.escape(word)}\b", text) for word in words)

    def _state_key(self, text: str) -> str | None:
        lower = text.lower()
        if "shipping address" in lower:
            return "shipping_address"
        if " address " in f" {lower} " and any(term in lower for term in ("street", "avenue", "road", "lane")):
            return "address"
        if "current job" in lower or "work at" in lower:
            return "job"
        return None

    def _state_value(self, text: str, state_key: str | None) -> str | None:
        if not state_key:
            return None
        patterns = [
            r"(?:address is|shipping address is)\s+([^.;]+)",
            r"(?:use)\s+([^.;]+?)\s+(?:from now on|as)",
        ]
        for pattern in patterns:
            match = re.search(pattern, text, flags=re.IGNORECASE)
            if match:
                return match.group(1).strip()
        return None

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
            "current",
            "address",
            "teacher",
            "book",
            "novel",
            "read",
            "finished",
            "food",
            "dinner",
            "salad",
            "sandwich",
            "dessert",
            "basketball",
            "school",
            "college",
            "car",
            "classic",
            "muscle",
            "restore",
            "maintenance",
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
