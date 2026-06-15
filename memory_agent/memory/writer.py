"""
Memory Writer —— 从对话中提取记忆（两层）。

主流 Agent 做法（Generative Agents + Mem0）：
  1) observation 层：每条对话轮次直接转成一条细粒度记忆（无需 LLM，批量 embed）
  2) summary 层：把**整个 session 的原始文本拼在一起**，一次 LLM 调用总结出高层记忆

速度关键：
  - 每个 session 只 1 次 LLM 调用（不再每条事实一次冲突判定）
  - 各 session 的总结**并发**执行（线程池），互不依赖
"""

from __future__ import annotations

import json
import re
from concurrent.futures import ThreadPoolExecutor

from memory.store import MemoryUnit, parse_date, annotate_relative_dates


# 高层记忆抽取 prompt：对标主流 agent 的「事实 + 反思」型高层记忆。
SUMMARY_PROMPT = """You are the memory module of a conversational agent. Read one full chat session between {speaker_a} and {speaker_b} and write down the high-level memories worth keeping long-term — the same kind a human would jot in a diary about this conversation.

Extract:
- Key facts about each person (identity, job, relationships, location, health, preferences).
- Events with their dates/times, plans, decisions, and changes/updates to earlier facts.
- One or two short reflections (higher-level takeaways) if warranted.
- EXHAUSTIVE LIST ITEMS: When a speaker mentions multiple items, activities, books, places, or any enumerated list, extract ALL items as separate memories OR keep them together in ONE memory - do NOT omit any items.

Rules:
- Each memory is ONE self-contained sentence; resolve pronouns to names.
- PRESERVE SPECIFIC DETAILS verbatim: proper nouns (book/movie/place/brand/car/song names), numbers, quantities, and list items. Do NOT generalize them away — e.g. write "Tim read Game of Thrones, The Hobbit, and The Alchemist", NOT "Tim read some fantasy books". If a turn enumerates several items, keep ALL of them in the memory.
- CRITICAL TIME NORMALIZATION: Convert EVERY relative time expression to an ABSOLUTE date using the session date below:
  * "two days ago" → compute exact date (e.g., if session is 12 July 2023, write "on 10 July 2023")
  * "last Friday" → compute the actual Friday date before the session
  * "yesterday" → the day before session date
  * "next month" → the specific month name and year
  * "the weekend of X" → compute the actual weekend dates (e.g., "weekend of 22-23 July 2023")
  * "a few days before X" → compute the approximate date range
  * "the week before X" → compute the actual week dates
  * If a time expression is genuinely vague (e.g., "a few years ago", "sometime in 2022"), keep it as stated but add any available context.
  * ALWAYS show your computation: write "[= computed date]" after the original expression for clarity.
- For questions about "what activities", "what books", "what places", extract ALL mentioned items, not just some.
- Skip greetings and pure small talk.
- Rate importance 1-10 (10 = core identity / major life event).

Return ONLY a JSON array, each item: {{"fact": "...", "about": "<name>", "importance": <int>}}

Session date: {date_time}
=== Session ===
{dialogue}
=== End ==="""


class MemoryWriter:
    def __init__(self, llm, max_workers: int = 8, verbose: bool = True):
        self.llm = llm
        self.max_workers = max_workers
        self.verbose = verbose

    # ---- observation 层：无 LLM，直接把每轮转成记忆 ----
    def build_observations(self, conversation: dict) -> list[MemoryUnit]:
        units = []
        for sess in conversation["sessions"]:
            ts = parse_date(sess["date_time"])
            for t in sess["turns"]:
                text = (t.get("text") or "").strip()
                if not text:
                    continue
                # D：确定性地把相对时间（"last Friday" 等）就地标注成绝对日期，
                # 让无 LLM 的 observation 层也带上精确时间，供时间题精排/生成使用。
                text = annotate_relative_dates(text, ts)
                # 带上说话人和日期，让这条观察记忆自包含
                units.append(MemoryUnit(
                    mem_id=-1, text=f"{t['speaker']} ({sess['date_time']}): {text}",
                    kind="observation", speaker=t["speaker"],
                    session_id=sess["session_id"], date_time=sess["date_time"],
                    importance=_observation_importance(text),
                    timestamp=ts, source_dia_ids=[t["dia_id"]],
                ))
        if self.verbose:
            print(f"[Writer] observation 层：{len(units)} 条细粒度记忆（无 LLM 调用）")
        return units

    # ---- summary 层：每 session 1 次 LLM，并发执行 ----
    def build_summaries(self, conversation: dict) -> list[MemoryUnit]:
        speaker_a = conversation.get("speaker_a", "A")
        speaker_b = conversation.get("speaker_b", "B")
        sessions = conversation["sessions"]

        def work(sess):
            dialogue = "\n".join(f"{t['speaker']}: {t['text']}" for t in sess["turns"])
            prompt = SUMMARY_PROMPT.format(
                speaker_a=speaker_a, speaker_b=speaker_b,
                date_time=sess["date_time"], dialogue=dialogue)
            raw = self.llm.generate(prompt, max_tokens=700)
            facts = _parse_json_array(raw)
            out = []
            for f in facts:
                text = str(f.get("fact", "")).strip()
                if not text:
                    continue
                out.append(MemoryUnit(
                    mem_id=-1, text=text, kind="summary",
                    speaker=str(f.get("about", "")).strip() or speaker_a,
                    session_id=sess["session_id"], date_time=sess["date_time"],
                    importance=_clamp_importance(f.get("importance", 5)),
                    timestamp=parse_date(sess["date_time"]),
                    source_dia_ids=[t["dia_id"] for t in sess["turns"]],
                ))
            if self.verbose:
                print(f"[Writer]   session {sess['session_id']} -> {len(out)} 条高层记忆")
            return out

        # 并发总结所有 session（IO 等待的是 LLM 服务，线程池足够）
        summaries = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as ex:
            for res in ex.map(work, sessions):
                summaries.extend(res)
        if self.verbose:
            print(f"[Writer] summary 层：{len(summaries)} 条高层记忆 "
                  f"（{len(sessions)} 次 LLM 调用，并发）")
        return summaries

    def merge_cross_session_memories(self, memories: list[MemoryUnit]) -> list[MemoryUnit]:
        """[已停用] 早期版本用关键词把"同主题"记忆拼成一条，但实测它把无关事实
        拼成乱句（如 "Audrey has four dogs..., and two Chihuahua mixes"），既损召回又
        污染生成上下文。近重复合并交给 updater 的 embedding dedup 即可，这里直接透传。
        保留方法签名以兼容 controller 调用与历史消融。"""
        return memories


def _clamp_importance(v) -> float:
    try:
        return float(max(1, min(10, int(v))))
    except (ValueError, TypeError):
        return 5.0


def _observation_importance(text: str) -> float:
    """Heuristic 1-10 importance for raw observation turns without extra LLM calls."""
    if not text:
        return 1.0

    raw = text.strip()
    low = raw.lower()
    words = re.findall(r"[a-z0-9']+", low)
    score = 2.0

    smalltalk = (
        "hi", "hello", "hey", "how are you", "how's it going", "thanks",
        "thank you", "nice", "cool", "awesome", "great to hear", "sounds good",
    )
    if len(words) <= 8 and any(p in low for p in smalltalk):
        score = 1.0

    time_patterns = (
        r"\b\d{1,2}\s+\w+\s+\d{4}\b",
        r"\b\d{4}\b",
        r"\b(yesterday|tomorrow|last|next|ago|week|month|year|today)\b",
        r"\[=\s*\d{1,2}\s+\w+\s+\d{4}\]",
    )
    if any(re.search(p, raw, re.IGNORECASE) for p in time_patterns):
        score += 2.0

    if re.search(r"\b\d+\b", raw):
        score += 1.0
    if re.search(r'"[^"]+"|\'[^\']+\'', raw):
        score += 1.0
    proper = re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b", raw)
    if len(proper) >= 2:
        score += 1.0

    preference_terms = (
        "like", "love", "enjoy", "favorite", "prefer", "hobby", "passion",
        "interested", "into ", "fan of", "dislike", "hate",
    )
    if any(t in low for t in preference_terms):
        score += 1.5

    profile_terms = (
        "job", "work", "career", "school", "college", "university", "family",
        "mother", "father", "sister", "brother", "partner", "wife", "husband",
        "friend", "dog", "cat", "pet", "live", "moved", "home", "health",
        "doctor", "hospital", "diagnosed", "therapy",
    )
    if any(t in low for t in profile_terms):
        score += 1.5

    event_terms = (
        "went", "visited", "traveled", "bought", "adopted", "started",
        "finished", "won", "lost", "failed", "passed", "met", "joined",
        "decided", "planning", "plan to", "will", "going to", "recently",
    )
    if any(t in low for t in event_terms):
        score += 1.5

    if "," in raw or " and " in low:
        score += 0.5

    major_terms = (
        "married", "divorced", "pregnant", "graduated", "promotion",
        "accident", "surgery", "death", "breakup", "engaged", "military",
        "award", "competition", "contest",
    )
    if any(t in low for t in major_terms):
        score += 2.0

    if len(words) <= 5 and score <= 3.0:
        score = min(score, 2.0)
    return float(max(1.0, min(10.0, round(score, 1))))


def _parse_json_array(raw: str) -> list:
    """从模型输出里稳健地抠出 JSON 数组（容忍 code fence / 前后多余文本）。"""
    if not raw:
        return []
    raw = re.sub(r"```(?:json)?", "", raw).strip()
    start, end = raw.find("["), raw.rfind("]")
    if start != -1 and end != -1 and end > start:
        raw = raw[start:end + 1]
    try:
        data = json.loads(raw)
        return data if isinstance(data, list) else []
    except json.JSONDecodeError:
        return []
