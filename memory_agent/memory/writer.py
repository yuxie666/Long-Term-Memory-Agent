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

from memory.store import MemoryUnit, parse_date


# 高层记忆抽取 prompt：对标主流 agent 的「事实 + 反思」型高层记忆。
SUMMARY_PROMPT = """You are the memory module of a conversational agent. Read one full chat session between {speaker_a} and {speaker_b} and write down the high-level memories worth keeping long-term — the same kind a human would jot in a diary about this conversation.

Extract:
- Key facts about each person (identity, job, relationships, location, health, preferences).
- Events with their dates/times, plans, decisions, and changes/updates to earlier facts.
- One or two short reflections (higher-level takeaways) if warranted.

Rules:
- Each memory is ONE self-contained sentence; resolve pronouns to names.
- PRESERVE SPECIFIC DETAILS verbatim: proper nouns (book/movie/place/brand/car/song names), numbers, quantities, and list items. Do NOT generalize them away — e.g. write "Tim read Game of Thrones, The Hobbit, and The Alchemist", NOT "Tim read some fantasy books". If a turn enumerates several items, keep ALL of them in the memory.
- IMPORTANT: convert every relative time expression into an ABSOLUTE date, computed from the session date given below. e.g. if the session date is 12 July 2023, then "two days ago" -> "on 10 July 2023", "last Friday" -> the actual date of that Friday, "yesterday" -> 11 July 2023, "next month" -> the corresponding month. Write the absolute date directly in the sentence. If a time expression is too vague to pin down (e.g. "a few years ago"), keep it as stated.
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
                # 带上说话人和日期，让这条观察记忆自包含
                units.append(MemoryUnit(
                    mem_id=-1, text=f"{t['speaker']} ({sess['date_time']}): {text}",
                    kind="observation", speaker=t["speaker"],
                    session_id=sess["session_id"], date_time=sess["date_time"],
                    importance=1.0, timestamp=ts, source_dia_ids=[t["dia_id"]],
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


def _clamp_importance(v) -> float:
    try:
        return float(max(1, min(10, int(v))))
    except (ValueError, TypeError):
        return 5.0


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
