"""
Memory Store —— 记忆存储、索引与持久化。

主流 Agent 记忆架构（Generative Agents memory-stream + Mem0 facts）采用**两层记忆**：
  - kind="observation"：细粒度观察（一条对话轮次一条），保留原始细节，无需 LLM —— 保证召回
  - kind="summary"    ：高层记忆（按 session 汇总抽取的事实/反思）—— 保证抽象与跨会话

两层都进同一个向量库，检索时**一起打分**（这是召回准确的关键，不只检索高层记忆）。

持久化：
  - cache（<key>.cache.json + .npy）：units + 向量矩阵，下次直接 load，抽取只做一次
  - 人类可读（<key>.highlevel.md）：让你直接看到抽取出的高层记忆
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Optional

import numpy as np

CACHE_VERSION = "v4"  # 抽取 prompt 强化(保留专有名词/列举项)，旧缓存(v3)失效需重建


def parse_date(date_time: str) -> Optional[datetime]:
    """把 LoCoMo 的 date_time（如 "9:39 am on 15 October, 2023"）解析成 datetime。"""
    s = date_time.strip().replace(",", "")
    for fmt in ("%I:%M %p on %d %B %Y", "%I:%M%p on %d %B %Y"):
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            continue
    return None


def conversation_key(conversation: dict) -> str:
    """根据对话内容算一个稳定 key（run_generation 不传 sample_id，用内容哈希兜底）。"""
    a = conversation.get("speaker_a", "A")
    b = conversation.get("speaker_b", "B")
    dia = "|".join(t["dia_id"] for s in conversation["sessions"] for t in s["turns"])
    h = hashlib.sha1(f"{a}|{b}|{dia}".encode("utf-8")).hexdigest()[:10]
    safe = lambda x: "".join(c for c in x if c.isalnum()) or "x"
    return f"{safe(a)}_{safe(b)}_{h}"


@dataclass
class MemoryUnit:
    """一条记忆单元。kind 区分两层：observation（细粒度）/ summary（高层）。"""
    mem_id: int
    text: str
    kind: str                       # "observation" | "summary"
    speaker: str
    session_id: int
    date_time: str
    importance: float = 1.0         # 重要性 1~10（observation 给基准分，summary 由 LLM 打分）
    timestamp: Optional[datetime] = None
    source_dia_ids: list = field(default_factory=list)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["timestamp"] = self.timestamp.isoformat() if self.timestamp else None
        return d

    @staticmethod
    def from_dict(d: dict) -> "MemoryUnit":
        ts = d.get("timestamp")
        return MemoryUnit(
            mem_id=d["mem_id"], text=d["text"], kind=d["kind"],
            speaker=d["speaker"], session_id=d["session_id"],
            date_time=d["date_time"], importance=d.get("importance", 1.0),
            timestamp=datetime.fromisoformat(ts) if ts else None,
            source_dia_ids=d.get("source_dia_ids", []),
        )


class MemoryStore:
    """两层记忆容器 + 向量索引 + 持久化。"""

    def __init__(self, cache_dir: str = None):
        from sentence_transformers import SentenceTransformer
        embed_model = os.getenv("EMBED_MODEL", "BAAI/bge-small-zh-v1.5")
        print(f"[Store] 加载 embedding 模型: {embed_model}")
        self._embedder = SentenceTransformer(embed_model)

        self.units: list[MemoryUnit] = []
        self._matrix: Optional[np.ndarray] = None      # (N, dim) 归一化向量
        self.cache_dir = cache_dir or os.path.join(
            os.path.dirname(os.path.dirname(__file__)), "experiments", "results", "mem_cache")
        os.makedirs(self.cache_dir, exist_ok=True)

    # ---- embedding（批量，速度关键）----
    def embed(self, texts) -> np.ndarray:
        if isinstance(texts, str):
            texts = [texts]
        vecs = self._embedder.encode(
            texts, normalize_embeddings=True, batch_size=64,
            show_progress_bar=False,
        )
        return np.asarray(vecs, dtype=np.float32)

    def add_batch(self, units: list[MemoryUnit]) -> None:
        """一次性加入多条记忆并批量编码（比逐条 add 快得多）。"""
        if not units:
            return
        base = len(self.units)
        for i, u in enumerate(units):
            u.mem_id = base + i
        vecs = self.embed([u.text for u in units])
        self.units.extend(units)
        self._matrix = vecs if self._matrix is None else np.vstack([self._matrix, vecs])

    def similarities(self, query: str) -> np.ndarray:
        if self._matrix is None or len(self.units) == 0:
            return np.zeros((0,), dtype=np.float32)
        return self._matrix @ self.embed(query)[0]

    def __len__(self) -> int:
        return len(self.units)

    # ---- 持久化：cache（机器读，复用）----
    def _cache_paths(self, key: str):
        base = os.path.join(self.cache_dir, f"{key}.{CACHE_VERSION}")
        return base + ".cache.json", base + ".npy"

    def has_cache(self, key: str) -> bool:
        j, n = self._cache_paths(key)
        return os.path.exists(j) and os.path.exists(n)

    def save_cache(self, key: str) -> None:
        j, n = self._cache_paths(key)
        with open(j, "w", encoding="utf-8") as f:
            json.dump([u.to_dict() for u in self.units], f, ensure_ascii=False, indent=2)
        np.save(n, self._matrix if self._matrix is not None else np.zeros((0, 1)))
        print(f"[Store] 已缓存 {len(self.units)} 条记忆 -> {os.path.basename(j)}")

    def load_cache(self, key: str) -> None:
        j, n = self._cache_paths(key)
        with open(j, encoding="utf-8") as f:
            self.units = [MemoryUnit.from_dict(d) for d in json.load(f)]
        self._matrix = np.load(n).astype(np.float32)
        print(f"[Store] 命中缓存，直接加载 {len(self.units)} 条记忆（跳过抽取）")

    # ---- 持久化：高层记忆（人读）----
    def dump_highlevel_md(self, key: str, conversation: dict) -> None:
        """把高层记忆写成 markdown，方便你直接查看抽取结果。"""
        path = os.path.join(self.cache_dir, f"{key}.highlevel.md")
        sums = [u for u in self.units if u.kind == "summary"]
        with open(path, "w", encoding="utf-8") as f:
            f.write(f"# 高层记忆 | {conversation.get('speaker_a')} & "
                    f"{conversation.get('speaker_b')}\n\n")
            f.write(f"共 {len(sums)} 条高层记忆 "
                    f"（另有 {len(self.units)-len(sums)} 条细粒度观察记忆未列出）\n\n")
            cur = None
            for u in sorted(sums, key=lambda x: x.session_id):
                if u.session_id != cur:
                    cur = u.session_id
                    f.write(f"\n## Session {u.session_id} @ {u.date_time}\n")
                f.write(f"- (imp={u.importance:.0f}) {u.text}\n")
        print(f"[Store] 高层记忆已导出 -> {os.path.basename(path)}")
