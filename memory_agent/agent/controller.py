"""
Agent Controller —— 主流程编排：写入 → 检索 → 生成（对标主流 Agent 记忆架构）。

记忆架构（两层，Generative Agents + Mem0）：
  ingest 时构建 observation（细粒度，无 LLM）+ summary（高层，每 session 1 次 LLM，并发），
  去重后一起进同一个向量库；检索时**两层一起打分**。

加速三件套：
  1) 抽取只做一次：构建完成后写 cache，重跑直接 load（has_cache → load_cache）
  2) 每 session 1 次 LLM + 并发；observation 层零 LLM
  3) 批量 embedding

可切换检索策略：vector / three_factor / hybrid（见 retriever.py）。
可追踪：每题把 检索记忆/完整 prompt/输出 落到 trace_*.jsonl。
"""

from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

_PKG_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PKG_ROOT))
sys.path.insert(0, str(_PKG_ROOT.parent / "eval_kit"))

from memory.store import MemoryStore, conversation_key      # noqa: E402
from memory.writer import MemoryWriter                       # noqa: E402
from memory.retriever import MemoryRetriever                 # noqa: E402
from memory.updater import MemoryUpdater                     # noqa: E402


ANSWER_PROMPT = """Answer a question about two people's past conversations based ONLY on the retrieved memories.

Guidelines:
1. Direct facts: If a memory directly states the answer, quote or state that fact.
2. One-step inference: If the answer is implied, infer the answer (e.g., "went surfing" implies "likes surfing").
3. Multi-hop reasoning: If the answer requires combining information from multiple memories, do so:
   - Memory A: "X recommended a book to Y"
   - Memory B: "X's favorite book is Z"
   - Conclusion: Y read book Z
4. Lists: For "what" questions, extract ALL matching items from ALL memories, not just the first match.
5. Hypothetical questions ("would", "likely"): Infer from preferences and past behavior with a brief reason.
6. When questions: Extract the date/time as stated in the memory.

Rule: Answer "unknown" ONLY when no memory contains or implies the answer. A reasoned inference beats "unknown".

Answer format: Short phrase or sentence. No preamble.

=== Memories ===
{memories}

=== Question ===
{question}

=== Answer ==="""

# 时间类问题专用 prompt：精度第一。保留记忆中的时间粒度，不妄自换算。
TEMPORAL_ANSWER_PROMPT = """Answer a TIME question about two people's past conversations. Date/time precision matters.

Rules:
1. Find the memory that describes the event, then extract its date/time.
2. Answer with the SAME GRANULARITY as stated in the memory:
   - If the memory says "first week of October 2023", answer "first week of October 2023" — do NOT convert to a specific day.
   - If it says "weekend of [date]", answer with that phrasing.
   - Keep qualifiers: "end of", "beginning of", "early", "mid", "late", etc.
3. Use only dates present in the memories. Do not compute or infer dates not explicitly stated.
4. If multiple dates exist, pick the one specifically tied to the event asked about.
5. Answer "unknown" only if no memory provides any date for this event.

Output only the date/time, as short as possible. No preamble.

=== Memories ===
{memories}

=== Question ===
{question}

=== Answer ==="""


# 查询扩展：对问题生成多个不同角度的检索查询，剥离问题框架，直指实体+属性。
QUERY_EXPANSION_PROMPT = """Given a question about two people's chat history, generate 2-4 short keyword search queries to find the answer memory.

Rules:
- Remove question words and negations ("besides", "other than", "what", "which", "how", "did", "does").
- Search for the entity/attribute being asked about, not the question framing.
- For exclusion questions (e.g., "what X besides Y"), target X (not Y).
- For list questions, target different items per query.
- For time questions, include the event and relevant time keywords.
- Each query: 2-5 words. Return ONLY a JSON array of strings.

Question: {question}
Queries:"""


class MemoryAgent:
    """长期记忆 Agent 主控。零参构造，符合 run_generation.py 接口。"""

    DEFAULT_CONFIG = {
        "retrieval_strategy": "hybrid",   # vector | three_factor | hybrid
        "update_mode": "dedup",            # append | dedup
        "top_k": 18,                       # 宽召回：分层 observation/summary/BM25 各取 top_k
        "rerank": True,                    # cross-encoder 精排（宽召回→窄生成）
        "final_k": 18,                      # 精排后最终喂给生成的条数
        "list_final_k": 12,                # 列举题临时调大，召更多同类项
        "query_expansion": True,           # 所有问题都做多角度扩展（A.2），改善单跳/多跳召回
        "bm25_as_separate_recall": True,   # BM25 作为独立召回路径（三路召回）
        "mmr": False,                      # MMR 多样性选择（已关闭）
        "mmr_lambda": 0.65,                # MMR 权衡参数
        "use_cache": True,                 # 抽取只做一次，命中缓存直接 load
        "trace": True,
    }

    def __init__(self, config: dict | None = None):
        self.cfg = {**self.DEFAULT_CONFIG, **(config or {})}
        self.llm = None                    # 延迟到需要抽取时才连 LLM
        self.llm_calls = 0
        self.store = MemoryStore()
        self.retriever = MemoryRetriever(
            self.store, strategy=self.cfg["retrieval_strategy"],
            top_k=self.cfg["top_k"],                       # 宽召回：分层各取 top_k
            rerank=self.cfg["rerank"],
            final_k=self.cfg["final_k"],                   # 窄生成：精排后只留 final_k
            bm25_as_separate_recall=self.cfg["bm25_as_separate_recall"],
            mmr=self.cfg["mmr"],                           # MMR 多样性选择
            mmr_lambda=self.cfg["mmr_lambda"])
        self._key = "conv"
        self._trace_path = None

    def _get_llm(self):
        if self.llm is None:
            from llm_client import LLMClient
            self.llm = LLMClient()
        return self.llm

    def _counted_llm(self):
        outer = self

        class _Counter:
            def generate(self, *a, **k):
                outer.llm_calls += 1
                return outer._get_llm().generate(*a, **k)
        return _Counter()

    # ---- 写入：构建两层记忆（带缓存）----
    def ingest(self, conversation: dict) -> None:
        base_key = conversation.get("sample_id") or conversation_key(conversation)
        # update_mode 改变入库内容，必须进 cache key，避免 append/dedup 缓存串味
        self._key = f"{base_key}.{self.cfg['update_mode']}"
        t0 = time.time()

        if self.cfg["use_cache"] and self.store.has_cache(self._key):
            self.store.load_cache(self._key)
        else:
            print(f"[Controller] ingest 构建记忆 | strategy={self.cfg['retrieval_strategy']} "
                  f"update={self.cfg['update_mode']}")
            writer = MemoryWriter(self._counted_llm())
            updater = MemoryUpdater(self.store, mode=self.cfg["update_mode"])
            # observation 层（零 LLM）+ summary 层（并发 LLM）
            obs = writer.build_observations(conversation)
            summ = writer.build_summaries(conversation)
            # 跨会话合并相似记忆
            #summ = writer.merge_cross_session_memories(summ)
            updater.write_batch(obs)
            updater.write_batch(summ)
            self.store.save_cache(self._key)                  # 抽取只做一次
            self.store.dump_highlevel_md(self._key, conversation)  # 人读高层记忆
            print(f"[Controller] 记忆构建完成 {len(self.store)} 条 "
                  f"| LLM 调用 {self.llm_calls} | 耗时 {time.time()-t0:.1f}s")

        if self.cfg["trace"]:
            self._init_trace()

    # ---- 回答：检索 + 生成 ----
    def answer(self, question: str) -> str:
        t0 = time.time()
        is_temporal = self._is_temporal(question)
        is_list = self._is_list(question)
        queries = self._expand_query(question)          # A.2：所有问题都做多角度扩展
        # 列举题临时调大 final_k，召更多同类项；其余用默认 final_k
        fk = self.cfg["list_final_k"] if is_list else self.cfg["final_k"]
        # 精排始终用原始问题（而非可能含噪的子查询），保证信噪比
        hits = self.retriever.retrieve(queries, rerank_query=question, final_k=fk)
        mem_block = "\n".join(f"- [{h.date_time}] ({h.kind}) {h.text}" for h in hits) \
            or "(no memories)"
        # D：时间题走专用 prompt（精度优先，保留粒度/限定词，不臆造日期）
        tmpl = TEMPORAL_ANSWER_PROMPT if is_temporal else ANSWER_PROMPT
        prompt = tmpl.format(memories=mem_block, question=question)
        self.llm_calls += 1
        ans = self._get_llm().generate(prompt, max_tokens=160).strip()
        ans = ans.strip().rstrip(".!?").strip()
        self._log_trace(question, hits, prompt, ans, time.time() - t0, queries)
        return ans

    @staticmethod
    def _is_temporal(question: str) -> bool:
        """时间类问题：when / what time / what date / how long，或显含时间词。"""
        q = question.lower().strip()
        if q.startswith(("when ", "what time", "what date", "how long")):
            return True
        return any(w in q for w in (" what year", " which year", " on what day"))

    @staticmethod
    def _is_list(question: str) -> bool:
        """列举/多项问题：需要召全多条同类记忆。"""
        q = question.lower().strip()
        signals = ("what activities", "what books", "what places", "what kinds",
                   "what types", "what sports", "what recipes", "what writings",
                   "what instruments", "what causes", "what tricks", "which cities",
                   "which books", "which places", "list", "all the", "which ones",
                   "what are", "what kind of")
        return any(s in q for s in signals)

    def _expand_query(self, question: str) -> list[str]:
        """把问题改写成多个不同角度的子查询（A.2，对所有问题生效）；
        关闭 / 解析失败时退化为原问题。始终保留原问题，避免改写漂移丢语义。"""
        if not self.cfg["query_expansion"]:
            return [question]
        from memory.writer import _parse_json_array
        self.llm_calls += 1
        raw = self._get_llm().generate(
            QUERY_EXPANSION_PROMPT.format(question=question), max_tokens=120)
        subs = [str(s).strip() for s in _parse_json_array(raw) if str(s).strip()]
        queries = [question] + subs[:4]
        if self.retriever.verbose:
            print(f"[Controller] 查询扩展 -> {len(queries)} 个子查询: {subs[:4]}")
        return queries

    # ---- 追踪日志 ----
    def _init_trace(self):
        d = _PKG_ROOT / "experiments" / "results"
        d.mkdir(parents=True, exist_ok=True)
        tag = f"{self.cfg['retrieval_strategy']}_{self.cfg['update_mode']}"
        self._trace_path = d / f"trace_{tag}_{self._key}.jsonl"
        open(self._trace_path, "w", encoding="utf-8").close()

    def _log_trace(self, question, hits, prompt, ans, latency, queries=None):
        if not self.cfg["trace"] or self._trace_path is None:
            return
        rec = {
            "ts": datetime.now().isoformat(), "question": question,
            "expanded_queries": queries or [question],
            "retrieved": [{"mem_id": h.mem_id, "kind": h.kind, "text": h.text,
                           "date_time": h.date_time, "importance": h.importance} for h in hits],
            "prompt": prompt, "prediction": ans, "latency_sec": round(latency, 3),
        }
        with open(self._trace_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


# =============================================================================
# 零参子类：每个对应一组实验配置，直接喂给 run_generation.py 的 module:ClassName。
# =============================================================================

class FullSystemAgent(MemoryAgent):
    """完整系统：hybrid 检索（向量为主 + 三因子加成）+ dedup。"""
    def __init__(self):
        super().__init__({"retrieval_strategy": "hybrid", "update_mode": "dedup"})


# ---- 探索方向 A：三种检索策略（更新固定 dedup）----
class AblA_VectorAgent(MemoryAgent):
    """A-1：只用向量相似度。"""
    def __init__(self):
        super().__init__({"retrieval_strategy": "vector", "update_mode": "dedup"})


class AblA_ThreeFactorAgent(MemoryAgent):
    """A-2：只用三因子打分。"""
    def __init__(self):
        super().__init__({"retrieval_strategy": "three_factor", "update_mode": "dedup"})


class AblA_HybridAgent(MemoryAgent):
    """A-3：向量 + 三因子（向量权重更高）。"""
    def __init__(self):
        super().__init__({"retrieval_strategy": "hybrid", "update_mode": "dedup"})


# ---- 探索方向 C：更新机制（检索固定 hybrid）----
class AblC_AppendAgent(MemoryAgent):
    """C-对照：只追加，不去重。"""
    def __init__(self):
        super().__init__({"retrieval_strategy": "hybrid", "update_mode": "append"})


class AblC_DedupAgent(MemoryAgent):
    """C-实验：近重复去重。"""
    def __init__(self):
        super().__init__({"retrieval_strategy": "hybrid", "update_mode": "dedup"})


# ---- 探索方向：查询扩展消融（其余固定为完整系统配置）----
class AblQ_NoExpansionAgent(MemoryAgent):
    """Q-对照：关闭查询扩展，单查询检索。"""
    def __init__(self):
        super().__init__({"retrieval_strategy": "hybrid", "update_mode": "dedup",
                          "query_expansion": False})


class AblQ_ExpansionAgent(MemoryAgent):
    """Q-实验：开启查询扩展，多子查询检索。"""
    def __init__(self):
        super().__init__({"retrieval_strategy": "hybrid", "update_mode": "dedup",
                          "query_expansion": True})


# ---- 探索方向：精排（reranker）消融 ----
class AblR_NoRerankAgent(MemoryAgent):
    """R-对照：不精排，宽召回结果直接截断喂生成。"""
    def __init__(self):
        super().__init__({"retrieval_strategy": "hybrid", "update_mode": "dedup",
                          "rerank": False})


class AblR_RerankAgent(MemoryAgent):
    """R-实验：cross-encoder 精排后取 final_k。"""
    def __init__(self):
        super().__init__({"retrieval_strategy": "hybrid", "update_mode": "dedup",
                          "rerank": True})
