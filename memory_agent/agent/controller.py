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


ANSWER_PROMPT = """You are an assistant answering a question about two people's past conversations. Use ONLY the retrieved memories below. Each memory is tagged with its date. Keep the answer short (a phrase or one sentence). For "when" questions, answer with the specific date. For questions that ask to list or enumerate (e.g. "what books / which places"), gather EVERY matching item across all memories. If the memories do not contain the answer, reply 'unknown'.

=== Retrieved memories ===
{memories}

=== Question ===
{question}

=== Answer ==="""


# 查询扩展：仅对"列举/多部分"问题触发，把原问题拆成覆盖不同子项的查询。
# 改进点：要求生成不同角度/不同实体的子查询，而非原问题的同义改写（同义改写只会
# 把更多边缘相关记忆拉进候选池、稀释信噪比，对单一事实型问题有害无益）。
QUERY_REWRITE_PROMPT = """The user's question asks for MULTIPLE items or spans several sub-topics. Break it into 2-4 focused search queries, each targeting a DIFFERENT specific item, entity, or aspect — do NOT just rephrase the whole question with synonyms. Each query is a short keyword phrase.

Example:
Question: "What schools did John play basketball in and how many years was he on the high school team?"
Queries: ["John middle school basketball", "John high school basketball team", "John college basketball", "John years on high school team"]

Return ONLY a JSON array of strings.

Question: {question}
Queries:"""


class MemoryAgent:
    """长期记忆 Agent 主控。零参构造，符合 run_generation.py 接口。"""

    DEFAULT_CONFIG = {
        "retrieval_strategy": "hybrid",   # vector | three_factor | hybrid
        "update_mode": "dedup",            # append | dedup
        "top_k": 8,                        # 宽召回：分层 observation/summary 各取 top_k
        "rerank": True,                    # cross-encoder 精排（宽召回→窄生成）
        "final_k": 6,                      # 精排后最终喂给生成的条数（窄，提升信噪比）
        "query_expansion": True,           # 仅对列举/多部分问题触发（见 _should_expand）
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
            final_k=self.cfg["final_k"])                   # 窄生成：精排后只留 final_k
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
        queries = self._expand_query(question)          # #2：按需查询扩展
        # 精排始终用原始问题（而非可能含噪的子查询），保证信噪比
        hits = self.retriever.retrieve(queries, rerank_query=question)
        mem_block = "\n".join(f"- [{h.date_time}] ({h.kind}) {h.text}" for h in hits) \
            or "(no memories)"
        prompt = ANSWER_PROMPT.format(memories=mem_block, question=question)
        self.llm_calls += 1
        ans = self._get_llm().generate(prompt, max_tokens=64).strip()
        self._log_trace(question, hits, prompt, ans, time.time() - t0, queries)
        return ans

    @staticmethod
    def _should_expand(question: str) -> bool:
        """只对'列举/多部分'问题做扩展。单一事实型（when/who/how long 等）不扩展——
        它们只需一条精确记忆，扩大召回只会引入噪声、拖累小模型生成。"""
        q = question.lower().strip()
        # 单一事实型问题的典型开头：不扩展
        single = ("when ", "what time", "what date", "how long", "how old",
                  "how many", "where ", "who is", "who was", "which ")
        if q.startswith(single):
            return False
        # 列举/多部分信号：扩展
        multi = (" and ", "list", "what books", "what places", "what kinds",
                 "what types", "which ones", "all the")
        return any(m in q for m in multi)

    def _expand_query(self, question: str) -> list[str]:
        """按需把问题改写成多个子查询；不触发 / 关闭 / 解析失败时退化为原问题。"""
        if not self.cfg["query_expansion"] or not self._should_expand(question):
            return [question]
        from memory.writer import _parse_json_array
        self.llm_calls += 1
        raw = self._get_llm().generate(
            QUERY_REWRITE_PROMPT.format(question=question), max_tokens=120)
        subs = [str(s).strip() for s in _parse_json_array(raw) if str(s).strip()]
        # 始终保留原问题，避免改写漂移丢掉关键语义
        queries = [question] + subs[:4]
        if self.retriever.verbose:
            print(f"[Controller] 查询扩展(列举型) -> {len(queries)} 个子查询: {subs[:4]}")
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
