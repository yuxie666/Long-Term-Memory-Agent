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


ANSWER_PROMPT = """Answer a question about two people's past conversations based ONLY on the retrieved memories. Make sure you read through ALL retrieved memories carefully.

INSTRUCTIONS:
1. DIRECT FACTS: If any memory directly states the answer, extract and state that fact.
2. ONE-STEP INFERENCE: If the answer is implied by the memories, infer it. Examples:
   - Memory: "went surfing" → Answer: "likes surfing"
   - Memory: "recommended book X" + Memory: "read recommended book" → Answer: "X"
3. MULTI-HOP REASONING: Combine information from multiple memories:
   - Memory A: "X recommended a book to Y"
   - Memory B: "X's favorite book is Z"
   - Conclusion: "Z"
4. LISTS: For "what" questions, extract EVERY matching item from ALL memories and list them comma-separated.
5. HYPOTHETICAL QUESTIONS ("would", "likely"): Infer from preferences and behavior with a reason.
6. WHEN QUESTIONS: Extract the date/time as stated in the memory.

MANDATORY RULES (MUST FOLLOW):
- YOU MUST extract information from the memories even if it's partial.
- YOU MUST NOT say "unknown" if ANY memory contains relevant information.
- If memories contain partial information, state what you know.
- For "what" questions, list ALL items you find, not just some.
- For "would" questions, use the memories to make a reasoned guess.

Answer "unknown" ONLY when ALL memories are completely irrelevant to the question.

Answer format: A phrase, list, or sentence that answers the question. No preamble.

=== Memories ===
{memories}

=== Question ===
{question}

=== Answer ==="""

# 时间类问题专用 prompt：智能时间推理 + 精度保持。
TEMPORAL_ANSWER_PROMPT = """Answer a TIME question about two people's past conversations. Date/time precision matters most.

INSTRUCTIONS:
1. Find the memory that describes the event in the question.
2. Extract the date/time from that memory.

TIME COMPUTATION RULES:
For examples:
- If the memory contains "[= computed date]" annotations, use those computed dates.
- If the memory says "weekend of 22 August 2022", compute: Saturday-Sunday = 20-21 August 2022 (weekend BEFORE the given date).
- If the memory says "the Friday before 14 August 2023", compute: Friday before = 11 August 2023.
- If the memory says "the week before 27 June 2023", compute: week of 19-25 June 2023.
- If the memory says "a few days before X", estimate: 3-5 days before X.
- If the memory says "end of October 2023", answer: "end of October 2023" (keep granularity).
- If the memory says "first week of October 2023", answer: "first week of October 2023" (keep granularity).
Based on examples above,you should make sure the time/date you compute is completely correct.
GRANULARITY RULES:
- Keep the SAME granularity as the memory when it's intentionally vague (e.g., "around March 2023", "sometime in 2022").
- Convert to specific dates when the memory provides enough information to compute.
- For duration questions ("how long"), answer with duration (e.g., "four months"), NOT a date range.

OUTPUT RULES:
- Output only the date/time answer, as short as possible.
- If the memory contains no date for this event, answer "unknown".
- No preamble, no explanation.

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
        "list_final_k": 20,                # 列举题调大，召更多同类项（从12增加到20）
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
        
        # 二次检索召回：仅在 hybrid 策略下生效
        # 如果模型回答 unknown，基于问题所在 session 的记忆重新召回
        if ans.lower() in ("unknown", "unknown.", "i don't know", "not sure"):
            if self.retriever.strategy == "hybrid":
                ans = self._session_based_secondary_retrieval(question, queries)
        
        self._log_trace(question, hits, prompt, ans, time.time() - t0, queries)
        return ans
    
    def _session_based_secondary_retrieval(self, question: str, queries: list) -> str:
        """基于问题所在 session 的二次检索召回。
        
        如果首次回答是 unknown，则只针对该问题所在的 session 的低层和高层记忆各召回 8 条，
        精排后返回给模型重新生成。
        """
        import numpy as np
        
        # 获取当前问题所在的 session（假设当前会话的所有记忆都属于同一个 session）
        # 这里简化处理：获取所有不同的 session_id，然后针对每个 session 进行召回
        session_ids = set(unit.session_id for unit in self.store.units)
        
        if not session_ids:
            return "unknown"
        
        # 获取问题的向量表示（用于计算语义相似度）
        q_vec = self.store.embed(question)[0]
        
        # 收集所有 session 的记忆（基于语义相似度排序）
        session_memories = []
        for session_id in session_ids:
            # 按 session 过滤记忆
            session_units = [unit for unit in self.store.units if unit.session_id == session_id]
            
            # 分别获取 observation 和 summary 记忆
            obs_units = [u for u in session_units if u.kind == "observation"]
            sum_units = [u for u in session_units if u.kind == "summary"]
            
            # 对 observation 层按语义相似度排序取前 8 条
            if obs_units:
                obs_texts = [u.text for u in obs_units]
                obs_vecs = self.store.embed(obs_texts)
                obs_sims = obs_vecs @ q_vec
                obs_indices = np.argsort(-obs_sims)[:8]
                obs_units_sorted = [obs_units[i] for i in obs_indices]
                session_memories.extend(obs_units_sorted)
            
            # 对 summary 层按语义相似度排序取前 8 条
            if sum_units:
                sum_texts = [u.text for u in sum_units]
                sum_vecs = self.store.embed(sum_texts)
                sum_sims = sum_vecs @ q_vec
                sum_indices = np.argsort(-sum_sims)[:8]
                sum_units_sorted = [sum_units[i] for i in sum_indices]
                session_memories.extend(sum_units_sorted)
        
        # 如果没有 session 记忆，返回 unknown
        if not session_memories:
            return "unknown"
        
        # 对所有收集的 session 记忆再次按语义相似度精排，取前 10 条（增加召回量）
        mem_texts = [m.text for m in session_memories]
        mem_vecs = self.store.embed(mem_texts)
        sims = mem_vecs @ q_vec
        top_indices = np.argsort(-sims)[:8]
        top_memories = [session_memories[i] for i in top_indices]
        
        # 构建记忆块
        mem_block = "\n".join(f"- [{m.date_time}] ({m.kind}) {m.text}" for m in top_memories)
        
        # 重新生成答案
        tmpl = TEMPORAL_ANSWER_PROMPT if self._is_temporal(question) else ANSWER_PROMPT
        prompt = tmpl.format(memories=mem_block, question=question)
        self.llm_calls += 1
        ans = self._get_llm().generate(prompt, max_tokens=160).strip()
        ans = ans.strip().rstrip(".!?").strip()
        
        # 强制提取答案：如果二次检索仍然返回 unknown，尝试从记忆中直接提取
        if ans.lower() in ("unknown", "unknown.", "i don't know", "not sure"):
            ans = self._force_extract_answer(top_memories, question)
        
        return ans
    
    def _force_extract_answer(self, memories: list, question: str) -> str:
        """强制从记忆中提取答案，避免返回 unknown。"""
        import re
        
        q_lower = question.lower()
        
        # 1. 对于列举类问题，提取所有匹配项
        if any(kw in q_lower for kw in ["what", "which", "list", "names", "items"]):
            items = []
            for mem in memories:
                text = mem.text.lower()
                # 提取逗号分隔或 "and" 连接的项目
                if "," in text or " and " in text:
                    parts = text.replace(" and ", ",").split(",")
                    for part in parts:
                        part = part.strip()
                        if len(part) > 3:
                            items.append(part)
            if items:
                return ", ".join(list(set(items)))[:100]
        
        # 2. 对于时间类问题，提取日期
        if any(kw in q_lower for kw in ["when", "date", "time", "year", "month", "day"]):
            for mem in memories:
                # 提取日期模式
                date_patterns = [
                    r'\d{1,2}\s+\w+\s+\d{4}',
                    r'\w+\s+\d{1,2},?\s+\d{4}',
                    r'\d{4}',
                ]
                for pattern in date_patterns:
                    match = re.search(pattern, mem.text)
                    if match:
                        return match.group(0)
        
        # 3. 对于 "would" 类问题，根据记忆做推测
        if any(kw in q_lower for kw in ["would", "likely", "should"]):
            for mem in memories:
                if "likes" in mem.text.lower() or "enjoys" in mem.text.lower():
                    return "Yes"
                if "dislikes" in mem.text.lower() or "hates" in mem.text.lower():
                    return "No"
            return "Maybe"
        
        # 4. 尝试提取专有名词（书名、人名等）
        for mem in memories:
            quoted = re.findall(r'"([^"]+)"', mem.text)
            if quoted:
                return quoted[0]
        
        # 5. 如果实在找不到，返回最相关记忆的内容摘要
        if memories:
            return memories[0].text[:100]
        
        # 最后才返回 unknown
        return "unknown"

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
