"""
Memory Retriever —— 三种可切换检索策略（探索方向 A 的对照核心）。

检索在**两层记忆上一起打分**（observation + summary），这是召回准确的关键：
高层记忆负责抽象/跨会话，细粒度观察负责具体细节（时间、数值、单次事件）。

  strategy="vector"        : 只看向量余弦相似度
  strategy="three_factor"  : Generative Agents 三因子，relevance/recency/importance
                             各自 min-max 归一化到 [0,1] 后等权相加
  strategy="hybrid"        : 向量相似度为主 + 三因子做小幅加成
                             score = w_rel*rel + w_rec*recency + w_imp*importance
                             （rel 权重远高，保证"向量相似度权重更高"）

分层检索：用同一套打分公式，对 observation / summary 两层**各取 top-k**再合并。
这样概括型 summary 不会把细粒度 observation 挤出 top-k（之前全局竞争时
细节层经常一条都进不来，导致需要原文细节的问题答不出）。
"""

from __future__ import annotations

import math
import os
import re
from collections import Counter

import numpy as np

from memory.store import MemoryStore, MemoryUnit


_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> list[str]:
    """极简英文分词：小写 + 取字母数字 token（LoCoMo 为英文，够用）。"""
    return _TOKEN_RE.findall(text.lower())


class _BM25:
    """零依赖 BM25，用于关键词级检索（专有名词/数字/书名等 token 精确匹配）。"""

    def __init__(self, docs_tokens: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.N = len(docs_tokens)
        self.doc_len = np.array([len(d) for d in docs_tokens], dtype=np.float32)
        self.avgdl = float(self.doc_len.mean()) if self.N else 0.0
        self.tf = [Counter(d) for d in docs_tokens]          # 每文档词频
        df = Counter()                                        # 文档频率
        for d in docs_tokens:
            for w in set(d):
                df[w] += 1
        # idf（带平滑），查询时按词查
        self.idf = {w: math.log(1 + (self.N - n + 0.5) / (n + 0.5)) for w, n in df.items()}

    def scores(self, query: str) -> np.ndarray:
        """返回 query 对每个文档的 BM25 分数 (N,)。"""
        out = np.zeros(self.N, dtype=np.float32)
        if self.N == 0:
            return out
        # 文档长度归一化项对所有词相同，预先算好
        norm = self.k1 * (1 - self.b + self.b * self.doc_len / self.avgdl)
        for w in set(_tokenize(query)):
            idf = self.idf.get(w)
            if idf is None:
                continue
            # 向量化：只遍历包含该词的文档
            f = np.array([tf.get(w, 0) for tf in self.tf], dtype=np.float32)
            mask = f > 0
            out[mask] += idf * (f[mask] * (self.k1 + 1)) / (f[mask] + norm[mask])
        return out


class _Reranker:
    """Cross-encoder 精排器（CPU，不占显存）。粗召回后用它对 (query, memory) 对打分，
    比双塔 embedding 更准，能把真正相关的记忆顶到最前、滤掉噪声。
    模型加载失败时降级为 None，调用方自动跳过精排。"""

    _instance = None  # 进程内单例，避免每个 Agent 实例重复加载

    @classmethod
    def get(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance if cls._instance.model is not None else None

    def __init__(self):
        self.model = None
        name = os.getenv("RERANK_MODEL", "BAAI/bge-reranker-base")
        try:
            from sentence_transformers import CrossEncoder
            print(f"[Reranker] 加载 cross-encoder: {name}")
            self.model = CrossEncoder(name)
        except Exception as e:
            print(f"[Reranker] 加载失败，跳过精排（降级为纯召回排序）：{e}")

    def rank(self, query: str, texts: list[str]) -> np.ndarray:
        """返回每个 text 对 query 的相关性分数 (N,)。"""
        pairs = [[query, t] for t in texts]
        return np.asarray(self.model.predict(pairs), dtype=np.float32)


class MemoryRetriever:
    def __init__(self, store: MemoryStore, strategy: str = "hybrid",
                 top_k: int = 8, recency_half_life_days: float = 60.0,
                 hybrid_weights=(1.0, 0.2, 0.2), tf_weights=(1.0, 1.0, 1.0),
                 per_layer: bool = True, bm25_weight: float = 0.5,
                 rerank: bool = True, final_k: int = 6, verbose: bool = True):
        assert strategy in ("vector", "three_factor", "hybrid")
        self.store = store
        self.strategy = strategy
        self.top_k = top_k              # 分层时 = 每层取的条数；非分层时 = 全局条数
        self.per_layer = per_layer      # True：observation / summary 各取 top_k 再合并
        self.rerank = rerank            # 是否用 cross-encoder 精排（宽召回→窄生成）
        self.final_k = final_k          # 精排后最终喂给生成的条数（窄，提升信噪比）
        self.half_life = recency_half_life_days
        # hybrid：relevance 权重 1.0，远高于 recency/importance 的 0.2，保证"向量相似度为主"
        self.w_rel, self.w_rec, self.w_imp = hybrid_weights
        self.tf = tf_weights                            # three_factor 三因子权重
        self.bm25_weight = bm25_weight                  # hybrid 下 BM25 关键词分的融合权重
        self.verbose = verbose
        self._bm25 = None
        self._bm25_n = -1                               # 构建时的库大小，用于失效判断

    def _get_bm25(self) -> "_BM25":
        """惰性构建 / 重建 BM25 索引（库大小变化时重建）。"""
        if self._bm25 is None or self._bm25_n != len(self.store):
            self._bm25 = _BM25([_tokenize(u.text) for u in self.store.units])
            self._bm25_n = len(self.store)
        return self._bm25

    def _score(self, query: str) -> np.ndarray:
        """按当前策略，对全部记忆单元算分数 (N,)。"""
        sims = self.store.similarities(query)           # cos ∈ [-1,1]
        if self.strategy == "vector":
            return sims
        recency = self._recency()
        importance = _minmax(np.array([u.importance for u in self.store.units]))
        if self.strategy == "three_factor":
            return (self.tf[0] * _minmax(sims)
                    + self.tf[1] * recency + self.tf[2] * importance)
        # hybrid：向量相关性 + BM25 关键词相关性（混合检索）+ 三因子小幅加成。
        # 向量召回语义近似，BM25 召回 token 精确匹配（书名/车型/专有名词），互补。
        bm25 = _minmax(self._get_bm25().scores(query))
        return (self.w_rel * _minmax(sims) + self.bm25_weight * bm25
                + self.w_rec * recency + self.w_imp * importance)

    def retrieve(self, query, rerank_query: str = None) -> list[MemoryUnit]:
        """检索。query 可以是单个字符串，也可以是多个子查询的列表（查询扩展）。
        多查询时，每条记忆取它在各子查询上的最高分（union 语义），再做分层选择。
        rerank_query：用于精排的查询（默认取 query 的第一个，通常是原始问题）。
        流程：宽召回（分层 top_k×2）→ cross-encoder 精排 → 取 final_k（窄，提升信噪比）。"""
        if len(self.store) == 0:
            return []
        queries = [query] if isinstance(query, str) else list(query)
        # 多查询逐元素取 max：任一子查询命中即算相关，适合"列举型"问题召全
        scores = np.max(np.stack([self._score(q) for q in queries]), axis=0)
        units = self.store.units

        if not self.per_layer:
            # 全局竞争：直接取 top_k（旧行为，留作消融对照）
            idx = list(np.argsort(-scores)[:min(self.top_k, len(scores))])
        else:
            # 分层：observation / summary 各取 top_k，再按分数合并去重
            picked = []
            for kind in ("summary", "observation"):
                kind_idx = [i for i, u in enumerate(units) if u.kind == kind]
                if not kind_idx:
                    continue
                kind_idx = np.array(kind_idx)
                order = kind_idx[np.argsort(-scores[kind_idx])]
                picked.extend(order[:self.top_k].tolist())
            # 合并后按全局分数从高到低排序，作为最终顺序
            idx = sorted(set(picked), key=lambda i: -scores[i])

        hits = [units[i] for i in idx]
        reranked = False

        # 精排：用 cross-encoder 对宽召回结果按原始问题重排，取 final_k（窄生成）
        if self.rerank and len(hits) > self.final_k:
            rk = _Reranker.get()
            if rk is not None:
                rq = rerank_query or queries[0]
                rscores = rk.rank(rq, [h.text for h in hits])
                order = np.argsort(-rscores)[:self.final_k]
                hits = [hits[i] for i in order]
                reranked = True
        if not reranked:
            # 未精排（关闭或模型不可用）：仍按召回分数截断到 final_k，
            # 使精排消融只隔离"是否精排"这一个变量（两组上下文条数一致）。
            hits = hits[:self.final_k]

        if self.verbose:
            kinds = {}
            for h in hits:
                kinds[h.kind] = kinds.get(h.kind, 0) + 1
            mode = f"per_layer top{self.top_k}" if self.per_layer else f"global top{self.top_k}"
            print(f"[Retriever] strategy={self.strategy} {mode} nq={len(queries)} "
                  f"rerank={'on' if reranked else 'off'} -> 最终 {len(hits)} 条 ({kinds})")
        return hits

    def _recency(self) -> np.ndarray:
        units = self.store.units
        times = [u.timestamp for u in units]
        valid = [t for t in times if t is not None]
        if not valid:
            return np.zeros(len(units))
        now = max(valid)
        ages = np.array([(now - t).total_seconds() / 86400.0 if t else 1e6 for t in times])
        return _minmax(0.5 ** (ages / self.half_life))


def _minmax(x: np.ndarray) -> np.ndarray:
    if x.size == 0:
        return x
    lo, hi = float(x.min()), float(x.max())
    if hi - lo < 1e-9:
        return np.ones_like(x)
    return (x - lo) / (hi - lo)
