"""
Vanilla RAG 基线：
  - 把原始对话按"每轮一个 chunk"切分
  - 用本地 sentence-transformers 加载 bge-small-zh-v1.5 编码（~100 MB，无需额外服务）
  - 回答问题时按余弦相似度检索 top-k，塞进 prompt

这是一个**故意做得很弱**的基线：
  - 直接把对话轮次当记忆，不做任何抽取/摘要/更新
  - 没有记忆管理机制（没有遗忘、没有去重、没有冲突处理）
你的系统应当在此基础上有明显提升，否则说明记忆模块没有发挥作用。

使用方式：
    # 1. 启动 chat LLM（例如 Qwen2.5-3B-Instruct-AWQ），监听 LLM_BASE_URL
    # 2. 运行（embedding 模型自动加载，无需单独起服务）：
    python run_generation.py --eval_set eval_set.json \
        --agent vanilla_rag_agent:VanillaRAGAgent \
        --output predictions_rag.json
"""

import os
import numpy as np
from llm_client import LLMClient


class VanillaRAGAgent:
    def __init__(self, top_k: int = 5):
        # 生成用的 LLM：从默认环境变量读配置
        self.llm = LLMClient()
        # embedding 用 sentence-transformers 本地加载，不占 GPU 显存（跑在 CPU 上），
        # 避免 8G 卡同时跑 vLLM + embedding 服务导致 OOM
        from sentence_transformers import SentenceTransformer
        embed_model = os.getenv("EMBED_MODEL", "BAAI/bge-small-zh-v1.5")
        self.embed_model = SentenceTransformer(embed_model)
        self.top_k = top_k
        self.chunks: list[str] = []
        self.embeddings: np.ndarray | None = None

    def ingest(self, conversation: dict) -> None:
        # 一个 chunk = 一条对话轮次，前面加上时间和说话人作为上下文标识
        chunks = []
        for sess in conversation["sessions"]:
            for turn in sess["turns"]:
                chunks.append(
                    f"[{sess['date_time']}] {turn['speaker']}: {turn['text']}"
                )
        self.chunks = chunks

        # 批量编码所有 chunk（sentence-transformers 自动处理 batching）
        vecs = self.embed_model.encode(chunks, normalize_embeddings=True)
        self.embeddings = np.array(vecs, dtype=np.float32)

    def _retrieve(self, query: str, k: int) -> list[str]:
        """对 query 编码后做余弦相似度检索，返回 top-k 的原文 chunk。"""
        qvec = self.embed_model.encode([query], normalize_embeddings=True)[0]
        sims = self.embeddings @ qvec.astype(np.float32)
        idx = np.argsort(-sims)[:k]
        return [self.chunks[i] for i in idx]

    def answer(self, question: str) -> str:
        retrieved = self._retrieve(question, self.top_k)
        ctx = "\n".join(retrieved)
        # prompt 保持英文以匹配 LoCoMo 原生语言
        prompt = (
            "You are answering a question about a past conversation. "
            "Use only the retrieved dialogue snippets below. Keep the answer short "
            "(a phrase or one sentence). If the snippets do not contain the answer, "
            "reply 'unknown'.\n\n"
            f"=== Retrieved snippets ===\n{ctx}\n\n"
            f"=== Question ===\n{question}\n\n"
            "=== Answer ==="
        )
        return self.llm.generate(prompt, max_tokens=64).strip()
