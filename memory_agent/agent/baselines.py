"""
基线 Agent：No-memory。
（另外两个基线 Full-context / Vanilla RAG 已由 eval_kit 提供，无需重复实现。）
"""

from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "eval_kit"))


class NoMemoryAgent:
    """只把当前 query 喂给 LLM，完全不看对话历史——准确率下界基线。"""

    def __init__(self):
        from llm_client import LLMClient
        self.llm = LLMClient()

    def ingest(self, conversation: dict) -> None:
        # 故意什么都不存
        print("[NoMemory] ingest 跳过（不保存任何历史）")

    def answer(self, question: str) -> str:
        prompt = ("Answer the question in a short phrase or one sentence. "
                  "If you cannot know the answer, reply 'unknown'.\n\n"
                  f"Question: {question}\nAnswer:")
        return self.llm.generate(prompt, max_tokens=64).strip()
