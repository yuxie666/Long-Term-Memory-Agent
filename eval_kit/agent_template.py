"""
Agent 接口模板。

学生需要在自己的文件中（例如 agent/my_agent.py）实现一个 MemoryAgent 类。
run_generation.py 会导入这个类，并依次调用 ingest() 和 answer()。

接口约定：
  - __init__(self): 每次构造都是全新的状态，不接受必填参数
  - ingest(self, conversation): 读入一段完整的多会话对话
  - answer(self, question) -> str: 基于已有记忆回答问题

重要：评测集中的每一段对话都会 new 一个新的 Agent 实例，
      不同对话之间状态不共享。你所有的记忆逻辑必须在单个实例内闭环。
"""

from typing import Protocol


class MemoryAgent(Protocol):
    """接口约定。在你自己的类中实现这些方法即可。"""

    def ingest(self, conversation: dict) -> None:
        """读入一段多会话对话。

        参数：
            conversation: dict，包含以下键：
                - speaker_a: str，说话人 A 的名字
                - speaker_b: str，说话人 B 的名字
                - sessions: list，每项为 {session_id, date_time, turns}
                    其中 turns = list of {speaker, dia_id, text}

        你应该在这里把对话加工成自己设计的记忆表示并存下来。
        对于每段对话，本方法只会被调用一次，且在所有 answer() 之前。
        """
        ...

    def answer(self, question: str) -> str:
        """基于已有记忆回答一个问题。

        参数：
            question: str

        返回：
            简短的自然语言答案（str）。请尽量简洁，参考答案通常是短语或单句。
        """
        ...


# -----------------------------------------------------------------------------
# 参考实现：最朴素的 "Full-Context" 基线
# ——把整段对话塞进 prompt，完全不做记忆管理。
# 你可以拿它当作流水线的烟雾测试（sanity check），
# 但不要当成自己的最终方案交上来。
# -----------------------------------------------------------------------------

class FullContextAgent:
    """把整段对话塞进 prompt。只有在上下文装得下时才有效。"""

    def __init__(self, max_turns: int = 500):
        # 这里采用延迟导入，避免仅使用接口声明时就强制加载 llm_client
        from llm_client import LLMClient
        self.llm = LLMClient()
        self.max_turns = max_turns
        self.history_text = ""

    def ingest(self, conversation: dict) -> None:
        lines = []
        for sess in conversation["sessions"]:
            lines.append(f"[Session {sess['session_id']} @ {sess['date_time']}]")
            for turn in sess["turns"]:
                lines.append(f"{turn['speaker']}: {turn['text']}")
        # 对话过长时只保留最近的若干轮
        if len(lines) > self.max_turns:
            lines = lines[-self.max_turns:]
        self.history_text = "\n".join(lines)

    def answer(self, question: str) -> str:
        # 为了不对 Judge 造成语言偏置，prompt 保持英文，与 LoCoMo 原文一致
        prompt = (
            "You are an assistant with access to a long conversation between two people. "
            "Answer the user's question using only information from the conversation. "
            "Keep the answer short (a phrase or one sentence). "
            "If the conversation does not contain the answer, reply 'unknown'.\n\n"
            f"=== Conversation ===\n{self.history_text}\n\n"
            f"=== Question ===\n{question}\n\n"
            "=== Answer ==="
        )
        return self.llm.generate(prompt, max_tokens=64).strip()
