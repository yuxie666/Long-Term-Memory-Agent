"""
一个轻量的 OpenAI 兼容 LLM 客户端封装，支持以下后端：
  - vLLM（本地部署）：vllm serve Qwen/Qwen2.5-3B-Instruct-AWQ
  - Ollama（本地）：ollama serve
  - DeepSeek API：https://api.deepseek.com/v1（推荐 Judge 用，极便宜）
  - 阿里云 DashScope：https://dashscope.aliyuncs.com/compatible-mode/v1
  - 任何 OpenAI 兼容 API（SiliconFlow 等）

构造时读取的环境变量（也可以通过构造参数覆盖）：
  LLM_BASE_URL   默认：http://localhost:8000/v1
  LLM_API_KEY    默认："EMPTY"（vLLM 本地服务不需要真实 key）
  LLM_MODEL      默认：Qwen/Qwen2.5-3B-Instruct-AWQ
"""

import os
import time
from typing import Optional

try:
    from openai import OpenAI
except ImportError:
    raise ImportError("请先执行 pip install openai>=1.0")


class LLMClient:
    def __init__(
        self,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        temperature: float = 0.0,
        max_retries: int = 3,
    ):
        # 优先使用显式传入的参数，其次用环境变量，最后用默认值
        self.base_url = base_url or os.getenv("LLM_BASE_URL", "http://localhost:8000/v1")
        self.api_key = api_key or os.getenv("LLM_API_KEY", "EMPTY")
        self.model = model or os.getenv("LLM_MODEL", "Qwen/Qwen2.5-3B-Instruct-AWQ")
        self.temperature = temperature
        self.max_retries = max_retries
        self.client = OpenAI(base_url=self.base_url, api_key=self.api_key)

    def generate(self, prompt: str, max_tokens: int = 256,
                 temperature: Optional[float] = None,
                 system: Optional[str] = None) -> str:
        """单轮对话生成（走 chat/completions 接口）。"""
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        temp = self.temperature if temperature is None else temperature

        # 简单的指数退避重试，防止偶尔的网络抖动
        last_err = None
        for attempt in range(self.max_retries):
            try:
                resp = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=temp,
                    max_tokens=max_tokens,
                )
                return resp.choices[0].message.content or ""
            except Exception as e:
                last_err = e
                if attempt < self.max_retries - 1:
                    time.sleep(2 ** attempt)
        raise RuntimeError(f"LLM 调用重试 {self.max_retries} 次后仍然失败：{last_err}")

    def embed(self, texts, model: Optional[str] = None) -> list:
        """可选：embedding 辅助方法。仅在后端支持 embeddings 接口时可用。"""
        model = model or os.getenv("EMBED_MODEL", "BAAI/bge-small-zh-v1.5")
        if isinstance(texts, str):
            texts = [texts]
        resp = self.client.embeddings.create(model=model, input=texts)
        return [item.embedding for item in resp.data]
