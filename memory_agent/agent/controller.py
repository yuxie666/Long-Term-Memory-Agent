from __future__ import annotations

import json
import os
import sys
import time
from importlib import import_module
from pathlib import Path

from memory_agent.memory import MemoryRetriever, MemoryStore, MemoryUpdater, MemoryWriter


ROOT = Path(__file__).resolve().parents[2]
EVAL_KIT = ROOT / "eval_kit"
if str(EVAL_KIT) not in sys.path:
    sys.path.insert(0, str(EVAL_KIT))

LLMClient = import_module("llm_client").LLMClient


class MemoryAgent:
    """End-to-end long-term memory agent used by eval_kit."""

    def __init__(self, top_k: int | None = None, log_dir: str | None = None):
        self.llm = LLMClient()
        self.writer = MemoryWriter(reflection_llm=lambda prompt: self.llm.generate(prompt, max_tokens=512))
        self.store = MemoryStore(auto_load=os.getenv("MEMORY_AUTO_LOAD", "0").lower() in {"1", "true", "yes"})
        self.updater = MemoryUpdater()
        self.retriever = MemoryRetriever(
            self.store,
            top_k=top_k or int(os.getenv("MEMORY_TOP_K", "16")),
            strategy=os.getenv("MEMORY_RETRIEVAL_STRATEGY", "hybrid"),
            recency_weight=float(os.getenv("MEMORY_RECENCY_WEIGHT", "0.15")),
            importance_weight=float(os.getenv("MEMORY_IMPORTANCE_WEIGHT", "0.2")),
            relevance_weight=float(os.getenv("MEMORY_RELEVANCE_WEIGHT", "0.65")),
            vector_weight=float(os.getenv("MEMORY_VECTOR_WEIGHT", "0.7")),
            factor_weight=float(os.getenv("MEMORY_FACTOR_WEIGHT", "0.3")),
            vector_threshold=float(os.getenv("MEMORY_VECTOR_THRESHOLD", "0.18")),
            factor_threshold=float(os.getenv("MEMORY_FACTOR_THRESHOLD", "0.25")),
            hybrid_threshold=float(os.getenv("MEMORY_HYBRID_THRESHOLD", "0.22")),
        )
        self.raw_log: list[str] = []
        default_log_dir = ROOT / "memory_agent" / "experiments" / "results" / "traces"
        self.log_dir = Path(log_dir or os.getenv("MEMORY_LOG_DIR", str(default_log_dir)))
        self.trace_id = f"trace_{int(time.time() * 1000)}_{id(self)}"
        self.ingest_stats: dict = {}
        self.answer_max_tokens = int(os.getenv("MEMORY_ANSWER_MAX_TOKENS", "512"))

    def ingest(self, conversation: dict) -> None:
        low_records, high_records, raw_log = self.writer.extract(conversation)
        self.raw_log = raw_log
        low_stats = self.updater.merge_into_store(self.store, low_records)
        high_stats = self.updater.merge_into_store(self.store, high_records)
        self.store.save()
        self.ingest_stats = {
            "raw_turns": len(raw_log),
            "low_memories": len(low_records),
            "high_memories": len(high_records),
            "stored_low_memories": self.store.count("low"),
            "stored_high_memories": self.store.count("high"),
            "stored_memories": len(self.store),
            "low_update_stats": low_stats,
            "high_update_stats": high_stats,
            "index_dir": str(self.store.persist_dir),
        }
        self._append_trace(
            {
                "event": "ingest",
                **self.ingest_stats,
                "raw_log": raw_log,
                "extracted_low_memories": [record.to_dict() for record in low_records],
                "extracted_high_memories": [record.to_dict() for record in high_records],
                "stored_memories_snapshot": [record.to_dict() for record in self.store.records],
            }
        )

    def answer(self, question: str) -> str:
        retrieved = self.retriever.retrieve(question)
        memory_text = "\n".join(self._format_memory_for_prompt(record, scores) for record, scores in retrieved)
        if not memory_text:
            memory_text = "No relevant memory found."

        prompt = (
            "You are an assistant with long-term memory from a past conversation. "
            "The memory list contains derived memory records, not raw dialogue. "
            "LOW memories are atomic facts or events extracted from dialogue. "
            "HIGH memories are reflective summaries generated from LOW memories and may include supporting evidence. "
            "Use active memories as current facts; superseded memories are historical only. "
            "Answer using only the memories. Keep the answer short "
            "(a phrase or one sentence). For list questions, combine all relevant retrieved facts. "
            "If multiple memories each contain part of the answer, synthesize them into one answer. "
            "For comparison or option questions, choose the option best supported by the memories. "
            "You may use general world knowledge only to map answer options to categories explicitly present "
            "in the memories, such as matching a car model to a classic/muscle/SUV category. "
            "For relationship questions, a cautious evidence-based description is allowed. "
            "If the memories truly do not contain enough evidence, reply 'unknown'.\n\n"
            f"=== Retrieved memories ===\n{memory_text}\n\n"
            f"=== Question ===\n{question}\n\n"
            "=== Answer ==="
        )
        answer = self.llm.generate(prompt, max_tokens=self.answer_max_tokens).strip()
        self._append_trace(
            {
                "event": "answer",
                "question": question,
                "retrieved": [
                    {"memory": record.to_dict(), "scores": scores}
                    for record, scores in retrieved
                ],
                "prompt": prompt,
                "answer": answer,
            }
        )
        return answer

    def _format_memory_for_prompt(self, record, scores: dict) -> str:
        status = record.metadata.get("status", "active")
        memory_type = record.metadata.get("memory_type", record.predicate)
        event_date = record.metadata.get("event_date")
        parts = [
            f"- [{record.memory_level.upper()} {memory_type}; status={status}] {record.text}",
            f"  source: session {record.session_id}, {record.date_time}; score={scores['score']}",
        ]
        if event_date:
            parts.append(f"  normalized_event_date: {event_date}")
        if record.memory_level == "high":
            evidence = self._supporting_low_memories(record)
            if evidence:
                parts.append("  supporting_low_memories:")
                parts.extend(f"    * {item}" for item in evidence)
        return "\n".join(parts)

    def _supporting_low_memories(self, record) -> list[str]:
        source_ids = set(record.metadata.get("source_memory_ids", []))
        if not source_ids:
            return []
        supporting = []
        for candidate in self.store.records:
            if candidate.id in source_ids:
                status = candidate.metadata.get("status", "active")
                if status == "superseded":
                    continue
                supporting.append(f"[status={status}] {candidate.text}")
            if len(supporting) >= 4:
                break
        return supporting

    def _append_trace(self, payload: dict) -> None:
        try:
            self.log_dir.mkdir(parents=True, exist_ok=True)
            path = self.log_dir / f"{self.trace_id}.jsonl"
            with path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(payload, ensure_ascii=False) + "\n")
        except Exception:
            pass


class NoMemoryAgent:
    def __init__(self):
        self.llm = LLMClient()
        self.answer_max_tokens = int(os.getenv("MEMORY_ANSWER_MAX_TOKENS", "512"))

    def ingest(self, conversation: dict) -> None:
        return None

    def answer(self, question: str) -> str:
        prompt = (
            "Answer the question briefly. If the answer cannot be inferred, reply 'unknown'.\n\n"
            f"Question: {question}\nAnswer:"
        )
        return self.llm.generate(prompt, max_tokens=self.answer_max_tokens).strip()


class FullContextAgent:
    def __init__(self, max_turns: int = 500):
        self.llm = LLMClient()
        self.max_turns = max_turns
        self.history_text = ""
        self.answer_max_tokens = int(os.getenv("MEMORY_ANSWER_MAX_TOKENS", "512"))

    def ingest(self, conversation: dict) -> None:
        lines = []
        for sess in conversation.get("sessions", []):
            lines.append(f"[Session {sess.get('session_id')} @ {sess.get('date_time', '')}]")
            for turn in sess.get("turns", []):
                lines.append(f"{turn.get('speaker', '')}: {turn.get('text', '')}")
        if len(lines) > self.max_turns:
            lines = lines[-self.max_turns :]
        self.history_text = "\n".join(lines)

    def answer(self, question: str) -> str:
        prompt = (
            "You are an assistant with access to a long conversation between two people. "
            "Answer the user's question using only information from the conversation. "
            "Keep the answer short. If the conversation does not contain the answer, reply 'unknown'.\n\n"
            f"=== Conversation ===\n{self.history_text}\n\n"
            f"=== Question ===\n{question}\n\n"
            "=== Answer ==="
        )
        return self.llm.generate(prompt, max_tokens=self.answer_max_tokens).strip()


class VanillaRAGAgent:
    """Compatibility wrapper around the provided baseline."""

    def __init__(self):
        ProvidedVanillaRAGAgent = import_module("vanilla_rag_agent").VanillaRAGAgent
        self.impl = ProvidedVanillaRAGAgent()

    def ingest(self, conversation: dict) -> None:
        self.impl.ingest(conversation)

    def answer(self, question: str) -> str:
        return self.impl.answer(question)
