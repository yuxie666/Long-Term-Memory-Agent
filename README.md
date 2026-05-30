# Long-Term Memory Dialogue Agent

This project implements the required `ingest(conversation)` / `answer(question)` agent interface for the supplied evaluation kit.

## Structure

```text
README.md
Agent_Memory.md
memory_agent/
  memory/
    store.py       # derived memory records and vector index
    writer.py      # raw dialogue -> structured memory units
    retriever.py   # relevance + importance + recency retrieval
    updater.py     # deduplication and soft conflict update
  agent/
    controller.py  # MemoryAgent plus baseline wrappers
  eval/
    run_eval.py    # convenience runner
memory_agent/experiments/results/
eval_kit/
  prepare_eval_set.py
  run_generation.py
  run_judge.py
```

The implementation explicitly separates raw dialogue logs from derived memory units. Raw turns are kept only for traceability; retrieval uses `MemoryRecord` objects such as `speaker said on date: fact`.

## Setup

```bash
pip install -r requirements.txt
```

Start an OpenAI-compatible generation server, for example:

```bash
vllm serve Qwen/Qwen2.5-3B-Instruct-AWQ --port 8000 --max-model-len 8192
```

Optional environment variables:

```bash
set LLM_BASE_URL=http://localhost:8000/v1
set LLM_API_KEY=EMPTY
set LLM_MODEL=Qwen/Qwen2.5-3B-Instruct-AWQ
set EMBED_MODEL=BAAI/bge-small-zh-v1.5
set MEMORY_TOP_K=8
```

If `sentence-transformers` or the embedding model is unavailable, the code falls back to a deterministic hashing embedder so the pipeline can still be debugged.

## Run

Prepare an evaluation set:

```bash
python eval_kit/prepare_eval_set.py --output eval_set.json --per_category 10 --seed 42
```

Run the memory agent:

```bash
python eval_kit/run_generation.py --eval_set eval_set.json --agent memory_agent.agent.controller:MemoryAgent --output memory_agent/experiments/results/predictions_memory.json --limit_conversations 2
```

Or use the convenience wrapper:

```bash
python -m memory_agent.eval.run_eval --eval_set eval_set.json --agent memory --limit_conversations 2
```

Baseline wrappers are also available:

```bash
python -m memory_agent.eval.run_eval --eval_set eval_set.json --agent no_memory
python -m memory_agent.eval.run_eval --eval_set eval_set.json --agent full_context
python -m memory_agent.eval.run_eval --eval_set eval_set.json --agent vanilla_rag
```

## Traces

Each agent instance writes JSONL traces to `memory_agent/experiments/results/traces/`. The traces include ingest statistics, retrieved memories, retrieval scores, the full prompt, and the model output for every answer.

## Method

The main system follows three ideas from the long-term memory literature:

- Generative Agents: retrieval combines relevance, importance, and recency.
- MemoryBank: older memories receive an exponential recency decay.
- Mem0: new memories pass through a simple deduplication/update step before entering the store.

For ablation, compare `MemoryAgent` against the provided `VanillaRAGAgent`, and vary `MEMORY_TOP_K` or the retrieval weights in `MemoryRetriever`.
