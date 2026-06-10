# 长期记忆对话 Agent (Long-Term Memory Agent)

从多会话历史中**抽取并结构化存储记忆**，回答新问题时**检索相关记忆辅助生成**，
并实现**冲突检测 + 更新/去重**的记忆管理机制。在 LoCoMo 评测集上端到端跑通。

## 文献定位

- **两层记忆（对标主流 Agent）** 参考 *Generative Agents* 的 memory-stream + *Mem0* 的 facts：
  - **observation 层**：每条对话轮次直接转成细粒度记忆（无 LLM）—— 保留时间/数值/单次事件等细节，保证召回。
  - **summary 层**：把整个 session 原始文本拼起来，一次 LLM 调用总结出高层记忆（事实 + 反思）。
  - 检索时**两层一起打分**（不是只检索高层记忆），这是召回准确的关键。
- **检索（探索方向 A）** 参考 *Generative Agents* 的三因子，提供 3 种可切换策略。
- **更新（探索方向 C）** 参考 *Mem0* 的去重，改为纯 embedding 近重复检测（无 LLM 热路径）。

## 速度与持久化

- **抽取只做一次**：ingest 构建完成后写 cache（`*.cache.json` + `*.npy`），重跑直接 load，跳过抽取。
- **每 session 仅 1 次 LLM 调用 + 并发**（线程池），observation 层零 LLM；embedding 批量编码。
- **高层记忆持久化为人读文件** `*.highlevel.md`，可直接查看抽取了哪些长期记忆。
- 缓存与持久化文件都在 `experiments/results/mem_cache/`。

## 三种检索策略（探索方向 A，可自由切换）

| strategy | 打分 |
|---|---|
| `vector` | 只用向量余弦相似度 |
| `three_factor` | relevance × recency × importance，各自 min-max 归一化后等权相加 |
| `hybrid` | 向量为主 + 三因子小幅加成：`cos01 + 0.15·recency + 0.15·importance`（向量权重更高）|

## 目录结构

```
memory_agent/
├── memory/
│   ├── store.py        # MemoryUnit(两层 kind) + numpy 向量索引 + cache/highlevel 持久化
│   ├── writer.py       # observation(无LLM) + summary(每session 1次LLM，并发)
│   ├── retriever.py    # 三种检索：vector / three_factor / hybrid（方向 A 消融）
│   └── updater.py      # 更新：append vs dedup 纯向量去重（方向 C 消融）
├── agent/
│   ├── controller.py   # 编排 写入→检索→生成 + 缓存 + 追踪日志 + 各实验配置子类
│   └── baselines.py    # No-memory 基线
├── eval/
│   └── run_eval.py     # 一键串起 4 基线 + 消融的生成阶段
└── experiments/results/  # predictions / results / trace_*.jsonl / mem_cache/
```

## 架构与数据流

```
ingest(conversation)  ── 命中 cache？──是──> load_cache（跳过抽取）
                          └─否─> observation(每轮, 无LLM) ┐
                                 summary(每session 1次LLM, 并发) ┘─dedup─> 向量库
                                 └─> save_cache + dump highlevel.md
answer(question)
  question ─Retriever(vector|three_factor|hybrid, 两层一起打分)─> top-k ─> prompt ─> LLM ─> 短答案
                                                                          └─> trace_*.jsonl
```

## 环境准备

```bash
pip install -r ../eval_kit/requirements.txt      # openai / numpy / sentence-transformers

# 生成模型（本地 vLLM，8G 3070 可跑）
vllm serve Qwen/Qwen2.5-3B-Instruct-AWQ --port 8000 --max-model-len 8192 --gpu-memory-utilization 0.75
export LLM_BASE_URL=http://localhost:8000/v1
export LLM_API_KEY=EMPTY
export LLM_MODEL=Qwen/Qwen2.5-3B-Instruct-AWQ
```
> embedding 用 `BAAI/bge-small-zh-v1.5`，跑在 CPU 上自动加载，不占显存。
> API Key 放 `.env`，勿提交到 git。

## 运行（先小后大）

```bash
# 1) 烟雾测试：先用 1 个对话跑通整条链路
python eval/run_eval.py --eval_set ../eval_set_small.json --limit_conversations 1 --only full_system

# 2) 小集跑全部实验（4 基线 + 方向 A/C 消融）
python eval/run_eval.py --eval_set ../eval_set_small.json

# 3) 只跑某几组
python eval/run_eval.py --eval_set ../eval_set_small.json --only ablC_append ablC_conflictupdate
```
predictions 落在 `experiments/results/predictions_<name>.json`，命令跑完会打印对应的
Judge 命令（Judge 用云端 API，见 `eval_kit/README.md` 第 2.3 节）：

```bash
python ../eval_kit/run_judge.py --predictions experiments/results/predictions_full_system.json \
       --output experiments/results/results_full_system.json --num_workers 4
```

## 实验矩阵（run_eval.py 中的 `--only` 名字）

| name | 说明 |
|---|---|
| `no_memory` / `full_context` / `vanilla_rag` / `full_system` | 4 个必做基线对照 |
| `ablA_vector` / `ablA_threefactor` / `ablA_hybrid` | 方向 A：三种检索策略 |
| `ablC_append` vs `ablC_dedup` | 方向 C：只追加 vs 近重复去重 |

每组只改一个变量，其余固定，保证消融干净。

## 可追踪性 & Bad case 分析

每题的 `[检索到的记忆 / 完整 prompt / 模型输出 / 延迟 / LLM 调用数]` 写入
`experiments/results/trace_<config>_<sample>.jsonl`。失败样例可据此定位是
**写入丢失**（记忆里压根没有）/ **检索未命中**（库里有但没召回）/ **生成未利用**
（召回了但答错）三环中的哪一环。

## 成本指标

控制台每段对话打印 `ingest LLM 调用` 次数，trace 里每题记录 `llm_calls`，
配合 `run_judge.py` 输出的平均延迟即可汇报「每题平均 LLM 调用次数」「端到端响应时间」。
