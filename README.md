# 长期记忆对话 Agent (Long-Term Memory Agent)

从多会话历史中**抽取并结构化存储记忆**，回答新问题时**检索相关记忆辅助生成**，
并实现**冲突检测 + 更新/去重**的记忆管理机制。在 LoCoMo 评测集上端到端跑通。

## 系统设计

- **两层记忆（对标主流 Agent）** 参考 *Generative Agents* 的 memory-stream + *Mem0* 的 facts：
  - **observation 层**：每条对话轮次直接转成细粒度记忆（无 LLM）—— 保留时间/数值/单次事件等细节，保证召回。
  - **summary 层**：把整个 session 原始文本拼起来，一次 LLM 调用总结出高层记忆（事实 + 反思）。
- **检索策略**  实现三路召回+二次重试机制，分别基于observation，summary,BM25关键词三路召回，并使用rerank精排，对于回答为unknow的问题，使用重试机制（三路检索信息太杂导致模型保守回答unknow），只基于当前session的记忆层进行语义检索。
-  **检索（探索方向 A）** 参考 *Generative Agents* 的三因子，提供 3 种可切换策略。
- **更新（探索方向 C）** 参考 *Mem0* 的更新，改为纯 embedding 近重复检测（无 LLM 热路径）。

## 速度与持久化

- **抽取只做一次**：ingest 构建完成后写 cache（`*.cache.json` + `*.npy`），重跑直接 load，跳过抽取。
- **每 session 仅 1 次 LLM 调用 + 并发**（线程池），observation 层零 LLM；embedding 批量编码。
- **高层记忆持久化为人读文件** `*.highlevel.md`，可直接查看抽取了哪些长期记忆。
- 缓存与持久化文件都在 `experiments/results/mem_cache/`。

## 三种检索策略（当前项目使用hybird混合检索策略）

| strategy | 打分 |
|---|---|
| `vector` | 只用向量余弦相似度 |
| `three_factor` | relevance × recency × importance，各自 min-max 归一化后等权相加 |
| `hybrid` | 向量为主 + 三因子小幅加成：`cos01 + 0.1·recency + 0.1·importance`（向量权重更高）|

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
  question ─Retriever(vector|three_factor|hybrid)─> top-k ─> prompt ─> LLM ─> 短答案
                                                                          └─> trace_*.jsonl
```

## 环境准备

```bash
pip install -r ../eval_kit/requirements.txt      # openai / numpy / sentence-transformers
```
> embedding 用 `BAAI/bge-small-zh-v1.5`，跑在 CPU 上自动加载，不占显存。
> API Key 放 `.env`或配置系统环境变量，勿提交到 git。


## 实验矩阵（run_eval.py 中的 `--only` 名字）

| name | 说明 |
|---|---|
| `no_memory` / `full_context` / `vanilla_rag` / `full_system` | 4 个必做基线对照 |
| `ablA_vector` / `ablA_threefactor` / `ablA_hybrid` | 方向 A：三种检索策略 |
| `ablC_append` vs `ablC_dedup` | 方向 C：只追加 vs 近重复去重 |

## 运行（以项目根目录下为例）

```bash
# 1) 我们的系统实现（检索策略：hybird混合检索，记忆处理机制：dedup合并去重）,即等同于ablA_hybird
python memory_agent/eval/run_eval.py --eval_set eval_set.json  --only full_system

#2）消融实验
#2.1）探索方向C：检索策略hybird不变，记忆处理机制改为append，只追加
python memory_agent/eval/run_eval.py --eval_set eval_set.json  --only ablC_append 

#2.2) 探索方向A（检索策略：three_factor只基于三因子打分，记忆处理机制：dedup合并去重）
python memory_agent/eval/run_eval.py --eval_set eval_set.json  --only ablA_three_factor

```
predictions 落在 `experiments/results/predictions_<name>.json`，命令跑完会打印对应的
Judge 命令（Judge 用云端 API，见 `eval_kit/README.md` 第 2.3 节）：

```bash
#运行不同的实验会生成不同名称的predictions，以full_system为例
python eval_kit/run_judge.py --predictions experiments/results/predictions_full_system.json \
       --output results_full_system.json --num_workers 4
```


每组只改一个变量，其余固定，保证消融干净。

## 可追踪性 & Bad case 分析

每题的 `[检索到的记忆 / 完整 prompt / 模型输出 / 延迟 / LLM 调用数]` 写入
`experiments/results/trace_<config>_<sample>.jsonl`。失败样例可据此定位是
**写入丢失**（记忆里压根没有）/ **检索未命中**（库里有但没召回）/ **生成未利用**
（召回了但答错）三环中的哪一环。

## 成本指标

控制台每段对话打印 `ingest LLM 调用` 次数，trace 里每题记录 `llm_calls`，
配合 `run_judge.py` 输出的平均延迟即可汇报「每题平均 LLM 调用次数」「端到端响应时间」。
