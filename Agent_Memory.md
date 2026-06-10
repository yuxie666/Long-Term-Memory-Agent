
# 长期记忆对话 Agent (Long-Term Memory Agent)

## 一、任务背景

大语言模型 (LLM) 本身是"无状态"的——每次对话结束后，它并不会记住你是谁、说过什么。如何让 Agent 在跨会话、跨时间的交互中"记住"用户信息，并在后续对话中合理地回忆、利用和更新这些记忆，是近两年非常活跃的研究方向。代表性工作包括 Generative Agents、MemoryBank、Mem0、A-MEM 等。

本次作业要求你们实现一个支持长期记忆的对话 Agent：从历史多轮对话中自动提取并管理记忆，在回答新问题时检索并利用相关记忆，并在长期对话评测集上测量效果。

## 二、任务说明

### 2.1 核心目标

- 从多会话对话中提取并结构化存储记忆
- 回答新问题时检索相关记忆辅助生成
- 实现基本的记忆管理机制（更新 / 冲突 / 遗忘，三选一即可）
- 在评测集上跑通并得到可量化的结果

### 2.2 必读文献（精选 4 篇）

|论文|关注点|
|---|---|
|**Generative Agents: Interactive Simulacra of Human Behavior** (Park et al., UIST 2023)|Recency × Importance × Relevance 三因子检索、Reflection|
|**MemoryBank: Enhancing Large Language Models with Long-Term Memory** (Zhong et al., AAAI 2024)|基于 Ebbinghaus 遗忘曲线的记忆动力学|
|**Mem0: Building Production-Ready AI Agents with Scalable Long-Term Memory** (Chhikara et al., ECAI 2025)|记忆提取 → 冲突检测 → 更新的工程化流程|
|**Evaluating Very Long-Term Conversational Memory of LLM Agents** (Maharana et al., ACL 2024)|长期对话记忆评测基准（本次评测数据来源）|

开题报告中需简要说明：你的方案参考/改进了哪篇工作的哪些思路。

## 三、模型与资源约定

### 3.1 推荐模型（开源、单卡 3070 8G 可跑）

| 模块              | 推荐选择                      | 显存占用    | 说明         |
| --------------- | ------------------------- | ------- | ---------- |
| 主 LLM（默认）       | `Qwen2.5-3B-Instruct-AWQ` | ~3–4 GB | 8G 显存可流畅运行，AWQ 量化版 |
| Embedding       | `BAAI/bge-small-zh-v1.5`  | ~100 MB | 本任务足够      |
| Embedding（可选加强） | `BAAI/bge-m3`             | ~2 GB   | 多语言、更强     |

> **注意**：8G 3070 无法流畅运行 Qwen2.5-7B（即使 AWQ 量化也会在长上下文时 OOM）。3B-AWQ 是本任务的推荐选择，质量足够完成记忆任务。

**所有对照实验必须使用同一个底座模型**，变化只发生在记忆模块。

### 3.2 部署方式

- 推荐 `vLLM` 起 OpenAI 兼容服务，便于切换模型：
    
    ```bash
    vllm serve Qwen/Qwen2.5-3B-Instruct-AWQ --max-model-len 8192
    ```
    
- 也可用 `transformers` 直接加载或 `Ollama`。
- 上下文长度建议设到 8K 即可，不需要拉到 32K 以上。

### 3.3 LLM-as-Judge 评测（云端 API）

Judge 评测对模型能力要求较高，推荐使用**云端 API**（无需本地 GPU）：

| 平台 | 模型推荐 | 费用 | 说明 |
|------|---------|------|------|
| **阿里云 DashScope**（推荐） | `qwen3-32b` 或 `qwen-plus` | 新用户有免费额度，通常够一次完整评测 | 注册即送 tokens，国内访问快 |
| **DeepSeek** | `deepseek-v4-flash` | 极低（约 ¥0.5/百万 tokens） | 性价比最高，API 兼容 OpenAI 格式 |

使用方式：设置环境变量指向云端 API 即可，无需启动本地 vLLM：

```bash
# 阿里云 DashScope
export LLM_BASE_URL="https://dashscope.aliyuncs.com/compatible-mode/v1"
export LLM_API_KEY="sk-xxxxxxxx"  # 在阿里云控制台获取
export LLM_MODEL="qwen-plus"

# 或者 DeepSeek
export LLM_BASE_URL="https://api.deepseek.com/v1"
export LLM_API_KEY="sk-xxxxxxxx"  # 在 DeepSeek 平台获取
export LLM_MODEL="deepseek-v4-flash"
```

> **推荐**：生成阶段用本地 3B-AWQ 模型，Judge 阶段用 DeepSeek V4 Flash（便宜 + 避免 self-evaluation bias）。
>
> ⚠️ **安全提醒**：请勿将 API Key 写入代码或提交到 git。建议将 key 放在 `.env` 文件中（已加入 `.gitignore`），或 `source env.sh` 手动加载。多人共享代码时，每人使用自己的 API Key，避免费用纠纷。

### 3.4 显卡使用规范

- **先用小样本（5–10 条）调通整条流水线**，再上全集评测，避免长时间占卡调 bug。
- 8G 显存下 vLLM 建议设置 `--gpu-memory-utilization 0.75`，预留空间给 embedding 模型。
- 若本地调试不便，也允许调用 SiliconFlow / DashScope 等平台的开源模型 API，**需要在报告中注明 API 费用与配置**。

## 四、系统设计要求

### 4.1 核心模块

```
长期记忆 Agent
├── Memory Store       记忆存储与索引（FAISS / Chroma）
├── Memory Writer      从对话中提取记忆（LLM prompt）
├── Memory Retriever   给定 query 检索相关记忆
├── Memory Updater     冲突 / 遗忘 / 去重（本次实现一种即可）
└── Agent Controller   主流程编排：写入 → 检索 → 生成
```

### 4.2 必做实现

1. 搭建可端到端运行的基线 Agent：输入多会话历史 + 新问题 → 输出答案。
2. **明确区分"原始对话日志"和"派生的记忆单元"**，不要直接把原始对话扔进向量库当记忆。
3. 所有记忆操作可追踪：每次问答时记录检索到的记忆内容、完整 prompt 和模型输出，便于事后分析。

### 4.3 探索方向（必选 1 个，选 2 个有加分）

|方向|说明|参考|
|---|---|---|
|A. 检索策略|对比稠密检索 vs. 三因子打分 (recency × importance × relevance)|Generative Agents|
|B. 记忆遗忘|对比无遗忘 vs. 时间衰减 / 遗忘曲线|MemoryBank|
|C. 记忆更新|对比只追加 vs. 冲突检测后更新/合并|Mem0|
|D. 写入粒度|对比每轮写入 vs. 会话结束批量提取 vs. 事件触发式写入|—|
|E. 反思机制|加入周期性 reflection 生成高阶记忆|Generative Agents|

每个探索方向必须做**对照消融**：相同数据、相同模型，只改这一个变量。不允许只报告"最终方案"而没有消融对比。

## 五、实验要求

### 5.1 评测数据

助教将提供一个基于 LoCoMo 抽取并简化的评测集（约 50 对话 / 200 问题），问题类型涵盖：

- 单会话回忆
- 跨会话回忆
- 时间推理
- 信息更新追踪

（助教会同时提供评测脚本，基于 LLM-as-Judge 打分，不需要自己实现指标计算。）
[下载工具包](https://cdn.jsdelivr.net/gh/zifengcheng/NJU_NLP_Project_2026@main/agent_momery/eval_kit.zip)

### 5.2 必做实验

1. **基线对照**：
    
    - _No-memory_：只把当前 query 喂给 LLM
    - _Full-context_：塞入全部对话历史（截断到模型上限）
    - _Vanilla RAG_：直接对原始对话切片做向量检索
    - _你的系统_
2. **探索方向消融**：针对所选方向做对照（至少 2 组配置）。
    
3. **按问题类型拆解**：分别报告上述 4 类问题的准确率，不允许只给一个总数。
    
4. **Bad case 分析**：从失败样例中挑 3–5 条，定位到底是写入丢失 / 检索未命中 / 生成未利用哪一环出了问题。
    

### 5.3 指标

- **QA 准确率**（主指标，助教脚本计算）
- **平均每题 LLM 调用次数**（成本指标）
- **平均端到端响应时间**

### 5.4 参考性能

| 指标        | 合格            | 良好                  |
| --------- | ------------- | ------------------- |
| 总体 QA 准确率 | ≥ Vanilla RAG | ≥ Vanilla RAG + 5pp |
| 系统稳定性     | 全集评测无崩溃       | —                   |

## 六、提交要求

```
memory_agent/
├── memory/
│   ├── store.py           # 存储与索引
│   ├── writer.py          # 提取与写入
│   ├── retriever.py       # 检索策略
│   └── updater.py         # 更新/遗忘/去重
├── agent/
│   └── controller.py      # 主流程编排
├── eval/
│   └── run_eval.py        # 评测入口
├── experiments/
│   └── results/           # 各组实验结果 JSON / 图表
├── README.md              # 环境与运行说明
├── report.pdf             # 报告（≤ 8 页）
```

**报告内容**

1. 方案概述与对文献的定位（1 页）
2. 系统架构图与模块说明
3. 实验设置、结果表、消融对比
4. Bad case 分析
5. 遇到的问题与反思

## 七、评分标准

|维度|说明|占比|
|---|---|---|
|功能完整性|基线能跑通、代码结构清晰、日志完备|25%|
|**探索深度**|所选方向动机明确、实现到位、消融干净|**30%**|
|实验严谨性|对照清洁、指标分类汇报、bad case 分析|25%|
|报告与展示|逻辑清晰、图表规范、demo 可理解|20%|

## 八、其他建议

- **调试路径**：先用 5–10 条样本打通"输入对话 → 写入记忆 → 检索 → 生成 → 评测"整条链路，再扩到全集。
- **不推荐**直接使用 LangChain 的 `ConversationBufferMemory` / `VectorStoreMemory` 完成作业——这会屏蔽你应该理解并设计的细节，严重影响"探索深度"评分。允许使用 LangChain / LlamaIndex 作为辅助组件（如 text splitter），但**核心记忆逻辑必须自己实现**。
- **Prompt 工程**建议保留在代码中而不是写死在字符串里，方便消融时替换。
- 鼓励在开题阶段就把 4 个 baseline 的骨架搭好，这样后续做探索时对照组是"免费"的。

#### 如有疑问，请联系助教: qqf@smail.nju.edu.cn
