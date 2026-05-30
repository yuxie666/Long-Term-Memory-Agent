# 长期记忆对话 Agent 评测工具包

本工具包提供期末大作业所需的统一评测流程：数据集准备、生成脚本。生成使用本地开源模型（单张 3070 8G 可跑），Judge 评测推荐使用云端 API。

## 目录结构

```
eval_kit/
├── prepare_eval_set.py      # 下载 LoCoMo + 分层抽样 → eval_set.json
├── run_generation.py        # 运行你的 Agent，输出 predictions.json
├── run_judge.py             # LLM-as-Judge 评测 → results.json
├── agent_template.py        # Agent 接口定义 + FullContextAgent 示例
├── vanilla_rag_agent.py     # Vanilla RAG 基线参考实现
├── llm_client.py            # OpenAI 兼容 LLM 客户端
├── metrics.py               # F1 / Exact Match 辅助指标
└── requirements.txt
```

## 依赖

```bash
pip install -r requirements.txt
# 本地部署 LLM 推荐 vLLM:
pip install vllm
```

---

## 完整流程（4 步）

### 第 1 步：准备评测集

```bash
python prepare_eval_set.py \
    --output eval_set.json \
    --per_category 40 \
    --seed 42
```

脚本会自动从 `github.com/snap-research/locomo` 克隆数据（约 2.7 MB），缓存到 `.locomo_cache/`，然后按类别分层抽样：默认每类 40 题，共 4 类（single_hop / temporal / multi_hop / open_domain），总计 160 题。

如果想用更小的数据集快速迭代，可以 `--per_category 10`（总 40 题）。

### 第 2 步：准备模型服务

本任务中，**生成模型**使用本地部署（3B-AWQ，8G 显存可跑），**Judge 评测**推荐使用云端 API（详见下方）。

#### 2.1 下载生成模型权重（只需一次）

vLLM 会在第一次启动时自动从 HuggingFace 下载模型到本地缓存（默认 `~/.cache/huggingface/`），之后直接复用。Qwen2.5-3B-Instruct-AWQ 约 **6 GB**。

**国内网络推荐用 HF 镜像**（最简单）：

```bash
pip install -U huggingface_hub
export HF_ENDPOINT=https://hf-mirror.com        # 把这行加到 ~/.bashrc 更省事
huggingface-cli download Qwen/Qwen2.5-3B-Instruct-AWQ
```

**或者用 ModelScope**（阿里镜像，国内更稳定）：

```bash
pip install modelscope
modelscope download --model Qwen/Qwen2.5-3B-Instruct-AWQ \
    --local_dir ./models/Qwen2.5-3B-Instruct-AWQ
```

用 ModelScope 下载后，启动 vLLM 时改用本地路径：

```bash
vllm serve ./models/Qwen2.5-3B-Instruct-AWQ --served-model-name Qwen/Qwen2.5-3B-Instruct-AWQ ...
```

`--served-model-name` 让 API 对外仍报告 `Qwen/Qwen2.5-3B-Instruct-AWQ`，后续脚本不用改模型名。

**共享机器小提示**：多个同学共用一台机器时，把 `HF_HOME` 指向共享目录可以避免重复下载：

```bash
export HF_HOME=/data/shared/hf_cache
```

#### 2.2 启动本地 vLLM 服务（生成用）

```bash
# 在独立的终端 / tmux 窗口中运行
vllm serve Qwen/Qwen2.5-3B-Instruct-AWQ \
    --port 8000 \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.75
```

服务起来后会在 `http://localhost:8000/v1` 提供 OpenAI 兼容 API。首次启动若没下好权重，vLLM 会自己触发下载（但走的是默认 HF 源，国内可能慢，所以推荐按上面先下好）。

**8G 3070 显存方案参考**：

| 方案 | 显存 | 说明 |
|---|---|---|
| `Qwen/Qwen2.5-3B-Instruct-AWQ` | ~3–4 GB | AWQ 量化，8G 显存流畅运行，**推荐** |
| `Qwen/Qwen2.5-3B-Instruct` | ~6 GB | 原始精度，8G 勉强可跑但余量紧张 |
| 云端 API（见下方） | 0 GB | 网络稳定时最省心 |

**Embedding 模型**（只有 Vanilla RAG 基线和部分探索方向需要）：

`vanilla_rag_agent.py` 已使用 `sentence-transformers` 直接加载 `BAAI/bge-small-zh-v1.5`（~100 MB，CPU 运行），无需额外起 embedding 服务。首次运行会自动从 HuggingFace 下载模型。8G 显存全部留给生成 LLM，不打折扣。如需自定义模型，设置环境变量 `EMBED_MODEL` 即可。

#### 2.3 配置 Judge 云端 API（推荐）

Judge 评测对模型能力要求较高，3B 模型做 Judge 不够可靠。推荐使用**云端 API**，便宜且无需本地 GPU：

| 平台 | 模型推荐 | 费用 | 说明 |
|------|---------|------|------|
| **DeepSeek**（推荐） | `deepseek-v4-flash` | ~¥0.5/百万 tokens | 性价比极高，OpenAI 兼容接口 |
| **阿里云 DashScope** | `qwen-plus` | 新用户免费额度通常够用 | 国内访问快，注册即送 |

使用方式：

```bash
# DeepSeek（推荐，便宜且避免 self-evaluation bias）
export LLM_BASE_URL="https://api.deepseek.com/v1"
export LLM_API_KEY="sk-xxxxxxxx"   # 在 platform.deepseek.com 获取
export LLM_MODEL="deepseek-v4-flash"

# 或者阿里云 DashScope
export LLM_BASE_URL="https://dashscope.aliyuncs.com/compatible-mode/v1"
export LLM_API_KEY="sk-xxxxxxxx"   # 在阿里云控制台获取
export LLM_MODEL="qwen-plus"
```

设置好环境变量后，`run_judge.py` 会自动使用云端 API 进行评测，无需本地 GPU。

> ⚠️ **安全提醒**：请勿将 API Key 写入代码或提交到 git。建议将 key 放在 `.env` 文件中（已加入 `.gitignore`），或使用环境变量脚本 `source env.sh` 在每次终端手动加载。多人共享代码时，每人使用自己的 API Key。

### 第 3 步：运行你的 Agent 生成预测

先跑通 FullContextAgent 验证环境：

```bash
python run_generation.py \
    --eval_set eval_set.json \
    --agent agent_template:FullContextAgent \
    --output predictions_fullctx.json \
    --limit_conversations 2     # 先跑 2 个对话验证
```

然后跑 Vanilla RAG 基线：

```bash
python run_generation.py \
    --eval_set eval_set.json \
    --agent vanilla_rag_agent:VanillaRAGAgent \
    --output predictions_rag.json
```

最后跑你自己的系统：

```bash
# 假设你的实现在 my_agent/controller.py，类名是 MyMemoryAgent
python run_generation.py \
    --eval_set eval_set.json \
    --agent my_agent.controller:MyMemoryAgent \
    --output predictions_mine.json
```

**断点续跑：** 加 `--resume`，已经跑过的 qa_id 会跳过，不怕中途挂掉。

### 第 4 步：LLM-as-Judge 评测

Judge 使用**云端 API**（DeepSeek 或 DashScope），通过环境变量配置（见第 2.3 节）：

```bash
python run_judge.py \
    --predictions predictions_mine.json \
    --output results_mine.json \
    --num_workers 4
```

`run_judge.py` 会从环境变量 `LLM_BASE_URL` / `LLM_MODEL` 读取 Judge 配置。也可以命令行覆盖：

```bash
python run_judge.py \
    --predictions predictions_mine.json \
    --output results_mine.json \
    --judge_base_url https://api.deepseek.com/v1 \
    --judge_model deepseek-v4-flash \
    --num_workers 4
```

输出示例：

```
===== Results =====
Judge: deepseek-v4-flash
Category        N    Score       F1       EM  Corr  Part  Wrng
multi_hop      40    0.387    0.241    0.075    12     7    21
open_domain    40    0.525    0.318    0.125    17     9    14
single_hop     40    0.712    0.456    0.250    25     7     8
temporal       40    0.400    0.198    0.050    14     4    22
OVERALL       160    0.506    0.303    0.125

Avg answer latency: 1.847s
```

---

## Agent 接口约定

你的 `MemoryAgent` 类必须实现：

```python
class MyMemoryAgent:
    def __init__(self):
        # 初始化你的记忆结构
        ...

    def ingest(self, conversation: dict) -> None:
        """读入一段完整多会话对话，构建记忆。
        
        conversation 结构：
        {
          "speaker_a": "Caroline",
          "speaker_b": "Melanie",
          "sessions": [
            {"session_id": 1, "date_time": "1:56 pm on 8 May 2023",
             "turns": [{"speaker": "Caroline", "dia_id": "D1:1", "text": "..."}, ...]},
            ...
          ]
        }
        """

    def answer(self, question: str) -> str:
        """基于已有记忆回答问题，返回一个简短字符串。"""
```

**关键规则：**

1. 每个对话新建一个 Agent 实例，**状态不跨对话共享**。
2. `ingest` 先调用一次，然后多次调用 `answer`。
3. `answer` 返回的字符串会直接和 gold 答案送给 Judge。**保持简短**——参考答案大多是短语或单句。

---

## 评测指标说明

### 主指标：Judge Score（三级评分）

- `CORRECT = 1.0`：预测捕获了参考答案的关键信息（允许释义、附加细节）
- `PARTIAL = 0.5`：方向正确但不完整（比如年份对了月份错了）
- `WRONG = 0.0`：缺失、矛盾、幻觉、离题

总分 = `mean(label_score)`，分类汇总和总体都会报告。

### 辅助指标

- **Token-level F1**：SQuAD 风格的词级 F1 overlap
- **Exact Match (EM)**：归一化后完全相等
- **平均延迟**：每题的 `answer()` 耗时

如果 Judge Score 高但 F1 极低，说明 Judge 可能过宽松，需要人工抽查；反之亦然。**报告中应至少报告 Judge Score 和 F1 两个数字**。

### 已知局限（必须在报告中声明）

如果 Agent 生成和 Judge 评测使用**同一模型族**（如都用 Qwen 系列），存在 **self-evaluation bias**（同族模型倾向给自己高分）。推荐做法：
- 生成用本地 `Qwen2.5-3B-Instruct-AWQ`
- Judge 用 **DeepSeek V4 Flash**（不同模型族，交叉验证更客观）
- 报告中声明生成模型和 Judge 模型的具体配置

如果因条件限制只能用同一模型族做 Judge，需在报告中说明并讨论 bias 影响。

---

## 常见问题

**Q: vLLM 启动 OOM？**
A: 降低 `--max-model-len`（例如 4096），或 `--gpu-memory-utilization 0.65`。8G 3070 建议用 3B-AWQ 版本，7B 即使 AWQ 量化在长上下文下也容易 OOM。

**Q: Judge 返回 "unparseable"？**
A: `run_judge.py` 已经容错处理（code fence / 额外文本 / 大小写）。如果仍大量失败，可能是 `max_tokens=128` 不够——把 `run_judge.py` 里的 `max_tokens` 调到 256。

**Q: 跑完没结果 / 卡住？**
A: 用 `--limit_conversations 1` 先跑一个对话验证流水线，确认问题在哪一步。

**Q: 我想加入 adversarial 类别？**
A: `prepare_eval_set.py --categories 1 2 3 4 5`。但注意：adversarial 题的参考答案在原数据里是 `adversarial_answer`（期望 Agent 识别为无法回答），直接和预测比对逻辑不合理，Judge prompt 需要改写。建议不要碰。

**Q: 我想跑全集？**
A: `--per_category 200`（最小类 multi_hop 只有 96，会自动取 96）。总量约 800 题，单卡跑完 Agent + Judge 大概 2–4 小时。建议平时用 40×4=160 题迭代，最后一次跑大集。

**Q: 能用 Ollama 代替 vLLM 吗？**
A: 可以。启动 `ollama run qwen2.5:3b`，然后 `--judge_base_url http://localhost:11434/v1`。但并发和吞吐不如 vLLM，且 Ollama 的 AWQ 支持有限。

---

## 引用与许可

- 数据集：Maharana et al., "Evaluating Very Long-Term Conversational Memory of LLM Agents", ACL 2024. [github.com/snap-research/locomo](https://github.com/snap-research/locomo)
- 许可证：数据集 CC BY-NC 4.0；本工具包代码 MIT。
