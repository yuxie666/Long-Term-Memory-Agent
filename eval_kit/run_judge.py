"""
基于 LLM-as-Judge 的评测脚本，对 run_generation.py 产生的预测结果打分。

Judge 推荐使用云端 API（DeepSeek V4 或阿里云 DashScope），无需本地 GPU，
避免与生成模型共用 8G 显存。输出分类别准确率、总体得分，以及 F1/EM 辅助指标。

使用方式：
    # 通过环境变量配置 Judge（推荐）：
    export LLM_BASE_URL="https://api.deepseek.com/v1"
    export LLM_API_KEY="sk-xxxxxxxx"
    export LLM_MODEL="deepseek-v4-flash"

    python run_judge.py \
        --predictions predictions.json \
        --output results.json

    # 或命令行指定：
    python run_judge.py \
        --predictions predictions.json \
        --output results.json \
        --judge_base_url https://dashscope.aliyuncs.com/compatible-mode/v1 \
        --judge_model qwen-plus

评分规则：
  CORRECT = 1.0    —— 预测捕捉到了参考答案的关键信息
  PARTIAL = 0.5    —— 方向正确但有偏差（粒度更粗、缺细节等）
  WRONG   = 0.0    —— 缺失、矛盾、幻觉、离题
  最终得分 = mean(label_score)，分类别和总体分别报告。
"""

import argparse
import json
import os
import re
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

from llm_client import LLMClient
from metrics import f1_score, exact_match


# 说明：Judge 的 prompt 保持英文，与 LoCoMo 题目语言一致，避免引入额外偏置。
JUDGE_SYSTEM = (
    "You are a strict but fair grading assistant. You grade whether a predicted answer "
    "to a question captures the same information as the reference answer. "
    "You output a single JSON object, nothing else."
)

JUDGE_PROMPT = """Grade the prediction against the reference.

Question: {question}
Reference answer: {reference}
Predicted answer: {prediction}

Apply this rubric:
- CORRECT: The prediction conveys the same key information as the reference. Paraphrasing, different wording, and additional relevant detail are fine. For dates/numbers, the values must match (format can differ).
- PARTIAL: The prediction is on the right topic and partially overlaps with the reference (e.g., correct category but missing specific item; correct year but wrong month), but is not fully accurate.
- WRONG: The prediction is missing, contradictory, hallucinated, or on the wrong topic.

Special cases:
- If the reference is a short entity/date and the prediction is a long sentence that clearly contains the correct information, label CORRECT.
- "I don't know" / empty prediction is WRONG unless the reference also indicates unanswerable.

Respond with JSON only, no other text:
{{"reasoning": "<one short sentence>", "label": "CORRECT" | "PARTIAL" | "WRONG"}}"""


# 三级标签到分数的映射
LABEL_SCORE = {"CORRECT": 1.0, "PARTIAL": 0.5, "WRONG": 0.0}


def parse_judge_output(text: str) -> dict:
    """从 Judge 的输出中提取 JSON，尽量兼容常见的格式问题（code fence、额外文本、大小写等）。"""
    if not text:
        return {"label": "WRONG", "reasoning": "empty judge output"}
    text = text.strip()
    # 去掉 ```json ... ``` 之类的 code fence
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    # 先尝试直接解析
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        # 退而求其次：抓第一个 {...} 块
        m = re.search(r"\{[\s\S]*\}", text)
        if not m:
            return {"label": "WRONG", "reasoning": f"unparseable: {text[:100]}"}
        try:
            obj = json.loads(m.group(0))
        except json.JSONDecodeError:
            return {"label": "WRONG", "reasoning": f"unparseable: {text[:100]}"}
    label = str(obj.get("label", "")).upper().strip()
    if label not in LABEL_SCORE:
        # 有时候模型会输出小写或带额外修饰，尝试在字符串中寻找合法标签
        for key in LABEL_SCORE:
            if key in label:
                label = key
                break
        else:
            label = "WRONG"
    return {"label": label, "reasoning": str(obj.get("reasoning", ""))[:300]}


def judge_one(client: LLMClient, pred_entry: dict) -> dict:
    """给单条预测打分。生成阶段就已经失败的条目直接判 WRONG。"""
    if pred_entry.get("error"):
        return {**pred_entry,
                "judge_label": "WRONG",
                "judge_reasoning": f"generation_error: {pred_entry['error']}",
                "judge_score": 0.0,
                "f1": 0.0,
                "em": 0.0}
    prompt = JUDGE_PROMPT.format(
        question=pred_entry["question"],
        reference=pred_entry["reference"],
        prediction=pred_entry["prediction"],
    )
    raw = client.generate(prompt, max_tokens=128, system=JUDGE_SYSTEM, temperature=0.0)
    parsed = parse_judge_output(raw)
    return {
        **pred_entry,
        "judge_raw": raw,
        "judge_label": parsed["label"],
        "judge_reasoning": parsed["reasoning"],
        "judge_score": LABEL_SCORE[parsed["label"]],
        "f1": f1_score(pred_entry["prediction"], pred_entry["reference"]),
        "em": exact_match(pred_entry["prediction"], pred_entry["reference"]),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True,
                        help="run_generation.py 生成的预测文件")
    parser.add_argument("--output", default="results.json",
                        help="评测结果输出路径")
    parser.add_argument("--judge_base_url", default=None,
                        help="覆盖 LLM_BASE_URL 环境变量")
    parser.add_argument("--judge_model", default=None,
                        help="覆盖 LLM_MODEL 环境变量")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="并发 Judge 请求数（云端 API 可适当调大，注意速率限制）")
    parser.add_argument("--limit", type=int, default=None,
                        help="只评测前 N 条（用于调试）")
    args = parser.parse_args()

    if args.judge_base_url:
        os.environ["LLM_BASE_URL"] = args.judge_base_url
    if args.judge_model:
        os.environ["LLM_MODEL"] = args.judge_model

    with open(args.predictions, encoding="utf-8") as f:
        preds = json.load(f)
    if args.limit:
        preds = preds[:args.limit]

    client = LLMClient(temperature=0.0)
    print(f"[Judge] 模型={client.model}  接口={client.base_url}  "
          f"并发={args.num_workers}  总数={len(preds)}")

    # 并发评测：Judge 任务彼此独立，可以并行
    graded = [None] * len(preds)
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.num_workers) as ex:
        futures = {ex.submit(judge_one, client, p): i for i, p in enumerate(preds)}
        done = 0
        for fut in as_completed(futures):
            i = futures[fut]
            try:
                graded[i] = fut.result()
            except Exception as e:
                # Judge 调用本身挂掉，兜底给 WRONG 但仍记录 F1/EM
                graded[i] = {
                    **preds[i],
                    "judge_label": "WRONG",
                    "judge_reasoning": f"judge_call_failed: {e}",
                    "judge_score": 0.0,
                    "f1": f1_score(preds[i]["prediction"], preds[i]["reference"]),
                    "em": exact_match(preds[i]["prediction"], preds[i]["reference"]),
                }
            done += 1
            if done % 20 == 0 or done == len(preds):
                elapsed = time.time() - t0
                print(f"  已评测 {done}/{len(preds)} "
                      f"（{elapsed:.1f}s，{elapsed/done:.2f}s/条）")

    # ---------- 聚合指标 ----------
    by_cat = defaultdict(lambda: {"n": 0, "score": 0.0, "f1": 0.0, "em": 0.0,
                                   "correct": 0, "partial": 0, "wrong": 0})
    for g in graded:
        name = g["category_name"]
        by_cat[name]["n"] += 1
        by_cat[name]["score"] += g["judge_score"]
        by_cat[name]["f1"] += g["f1"]
        by_cat[name]["em"] += g["em"]
        label_key = g["judge_label"].lower()
        if label_key in by_cat[name]:
            by_cat[name][label_key] += 1

    # 取均值
    for name, d in by_cat.items():
        d["score"] = round(d["score"] / d["n"], 4)
        d["f1"] = round(d["f1"] / d["n"], 4)
        d["em"] = round(d["em"] / d["n"], 4)

    overall_n = sum(d["n"] for d in by_cat.values())
    overall = {
        "n": overall_n,
        "score": round(sum(g["judge_score"] for g in graded) / overall_n, 4),
        "f1": round(sum(g["f1"] for g in graded) / overall_n, 4),
        "em": round(sum(g["em"] for g in graded) / overall_n, 4),
        "avg_latency_sec": round(
            sum(g.get("latency_sec", 0) for g in graded) / overall_n, 3),
    }

    results = {
        "overall": overall,
        "by_category": dict(by_cat),
        "judge_model": client.model,
        "predictions_file": args.predictions,
        "graded": graded,
    }
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    # ---------- 打印表格 ----------
    print("\n===== 评测结果 =====")
    print(f"Judge 模型：{client.model}")
    print(f"{'类别':<14}{'题数':>5}{'得分':>9}{'F1':>9}{'EM':>9}"
          f"{'正确':>6}{'部分':>6}{'错误':>6}")
    for name in sorted(by_cat.keys()):
        d = by_cat[name]
        print(f"{name:<14}{d['n']:>5}{d['score']:>9.3f}{d['f1']:>9.3f}"
              f"{d['em']:>9.3f}{d['correct']:>6}{d['partial']:>6}{d['wrong']:>6}")
    print(f"{'总体':<14}{overall['n']:>5}{overall['score']:>9.3f}"
          f"{overall['f1']:>9.3f}{overall['em']:>9.3f}")
    print(f"\n平均回答耗时：{overall['avg_latency_sec']}s")
    print(f"结果已保存 -> {args.output}")


if __name__ == "__main__":
    main()
