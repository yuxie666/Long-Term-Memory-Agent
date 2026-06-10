"""
运行学生实现的 MemoryAgent，在评测集上生成预测结果。

使用方式：
    python run_generation.py \
        --eval_set eval_set.json \
        --agent my_agent:MyMemoryAgent \
        --output predictions.json

--agent 参数格式为 "模块路径:类名"。脚本会导入该类并以零参方式实例化，
然后按 ingest() → answer() 的顺序调用。

如果你的 Agent 需要构造参数，可以在模块里写一个工厂函数包装一下，
或者直接修改本脚本中 AgentCls() 处的调用。
"""

import argparse
import importlib
import json
import sys
import time
import traceback
from pathlib import Path


def load_agent_class(spec: str):
    """解析 'module.path:ClassName' 格式并动态导入类。"""
    if ":" not in spec:
        raise ValueError(f"agent 参数必须是 'module:ClassName' 格式，当前为 {spec}")
    mod_path, cls_name = spec.split(":", 1)
    # 把当前目录加入 import path，这样学生可以直接指向本地模块
    if "" not in sys.path:
        sys.path.insert(0, "")
    mod = importlib.import_module(mod_path)
    return getattr(mod, cls_name)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_set", required=True,
                        help="由 prepare_eval_set.py 生成的评测集路径")
    parser.add_argument("--agent", required=True,
                        help="你的 Agent 类，格式：module:ClassName")
    parser.add_argument("--output", default="predictions.json",
                        help="预测结果输出路径")
    parser.add_argument("--limit_conversations", type=int, default=None,
                        help="只跑前 N 段对话（用于快速调试）")
    parser.add_argument("--resume", action="store_true",
                        help="跳过 --output 中已经出现过的 qa_id（断点续跑）")
    args = parser.parse_args()

    with open(args.eval_set, encoding="utf-8") as f:
        eval_set = json.load(f)
    if args.limit_conversations:
        eval_set = eval_set[:args.limit_conversations]

    # 加载已有预测（用于 --resume）
    done_ids = set()
    predictions = []
    out_path = Path(args.output)
    if args.resume and out_path.exists():
        with open(out_path, encoding="utf-8") as f:
            predictions = json.load(f)
        done_ids = {p["qa_id"] for p in predictions}
        print(f"[续跑] 已加载 {len(done_ids)} 条现有预测")

    AgentCls = load_agent_class(args.agent)

    total_convs = len(eval_set)
    total_qas = sum(len(s["qa_list"]) for s in eval_set)
    print(f"[初始化] 共 {total_convs} 段对话，{total_qas} 题，Agent={args.agent}")

    qa_done = 0
    for i, sample in enumerate(eval_set):
        sample_id = sample["sample_id"]
        # 过滤掉本对话中已经有预测结果的 QA
        remaining_qas = [qa for qa in sample["qa_list"] if qa["qa_id"] not in done_ids]
        if not remaining_qas:
            qa_done += len(sample["qa_list"])
            continue

        print(f"[{i+1}/{total_convs}] {sample_id}：正在 ingest "
              f"{len(sample['conversation']['sessions'])} 个 session ...")
        t0 = time.time()
        try:
            # 每段对话都 new 一个新的 Agent 实例，状态隔离
            agent = AgentCls()
            agent.ingest(sample["conversation"])
        except Exception as e:
            print(f"  [错误] ingest 失败：{e}")
            traceback.print_exc()
            # ingest 失败则把本对话所有 QA 都标记为失败
            for qa in remaining_qas:
                predictions.append({
                    "qa_id": qa["qa_id"],
                    "question": qa["question"],
                    "reference": qa["answer"],
                    "category": qa["category"],
                    "category_name": qa["category_name"],
                    "prediction": "",
                    "error": f"ingest_failed: {e}",
                    "latency_sec": 0.0,
                })
                qa_done += 1
            _save(out_path, predictions)
            continue
        ingest_time = time.time() - t0

        # 对当前对话的每道题依次调用 answer()
        for qa in remaining_qas:
            t1 = time.time()
            try:
                pred = agent.answer(qa["question"])
                err = None
            except Exception as e:
                pred = ""
                err = f"answer_failed: {e}"
                traceback.print_exc()
            latency = time.time() - t1
            predictions.append({
                "qa_id": qa["qa_id"],
                "question": qa["question"],
                "reference": qa["answer"],
                "category": qa["category"],
                "category_name": qa["category_name"],
                "prediction": str(pred).strip(),
                "error": err,
                "latency_sec": round(latency, 3),
            })
            qa_done += 1
        # 每处理完一段对话就落盘一次，防止中途挂掉全丢
        _save(out_path, predictions)
        print(f"  ingest 耗时 {ingest_time:.1f}s，"
              f"回答了 {len(remaining_qas)} 题，进度 {qa_done}/{total_qas}")

    print(f"[完成] 共保存 {len(predictions)} 条预测 -> {out_path}")


def _save(path: Path, preds: list):
    """把预测结果落盘。每段对话结束都会调用一次，保证崩溃可恢复。"""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(preds, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
