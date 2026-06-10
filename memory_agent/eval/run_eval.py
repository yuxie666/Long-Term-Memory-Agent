"""
评测入口 run_eval.py —— 一键串起「4 基线 + 探索消融」的生成阶段。

它只是对 eval_kit/run_generation.py 的薄封装：把要跑的 (名字, agent spec) 列表
依次调起来，predictions 各自落到 experiments/results/ 下。Judge 阶段仍用
eval_kit/run_judge.py（需要云端 API），本脚本最后会打印每组的 judge 命令。

用法：
    # 先确认 LLM 服务可用（本地 vLLM 或云端 API，见 README）
    python eval/run_eval.py --eval_set ../eval_set_small.json --limit_conversations 1
    python eval/run_eval.py --eval_set ../eval_set_small.json        # 跑全部小集
    python eval/run_eval.py --eval_set ../eval_set_small.json --only full_system
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

PKG_ROOT = Path(__file__).resolve().parents[1]          # memory_agent/
EVAL_KIT = PKG_ROOT.parent / "eval_kit"
RESULTS = PKG_ROOT / "experiments" / "results"

# 实验矩阵：name -> agent spec（module:ClassName，相对 eval_kit 工作目录可导入）
EXPERIMENTS = {
    # ---- 4 个 baseline ----
    "no_memory":     "agent.baselines:NoMemoryAgent",
    "full_context":  "agent_template:FullContextAgent",       # eval_kit 自带
    "vanilla_rag":   "vanilla_rag_agent:VanillaRAGAgent",      # eval_kit 自带
    "full_system":   "agent.controller:FullSystemAgent",
    # ---- 探索方向 A：三种检索策略消融 ----
    "ablA_vector":       "agent.controller:AblA_VectorAgent",
    "ablA_threefactor":  "agent.controller:AblA_ThreeFactorAgent",
    "ablA_hybrid":       "agent.controller:AblA_HybridAgent",
    # ---- 探索方向 C：更新机制消融 ----
    "ablC_append":  "agent.controller:AblC_AppendAgent",
    "ablC_dedup":   "agent.controller:AblC_DedupAgent",
    # ---- 查询扩展消融 ----
    "ablQ_noexpand":  "agent.controller:AblQ_NoExpansionAgent",
    "ablQ_expand":    "agent.controller:AblQ_ExpansionAgent",
    # ---- 精排消融 ----
    "ablR_norerank":  "agent.controller:AblR_NoRerankAgent",
    "ablR_rerank":    "agent.controller:AblR_RerankAgent",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval_set", required=True)
    ap.add_argument("--limit_conversations", type=int, default=None)
    ap.add_argument("--only", nargs="*", default=None,
                    help="只跑指定实验名（默认全跑）")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    RESULTS.mkdir(parents=True, exist_ok=True)
    eval_set = str(Path(args.eval_set).resolve())
    names = args.only or list(EXPERIMENTS.keys())

    # 让子进程能 import 到 memory_agent 包和 eval_kit 里的模块
    import os
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        [str(PKG_ROOT), str(EVAL_KIT), env.get("PYTHONPATH", "")]
    )

    judge_cmds = []
    for name in names:
        spec = EXPERIMENTS[name]
        out = RESULTS / f"predictions_{name}.json"
        print(f"\n{'='*70}\n[run_eval] === {name} ({spec}) ===\n{'='*70}")
        cmd = [sys.executable, str(EVAL_KIT / "run_generation.py"),
               "--eval_set", eval_set, "--agent", spec, "--output", str(out)]
        if args.limit_conversations:
            cmd += ["--limit_conversations", str(args.limit_conversations)]
        if args.resume:
            cmd += ["--resume"]
        # 在 eval_kit 目录下运行，使 llm_client / agent_template 等可直接 import
        subprocess.run(cmd, cwd=str(EVAL_KIT), env=env, check=True)

        res = RESULTS / f"results_{name}.json"
        judge_cmds.append(
            f"python {EVAL_KIT/'run_judge.py'} --predictions {out} --output {res} --num_workers 4"
        )

    print(f"\n{'='*70}\n[run_eval] 生成阶段完成。下一步用云端 API 跑 Judge：\n")
    for c in judge_cmds:
        print("  " + c)


if __name__ == "__main__":
    main()
