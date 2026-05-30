from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
EVAL_KIT = ROOT / "eval_kit"


AGENTS = {
    "no_memory": "memory_agent.agent.controller:NoMemoryAgent",
    "full_context": "memory_agent.agent.controller:FullContextAgent",
    "vanilla_rag": "memory_agent.agent.controller:VanillaRAGAgent",
    "memory": "memory_agent.agent.controller:MemoryAgent",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_set", required=True)
    parser.add_argument("--agent", choices=AGENTS.keys(), default="memory")
    parser.add_argument("--output", default=None)
    parser.add_argument("--limit_conversations", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    output = args.output or str(
        ROOT / "memory_agent" / "experiments" / "results" / f"predictions_{args.agent}.json"
    )
    cmd = [
        sys.executable,
        str(EVAL_KIT / "run_generation.py"),
        "--eval_set",
        args.eval_set,
        "--agent",
        AGENTS[args.agent],
        "--output",
        output,
    ]
    if args.limit_conversations:
        cmd += ["--limit_conversations", str(args.limit_conversations)]
    if args.resume:
        cmd.append("--resume")
    subprocess.run(cmd, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
