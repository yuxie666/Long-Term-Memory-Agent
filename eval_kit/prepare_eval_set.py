"""
评测集准备脚本：下载 LoCoMo 数据集并按类别分层抽样。

使用方式：
    python prepare_eval_set.py \
        --output eval_set.json \
        --per_category 40 \
        --seed 42

输出 JSON 文件结构：
    [
      {
        "sample_id": "conv_0",
        "conversation": {
            "speaker_a": "Caroline",
            "speaker_b": "Melanie",
            "sessions": [
                {"session_id": 1, "date_time": "...", "turns": [{"speaker": "...", "text": "..."}, ...]},
                ...
            ]
        },
        "qa_list": [
            {"qa_id": "conv_0_q3", "question": "...", "answer": "...",
             "category": 1, "category_name": "single_hop"},
            ...
        ]
      },
      ...
    ]

注意：LoCoMo 遵循 CC BY-NC 4.0 许可，仅限研究/教学使用，不得重分发。
来源：https://github.com/snap-research/locomo
"""

import argparse
import json
import random
import subprocess
import tempfile
from collections import defaultdict
from pathlib import Path

# 类别 ID → 可读名称
# 通过抽样检查 locomo10.json 中每个类别的示例 QA 人工核对得到。
# 注意：论文正文中列出的顺序与 JSON 文件中使用的 ID 顺序并不一致，这里以 JSON 为准。
CATEGORY_NAMES = {
    1: "single_hop",     # 单会话事实问答
    2: "temporal",       # 时间推理（When/日期类问题）
    3: "multi_hop",      # 跨证据多跳推理
    4: "open_domain",    # 开放域/常识
    5: "adversarial",    # 对抗性问题（期望识别为无法回答）
}

LOCOMO_REPO = "https://github.com/snap-research/locomo.git"
LOCOMO_DATA_PATH = "data/locomo10.json"


def download_locomo(cache_dir: Path) -> Path:
    """浅克隆 LoCoMo 仓库并返回 locomo10.json 的路径。"""
    data_file = cache_dir / "locomo10.json"
    if data_file.exists():
        print(f"[缓存] 使用已缓存的文件：{data_file}")
        return data_file

    print(f"[下载] 正在克隆 {LOCOMO_REPO} ...")
    with tempfile.TemporaryDirectory() as tmp:
        subprocess.run(
            ["git", "clone", "--depth", "1", LOCOMO_REPO, tmp],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        src = Path(tmp) / LOCOMO_DATA_PATH
        if not src.exists():
            raise FileNotFoundError(f"克隆的仓库中找不到 {LOCOMO_DATA_PATH}")
        cache_dir.mkdir(parents=True, exist_ok=True)
        data_file.write_bytes(src.read_bytes())
    print(f"[下载] 已保存到 {data_file}")
    return data_file


def flatten_conversation(raw_conv: dict) -> dict:
    """把 LoCoMo 原始的嵌套 session dict 展平成有序的 session 列表。"""
    sessions = []
    # session 键形如 'session_1', 'session_2'...，对应的时间戳是 'session_N_date_time'
    session_ids = sorted(
        [int(k.split("_")[1]) for k in raw_conv.keys()
         if k.startswith("session_") and not k.endswith("_date_time")]
    )
    for sid in session_ids:
        turns = raw_conv[f"session_{sid}"]
        date_time = raw_conv.get(f"session_{sid}_date_time", "")
        sessions.append({
            "session_id": sid,
            "date_time": date_time,
            "turns": [
                {"speaker": t["speaker"], "dia_id": t.get("dia_id", ""), "text": t["text"]}
                for t in turns
            ],
        })
    return {
        "speaker_a": raw_conv.get("speaker_a", ""),
        "speaker_b": raw_conv.get("speaker_b", ""),
        "sessions": sessions,
    }


def build_eval_set(
    raw_data: list,
    per_category: int,
    include_categories: list,
    seed: int,
) -> list:
    """对每个指定类别随机抽样 per_category 条问题，再按对话分组组织输出。"""
    rng = random.Random(seed)

    # 先收集所有符合条件的 QA，记录它们所属的对话下标
    qa_by_category = defaultdict(list)  # cat -> [(conv_idx, qa_idx, qa_dict)]
    for conv_idx, conv in enumerate(raw_data):
        for qa_idx, qa in enumerate(conv["qa"]):
            cat = qa.get("category")
            if cat in include_categories:
                qa_by_category[cat].append((conv_idx, qa_idx, qa))

    # 分层抽样
    print("[抽样] 各类别可用题数：")
    for cat in sorted(qa_by_category.keys()):
        print(f"  类别 {cat} ({CATEGORY_NAMES[cat]})：{len(qa_by_category[cat])} 题")

    sampled_keys = set()  # (conv_idx, qa_idx) 的集合
    for cat, pool in qa_by_category.items():
        n = min(per_category, len(pool))
        if n < per_category:
            print(f"  [警告] 类别 {cat} 只有 {n} 题，少于请求的 {per_category} 题")
        picked = rng.sample(pool, n)
        for conv_idx, qa_idx, _ in picked:
            sampled_keys.add((conv_idx, qa_idx))

    # 按对话分组：同一对话的多个 QA 共享同一份 ingest 过程
    by_conv = defaultdict(list)
    for conv_idx, qa_idx in sampled_keys:
        by_conv[conv_idx].append(qa_idx)

    eval_set = []
    for conv_idx in sorted(by_conv.keys()):
        conv = raw_data[conv_idx]
        sample_id = conv.get("sample_id", f"conv_{conv_idx}")
        qa_list = []
        for qa_idx in sorted(by_conv[conv_idx]):
            qa = conv["qa"][qa_idx]
            qa_list.append({
                "qa_id": f"{sample_id}_q{qa_idx}",
                "question": qa["question"],
                # 对抗性类别（cat=5）用的字段是 adversarial_answer，这里做兼容
                "answer": qa.get("answer", qa.get("adversarial_answer", "")),
                "category": qa["category"],
                "category_name": CATEGORY_NAMES[qa["category"]],
            })
        eval_set.append({
            "sample_id": sample_id,
            "conversation": flatten_conversation(conv["conversation"]),
            "qa_list": qa_list,
        })

    return eval_set


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="eval_set.json",
                        help="评测集输出路径")
    parser.add_argument("--per_category", type=int, default=40,
                        help="每个类别采样的题数")
    parser.add_argument("--categories", nargs="+", type=int, default=[1, 2, 3, 4],
                        help="要纳入评测的类别 ID（默认排除对抗性类别 5）")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--cache_dir", default=".locomo_cache",
                        help="下载数据集的缓存目录")
    args = parser.parse_args()

    cache_dir = Path(args.cache_dir)
    data_file = download_locomo(cache_dir)

    with open(data_file) as f:
        raw_data = json.load(f)
    print(f"[加载] 共 {len(raw_data)} 段对话")

    eval_set = build_eval_set(
        raw_data,
        per_category=args.per_category,
        include_categories=args.categories,
        seed=args.seed,
    )

    total_qa = sum(len(s["qa_list"]) for s in eval_set)
    print(f"[输出] {len(eval_set)} 段对话，共 {total_qa} 题")
    per_cat = defaultdict(int)
    for s in eval_set:
        for qa in s["qa_list"]:
            per_cat[qa["category_name"]] += 1
    for name in sorted(per_cat.keys()):
        print(f"  {name}：{per_cat[name]} 题")

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(eval_set, f, ensure_ascii=False, indent=2)
    print(f"[已保存] -> {args.output}")


if __name__ == "__main__":
    main()
