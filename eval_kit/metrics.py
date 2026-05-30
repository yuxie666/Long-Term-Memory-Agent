"""
辅助字符串匹配指标：SQuAD 风格的 F1 和 Exact Match。

这两个指标与 LLM Judge 配合使用作为交叉验证：
LLM Judge 是主指标，但它有一定波动，F1/EM 计算廉价，
可以用来检测 Judge 异常（例如 Judge 给了 90% 但 F1 接近 0，就有问题）。
"""

import re
import string
from collections import Counter


def normalize(s: str) -> str:
    """SQuAD 风格的文本归一化：小写、去标点、去冠词、压空白。"""
    if s is None:
        return ""
    s = s.lower()
    s = re.sub(r"\b(a|an|the)\b", " ", s)          # 去冠词
    s = "".join(ch for ch in s if ch not in set(string.punctuation))  # 去标点
    s = re.sub(r"\s+", " ", s).strip()             # 压缩连续空白
    return s


def f1_score(pred: str, ref: str) -> float:
    """词级 F1：预测和参考之间的词重叠度。"""
    pred_tokens = normalize(pred).split()
    ref_tokens = normalize(ref).split()
    if not pred_tokens or not ref_tokens:
        # 两边都为空算完全一致，否则 F1=0
        return float(pred_tokens == ref_tokens)
    common = Counter(pred_tokens) & Counter(ref_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


def exact_match(pred: str, ref: str) -> float:
    """归一化后完全相等返回 1.0，否则 0.0。"""
    return float(normalize(pred) == normalize(ref))
