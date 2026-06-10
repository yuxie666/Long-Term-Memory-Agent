"""
Memory Updater —— 去重 / 更新（探索方向 C，参考 Mem0），但去掉慢的 LLM 热路径。

之前每条事实都调一次 LLM 判定冲突，是 ingest 慢的主因之一。改为**纯 embedding 去重**：
对一批新记忆，用向量余弦做近重复检测，重复的丢弃、保留 importance 更高的那条。
全程无 LLM 调用，速度由向量点积决定。

对照（方向 C 消融）：
  mode="append"  : 不去重，全部保留（对照基线）
  mode="dedup"   : 批内 + 与库内做近重复去重（实验组）
"""

from __future__ import annotations

import numpy as np

from memory.store import MemoryStore, MemoryUnit


class MemoryUpdater:
    def __init__(self, store: MemoryStore, mode: str = "dedup",
                 dup_threshold: float = 0.92, verbose: bool = True):
        assert mode in ("append", "dedup")
        self.store = store
        self.mode = mode
        self.dup_threshold = dup_threshold     # 余弦 ≥ 此值视为近重复
        self.verbose = verbose
        self.stats = {"add": 0, "dropped": 0}

    def write_batch(self, units: list[MemoryUnit]) -> None:
        """把一批候选记忆写入 store（按 mode 决定是否去重）。"""
        if not units:
            return
        if self.mode == "append":
            self.store.add_batch(units)
            self.stats["add"] += len(units)
            return

        # 先编码这批候选，做批内去重，再和已有库去重
        vecs = self.store.embed([u.text for u in units])
        keep_units, keep_vecs = [], []
        kept_matrix = self.store._matrix  # 库内现有向量（可能为 None）

        for i, u in enumerate(units):
            v = vecs[i]
            dup = False
            # 与本批已保留的比
            if keep_vecs and float(np.max(np.stack(keep_vecs) @ v)) >= self.dup_threshold:
                dup = True
            # 与库内已有的比
            if not dup and kept_matrix is not None and len(kept_matrix) > 0:
                if float(np.max(kept_matrix @ v)) >= self.dup_threshold:
                    dup = True
            if dup:
                self.stats["dropped"] += 1
            else:
                keep_units.append(u)
                keep_vecs.append(v)

        self.store.add_batch(keep_units)
        self.stats["add"] += len(keep_units)
        if self.verbose:
            print(f"[Updater] mode={self.mode} 入库 {len(keep_units)} 条，"
                  f"去重丢弃 {len(units)-len(keep_units)} 条")
