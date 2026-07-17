from __future__ import annotations

from typing import Optional, Tuple

import torch

from adaptive_snapkv.monkeypatch.entrokv_utils import (
    resolve_retention_capacity,
)
from adaptive_snapkv.monkeypatch.snapkv_utils import (
    SnapKVCluster,
)


class RatioSnapKVCluster(SnapKVCluster):
    """
    按当前样本 q_len 动态设置 Fixed SnapKV 的绝对容量。
    """

    def __init__(
        self,
        *args,
        retention_ratio: float,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)

        if not 0.0 < retention_ratio <= 1.0:
            raise ValueError(
                "retention_ratio 必须位于 (0,1]"
            )

        if self.pyram_mode:
            raise ValueError(
                "Ratio Fixed SnapKV 不应启用 pyram_mode"
            )

        self.retention_ratio = float(retention_ratio)
        self.last_target_final_capacity = None
        self.last_target_history_capacity = None
        self.last_effective_retention_ratio = None

    def update_kv(
        self,
        origin_key_states: torch.Tensor,
        query_states: torch.Tensor,
        origin_value_states: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        q_len = int(query_states.shape[-2])

        (
            target_final_capacity,
            target_history_capacity,
            effective_ratio,
        ) = resolve_retention_capacity(
            q_len=q_len,
            window_size=self.window_size,
            capacity_mode="ratio",
            absolute_capacity=self.max_capacity_prompt,
            retention_ratio=self.retention_ratio,
        )

        self.last_target_final_capacity = (
            target_final_capacity
        )

        self.last_target_history_capacity = (
            target_history_capacity
        )

        self.last_effective_retention_ratio = (
            effective_ratio
        )

        original_capacity = self.max_capacity_prompt

        # 上游 SnapKV 读取 self.max_capacity_prompt；
        # 仅在本次调用期间临时替换。
        self.max_capacity_prompt = target_final_capacity

        try:
            return super().update_kv(
                origin_key_states,
                query_states,
                origin_value_states,
            )
        finally:
            self.max_capacity_prompt = original_capacity
