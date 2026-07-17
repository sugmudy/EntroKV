from __future__ import annotations

import math
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F

from adaptive_snapkv.monkeypatch.entrokv_utils import (
    allocate_entrokv_budgets,
    compute_window_entropy,
    resolve_retention_capacity,
)
from adaptive_snapkv.monkeypatch.snapkv_utils import (
    AdaptiveSnapKVCluster,
    repeat_kv,
)


class EntroKVCluster(AdaptiveSnapKVCluster):
    """
    EntroKV + SnapKV GQA implementation.

    Supported normalization scopes
    ------------------------------
    per_layer:
        每一层内部的 8 个 KV heads 单独归一化。
        对应阶段 3B。

    global:
        在全部 layer-head 上统一归一化。
        当前层只收集 entropy、token score 和原始 KV；
        所有层完成后统一压缩。
    """

    def __init__(
        self,
        *args,
        entrokv_alpha: float = 0.5,
        entrokv_h_bar: float = 0.3,
        entrokv_debug: bool = False,
        entrokv_scope: str = "per_layer",
        capacity_mode: str = "absolute",
        retention_ratio: Optional[float] = None,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)

        if not self.gqa_support:
            raise ValueError(
                "当前 EntroKVCluster 只实现 GQA 路径"
            )

        if self.pyram_mode:
            raise ValueError(
                "EntroKV 与 PyramidKV layer schedule "
                "不能同时启用"
            )

        if entrokv_scope not in {"per_layer", "global"}:
            raise ValueError(
                "entrokv_scope 必须为 "
                "'per_layer' 或 'global'，"
                f"实际为 {entrokv_scope!r}"
            )

        if capacity_mode not in {"absolute", "ratio"}:
            raise ValueError(
                "capacity_mode 必须为 "
                "'absolute' 或 'ratio'，"
                f"实际为 {capacity_mode!r}"
            )

        if capacity_mode == "ratio":
            if retention_ratio is None:
                raise ValueError(
                    "ratio 模式必须提供 retention_ratio"
                )

            if not 0.0 < retention_ratio <= 1.0:
                raise ValueError(
                    "retention_ratio 必须位于 (0,1]"
                )

        self.entrokv_alpha = float(entrokv_alpha)
        self.entrokv_h_bar = float(entrokv_h_bar)
        self.entrokv_debug = bool(entrokv_debug)
        self.entrokv_scope = entrokv_scope
        self.capacity_mode = capacity_mode
        self.retention_ratio = retention_ratio
        self.budget_mode = "entrokv"

        # 最近一次 prefill 的记录。
        self.last_query_entropy = None
        self.last_kv_entropy = None
        self.last_raw_budgets = None
        self.last_history_budgets = None
        self.last_prefill_head_lens = None
        self.last_prefill_q_len = None

        self.last_target_final_capacity = None
        self.last_target_history_capacity = None
        self.last_effective_retention_ratio = None
        self.last_prefill_flattened_rows = None

        # Global scope 下的 GPU 临时状态。
        self.pending_token_score = None
        self.pending_kv_entropy = None
        self.pending_query_entropy = None
        self.pending_q_len = None
        self.pending_history_length = None

    def _resolve_current_capacity(
        self,
        q_len: int,
    ) -> Tuple[int, int, float]:
        configured_final_capacity = (
            self.base_capacity + self.window_size
        )

        return resolve_retention_capacity(
            q_len=q_len,
            window_size=self.window_size,
            capacity_mode=self.capacity_mode,
            absolute_capacity=configured_final_capacity,
            retention_ratio=self.retention_ratio,
        )

    def _init_gqa_metadata(
        self,
        num_query_heads: int,
        k_lens: List[int],
        device: torch.device,
    ) -> None:
        num_kv_heads = (
            num_query_heads // self.num_key_value_groups
        )

        if len(k_lens) != num_kv_heads:
            raise AssertionError(
                f"k_lens 应有 {num_kv_heads} 项，"
                f"实际为 {len(k_lens)}"
            )

        self.head_lens = torch.tensor(
            k_lens,
            dtype=torch.int32,
            device=device,
        )

        self.klen_sum = int(self.head_lens.sum().item())
        self.max_seqlen_k = int(
            self.head_lens.max().item()
        )

        cumulative = torch.cumsum(
            self.head_lens,
            dim=0,
            dtype=torch.int32,
        )

        self.cu_headlens = cumulative

        self.cu_klen = torch.cat(
            [
                torch.zeros(
                    1,
                    dtype=torch.int32,
                    device=device,
                ),
                cumulative,
            ],
            dim=0,
        )

        self.layer_qlens = torch.ones(
            num_kv_heads,
            dtype=torch.int32,
            device=device,
        )

        self.qlen_sum = num_kv_heads

        self.cu_qlen = torch.arange(
            0,
            num_kv_heads + 1,
            dtype=torch.int32,
            device=device,
        )

        self.cu_offset = torch.arange(
            0,
            num_kv_heads + 1,
            dtype=torch.int32,
            device=device,
        )

        self.cu_head_offset = torch.arange(
            1,
            num_kv_heads + 1,
            dtype=torch.int32,
            device=device,
        )

    def _compute_attention_products(
        self,
        repeated_key_states: torch.Tensor,
        query_states: torch.Tensor,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        batch_size, num_query_heads, q_len, head_dim = (
            query_states.shape
        )

        window_size = int(self.window_size)

        if q_len <= window_size:
            raise ValueError(
                f"q_len={q_len} 必须大于 "
                f"window_size={window_size}"
            )

        expected_shape = (
            batch_size,
            num_query_heads,
            q_len,
        )

        if repeated_key_states.shape[:3] != expected_shape:
            raise AssertionError(
                "repeat_kv 后的 K shape 与 Q 不匹配："
                f"K={tuple(repeated_key_states.shape)}, "
                f"Q={tuple(query_states.shape)}"
            )

        attention_logits = torch.matmul(
            query_states[..., -window_size:, :],
            repeated_key_states.transpose(2, 3),
        ) / math.sqrt(head_dim)

        causal_mask = torch.full(
            (window_size, window_size),
            torch.finfo(attention_logits.dtype).min,
            dtype=attention_logits.dtype,
            device=attention_logits.device,
        )

        position = torch.arange(
            window_size,
            device=attention_logits.device,
        )

        causal_mask.masked_fill_(
            position
            < (position + 1).view(window_size, 1),
            0,
        )

        attention_logits[
            :,
            :,
            -window_size:,
            -window_size:,
        ] += causal_mask[None, None, :, :]

        attention_weights = F.softmax(
            attention_logits,
            dim=-1,
            dtype=torch.float32,
        ).to(query_states.dtype)

        historical_length = q_len - window_size

        query_entropy, kv_entropy = (
            compute_window_entropy(
                attn_weights=attention_weights,
                window_size=window_size,
                num_key_value_groups=(
                    self.num_key_value_groups
                ),
                gqa_func=self.gqa_func,
            )
        )

        token_score = attention_weights[
            ...,
            :historical_length,
        ].mean(dim=-2)

        num_kv_heads = (
            num_query_heads // self.num_key_value_groups
        )

        token_score = token_score.view(
            batch_size,
            num_kv_heads,
            self.num_key_value_groups,
            historical_length,
        )

        if self.gqa_func == "mean":
            token_score = token_score.mean(dim=2)
        elif self.gqa_func == "max":
            token_score = token_score.max(dim=2).values
        else:
            raise ValueError(
                f"不支持的 gqa_func={self.gqa_func!r}"
            )

        if self.pooling == "avgpool":
            token_score = F.avg_pool1d(
                token_score,
                kernel_size=self.kernel_size,
                padding=self.kernel_size // 2,
                stride=1,
            )
        elif self.pooling == "maxpool":
            token_score = F.max_pool1d(
                token_score,
                kernel_size=self.kernel_size,
                padding=self.kernel_size // 2,
                stride=1,
            )
        else:
            raise ValueError(
                f"不支持的 pooling={self.pooling!r}"
            )

        return token_score, query_entropy, kv_entropy

    def _compress_from_budget(
        self,
        origin_key_states: torch.Tensor,
        origin_value_states: torch.Tensor,
        token_score: torch.Tensor,
        history_budgets: torch.Tensor,
        raw_budgets: torch.Tensor,
        query_entropy: torch.Tensor,
        kv_entropy: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, num_kv_heads, q_len, head_dim = (
            origin_key_states.shape
        )

        if batch_size != 1:
            raise AssertionError(
                "flattened cache 当前只支持 batch_size=1"
            )

        historical_length = q_len - self.window_size

        if tuple(history_budgets.shape) != (
            1,
            num_kv_heads,
        ):
            raise AssertionError(
                "history_budgets shape 错误："
                f"{tuple(history_budgets.shape)}"
            )

        if int(history_budgets.max().item()) > (
            historical_length
        ):
            raise AssertionError(
                "存在 quota 超过历史候选长度："
                f"max_quota={int(history_budgets.max())}, "
                f"history={historical_length}"
            )

        _, sorted_token_indices = token_score.sort(
            dim=-1,
            descending=True,
        )

        origin_key_by_head = torch.split(
            origin_key_states,
            1,
            dim=1,
        )

        origin_value_by_head = torch.split(
            origin_value_states,
            1,
            dim=1,
        )

        sorted_indices_by_head = (
            sorted_token_indices.split(1, dim=1)
        )

        flattened_keys = []
        flattened_values = []
        k_lens = []

        for head_idx in range(num_kv_heads):
            quota = int(
                history_budgets[0, head_idx].item()
            )

            # 不压缩时保留原始顺序。
            if quota == historical_length:
                cache_index = torch.arange(
                    historical_length,
                    dtype=torch.long,
                    device=origin_key_states.device,
                ).view(1, 1, -1)
            else:
                cache_index = sorted_indices_by_head[
                    head_idx
                ][..., :quota]

            cache_index = cache_index.view(
                1,
                1,
                quota,
                1,
            ).expand(
                -1,
                -1,
                -1,
                head_dim,
            )

            history_k = origin_key_by_head[head_idx][
                :,
                :,
                :-self.window_size,
                :,
            ].gather(
                dim=2,
                index=cache_index,
            )

            history_v = origin_value_by_head[head_idx][
                :,
                :,
                :-self.window_size,
                :,
            ].gather(
                dim=2,
                index=cache_index,
            )

            recent_k = origin_key_by_head[head_idx][
                :,
                :,
                -self.window_size:,
                :,
            ]

            recent_v = origin_value_by_head[head_idx][
                :,
                :,
                -self.window_size:,
                :,
            ]

            selected_k = torch.cat(
                [history_k, recent_k],
                dim=2,
            )

            selected_v = torch.cat(
                [history_v, recent_v],
                dim=2,
            )

            head_length = quota + self.window_size
            k_lens.append(head_length)

            flattened_keys.append(
                selected_k.reshape(-1, head_dim)
            )

            flattened_values.append(
                selected_v.reshape(-1, head_dim)
            )

        self._init_gqa_metadata(
            num_query_heads=(
                num_kv_heads
                * self.num_key_value_groups
            ),
            k_lens=k_lens,
            device=origin_key_states.device,
        )

        flattened_keys = torch.cat(
            flattened_keys,
            dim=0,
        )

        flattened_values = torch.cat(
            flattened_values,
            dim=0,
        )

        if flattened_keys.shape[0] != self.klen_sum:
            raise AssertionError(
                "flattened K 行数与 klen_sum 不一致"
            )

        if flattened_values.shape[0] != self.klen_sum:
            raise AssertionError(
                "flattened V 行数与 klen_sum 不一致"
            )

        self.last_query_entropy = (
            query_entropy.detach().cpu().clone()
        )

        self.last_kv_entropy = (
            kv_entropy.detach().cpu().clone()
        )

        self.last_raw_budgets = (
            raw_budgets.detach().cpu().clone()
        )

        self.last_history_budgets = (
            history_budgets.detach().cpu().clone()
        )

        self.last_prefill_head_lens = (
            self.head_lens.detach().cpu().clone()
        )

        self.last_prefill_q_len = q_len
        self.last_prefill_flattened_rows = int(
            flattened_keys.shape[0]
        )

        if self.entrokv_debug:
            self._print_debug()

        return flattened_keys, flattened_values

    def _print_debug(self) -> None:
        entropy_list = self.last_kv_entropy[
            0
        ].tolist()

        budget_list = self.last_history_budgets[
            0
        ].tolist()

        lengths = self.last_prefill_head_lens.tolist()

        print(
            f"[EntroKV-{self.entrokv_scope}] "
            f"layer={self.layer_idx:02d}, "
            f"q_len={self.last_prefill_q_len}",
            flush=True,
        )

        print(
            f"  KV entropy:      {entropy_list}",
            flush=True,
        )

        print(
            f"  history budgets: {budget_list}",
            flush=True,
        )

        print(
            f"  layer history total: {sum(budget_list)}",
            flush=True,
        )

        print(
            f"  prefill lengths: {lengths}",
            flush=True,
        )

        print(
            f"  layer prefill total: {sum(lengths)}",
            flush=True,
        )

    def update_kv_gqa(
        self,
        origin_key_states: torch.Tensor,
        query_states: torch.Tensor,
        origin_value_states: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        repeated_key_states = repeat_kv(
            origin_key_states,
            self.num_key_value_groups,
        )

        batch_size, num_query_heads, q_len, head_dim = (
            query_states.shape
        )

        _, num_kv_heads, kv_len, kv_head_dim = (
            origin_key_states.shape
        )

        if batch_size != 1:
            raise AssertionError(
                "当前实现只支持 batch_size=1"
            )

        if q_len != kv_len:
            raise AssertionError(
                f"Q/K 长度不一致：{q_len} != {kv_len}"
            )

        if head_dim != kv_head_dim:
            raise AssertionError(
                "Q/K head_dim 不一致"
            )

        if (
            num_query_heads
            != num_kv_heads * self.num_key_value_groups
        ):
            raise AssertionError(
                "GQA query/KV head 映射错误"
            )

        (
            target_final_capacity,
            target_history_capacity,
            effective_ratio,
        ) = self._resolve_current_capacity(q_len)

        self.last_target_final_capacity = (
            target_final_capacity
        )

        self.last_target_history_capacity = (
            target_history_capacity
        )

        self.last_effective_retention_ratio = (
            effective_ratio
        )

        token_score, query_entropy, kv_entropy = (
            self._compute_attention_products(
                repeated_key_states,
                query_states,
            )
        )

        historical_length = q_len - self.window_size

        # --------------------------------------------------
        # Global mode：暂不压缩，等待全部层 entropy。
        # --------------------------------------------------
        if self.entrokv_scope == "global":
            self.pending_token_score = token_score.detach()
            self.pending_query_entropy = (
                query_entropy.detach()
            )
            self.pending_kv_entropy = kv_entropy.detach()
            self.pending_q_len = int(q_len)
            self.pending_history_length = int(
                historical_length
            )

            self.last_query_entropy = (
                query_entropy.detach().cpu().clone()
            )

            self.last_kv_entropy = (
                kv_entropy.detach().cpu().clone()
            )

            # 暂时用 full-cache metadata；
            # 全部层完成后会被 global finalizer 替换。
            self._init_gqa_metadata(
                num_query_heads=num_query_heads,
                k_lens=[q_len] * num_kv_heads,
                device=origin_key_states.device,
            )

            return (
                origin_key_states.reshape(-1, head_dim),
                origin_value_states.reshape(-1, head_dim),
            )

        # --------------------------------------------------
        # Stage 3B per-layer mode。
        # --------------------------------------------------
        history_budgets, raw_budgets = (
            allocate_entrokv_budgets(
                kv_entropy=kv_entropy,
                base_capacity=target_history_capacity,
                alpha=self.entrokv_alpha,
                h_bar=self.entrokv_h_bar,
                candidate_length=historical_length,
                min_capacity=1,
            )
        )

        return self._compress_from_budget(
            origin_key_states=origin_key_states,
            origin_value_states=origin_value_states,
            token_score=token_score,
            history_budgets=history_budgets,
            raw_budgets=raw_budgets,
            query_entropy=query_entropy,
            kv_entropy=kv_entropy,
        )

    def finalize_global_layer(
        self,
        origin_key_states: torch.Tensor,
        origin_value_states: torch.Tensor,
        history_budgets: torch.Tensor,
        raw_budgets: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        由 global finalizer 在全部层 entropy 收集完成后调用。
        """

        if self.entrokv_scope != "global":
            raise RuntimeError(
                "finalize_global_layer 只能用于 global scope"
            )

        if self.pending_token_score is None:
            raise RuntimeError(
                f"Layer {self.layer_idx} "
                "缺少 pending token score"
            )

        output = self._compress_from_budget(
            origin_key_states=origin_key_states,
            origin_value_states=origin_value_states,
            token_score=self.pending_token_score,
            history_budgets=history_budgets,
            raw_budgets=raw_budgets,
            query_entropy=self.pending_query_entropy,
            kv_entropy=self.pending_kv_entropy,
        )

        # 释放 GPU 临时状态。
        self.pending_token_score = None
        self.pending_query_entropy = None
        self.pending_kv_entropy = None
        self.pending_q_len = None
        self.pending_history_length = None

        return output
