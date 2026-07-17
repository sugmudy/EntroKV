from __future__ import annotations

import math
from typing import Optional, Tuple

import torch


def compute_window_entropy(
    attn_weights: torch.Tensor,
    window_size: int,
    num_key_value_groups: int,
    gqa_func: str = "mean",
    eps: float = 1e-12,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    计算 observation window 上的归一化注意力熵。

    Parameters
    ----------
    attn_weights:
        已经经过 softmax 的注意力权重，shape 为：

            [batch, num_query_heads, window_size, total_key_length]

        当前 AdaKV/SnapKV 中，它对应最近 window_size 个 query
        对完整 prompt key 的 attention。

    window_size:
        SnapKV observation window 长度，例如 32。

    num_key_value_groups:
        每个 KV head 对应多少个 query heads。
        Llama-3.1-8B 中为 32 / 8 = 4。

    gqa_func:
        如何将共享同一 KV head 的 query-head entropy 聚合。
        支持 "mean" 或 "max"。

    Returns
    -------
    query_head_entropy:
        [batch, num_query_heads]

    kv_head_entropy:
        [batch, num_kv_heads]

    Notes
    -----
    论文定义是在最近 w 个 query 上，计算其对历史 cached positions
    的注意力分布熵。

    当前 attn_weights 的 softmax 包含 recent window，因此截取历史
    部分后，需要重新归一化，使每个 query 对历史 token 的概率和为 1。
    """

    if attn_weights.ndim != 4:
        raise ValueError(
            "attn_weights 必须为四维张量 "
            "[batch, query_heads, window, key_length]，"
            f"实际 shape={tuple(attn_weights.shape)}"
        )

    batch_size, num_query_heads, actual_window, total_key_length = (
        attn_weights.shape
    )

    if actual_window != window_size:
        raise ValueError(
            f"attention query window={actual_window}，"
            f"但配置 window_size={window_size}"
        )

    if num_key_value_groups <= 0:
        raise ValueError("num_key_value_groups 必须大于 0")

    if num_query_heads % num_key_value_groups != 0:
        raise ValueError(
            f"query heads={num_query_heads} 不能被 "
            f"GQA group size={num_key_value_groups} 整除"
        )

    historical_length = total_key_length - window_size

    if historical_length <= 1:
        # log(1)=0，且只有一个历史位置时熵没有区分意义。
        query_entropy = torch.zeros(
            (batch_size, num_query_heads),
            dtype=torch.float32,
            device=attn_weights.device,
        )
    else:
        # [B, QH, W, historical_length]
        historical_attention = attn_weights[
            ..., :historical_length
        ].float()

        # 原 softmax 包含了 recent-window key。
        # 截取历史部分后重新归一化，得到真正的历史位置概率分布。
        historical_mass = historical_attention.sum(
            dim=-1,
            keepdim=True,
        ).clamp_min(eps)

        historical_probability = (
            historical_attention / historical_mass
        )

        safe_probability = historical_probability.clamp_min(eps)

        # 对历史位置求 Shannon entropy：
        # [B, QH, W, N] -> [B, QH, W]
        entropy_per_query = -(
            historical_probability * safe_probability.log()
        ).sum(dim=-1)

        # 对 observation window 中的 w 个 query 求平均：
        # [B, QH, W] -> [B, QH]
        query_entropy = entropy_per_query.mean(dim=-1)

        # 除以理论最大熵 log(N)，归一化到 [0, 1]。
        query_entropy = query_entropy / math.log(
            historical_length
        )

        query_entropy = query_entropy.clamp(0.0, 1.0)

    num_kv_heads = num_query_heads // num_key_value_groups

    # Llama GQA:
    # [B, 32] -> [B, 8, 4]
    grouped_entropy = query_entropy.view(
        batch_size,
        num_kv_heads,
        num_key_value_groups,
    )

    if gqa_func == "mean":
        kv_entropy = grouped_entropy.mean(dim=-1)
    elif gqa_func == "max":
        kv_entropy = grouped_entropy.max(dim=-1).values
    else:
        raise ValueError(
            f"不支持的 gqa_func={gqa_func!r}，"
            "只支持 'mean' 和 'max'"
        )

    return query_entropy, kv_entropy


def _round_budgets_with_exact_sum(
    raw_budgets: torch.Tensor,
    target_total: int,
    min_capacity: int,
    max_capacity: int,
) -> torch.Tensor:
    """
    将浮点预算转换成整数预算，并严格保证每个 batch 的总和。

    采用带上下界的 largest-remainder / residual correction。
    KV head 数很少，例如 Llama 中只有 8 个，因此逐步修正开销可忽略。
    """

    if raw_budgets.ndim != 2:
        raise ValueError(
            "raw_budgets 必须为 [batch, num_kv_heads]"
        )

    batch_size, num_heads = raw_budgets.shape

    min_total = num_heads * min_capacity
    max_total = num_heads * max_capacity

    if not min_total <= target_total <= max_total:
        raise ValueError(
            "目标预算无法满足上下界："
            f"min_total={min_total}, "
            f"target_total={target_total}, "
            f"max_total={max_total}"
        )

    results = []

    for batch_idx in range(batch_size):
        raw = raw_budgets[batch_idx]

        capacities = torch.floor(raw).to(torch.int64)
        capacities = capacities.clamp(
            min=min_capacity,
            max=max_capacity,
        )

        current_total = int(capacities.sum().item())

        # 总量不足：给 raw - current 最大的 head 逐个补 1。
        while current_total < target_total:
            can_increase = capacities < max_capacity

            if not bool(can_increase.any()):
                raise RuntimeError(
                    "无法继续增加预算，但尚未达到目标总量"
                )

            priority = raw - capacities.to(raw.dtype)
            priority = torch.where(
                can_increase,
                priority,
                torch.full_like(priority, -torch.inf),
            )

            selected_head = int(priority.argmax().item())
            capacities[selected_head] += 1
            current_total += 1

        # 总量超出：从 current - raw 最大的 head 逐个减 1。
        while current_total > target_total:
            can_decrease = capacities > min_capacity

            if not bool(can_decrease.any()):
                raise RuntimeError(
                    "无法继续减少预算，但仍高于目标总量"
                )

            priority = capacities.to(raw.dtype) - raw
            priority = torch.where(
                can_decrease,
                priority,
                torch.full_like(priority, -torch.inf),
            )

            selected_head = int(priority.argmax().item())
            capacities[selected_head] -= 1
            current_total -= 1

        if int(capacities.sum().item()) != target_total:
            raise AssertionError(
                "整数预算总量修正失败："
                f"{int(capacities.sum())} != {target_total}"
            )

        results.append(capacities)

    return torch.stack(results, dim=0).to(torch.int32)


def allocate_entrokv_budgets(
    kv_entropy: torch.Tensor,
    base_capacity: int,
    alpha: float = 0.5,
    h_bar: float = 0.3,
    candidate_length: Optional[int] = None,
    min_capacity: int = 1,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    根据 EntroKV 信号为每个物理 KV head 分配历史 token quota。

    Parameters
    ----------
    kv_entropy:
        [batch, num_kv_heads]，每个 KV head 的归一化熵。

    base_capacity:
        固定 SnapKV 中每个 head 的历史 token quota。

        注意：AdaptiveSnapKVCluster 中：

            self.base_capacity = configured_capacity - window_size

        例如：
            configured capacity = 256
            window size = 32
            base_capacity = 224

    alpha:
        当前样本 entropy 与全局 H_bar 的插值权重。

    h_bar:
        Llama-3.1-8B-Instruct 的论文默认全局基线熵，取 0.3。

    candidate_length:
        当前可供选择的历史 token 数，即 q_len - window_size。

    min_capacity:
        每个 KV head 至少保留多少个历史 token。
        当前设置 1 仅作为数值与 top-k 安全下限。

    Returns
    -------
    integer_budgets:
        [batch, num_kv_heads]，整数历史 quota。

    raw_budgets:
        离散化前的浮点 quota，便于调试和记录。

    Important
    ---------
    当前阶段采用“每层严格预算守恒”：

        sum_h K_l^h = num_kv_heads * base_capacity

    这样 Fixed SnapKV 和 EntroKV 在每层拥有完全相同的总 KV 数量，
    差别只在于预算如何在 head 之间重新分配。
    """

    if kv_entropy.ndim != 2:
        raise ValueError(
            "kv_entropy 必须为 [batch, num_kv_heads]，"
            f"实际 shape={tuple(kv_entropy.shape)}"
        )

    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"alpha 必须位于 [0,1]，实际为 {alpha}")

    if not 0.0 <= h_bar <= 1.0:
        raise ValueError(
            f"h_bar 必须位于 [0,1]，实际为 {h_bar}"
        )

    if base_capacity <= 0:
        raise ValueError(
            f"base_capacity 必须大于 0，实际为 {base_capacity}"
        )

    batch_size, num_kv_heads = kv_entropy.shape

    # EntroKV 的稳定化信号：
    #
    # s_l^h = (1-alpha) * H_bar + alpha * H_l^h
    allocation_signal = (
        (1.0 - alpha) * h_bar
        + alpha * kv_entropy.float()
    )

    allocation_signal = allocation_signal.clamp_min(1e-12)

    # 为了让 EntroKV 与 Fixed SnapKV 使用严格相同的总预算，
    # 先在当前层内归一化：
    #
    # raw_h = H * base_capacity * s_h / sum_j s_j
    target_total = num_kv_heads * base_capacity

    raw_budgets = (
        allocation_signal
        / allocation_signal.sum(dim=-1, keepdim=True)
        * float(target_total)
    )

    if candidate_length is None:
        max_capacity = target_total
    else:
        if candidate_length < min_capacity:
            raise ValueError(
                f"candidate_length={candidate_length} "
                f"小于 min_capacity={min_capacity}"
            )
        max_capacity = int(candidate_length)

    integer_budgets = _round_budgets_with_exact_sum(
        raw_budgets=raw_budgets,
        target_total=target_total,
        min_capacity=min_capacity,
        max_capacity=max_capacity,
    )

    expected_shape = (batch_size, num_kv_heads)

    if tuple(integer_budgets.shape) != expected_shape:
        raise AssertionError(
            f"预算 shape 错误：{tuple(integer_budgets.shape)} "
            f"!= {expected_shape}"
        )

    if not torch.all(
        integer_budgets.sum(dim=-1) == target_total
    ):
        raise AssertionError(
            "存在 batch 的 EntroKV 总预算不守恒"
        )

    return integer_budgets, raw_budgets


def resolve_retention_capacity(
    q_len: int,
    window_size: int,
    capacity_mode: str,
    absolute_capacity: int,
    retention_ratio: Optional[float],
) -> Tuple[int, int, float]:
    """
    将 absolute capacity 或 retention ratio 转换为当前样本的容量。

    Returns
    -------
    final_capacity:
        每个 KV head 最终平均保留的 token 数，包含 recent window。

    history_capacity:
        每个 KV head 平均保留的历史 token 数，不包含 recent window。

    effective_ratio:
        final_capacity / q_len。
    """

    if q_len <= 0:
        raise ValueError(f"q_len 必须大于 0，实际为 {q_len}")

    if window_size < 0:
        raise ValueError(
            f"window_size 不能小于 0，实际为 {window_size}"
        )

    if capacity_mode not in {"absolute", "ratio"}:
        raise ValueError(
            "capacity_mode 必须为 'absolute' 或 'ratio'，"
            f"实际为 {capacity_mode!r}"
        )

    # 序列比 observation window 还短时，不进行压缩。
    if q_len <= window_size:
        return q_len, 0, 1.0

    if capacity_mode == "absolute":
        requested_final_capacity = int(absolute_capacity)

        if requested_final_capacity <= 0:
            raise ValueError(
                "absolute_capacity 必须大于 0，"
                f"实际为 {absolute_capacity}"
            )

    else:
        if retention_ratio is None:
            raise ValueError(
                "capacity_mode='ratio' 时必须提供 "
                "retention_ratio"
            )

        if not 0.0 < retention_ratio <= 1.0:
            raise ValueError(
                "retention_ratio 必须位于 (0,1]，"
                f"实际为 {retention_ratio}"
            )

        # 与论文离散化形式保持一致，使用 floor。
        requested_final_capacity = math.floor(
            q_len * retention_ratio
        )

    # 至少保留整个 recent window 和一个历史 token。
    final_capacity = max(
        window_size + 1,
        requested_final_capacity,
    )

    # 不能超过原始上下文长度。
    final_capacity = min(final_capacity, q_len)

    history_capacity = final_capacity - window_size

    effective_ratio = final_capacity / q_len

    return (
        int(final_capacity),
        int(history_capacity),
        float(effective_ratio),
    )


def allocate_global_entrokv_budgets(
    kv_entropy: torch.Tensor,
    base_history_capacity: int,
    alpha: float = 0.5,
    h_bar: float = 0.3,
    candidate_length: Optional[int] = None,
    min_capacity: int = 1,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    在全部 layer-head 上进行 EntroKV 全局预算分配。

    Parameters
    ----------
    kv_entropy:
        [num_layers, num_kv_heads]

    base_history_capacity:
        固定 SnapKV 基线中，每个 KV head 平均拥有的历史 quota。

    Returns
    -------
    integer_budgets:
        [num_layers, num_kv_heads]

    raw_budgets:
        [num_layers, num_kv_heads]

    Budget conservation
    -------------------
        sum_{l,h} K_l^h
        =
        num_layers * num_kv_heads * base_history_capacity
    """

    if kv_entropy.ndim != 2:
        raise ValueError(
            "kv_entropy 必须为 "
            "[num_layers, num_kv_heads]，"
            f"实际 shape={tuple(kv_entropy.shape)}"
        )

    if not 0.0 <= alpha <= 1.0:
        raise ValueError(
            f"alpha 必须位于 [0,1]，实际为 {alpha}"
        )

    if not 0.0 <= h_bar <= 1.0:
        raise ValueError(
            f"h_bar 必须位于 [0,1]，实际为 {h_bar}"
        )

    if base_history_capacity <= 0:
        raise ValueError(
            "base_history_capacity 必须大于 0，"
            f"实际为 {base_history_capacity}"
        )

    num_layers, num_kv_heads = kv_entropy.shape

    allocation_signal = (
        (1.0 - alpha) * h_bar
        + alpha * kv_entropy.float()
    ).clamp_min(1e-12)

    target_total = (
        num_layers
        * num_kv_heads
        * base_history_capacity
    )

    raw_budgets = (
        allocation_signal
        / allocation_signal.sum()
        * float(target_total)
    )

    if candidate_length is None:
        max_capacity = target_total
    else:
        if candidate_length < min_capacity:
            raise ValueError(
                f"candidate_length={candidate_length} "
                f"小于 min_capacity={min_capacity}"
            )

        max_capacity = int(candidate_length)

    flattened_integer_budgets = (
        _round_budgets_with_exact_sum(
            raw_budgets=raw_budgets.reshape(1, -1),
            target_total=target_total,
            min_capacity=min_capacity,
            max_capacity=max_capacity,
        )
    )

    integer_budgets = flattened_integer_budgets.view(
        num_layers,
        num_kv_heads,
    )

    if int(integer_budgets.sum().item()) != target_total:
        raise AssertionError(
            "全局 EntroKV 整数预算不守恒："
            f"{int(integer_budgets.sum())} != "
            f"{target_total}"
        )

    return integer_budgets, raw_budgets
