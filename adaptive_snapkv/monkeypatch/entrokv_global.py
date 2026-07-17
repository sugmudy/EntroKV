from __future__ import annotations

import torch

from adaptive_snapkv.monkeypatch.entrokv_utils import (
    allocate_global_entrokv_budgets,
)


def finalize_entrokv_global_cache(
    llama_model,
    cache,
) -> None:
    """
    在 Llama 的全部 decoder layers 完成 prefill 后：

      1. 收集 [L,Hkv] entropy；
      2. 计算全局 quota；
      3. 对每一层执行 SnapKV top-k；
      4. 将 full flattened cache 替换为 compressed cache。
    """

    layers = llama_model.layers

    num_layers = len(layers)

    if num_layers == 0:
        raise RuntimeError("模型没有 decoder layers")

    clusters = [
        layer.self_attn.kv_cluster
        for layer in layers
    ]

    for layer_idx, cluster in enumerate(clusters):
        if cluster.__class__.__name__ != "EntroKVCluster":
            raise RuntimeError(
                f"Layer {layer_idx} 不是 EntroKVCluster"
            )

        if cluster.entrokv_scope != "global":
            raise RuntimeError(
                f"Layer {layer_idx} 不是 global scope"
            )

        if cluster.pending_kv_entropy is None:
            raise RuntimeError(
                f"Layer {layer_idx} 没有收集 entropy"
            )

    q_lens = {
        int(cluster.pending_q_len)
        for cluster in clusters
    }

    if len(q_lens) != 1:
        raise RuntimeError(
            f"不同层 q_len 不一致：{q_lens}"
        )

    q_len = next(iter(q_lens))

    num_kv_heads = int(
        llama_model.config.num_key_value_heads
    )

    head_dim = int(
        llama_model.config.hidden_size
        // llama_model.config.num_attention_heads
    )

    window_size = int(clusters[0].window_size)

    target_history_capacities = {
        int(cluster.last_target_history_capacity)
        for cluster in clusters
    }

    target_final_capacities = {
        int(cluster.last_target_final_capacity)
        for cluster in clusters
    }

    if len(target_history_capacities) != 1:
        raise RuntimeError(
            "不同层 target history capacity 不一致"
        )

    if len(target_final_capacities) != 1:
        raise RuntimeError(
            "不同层 target final capacity 不一致"
        )

    base_history_capacity = next(
        iter(target_history_capacities)
    )

    target_final_capacity = next(
        iter(target_final_capacities)
    )

    historical_length = q_len - window_size

    # [32, 8]
    global_entropy = torch.cat(
        [
            cluster.pending_kv_entropy
            for cluster in clusters
        ],
        dim=0,
    )

    global_budgets, global_raw_budgets = (
        allocate_global_entrokv_budgets(
            kv_entropy=global_entropy,
            base_history_capacity=(
                base_history_capacity
            ),
            alpha=clusters[0].entrokv_alpha,
            h_bar=clusters[0].entrokv_h_bar,
            candidate_length=historical_length,
            min_capacity=1,
        )
    )

    expected_history_total = (
        num_layers
        * num_kv_heads
        * base_history_capacity
    )

    if int(global_budgets.sum().item()) != (
        expected_history_total
    ):
        raise AssertionError(
            "全局历史预算不守恒"
        )

    layer_history_totals = []
    layer_prefill_totals = []

    for layer_idx, cluster in enumerate(clusters):
        raw_flattened_k = cache.key_cache[layer_idx]
        raw_flattened_v = cache.value_cache[layer_idx]

        expected_raw_rows = (
            num_kv_heads * q_len
        )

        if raw_flattened_k.ndim != 2:
            raise AssertionError(
                f"Layer {layer_idx} raw K 不是二维 tensor"
            )

        if raw_flattened_k.shape[0] != expected_raw_rows:
            raise AssertionError(
                f"Layer {layer_idx} raw K rows 错误："
                f"{raw_flattened_k.shape[0]} != "
                f"{expected_raw_rows}"
            )

        origin_key_states = raw_flattened_k.view(
            1,
            num_kv_heads,
            q_len,
            head_dim,
        )

        origin_value_states = raw_flattened_v.view(
            1,
            num_kv_heads,
            q_len,
            head_dim,
        )

        layer_budgets = global_budgets[
            layer_idx:layer_idx + 1
        ]

        layer_raw_budgets = global_raw_budgets[
            layer_idx:layer_idx + 1
        ]

        compressed_k, compressed_v = (
            cluster.finalize_global_layer(
                origin_key_states=origin_key_states,
                origin_value_states=(
                    origin_value_states
                ),
                history_budgets=layer_budgets,
                raw_budgets=layer_raw_budgets,
            )
        )

        cache.key_cache[layer_idx] = compressed_k
        cache.value_cache[layer_idx] = compressed_v

        layer_history_total = int(
            layer_budgets.sum().item()
        )

        layer_prefill_total = int(
            cluster.head_lens.sum().item()
        )

        layer_history_totals.append(
            layer_history_total
        )

        layer_prefill_totals.append(
            layer_prefill_total
        )

    expected_prefill_total = (
        expected_history_total
        + num_layers * num_kv_heads * window_size
    )

    actual_prefill_total = sum(
        layer_prefill_totals
    )

    if actual_prefill_total != expected_prefill_total:
        raise AssertionError(
            "全局 prefill 总量不守恒："
            f"{actual_prefill_total} != "
            f"{expected_prefill_total}"
        )

    # 保存到 LlamaModel，供评测和可视化读取。
    llama_model.entrokv_last_global_entropy = (
        global_entropy.detach().cpu().clone()
    )

    llama_model.entrokv_last_global_raw_budgets = (
        global_raw_budgets.detach().cpu().clone()
    )

    llama_model.entrokv_last_global_budgets = (
        global_budgets.detach().cpu().clone()
    )

    llama_model.entrokv_last_layer_history_totals = (
        torch.tensor(
            layer_history_totals,
            dtype=torch.int64,
        )
    )

    llama_model.entrokv_last_layer_prefill_totals = (
        torch.tensor(
            layer_prefill_totals,
            dtype=torch.int64,
        )
    )

    llama_model.entrokv_last_target_history_capacity = (
        base_history_capacity
    )

    llama_model.entrokv_last_target_final_capacity = (
        target_final_capacity
    )

    llama_model.entrokv_last_global_history_total = (
        expected_history_total
    )

    llama_model.entrokv_last_global_prefill_total = (
        expected_prefill_total
    )

    if clusters[0].entrokv_debug:
        print(
            "\n[EntroKV-global] allocation finalized",
            flush=True,
        )

        print(
            f"  budget matrix shape: "
            f"{tuple(global_budgets.shape)}",
            flush=True,
        )

        print(
            f"  target final capacity/head: "
            f"{target_final_capacity}",
            flush=True,
        )

        print(
            f"  target history capacity/head: "
            f"{base_history_capacity}",
            flush=True,
        )

        print(
            f"  global history total: "
            f"{int(global_budgets.sum())} / "
            f"{expected_history_total}",
            flush=True,
        )

        print(
            f"  global prefill total: "
            f"{actual_prefill_total} / "
            f"{expected_prefill_total}",
            flush=True,
        )

        print(
            f"  layer history min/max: "
            f"{min(layer_history_totals)} / "
            f"{max(layer_history_totals)}",
            flush=True,
        )
