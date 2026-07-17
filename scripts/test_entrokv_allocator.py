#!/usr/bin/env python3

import torch

from adaptive_snapkv.monkeypatch.entrokv_utils import (
    allocate_entrokv_budgets,
    compute_window_entropy,
)


def build_synthetic_attention() -> torch.Tensor:
    """
    创建 32 个 query heads、8 个 KV heads 的合成注意力。

    KV head 0：非常集中，低熵
    KV head 7：接近均匀，高熵

    每个 KV head 对应连续 4 个 query heads。
    """

    batch_size = 1
    query_heads = 32
    kv_heads = 8
    groups = 4
    window_size = 2
    historical_length = 16
    total_length = historical_length + window_size

    attention = torch.zeros(
        batch_size,
        query_heads,
        window_size,
        total_length,
        dtype=torch.float32,
    )

    uniform = torch.full(
        (historical_length,),
        1.0 / historical_length,
    )

    peaked = torch.zeros(historical_length)
    peaked[0] = 1.0

    for kv_head in range(kv_heads):
        # kv_head=0 时 mixture=0：完全 peaked
        # kv_head=7 时 mixture=1：完全 uniform
        mixture = kv_head / (kv_heads - 1)

        probability = (
            (1.0 - mixture) * peaked
            + mixture * uniform
        )

        for local_query_head in range(groups):
            query_head = kv_head * groups + local_query_head

            for query_idx in range(window_size):
                attention[
                    0,
                    query_head,
                    query_idx,
                    :historical_length,
                ] = probability

    return attention


def main() -> None:
    attention = build_synthetic_attention()

    query_entropy, kv_entropy = compute_window_entropy(
        attn_weights=attention,
        window_size=2,
        num_key_value_groups=4,
        gqa_func="mean",
    )

    print("query entropy shape:", tuple(query_entropy.shape))
    print("KV entropy shape:   ", tuple(kv_entropy.shape))
    print("KV entropy:         ", kv_entropy[0].tolist())

    assert tuple(query_entropy.shape) == (1, 32)
    assert tuple(kv_entropy.shape) == (1, 8)

    assert torch.all(kv_entropy >= 0)
    assert torch.all(kv_entropy <= 1)

    # 合成分布从 head 0 到 head 7 越来越均匀，
    # 所以熵应该单调不减。
    assert torch.all(
        kv_entropy[0, 1:] >= kv_entropy[0, :-1]
    )

    # alpha=0：完全依赖 H_bar，应该退化为固定均匀预算。
    fixed_budgets, fixed_raw = allocate_entrokv_budgets(
        kv_entropy=kv_entropy,
        base_capacity=224,
        alpha=0.0,
        h_bar=0.3,
        candidate_length=16_000,
    )

    print("\nalpha=0 raw budgets:")
    print(fixed_raw[0].tolist())

    print("alpha=0 integer budgets:")
    print(fixed_budgets[0].tolist())

    assert torch.all(fixed_budgets == 224)
    assert int(fixed_budgets.sum()) == 8 * 224

    # alpha=0.5：高熵 head 应获得更多预算。
    dynamic_budgets, dynamic_raw = allocate_entrokv_budgets(
        kv_entropy=kv_entropy,
        base_capacity=224,
        alpha=0.5,
        h_bar=0.3,
        candidate_length=16_000,
    )

    print("\nalpha=0.5 raw budgets:")
    print(dynamic_raw[0].tolist())

    print("alpha=0.5 integer budgets:")
    print(dynamic_budgets[0].tolist())

    print("dynamic total:", int(dynamic_budgets.sum()))

    assert int(dynamic_budgets.sum()) == 8 * 224

    assert (
        int(dynamic_budgets[0, -1])
        > int(dynamic_budgets[0, 0])
    )

    # 高熵程度越高，预算整体应呈非下降趋势。
    assert torch.all(
        dynamic_budgets[0, 1:]
        >= dynamic_budgets[0, :-1]
    )

    print("\n[PASS] Entropy shape and range are correct.")
    print("[PASS] GQA entropy aggregation is correct.")
    print("[PASS] alpha=0 degenerates to fixed allocation.")
    print("[PASS] High-entropy KV heads receive larger budgets.")
    print("[PASS] Integer budget total is exactly conserved.")


if __name__ == "__main__":
    main()
