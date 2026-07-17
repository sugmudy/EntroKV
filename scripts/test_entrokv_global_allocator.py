#!/usr/bin/env python3

import torch

from adaptive_snapkv.monkeypatch.entrokv_utils import (
    allocate_global_entrokv_budgets,
    resolve_retention_capacity,
)


def main() -> None:
    q_len = 5265
    window_size = 32
    ratio = 0.3

    final_capacity, history_capacity, effective_ratio = (
        resolve_retention_capacity(
            q_len=q_len,
            window_size=window_size,
            capacity_mode="ratio",
            absolute_capacity=256,
            retention_ratio=ratio,
        )
    )

    print("q_len:", q_len)
    print("target ratio:", ratio)
    print("final capacity/head:", final_capacity)
    print("history capacity/head:", history_capacity)
    print("effective ratio:", effective_ratio)

    assert final_capacity == 1579
    assert history_capacity == 1547

    entropy = torch.linspace(
        0.0,
        1.0,
        steps=32 * 8,
        dtype=torch.float32,
    ).view(32, 8)

    budgets, raw = allocate_global_entrokv_budgets(
        kv_entropy=entropy,
        base_history_capacity=history_capacity,
        alpha=0.5,
        h_bar=0.3,
        candidate_length=q_len - window_size,
    )

    expected_total = 32 * 8 * history_capacity

    print("\ndynamic budget shape:", tuple(budgets.shape))
    print("dynamic budget total:", int(budgets.sum()))
    print(
        "layer history totals:",
        budgets.sum(dim=-1).tolist(),
    )

    assert tuple(budgets.shape) == (32, 8)
    assert int(budgets.sum()) == expected_total
    assert budgets.max() > budgets.min()

    fixed_budgets, _ = allocate_global_entrokv_budgets(
        kv_entropy=entropy,
        base_history_capacity=history_capacity,
        alpha=0.0,
        h_bar=0.3,
        candidate_length=q_len - window_size,
    )

    assert torch.all(fixed_budgets == history_capacity)
    assert int(fixed_budgets.sum()) == expected_total

    print("\n[PASS] Ratio capacity is correct.")
    print("[PASS] Global budget shape is [32,8].")
    print("[PASS] Dynamic global budget is non-uniform.")
    print("[PASS] Global integer budget is exactly conserved.")
    print("[PASS] alpha=0 globally degenerates to fixed budget.")


if __name__ == "__main__":
    main()
