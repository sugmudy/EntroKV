#!/usr/bin/env python3

from __future__ import annotations

import argparse
import math
import os
import time
from typing import Dict

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)

import adaptive_snapkv.monkeypatch.snapkv_utils as snapkv_utils
from adaptive_snapkv.monkeypatch.monkeypatch import (
    config_compress,
    replace_llama_adaptive,
)


FINAL_CACHE_ROWS: Dict[int, int] = {}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def install_cache_recorder() -> None:
    original_update = (
        snapkv_utils.DynamicCacheSplitHeadFlatten.update
    )

    def wrapped_update(
        self,
        key_states,
        value_states,
        layer_idx,
        cache_kwargs=None,
    ):
        output_k, output_v = original_update(
            self,
            key_states,
            value_states,
            layer_idx,
            cache_kwargs,
        )

        if (
            isinstance(output_k, torch.Tensor)
            and output_k.ndim == 2
        ):
            FINAL_CACHE_ROWS[int(layer_idx)] = int(
                output_k.shape[0]
            )

        return output_k, output_v

    snapkv_utils.DynamicCacheSplitHeadFlatten.update = (
        wrapped_update
    )


def build_input(tokenizer, minimum_tokens: int):
    filler = (
        "alpha beta gamma delta epsilon zeta eta theta "
        "iota kappa lambda. "
    )

    repeats = 200

    while True:
        prompt = (
            "Read the following context carefully.\n\n"
            f"{filler * repeats}\n\n"
            "The secret code is ORANGE-739.\n\n"
            f"{filler * repeats}\n\n"
            "Question: What is the secret code? "
            "Return only the code."
        )

        input_ids = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            add_generation_prompt=True,
            return_tensors="pt",
        )

        if input_ids.shape[-1] >= minimum_tokens:
            return input_ids

        repeats += 50


def validate(
    model,
    q_len: int,
    ratio: float,
    window_size: int,
    alpha: float,
    h_bar: float,
    decode_count: int,
) -> None:
    llama = model.model

    num_layers = model.config.num_hidden_layers
    num_kv_heads = model.config.num_key_value_heads

    target_final_capacity = max(
        window_size + 1,
        math.floor(q_len * ratio),
    )

    target_final_capacity = min(
        q_len,
        target_final_capacity,
    )

    target_history_capacity = (
        target_final_capacity - window_size
    )

    expected_global_history_total = (
        num_layers
        * num_kv_heads
        * target_history_capacity
    )

    expected_global_prefill_total = (
        num_layers
        * num_kv_heads
        * target_final_capacity
    )

    entropy = llama.entrokv_last_global_entropy
    raw_budgets = (
        llama.entrokv_last_global_raw_budgets
    )
    budgets = llama.entrokv_last_global_budgets

    require(
        tuple(entropy.shape)
        == (num_layers, num_kv_heads),
        f"entropy shape 错误：{tuple(entropy.shape)}",
    )

    require(
        tuple(budgets.shape)
        == (num_layers, num_kv_heads),
        f"budget shape 错误：{tuple(budgets.shape)}",
    )

    signal = (
        (1.0 - alpha) * h_bar
        + alpha * entropy.float()
    )

    expected_raw = (
        signal
        / signal.sum()
        * float(expected_global_history_total)
    )

    require(
        torch.allclose(
            raw_budgets.float(),
            expected_raw.float(),
            atol=1e-3,
            rtol=1e-5,
        ),
        "global raw budget 不符合归一化公式",
    )

    require(
        int(budgets.sum().item())
        == expected_global_history_total,
        (
            f"global history budget 不守恒："
            f"{int(budgets.sum())} != "
            f"{expected_global_history_total}"
        ),
    )

    layer_history_totals = budgets.sum(dim=-1)

    current_global_total = 0

    for layer_idx, decoder_layer in enumerate(
        llama.layers
    ):
        cluster = decoder_layer.self_attn.kv_cluster

        require(
            cluster.entrokv_scope == "global",
            f"Layer {layer_idx} 不是 global scope",
        )

        require(
            torch.equal(
                cluster.last_history_budgets[0],
                budgets[layer_idx],
            ),
            f"Layer {layer_idx} budget 记录不一致",
        )

        expected_prefill_lens = (
            budgets[layer_idx].to(torch.int64)
            + window_size
        )

        require(
            torch.equal(
                cluster.last_prefill_head_lens.to(
                    torch.int64
                ),
                expected_prefill_lens,
            ),
            (
                f"Layer {layer_idx} prefill lengths "
                "与 quota+window 不一致"
            ),
        )

        current_lens = (
            cluster.head_lens
            .detach()
            .cpu()
            .to(torch.int64)
        )

        increments = (
            current_lens
            - expected_prefill_lens
        )

        require(
            bool(torch.all(increments == decode_count)),
            (
                f"Layer {layer_idx} decode append "
                "数量错误"
            ),
        )

        current_total = int(current_lens.sum().item())
        current_global_total += current_total

        require(
            int(cluster.klen_sum) == current_total,
            f"Layer {layer_idx} klen_sum 错误",
        )

        require(
            int(cluster.cu_klen[-1].item())
            == current_total,
            f"Layer {layer_idx} cu_klen[-1] 错误",
        )

        require(
            FINAL_CACHE_ROWS[layer_idx]
            == current_total,
            f"Layer {layer_idx} flattened rows 错误",
        )

    expected_current_global_total = (
        expected_global_prefill_total
        + num_layers * num_kv_heads * decode_count
    )

    require(
        current_global_total
        == expected_current_global_total,
        (
            f"decode 后 global total 错误："
            f"{current_global_total} != "
            f"{expected_current_global_total}"
        ),
    )

    if alpha == 0.0:
        require(
            bool(
                torch.all(
                    budgets
                    == target_history_capacity
                )
            ),
            "alpha=0 未退化为全局固定预算",
        )
    else:
        require(
            int(layer_history_totals.max())
            > int(layer_history_totals.min()),
            "alpha>0 但不同层总预算仍完全相同",
        )

    print("\n" + "=" * 76)
    print("EntroKV Stage 4A validation summary")
    print("=" * 76)

    print(f"Input length:                 {q_len}")
    print(f"Requested retention ratio:    {ratio}")
    print(
        f"Target final capacity/head:   "
        f"{target_final_capacity}"
    )
    print(
        f"Effective retention ratio:    "
        f"{target_final_capacity / q_len:.6f}"
    )
    print(
        f"Target history capacity/head: "
        f"{target_history_capacity}"
    )
    print(
        f"Budget matrix shape:          "
        f"{tuple(budgets.shape)}"
    )
    print(
        f"Global history total:         "
        f"{int(budgets.sum())}"
    )
    print(
        f"Expected history total:       "
        f"{expected_global_history_total}"
    )
    print(
        f"Global prefill total:         "
        f"{expected_global_prefill_total}"
    )
    print(
        f"Global final total:           "
        f"{current_global_total}"
    )
    print(
        f"Layer history min/max:        "
        f"{int(layer_history_totals.min())} / "
        f"{int(layer_history_totals.max())}"
    )

    selected_layers = [0, 1, 16, 31]

    for layer_idx in selected_layers:
        print(f"\nLayer {layer_idx:02d}")
        print(
            "  entropy: "
            f"{[round(x, 6) for x in entropy[layer_idx].tolist()]}"
        )
        print(
            "  history budgets: "
            f"{budgets[layer_idx].tolist()}"
        )
        print(
            "  layer history total: "
            f"{int(layer_history_totals[layer_idx])}"
        )

    print(
        "\n[PASS] Ratio-based capacity is correct."
    )
    print(
        "[PASS] Global entropy matrix is [32,8]."
    )
    print(
        "[PASS] Budgets are normalized across all "
        "layer-head pairs."
    )
    print(
        "[PASS] Global integer history budget is "
        "exactly conserved."
    )
    print(
        "[PASS] Layer totals can vary under alpha>0."
    )
    print(
        "[PASS] Flattened cache and varlen decode "
        "remain consistent."
    )
    print(
        "[PASS] EntroKV Stage 4A is fully validated."
    )


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model-path",
        default=os.environ.get(
            "MODEL_PATH",
            "/root/autodl-tmp/models/"
            "Meta-Llama-3.1-8B-Instruct",
        ),
    )

    parser.add_argument(
        "--retention-ratio",
        type=float,
        default=0.3,
    )

    parser.add_argument(
        "--window-size",
        type=int,
        default=32,
    )

    parser.add_argument(
        "--alpha",
        type=float,
        default=0.5,
    )

    parser.add_argument(
        "--h-bar",
        type=float,
        default=0.3,
    )

    parser.add_argument(
        "--minimum-input-tokens",
        type=int,
        default=4200,
    )

    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=6,
    )

    parser.add_argument(
        "--debug",
        action="store_true",
    )

    args = parser.parse_args()

    install_cache_recorder()
    replace_llama_adaptive()

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        local_files_only=True,
        use_fast=True,
    )

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        local_files_only=True,
        torch_dtype=torch.bfloat16,
        device_map={"": 0},
        attn_implementation="flash_attention_2",
        low_cpu_mem_usage=True,
    )

    model = config_compress(
        model,
        window_size=args.window_size,
        base_capacity=256,
        kernel_size=7,
        pooling="maxpool",
        floor_alpha=0.5,
        pyram_mode=False,
        beta=20,
        skip=0,
        gqa_support=True,
        gqa_func="mean",
        budget_mode="entrokv",
        entrokv_alpha=args.alpha,
        entrokv_h_bar=args.h_bar,
        entrokv_debug=args.debug,
        entrokv_scope="global",
        capacity_mode="ratio",
        retention_ratio=args.retention_ratio,
    )

    model.eval()
    model.generation_config.temperature = None
    model.generation_config.top_p = None

    input_ids = build_input(
        tokenizer,
        args.minimum_input_tokens,
    )

    input_ids = input_ids.to(
        next(model.parameters()).device
    )

    attention_mask = torch.ones_like(input_ids)

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()

    start = time.perf_counter()

    with torch.inference_mode():
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=args.max_new_tokens,
            min_new_tokens=args.max_new_tokens,
            do_sample=False,
            use_cache=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    response = tokenizer.decode(
        outputs[0, input_ids.shape[-1]:],
        skip_special_tokens=True,
    )

    print(f"\nResponse: {response!r}")
    print(f"Elapsed: {elapsed:.3f} s")
    print(
        "Peak allocated: "
        f"{torch.cuda.max_memory_allocated() / 1024**3:.3f} GB"
    )
    print(
        "Peak reserved:  "
        f"{torch.cuda.max_memory_reserved() / 1024**3:.3f} GB"
    )

    require(
        "ORANGE-739" in response,
        "Secret retrieval failed",
    )

    validate(
        model=model,
        q_len=input_ids.shape[-1],
        ratio=args.retention_ratio,
        window_size=args.window_size,
        alpha=args.alpha,
        h_bar=args.h_bar,
        decode_count=args.max_new_tokens - 1,
    )


if __name__ == "__main__":
    main()
