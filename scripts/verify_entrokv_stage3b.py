#!/usr/bin/env python3

from __future__ import annotations

import argparse
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


def install_cache_row_recorder() -> None:
    """
    记录每层 flattened cache 的最终二维行数。
    """

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


def build_long_input(
    tokenizer,
    minimum_tokens: int,
) -> torch.Tensor:
    repeat_count = 200

    filler = (
        "alpha beta gamma delta epsilon zeta eta theta "
        "iota kappa lambda. "
    )

    while True:
        prompt = (
            "Read the following context carefully.\n\n"
            f"{filler * repeat_count}\n\n"
            "The secret code is ORANGE-739.\n\n"
            f"{filler * repeat_count}\n\n"
            "Question: What is the secret code? "
            "Return only the code."
        )

        messages = [
            {
                "role": "user",
                "content": prompt,
            }
        ]

        input_ids = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
        )

        if input_ids.shape[-1] >= minimum_tokens:
            return input_ids

        repeat_count += 50


def validate_model(
    model,
    capacity: int,
    window_size: int,
    alpha: float,
    h_bar: float,
    expected_decode_count: int,
) -> None:
    num_layers = int(model.config.num_hidden_layers)
    num_query_heads = int(model.config.num_attention_heads)
    num_kv_heads = int(model.config.num_key_value_heads)

    groups = num_query_heads // num_kv_heads
    history_capacity = capacity - window_size

    expected_history_total = (
        num_kv_heads * history_capacity
    )

    expected_prefill_total = (
        expected_history_total
        + num_kv_heads * window_size
    )

    selected_layers = {
        0,
        1,
        num_layers // 2,
        num_layers - 1,
    }

    nonuniform_layers = 0

    print("\n" + "=" * 76)
    print("EntroKV Stage 3B validation summary")
    print("=" * 76)

    for layer_idx, decoder_layer in enumerate(
        model.model.layers
    ):
        cluster = decoder_layer.self_attn.kv_cluster

        require(
            cluster.__class__.__name__ == "EntroKVCluster",
            (
                f"Layer {layer_idx} 未使用 EntroKVCluster，"
                f"实际为 {cluster.__class__.__name__}"
            ),
        )

        query_entropy = cluster.last_query_entropy
        kv_entropy = cluster.last_kv_entropy
        raw_budgets = cluster.last_raw_budgets
        history_budgets = cluster.last_history_budgets
        prefill_head_lens = cluster.last_prefill_head_lens

        require(
            query_entropy is not None,
            f"Layer {layer_idx} 缺少 query entropy",
        )

        require(
            kv_entropy is not None,
            f"Layer {layer_idx} 缺少 KV entropy",
        )

        require(
            raw_budgets is not None,
            f"Layer {layer_idx} 缺少 raw budgets",
        )

        require(
            history_budgets is not None,
            f"Layer {layer_idx} 缺少 integer budgets",
        )

        require(
            tuple(query_entropy.shape)
            == (1, num_query_heads),
            (
                f"Layer {layer_idx} query entropy shape 错误："
                f"{tuple(query_entropy.shape)}"
            ),
        )

        require(
            tuple(kv_entropy.shape)
            == (1, num_kv_heads),
            (
                f"Layer {layer_idx} KV entropy shape 错误："
                f"{tuple(kv_entropy.shape)}"
            ),
        )

        require(
            groups == 4,
            f"预期 GQA groups=4，实际为 {groups}",
        )

        require(
            bool(
                torch.all(
                    (query_entropy >= 0)
                    & (query_entropy <= 1)
                )
            ),
            f"Layer {layer_idx} query entropy 超出 [0,1]",
        )

        require(
            bool(
                torch.all(
                    (kv_entropy >= 0)
                    & (kv_entropy <= 1)
                )
            ),
            f"Layer {layer_idx} KV entropy 超出 [0,1]",
        )

        # 验证真实 raw budget 确实由 EntroKV 公式计算。
        signal = (
            (1.0 - alpha) * h_bar
            + alpha * kv_entropy.float()
        )

        expected_raw = (
            signal
            / signal.sum(dim=-1, keepdim=True)
            * float(expected_history_total)
        )

        require(
            torch.allclose(
                raw_budgets.float(),
                expected_raw.float(),
                atol=1e-4,
                rtol=1e-5,
            ),
            (
                f"Layer {layer_idx} raw budget "
                "不符合 EntroKV allocation formula"
            ),
        )

        require(
            int(history_budgets.sum().item())
            == expected_history_total,
            (
                f"Layer {layer_idx} 历史预算不守恒："
                f"{int(history_budgets.sum())} != "
                f"{expected_history_total}"
            ),
        )

        expected_prefill_lens = (
            history_budgets[0].to(torch.int64)
            + window_size
        )

        require(
            torch.equal(
                prefill_head_lens.to(torch.int64),
                expected_prefill_lens,
            ),
            (
                f"Layer {layer_idx} prefill head_lens "
                "不等于 history budget + window"
            ),
        )

        require(
            int(prefill_head_lens.sum().item())
            == expected_prefill_total,
            (
                f"Layer {layer_idx} prefill 总长度错误："
                f"{int(prefill_head_lens.sum())} != "
                f"{expected_prefill_total}"
            ),
        )

        current_head_lens = (
            cluster.head_lens
            .detach()
            .cpu()
            .to(torch.int64)
        )

        increments = (
            current_head_lens
            - prefill_head_lens.to(torch.int64)
        )

        require(
            bool(torch.all(increments == increments[0])),
            (
                f"Layer {layer_idx} decode 后不同 KV head "
                "增加的 token 数不一致"
            ),
        )

        actual_decode_count = int(increments[0].item())

        require(
            actual_decode_count == expected_decode_count,
            (
                f"Layer {layer_idx} decode count 错误："
                f"{actual_decode_count} != "
                f"{expected_decode_count}"
            ),
        )

        current_total = int(
            current_head_lens.sum().item()
        )

        require(
            int(cluster.klen_sum) == current_total,
            (
                f"Layer {layer_idx} klen_sum 错误："
                f"{cluster.klen_sum} != {current_total}"
            ),
        )

        current_cu_klen = (
            cluster.cu_klen
            .detach()
            .cpu()
            .to(torch.int64)
        )

        require(
            int(current_cu_klen[-1].item())
            == current_total,
            (
                f"Layer {layer_idx} cu_klen[-1] 错误："
                f"{int(current_cu_klen[-1])} != "
                f"{current_total}"
            ),
        )

        require(
            torch.equal(
                current_cu_klen[1:]
                - current_cu_klen[:-1],
                current_head_lens,
            ),
            (
                f"Layer {layer_idx} cu_klen 相邻差值 "
                "不等于 head_lens"
            ),
        )

        require(
            int(cluster.max_seqlen_k)
            == int(current_head_lens.max().item()),
            (
                f"Layer {layer_idx} max_seqlen_k 错误"
            ),
        )

        require(
            layer_idx in FINAL_CACHE_ROWS,
            f"Layer {layer_idx} 没有记录 flattened rows",
        )

        require(
            FINAL_CACHE_ROWS[layer_idx] == current_total,
            (
                f"Layer {layer_idx} flattened rows 错误："
                f"{FINAL_CACHE_ROWS[layer_idx]} != "
                f"{current_total}"
            ),
        )

        if int(history_budgets.min()) != int(
            history_budgets.max()
        ):
            nonuniform_layers += 1

        if layer_idx in selected_layers:
            print(f"Layer {layer_idx:02d}")
            print(
                "  KV entropy:       "
                f"{[round(x, 6) for x in kv_entropy[0].tolist()]}"
            )
            print(
                "  history budgets:  "
                f"{history_budgets[0].tolist()}"
            )
            print(
                "  history total:    "
                f"{int(history_budgets.sum())}"
            )
            print(
                "  prefill lengths:  "
                f"{prefill_head_lens.tolist()}"
            )
            print(
                "  prefill total:    "
                f"{int(prefill_head_lens.sum())}"
            )
            print(
                "  decode count:     "
                f"{actual_decode_count}"
            )
            print(
                "  current total:    "
                f"{current_total}"
            )
            print(
                "  flattened rows:   "
                f"{FINAL_CACHE_ROWS[layer_idx]}"
            )
            print(
                "  cu_klen[-1]:      "
                f"{int(current_cu_klen[-1])}"
            )
            print(
                "  klen_sum:         "
                f"{int(cluster.klen_sum)}"
            )

    if alpha == 0.0:
        require(
            nonuniform_layers == 0,
            (
                "alpha=0 时应全部退化成均匀预算，"
                f"但有 {nonuniform_layers} 层非均匀"
            ),
        )
    else:
        require(
            nonuniform_layers > 0,
            "alpha>0，但所有层预算仍完全均匀",
        )

    print(f"\nValidated layers: {num_layers}")
    print(
        f"Non-uniform layers: {nonuniform_layers}/{num_layers}"
    )
    print(
        f"Expected history total/layer: "
        f"{expected_history_total}"
    )
    print(
        f"Expected prefill total/layer: "
        f"{expected_prefill_total}"
    )
    print(
        f"Expected final total/layer: "
        f"{expected_prefill_total + num_kv_heads * expected_decode_count}"
    )

    print(
        "[PASS] Real attention entropy was computed "
        "for all query heads."
    )
    print(
        "[PASS] 32 query-head entropies were mapped "
        "to 8 physical KV heads."
    )
    print(
        "[PASS] EntroKV raw and integer budgets "
        "are correct."
    )
    print(
        "[PASS] Per-layer total KV budget is "
        "exactly conserved."
    )
    print(
        "[PASS] SnapKV top-k, flattened cache and "
        "varlen decode are consistent."
    )
    print(
        "[PASS] EntroKV Stage 3B integration "
        "is fully validated."
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
        "--capacity",
        type=int,
        default=256,
    )

    parser.add_argument(
        "--window-size",
        type=int,
        default=32,
    )

    parser.add_argument(
        "--kernel-size",
        type=int,
        default=7,
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

    require(
        args.capacity > args.window_size,
        "capacity 必须大于 window_size",
    )

    install_cache_row_recorder()

    # EntroKV 复用 Adaptive Llama 的
    # flattened cache 和 varlen decode 路径。
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
        base_capacity=args.capacity,
        kernel_size=args.kernel_size,
        pooling="maxpool",
        floor_alpha=0.5,   # EntroKV mode 中不参与预算计算
        pyram_mode=False,
        beta=20,
        skip=0,
        gqa_support=True,
        gqa_func="mean",
        budget_mode="entrokv",
        entrokv_alpha=args.alpha,
        entrokv_h_bar=args.h_bar,
        entrokv_debug=args.debug,
    )

    model.eval()

    model.generation_config.temperature = None
    model.generation_config.top_p = None

    input_ids = build_long_input(
        tokenizer,
        minimum_tokens=args.minimum_input_tokens,
    )

    device = next(model.parameters()).device
    input_ids = input_ids.to(device)

    attention_mask = torch.ones_like(input_ids)

    print(f"Model path:   {args.model_path}")
    print(f"Input length: {input_ids.shape[-1]}")
    print(f"Capacity:     {args.capacity}")
    print(f"Window:       {args.window_size}")
    print(f"Alpha:        {args.alpha}")
    print(f"H_bar:        {args.h_bar}")

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()

    start_time = time.perf_counter()

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

    elapsed = time.perf_counter() - start_time

    generated_ids = outputs[
        0,
        input_ids.shape[-1]:,
    ]

    response = tokenizer.decode(
        generated_ids,
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

    if "ORANGE-739" in response:
        print("[PASS] Secret retrieval answer is correct.")
    else:
        print(
            "[WARN] Secret retrieval answer is not exact; "
            "structural validation will continue."
        )

    # 生成 N 个 token 时，第一个 token 来自 prefill logits；
    # 后续 N-1 个 token 才执行 q_len=1 decode forward。
    expected_decode_count = args.max_new_tokens - 1

    validate_model(
        model=model,
        capacity=args.capacity,
        window_size=args.window_size,
        alpha=args.alpha,
        h_bar=args.h_bar,
        expected_decode_count=expected_decode_count,
    )


if __name__ == "__main__":
    main()
