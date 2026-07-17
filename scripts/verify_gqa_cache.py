#!/usr/bin/env python3

import argparse
import os
import time
from collections import defaultdict

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import adaptive_snapkv.monkeypatch.snapkv_utils as snapkv_utils
from adaptive_snapkv.monkeypatch.monkeypatch import (
    config_compress,
    replace_llama_adaptive,
    replace_llama_fixed,
)


# ============================================================
# 全局调试记录
# ============================================================

FIXED_RECORDS = {}

ADAPTIVE_PREFILL_LENS = {}
ADAPTIVE_PREFILL_RECORDS = {}

# 每层真实执行了多少次 q_len=1 decode cache update
DECODE_COUNTS = defaultdict(int)

# DynamicCacheSplitHeadFlatten 中每层最后一次记录到的 flattened 行数
FINAL_CACHE_ROWS = {}


def require(condition: bool, message: str) -> None:
    """比普通 assert 更明确地报告验证失败原因。"""
    if not condition:
        raise AssertionError(message)


# ============================================================
# 阶段一：Fixed SnapKV 验证
# ============================================================

def install_fixed_debug_wrapper() -> None:
    original_update_kv = snapkv_utils.SnapKVCluster.update_kv

    def wrapped_update_kv(
        self,
        origin_key_states: torch.Tensor,
        query_states: torch.Tensor,
        origin_value_states: torch.Tensor,
    ):
        """
        输入：
          query_states:       [bs, 32 query heads, seq_len, head_dim]
          origin_key_states:  [bs,  8 KV heads,    seq_len, head_dim]

        输出：
          compressed K/V:     [bs, 8 KV heads, retained_len, head_dim]
        """

        layer_idx = int(self.layer_idx)

        require(query_states.ndim == 4, "query_states 必须是 4 维")
        require(origin_key_states.ndim == 4, "origin_key_states 必须是 4 维")
        require(origin_value_states.ndim == 4, "origin_value_states 必须是 4 维")

        bsz, query_heads, q_len, head_dim = query_states.shape
        kb, kv_heads, k_len, k_head_dim = origin_key_states.shape

        require(bsz == kb, f"batch size 不一致：Q={bsz}, K={kb}")
        require(q_len == k_len, f"prefill Q/K 长度不一致：Q={q_len}, K={k_len}")
        require(head_dim == k_head_dim, "Q/K head_dim 不一致")

        groups = int(self.num_key_value_groups)

        require(
            query_heads == kv_heads * groups,
            (
                f"GQA 映射错误：query_heads={query_heads}, "
                f"kv_heads={kv_heads}, groups={groups}"
            ),
        )

        # 根据上游代码实际运算推导出的中间 shape。
        # attention: 最近 window 个 query 对所有 key
        window = int(self.window_size)
        historical_len = max(q_len - window, 0)

        raw_attention_shape = (bsz, query_heads, window, q_len)
        query_head_score_shape = (bsz, query_heads, historical_len)
        grouped_score_shape = (bsz, kv_heads, groups, historical_len)
        kv_head_score_shape = (bsz, kv_heads, historical_len)

        # 用一个简单 tensor 验证 view 后每 4 个连续 query heads
        # 正确映射到一个 KV head：
        #
        # KV0 <- Q0,Q1,Q2,Q3
        # KV1 <- Q4,Q5,Q6,Q7
        # ...
        if layer_idx == 0:
            synthetic_query_head_ids = torch.arange(
                query_heads, device=query_states.device, dtype=torch.float32
            ).view(1, query_heads, 1)

            grouped_ids = synthetic_query_head_ids.view(
                1, kv_heads, groups, 1
            )

            grouped_means = grouped_ids.mean(dim=2).view(-1).cpu().tolist()

            print("\n[Fixed / Layer 0] GQA mapping")
            for kv_idx in range(kv_heads):
                q_start = kv_idx * groups
                q_end = q_start + groups - 1
                print(
                    f"  KV head {kv_idx}: "
                    f"query heads {q_start}-{q_end}, "
                    f"synthetic mean={grouped_means[kv_idx]:.1f}"
                )

            print("\n[Fixed / Layer 0] Intermediate shapes")
            print(f"  query_states:          {tuple(query_states.shape)}")
            print(f"  original_key_states:   {tuple(origin_key_states.shape)}")
            print(f"  raw_attention:         {raw_attention_shape}")
            print(f"  query_head_scores:     {query_head_score_shape}")
            print(f"  grouped_scores:        {grouped_score_shape}")
            print(f"  KV_head_scores:        {kv_head_score_shape}")
            print(
                f"  historical quota/head: "
                f"{self.max_capacity_prompt - self.window_size}"
            )
            print(f"  recent window/head:    {self.window_size}")

        # 调用仓库原始 Fixed SnapKV
        output_key_states, output_value_states = original_update_kv(
            self,
            origin_key_states,
            query_states,
            origin_value_states,
        )

        expected_retained_len = (
            q_len
            if q_len < self.max_capacity_prompt
            else self.max_capacity_prompt
        )

        expected_output_shape = (
            bsz,
            kv_heads,
            expected_retained_len,
            head_dim,
        )

        require(
            tuple(output_key_states.shape) == expected_output_shape,
            (
                f"Layer {layer_idx} 压缩后 K shape 错误："
                f"actual={tuple(output_key_states.shape)}, "
                f"expected={expected_output_shape}"
            ),
        )

        require(
            tuple(output_value_states.shape) == expected_output_shape,
            (
                f"Layer {layer_idx} 压缩后 V shape 错误："
                f"actual={tuple(output_value_states.shape)}, "
                f"expected={expected_output_shape}"
            ),
        )

        FIXED_RECORDS[layer_idx] = {
            "query_heads": query_heads,
            "kv_heads": kv_heads,
            "groups": groups,
            "input_len": q_len,
            "output_len": output_key_states.shape[-2],
            "output_shape": tuple(output_key_states.shape),
        }

        return output_key_states, output_value_states

    snapkv_utils.SnapKVCluster.update_kv = wrapped_update_kv


# ============================================================
# 阶段二：Adaptive flattened GQA 验证
# ============================================================

def install_adaptive_debug_wrappers() -> None:
    original_update_kv_gqa = (
        snapkv_utils.AdaptiveSnapKVCluster.update_kv_gqa
    )

    original_flatten_cache_update = (
        snapkv_utils.DynamicCacheSplitHeadFlatten.update
    )

    def wrapped_update_kv_gqa(
        self,
        origin_key_states: torch.Tensor,
        query_states: torch.Tensor,
        origin_value_states: torch.Tensor,
    ):
        layer_idx = int(self.layer_idx)

        bsz, query_heads, q_len, head_dim = query_states.shape
        _, kv_heads, kv_len, kv_head_dim = origin_key_states.shape
        groups = int(self.num_key_value_groups)

        require(bsz == 1, "AdaKV 当前 flattened 实现只支持 batch_size=1")
        require(q_len == kv_len, "prefill Q/K 长度不一致")
        require(head_dim == kv_head_dim, "Q/K head_dim 不一致")
        require(
            query_heads == kv_heads * groups,
            (
                f"Layer {layer_idx} GQA 映射错误："
                f"{query_heads} != {kv_heads} * {groups}"
            ),
        )

        flattened_k, flattened_v = original_update_kv_gqa(
            self,
            origin_key_states,
            query_states,
            origin_value_states,
        )

        require(flattened_k.ndim == 2, "Adaptive K 应为二维 flattened tensor")
        require(flattened_v.ndim == 2, "Adaptive V 应为二维 flattened tensor")
        require(
            flattened_k.shape == flattened_v.shape,
            "flattened K/V shape 不一致",
        )

        head_lens = self.head_lens.detach().cpu().clone()
        cu_klen = self.cu_klen.detach().cpu().clone()

        require(
            head_lens.numel() == kv_heads,
            (
                f"Layer {layer_idx} head_lens 应有 {kv_heads} 项，"
                f"实际为 {head_lens.numel()}"
            ),
        )

        require(
            cu_klen.numel() == kv_heads + 1,
            (
                f"Layer {layer_idx} cu_klen 应有 {kv_heads + 1} 项，"
                f"实际为 {cu_klen.numel()}"
            ),
        )

        retained_sum = int(head_lens.sum().item())

        require(
            flattened_k.shape[0] == retained_sum,
            (
                f"Layer {layer_idx} flattened 行数与 head_lens.sum 不一致："
                f"rows={flattened_k.shape[0]}, sum={retained_sum}"
            ),
        )

        require(
            int(cu_klen[0].item()) == 0,
            f"Layer {layer_idx} cu_klen[0] 必须为 0",
        )

        require(
            int(cu_klen[-1].item()) == retained_sum,
            (
                f"Layer {layer_idx} cu_klen[-1] 错误："
                f"{int(cu_klen[-1])} != {retained_sum}"
            ),
        )

        cu_differences = cu_klen[1:] - cu_klen[:-1]

        require(
            torch.equal(cu_differences.to(torch.int64),
                        head_lens.to(torch.int64)),
            (
                f"Layer {layer_idx} cu_klen 相邻差值 "
                f"不等于 head_lens"
            ),
        )

        require(
            int(self.max_seqlen_k) == int(head_lens.max().item()),
            (
                f"Layer {layer_idx} max_seqlen_k 错误："
                f"{self.max_seqlen_k} != {int(head_lens.max())}"
            ),
        )

        require(
            int(self.klen_sum) == retained_sum,
            (
                f"Layer {layer_idx} prefill klen_sum 错误："
                f"{self.klen_sum} != {retained_sum}"
            ),
        )

        ADAPTIVE_PREFILL_LENS[layer_idx] = head_lens
        ADAPTIVE_PREFILL_RECORDS[layer_idx] = {
            "query_heads": query_heads,
            "kv_heads": kv_heads,
            "groups": groups,
            "input_len": q_len,
            "head_lens": head_lens.tolist(),
            "flattened_rows": flattened_k.shape[0],
            "cu_klen": cu_klen.tolist(),
        }

        return flattened_k, flattened_v

    def wrapped_flatten_cache_update(
        self,
        key_states,
        value_states,
        layer_idx,
        cache_kwargs=None,
    ):
        layer_idx = int(layer_idx)

        # 当 layer 已经存在，并再次输入 [1, kv_heads, 1, dim]，
        # 说明进入 decode cache append。
        is_decode_append = (
            len(self.key_cache) > layer_idx
            and isinstance(key_states, torch.Tensor)
            and key_states.ndim == 4
            and key_states.shape[-2] == 1
        )

        if is_decode_append:
            require(
                cache_kwargs is not None,
                f"Layer {layer_idx} decode 时缺少 cache_kwargs",
            )
            require(
                "head_lens" in cache_kwargs,
                f"Layer {layer_idx} decode 时缺少 head_lens",
            )
            require(
                "cu_klen" in cache_kwargs,
                f"Layer {layer_idx} decode 时缺少 cu_klen",
            )

            DECODE_COUNTS[layer_idx] += 1

        output_k, output_v = original_flatten_cache_update(
            self,
            key_states,
            value_states,
            layer_idx,
            cache_kwargs,
        )

        if isinstance(output_k, torch.Tensor) and output_k.ndim == 2:
            FINAL_CACHE_ROWS[layer_idx] = int(output_k.shape[0])

        return output_k, output_v

    snapkv_utils.AdaptiveSnapKVCluster.update_kv_gqa = (
        wrapped_update_kv_gqa
    )

    snapkv_utils.DynamicCacheSplitHeadFlatten.update = (
        wrapped_flatten_cache_update
    )


# ============================================================
# 结果汇总
# ============================================================

def validate_fixed_summary(model) -> None:
    expected_layers = int(model.config.num_hidden_layers)

    require(
        len(FIXED_RECORDS) == expected_layers,
        (
            f"只验证到 {len(FIXED_RECORDS)} 层，"
            f"模型应该有 {expected_layers} 层"
        ),
    )

    print("\n" + "=" * 72)
    print("Fixed SnapKV validation summary")
    print("=" * 72)

    selected_layers = [0, 1, expected_layers // 2, expected_layers - 1]

    for layer_idx in selected_layers:
        record = FIXED_RECORDS[layer_idx]
        print(
            f"Layer {layer_idx:02d}: "
            f"Q-heads={record['query_heads']}, "
            f"KV-heads={record['kv_heads']}, "
            f"groups={record['groups']}, "
            f"input={record['input_len']}, "
            f"retained={record['output_len']}, "
            f"shape={record['output_shape']}"
        )

    all_output_lengths = {
        record["output_len"] for record in FIXED_RECORDS.values()
    }

    require(
        len(all_output_lengths) == 1,
        f"不同层 Fixed cache length 不一致：{all_output_lengths}",
    )

    print(f"\nValidated layers: {len(FIXED_RECORDS)}")
    print(f"Uniform retained length: {next(iter(all_output_lengths))}")
    print("[PASS] Fixed SnapKV GQA mapping and cache length are correct.")


def validate_adaptive_summary(model) -> None:
    expected_layers = int(model.config.num_hidden_layers)
    expected_kv_heads = int(model.config.num_key_value_heads)

    require(
        len(ADAPTIVE_PREFILL_LENS) == expected_layers,
        (
            f"只验证到 {len(ADAPTIVE_PREFILL_LENS)} 层，"
            f"模型应该有 {expected_layers} 层"
        ),
    )

    print("\n" + "=" * 72)
    print("Adaptive SnapKV validation summary")
    print("=" * 72)

    klen_sum_mismatches = []

    selected_layers = [0, 1, expected_layers // 2, expected_layers - 1]

    for layer_idx, decoder_layer in enumerate(model.model.layers):
        cluster = decoder_layer.self_attn.kv_cluster

        prefill_head_lens = ADAPTIVE_PREFILL_LENS[layer_idx]
        decode_count = int(DECODE_COUNTS[layer_idx])

        current_head_lens = (
            cluster.head_lens.detach().cpu().to(torch.int64)
        )

        require(
            current_head_lens.numel() == expected_kv_heads,
            (
                f"Layer {layer_idx} decode 后 head_lens 数量错误："
                f"{current_head_lens.numel()} != {expected_kv_heads}"
            ),
        )

        expected_current_lens = (
            prefill_head_lens.to(torch.int64) + decode_count
        )

        require(
            torch.equal(current_head_lens, expected_current_lens),
            (
                f"Layer {layer_idx} decode 后 head_lens 更新错误\n"
                f"prefill={prefill_head_lens.tolist()}\n"
                f"decode_count={decode_count}\n"
                f"actual={current_head_lens.tolist()}\n"
                f"expected={expected_current_lens.tolist()}"
            ),
        )

        current_sum = int(current_head_lens.sum().item())

        require(
            layer_idx in FINAL_CACHE_ROWS,
            f"Layer {layer_idx} 没有记录 flattened cache 行数",
        )

        require(
            FINAL_CACHE_ROWS[layer_idx] == current_sum,
            (
                f"Layer {layer_idx} 最终 flattened 行数错误："
                f"{FINAL_CACHE_ROWS[layer_idx]} != {current_sum}"
            ),
        )

        current_cu = cluster.cu_klen.detach().cpu().to(torch.int64)
        expected_cu = torch.cat(
            [
                torch.zeros(1, dtype=torch.int64),
                torch.cumsum(current_head_lens, dim=0),
            ],
            dim=0,
        )

        require(
            torch.equal(current_cu, expected_cu),
            (
                f"Layer {layer_idx} decode 后 cu_klen 错误\n"
                f"actual={current_cu.tolist()}\n"
                f"expected={expected_cu.tolist()}"
            ),
        )

        # 上游 Llama Adaptive 路径中很可能存在：
        # klen_sum += self.num_heads
        #
        # 对 GQA 来说物理 KV heads 是 8，不应该增加 32。
        if int(cluster.klen_sum) != current_sum:
            klen_sum_mismatches.append(
                {
                    "layer": layer_idx,
                    "actual": int(cluster.klen_sum),
                    "expected": current_sum,
                    "difference": int(cluster.klen_sum) - current_sum,
                    "decode_count": decode_count,
                }
            )

        if layer_idx in selected_layers:
            print(
                f"Layer {layer_idx:02d}: "
                f"prefill head_lens="
                f"{ADAPTIVE_PREFILL_LENS[layer_idx].tolist()}"
            )
            print(
                f"          decode_count={decode_count}, "
                f"current head_lens={current_head_lens.tolist()}"
            )
            print(
                f"          flattened_rows={FINAL_CACHE_ROWS[layer_idx]}, "
                f"cu_klen[-1]={int(current_cu[-1])}, "
                f"klen_sum={int(cluster.klen_sum)}"
            )

    print(f"\nValidated layers: {expected_layers}")
    print(
        "[PASS] head_lens, flattened rows and cu_klen "
        "are internally consistent."
    )

    if klen_sum_mismatches:
        print("\n[WARN] Detected klen_sum metadata mismatch.")
        print(
            "This matches the suspected GQA bug in "
            "adaptive_llama_hijack.py."
        )

        for item in klen_sum_mismatches[:4]:
            print(
                f"  Layer {item['layer']:02d}: "
                f"klen_sum={item['actual']}, "
                f"expected={item['expected']}, "
                f"difference={item['difference']}, "
                f"decode_count={item['decode_count']}"
            )

        print(
            f"  Total affected layers: {len(klen_sum_mismatches)}"
        )
    else:
        print("[PASS] klen_sum is also consistent.")
        print("[PASS] Adaptive flattened GQA path is fully validated.")


# ============================================================
# 构造长输入
# ============================================================

def build_long_input(tokenizer, minimum_tokens: int) -> torch.Tensor:
    repeat_count = 200
    filler_unit = (
        "alpha beta gamma delta epsilon zeta eta theta "
        "iota kappa lambda. "
    )

    while True:
        left_filler = filler_unit * repeat_count
        right_filler = filler_unit * repeat_count

        prompt = (
            "Read the following context carefully.\n\n"
            f"{left_filler}\n\n"
            "The secret code is ORANGE-739.\n\n"
            f"{right_filler}\n\n"
            "Question: What is the secret code? "
            "Return only the code."
        )

        messages = [{"role": "user", "content": prompt}]

        input_ids = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
        )

        if input_ids.shape[-1] >= minimum_tokens:
            return input_ids

        repeat_count += 50


# ============================================================
# 主程序
# ============================================================

def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--mode",
        choices=["fixed", "adaptive"],
        required=True,
    )

    parser.add_argument(
        "--model-path",
        default=os.environ.get(
            "MODEL_PATH",
            "/root/autodl-tmp/models/Meta-Llama-3.1-8B-Instruct",
        ),
    )

    parser.add_argument("--capacity", type=int, default=256)
    parser.add_argument("--window-size", type=int, default=32)
    parser.add_argument("--kernel-size", type=int, default=7)
    parser.add_argument("--minimum-input-tokens", type=int, default=4200)
    parser.add_argument("--max-new-tokens", type=int, default=6)

    args = parser.parse_args()

    require(
        args.capacity > args.window_size,
        "capacity 必须大于 observation window",
    )

    print(f"Mode: {args.mode}")
    print(f"Model path: {args.model_path}")
    print(f"Capacity: {args.capacity}")
    print(f"Window size: {args.window_size}")

    # 必须在模型加载前替换 Transformers 类方法
    if args.mode == "fixed":
        install_fixed_debug_wrapper()
        replace_llama_fixed()
    else:
        install_adaptive_debug_wrappers()
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
        floor_alpha=0.5,
        pyram_mode=False,
        beta=20,
        skip=0,
        gqa_support=True,
        gqa_func="mean",
    )

    model.eval()

    # 清理 greedy decoding 下无效的 generation warning
    model.generation_config.temperature = None
    model.generation_config.top_p = None

    input_ids = build_long_input(
        tokenizer,
        minimum_tokens=args.minimum_input_tokens,
    )

    device = next(model.parameters()).device
    input_ids = input_ids.to(device)
    attention_mask = torch.ones_like(input_ids)

    print(f"Input length: {input_ids.shape[-1]} tokens")

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

    generated_ids = outputs[0, input_ids.shape[-1]:]
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

    if args.mode == "fixed":
        validate_fixed_summary(model)
    else:
        validate_adaptive_summary(model)


if __name__ == "__main__":
    main()
