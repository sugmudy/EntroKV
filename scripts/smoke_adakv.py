import argparse
import os
import time

import torch


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=["full", "fix"],
        required=True,
    )
    parser.add_argument(
        "--capacity",
        type=int,
        default=256,
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # 必须在加载模型前 monkeypatch
    if args.mode == "fix":
        from adaptive_snapkv.monkeypatch.monkeypatch import (
            config_compress,
            replace_llama_fixed,
        )

        replace_llama_fixed()

    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
    )

    model_path = os.environ["MODEL_PATH"]

    print("Mode:", args.mode)
    print("Model path:", model_path)

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map={"": 0},
        attn_implementation="flash_attention_2",
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    ).eval()

    if args.mode == "fix":
        model = config_compress(
            model,
            window_size=32,
            base_capacity=args.capacity,
            kernel_size=7,
            pooling="maxpool",
            floor_alpha=0.5,
            pyram_mode=False,
            gqa_support=True,
            gqa_func="mean",
        )

    filler = (
        "This paragraph contains ordinary background information "
        "about cities, books, science, weather, music, and daily life. "
    )

    context = filler * 180

    messages = [
        {
            "role": "user",
            "content": (
                context
                + "\nImportant fact: the secret code is ORANGE-739.\n"
                + filler * 20
                + "\nQuestion: What is the secret code? "
                  "Answer with only the code."
            ),
        }
    ]

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        add_special_tokens=False,
    ).to("cuda")

    context_length = inputs.input_ids.shape[-1]

    print("Context length:", context_length)

    if args.mode == "fix":
        print("SnapKV capacity:", args.capacity)
        print("Observation window:", 32)

        if context_length <= args.capacity:
            raise RuntimeError(
                "输入没有超过 capacity，压缩不会被触发。"
            )

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()

    start = time.perf_counter()

    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=24,
            do_sample=False,
            num_beams=1,
            use_cache=True,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
        )

    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    response = tokenizer.decode(
        outputs[0, context_length:],
        skip_special_tokens=True,
    )

    peak_allocated = (
        torch.cuda.max_memory_allocated() / 1024**3
    )
    peak_reserved = (
        torch.cuda.max_memory_reserved() / 1024**3
    )

    print("\n===== Result =====")
    print("Mode:", args.mode)
    print("Response:", response)
    print(f"Elapsed: {elapsed:.3f} s")
    print(f"Peak allocated: {peak_allocated:.3f} GB")
    print(f"Peak reserved: {peak_reserved:.3f} GB")


if __name__ == "__main__":
    main()
