#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import json
import math
import os
import random
import subprocess
import sys
import time
import types
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import numpy as np
import torch
import transformers
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from adaptive_snapkv.monkeypatch.monkeypatch import (  # noqa: E402
    config_compress,
    replace_llama_adaptive,
    replace_llama_fixed,
)

DEFAULT_TASKS = ["qasper", "hotpotqa", "passage_retrieval_en"]
NO_CHAT_TASKS = {"trec", "triviaqa", "samsum", "lsht", "lcc", "repobench-p"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Unified LongBench runner for Full Cache, Fixed SnapKV and EntroKV."
    )
    parser.add_argument("--method", choices=["full", "fixed", "entrokv"], required=True)
    parser.add_argument(
        "--model-path",
        default=os.environ.get(
            "MODEL_PATH", "/root/autodl-tmp/models/Meta-Llama-3.1-8B-Instruct"
        ),
    )
    parser.add_argument(
        "--dataset-source",
        default=os.environ.get(
            "LONGBENCH_PATH", "zai-org/LongBench"
        ),
        help="Local LongBench repository/dataset path or a datasets hub id.",
    )
    parser.add_argument("--tasks", nargs="+", default=DEFAULT_TASKS)
    parser.add_argument("--output-root", default="outputs/longbench_stage4b")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--sample-start", type=int, default=0)
    parser.add_argument("--sample-limit", type=int, default=1)
    parser.add_argument("--max-input-tokens", type=int, default=8192)
    parser.add_argument("--retention-ratio", type=float, default=0.30)
    parser.add_argument("--window-size", type=int, default=32)
    parser.add_argument("--kernel-size", type=int, default=7)
    parser.add_argument("--pooling", choices=["maxpool", "avgpool"], default="maxpool")
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--h-bar", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--longbench-e", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument(
        "--save-allocation",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--local-files-only",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def git_commit() -> Optional[str]:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
        ).strip()
    except Exception:
        return None


def auto_run_name(args: argparse.Namespace) -> str:
    if args.method == "full":
        return "full"
    if args.method == "fixed":
        return f"fixed_r{args.retention_ratio:.2f}"
    return f"entrokv_r{args.retention_ratio:.2f}_a{args.alpha:.2f}"


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    tmp.replace(path)


def append_jsonl(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False)
        f.write("\n")
        f.flush()


def completed_indices(path: Path) -> set[int]:
    done: set[int] = set()
    if not path.exists():
        return done
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("status") == "ok":
                done.add(int(record["sample_index"]))
    return done


def install_method(method: str) -> None:
    # Run exactly one method per process. These functions monkeypatch global
    # Transformers classes and should not be switched inside one Python process.
    if method == "fixed":
        replace_llama_fixed()
    elif method == "entrokv":
        replace_llama_adaptive()


def load_model_and_tokenizer(args: argparse.Namespace):
    install_method(args.method)

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        local_files_only=args.local_files_only,
        use_fast=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        local_files_only=args.local_files_only,
        torch_dtype=torch.bfloat16,
        device_map={"": 0},
        attn_implementation="flash_attention_2",
        low_cpu_mem_usage=True,
    )

    if model.config.model_type != "llama":
        raise ValueError(
            f"Stage 4B currently validates Llama only; got {model.config.model_type!r}."
        )

    if args.method == "fixed":
        config_compress(
            model,
            window_size=args.window_size,
            base_capacity=256,  # ignored by ratio mode after each sample length is known
            kernel_size=args.kernel_size,
            pooling=args.pooling,
            floor_alpha=0.5,
            pyram_mode=False,
            skip=0,
            gqa_support=True,
            gqa_func="mean",
            capacity_mode="ratio",
            retention_ratio=args.retention_ratio,
        )
    elif args.method == "entrokv":
        config_compress(
            model,
            window_size=args.window_size,
            base_capacity=256,
            kernel_size=args.kernel_size,
            pooling=args.pooling,
            floor_alpha=0.5,
            pyram_mode=False,
            skip=0,
            gqa_support=True,
            gqa_func="mean",
            budget_mode="entrokv",
            entrokv_alpha=args.alpha,
            entrokv_h_bar=args.h_bar,
            entrokv_debug=False,
            entrokv_scope="global",
            capacity_mode="ratio",
            retention_ratio=args.retention_ratio,
        )

    model.eval()
    model.generation_config.temperature = None
    model.generation_config.top_p = None
    return model, tokenizer


def load_longbench_task(source: str, task: str, longbench_e: bool):
    config_name = f"{task}_e" if longbench_e else task
    source_path = Path(source).expanduser()

    if source_path.exists():
        data_dir = source_path / "data"
        kwargs: Dict[str, Any] = {}
        if data_dir.exists():
            kwargs["data_dir"] = str(data_dir)
        return load_dataset(str(source_path), config_name, split="test", **kwargs)

    return load_dataset(source, config_name, split="test")


def task_indices(dataset_size: int, start: int, limit: int) -> list[int]:
    if start < 0 or start >= dataset_size:
        raise ValueError(f"sample-start={start} is outside dataset size {dataset_size}")
    if limit <= 0:
        end = dataset_size
    else:
        end = min(dataset_size, start + limit)
    return list(range(start, end))


def encode_prompt(
    tokenizer,
    prompt: str,
    task: str,
    max_input_tokens: int,
) -> tuple[torch.Tensor, int, bool]:
    if task in NO_CHAT_TASKS:
        ids = tokenizer(
            prompt,
            add_special_tokens=True,
            truncation=False,
            return_tensors="pt",
        ).input_ids[0]
    else:
        chat_text = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        ids = tokenizer(
            chat_text,
            add_special_tokens=False,
            truncation=False,
            return_tensors="pt",
        ).input_ids[0]

    raw_len = int(ids.numel())
    truncated = raw_len > max_input_tokens
    if truncated:
        left = (max_input_tokens + 1) // 2
        right = max_input_tokens - left
        ids = torch.cat([ids[:left], ids[-right:]], dim=0)

    return ids.unsqueeze(0), raw_len, truncated


def infer_q_len(args: tuple[Any, ...], kwargs: Dict[str, Any]) -> Optional[int]:
    tensor = kwargs.get("input_ids")
    if tensor is None:
        tensor = kwargs.get("inputs_embeds")
    if tensor is None and args and isinstance(args[0], torch.Tensor):
        tensor = args[0]
    if not isinstance(tensor, torch.Tensor) or tensor.ndim < 2:
        return None
    return int(tensor.shape[1])


class CudaForwardRecorder:
    def __init__(self, backbone) -> None:
        self.backbone = backbone
        self.original_forward = backbone.forward
        self.records: list[tuple[Optional[int], torch.cuda.Event, torch.cuda.Event]] = []

        def wrapped(_module, *args, **kwargs):
            q_len = infer_q_len(args, kwargs)
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            output = self.original_forward(*args, **kwargs)
            end.record()
            self.records.append((q_len, start, end))
            return output

        backbone.forward = types.MethodType(wrapped, backbone)

    def reset(self) -> None:
        self.records.clear()

    def summarize(self) -> Dict[str, Any]:
        torch.cuda.synchronize()
        rows = [
            (q_len, float(start.elapsed_time(end)))
            for q_len, start, end in self.records
        ]
        prefill_times = [ms for q_len, ms in rows if q_len is not None and q_len > 1]
        decode_times = [ms for q_len, ms in rows if q_len == 1]
        return {
            "forward_calls": len(rows),
            "prefill_forward_calls": len(prefill_times),
            "decode_forward_calls": len(decode_times),
            "prefill_gpu_ms": float(sum(prefill_times)),
            "decode_gpu_ms": float(sum(decode_times)),
            "decode_gpu_ms_per_forward": (
                float(sum(decode_times) / len(decode_times)) if decode_times else 0.0
            ),
        }


def post_process(prediction: str, task: str) -> str:
    if task in {"trec", "triviaqa", "samsum", "lsht"}:
        return prediction.lstrip("\n").split("\n")[0]
    return prediction


def tensor_to_list(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    return value


def collect_cache_stats(
    model,
    method: str,
    input_tokens: int,
    decode_forward_calls: int,
    retention_ratio: float,
    save_allocation: bool,
) -> Dict[str, Any]:
    num_layers = int(model.config.num_hidden_layers)
    num_kv_heads = int(model.config.num_key_value_heads)
    head_dim = int(model.config.hidden_size // model.config.num_attention_heads)

    entropy = None
    budgets = None
    layer_history_totals = None

    if method == "full":
        target_final_capacity = input_tokens
        target_history_capacity = input_tokens
        effective_retention_ratio = 1.0
        prefill_cache_entries = num_layers * num_kv_heads * input_tokens
    elif method == "fixed":
        clusters = [layer.self_attn.kv_cluster for layer in model.model.layers]
        target_caps = {int(cluster.last_target_final_capacity) for cluster in clusters}
        history_caps = {int(cluster.last_target_history_capacity) for cluster in clusters}
        effective_ratios = {
            round(float(cluster.last_effective_retention_ratio), 12)
            for cluster in clusters
        }
        if len(target_caps) != 1 or len(history_caps) != 1 or len(effective_ratios) != 1:
            raise AssertionError("Fixed SnapKV layers disagree on ratio capacity metadata")
        target_final_capacity = target_caps.pop()
        target_history_capacity = history_caps.pop()
        effective_retention_ratio = effective_ratios.pop()
        prefill_cache_entries = num_layers * num_kv_heads * target_final_capacity
    else:
        llama = model.model
        target_final_capacity = int(llama.entrokv_last_target_final_capacity)
        target_history_capacity = int(llama.entrokv_last_target_history_capacity)
        effective_retention_ratio = target_final_capacity / input_tokens
        prefill_cache_entries = int(llama.entrokv_last_global_prefill_total)
        budget_tensor = llama.entrokv_last_global_budgets
        entropy_tensor = llama.entrokv_last_global_entropy
        expected_shape = (num_layers, num_kv_heads)
        if tuple(budget_tensor.shape) != expected_shape:
            raise AssertionError(
                f"EntroKV budget shape {tuple(budget_tensor.shape)} != {expected_shape}"
            )
        expected_history_total = num_layers * num_kv_heads * target_history_capacity
        if int(budget_tensor.sum().item()) != expected_history_total:
            raise AssertionError("EntroKV global history budget is not conserved")
        layer_history_totals = budget_tensor.sum(dim=-1)
        if save_allocation:
            budgets = tensor_to_list(budget_tensor)
            entropy = tensor_to_list(entropy_tensor)

    final_cache_entries = (
        prefill_cache_entries + num_layers * num_kv_heads * decode_forward_calls
    )
    dtype_bytes = 2  # bfloat16
    estimated_final_kv_bytes = final_cache_entries * 2 * head_dim * dtype_bytes

    return {
        "requested_retention_ratio": (
            None if method == "full" else float(retention_ratio)
        ),
        "effective_retention_ratio": float(effective_retention_ratio),
        "target_final_capacity_per_head": int(target_final_capacity),
        "target_history_capacity_per_head": int(target_history_capacity),
        "prefill_cache_entries": int(prefill_cache_entries),
        "final_cache_entries": int(final_cache_entries),
        "estimated_final_kv_gb": float(estimated_final_kv_bytes / 1024**3),
        "entropy": entropy,
        "history_budgets": budgets,
        "layer_history_totals": tensor_to_list(layer_history_totals),
    }


def run_generation(
    model,
    tokenizer,
    recorder: CudaForwardRecorder,
    input_ids: torch.Tensor,
    max_new_tokens: int,
) -> tuple[str, int, float, Dict[str, Any], float, float]:
    device = next(model.parameters()).device
    input_ids = input_ids.to(device)
    attention_mask = torch.ones_like(input_ids)

    recorder.reset()
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()

    wall_start = time.perf_counter()
    with torch.inference_mode():
        output = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            num_beams=1,
            do_sample=False,
            use_cache=True,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    torch.cuda.synchronize()
    wall_seconds = time.perf_counter() - wall_start

    timing = recorder.summarize()
    peak_allocated_gb = torch.cuda.max_memory_allocated() / 1024**3
    peak_reserved_gb = torch.cuda.max_memory_reserved() / 1024**3

    generated_ids = output[0, input_ids.shape[-1] :]
    prediction = tokenizer.decode(generated_ids, skip_special_tokens=True)
    generated_tokens = int(generated_ids.numel())

    del output, attention_mask, input_ids, generated_ids
    return (
        prediction,
        generated_tokens,
        wall_seconds,
        timing,
        float(peak_allocated_gb),
        float(peak_reserved_gb),
    )


def main() -> None:
    args = parse_args()
    if not 0.0 < args.retention_ratio <= 1.0:
        raise ValueError("retention-ratio must be in (0,1]")
    if args.max_input_tokens <= args.window_size:
        raise ValueError("max-input-tokens must be larger than window-size")

    seed_everything(args.seed)
    run_name = args.run_name or auto_run_name(args)
    output_dir = (REPO_ROOT / args.output_root / run_name).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    config_dir = THIS_DIR / "config"
    prompt_map = load_json(config_dir / "dataset2prompt.json")
    max_gen_map = load_json(config_dir / "dataset2maxlen.json")

    unknown_tasks = [task for task in args.tasks if task not in prompt_map]
    if unknown_tasks:
        raise KeyError(f"Unknown tasks: {unknown_tasks}")

    run_config = {
        "schema_version": 1,
        "run_name": run_name,
        "method": args.method,
        "model_path": args.model_path,
        "dataset_source": args.dataset_source,
        "tasks": args.tasks,
        "sample_start": args.sample_start,
        "sample_limit": args.sample_limit,
        "max_input_tokens": args.max_input_tokens,
        "retention_ratio": args.retention_ratio,
        "window_size": args.window_size,
        "kernel_size": args.kernel_size,
        "pooling": args.pooling,
        "alpha": args.alpha,
        "h_bar": args.h_bar,
        "seed": args.seed,
        "longbench_e": args.longbench_e,
        "git_commit": git_commit(),
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "cuda_version": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
    }
    config_path = output_dir / "run_config.json"
    if config_path.exists() and not args.overwrite:
        old_config = load_json(config_path)
        comparable_keys = [
            "method",
            "model_path",
            "dataset_source",
            "max_input_tokens",
            "retention_ratio",
            "window_size",
            "kernel_size",
            "pooling",
            "alpha",
            "h_bar",
            "longbench_e",
        ]
        mismatches = {
            key: (old_config.get(key), run_config.get(key))
            for key in comparable_keys
            if old_config.get(key) != run_config.get(key)
        }
        if mismatches:
            raise RuntimeError(
                f"Existing run_config conflicts with current arguments: {mismatches}"
            )
    else:
        write_json(config_path, run_config)

    print(json.dumps(run_config, ensure_ascii=False, indent=2))
    model, tokenizer = load_model_and_tokenizer(args)
    recorder = CudaForwardRecorder(model.model)

    for task in args.tasks:
        dataset = load_longbench_task(args.dataset_source, task, args.longbench_e)
        indices = task_indices(len(dataset), args.sample_start, args.sample_limit)
        out_path = output_dir / f"{task}.jsonl"
        error_path = output_dir / f"{task}.errors.jsonl"
        if args.overwrite:
            out_path.unlink(missing_ok=True)
            error_path.unlink(missing_ok=True)
        done = completed_indices(out_path)

        print(f"\n[{run_name}] task={task}, selected={indices}, completed={sorted(done)}")
        for sample_index in tqdm(indices, desc=f"{run_name}:{task}"):
            if sample_index in done:
                continue
            sample = dataset[sample_index]
            try:
                prompt = prompt_map[task].format(**sample)
                input_ids, raw_input_tokens, truncated = encode_prompt(
                    tokenizer,
                    prompt,
                    task,
                    args.max_input_tokens,
                )
                input_tokens = int(input_ids.shape[-1])
                max_new_tokens = int(max_gen_map[task])

                (
                    prediction,
                    generated_tokens,
                    wall_seconds,
                    timing,
                    peak_allocated_gb,
                    peak_reserved_gb,
                ) = run_generation(
                    model,
                    tokenizer,
                    recorder,
                    input_ids,
                    max_new_tokens,
                )
                prediction = post_process(prediction, task)

                cache_stats = collect_cache_stats(
                    model=model,
                    method=args.method,
                    input_tokens=input_tokens,
                    decode_forward_calls=int(timing["decode_forward_calls"]),
                    retention_ratio=args.retention_ratio,
                    save_allocation=args.save_allocation,
                )

                record = {
                    "schema_version": 1,
                    "status": "ok",
                    "run_name": run_name,
                    "method": args.method,
                    "task": task,
                    "sample_index": sample_index,
                    "prediction": prediction,
                    "answers": sample["answers"],
                    "all_classes": sample.get("all_classes", []),
                    "benchmark_length": sample.get("length"),
                    "raw_input_tokens": raw_input_tokens,
                    "input_tokens": input_tokens,
                    "truncated": truncated,
                    "max_input_tokens": args.max_input_tokens,
                    "max_new_tokens": max_new_tokens,
                    "generated_tokens": generated_tokens,
                    "wall_seconds": wall_seconds,
                    "peak_allocated_gb": peak_allocated_gb,
                    "peak_reserved_gb": peak_reserved_gb,
                    **timing,
                    **cache_stats,
                }
                append_jsonl(out_path, record)
            except Exception as exc:
                error_record = {
                    "schema_version": 1,
                    "status": "error",
                    "run_name": run_name,
                    "method": args.method,
                    "task": task,
                    "sample_index": sample_index,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
                append_jsonl(error_path, error_record)
                if not args.continue_on_error:
                    raise
            finally:
                gc.collect()
                torch.cuda.empty_cache()

    print(f"\n[PASS] Stage 4B runner completed: {output_dir}")


if __name__ == "__main__":
    main()
