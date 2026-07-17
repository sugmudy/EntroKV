#!/usr/bin/env python3

from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]

MONKEYPATCH_PATH = (
    ROOT
    / "adaptive_snapkv"
    / "monkeypatch"
    / "monkeypatch.py"
)

SNAPKV_UTILS_PATH = (
    ROOT
    / "adaptive_snapkv"
    / "monkeypatch"
    / "snapkv_utils.py"
)


def replace_once(
    text: str,
    old: str,
    new: str,
    description: str,
) -> str:
    count = text.count(old)

    if count != 1:
        raise RuntimeError(
            f"{description}: 预期找到 1 处，实际找到 {count} 处"
        )

    return text.replace(old, new, 1)


def patch_monkeypatch() -> None:
    text = MONKEYPATCH_PATH.read_text(encoding="utf-8")

    if "entrokv_h_bar" in text:
        print("monkeypatch.py already patched")
        return

    old_signature = (
        'def config_compress(model, window_size=32, '
        'base_capacity=1024, kernel_size=7, '
        'pooling="maxpool", floor_alpha=0.5, '
        'pyram_mode = False, beta = 20, skip=0, '
        'gqa_support=False,gqa_func="mean"):'
    )

    new_signature = '''def config_compress(
    model,
    window_size=32,
    base_capacity=1024,
    kernel_size=7,
    pooling="maxpool",
    floor_alpha=0.5,
    pyram_mode=False,
    beta=20,
    skip=0,
    gqa_support=False,
    gqa_func="mean",
    budget_mode="adakv",
    entrokv_alpha=0.5,
    entrokv_h_bar=0.3,
    entrokv_debug=False,
):'''

    text = replace_once(
        text,
        old_signature,
        new_signature,
        "修改 config_compress 函数签名",
    )

    old_config_tail = '''    model.model.config.gqa_support = gqa_support
    model.model.config.gqa_func = gqa_func

    return model
'''

    new_config_tail = '''    model.model.config.gqa_support = gqa_support
    model.model.config.gqa_func = gqa_func

    if budget_mode not in {"adakv", "entrokv"}:
        raise ValueError(
            f"budget_mode must be 'adakv' or 'entrokv', "
            f"got {budget_mode!r}"
        )

    model.model.config.budget_mode = budget_mode
    model.model.config.entrokv_alpha = entrokv_alpha
    model.model.config.entrokv_h_bar = entrokv_h_bar
    model.model.config.entrokv_debug = entrokv_debug

    return model
'''

    text = replace_once(
        text,
        old_config_tail,
        new_config_tail,
        "添加 EntroKV 配置字段",
    )

    MONKEYPATCH_PATH.write_text(
        text,
        encoding="utf-8",
    )

    print("Patched:", MONKEYPATCH_PATH)


def patch_snapkv_utils() -> None:
    text = SNAPKV_UTILS_PATH.read_text(encoding="utf-8")

    if "cluster_cls = EntroKVCluster" in text:
        print("snapkv_utils.py already patched")
        return

    new_function = '''def init_adaptive_snapkv(self):
    assert hasattr(self.config, "window_size"), "window_size not set"
    assert hasattr(self.config, "kernel_size"), "kernel_size not set"
    assert hasattr(self.config, "pooling"), "pooling not set"
    assert hasattr(self.config, "base_capacity"), "base_capacity not set"
    assert hasattr(self.config, "floor_alpha"), "floor_alpha not set"
    assert self.config.floor_alpha is not None

    if not hasattr(self, "kv_cluster"):
        budget_mode = getattr(
            self.config,
            "budget_mode",
            "adakv",
        )

        common_kwargs = dict(
            window_size=self.config.window_size,
            base_capacity=self.config.base_capacity,
            kernel_size=self.config.kernel_size,
            pooling=self.config.pooling,
            floor_alpha=self.config.floor_alpha,
            skip=self.config.skip,
            layer_idx=self.layer_idx,
            normalize=self.config.normalize,
            num_hidden_layers=self.config.num_hidden_layers,
            pyram_mode=self.config.pyram_mode,
            pyram_beta=self.config.pyram_beta,
            gqa_support=self.config.gqa_support,
            num_key_value_groups=(
                self.config.num_attention_heads
                // self.config.num_key_value_heads
            ),
            gqa_func=self.config.gqa_func,
        )

        if budget_mode == "entrokv":
            # Local import avoids a module-level circular import:
            # entrokv_cluster imports AdaptiveSnapKVCluster
            # from this file.
            from adaptive_snapkv.monkeypatch.entrokv_cluster import (
                EntroKVCluster,
            )

            cluster_cls = EntroKVCluster

            common_kwargs.update(
                entrokv_alpha=getattr(
                    self.config,
                    "entrokv_alpha",
                    0.5,
                ),
                entrokv_h_bar=getattr(
                    self.config,
                    "entrokv_h_bar",
                    0.3,
                ),
                entrokv_debug=getattr(
                    self.config,
                    "entrokv_debug",
                    False,
                ),
            )

        elif budget_mode == "adakv":
            cluster_cls = AdaptiveSnapKVCluster

        else:
            raise ValueError(
                f"Unsupported budget_mode={budget_mode!r}"
            )

        self.kv_cluster = cluster_cls(**common_kwargs)

        if self.config.gqa_support:
            if self.config.model_type != "mistral":
                warnings.warn(
                    "GQA currently supports only for "
                    "mistral-7B-v0.2 model"
                )

        print(
            f"Compress config({budget_mode}): "
            f"window_size={self.kv_cluster.window_size}, "
            f"base_capacity={self.kv_cluster.base_capacity}, "
            f"kernel_size={self.kv_cluster.kernel_size}, "
            f"pooling={self.kv_cluster.pooling}, "
            f"floor_alpha={self.kv_cluster.floor_ratio}, "
            f"pyram_mode={self.kv_cluster.pyram_mode}, "
            f"alpha={getattr(self.kv_cluster, 'entrokv_alpha', None)}, "
            f"h_bar={getattr(self.kv_cluster, 'entrokv_h_bar', None)}",
            flush=True,
        )
'''

    pattern = re.compile(
        r"def init_adaptive_snapkv\(self\):.*?"
        r"(?=\n\n\nclass StreamingLLMKVCluster)",
        flags=re.DOTALL,
    )

    text, replacement_count = pattern.subn(
        new_function,
        text,
        count=1,
    )

    if replacement_count != 1:
        raise RuntimeError(
            "替换 init_adaptive_snapkv 失败："
            f"replacement_count={replacement_count}"
        )

    SNAPKV_UTILS_PATH.write_text(
        text,
        encoding="utf-8",
    )

    print("Patched:", SNAPKV_UTILS_PATH)


def main() -> None:
    patch_monkeypatch()
    patch_snapkv_utils()
    print("Stage 3B source patch completed.")


if __name__ == "__main__":
    main()
