#!/usr/bin/env python3

from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]

MONKEYPATCH = (
    ROOT
    / "adaptive_snapkv"
    / "monkeypatch"
    / "monkeypatch.py"
)

SNAPKV_UTILS = (
    ROOT
    / "adaptive_snapkv"
    / "monkeypatch"
    / "snapkv_utils.py"
)

LLAMA_HIJACK = (
    ROOT
    / "adaptive_snapkv"
    / "monkeypatch"
    / "adaptive_llama_hijack.py"
)


def replace_once(
    text: str,
    old: str,
    new: str,
    name: str,
) -> str:
    count = text.count(old)

    if count != 1:
        raise RuntimeError(
            f"{name}: 预期找到 1 处，实际找到 {count} 处"
        )

    return text.replace(old, new, 1)


def patch_config_compress() -> None:
    text = MONKEYPATCH.read_text(encoding="utf-8")

    if "entrokv_scope=\"per_layer\"" not in text:
        old = '''    entrokv_debug=False,
):'''

        new = '''    entrokv_debug=False,
    entrokv_scope="per_layer",
    capacity_mode="absolute",
    retention_ratio=None,
):'''

        text = replace_once(
            text,
            old,
            new,
            "扩展 config_compress 参数",
        )

    if (
        "model.model.config.entrokv_scope"
        not in text
    ):
        old = '''    model.model.config.entrokv_debug = entrokv_debug

    return model
'''

        new = '''    model.model.config.entrokv_debug = entrokv_debug

    if entrokv_scope not in {"per_layer", "global"}:
        raise ValueError(
            f"Unsupported entrokv_scope={entrokv_scope!r}"
        )

    if capacity_mode not in {"absolute", "ratio"}:
        raise ValueError(
            f"Unsupported capacity_mode={capacity_mode!r}"
        )

    if capacity_mode == "ratio":
        if retention_ratio is None:
            raise ValueError(
                "ratio mode requires retention_ratio"
            )
        if not 0.0 < retention_ratio <= 1.0:
            raise ValueError(
                "retention_ratio must be in (0,1]"
            )

    model.model.config.entrokv_scope = entrokv_scope
    model.model.config.capacity_mode = capacity_mode
    model.model.config.retention_ratio = retention_ratio

    return model
'''

        text = replace_once(
            text,
            old,
            new,
            "写入 Stage 4A config",
        )

    MONKEYPATCH.write_text(text, encoding="utf-8")
    print("Patched:", MONKEYPATCH)


def patch_init_snapkv() -> None:
    text = SNAPKV_UTILS.read_text(encoding="utf-8")

    new_function = '''def init_snapkv(self):

    assert hasattr(self.config, "window_size"), "window_size not set"
    assert hasattr(self.config, "kernel_size"), "kernel_size not set"
    assert hasattr(self.config, "pooling"), "pooling not set"
    assert hasattr(self.config, "base_capacity"), "base_capacity not set"

    if not hasattr(self, "kv_cluster"):
        capacity_mode = getattr(
            self.config,
            "capacity_mode",
            "absolute",
        )

        cluster_kwargs = dict(
            window_size=self.config.window_size,
            max_capacity_prompt=self.config.base_capacity,
            kernel_size=self.config.kernel_size,
            pooling=self.config.pooling,
            layer_idx=self.layer_idx,
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

        if capacity_mode == "ratio":
            from adaptive_snapkv.monkeypatch.ratio_snapkv_cluster import (
                RatioSnapKVCluster,
            )

            self.kv_cluster = RatioSnapKVCluster(
                **cluster_kwargs,
                retention_ratio=self.config.retention_ratio,
            )
        elif capacity_mode == "absolute":
            self.kv_cluster = SnapKVCluster(
                **cluster_kwargs
            )
        else:
            raise ValueError(
                f"Unsupported capacity_mode={capacity_mode!r}"
            )

        if self.config.gqa_support:
            if self.config.model_type != "mistral":
                warnings.warn(
                    "GQA currently supports only for "
                    "mistral-7B-v0.2 model"
                )

        print(
            f"Compress config(Snap-{capacity_mode}): "
            f"window_size={self.kv_cluster.window_size}, "
            f"configured_capacity="
            f"{self.kv_cluster.max_capacity_prompt}, "
            f"retention_ratio="
            f"{getattr(self.kv_cluster, 'retention_ratio', None)}, "
            f"kernel_size={self.kv_cluster.kernel_size}, "
            f"pooling={self.kv_cluster.pooling}",
            flush=True,
        )
'''

    pattern = re.compile(
        r"def init_snapkv\(self\):.*?"
        r"(?=\ndef init_adaptive_snapkv\(self\):)",
        flags=re.DOTALL,
    )

    text, count = pattern.subn(
        new_function + "\n",
        text,
        count=1,
    )

    if count != 1:
        raise RuntimeError(
            f"替换 init_snapkv 失败：{count}"
        )

    SNAPKV_UTILS.write_text(text, encoding="utf-8")
    print("Patched:", SNAPKV_UTILS)


def patch_entrokv_init_args() -> None:
    text = SNAPKV_UTILS.read_text(encoding="utf-8")

    if "entrokv_scope=getattr(" in text:
        print("EntroKV init args already patched")
        return

    old = '''                entrokv_debug=getattr(
                    self.config,
                    "entrokv_debug",
                    False,
                ),
            )'''

    new = '''                entrokv_debug=getattr(
                    self.config,
                    "entrokv_debug",
                    False,
                ),
                entrokv_scope=getattr(
                    self.config,
                    "entrokv_scope",
                    "per_layer",
                ),
                capacity_mode=getattr(
                    self.config,
                    "capacity_mode",
                    "absolute",
                ),
                retention_ratio=getattr(
                    self.config,
                    "retention_ratio",
                    None,
                ),
            )'''

    text = replace_once(
        text,
        old,
        new,
        "传递 EntroKV Stage 4A 参数",
    )

    SNAPKV_UTILS.write_text(text, encoding="utf-8")
    print("Patched EntroKV args:", SNAPKV_UTILS)


def patch_llama_global_finalize() -> None:
    text = LLAMA_HIJACK.read_text(encoding="utf-8")

    if "finalize_entrokv_global_cache" in text:
        print("Llama global finalizer already patched")
        return

    old = '''    next_cache = next_decoder_cache if use_cache else None
    if return_legacy_cache:
'''

    new = '''    # EntroKV global scope:
    # all layers have now produced entropy and full prefill KV.
    # Perform one global layer-head allocation before converting
    # the cache back to legacy format for generation.
    if (
        use_cache
        and next_decoder_cache is not None
        and inputs_embeds.shape[1] != 1
        and getattr(
            self.config,
            "budget_mode",
            "adakv",
        ) == "entrokv"
        and getattr(
            self.config,
            "entrokv_scope",
            "per_layer",
        ) == "global"
    ):
        from adaptive_snapkv.monkeypatch.entrokv_global import (
            finalize_entrokv_global_cache,
        )

        finalize_entrokv_global_cache(
            llama_model=self,
            cache=next_decoder_cache,
        )

    next_cache = next_decoder_cache if use_cache else None
    if return_legacy_cache:
'''

    text = replace_once(
        text,
        old,
        new,
        "插入 global cache finalizer",
    )

    LLAMA_HIJACK.write_text(text, encoding="utf-8")
    print("Patched:", LLAMA_HIJACK)


def main() -> None:
    patch_config_compress()
    patch_init_snapkv()
    patch_entrokv_init_args()
    patch_llama_global_finalize()
    print("Stage 4A patch completed.")


if __name__ == "__main__":
    main()
