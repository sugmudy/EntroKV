# Stage 4C: EntroKV Dynamic-Budget Visualization

This stage consumes the per-sample EntroKV allocation fields already saved by `stage4b_runner.py`:

- `history_budgets`: `[num_layers, num_kv_heads]`
- `entropy`: `[num_layers, num_kv_heads]`
- `target_history_capacity_per_head`
- `input_tokens`, `truncated`, and retention metadata

No model inference is required for visualization.

## Why normalize before averaging

LongBench samples have different input lengths, so directly averaging absolute token budgets would bias the heatmap toward longer samples. The main visualization therefore uses the allocation multiplier

```text
M[s,l,h] = history_budgets[s,l,h] / target_history_capacity_per_head[s]
```

Interpretation:

- `M > 1`: the layer-head receives more history budget than uniform fixed SnapKV;
- `M = 1`: the same history budget as uniform fixed SnapKV;
- `M < 1`: the layer-head is compressed more aggressively.

The script verifies that every sample conserves the global budget and that the mean multiplier over all layer-heads is one.

## Dependencies

```bash
python -m pip install matplotlib
```

`numpy` is already required by the Stage 4B runner.

## Run on the 100-sample experiment

```bash
cd /root/autodl-tmp/entrokv_repro/AdaKV

git fetch origin
git checkout stage4c-dynamic-budget-visualization

EXPECTED_SAMPLES=100 \
INPUT_ROOT=outputs/longbench_stage4b \
RUN_NAME=entrokv_r0.30_a0.50 \
bash experiments/LongBench/run_stage4c_visualization.sh
```

To visualize a frozen copy such as `outputs/longbench_stage4b_main100`:

```bash
EXPECTED_SAMPLES=100 \
INPUT_ROOT=outputs/longbench_stage4b_main100 \
bash experiments/LongBench/run_stage4c_visualization.sh
```

For the current 50-sample repository snapshot, set:

```bash
EXPECTED_SAMPLES=50 bash experiments/LongBench/run_stage4c_visualization.sh
```

## Generated outputs

The default output directory is:

```text
outputs/longbench_stage4b/visualization/entrokv_r0.30_a0.50/
```

Main figures are saved as both PNG and PDF:

1. `01_task_budget_multiplier_heatmaps`
   - one shared-scale layer-head heatmap for each task;
   - satisfies the required layer-head heatmap and multi-task comparison.
2. `02_pairwise_task_difference_heatmaps`
   - Qasper vs. HotpotQA, Qasper vs. PassageRetrieval-en, and HotpotQA vs. PassageRetrieval-en;
   - directly visualizes task-wise heterogeneity.
3. `03_layerwise_budget_profile`
   - averages over samples and KV heads, showing where budget concentrates across layers.
4. `04_headwise_budget_profile`
   - averages over samples and layers, showing persistent KV-head preferences.
5. `05_entropy_budget_relationship`
   - checks whether mean entropy and mean allocated budget follow the expected monotonic relation.
6. `individual/`
   - separate task heatmaps suitable for reports and presentations.

Generated tables:

- `tables/task_summary.csv`
- `tables/layer_head_summary.csv`
- `tables/top_bottom_layer_heads.csv`
- `tables/task_allocation_similarity.csv`
- `tables/pairwise_task_difference_summary.csv`
- `visualization_summary.json`
- `analysis.md`

## Recommended interpretation order

1. Use the shared-scale heatmap to identify stable high- and low-budget regions.
2. Use pairwise difference heatmaps to determine whether task differences are localized to specific layers or heads.
3. Use layer-wise and head-wise profiles to summarize the two axes separately.
4. Use `top_bottom_layer_heads.csv` to quote exact layer-head positions and allocation multipliers.
5. Use the entropy-budget plot as an implementation sanity check, not as proof that entropy equals semantic importance.

Avoid assigning a fixed semantic role to a single attention head based only on one heatmap. The defensible conclusion is statistical: under the current model, tasks, prompts, retention ratio, and sample set, certain layer-head positions consistently receive relatively more or less KV history budget.
