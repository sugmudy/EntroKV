#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Dict, Iterable

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parents[1]
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from metrics import (  # noqa: E402
    classification_score,
    code_sim_score,
    count_score,
    qa_f1_score,
    qa_f1_zh_score,
    retrieval_score,
    retrieval_zh_score,
    rouge_score,
    rouge_zh_score,
)

DEFAULT_TASKS = ["qasper", "hotpotqa", "passage_retrieval_en"]

DATASET2METRIC: Dict[str, Callable[..., float]] = {
    "narrativeqa": qa_f1_score,
    "qasper": qa_f1_score,
    "multifieldqa_en": qa_f1_score,
    "multifieldqa_zh": qa_f1_zh_score,
    "hotpotqa": qa_f1_score,
    "2wikimqa": qa_f1_score,
    "musique": qa_f1_score,
    "dureader": rouge_zh_score,
    "gov_report": rouge_score,
    "qmsum": rouge_score,
    "multi_news": rouge_score,
    "vcsum": rouge_zh_score,
    "trec": classification_score,
    "triviaqa": qa_f1_score,
    "samsum": rouge_score,
    "lsht": classification_score,
    "passage_retrieval_en": retrieval_score,
    "passage_count": count_score,
    "passage_retrieval_zh": retrieval_zh_score,
    "lcc": code_sim_score,
    "repobench-p": code_sim_score,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Stage 4B LongBench runs.")
    parser.add_argument("--output-root", default="outputs/longbench_stage4b")
    parser.add_argument(
        "--runs",
        nargs="+",
        default=["full", "fixed_r0.30", "entrokv_r0.30_a0.50"],
    )
    parser.add_argument("--tasks", nargs="+", default=DEFAULT_TASKS)
    parser.add_argument(
        "--require-matched-samples",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser.parse_args()


def read_jsonl(path: Path) -> list[Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(path)
    rows: list[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("status") == "ok":
                rows.append(row)
    if not rows:
        raise RuntimeError(f"No successful records found in {path}")
    return rows


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def mean_or_none(values: Iterable[Any]) -> float | None:
    clean = [float(v) for v in values if v is not None]
    return float(statistics.fmean(clean)) if clean else None


def score_record(task: str, row: Dict[str, Any]) -> float:
    metric = DATASET2METRIC[task]
    prediction = row["prediction"]
    if task in {"trec", "triviaqa", "samsum", "lsht"}:
        prediction = prediction.lstrip("\n").split("\n")[0]

    answers = row["answers"]
    all_classes = row.get("all_classes", [])
    return max(
        float(metric(prediction, answer, all_classes=all_classes))
        for answer in answers
    )


def aggregate(run_name: str, task: str, rows: list[Dict[str, Any]]) -> Dict[str, Any]:
    scored_rows = []
    for row in rows:
        score = score_record(task, row)
        scored = dict(row)
        scored["sample_score"] = score
        scored_rows.append(scored)

    return {
        "run_name": run_name,
        "method": rows[0]["method"],
        "task": task,
        "samples": len(rows),
        "score": round(100.0 * statistics.fmean(r["sample_score"] for r in scored_rows), 4),
        "mean_input_tokens": mean_or_none(r.get("input_tokens") for r in rows),
        "mean_raw_input_tokens": mean_or_none(r.get("raw_input_tokens") for r in rows),
        "truncation_rate": mean_or_none(1.0 if r.get("truncated") else 0.0 for r in rows),
        "mean_generated_tokens": mean_or_none(r.get("generated_tokens") for r in rows),
        "mean_wall_seconds": mean_or_none(r.get("wall_seconds") for r in rows),
        "mean_prefill_gpu_ms": mean_or_none(r.get("prefill_gpu_ms") for r in rows),
        "mean_decode_gpu_ms": mean_or_none(r.get("decode_gpu_ms") for r in rows),
        "mean_decode_gpu_ms_per_forward": mean_or_none(
            r.get("decode_gpu_ms_per_forward") for r in rows
        ),
        "mean_peak_allocated_gb": mean_or_none(r.get("peak_allocated_gb") for r in rows),
        "mean_peak_reserved_gb": mean_or_none(r.get("peak_reserved_gb") for r in rows),
        "mean_effective_retention_ratio": mean_or_none(
            r.get("effective_retention_ratio") for r in rows
        ),
        "mean_target_final_capacity_per_head": mean_or_none(
            r.get("target_final_capacity_per_head") for r in rows
        ),
        "mean_final_cache_entries": mean_or_none(r.get("final_cache_entries") for r in rows),
        "mean_estimated_final_kv_gb": mean_or_none(
            r.get("estimated_final_kv_gb") for r in rows
        ),
        "scored_rows": scored_rows,
    }


def assert_fairness(
    data: Dict[str, Dict[str, list[Dict[str, Any]]]],
    runs: list[str],
    tasks: list[str],
    require_matched: bool,
) -> None:
    for task in tasks:
        by_run = {
            run: {int(row["sample_index"]): row for row in data[run][task]}
            for run in runs
        }
        index_sets = {run: set(rows) for run, rows in by_run.items()}
        if require_matched and len({frozenset(v) for v in index_sets.values()}) != 1:
            raise AssertionError(f"Sample indices differ for task={task}: {index_sets}")

        common = set.intersection(*(set(rows) for rows in by_run.values()))
        for sample_index in sorted(common):
            input_lengths = {
                run: int(by_run[run][sample_index]["input_tokens"])
                for run in runs
            }
            if len(set(input_lengths.values())) != 1:
                raise AssertionError(
                    f"Input token lengths differ for task={task}, sample={sample_index}: "
                    f"{input_lengths}"
                )

            compressed = [
                run
                for run in runs
                if by_run[run][sample_index]["method"] in {"fixed", "entrokv"}
            ]
            if len(compressed) >= 2:
                capacities = {
                    run: int(
                        by_run[run][sample_index]["target_final_capacity_per_head"]
                    )
                    for run in compressed
                }
                if len(set(capacities.values())) != 1:
                    raise AssertionError(
                        f"Fixed/EntroKV capacities differ for task={task}, "
                        f"sample={sample_index}: {capacities}"
                    )


def main() -> None:
    args = parse_args()
    output_root = (REPO_ROOT / args.output_root).resolve()
    eval_dir = output_root / "evaluation"
    eval_dir.mkdir(parents=True, exist_ok=True)

    data: Dict[str, Dict[str, list[Dict[str, Any]]]] = defaultdict(dict)
    for run in args.runs:
        for task in args.tasks:
            data[run][task] = read_jsonl(output_root / run / f"{task}.jsonl")

    assert_fairness(
        data=data,
        runs=args.runs,
        tasks=args.tasks,
        require_matched=args.require_matched_samples,
    )

    summary_rows: list[Dict[str, Any]] = []
    nested_summary: Dict[str, Any] = {}

    for run in args.runs:
        nested_summary[run] = {"tasks": {}}
        task_scores = []
        for task in args.tasks:
            result = aggregate(run, task, data[run][task])
            scored_rows = result.pop("scored_rows")
            summary_rows.append(result)
            nested_summary[run]["tasks"][task] = result
            task_scores.append(float(result["score"]))

            scored_path = eval_dir / run / f"{task}.scored.jsonl"
            scored_path.parent.mkdir(parents=True, exist_ok=True)
            with scored_path.open("w", encoding="utf-8") as f:
                for row in scored_rows:
                    json.dump(row, f, ensure_ascii=False)
                    f.write("\n")

        nested_summary[run]["macro_score"] = round(statistics.fmean(task_scores), 4)

    write_json(eval_dir / "summary.json", nested_summary)

    csv_fields = [
        "run_name",
        "method",
        "task",
        "samples",
        "score",
        "mean_input_tokens",
        "mean_raw_input_tokens",
        "truncation_rate",
        "mean_generated_tokens",
        "mean_wall_seconds",
        "mean_prefill_gpu_ms",
        "mean_decode_gpu_ms",
        "mean_decode_gpu_ms_per_forward",
        "mean_peak_allocated_gb",
        "mean_peak_reserved_gb",
        "mean_effective_retention_ratio",
        "mean_target_final_capacity_per_head",
        "mean_final_cache_entries",
        "mean_estimated_final_kv_gb",
    ]
    with (eval_dir / "summary.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields)
        writer.writeheader()
        for row in summary_rows:
            writer.writerow({key: row.get(key) for key in csv_fields})

    print("\nStage 4B evaluation summary")
    print("=" * 88)
    for run in args.runs:
        print(f"{run}: macro_score={nested_summary[run]['macro_score']}")
        for task in args.tasks:
            result = nested_summary[run]["tasks"][task]
            print(
                f"  {task:24s} score={result['score']:8.4f} "
                f"n={result['samples']:3d} "
                f"prefill_ms={result['mean_prefill_gpu_ms']:.3f} "
                f"peak_GB={result['mean_peak_allocated_gb']:.3f}"
            )

    print(f"\n[PASS] Matched-sample fairness checks passed.")
    print(f"[PASS] Summary written to {eval_dir / 'summary.json'}")
    print(f"[PASS] CSV written to {eval_dir / 'summary.csv'}")


if __name__ == "__main__":
    main()
