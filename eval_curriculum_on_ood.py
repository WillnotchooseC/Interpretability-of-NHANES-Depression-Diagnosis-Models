"""
Evaluate plan_composition_curriculum_v1 (curriculum-trained via train.py
--stage_files on curriculum_stage1-4, <PLAN>/<DECOMPOSITION> format) against
ood_data/test_dataset_ood_rebalanced.json's reference `output`, for direct
comparison with decomp_v1_ood's numbers on the same test set
(eval_decomp_v1_ood.py / eval_decomp_v1_ood.log).

The curriculum model always emits a <PLAN>...</PLAN> header before
<DECOMPOSITION>...</DECOMPOSITION>; evaluate.py's parse_output/truncate_output
only match "Step N:"/"Sub-question:" lines, so the <PLAN> block is silently
skipped and scoring is directly comparable to decomp_v1_ood's plain-format
output (same mechanism eval_plan_composition.py relies on).

This is a one-off comparison script, not part of the documented pipeline in
CLAUDE.md -- adapted from eval_decomp_v1_ood.py with the model path/output
root/label parameterized so it writes to its own directory instead of
overwriting nhanes_model/decomp_v1_ood's existing eval outputs.

Usage:
    python eval_curriculum_on_ood.py
    python eval_curriculum_on_ood.py --limit 10
"""

import argparse
import json
import os

from few_shot_cypher import launch_sglang_server, SGLANG_PORT
from run_finetuned_pipeline_eval import (
    truncate_decomposition, sglang_chat_complete_batch, hard_shutdown_sglang,
)
from evaluate import (
    get_openai_client, evaluate_experiment, print_metrics, save_results,
    EMBEDDING_MODEL, GPT_MODEL,
)

DECOMP_MODEL_PATH = "./nhanes_model/plan_composition_curriculum_v1/merged_model"
TEST_FILE = "ood_data/test_dataset_ood_rebalanced.json"
OUT_ROOT = "nhanes_model/plan_composition_curriculum_v1/ood_eval"
LABEL = "curriculum_v1_ood"
RAW_OUTPUTS_CACHE = os.path.join(OUT_ROOT, f"{LABEL}_raw_outputs.json")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_only", action="store_true")
    parser.add_argument("--truncated_only", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--decomp_model", type=str, default=DECOMP_MODEL_PATH)
    parser.add_argument("--max_new_tokens", type=int, default=384)
    parser.add_argument("--port", type=int, default=SGLANG_PORT)
    parser.add_argument("--regenerate", action="store_true",
                        help="Ignore any cached raw outputs and regenerate from scratch.")
    args = parser.parse_args()

    os.makedirs(OUT_ROOT, exist_ok=True)

    with open(TEST_FILE) as f:
        raw_data = json.load(f)
    if args.limit:
        raw_data = raw_data[: args.limit]
    print(f"Loaded {len(raw_data)} samples from {TEST_FILE}")

    if not args.regenerate and os.path.exists(RAW_OUTPUTS_CACHE):
        with open(RAW_OUTPUTS_CACHE) as f:
            cached = json.load(f)
        if len(cached.get("raw_outputs", [])) == len(raw_data):
            print(f"Loaded cached raw outputs from {RAW_OUTPUTS_CACHE} — skipping SGLang generation.")
            raw_outputs = cached["raw_outputs"]
        else:
            raw_outputs = None
    else:
        raw_outputs = None

    if raw_outputs is None:
        print(f"Launching SGLang for {args.decomp_model} ...")
        proc = launch_sglang_server(args.decomp_model, args.port)
        if proc is None:
            print("ERROR: failed to launch curriculum SGLang server.")
            return
        try:
            inputs = [item["input"] for item in raw_data]
            raw_outputs = sglang_chat_complete_batch(
                inputs, f"http://localhost:{args.port}", args.max_new_tokens, label=LABEL)
        finally:
            print("Tearing down SGLang server...")
            hard_shutdown_sglang(proc, args.port)

        with open(RAW_OUTPUTS_CACHE, "w") as f:
            json.dump({"raw_outputs": raw_outputs}, f, ensure_ascii=False)
        print(f"Saved raw outputs to {RAW_OUTPUTS_CACHE}")

    samples = [
        {
            "input": item["input"],
            "reference_output": item["output"],
            "model_raw_output": raw_out,
            "model_output": truncate_decomposition(raw_out),
        }
        for item, raw_out in zip(raw_data, raw_outputs)
    ]

    client = get_openai_client()

    variants = []
    if not args.truncated_only:
        variants.append(("raw", "model_raw_output"))
    if not args.raw_only:
        variants.append(("truncated", "model_output"))

    all_metrics = {}
    for label, field in variants:
        predictions = [
            {"input": s["input"], "reference": s["reference_output"], "prediction": s[field]}
            for s in samples
        ]
        evals, metrics = evaluate_experiment(f"{LABEL}_{label}", predictions, client,
                                              EMBEDDING_MODEL, GPT_MODEL)
        print_metrics(metrics, f"{LABEL} ({label})")
        out_dir = os.path.join(OUT_ROOT, label)
        save_results(f"{LABEL}_{label}", evals, metrics, predictions, out_dir)
        all_metrics[label] = metrics

    with open(os.path.join(OUT_ROOT, f"{LABEL}_eval_summary.json"), "w") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"\nSaved summary to {OUT_ROOT}/{LABEL}_eval_summary.json")


if __name__ == "__main__":
    main()
