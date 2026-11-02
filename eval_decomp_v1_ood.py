"""
Evaluate decomp_v1_ood (SFT trained on ood_data/train_dataset_ood_rebalanced.json,
same way as decomp_v1 -- see train.py --train_file) against
ood_data/test_dataset_ood_rebalanced.json's reference `output`, reusing
evaluate.py's exact 3-dimension methodology (semantic similarity via
embeddings + Hungarian matching, sub-question count accuracy, redundancy
detection via embedding + GPT-4o verification) so the numbers are directly
comparable to decomp_v1's in-distribution numbers from eval_decomp_v1.py.

Unlike eval_decomp_v1.py (which reuses decomp_v1's already-cached
generations), this script has its own generation phase since decomp_v1_ood
has never been run against the ood test set before: it launches SGLang for
nhanes_model/decomp_v1_ood/merged_model, greedy-decodes a decomposition for
every ood test question, then evaluates both the raw and truncated output.

Usage:
    python eval_decomp_v1_ood.py                  # both raw and truncated
    python eval_decomp_v1_ood.py --raw_only
    python eval_decomp_v1_ood.py --truncated_only
    python eval_decomp_v1_ood.py --limit 10        # smoke test
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

DECOMP_MODEL_PATH = "./nhanes_model/decomp_v1_ood/merged_model"
TEST_FILE = "ood_data/test_dataset_ood_rebalanced.json"
OUT_ROOT = "nhanes_model/decomp_v1_ood"
RAW_OUTPUTS_CACHE = os.path.join(OUT_ROOT, "decomp_v1_ood_raw_outputs.json")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_only", action="store_true")
    parser.add_argument("--truncated_only", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--decomp_model", type=str, default=DECOMP_MODEL_PATH)
    parser.add_argument("--max_new_tokens", type=int, default=320)
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
            print("ERROR: failed to launch decomp_v1_ood SGLang server.")
            return
        try:
            inputs = [item["input"] for item in raw_data]
            raw_outputs = sglang_chat_complete_batch(
                inputs, f"http://localhost:{args.port}", args.max_new_tokens, label="decomp_ood")
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
        evals, metrics = evaluate_experiment(f"decomp_v1_ood_{label}", predictions, client,
                                              EMBEDDING_MODEL, GPT_MODEL)
        print_metrics(metrics, f"decomp_v1_ood ({label})")
        out_dir = os.path.join(OUT_ROOT, label)
        save_results(f"decomp_v1_ood_{label}", evals, metrics, predictions, out_dir)
        all_metrics[label] = metrics

    with open(os.path.join(OUT_ROOT, "decomp_v1_ood_eval_summary.json"), "w") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"\nSaved summary to {OUT_ROOT}/decomp_v1_ood_eval_summary.json")


if __name__ == "__main__":
    main()
