"""
Evaluate decomp_v1's decomposition quality against data/test_dataset.json's
reference `output`, reusing evaluate.py's exact 3-dimension methodology
(semantic similarity via embeddings + Hungarian matching, sub-question count
accuracy, redundancy detection via embedding + GPT-4o verification) so the
numbers are directly comparable to the exp1-6 table in README.md.

Uses the already-cached decomp_v1 generations from
cypher_results/decomp_only_samples.json (produced by
run_finetuned_pipeline_eval.py's Phase 1) — no SGLang/GPU inference needed.

Usage:
    python eval_decomp_v1.py              # both raw and truncated
    python eval_decomp_v1.py --raw_only
    python eval_decomp_v1.py --truncated_only
"""

import argparse
import json
import os

from evaluate import (
    get_openai_client, evaluate_experiment, print_metrics, save_results,
    EMBEDDING_MODEL, GPT_MODEL,
)

SAMPLES_FILE = "cypher_results/decomp_only_samples.json"
OUT_ROOT = "nhanes_model/decomp_v1"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_only", action="store_true")
    parser.add_argument("--truncated_only", action="store_true")
    args = parser.parse_args()

    samples = json.load(open(SAMPLES_FILE))
    print(f"Loaded {len(samples)} decomp_v1 samples from {SAMPLES_FILE}")

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
        evals, metrics = evaluate_experiment(f"decomp_v1_{label}", predictions, client,
                                              EMBEDDING_MODEL, GPT_MODEL)
        print_metrics(metrics, f"decomp_v1 ({label})")
        out_dir = os.path.join(OUT_ROOT, label)
        save_results(f"decomp_v1_{label}", evals, metrics, predictions, out_dir)
        all_metrics[label] = metrics

    with open(os.path.join(OUT_ROOT, "decomp_v1_eval_summary.json"), "w") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"\nSaved summary to {OUT_ROOT}/decomp_v1_eval_summary.json")


if __name__ == "__main__":
    main()
