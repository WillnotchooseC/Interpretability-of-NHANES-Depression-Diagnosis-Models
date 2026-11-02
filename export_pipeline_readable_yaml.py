"""
Reformat finetuned_pipeline_samples.json into a human-readable YAML, in the
same per-example style as test_dataset_gold_cypher.yaml (one block per test
sample, one sub-block per searching step) -- but with method1 (few-shot) and
method2 (schema+retrieval) Cypher/results/gold-score placed directly
alongside gold, instead of three separate hard-to-cross-reference JSON files.

Usage:
    python export_pipeline_readable_yaml.py
    python export_pipeline_readable_yaml.py --samples cypher_results_curriculum_v1_updated/finetuned_pipeline_samples.json --out cypher_results_curriculum_v1_updated/method_comparison_readable.yaml
"""

import argparse
import json

import yaml


def literal_str_representer(dumper, data):
    if "\n" in data:
        return dumper.represent_scalar("tag:yaml.org,2002:str", data, style="|")
    return dumper.represent_scalar("tag:yaml.org,2002:str", data)


yaml.add_representer(str, literal_str_representer)


def method_block(method_data, qi):
    res = (method_data or {}).get(str(qi)) or (method_data or {}).get(qi)
    if res is None:
        return None
    return {
        "cypher": res.get("cypher") or "",
        "exec_success": res.get("exec_success"),
        "has_results": res.get("has_results"),
        "error": res.get("error"),
        "result_rows": res.get("rows_data") or [],
        "gold_score": res.get("gold_score"),
    }


def build_entry(sample):
    searching = [sq for sq in sample["model_subqs"] if sq["type"] == "searching"]
    entry = {
        "id": sample["test_id"],
        "input": sample["input"],
        "gold_alignable": sample["alignable"],
        "model_decomposition": sample["model_output"],
    }
    if not sample["alignable"]:
        entry["note"] = (
            f"Model produced {sample['model_n_searching']} searching sub-question(s), "
            f"reference has {sample['ref_n_searching']} -- not positionally alignable to "
            f"gold, so no per-step Cypher comparison was run for this sample."
        )
        return entry

    steps = []
    for qi, sq in enumerate(searching):
        gold_q = sample["gold_queries"][qi] if qi < len(sample["gold_queries"]) else {}
        step_entry = {
            "step": qi + 1,
            "sub_question": sq["question"],
            "gold": {
                "cypher": gold_q.get("cypher") or "",
                "result": gold_q.get("gold_result") or [],
            },
            "method1_fewshot": method_block(sample.get("method1_fewshot"), qi),
            "method2_schema": method_block(sample.get("method2_schema"), qi),
        }
        steps.append(step_entry)
    entry["queries"] = steps
    return entry


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=str,
                         default="cypher_results_curriculum_v1_updated/finetuned_pipeline_samples.json")
    parser.add_argument("--out", type=str,
                         default="cypher_results_curriculum_v1_updated/method_comparison_readable.yaml")
    args = parser.parse_args()

    samples = json.load(open(args.samples, encoding="utf-8"))
    entries = [build_entry(s) for s in samples]

    with open(args.out, "w", encoding="utf-8") as f:
        yaml.dump(entries, f, allow_unicode=True, sort_keys=False, width=100000)

    print(f"Wrote {len(entries)} entries to {args.out}")


if __name__ == "__main__":
    main()
