"""
Evaluate plan_composition_v1 (SFT run trained on data/plan_composition_{train,valid,test}.json,
output format `<PLAN>...</PLAN><DECOMPOSITION>...</DECOMPOSITION>`) on data/plan_composition_test.json.

Reuses evaluate.py's exact 3-dimension methodology (semantic similarity via embeddings +
Hungarian matching, sub-question count accuracy, redundancy detection) against the
<DECOMPOSITION> portion of the output -- parse_output/truncate_output only match
"Step N:"/"Sub-question: ... # Type:" lines, so the <PLAN> block's lines are silently
skipped by that machinery, same as for any other non-matching text.

Additionally checks the <PLAN> block itself (new in this data format, not covered by
evaluate.py): whether it's present, and whether its BINDINGS line is valid JSON.

Usage:
    python eval_plan_composition.py --run_name plan_composition_v1   # launch SGLang + full pipeline
    python eval_plan_composition.py --run_name plan_composition_v2 --sglang_url http://localhost:30000
    python eval_plan_composition.py --run_name plan_composition_v1 --skip_inference  # reuse cached predictions
"""

import argparse
import json
import os
import re
import time
import concurrent.futures

import requests

from evaluate import (
    get_openai_client, evaluate_experiment, print_metrics, save_results,
    truncate_output, launch_sglang_server, shutdown_sglang_server,
    EMBEDDING_MODEL, GPT_MODEL, SGLANG_PORT, MAX_RETRIES, SGLANG_TIMEOUT,
    SYSTEM_PROMPT,
)

TEST_FILE = "data/plan_composition_test.json"

PLAN_RE = re.compile(r"<PLAN>\s*(.*?)\s*</PLAN>", re.DOTALL)
BINDINGS_RE = re.compile(r"BINDINGS\s*=\s*(\{.*\})")


def run_inference_sglang_batch(sglang_url, max_workers=16, test_file=TEST_FILE):
    test_data = json.load(open(test_file, "r", encoding="utf-8"))
    print(f"  Test samples: {len(test_data)}")

    def infer_one(idx_sample):
        idx, sample = idx_sample
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": sample["input"]},
        ]
        payload = {
            "model": "default",
            "messages": messages,
            "max_tokens": 512,
            "temperature": 0,
            "top_p": 1.0,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        for attempt in range(MAX_RETRIES):
            try:
                resp = requests.post(
                    f"{sglang_url}/v1/chat/completions",
                    json=payload,
                    timeout=SGLANG_TIMEOUT,
                )
                resp.raise_for_status()
                pred_raw = resp.json()["choices"][0]["message"]["content"]
                return idx, pred_raw
            except Exception as e:
                if attempt < MAX_RETRIES - 1:
                    time.sleep(1)
                else:
                    print(f"    [{idx}] failed: {e}")
        return idx, ""

    t_start = time.time()
    results = [None] * len(test_data)
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = executor.map(infer_one, enumerate(test_data))
        for idx, pred_raw in futures:
            results[idx] = {
                "input": test_data[idx]["input"],
                "reference": test_data[idx]["output"],
                "prediction_raw": pred_raw,
            }
    elapsed = time.time() - t_start
    print(f"  Done! {len(results)} samples in {elapsed:.1f}s ({len(results)/elapsed:.1f} samples/s)")
    return results


def check_plan_block(text: str) -> dict:
    m = PLAN_RE.search(text)
    if not m:
        return {"plan_present": False, "bindings_valid_json": False}
    body = m.group(1)
    bm = BINDINGS_RE.search(body)
    if not bm:
        return {"plan_present": True, "bindings_valid_json": False}
    try:
        json.loads(bm.group(1))
        return {"plan_present": True, "bindings_valid_json": True}
    except Exception:
        return {"plan_present": True, "bindings_valid_json": False}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_name", type=str, required=True,
                        help="nhanes_model/<run_name>/merged_model to evaluate, e.g. plan_composition_v2")
    parser.add_argument("--test_file", type=str, default=TEST_FILE)
    parser.add_argument("--sglang_url", type=str, default=None,
                        help="Use an already-running SGLang server instead of launching one")
    parser.add_argument("--port", type=int, default=SGLANG_PORT)
    parser.add_argument("--skip_inference", action="store_true")
    parser.add_argument("--skip_redundancy", action="store_true", default=True,
                         help="Skip Dim 3 (redundancy, needs GPT-4o calls) and only report "
                              "semantic similarity / count / type. Default on.")
    parser.add_argument("--with_redundancy", dest="skip_redundancy", action="store_false",
                         help="Include the redundancy dimension (GPT-4o calls).")
    args = parser.parse_args()

    merged_model = os.path.join("nhanes_model", args.run_name, "merged_model")
    # Testing on a non-default test file writes to its own subdir so it never
    # clobbers the standard plan_composition_test.json results for this run.
    if os.path.abspath(args.test_file) == os.path.abspath(TEST_FILE):
        out_root = os.path.join("nhanes_model", args.run_name)
    else:
        tag = os.path.splitext(os.path.basename(args.test_file))[0]
        out_root = os.path.join("nhanes_model", args.run_name, f"eval_{tag}")
    pred_file = os.path.join(out_root, "test_predictions.json")
    os.makedirs(out_root, exist_ok=True)

    sglang_process = None
    try:
        if args.skip_inference and os.path.exists(pred_file):
            print(f"Loading existing predictions: {pred_file}")
            raw_predictions = json.load(open(pred_file, "r", encoding="utf-8"))
        else:
            server_url = args.sglang_url or f"http://localhost:{args.port}"
            if not args.sglang_url:
                print(f"Launching SGLang for {merged_model}...")
                sglang_process = launch_sglang_server(model_path=merged_model, port=args.port)
                if sglang_process is None:
                    print("ERROR: failed to launch SGLang.")
                    return
            raw_predictions = run_inference_sglang_batch(server_url, test_file=args.test_file)
            with open(pred_file, "w", encoding="utf-8") as f:
                json.dump(raw_predictions, f, indent=2, ensure_ascii=False)
            print(f"Predictions saved: {pred_file}")

        # <PLAN> block check (supplementary, not part of evaluate.py's 3 dimensions)
        plan_checks = [check_plan_block(p["prediction_raw"]) for p in raw_predictions]
        n = len(plan_checks)
        plan_present_rate = sum(c["plan_present"] for c in plan_checks) / n * 100
        bindings_valid_rate = sum(c["bindings_valid_json"] for c in plan_checks) / n * 100
        print(f"\n  <PLAN> block present:        {plan_present_rate:.1f}%")
        print(f"  BINDINGS valid JSON:         {bindings_valid_rate:.1f}%")

        # Truncate at the DECOMPOSITION portion for evaluate.py's 3-dim methodology
        truncated_predictions = []
        for p in raw_predictions:
            tp = {"input": p["input"], "reference": p["reference"],
                  "prediction": truncate_output(p["prediction_raw"])}
            truncated_predictions.append(tp)

        client = get_openai_client()
        evals, metrics = evaluate_experiment(
            args.run_name, truncated_predictions, client,
            emb_model=EMBEDDING_MODEL, gpt_model=GPT_MODEL,
            skip_redundancy=args.skip_redundancy,
        )
        metrics["plan_present_rate"] = plan_present_rate
        metrics["bindings_valid_json_rate"] = bindings_valid_rate

        print_metrics(metrics, args.run_name)
        save_results(args.run_name, evals, metrics, truncated_predictions, out_root)

        with open(os.path.join(out_root, "metrics.json"), "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"\nSaved metrics to {out_root}/metrics.json")

    finally:
        if sglang_process:
            shutdown_sglang_server(sglang_process)


if __name__ == "__main__":
    main()
