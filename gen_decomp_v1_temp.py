"""
Regenerate decomp_v1's decomposition on data/test_dataset.json at a given
sampling temperature (default: the run_finetuned_pipeline_eval.py pipeline
always uses temperature=0 / greedy decoding — this script exists to probe
what happens under real sampling randomness, e.g. --temperature 0.8).

Usage:
    python gen_decomp_v1_temp.py --temperature 0.8
"""

import argparse
import json
import os
import re
import time
import concurrent.futures

import requests

from few_shot_cypher import launch_sglang_server, SGLANG_PORT
from run_finetuned_pipeline_eval import (
    DECOMP_MODEL_PATH, DECOMP_SYSTEM_PROMPT, TEST_FILE,
    truncate_decomposition, hard_shutdown_sglang,
)
from run_cypher_qa_comparison import extract_all_subquestions

OUT_DIR = "cypher_results"


def sglang_chat_complete(question, url, max_tokens, temperature, retries=3):
    payload = {
        "model": "default",
        "messages": [
            {"role": "system", "content": DECOMP_SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": 1.0 if temperature == 0 else 0.95,
        "stop": ["�", "\nStep by step:", "\nuser", "\nassistant", "<|im_start|>"],
        "chat_template_kwargs": {"enable_thinking": False},
    }
    for attempt in range(retries):
        try:
            resp = requests.post(f"{url}/v1/chat/completions", json=payload, timeout=60)
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"]
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(2)
            else:
                print(f"    sglang_chat_complete failed: {e}")
                return ""


def sglang_chat_complete_batch(questions, url, max_tokens, temperature, max_workers=16, label=""):
    results = [""] * len(questions)

    def one(idx_q):
        idx, q = idx_q
        return idx, sglang_chat_complete(q, url, max_tokens, temperature)

    t0 = time.time()
    done = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as ex:
        for idx, text in ex.map(one, enumerate(questions)):
            results[idx] = text
            done += 1
            if done % 20 == 0 or done == len(questions):
                print(f"    [{label}] {done}/{len(questions)}  ({time.time()-t0:.0f}s elapsed)")
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--temperature", type=float, required=True)
    parser.add_argument("--n_samples", type=int, default=1,
                         help="Generate this many independent samples per question (default 1).")
    parser.add_argument("--max_new_tokens", type=int, default=320)
    parser.add_argument("--port", type=int, default=SGLANG_PORT)
    parser.add_argument("--model_path", type=str, default=DECOMP_MODEL_PATH,
                         help="Override the model served (e.g. a GRPO-merged checkpoint).")
    parser.add_argument("--out_suffix", type=str, default="",
                         help="Appended to the output filename, e.g. '_grpo', to avoid "
                              "overwriting a baseline run's samples file.")
    args = parser.parse_args()

    raw_data = json.load(open(TEST_FILE))
    print(f"Loaded {len(raw_data)} samples from {TEST_FILE}")

    server_url = f"http://localhost:{args.port}"
    proc = launch_sglang_server(args.model_path, args.port)
    if proc is None:
        print("ERROR: failed to launch decomp_v1 SGLang server.")
        return

    try:
        inputs = [item["input"] for item in raw_data]
        all_runs = []
        for run_i in range(args.n_samples):
            label = f"temp{args.temperature}-run{run_i+1}"
            outs = sglang_chat_complete_batch(
                inputs, server_url, args.max_new_tokens, args.temperature, label=label)
            all_runs.append(outs)
    finally:
        hard_shutdown_sglang(proc, args.port)

    samples = []
    for i, item in enumerate(raw_data):
        ref = item["output"]
        ref_subqs = extract_all_subquestions(ref)
        runs_info = []
        for run_i, outs in enumerate(all_runs):
            raw_out = outs[i]
            trunc = truncate_decomposition(raw_out)
            exact = trunc.strip() == ref.strip()
            subqs = extract_all_subquestions(trunc)
            runs_info.append({
                "raw_output": raw_out, "truncated_output": trunc,
                "exact_match_ref": exact, "n_searching": sum(1 for s in subqs if s["type"] == "searching"),
                "n_total": len(subqs),
            })
        samples.append({
            "index": i, "input": item["input"], "reference_output": ref,
            "ref_n_searching": sum(1 for s in ref_subqs if s["type"] == "searching"),
            "runs": runs_info,
        })

    n_exact = sum(1 for s in samples if s["runs"][0]["exact_match_ref"])
    print(f"\nExact match to reference (run 1, temp={args.temperature}): {n_exact}/{len(samples)} "
          f"({n_exact/len(samples)*100:.1f}%)")

    if args.n_samples > 1:
        # for samples with >1 run, how often do independent samples at this temperature agree with each other
        n_self_consistent = sum(
            1 for s in samples
            if len({r["truncated_output"].strip() for r in s["runs"]}) == 1
        )
        print(f"Self-consistent across {args.n_samples} independent samples: "
              f"{n_self_consistent}/{len(samples)} ({n_self_consistent/len(samples)*100:.1f}%)")

    out_path = os.path.join(OUT_DIR, f"decomp_v1_temp{args.temperature}{args.out_suffix}_samples.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(samples, f, indent=2, ensure_ascii=False)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
