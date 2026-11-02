"""
Generate repeat inference runs (rep2, rep3) of the two *_corrected SFT runs on
ood_data/curriculum_test_120.json, for the 3-repetition statistical comparison
of sub-question recall / hallucination / full-sample coverage.

rep1 is the already-cached eval_plan_composition.py run at
  nhanes_model/<run>/eval_curriculum_test_120/test_predictions.json
so this script only produces rep2 and rep3, writing to
  nhanes_model/<run>/eval_curriculum_test_120_rep{2,3}/test_predictions.json

Decoding matches rep1 exactly (eval_plan_composition.run_inference_sglang_batch:
temperature=0, top_p=1.0, max_tokens=512, enable_thinking=False), so the three
reps differ only through SGLang's own run-to-run nondeterminism (batch
composition / reduction order), not through a changed sampling config.

A FRESH SGLang server is launched for EVERY rep. An earlier version reused one
server for both reps of a model to save the ~60s model load, and the second rep
degraded badly (275s -> 719-867s for the same 120 requests) until requests blew
past evaluate.SGLANG_TIMEOUT=60s; run_inference_sglang_batch then returns "" for
those samples, which scores as "nothing covered, nothing hallucinated" and
silently poisons the metrics (an empty prediction has no sub-questions, so the
hallucination denominator is 0 and the rate looks like a perfect 0%). Hence:
one server per rep, plus a raised per-request timeout as defence in depth.

Usage:
    python gen_test120_reps.py
    python gen_test120_reps.py --runs ood_curriculum_progressive_corrected
    python gen_test120_reps.py --reps 2 3
"""

import argparse
import json
import os

import eval_plan_composition
from evaluate import launch_sglang_server, shutdown_sglang_server, SGLANG_PORT
from eval_plan_composition import run_inference_sglang_batch

TEST_FILE = "ood_data/curriculum_test_120.json"

RUNS = [
    "ood_curriculum_progressive_corrected",
    "ood_curriculum_baseline_mixed1076_corrected",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", default=RUNS)
    ap.add_argument("--reps", nargs="+", type=int, default=[2, 3])
    ap.add_argument("--test_file", default=TEST_FILE)
    ap.add_argument("--port", type=int, default=SGLANG_PORT)
    ap.add_argument("--timeout", type=int, default=300,
                    help="Per-request read timeout (evaluate.py's default 60s is "
                         "too tight once the server slows down).")
    args = ap.parse_args()

    # run_inference_sglang_batch reads SGLANG_TIMEOUT as a module global at call
    # time, so raising it here applies without touching evaluate.py's default.
    eval_plan_composition.SGLANG_TIMEOUT = args.timeout

    for run_name in args.runs:
        merged_model = os.path.join("nhanes_model", run_name, "merged_model")
        if not os.path.isdir(merged_model):
            print(f"SKIP {run_name}: {merged_model} not found")
            continue

        print(f"\n{'='*70}")
        print(f"  {run_name}: reps {args.reps} on {args.test_file}")
        print(f"{'='*70}")

        for rep in args.reps:
            out_dir = os.path.join("nhanes_model", run_name,
                                   f"eval_curriculum_test_120_rep{rep}")
            os.makedirs(out_dir, exist_ok=True)
            pred_file = os.path.join(out_dir, "test_predictions.json")

            print(f"\n  --- rep{rep} (fresh server) ---")
            sglang_process = None
            preds = None
            try:
                sglang_process = launch_sglang_server(model_path=merged_model, port=args.port)
                if sglang_process is None:
                    print("ERROR: failed to launch SGLang.")
                else:
                    server_url = f"http://localhost:{args.port}"
                    preds = run_inference_sglang_batch(server_url, test_file=args.test_file)
            finally:
                if sglang_process:
                    shutdown_sglang_server(sglang_process)

            if preds is None:
                continue
            n_empty = sum(1 for p in preds if not p["prediction_raw"].strip())
            if n_empty:
                print(f"  WARNING: {n_empty}/{len(preds)} predictions came back EMPTY "
                      f"(timeouts). NOT saving -- rerun this rep.")
                continue
            with open(pred_file, "w", encoding="utf-8") as f:
                json.dump(preds, f, indent=2, ensure_ascii=False)
            print(f"  Saved: {pred_file}  (0 empty)")

    print("\nAll reps generated.")


if __name__ == "__main__":
    main()
