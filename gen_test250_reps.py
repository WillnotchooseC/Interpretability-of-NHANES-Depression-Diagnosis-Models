"""
Repeated greedy inference of one SFT run on ood_data/curriculum_test_250.json
(250 questions with a `split` field: seen 150 / unseen 100, updated 2026-09-12),
for a repetition-based statistical evaluation that is judged EXTERNALLY --
this script does no scoring and makes no OpenAI calls.

Reps run strictly one after another, each on a FRESH SGLang server (never
reused across reps -- see gen_test120_reps.py / CLAUDE.md Gotchas). Decoding is
eval_plan_composition.run_inference_sglang_batch's exact config (temperature=0,
top_p=1.0, max_tokens=512, enable_thinking=False), so reps differ only through
SGLang's batching nondeterminism.

Output per rep (dir name follows the test file's stem):
  nhanes_model/<run>/eval_<test_stem>_rep{N}/test_predictions.json
one record per question: input, split, reference, prediction_raw, prediction
(= truncate_output(prediction_raw)), predicted_subquestions (parsed).
A rep with ANY empty generation (= request timeout) is NOT saved.

Usage:
    python gen_test250_reps.py                                   # progressive_corrected, reps 1 2 3
    python gen_test250_reps.py --test_file ood_data/curriculum_test_200.json --reps 1   # other test set, output dir follows the file stem
    python gen_test250_reps.py --reps 2                          # redo one rep
    python gen_test250_reps.py --run ood_curriculum_baseline_mixed1076_corrected
"""

import argparse
import json
import os
import time

import eval_plan_composition
from evaluate import launch_sglang_server, shutdown_sglang_server, SGLANG_PORT, parse_output, truncate_output
from eval_plan_composition import run_inference_sglang_batch

TEST_FILE = "ood_data/curriculum_test_250.json"
DEFAULT_RUN = "ood_curriculum_progressive_corrected"


def out_path(run_name, rep, test_file=TEST_FILE):
    # eval_<test-file stem>_rep{N}, e.g. eval_curriculum_test_250_rep1 / eval_curriculum_test_200_rep1
    stem = os.path.splitext(os.path.basename(test_file))[0]
    return os.path.join("nhanes_model", run_name, f"eval_{stem}_rep{rep}", "test_predictions.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=DEFAULT_RUN)
    ap.add_argument("--reps", nargs="+", type=int, default=[1, 2, 3])
    ap.add_argument("--test_file", default=TEST_FILE)
    ap.add_argument("--port", type=int, default=SGLANG_PORT)
    ap.add_argument("--timeout", type=int, default=300, help="per-request read timeout (s)")
    ap.add_argument("--overwrite", action="store_true", help="regenerate reps whose file already exists")
    args = ap.parse_args()

    eval_plan_composition.SGLANG_TIMEOUT = args.timeout
    merged_model = os.path.join("nhanes_model", args.run, "merged_model")
    if not os.path.isdir(merged_model):
        raise SystemExit(f"missing merged model: {merged_model}")
    test_data = json.load(open(args.test_file, encoding="utf-8"))
    print(f"{args.run}: reps {args.reps} on {args.test_file} ({len(test_data)} samples), sequential")

    for rep in args.reps:
        pred_file = out_path(args.run, rep, args.test_file)
        if os.path.exists(pred_file) and not args.overwrite:
            print(f"\n  rep{rep}: exists, skipping ({pred_file}); use --overwrite to redo")
            continue
        os.makedirs(os.path.dirname(pred_file), exist_ok=True)

        print(f"\n{'='*70}\n  rep{rep}  (fresh server)\n{'='*70}")
        t0 = time.time()
        proc, preds = None, None
        try:
            proc = launch_sglang_server(model_path=merged_model, port=args.port)
            if proc is None:
                raise SystemExit("SGLang failed to start")
            preds = run_inference_sglang_batch(f"http://localhost:{args.port}", test_file=args.test_file)
        finally:
            if proc:
                shutdown_sglang_server(proc)

        n_empty = sum(1 for p in preds if not p["prediction_raw"].strip())
        if n_empty:
            raise SystemExit(f"ABORT rep{rep}: {n_empty}/{len(preds)} EMPTY generations (timeouts). "
                             f"Not saving -- rerun with --reps {rep}.")

        for p, s in zip(preds, test_data):
            assert p["input"] == s["input"]
            p["split"] = s.get("split")
            p["prediction"] = truncate_output(p["prediction_raw"])
            p["predicted_subquestions"] = [
                {"step": q.step_num, "text": q.sub_question, "type": q.sq_type}
                for q in parse_output(p["prediction"])
            ]
        n_noparse = sum(1 for p in preds if not p["predicted_subquestions"])
        json.dump(preds, open(pred_file, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
        print(f"  Saved: {pred_file}  ({len(preds)} samples, 0 empty, "
              f"{n_noparse} with no parseable sub-question, rep took {time.time()-t0:.0f}s)")

    print("\nDone.")


if __name__ == "__main__":
    main()
