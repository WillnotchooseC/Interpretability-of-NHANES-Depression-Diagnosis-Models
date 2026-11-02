"""
Single-rep sub-question comparison of the two *_corrected SFT runs on
ood_data/curriculum_test_250.json (250 questions, added 2026-09-12).

Same three metrics, same GPT-4o judge and same statistics as
eval_corrected_3rep_subq_ci.py, just one inference pass per model:

  1. Sub-question RECALL        -- reference sub-Qs whose information need the
                                   predicted set collectively covers
  2. Sub-question HALLUCINATION -- predicted sub-Qs grounded in no reference
                                   sub-question's information need
  3. FULL-SAMPLE COVERAGE       -- samples where every reference sub-Q is covered

Phase 1 (inference): a FRESH SGLang server per model (never reused -- see
gen_test120_reps.py for why), decoding identical to eval_plan_composition
(temperature=0, top_p=1.0, max_tokens=512, enable_thinking=False). Predictions
are written to nhanes_model/<run>/eval_curriculum_test_250/test_predictions.json
and a run with ANY empty prediction (= timeout) is refused, not scored.

Phase 2 (judging): eval_progressive_vs_baseline_subq_ci.judge_sample verbatim,
verdicts content-cached in cypher_results/corrected_test250_judge_cache.json.

Statistics: point estimate + 95% cluster-bootstrap CI over samples (Wilson also
reported); paired model comparison via paired cluster bootstrap on the delta,
paired Wilcoxon on per-sample rates, exact McNemar on full coverage.

curriculum_test_250.json carries no seen/unseen template labels (the test_120
UNSEEN_IDX split is index-based and does not transfer), so only "overall" is
reported unless --unseen_idx <json list of indices> is supplied.

Usage:
    python eval_corrected_test250_subq.py                   # inference + judge
    python eval_corrected_test250_subq.py --skip_inference  # reuse cached predictions
    python eval_corrected_test250_subq.py --skip_judge      # reuse cached verdicts too
    python eval_corrected_test250_subq.py --limit 10 --workers 4
"""

import argparse
import json
import os
import time

import numpy as np
from scipy import stats as sps

import eval_plan_composition
from evaluate import get_openai_client, launch_sglang_server, shutdown_sglang_server, SGLANG_PORT
from eval_plan_composition import run_inference_sglang_batch
from eval_progressive_vs_baseline_subq_ci import judge_all, load_samples, wilson_ci
from eval_corrected_3rep_subq_ci import metric_values, mcnemar_exact

TEST_FILE = "ood_data/curriculum_test_250.json"
EVAL_SUBDIR = "eval_curriculum_test_250"
MODELS = {
    "progressive_corrected": "ood_curriculum_progressive_corrected",
    "baseline_corrected": "ood_curriculum_baseline_mixed1076_corrected",
}
N_BOOT = 10000
SEED = 20260912


def pred_path(run_name):
    return os.path.join("nhanes_model", run_name, EVAL_SUBDIR, "test_predictions.json")


def run_inference(run_name, test_file, port, timeout):
    merged_model = os.path.join("nhanes_model", run_name, "merged_model")
    if not os.path.isdir(merged_model):
        raise FileNotFoundError(merged_model)
    out = pred_path(run_name)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    eval_plan_composition.SGLANG_TIMEOUT = timeout

    print(f"\n{'='*70}\n  inference: {run_name} on {test_file} (fresh server)\n{'='*70}")
    proc = None
    preds = None
    try:
        proc = launch_sglang_server(model_path=merged_model, port=port)
        if proc is None:
            raise RuntimeError(f"SGLang failed to start for {run_name}")
        preds = run_inference_sglang_batch(f"http://localhost:{port}", test_file=test_file)
    finally:
        if proc:
            shutdown_sglang_server(proc)

    n_empty = sum(1 for p in preds if not p["prediction_raw"].strip())
    if n_empty:
        raise SystemExit(f"ABORT: {n_empty}/{len(preds)} predictions EMPTY (timeouts) for "
                         f"{run_name}; not saving. Rerun inference.")
    json.dump(preds, open(out, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    print(f"  Saved: {out}  (0 empty)")


def per_sample_counts(samples, verdicts):
    out = []
    for s, (ref_cov, pred_grounded) in zip(samples, verdicts):
        n_ref, n_cov = len(s["ref"]), sum(ref_cov)
        out.append({
            "idx": s["idx"], "split": s["split"], "input": s["input"],
            "n_ref": n_ref, "n_covered": n_cov,
            "n_pred": len(s["pred"]), "n_halluc": sum(1 for g in pred_grounded if not g),
            "full_cov": 1.0 if (n_ref > 0 and n_cov == n_ref) else 0.0,
            "missing_refs": [q.sub_question for q, c in zip(s["ref"], ref_cov) if not c],
            "hallucinated_preds": [q.sub_question for q, g in zip(s["pred"], pred_grounded) if not g],
        })
    return out


def cluster_bootstrap_ci(rows_by_model, split, metric, paired=False, n_boot=N_BOOT, seed=SEED):
    labels = list(rows_by_model)
    idxs = [i for i, r in enumerate(rows_by_model[labels[0]]) if split == "overall" or r["split"] == split]
    rng = np.random.default_rng(seed)
    n = len(idxs)

    def stat(rows, sel):
        return metric_values([rows[i] for i in sel])[metric]

    draws = []
    for _ in range(n_boot):
        sel = [idxs[j] for j in rng.integers(0, n, size=n)]
        if paired:
            draws.append(stat(rows_by_model[labels[0]], sel) - stat(rows_by_model[labels[1]], sel))
        else:
            draws.append(stat(rows_by_model[labels[0]], sel))
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return float(lo), float(hi), float(np.mean(draws))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test_file", default=TEST_FILE)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--port", type=int, default=SGLANG_PORT)
    ap.add_argument("--timeout", type=int, default=300, help="per-request SGLang read timeout")
    ap.add_argument("--skip_inference", action="store_true")
    ap.add_argument("--skip_judge", action="store_true")
    ap.add_argument("--unseen_idx", default=None,
                    help="optional JSON file with a list of sample indices to treat as unseen_40-style OOD split")
    ap.add_argument("--out_dir", default="cypher_results")
    ap.add_argument("--cache", default="cypher_results/corrected_test250_judge_cache.json")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # ---- phase 1: inference ----
    for label, run_name in MODELS.items():
        if args.skip_inference or os.path.exists(pred_path(run_name)):
            if os.path.exists(pred_path(run_name)):
                print(f"predictions exist: {label} -> {pred_path(run_name)}")
                continue
            raise FileNotFoundError(f"--skip_inference but no predictions for {label}: {pred_path(run_name)}")
        run_inference(run_name, args.test_file, args.port, args.timeout)

    unseen = set(json.load(open(args.unseen_idx))) if args.unseen_idx else set()
    splits = ["overall"] + (["seen", "unseen"] if unseen else [])

    # ---- phase 2: judge ----
    cache = json.load(open(args.cache)) if os.path.exists(args.cache) else {}
    client = None
    rows = {}
    for label, run_name in MODELS.items():
        samples = load_samples(pred_path(run_name), limit=args.limit)
        for s in samples:  # test_120's index-based split does not apply here
            s["split"] = "unseen" if s["idx"] in unseen else ("seen" if unseen else "overall")
        n_no_subq = sum(1 for s in samples if len(s["pred"]) == 0)
        if n_no_subq:
            raise SystemExit(f"ABORT: {label} has {n_no_subq}/{len(samples)} samples with zero parsed "
                             f"sub-questions -- likely inference timeouts; regenerate before scoring.")
        key = f"{label}|{args.test_file}|{args.limit}"
        if key in cache:
            verdicts = [(v[0], v[1]) for v in cache[key]]
            print(f"cache hit: {label}")
        elif args.skip_judge:
            raise SystemExit(f"--skip_judge but no cached verdicts for {key}")
        else:
            if client is None:
                client = get_openai_client()
            print(f"Judging {label}: {len(samples)} samples ...")
            t0 = time.time()
            verdicts = judge_all(client, samples, args.workers)
            print(f"  done in {time.time()-t0:.0f}s")
            cache[key] = [[list(v[0]), list(v[1])] for v in verdicts]
            json.dump(cache, open(args.cache, "w"))
        rows[label] = per_sample_counts(samples, verdicts)

    labels = list(MODELS)
    assert [r["idx"] for r in rows[labels[0]]] == [r["idx"] for r in rows[labels[1]]]

    # ---- summary ----
    summary = {}
    for label in labels:
        summary[label] = {}
        for split in splits:
            sub = [r for r in rows[label] if split == "overall" or r["split"] == split]
            mv = metric_values(sub)
            entry = {"n_samples": mv["_n"], "ref_subq_total": mv["_tot_ref"],
                     "ref_subq_covered": mv["_tot_cov"], "pred_subq_total": mv["_tot_pred"],
                     "pred_subq_hallucinated": mv["_tot_hal"],
                     "full_cov_samples": int(round(mv["full_cov_pct"] / 100 * mv["_n"]))}
            for metric in ["recall_pct", "halluc_pct", "full_cov_pct"]:
                lo, hi, _ = cluster_bootstrap_ci({label: rows[label]}, split, metric)
                if metric == "recall_pct":
                    w = wilson_ci(mv["_tot_cov"], mv["_tot_ref"])
                elif metric == "halluc_pct":
                    w = wilson_ci(mv["_tot_hal"], mv["_tot_pred"])
                else:
                    w = wilson_ci(entry["full_cov_samples"], mv["_n"])
                entry[metric] = {"value": mv[metric], "ci95_cluster_bootstrap": [lo, hi],
                                 "ci95_wilson": list(w)}
            summary[label][split] = entry

    # ---- paired comparison ----
    significance = {}
    for split in splits:
        sig = {}
        for metric in ["recall_pct", "halluc_pct", "full_cov_pct"]:
            lo, hi, mean_d = cluster_bootstrap_ci(
                {labels[0]: rows[labels[0]], labels[1]: rows[labels[1]]}, split, metric, paired=True)
            sig[metric] = {"delta_progressive_minus_baseline": mean_d,
                           "ci95_paired_bootstrap": [lo, hi],
                           "significant_at_95": (lo > 0) or (hi < 0)}
        sel = [i for i, r in enumerate(rows[labels[0]]) if split == "overall" or r["split"] == split]
        for metric, num, den in [("recall_pct", "n_covered", "n_ref"), ("halluc_pct", "n_halluc", "n_pred")]:
            a = [rows[labels[0]][i][num] / max(rows[labels[0]][i][den], 1e-9) for i in sel]
            b = [rows[labels[1]][i][num] / max(rows[labels[1]][i][den], 1e-9) for i in sel]
            diff = np.array(a) - np.array(b)
            w_p = float(sps.wilcoxon(a, b, zero_method="wilcox")[1]) if np.any(diff != 0) else 1.0
            sig[metric]["wilcoxon_p"] = w_p
        a = [rows[labels[0]][i]["full_cov"] for i in sel]
        b = [rows[labels[1]][i]["full_cov"] for i in sel]
        n01, n10, p = mcnemar_exact(a, b)
        sig["full_cov_pct"]["mcnemar"] = {"baseline_only_wins": n01, "progressive_only_wins": n10, "p": p}
        significance[split] = sig

    # ---- report ----
    print(f"\n{'='*100}")
    print(f"  SUB-QUESTION COMPARISON on {args.test_file} (1 rep, temperature=0, GPT-4o judge)")
    print(f"  value  [95% CI: cluster bootstrap over samples, B={N_BOOT}]")
    print(f"{'='*100}")
    for split in splits:
        print(f"\n  ===== {split}  (n={summary[labels[0]][split]['n_samples']}) =====")
        for metric, name in [("recall_pct", "Sub-Q recall        "),
                             ("halluc_pct", "Sub-Q hallucination "),
                             ("full_cov_pct", "Full-coverage samples")]:
            print(f"    {name}")
            for label in labels:
                m = summary[label][split][metric]
                e = summary[label][split]
                counts = {"recall_pct": f"({e['ref_subq_covered']}/{e['ref_subq_total']})",
                          "halluc_pct": f"({e['pred_subq_hallucinated']}/{e['pred_subq_total']})",
                          "full_cov_pct": f"({e['full_cov_samples']}/{e['n_samples']})"}[metric]
                print(f"      {label:24s} {m['value']:5.1f}  [{m['ci95_cluster_bootstrap'][0]:5.1f}, "
                      f"{m['ci95_cluster_bootstrap'][1]:5.1f}]  {counts}")
            s = significance[split][metric]
            mark = "*" if s["significant_at_95"] else " "
            test = (f"wilcoxon_p={s['wilcoxon_p']:.4g}" if "wilcoxon_p" in s
                    else f"mcnemar_p={s['mcnemar']['p']:.4g}")
            print(f"      {'Δ (prog - base)':24s} {s['delta_progressive_minus_baseline']:+5.1f}  "
                  f"[{s['ci95_paired_bootstrap'][0]:+5.1f}, {s['ci95_paired_bootstrap'][1]:+5.1f}] {mark}  {test}")

    out = {"config": {"test_file": args.test_file, "reps": 1, "temperature": 0, "n_boot": N_BOOT,
                      "judge": "gpt-4o", "models": MODELS, "limit": args.limit},
           "summary": summary, "significance": significance}
    p_out = os.path.join(args.out_dir, "corrected_test250_subq_summary.json")
    json.dump(out, open(p_out, "w"), indent=2)
    json.dump(rows, open(os.path.join(args.out_dir, "corrected_test250_subq_per_sample.json"), "w"),
              indent=2, ensure_ascii=False)
    print(f"\nSaved -> {p_out}")


if __name__ == "__main__":
    main()
