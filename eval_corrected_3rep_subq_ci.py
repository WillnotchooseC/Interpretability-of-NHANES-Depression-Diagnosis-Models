"""
3-repetition statistical comparison of the two *_corrected SFT runs on
ood_data/curriculum_test_120.json, scored ONLY on the sub-question axes that
matter downstream (the sub-questions are what the KG pipeline actually consumes;
<PLAN> / "Step N:" text is auxiliary chain-of-thought and is ignored by
parse_output):

  1. Sub-question RECALL      -- reference sub-Qs whose information need the
                                 predicted set collectively covers
  2. Sub-question HALLUCINATION -- predicted sub-Qs grounded in no reference
                                 sub-question's information need
  3. FULL-SAMPLE COVERAGE     -- samples where every reference sub-Q is covered

Reps:
  rep1 = nhanes_model/<run>/eval_curriculum_test_120/test_predictions.json
         (the cached eval_plan_composition.py run)
  rep2/rep3 = nhanes_model/<run>/eval_curriculum_test_120_rep{2,3}/...
         (produced by gen_test120_reps.py, identical decoding config)

Judging reuses eval_progressive_vs_baseline_subq_ci.judge_sample verbatim, so
these numbers are directly comparable to that script's single-rep output for the
non-corrected models. Each rep is judged independently, so cross-rep spread
captures BOTH SGLang's run-to-run nondeterminism and GPT-4o judge noise.

Statistics:
  - Per-rep point estimate, plus across-rep mean +/- std.
  - 95% CI by CLUSTER bootstrap resampling SAMPLES (not sub-questions): recall
    and hallucination are ratios-of-sums over sub-questions nested within
    samples, so a plain binomial/Wilson interval treats correlated sub-questions
    as independent and is too narrow. Wilson is still reported in the JSON for
    continuity with eval_progressive_vs_baseline_subq_ci.py.
  - Model comparison is PAIRED (both models answer the same 120 questions):
    paired cluster bootstrap on the difference (shared resampled indices),
    McNemar exact on per-sample full coverage (majority vote across reps), and
    paired Wilcoxon on rep-averaged per-sample rates.

Usage:
    python eval_corrected_3rep_subq_ci.py
    python eval_corrected_3rep_subq_ci.py --limit 10 --workers 4
    python eval_corrected_3rep_subq_ci.py --skip_judge   # reuse cached verdicts
"""

import argparse
import json
import math
import os
import time

import numpy as np
from scipy import stats as sps

from evaluate import get_openai_client
from eval_progressive_vs_baseline_subq_ci import (
    judge_all, load_samples, wilson_ci, UNSEEN_IDX,
)

MODELS = {
    "progressive_corrected": "ood_curriculum_progressive_corrected",
    "baseline_corrected": "ood_curriculum_baseline_mixed1076_corrected",
}
REPS = [1, 2, 3]
SPLITS = ["overall", "seen_80", "unseen_40"]
N_BOOT = 10000
SEED = 20260911


def pred_path(run_name, rep):
    sub = "eval_curriculum_test_120" if rep == 1 else f"eval_curriculum_test_120_rep{rep}"
    return os.path.join("nhanes_model", run_name, sub, "test_predictions.json")


def per_sample_counts(samples, verdicts):
    """One record per sample: raw counts needed by every metric downstream."""
    out = []
    for s, (ref_cov, pred_grounded) in zip(samples, verdicts):
        n_ref = len(s["ref"])
        n_cov = sum(ref_cov)
        out.append({
            "idx": s["idx"], "split": s["split"],
            "n_ref": n_ref, "n_covered": n_cov,
            "n_pred": len(s["pred"]),
            "n_halluc": sum(1 for g in pred_grounded if not g),
            "full_cov": 1.0 if (n_ref > 0 and n_cov == n_ref) else 0.0,
        })
    return out


def metric_values(rows):
    """Point estimates for the three metrics over a list of per-sample rows."""
    tot_ref = sum(r["n_ref"] for r in rows)
    tot_cov = sum(r["n_covered"] for r in rows)
    tot_pred = sum(r["n_pred"] for r in rows)
    tot_hal = sum(r["n_halluc"] for r in rows)
    n = len(rows)
    return {
        "recall_pct": tot_cov / max(tot_ref, 1e-9) * 100,
        "halluc_pct": tot_hal / max(tot_pred, 1e-9) * 100,
        "full_cov_pct": sum(r["full_cov"] for r in rows) / max(n, 1) * 100,
        "_tot_ref": tot_ref, "_tot_cov": tot_cov,
        "_tot_pred": tot_pred, "_tot_hal": tot_hal,
        "_n": n,
    }


def rep_average(rep_rows):
    """Average the 3 reps' counts per sample (same sample order in every rep)."""
    n_samples = len(rep_rows[0])
    avg = []
    for i in range(n_samples):
        rs = [rr[i] for rr in rep_rows]
        assert len({r["idx"] for r in rs}) == 1, "rep sample order mismatch"
        avg.append({
            "idx": rs[0]["idx"], "split": rs[0]["split"],
            "n_ref": float(np.mean([r["n_ref"] for r in rs])),
            "n_covered": float(np.mean([r["n_covered"] for r in rs])),
            "n_pred": float(np.mean([r["n_pred"] for r in rs])),
            "n_halluc": float(np.mean([r["n_halluc"] for r in rs])),
            "full_cov": float(np.mean([r["full_cov"] for r in rs])),
            # majority vote across reps, for the paired McNemar test
            "full_cov_majority": 1.0 if np.mean([r["full_cov"] for r in rs]) >= 0.5 else 0.0,
        })
    return avg


def cluster_bootstrap_ci(rows_by_model, split, metric, paired=False, n_boot=N_BOOT, seed=SEED):
    """
    Percentile CI by resampling SAMPLES (clusters) with replacement.

    rows_by_model: {label: [per-sample rows]} (rep-averaged), same sample order.
    If paired, returns the CI of (first label - second label) using shared draws.
    """
    labels = list(rows_by_model)
    idxs = [i for i, r in enumerate(rows_by_model[labels[0]]) if split == "overall" or r["split"] == split]
    rng = np.random.default_rng(seed)
    n = len(idxs)

    def stat(rows, sel):
        sub = [rows[i] for i in sel]
        v = metric_values(sub)
        return v[metric]

    draws = []
    for _ in range(n_boot):
        sel = [idxs[j] for j in rng.integers(0, n, size=n)]
        if paired:
            draws.append(stat(rows_by_model[labels[0]], sel) - stat(rows_by_model[labels[1]], sel))
        else:
            draws.append(stat(rows_by_model[labels[0]], sel))
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return float(lo), float(hi), float(np.mean(draws))


def mcnemar_exact(a, b):
    """Exact McNemar on paired binaries a,b (1=success). Returns (n01, n10, p)."""
    n01 = sum(1 for x, y in zip(a, b) if x == 0 and y == 1)
    n10 = sum(1 for x, y in zip(a, b) if x == 1 and y == 0)
    n = n01 + n10
    if n == 0:
        return n01, n10, 1.0
    p = min(1.0, 2 * sps.binom.cdf(min(n01, n10), n, 0.5))
    return n01, n10, float(p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out_dir", default="cypher_results")
    ap.add_argument("--cache", default="cypher_results/corrected_3rep_judge_cache.json")
    ap.add_argument("--skip_judge", action="store_true",
                    help="Reuse cached judge verdicts only; fail if any rep is missing.")
    ap.add_argument("--max_empty", type=int, default=0,
                    help="Abort if a rep has more than this many samples with zero parsed "
                         "sub-questions (default 0: any empty prediction is treated as a "
                         "failed inference run, not as data).")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    cache = {}
    if os.path.exists(args.cache):
        cache = json.load(open(args.cache))

    client = None
    # {label: {rep: [per-sample rows]}}
    rows = {label: {} for label in MODELS}
    pred_texts = {label: {} for label in MODELS}

    for label, run_name in MODELS.items():
        for rep in REPS:
            path = pred_path(run_name, rep)
            if not os.path.exists(path):
                raise FileNotFoundError(f"missing predictions for {label} rep{rep}: {path}")
            samples = load_samples(path, limit=args.limit)

            # Guard against silently-poisoned reps. An inference timeout makes
            # run_inference_sglang_batch return "" for that sample; the empty
            # prediction then scores as "nothing covered" (deflating recall) AND
            # contributes 0 predicted sub-questions (so the hallucination
            # denominator shrinks and the rate can even look like a perfect 0%).
            # That failure is indistinguishable from a plausible bad result, so
            # refuse to score it rather than report it.
            n_no_subq = sum(1 for s in samples if len(s["pred"]) == 0)
            if n_no_subq > args.max_empty:
                raise SystemExit(
                    f"ABORT: {label} rep{rep} has {n_no_subq}/{len(samples)} samples with "
                    f"zero parsed sub-questions (> --max_empty={args.max_empty}).\n"
                    f"  {path}\n"
                    f"  This is almost certainly inference timeouts, not model behaviour "
                    f"(check the generation log for 'Read timed out'). Regenerate this rep "
                    f"with gen_test120_reps.py before scoring."
                )
            if n_no_subq:
                print(f"  note: {label} rep{rep} has {n_no_subq} sample(s) with no parsed "
                      f"sub-questions (within --max_empty tolerance)")

            pred_texts[label][rep] = [
                " | ".join(q.sub_question for q in s["pred"]) for s in samples
            ]
            key = f"{label}|rep{rep}|{args.limit}"
            if key in cache:
                verdicts = [(v[0], v[1]) for v in cache[key]]
                print(f"cache hit: {label} rep{rep}")
            elif args.skip_judge:
                raise SystemExit(f"--skip_judge but no cached verdicts for {key}")
            else:
                if client is None:
                    client = get_openai_client()
                print(f"Judging {label} rep{rep}: {len(samples)} samples ...")
                t0 = time.time()
                verdicts = judge_all(client, samples, args.workers)
                print(f"  done in {time.time()-t0:.0f}s")
                cache[key] = [[list(v[0]), list(v[1])] for v in verdicts]
                json.dump(cache, open(args.cache, "w"))
            rows[label][rep] = per_sample_counts(samples, verdicts)

    # ---- run-to-run determinism check (did the reps actually differ?) ----
    determinism = {}
    for label in MODELS:
        base = pred_texts[label][1]
        same = {}
        for rep in REPS[1:]:
            ident = sum(1 for a, b in zip(base, pred_texts[label][rep]) if a == b)
            same[f"rep1_vs_rep{rep}_identical_samples"] = f"{ident}/{len(base)}"
        determinism[label] = same

    # ---- per-rep point estimates + across-rep mean/std ----
    summary = {}
    avg_rows = {label: rep_average([rows[label][r] for r in REPS]) for label in MODELS}

    for label in MODELS:
        summary[label] = {}
        for split in SPLITS:
            per_rep = []
            for rep in REPS:
                sub = [r for r in rows[label][rep] if split == "overall" or r["split"] == split]
                per_rep.append(metric_values(sub))
            entry = {"n_samples": per_rep[0]["_n"]}
            for metric in ["recall_pct", "halluc_pct", "full_cov_pct"]:
                vals = [pr[metric] for pr in per_rep]
                lo, hi, _ = cluster_bootstrap_ci({label: avg_rows[label]}, split, metric)
                # Wilson on the rep-1 counts, for continuity with the single-rep script
                if metric == "recall_pct":
                    w = wilson_ci(per_rep[0]["_tot_cov"], per_rep[0]["_tot_ref"])
                elif metric == "halluc_pct":
                    w = wilson_ci(per_rep[0]["_tot_hal"], per_rep[0]["_tot_pred"])
                else:
                    w = wilson_ci(round(per_rep[0]["full_cov_pct"] / 100 * per_rep[0]["_n"]), per_rep[0]["_n"])
                entry[metric] = {
                    "per_rep": [round(v, 2) for v in vals],
                    "mean": float(np.mean(vals)),
                    "std": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
                    "ci95_cluster_bootstrap": [lo, hi],
                    "ci95_wilson_rep1": list(w),
                }
            summary[label][split] = entry

    # ---- paired model comparison ----
    labels = list(MODELS)
    significance = {}
    for split in SPLITS:
        sig = {}
        for metric in ["recall_pct", "halluc_pct", "full_cov_pct"]:
            lo, hi, mean_d = cluster_bootstrap_ci(
                {labels[0]: avg_rows[labels[0]], labels[1]: avg_rows[labels[1]]},
                split, metric, paired=True,
            )
            sig[metric] = {
                "delta_progressive_minus_baseline": mean_d,
                "ci95_paired_bootstrap": [lo, hi],
                "significant_at_95": (lo > 0) or (hi < 0),
            }
        # paired Wilcoxon on rep-averaged per-sample rates
        sel = [i for i, r in enumerate(avg_rows[labels[0]]) if split == "overall" or r["split"] == split]
        for metric, num, den in [("recall_pct", "n_covered", "n_ref"),
                                 ("halluc_pct", "n_halluc", "n_pred")]:
            a = [avg_rows[labels[0]][i][num] / max(avg_rows[labels[0]][i][den], 1e-9) for i in sel]
            b = [avg_rows[labels[1]][i][num] / max(avg_rows[labels[1]][i][den], 1e-9) for i in sel]
            diff = np.array(a) - np.array(b)
            if np.any(diff != 0):
                w_stat, w_p = sps.wilcoxon(a, b, zero_method="wilcox")
            else:
                w_stat, w_p = float("nan"), 1.0
            sig[metric]["wilcoxon_p"] = float(w_p)
        # McNemar on majority-vote full coverage
        a = [avg_rows[labels[0]][i]["full_cov_majority"] for i in sel]
        b = [avg_rows[labels[1]][i]["full_cov_majority"] for i in sel]
        n01, n10, p = mcnemar_exact(a, b)
        sig["full_cov_pct"]["mcnemar"] = {
            "baseline_only_wins": n01, "progressive_only_wins": n10, "p": p,
        }
        significance[split] = sig

    # ---- report ----
    def fmt(entry, metric):
        m = entry[metric]
        return (f"{m['mean']:5.1f} ± {m['std']:.1f}  "
                f"[{m['ci95_cluster_bootstrap'][0]:5.1f}, {m['ci95_cluster_bootstrap'][1]:5.1f}]  "
                f"reps={m['per_rep']}")

    print(f"\n{'='*100}")
    print("  3-REP SUB-QUESTION COMPARISON (temperature=0, 120 questions, GPT-4o judge)")
    print(f"  mean ± std across 3 reps  [95% CI: cluster bootstrap over samples, B={N_BOOT}]")
    print(f"{'='*100}")
    for split in SPLITS:
        print(f"\n  ===== {split}  (n={summary[labels[0]][split]['n_samples']}) =====")
        for metric, name in [("recall_pct", "Sub-Q recall        "),
                             ("halluc_pct", "Sub-Q hallucination "),
                             ("full_cov_pct", "Full-coverage samples")]:
            print(f"    {name}")
            for label in labels:
                print(f"      {label:24s} {fmt(summary[label][split], metric)}")
            s = significance[split][metric]
            mark = "*" if s["significant_at_95"] else " "
            if "wilcoxon_p" in s:
                test = f"wilcoxon_p={s['wilcoxon_p']:.4g}"
            else:
                test = f"mcnemar_p={s['mcnemar']['p']:.4g}"
            print(f"      {'Δ (prog - base)':24s} {s['delta_progressive_minus_baseline']:+5.1f}  "
                  f"[{s['ci95_paired_bootstrap'][0]:+5.1f}, "
                  f"{s['ci95_paired_bootstrap'][1]:+5.1f}] {mark}  {test}")

    print(f"\n  Run-to-run determinism (identical predicted sub-question strings vs rep1):")
    for label, d in determinism.items():
        print(f"    {label:24s} {d}")

    out = {
        "config": {"test_file": "ood_data/curriculum_test_120.json", "reps": REPS,
                   "temperature": 0, "n_boot": N_BOOT, "judge": "gpt-4o",
                   "models": MODELS},
        "determinism": determinism,
        "summary": summary,
        "significance": significance,
    }
    p_out = os.path.join(args.out_dir, "corrected_3rep_subq_summary.json")
    json.dump(out, open(p_out, "w"), indent=2)
    json.dump({l: avg_rows[l] for l in labels},
              open(os.path.join(args.out_dir, "corrected_3rep_subq_per_sample.json"), "w"), indent=2)
    print(f"\nSaved -> {p_out}")


if __name__ == "__main__":
    main()
