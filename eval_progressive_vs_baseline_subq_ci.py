"""
GPT-4o judged sub-question COVERAGE + HALLUCINATION comparison:
ood_curriculum_progressive (latest curriculum/progressive SFT run) vs
ood_curriculum_baseline_mixed1076 (latest non-progressive baseline SFT run),
both evaluated on ood_data/curriculum_test_120.json.

Spec (per user request):
  - COVERAGE: for every REFERENCE sub-question, does the model's predicted set
    collectively express the same information need? A single predicted
    sub-question may legitimately cover 2+ reference sub-questions (semantic
    coverage, not positional/1:1 alignment) — reuses judge_subq_coverage.py's
    "collectively covers" spec.
  - HALLUCINATION: for every PREDICTED sub-question, is it grounded in (helps
    answer) some reference sub-question's information need, or does it
    introduce content the reference decomposition never asked for? Both
    directions are judged in one GPT-4o call per sample.
  - Broken out by seen_80 (training-template) vs unseen_40 (OOD-template)
    subsets — same UNSEEN_IDX split as score_experiments.py — plus overall.
  - Every proportion reported with a Wilson 95% CI.

Reuses cached greedy-decoding predictions already on disk (both runs were
already evaluated on this exact test file, same sample order):
  nhanes_model/ood_curriculum_progressive/eval_curriculum_test_120/test_predictions.json
  nhanes_model/ood_curriculum_baseline_mixed1076/eval_curriculum_test_120/test_predictions.json

Usage:
    python eval_progressive_vs_baseline_subq_ci.py
    python eval_progressive_vs_baseline_subq_ci.py --limit 10
"""

import argparse
import concurrent.futures
import json
import math
import re
import time

from evaluate import get_openai_client, parse_output, truncate_output, GPT_MODEL

MAX_RETRIES = 8

# Same OOD-template split as score_experiments.py, over curriculum_test_120's
# fixed sample order (curriculum_test_120.json / curriculum_test_120_gold.json
# / both models' eval_curriculum_test_120/test_predictions.json all share it).
UNSEEN_IDX = set([10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89,
                  100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114,
                  115, 116, 117, 118, 119])

PRED_FILES = {
    "progressive": "nhanes_model/ood_curriculum_progressive/eval_curriculum_test_120/test_predictions.json",
    "baseline_mixed1076": "nhanes_model/ood_curriculum_baseline_mixed1076/eval_curriculum_test_120/test_predictions.json",
}


def wilson_ci(k, n, z=1.96):
    """Wilson score interval for a binomial proportion, percent scale."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, center - half) * 100, min(1.0, center + half) * 100)


def judge_sample(client, inp, ref_subs, pred_subs, model=GPT_MODEL):
    ref_list = "\n".join(f"  R{i+1} [{q.sq_type}]: {q.sub_question}" for i, q in enumerate(ref_subs))
    pred_list = "\n".join(f"  P{j+1} [{q.sq_type}]: {q.sub_question}" for j, q in enumerate(pred_subs)) or "  (none)"
    prompt = f"""You are grading a question-decomposition model on two axes: coverage and hallucination.

Original question: "{inp}"

REFERENCE sub-questions (the expected decomposition):
{ref_list}

MODEL's predicted sub-questions:
{pred_list}

TASK 1 — COVERAGE. For EACH reference sub-question Ri, decide:
  - "covered": true if one or more predicted sub-questions together express the
    same information need as Ri. The model MAY be more explicit (spelling out
    model names, the training ratio, "the retrieved probabilities", the patient
    id) and MAY split Ri across several predicted sub-questions - still covered.
    A single predicted sub-question MAY also cover 2+ reference sub-questions at
    once if it bundles their information needs - still covered.
    Only mark false if the information Ri asks for is genuinely absent.
  - "covering": list of predicted indices (e.g. ["P1","P2"]) that cover it, or [].

TASK 2 — HALLUCINATION. For EACH predicted sub-question Pj, decide:
  - "grounded": true if Pj corresponds to (helps answer) the information need of
    at least one reference sub-question above (including partial/split coverage).
    false if Pj asks about something the reference decomposition never asked for
    - e.g. a different entity/attribute/relation, a fabricated detail, or content
    unrelated to answering the original question as scoped by the reference.

Respond JSON only, no markdown:
{{"ref_results": [{{"ref": 1, "covered": true, "covering": ["P1"]}}],
  "pred_results": [{{"pred": 1, "grounded": true}}]}}"""

    for attempt in range(MAX_RETRIES):
        try:
            resp = client.chat.completions.create(
                model=model, messages=[{"role": "user", "content": prompt}],
                temperature=0, max_tokens=1500,
            )
            raw = resp.choices[0].message.content.strip()
            data = json.loads(re.sub(r"```json\s*|\s*```", "", raw).strip())
            ref_res = data.get("ref_results", [])
            pred_res = data.get("pred_results", [])
            ref_out = []
            for i in range(len(ref_subs)):
                r = ref_res[i] if i < len(ref_res) else {}
                ref_out.append(bool(r.get("covered", False)))
            pred_out = []
            for j in range(len(pred_subs)):
                r = pred_res[j] if j < len(pred_res) else {}
                pred_out.append(bool(r.get("grounded", True)))
            return ref_out, pred_out
        except Exception as e:
            msg = str(e)
            wait = 2.0 * (attempt + 1)
            m = re.search(r"try again in ([\d.]+)s", msg)
            if m:
                wait = float(m.group(1)) + 0.5
            if attempt < MAX_RETRIES - 1:
                time.sleep(wait)
            else:
                print(f"  judge failed: {e}")
    return [False] * len(ref_subs), [True] * len(pred_subs)


def load_samples(pred_file, limit=None):
    raw = json.load(open(pred_file, encoding="utf-8"))
    if limit:
        raw = raw[:limit]
    samples = []
    for idx, p in enumerate(raw):
        pred_text = p.get("prediction") or truncate_output(p["prediction_raw"])
        samples.append({
            "idx": idx,
            "input": p["input"],
            "ref": parse_output(p["reference"]),
            "pred": parse_output(pred_text),
            "split": "unseen_40" if idx in UNSEEN_IDX else "seen_80",
        })
    return samples


def judge_all(client, samples, workers):
    def work(i):
        s = samples[i]
        return i, judge_sample(client, s["input"], s["ref"], s["pred"])

    verdicts = [None] * len(samples)
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
        for i, v in ex.map(work, range(len(samples))):
            verdicts[i] = v
    return verdicts


def aggregate(samples, verdicts):
    """Returns per-split + overall stats, plus a per-sample record list."""
    subsets = {"overall": lambda s: True, "seen_80": lambda s: s["split"] == "seen_80",
               "unseen_40": lambda s: s["split"] == "unseen_40"}
    per_sample = []

    for s, (ref_cov, pred_grounded) in zip(samples, verdicts):
        n_ref = len(s["ref"])
        n_cov = sum(ref_cov)
        n_pred = len(s["pred"])
        n_halluc = sum(1 for g in pred_grounded if not g)
        per_sample.append({
            "idx": s["idx"], "input": s["input"], "split": s["split"],
            "n_ref": n_ref, "n_covered": n_cov, "full_coverage": (n_ref > 0 and n_cov == n_ref),
            "n_pred": n_pred, "n_hallucinated": n_halluc, "has_hallucination": n_halluc > 0,
            "missing_refs": [q.sub_question for q, c in zip(s["ref"], ref_cov) if not c],
            "hallucinated_preds": [q.sub_question for q, g in zip(s["pred"], pred_grounded) if not g],
        })

    stats = {}
    for name, pred in subsets.items():
        rows = [r for r in per_sample if pred({"split": r["split"]})]
        n_samples = len(rows)
        tot_ref = sum(r["n_ref"] for r in rows)
        tot_cov = sum(r["n_covered"] for r in rows)
        tot_pred = sum(r["n_pred"] for r in rows)
        tot_halluc = sum(r["n_hallucinated"] for r in rows)
        n_full = sum(1 for r in rows if r["full_coverage"])
        n_has_halluc = sum(1 for r in rows if r["has_hallucination"])

        cov_pct = tot_cov / max(tot_ref, 1) * 100
        cov_ci = wilson_ci(tot_cov, tot_ref)
        halluc_pct = tot_halluc / max(tot_pred, 1) * 100
        halluc_ci = wilson_ci(tot_halluc, tot_pred)
        full_pct = n_full / max(n_samples, 1) * 100
        full_ci = wilson_ci(n_full, n_samples)
        has_halluc_pct = n_has_halluc / max(n_samples, 1) * 100
        has_halluc_ci = wilson_ci(n_has_halluc, n_samples)

        stats[name] = {
            "n_samples": n_samples,
            "ref_subq_coverage_pct": cov_pct, "ref_subq_coverage_ci95": cov_ci,
            "ref_subq_total": tot_ref, "ref_subq_covered": tot_cov,
            "full_sample_coverage_pct": full_pct, "full_sample_coverage_ci95": full_ci,
            "pred_subq_hallucination_pct": halluc_pct, "pred_subq_hallucination_ci95": halluc_ci,
            "pred_subq_total": tot_pred, "pred_subq_hallucinated": tot_halluc,
            "sample_has_hallucination_pct": has_halluc_pct, "sample_has_hallucination_ci95": has_halluc_ci,
        }
    return stats, per_sample


def print_report(label, stats):
    print(f"\n{'='*72}")
    print(f"  {label}")
    print(f"{'='*72}")
    for name in ["overall", "seen_80", "unseen_40"]:
        s = stats[name]
        print(f"\n  -- {name}  (n={s['n_samples']}) --")
        print(f"    Sub-question coverage:      {s['ref_subq_coverage_pct']:.1f}%  "
              f"[{s['ref_subq_coverage_ci95'][0]:.1f}, {s['ref_subq_coverage_ci95'][1]:.1f}]  "
              f"({s['ref_subq_covered']}/{s['ref_subq_total']})")
        print(f"    Full-sample coverage:       {s['full_sample_coverage_pct']:.1f}%  "
              f"[{s['full_sample_coverage_ci95'][0]:.1f}, {s['full_sample_coverage_ci95'][1]:.1f}]")
        print(f"    Predicted sub-Q hallucination: {s['pred_subq_hallucination_pct']:.1f}%  "
              f"[{s['pred_subq_hallucination_ci95'][0]:.1f}, {s['pred_subq_hallucination_ci95'][1]:.1f}]  "
              f"({s['pred_subq_hallucinated']}/{s['pred_subq_total']})")
        print(f"    Samples w/ >=1 hallucination:  {s['sample_has_hallucination_pct']:.1f}%  "
              f"[{s['sample_has_hallucination_ci95'][0]:.1f}, {s['sample_has_hallucination_ci95'][1]:.1f}]")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out_dir", default="cypher_results")
    args = ap.parse_args()

    client = get_openai_client()
    all_stats = {}
    for label, pred_file in PRED_FILES.items():
        samples = load_samples(pred_file, limit=args.limit)
        print(f"Judging {label}: {len(samples)} samples from {pred_file} ...")
        t0 = time.time()
        verdicts = judge_all(client, samples, args.workers)
        print(f"  done in {time.time()-t0:.0f}s")
        stats, per_sample = aggregate(samples, verdicts)
        print_report(label, stats)
        all_stats[label] = stats

        with open(f"{args.out_dir}/{label}_subq_coverage_halluc_ci_per_sample.json", "w") as f:
            json.dump(per_sample, f, indent=2)

    out_path = f"{args.out_dir}/progressive_vs_baseline_mixed1076_subq_coverage_halluc_ci.json"
    with open(out_path, "w") as f:
        json.dump(all_stats, f, indent=2)
    print(f"\nSaved summary -> {out_path}")


if __name__ == "__main__":
    main()
