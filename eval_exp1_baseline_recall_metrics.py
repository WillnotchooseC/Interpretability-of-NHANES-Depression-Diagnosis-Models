"""
Baseline (exp1, no GraphRAG) metrics on ood_data/curriculum_test_120_gold.json,
using progressive-model decompositions and the already-executed Cypher output
cached in results_exp1_baseline_fewshot/ (non-version3 - see
eval_exp3_graphrag_recall_metrics.py's EXP3_FILE comment for why the
non-version3, provenance-audit-free few-shot set is used here too, for the
same reason: Cypher generation is the base model via in-context few-shot, not
the fine-tuned decomposition model under OOD test).

Same methodology and metrics as eval_exp3_graphrag_recall_metrics.py, minus
the irrelevant-grounding judge - exp1 (USE_GRAPHRAG=False) never does the
exact entity/path grounding lookup, so there is no `primary_grounding_rows`
field to audit for that experiment:
  1. GOLD RECALL (strict full-match / lenient >=50% / soft coverage mean)
  2. CYPHER EXECUTION FAILURE RATE
  3. FINAL ANSWER MATCH vs gold_answer (GPT-4o judge)
All broken out overall / seen_80 / unseen_40, each with a 95% CI (Wilson for
proportions, bootstrap for continuous means).

Usage:
    python eval_exp1_baseline_recall_metrics.py
    python eval_exp1_baseline_recall_metrics.py --limit 10
"""

import argparse
import concurrent.futures
import json
import math
import re
import time

import numpy as np
from neo4j import GraphDatabase

from entity_retrieval import build_vocab
from run_finetuned_pipeline_eval import score_against_gold, build_echo_values
from evaluate import get_openai_client, GPT_MODEL

NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "12345678"

GOLD_FILE = "ood_data/curriculum_test_120_gold.json"
EXP1_FILE = "results_exp1_baseline_fewshot/progressive_decomp_for_experiments_exp1_baseline_fewshot.json"

BIOMARKER_ALIASES = {
    "25-hydroxyvitamin d3": "vitamin d3",
    "25-hydroxyvitamin d2": "vitamin d2",
}


def _canon_row(row):
    out = {}
    for k, v in row.items():
        if isinstance(v, str) and v.strip().lower() in BIOMARKER_ALIASES:
            v = BIOMARKER_ALIASES[v.strip().lower()]
        out[k] = v
    return out


def _canon_rows(rows):
    return [_canon_row(r) if isinstance(r, dict) else r for r in rows]


UNSEEN_IDX = set([10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89,
                  100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114,
                  115, 116, 117, 118, 119])

ANSWER_JUDGE_PROMPT = """You are grading a knowledge-graph QA system's final answer against a gold answer.

Question: {question}

Gold answer: {gold}

Model answer: {pred}

Grade the model answer on:
- "correct": do ALL the specific facts it states (numeric values, predicted
  labels, directions, which ratio/model is highest/lowest, rankings) agree with
  the gold answer? Minor rounding (0.19 vs 0.190) is fine. false if any stated
  fact is wrong or contradicts gold.
- "complete": does it answer every part the question asks for (all requested
  values AND any comparison / highest-lowest / ranking)? false if part is missing.
- "no_hallucination": true if it does NOT invent facts absent from the gold /
  evidence. false if it adds unsupported specific claims.
- "score": overall answer quality 0.0-1.0 (1.0 = fully correct, complete, clean).

Respond JSON only:
{{"correct": true, "complete": true, "no_hallucination": true, "score": 1.0}}"""


def wilson_ci(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, center - half) * 100, min(1.0, center + half) * 100)


def bootstrap_mean_ci(values, n_boot=5000, z_lo=2.5, z_hi=97.5, seed=0):
    if not values:
        return (0.0, 0.0, 0.0)
    arr = np.array(values, dtype=float)
    rng = np.random.RandomState(seed)
    means = [rng.choice(arr, size=len(arr), replace=True).mean() for _ in range(n_boot)]
    lo, hi = np.percentile(means, [z_lo, z_hi])
    return (float(arr.mean()), float(lo), float(hi))


def judge_answer(client, question, gold, pred, max_retries=8):
    if not pred.strip():
        return {"correct": False, "complete": False, "no_hallucination": False, "score": 0.0}
    prompt = ANSWER_JUDGE_PROMPT.format(question=question, gold=gold, pred=pred)
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=GPT_MODEL, messages=[{"role": "user", "content": prompt}],
                temperature=0, max_tokens=400,
            )
            raw = resp.choices[0].message.content.strip()
            d = json.loads(re.sub(r"```json\s*|\s*```", "", raw).strip())
            return {
                "correct": bool(d.get("correct", False)),
                "complete": bool(d.get("complete", False)),
                "no_hallucination": bool(d.get("no_hallucination", False)),
                "score": float(d.get("score", 0.0)),
            }
        except Exception as e:
            msg = str(e)
            wait = 2.0 * (attempt + 1)
            m = re.search(r"try again in ([\d.]+)s", msg)
            if m:
                wait = float(m.group(1)) + 0.5
            if attempt < max_retries - 1:
                time.sleep(wait)
            else:
                print(f"  judge failed: {msg[:150]}")
    return {"correct": False, "complete": False, "no_hallucination": False, "score": 0.0}


def gold_rows_for(sample):
    rows = []
    for g in sample.get("gold_results", []):
        r = g.get("result")
        if isinstance(r, list):
            rows.extend(x for x in r if isinstance(x, dict))
        elif isinstance(r, dict):
            rows.append(r)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    gold = json.load(open(GOLD_FILE, encoding="utf-8"))
    exp1 = json.load(open(EXP1_FILE, encoding="utf-8"))
    details = exp1["details"]
    assert len(details) == len(gold), f"{len(details)} details vs {len(gold)} gold samples"

    if args.limit:
        gold = gold[:args.limit]
        details = details[:args.limit]

    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    driver.verify_connectivity()
    vocab = build_vocab(driver)
    client = get_openai_client()

    per_sample = []
    for i, (g, d) in enumerate(zip(gold, details)):
        actual_rows, exec_flags = [], []
        for step in d.get("steps", []):
            exec_flags.append(bool(step.get("exec_success")))
            actual_rows.extend(step.get("records") or [])
        gold_rows = _canon_rows(gold_rows_for(g))
        actual_rows = _canon_rows(actual_rows)
        echoes = build_echo_values(g["input"], vocab)
        sc = score_against_gold(actual_rows, gold_rows, echoes)
        per_sample.append({
            "idx": i, "input": g["input"],
            "split": "unseen_40" if i in UNSEEN_IDX else "seen_80",
            "n_steps": len(exec_flags), "exec_fail_steps": sum(1 for f in exec_flags if not f),
            "n_gold_rows": sc["n_gold_rows"], "full_matches": sc["full_matches"] or 0,
            "partial_matches": sc["partial_matches"] or 0, "coverage": sc["coverage"],
            "gold_answer": g.get("gold_answer", ""), "final_answer": d.get("final_answer") or "",
        })

    print(f"Judging {len(per_sample)} final answers ...")

    def work_answer(i):
        s = per_sample[i]
        return i, judge_answer(client, s["input"], s["gold_answer"], s["final_answer"])

    t0 = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as ex:
        for i, v in ex.map(work_answer, range(len(per_sample))):
            s = per_sample[i]
            s["answer_correct"] = v["correct"]
            s["answer_complete"] = v["complete"]
            s["answer_no_hallucination"] = v["no_hallucination"]
            s["answer_score"] = v["score"]
    print(f"  answer judge done in {time.time()-t0:.0f}s")

    def agg(rows):
        n = len(rows)
        tot_steps = sum(r["n_steps"] for r in rows)
        fail_steps = sum(r["exec_fail_steps"] for r in rows)
        tot_gold = sum(r["n_gold_rows"] for r in rows)
        full = sum(r["full_matches"] for r in rows)
        partial = sum(r["partial_matches"] for r in rows)
        cov_vals = [r["coverage"] for r in rows if r["coverage"] is not None]
        cov_mean, cov_lo, cov_hi = bootstrap_mean_ci(cov_vals)
        n_correct = sum(1 for r in rows if r["answer_correct"])
        n_complete = sum(1 for r in rows if r["answer_complete"])
        n_no_halluc = sum(1 for r in rows if r["answer_no_hallucination"])
        score_vals = [r["answer_score"] for r in rows]
        score_mean, score_lo, score_hi = bootstrap_mean_ci(score_vals)
        return {
            "n_samples": n,
            "cypher_exec_fail_rate_pct": fail_steps / max(tot_steps, 1) * 100,
            "cypher_exec_fail_ci95": wilson_ci(fail_steps, tot_steps),
            "cypher_steps_total": tot_steps, "cypher_steps_failed": fail_steps,
            "gold_recall_full_pct": full / max(tot_gold, 1) * 100,
            "gold_recall_full_ci95": wilson_ci(full, tot_gold),
            "gold_recall_partial_pct": partial / max(tot_gold, 1) * 100,
            "gold_recall_partial_ci95": wilson_ci(partial, tot_gold),
            "gold_rows_total": tot_gold, "gold_rows_full_matched": full,
            "gold_coverage_soft_mean_pct": cov_mean * 100, "gold_coverage_soft_ci95_pct": (cov_lo * 100, cov_hi * 100),
            "answer_correct_pct": n_correct / max(n, 1) * 100, "answer_correct_ci95": wilson_ci(n_correct, n),
            "answer_complete_pct": n_complete / max(n, 1) * 100, "answer_complete_ci95": wilson_ci(n_complete, n),
            "answer_no_hallucination_pct": n_no_halluc / max(n, 1) * 100, "answer_no_hallucination_ci95": wilson_ci(n_no_halluc, n),
            "answer_score_mean": score_mean, "answer_score_ci95": (score_lo, score_hi),
        }

    stats = {
        "overall": agg(per_sample),
        "seen_80": agg([r for r in per_sample if r["split"] == "seen_80"]),
        "unseen_40": agg([r for r in per_sample if r["split"] == "unseen_40"]),
    }

    driver.close()

    def pct_ci(v, ci):
        return f"{v:5.1f}%  [{ci[0]:.1f}, {ci[1]:.1f}]"

    print(f"\n{'='*78}\n  exp1_baseline_fewshot metrics (progressive decomp, no GraphRAG)\n{'='*78}")
    for name in ["overall", "seen_80", "unseen_40"]:
        s = stats[name]
        print(f"\n  -- {name}  (n={s['n_samples']}) --")
        print(f"    Cypher exec failure rate:        {pct_ci(s['cypher_exec_fail_rate_pct'], s['cypher_exec_fail_ci95'])}  ({s['cypher_steps_failed']}/{s['cypher_steps_total']})")
        print(f"    Gold recall (strict full-match): {pct_ci(s['gold_recall_full_pct'], s['gold_recall_full_ci95'])}  ({s['gold_rows_full_matched']}/{s['gold_rows_total']})")
        print(f"    Gold recall (lenient >=50%):     {pct_ci(s['gold_recall_partial_pct'], s['gold_recall_partial_ci95'])}")
        print(f"    Gold coverage (soft mean):       {s['gold_coverage_soft_mean_pct']:5.1f}%  [{s['gold_coverage_soft_ci95_pct'][0]:.1f}, {s['gold_coverage_soft_ci95_pct'][1]:.1f}]")
        print(f"    Final answer correct:            {pct_ci(s['answer_correct_pct'], s['answer_correct_ci95'])}")
        print(f"    Final answer complete:           {pct_ci(s['answer_complete_pct'], s['answer_complete_ci95'])}")
        print(f"    Final answer no-hallucination:   {pct_ci(s['answer_no_hallucination_pct'], s['answer_no_hallucination_ci95'])}")
        print(f"    Final answer score (mean):       {s['answer_score_mean']:.3f}  [{s['answer_score_ci95'][0]:.3f}, {s['answer_score_ci95'][1]:.3f}]")

    with open("cypher_results/exp1_baseline_recall_metrics_summary.json", "w") as f:
        json.dump(stats, f, indent=2)
    with open("cypher_results/exp1_baseline_recall_metrics_per_sample.json", "w") as f:
        json.dump(per_sample, f, indent=2, ensure_ascii=False)
    print("\nSaved -> cypher_results/exp1_baseline_recall_metrics_summary.json + _per_sample.json")


if __name__ == "__main__":
    main()
