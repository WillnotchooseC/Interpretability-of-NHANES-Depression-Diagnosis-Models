"""
Per-sub-question (per gold decomposition step) coverage of gold_results by
GraphRAG's actually-executed Cypher (exp3 rescue, version3), on
ood_data/curriculum_test_120_gold.json.

Unlike eval_exp3_graphrag_recall_metrics.py (which pools ALL of a sample's
gold_results rows together before matching, since the model's own predicted
decomposition can have a different number/order of searching steps than the
gold decomposition — so predicted-step-i cannot be safely assumed to
correspond to gold-step-i), this reports coverage separately for EACH
gold_results entry ("step": N, "result": [...]) — i.e. each of the gold
decomposition's own sub-questions may carry several result dicts, and this
answers "of a given sub-question's expected result rows, how many did
GraphRAG actually find?" per sub-question, not just one number per sample.

Matching is still against the full pool of rows GraphRAG's Cypher actually
retrieved for that sample (same greedy field-value-overlap containment,
echo-filtered, as run_finetuned_pipeline_eval.score_against_gold) - only the
GOLD side is now split per step instead of pooled.

Usage:
    python eval_exp3_per_subquestion_coverage.py
    python eval_exp3_per_subquestion_coverage.py --limit 10
"""

import argparse
import json
import math

from neo4j import GraphDatabase

from entity_retrieval import build_vocab
from run_finetuned_pipeline_eval import build_echo_values, _row_vals
from evaluate import parse_output

NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "12345678"

GOLD_FILE = "ood_data/curriculum_test_120_gold.json"
# See eval_exp3_graphrag_recall_metrics.py's EXP3_FILE comment: non-version3 is
# used here because its few-shot demos aren't provenance-audit-constrained,
# which matters for Cypher-generation quality, not decomposition OOD testing.
EXP3_FILE = "results_exp3_fewshot_graphrag_rescue/progressive_decomp_for_experiments_exp3_fewshot_graphrag_rescue.json"

# gold_results uses the clinical lab-test name for these two biomarkers, but the
# graph's `biomarker.name` property only ever uses the short form (confirmed via
# `MATCH (b:biomarker) RETURN DISTINCT b.name`) - see eval_exp3_graphrag_recall_metrics.py
# for the full note. No other biomarker name has this mismatch.
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


def wilson_ci(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, center - half) * 100, min(1.0, center + half) * 100)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    gold = json.load(open(GOLD_FILE, encoding="utf-8"))
    exp3 = json.load(open(EXP3_FILE, encoding="utf-8"))
    details = exp3["details"]
    assert len(details) == len(gold)
    if args.limit:
        gold, details = gold[:args.limit], details[:args.limit]

    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    driver.verify_connectivity()
    vocab = build_vocab(driver)

    subq_rows = []  # one row per (sample, gold step)
    for i, (g, d) in enumerate(zip(gold, details)):
        split = "unseen_40" if i in UNSEEN_IDX else "seen_80"
        actual_rows = []
        for step in d.get("steps", []):
            actual_rows.extend(step.get("records") or [])
        actual_rows = _canon_rows(actual_rows)
        echoes = build_echo_values(g["input"], vocab)

        sub_text = {s.step_num: s.sub_question for s in parse_output(g["output"])}

        for gr in g.get("gold_results", []):
            step_num = gr.get("step")
            result = gr.get("result")
            rows_raw = result if isinstance(result, list) else ([result] if isinstance(result, dict) else [])
            rows_raw = [r for r in rows_raw if isinstance(r, dict)]
            rows = _canon_rows(rows_raw)  # normalized for matching; rows_raw kept for display

            # which specific dicts were/weren't covered (full-match)
            covered_flags = []
            remaining = [( _row_vals(r) - echoes, r) for r in actual_rows]
            for gold_row in rows:
                gvals = _row_vals(gold_row) - echoes
                if not gvals:
                    covered_flags.append(True)  # nothing to check (all echoed)
                    continue
                best_idx, best_score = None, -1.0
                for k, (avals, _raw) in enumerate(remaining):
                    if not avals:
                        continue
                    s = len(gvals & avals) / len(gvals)
                    if s > best_score:
                        best_score, best_idx = s, k
                is_full = best_idx is not None and best_score >= 0.999
                covered_flags.append(is_full)
                if best_idx is not None and best_score > 0:
                    del remaining[best_idx]

            subq_rows.append({
                "idx": i, "split": split, "input": g["input"], "step": step_num,
                "sub_question": sub_text.get(step_num, "(unknown)"),
                "n_results": len(rows), "n_covered": sum(covered_flags),
                "coverage_pct": (sum(covered_flags) / len(rows) * 100) if rows else None,
                "results_covered": [{"result": r, "covered": c} for r, c in zip(rows_raw, covered_flags)],
            })

    driver.close()

    def agg(rows_subset):
        rows_subset = [r for r in rows_subset if r["n_results"] > 0]
        n_steps = len(rows_subset)
        tot_dicts = sum(r["n_results"] for r in rows_subset)
        tot_covered = sum(r["n_covered"] for r in rows_subset)
        n_full_step = sum(1 for r in rows_subset if r["n_covered"] == r["n_results"])
        n_zero_step = sum(1 for r in rows_subset if r["n_covered"] == 0)
        return {
            "n_subquestions": n_steps,
            "dict_level_coverage_pct": tot_covered / max(tot_dicts, 1) * 100,
            "dict_level_ci95": wilson_ci(tot_covered, tot_dicts),
            "dicts_total": tot_dicts, "dicts_covered": tot_covered,
            "subq_fully_covered_pct": n_full_step / max(n_steps, 1) * 100,
            "subq_fully_covered_ci95": wilson_ci(n_full_step, n_steps),
            "subq_zero_covered_pct": n_zero_step / max(n_steps, 1) * 100,
            "subq_zero_covered_ci95": wilson_ci(n_zero_step, n_steps),
        }

    summary = {
        "overall": agg(subq_rows),
        "seen_80": agg([r for r in subq_rows if r["split"] == "seen_80"]),
        "unseen_40": agg([r for r in subq_rows if r["split"] == "unseen_40"]),
    }

    print(f"\n{'='*90}\n  Per-sub-question gold_results coverage (exp3 rescue, dict-level, full-match)\n{'='*90}")
    for name in ["overall", "seen_80", "unseen_40"]:
        s = summary[name]
        print(f"\n  -- {name}  ({s['n_subquestions']} sub-questions with >=1 gold result) --")
        print(f"    Dict-level coverage:        {s['dict_level_coverage_pct']:5.1f}%  [{s['dict_level_ci95'][0]:.1f}, {s['dict_level_ci95'][1]:.1f}]  ({s['dicts_covered']}/{s['dicts_total']})")
        print(f"    Sub-question fully covered: {s['subq_fully_covered_pct']:5.1f}%  [{s['subq_fully_covered_ci95'][0]:.1f}, {s['subq_fully_covered_ci95'][1]:.1f}]")
        print(f"    Sub-question zero covered:  {s['subq_zero_covered_pct']:5.1f}%  [{s['subq_zero_covered_ci95'][0]:.1f}, {s['subq_zero_covered_ci95'][1]:.1f}]")

    with open("cypher_results/exp3_per_subquestion_coverage_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    with open("cypher_results/exp3_per_subquestion_coverage_detail.json", "w") as f:
        json.dump(subq_rows, f, indent=2, ensure_ascii=False, default=str)
    print("\nSaved -> cypher_results/exp3_per_subquestion_coverage_summary.json + _detail.json")


if __name__ == "__main__":
    main()
