"""
Ablation: how much of GraphRAG v2's unseen-template advantage over the few-shot
baseline comes from ONE prompt rule rather than from graph retrieval?

Arms (all on ood_data/curriculum_test_200_gold.json, same decomposition input
ood_data/progressive_corrected_decomp_test200.json, same Qwen3-8B-Base greedy
Cypher LLM, same 12 demos / STATIC_SCHEMA / HARD_RULES, sequential steps):

  fewshot            exp1_5_baseline_original_input   no graph access, no rule
  fewshot_notruncate exp1_6_baseline_notruncate       no graph access, + rule
  graphrag_v2        exp2_fewshot_graphrag_v2         grounding + rules 10-13
                                                      + repair + the rule

The rule (`no_truncate_rule`) is injected iff infer_query_type(sub-question) ==
"local_max_abs" and tells the model not to add a LIMIT, because the sub-question's
literal top-K wording can drop a whole SHAP sign/direction that the ORIGINAL
question needs. It is live in exp2/v2 but was absent from exp1.5, which makes it
a confound in "GraphRAG wins on unseen". exp1_6 adds exactly that rule to the
few-shot arm and nothing else (trigger verified identical to v2 on all 39/39
local_max_abs steps; the other 250 prompts byte-identical to exp1.5's).

Metric: graph_coverage per sample = run_finetuned_pipeline_eval.score_against_gold
(pooled value-set row containment vs pooled gold_results, filter-echo values
removed) -- identical to score_experiments*.py / score_test200_arms.py, so the
numbers are comparable to every earlier table. No OpenAI calls.

Beyond the usual overall/seen/unseen split this also reports the STRUCTURAL
slices that localise the effect (measured 2026-10-14): single-step samples,
multi-step samples without coreference, and multi-step samples whose later
sub-question refers back to an earlier step ("those 4 selected biomarkers") --
the last group is where the whole few-shot-vs-v2 unseen gap lives.

Usage:
    python score_test200_notruncate_ablation.py
    python score_test200_notruncate_ablation.py --rep 1 --out cypher_results/test200_notruncate_ablation.json
"""

import argparse
import json
import os
import re

import numpy as np
from neo4j import GraphDatabase
from scipy import stats as sps

from run_finetuned_pipeline_eval import score_against_gold, build_echo_values
from entity_retrieval import build_vocab
from score_experiments import gold_rows_for

NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "12345678")

ARMS = {
    "fewshot": "cypher_results/test200_fewshot_exp1_5_3reps.json",
    "fewshot_notruncate": "cypher_results/test200_fewshot_notruncate_exp1_6_reps.json",
    "graphrag_v2": "cypher_results/test200_graphrag_v2_exp2_3reps.json",
}
PAIRS = [("fewshot_notruncate", "fewshot"),   # what the rule alone buys
         ("graphrag_v2", "fewshot_notruncate"),  # what graph retrieval adds on top of the rule
         ("graphrag_v2", "fewshot")]          # the original headline gap
COREF_RE = re.compile(r"\bthose\b|\bthat biomarker\b|\bselected biomarker", re.I)
LIMIT_RE = re.compile(r"\bLIMIT\b", re.I)
N_BOOT = 10000


def boot_ci(x, seed=0):
    rng = np.random.default_rng(seed)
    x = np.asarray(x, float)
    if len(x) == 0:
        return (float("nan"), float("nan"))
    draws = [rng.choice(x, size=len(x), replace=True).mean() for _ in range(N_BOOT)]
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return float(lo), float(hi)


def structure_of(steps):
    """Structural class of a sample, from its own sub-questions only."""
    n = len(steps)
    coref = any(COREF_RE.search(s.get("question") or "") for s in steps)
    if n <= 1:
        return "single_step"
    return "multistep_coref" if coref else "multistep_no_coref"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rep", type=int, default=1)
    ap.add_argument("--out", default="cypher_results/test200_notruncate_ablation.json")
    args = ap.parse_args()

    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    driver.verify_connectivity()
    vocab = build_vocab(driver)
    driver.close()

    per_arm, missing = {}, []
    for arm, path in ARMS.items():
        if not os.path.exists(path):
            missing.append(arm)
            continue
        recs = json.load(open(path, encoding="utf-8"))["records"]
        rows = []
        for r in recs:
            steps = r["reps"][f"rep{args.rep}"]["steps"]
            actual = [x for st in steps for x in (st.get("records") or [])]
            sc = score_against_gold(actual, gold_rows_for(r), build_echo_values(r["input"], vocab))
            rows.append({
                "idx": r["idx"], "split": r["split"], "structure": structure_of(steps),
                "graph_coverage": sc["coverage"] if sc["coverage"] is not None else 0.0,
                "n_steps": len(steps),
                "exec_ok": sum(1 for st in steps if st.get("exec_success")),
                "nonempty": sum(1 for st in steps if st.get("has_results")),
                "step1_has_limit": bool(steps and LIMIT_RE.search(steps[0].get("cypher") or "")),
                "step1_rows": (steps[0].get("rows") if steps else None),
                "total_rows": sum((st.get("rows") or 0) for st in steps),
            })
        per_arm[arm] = sorted(rows, key=lambda r: r["idx"])
    if missing:
        print(f"!! missing arms (run them first): {missing}")
    if "fewshot_notruncate" in missing:
        raise SystemExit("the ablation arm has no results yet -- nothing to compare")

    SLICES = {
        "overall": lambda r: True,
        "seen": lambda r: r["split"] == "seen",
        "unseen": lambda r: r["split"] == "unseen",
        "unseen_single_step": lambda r: r["split"] == "unseen" and r["structure"] == "single_step",
        "unseen_multistep_no_coref": lambda r: r["split"] == "unseen" and r["structure"] == "multistep_no_coref",
        "unseen_multistep_coref": lambda r: r["split"] == "unseen" and r["structure"] == "multistep_coref",
    }

    summary = {}
    for arm, rows in per_arm.items():
        summary[arm] = {}
        for sp, keep in SLICES.items():
            sub = [r for r in rows if keep(r)]
            if not sub:
                continue
            cov = [r["graph_coverage"] for r in sub]
            n_steps = sum(r["n_steps"] for r in sub)
            summary[arm][sp] = {
                "n": len(sub),
                "graph_coverage_mean": float(np.mean(cov)),
                "graph_coverage_ci95": boot_ci(cov),
                "exec_rate": sum(r["exec_ok"] for r in sub) / max(n_steps, 1),
                "nonempty_rate": sum(r["nonempty"] for r in sub) / max(n_steps, 1),
                "step1_limit_rate": float(np.mean([r["step1_has_limit"] for r in sub])),
                "rows_returned_total": sum(r["total_rows"] for r in sub),
            }

    sig = {}
    for a, b in PAIRS:
        if a not in per_arm or b not in per_arm:
            continue
        sig[f"{a}_minus_{b}"] = {}
        for sp, keep in SLICES.items():
            ia = [r for r in per_arm[a] if keep(r)]
            ib = [r for r in per_arm[b] if keep(r)]
            if not ia or len(ia) != len(ib):
                continue
            assert [r["idx"] for r in ia] == [r["idx"] for r in ib], f"{a}/{b} idx misalignment in {sp}"
            x = np.array([r["graph_coverage"] for r in ia])
            y = np.array([r["graph_coverage"] for r in ib])
            d = x - y
            if np.any(d != 0):
                w_p = float(sps.wilcoxon(x, y, zero_method="wilcox")[1])
                cohen = float(d.mean() / d.std(ddof=1)) if d.std(ddof=1) > 0 else 0.0
            else:
                w_p, cohen = 1.0, 0.0
            lo, hi = boot_ci(d)
            sig[f"{a}_minus_{b}"][sp] = {
                "delta_mean": float(d.mean()), "ci95_paired_bootstrap": [lo, hi],
                "significant_at_95": bool(lo > 0 or hi < 0), "wilcoxon_p": w_p, "cohen_d": cohen,
                "a_better": int((d > 0).sum()), "b_better": int((d < 0).sum()), "tie": int((d == 0).sum()),
            }

    # ---- attribution: fraction of v2's advantage reproduced by the rule alone ----
    attribution = {}
    if {"fewshot", "fewshot_notruncate", "graphrag_v2"} <= set(per_arm):
        for sp in SLICES:
            try:
                rule = sig["fewshot_notruncate_minus_fewshot"][sp]["delta_mean"]
                full = sig["graphrag_v2_minus_fewshot"][sp]["delta_mean"]
            except KeyError:
                continue
            attribution[sp] = {
                "rule_only_delta": rule,
                "v2_total_delta": full,
                "share_of_v2_gap_from_rule": (rule / full) if abs(full) > 1e-9 else None,
                "residual_delta_attributable_to_graphrag": full - rule,
            }

    # ---- report ----
    W = 104
    print(f"\n{'='*W}\n  test_200 no-truncate-rule ablation (rep{args.rep}) -- graph_coverage vs pooled gold_results\n{'='*W}")
    for sp in SLICES:
        any_arm = next((a for a in per_arm if sp in summary[a]), None)
        if any_arm is None:
            continue
        print(f"\n  ===== {sp} (n={summary[any_arm][sp]['n']}) =====")
        print(f"    {'arm':20s} {'coverage':>9s}  {'95% CI':>16s}  {'exec':>6s}  {'nonempty':>8s}  {'step1 LIMIT':>11s}  {'rows':>7s}")
        for arm in ARMS:
            if arm not in summary or sp not in summary[arm]:
                continue
            e = summary[arm][sp]
            print(f"    {arm:20s} {e['graph_coverage_mean']:9.4f}  [{e['graph_coverage_ci95'][0]:.3f}, {e['graph_coverage_ci95'][1]:.3f}]"
                  f"  {e['exec_rate']:6.3f}  {e['nonempty_rate']:8.3f}  {e['step1_limit_rate']:11.3f}  {e['rows_returned_total']:7d}")
        for name, per in sig.items():
            if sp not in per:
                continue
            g = per[sp]
            mark = "*" if g["significant_at_95"] else " "
            print(f"    Δ {name:38s} {g['delta_mean']:+.4f} [{g['ci95_paired_bootstrap'][0]:+.3f}, {g['ci95_paired_bootstrap'][1]:+.3f}]{mark} "
                  f"p={g['wilcoxon_p']:.3g} d={g['cohen_d']:+.2f}  w/l/t={g['a_better']}/{g['b_better']}/{g['tie']}")

    if attribution:
        print(f"\n{'='*W}\n  ATTRIBUTION: how much of v2's gain over few-shot is the ONE prompt rule?\n{'='*W}")
        print(f"    {'slice':28s} {'rule only':>10s} {'v2 total':>10s} {'rule share':>11s} {'residual (graphrag)':>20s}")
        for sp, a in attribution.items():
            share = "n/a" if a["share_of_v2_gap_from_rule"] is None else f"{a['share_of_v2_gap_from_rule']:.0%}"
            print(f"    {sp:28s} {a['rule_only_delta']:+10.4f} {a['v2_total_delta']:+10.4f} {share:>11s} "
                  f"{a['residual_delta_attributable_to_graphrag']:+20.4f}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump({"rep": args.rep, "arms": ARMS, "summary": summary,
               "significance": sig, "attribution": attribution, "per_sample": per_arm},
              open(args.out, "w"), indent=2)
    print(f"\nSaved -> {args.out}")


if __name__ == "__main__":
    main()
