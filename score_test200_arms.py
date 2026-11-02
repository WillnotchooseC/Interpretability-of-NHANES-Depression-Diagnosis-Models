"""
Gold-scored comparison of the test_200 Cypher arms (few-shot / GraphRAG v1 /
GraphRAG v2) from the merged cypher_results/test200_*_reps.json files.

Metric: graph_coverage per sample = run_finetuned_pipeline_eval.score_against_gold
(greedy value-set row containment of ALL retrieved rows pooled across a sample's
steps vs the pooled gold_results rows, filter-echo values removed via
build_echo_values) -- identical to score_experiments.py / score_experiments_reps.py,
so numbers are comparable to the test_120 eval. No OpenAI calls (answer cosine is
not computed: answers were not synthesised).

Also reports exec/non-empty rates, per-split means, and PAIRED comparisons on
the same 200 samples (Wilcoxon, Cohen's d, cluster-bootstrap CI of the mean delta).

Usage:
    python score_test200_arms.py                       # rep1 of every arm present
    python score_test200_arms.py --rep 2
"""

import argparse
import json
import os

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
    "graphrag_v1": "cypher_results/test200_graphrag_exp2_3reps.json",
    "graphrag_v2": "cypher_results/test200_graphrag_v2_exp2_3reps.json",
    "fewshot_notruncate": "cypher_results/test200_fewshot_notruncate_exp1_6_reps.json",
}
SPLITS = ["overall", "seen", "unseen"]
N_BOOT = 10000


def boot_ci(x, seed=0):
    rng = np.random.default_rng(seed)
    x = np.asarray(x, float)
    draws = [rng.choice(x, size=len(x), replace=True).mean() for _ in range(N_BOOT)]
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return float(lo), float(hi)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rep", type=int, default=1)
    ap.add_argument("--out", default="cypher_results/test200_arms_score.json")
    args = ap.parse_args()

    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    driver.verify_connectivity()
    vocab = build_vocab(driver)
    driver.close()

    per_arm = {}
    for arm, path in ARMS.items():
        if not os.path.exists(path):
            print(f"skip {arm}: {path} missing")
            continue
        recs = json.load(open(path, encoding="utf-8"))["records"]
        rows = []
        for r in recs:
            steps = r["reps"][f"rep{args.rep}"]["steps"]
            actual = [x for st in steps for x in (st.get("records") or [])]
            sc = score_against_gold(actual, gold_rows_for(r), build_echo_values(r["input"], vocab))
            rows.append({
                "idx": r["idx"], "split": r["split"],
                "graph_coverage": sc["coverage"] if sc["coverage"] is not None else 0.0,
                "gold_full_match": 1.0 if (sc["full_matches"] is not None and sc["n_gold_rows"]
                                            and sc["full_matches"] == sc["n_gold_rows"]) else 0.0,
                "n_steps": len(steps),
                "exec_ok": sum(1 for st in steps if st.get("exec_success")),
                "nonempty": sum(1 for st in steps if st.get("has_results")),
                "all_steps_nonempty": 1.0 if steps and all(st.get("has_results") for st in steps) else 0.0,
                "repaired": sum(1 for st in steps if st.get("repaired_concatenated_query")),
            })
        per_arm[arm] = rows

    summary = {}
    for arm, rows in per_arm.items():
        summary[arm] = {}
        for sp in SPLITS:
            sub = [r for r in rows if sp == "overall" or r["split"] == sp]
            cov = [r["graph_coverage"] for r in sub]
            n_steps = sum(r["n_steps"] for r in sub)
            summary[arm][sp] = {
                "n": len(sub),
                "graph_coverage_mean": float(np.mean(cov)),
                "graph_coverage_ci95": boot_ci(cov),
                "gold_full_match_rate": float(np.mean([r["gold_full_match"] for r in sub])),
                "exec_rate": sum(r["exec_ok"] for r in sub) / max(n_steps, 1),
                "nonempty_rate": sum(r["nonempty"] for r in sub) / max(n_steps, 1),
                "all_steps_nonempty_rate": float(np.mean([r["all_steps_nonempty"] for r in sub])),
                "repaired_steps": sum(r["repaired"] for r in sub),
            }

    pairs = [("graphrag_v1", "fewshot"), ("graphrag_v2", "fewshot"), ("graphrag_v2", "graphrag_v1"),
         ("fewshot_notruncate", "fewshot"), ("graphrag_v2", "fewshot_notruncate")]
    sig = {}
    for a, b in pairs:
        if a not in per_arm or b not in per_arm:
            continue
        sig[f"{a}_minus_{b}"] = {}
        for sp in SPLITS:
            ia = [r for r in per_arm[a] if sp == "overall" or r["split"] == sp]
            ib = [r for r in per_arm[b] if sp == "overall" or r["split"] == sp]
            x = np.array([r["graph_coverage"] for r in ia]); y = np.array([r["graph_coverage"] for r in ib])
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

    print(f"\n{'='*96}\n  test_200 gold-scored comparison (rep{args.rep}); graph_coverage = pooled row containment vs gold_results\n{'='*96}")
    for sp in SPLITS:
        print(f"\n  ===== {sp} (n={summary[next(iter(summary))][sp]['n']}) =====")
        print(f"    {'arm':13s} {'coverage':>9s}  {'95% CI':>16s}  {'full':>6s}  {'exec':>6s}  {'nonempty':>8s}  {'all-ne':>6s}  repaired")
        for arm, s in summary.items():
            e = s[sp]
            print(f"    {arm:13s} {e['graph_coverage_mean']:9.4f}  [{e['graph_coverage_ci95'][0]:.3f}, {e['graph_coverage_ci95'][1]:.3f}]"
                  f"  {e['gold_full_match_rate']:6.3f}  {e['exec_rate']:6.3f}  {e['nonempty_rate']:8.3f}  {e['all_steps_nonempty_rate']:6.3f}  {e['repaired_steps']}")
        for name, per in sig.items():
            g = per[sp]; mark = "*" if g["significant_at_95"] else " "
            print(f"    Δ {name:26s} {g['delta_mean']:+.4f} [{g['ci95_paired_bootstrap'][0]:+.3f}, {g['ci95_paired_bootstrap'][1]:+.3f}]{mark} "
                  f"wilcoxon_p={g['wilcoxon_p']:.3g} d={g['cohen_d']:+.2f}  wins/losses/ties={g['a_better']}/{g['b_better']}/{g['tie']}")

    json.dump({"rep": args.rep, "summary": summary, "significance": sig, "per_sample": per_arm},
              open(args.out, "w"), indent=2)
    print(f"\nSaved -> {args.out}")


if __name__ == "__main__":
    main()
