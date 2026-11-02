"""
Score the exp1_5 / exp2 / exp3 GraphRAG-ablation outputs across 3 repetitions
each and run a statistical evaluation.

exp1 here == experiment1_5_baseline_original_input.py (baseline few-shot that
DOES see the original question in the Cypher prompt), per the user's spec.

Repetitions:
  rep1 = results_<name>/                (the pre-existing first run)
  rep2 = results_<name>_run2/
  rep3 = results_<name>_run3/
all on --files ood_data/progressive_decomp_for_experiments.json, scored against
ood_data/curriculum_test_120_gold.json.

Two axes (same as score_experiments.py):
  - graph_coverage : value-set containment of pooled retrieved rows vs gold_results
                     (run_finetuned_pipeline_eval.score_against_gold)
  - answer_cosine  : text-embedding-3-large cosine of final_answer vs gold_answer

Outputs:
  experiments_reps_per_sample.json   {exp: {rep: [per-sample rows]}}
  experiments_reps_summary.json      per-rep means + across-rep mean/std, by subset
  experiments_reps_significance.json pairwise paired tests (rep-averaged + per-rep)
"""
import json
import numpy as np
from scipy import stats
from neo4j import GraphDatabase

from entity_retrieval import build_vocab
from run_finetuned_pipeline_eval import score_against_gold, build_echo_values
from evaluate import get_openai_client, get_embeddings, EMBEDDING_MODEL

NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "12345678"
GOLD_FILE = "ood_data/curriculum_test_120_gold.json"
INPUT_BASENAME = "progressive_decomp_for_experiments"

EXPERIMENTS = {
    "exp1_5_baseline_original_input": "exp1_5_baseline_original_input",
    "exp2_fewshot_graphrag":          "exp2_fewshot_graphrag",
    "exp3_fewshot_graphrag_rescue":   "exp3_fewshot_graphrag_rescue",
}
PAIRS = [
    ("exp1_5_baseline_original_input", "exp2_fewshot_graphrag"),
    ("exp1_5_baseline_original_input", "exp3_fewshot_graphrag_rescue"),
    ("exp2_fewshot_graphrag",          "exp3_fewshot_graphrag_rescue"),
]
REPS = [1, 2, 3]

UNSEEN_IDX = set([10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89,
                  100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114,
                  115, 116, 117, 118, 119])

SUBSETS = {
    "all":       lambda i: True,
    "seen_80":   lambda i: i not in UNSEEN_IDX,
    "unseen_40": lambda i: i in UNSEEN_IDX,
}


def rep_dir(exp_name, rep):
    return f"results_{exp_name}" if rep == 1 else f"results_{exp_name}_run{rep}"


def rep_path(exp_name, rep):
    return f"{rep_dir(exp_name, rep)}/{INPUT_BASENAME}_{exp_name}.json"


def cos(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def gold_rows_for(sample):
    rows = []
    for g in sample.get("gold_results", []):
        r = g.get("result")
        if isinstance(r, list):
            rows.extend(x for x in r if isinstance(x, dict))
        elif isinstance(r, dict):
            rows.append(r)
    return rows


def cohens_d_paired(x, y):
    d = np.asarray(x, float) - np.asarray(y, float)
    sd = d.std(ddof=1)
    return float(d.mean() / sd) if sd > 0 else 0.0


def mean_std(vals):
    v = [x for x in vals if x is not None]
    if not v:
        return {"mean": None, "std": None, "n": 0}
    return {"mean": round(float(np.mean(v)), 4),
            "std": round(float(np.std(v, ddof=1)), 4) if len(v) > 1 else 0.0,
            "n": len(v)}


def main():
    gold = json.load(open(GOLD_FILE, encoding="utf-8"))
    n = len(gold)
    gold_rows_all = [gold_rows_for(g) for g in gold]

    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    driver.verify_connectivity()
    vocab = build_vocab(driver)
    driver.close()
    echoes_all = [build_echo_values(g["input"], vocab) for g in gold]

    client = get_openai_client()
    gold_emb = get_embeddings(client, [g.get("gold_answer", "") for g in gold], model=EMBEDDING_MODEL)

    # per_sample[exp][rep] = list of n row dicts
    per_sample = {e: {} for e in EXPERIMENTS}

    for exp_name in EXPERIMENTS:
        for rep in REPS:
            path = rep_path(exp_name, rep)
            try:
                data = json.load(open(path, encoding="utf-8"))
            except FileNotFoundError:
                print(f"  MISSING {path} -- skipping {exp_name} rep{rep}")
                continue
            details = data["details"]
            assert len(details) == n, f"{exp_name} rep{rep}: {len(details)} vs {n}"

            final_answers = [d.get("final_answer") or "" for d in details]
            embed_inputs = [a if a.strip() else "(no answer was generated)" for a in final_answers]
            empty_idx = [i for i, a in enumerate(final_answers) if not a.strip()]
            if empty_idx:
                print(f"  {exp_name} rep{rep}: empty final_answer at {empty_idx}")
            ans_emb = get_embeddings(client, embed_inputs, model=EMBEDDING_MODEL)

            rows = []
            for i, d in enumerate(details):
                actual_rows = []
                steps = d.get("steps", [])
                for step in steps:
                    actual_rows.extend(step.get("records") or [])
                sc = score_against_gold(actual_rows, gold_rows_all[i], echoes_all[i])
                exec_ok = bool(steps) and all(s.get("exec_success") for s in steps)
                nonempty = bool(steps) and all(s.get("has_results") for s in steps)
                full_cov = None
                if sc["coverage"] is not None:
                    full_cov = int(sc["full_matches"] == sc["n_gold_rows"] and sc["n_gold_rows"] > 0)
                rows.append({
                    "index": i,
                    "input": gold[i]["input"],
                    "graph_coverage": sc["coverage"],
                    "full_cov": full_cov,
                    "answer_cosine": round(cos(ans_emb[i], gold_emb[i]), 4),
                    "exec_success": int(exec_ok),
                    "has_results": int(nonempty),
                    "n_searching": d.get("num_searching_subquestions", len(steps)),
                })
            per_sample[exp_name][str(rep)] = rows
            print(f"  scored {exp_name} rep{rep}")

    json.dump(per_sample, open("experiments_reps_per_sample.json", "w"),
              indent=2, ensure_ascii=False, default=str)

    # ---- summary: per-rep subset means + across-rep mean/std ----
    summary = {}
    for exp_name, reps in per_sample.items():
        summary[exp_name] = {}
        for sub_name, filt in SUBSETS.items():
            per_rep = {}
            gc_means, ac_means, ex_means, fc_means = [], [], [], []
            for rep, rows in reps.items():
                sel = [r for r in rows if filt(r["index"])]
                gc = [r["graph_coverage"] for r in sel if r["graph_coverage"] is not None]
                ac = [r["answer_cosine"] for r in sel]
                ex = [r["exec_success"] for r in sel]
                fc = [r["full_cov"] for r in sel if r["full_cov"] is not None]
                gc_m = round(float(np.mean(gc)), 4) if gc else None
                ac_m = round(float(np.mean(ac)), 4) if ac else None
                ex_m = round(float(np.mean(ex)), 4) if ex else None
                fc_m = round(float(np.mean(fc)), 4) if fc else None
                per_rep[rep] = {"graph_coverage": gc_m, "answer_cosine": ac_m,
                                "exec_rate": ex_m, "full_cov_rate": fc_m,
                                "graph_coverage_n": len(gc), "n": len(sel)}
                gc_means.append(gc_m); ac_means.append(ac_m)
                ex_means.append(ex_m); fc_means.append(fc_m)
            summary[exp_name][sub_name] = {
                "per_rep": per_rep,
                "across_reps": {
                    "graph_coverage": mean_std(gc_means),
                    "answer_cosine":  mean_std(ac_means),
                    "exec_rate":      mean_std(ex_means),
                    "full_cov_rate":  mean_std(fc_means),
                },
            }
    json.dump(summary, open("experiments_reps_summary.json", "w"), indent=2, ensure_ascii=False)

    # ---- significance: rep-averaged per-sample vectors + per-rep p-values ----
    def rep_avg_vec(exp_name, key):
        reps = per_sample[exp_name]
        out = []
        for i in range(n):
            vals = [reps[r][i][key] for r in reps if reps[r][i][key] is not None]
            out.append(float(np.mean(vals)) if vals else None)
        return out

    sig = {}
    for a, b in PAIRS:
        pair_key = f"{a}__vs__{b}"
        sig[pair_key] = {}
        for sub_name, filt in SUBSETS.items():
            entry = {}
            for key in ("graph_coverage", "answer_cosine"):
                va, vb = rep_avg_vec(a, key), rep_avg_vec(b, key)
                pairs = [(x, y) for idx, (x, y) in enumerate(zip(va, vb))
                         if filt(idx) and x is not None and y is not None]
                if len(pairs) < 3:
                    entry[key] = {"n_pairs": len(pairs)}
                    continue
                xa = np.array([p[0] for p in pairs]); xb = np.array([p[1] for p in pairs])
                t_p = float(stats.ttest_rel(xa, xb).pvalue)
                try:
                    w_p = float(stats.wilcoxon(xa, xb, zero_method="wilcox").pvalue)
                except ValueError:
                    w_p = 1.0
                # per-rep Wilcoxon for consistency check
                per_rep_w = []
                common = set(per_sample[a]) & set(per_sample[b])
                for r in sorted(common):
                    ra = per_sample[a][r]; rb = per_sample[b][r]
                    pr = [(ra[i][key], rb[i][key]) for i in range(n)
                          if filt(i) and ra[i][key] is not None and rb[i][key] is not None]
                    if len(pr) >= 3:
                        try:
                            per_rep_w.append(round(float(stats.wilcoxon(
                                [p[0] for p in pr], [p[1] for p in pr]).pvalue), 5))
                        except ValueError:
                            per_rep_w.append(1.0)
                entry[key] = {
                    "n_pairs": len(pairs),
                    "mean_a": round(float(xa.mean()), 4),
                    "mean_b": round(float(xb.mean()), 4),
                    "delta_b_minus_a": round(float(xb.mean() - xa.mean()), 4),
                    "paired_t_p": round(t_p, 6),
                    "wilcoxon_p": round(w_p, 6),
                    "cohens_d_paired": round(cohens_d_paired(xb, xa), 4),
                    "per_rep_wilcoxon_p": per_rep_w,
                }
            # McNemar on rep-majority full_cov
            fa, fb = rep_avg_vec(a, "full_cov"), rep_avg_vec(b, "full_cov")
            b01 = c10 = both = neither = 0
            for idx in range(n):
                if not filt(idx) or fa[idx] is None or fb[idx] is None:
                    continue
                ba = 1 if fa[idx] >= 0.5 else 0
                bb = 1 if fb[idx] >= 0.5 else 0
                if ba and bb: both += 1
                elif ba and not bb: c10 += 1
                elif bb and not ba: b01 += 1
                else: neither += 1
            disc = b01 + c10
            if disc > 0:
                mcp = float(stats.binomtest(min(b01, c10), disc, 0.5).pvalue)
            else:
                mcp = 1.0
            entry["full_cov_mcnemar"] = {
                "a_only": c10, "b_only": b01, "both": both, "neither": neither,
                "discordant": disc, "exact_binom_p": round(mcp, 6),
            }
            sig[pair_key][sub_name] = entry
    json.dump(sig, open("experiments_reps_significance.json", "w"), indent=2, ensure_ascii=False)

    # ---- console ----
    print("\n" + "=" * 100)
    print(f"{'experiment':<34}{'subset':<11}{'graph_coverage (mean±std over 3 reps)':<40}{'answer_cosine':<22}")
    print("-" * 100)
    for exp_name, s in summary.items():
        for sub_name in SUBSETS:
            gc = s[sub_name]["across_reps"]["graph_coverage"]
            ac = s[sub_name]["across_reps"]["answer_cosine"]
            print(f"{exp_name:<34}{sub_name:<11}"
                  f"{str(gc['mean'])+' ± '+str(gc['std']):<40}"
                  f"{str(ac['mean'])+' ± '+str(ac['std']):<22}")
    print("\n" + "=" * 100)
    print("PAIRWISE (rep-averaged per-sample, paired):")
    for pk, subs in sig.items():
        print(f"\n{pk}")
        for sub_name, e in subs.items():
            gc = e.get("graph_coverage", {})
            if "wilcoxon_p" in gc:
                print(f"  {sub_name:<10} graph_coverage  Δ={gc['delta_b_minus_a']:+.4f}  "
                      f"wilcoxon_p={gc['wilcoxon_p']:.4g}  paired_t_p={gc['paired_t_p']:.4g}  "
                      f"d={gc['cohens_d_paired']:+.3f}  per_rep_w={gc['per_rep_wilcoxon_p']}")
            mc = e["full_cov_mcnemar"]
            print(f"  {sub_name:<10} full_cov McNemar  a_only={mc['a_only']} b_only={mc['b_only']} "
                  f"disc={mc['discordant']}  p={mc['exact_binom_p']:.4g}")
    print("\nSaved -> experiments_reps_{per_sample,summary,significance}.json")


if __name__ == "__main__":
    main()
