"""
compute_pipeline_significance.py - Paired significance testing between Method 1
(fewshot) and Method 2 (schema+retrieval / GraphRAG) on decomp_v1's finetuned
pipeline eval (cypher_results/finetuned_pipeline_samples.json).

Both methods are run on the SAME decomp_v1-decomposed searching sub-questions,
so per-sub-question outcomes are paired: McNemar's exact test for the binary
metrics (exec_success, has_results, gold_full_match), paired t-test + Wilcoxon
signed-rank + Cohen's d for the continuous gold_coverage metric (mirrors
compare_grpo_stability.py's methodology for consistency with the rest of the repo).

Usage:
    python compute_pipeline_significance.py
"""

import json
import numpy as np
from scipy import stats

SAMPLES_FILE = "cypher_results/finetuned_pipeline_samples.json"
OUT_FILE = "cypher_results/finetuned_pipeline_method_significance.json"


def mcnemar(a, b):
    a, b = np.array(a), np.array(b)
    b01 = int(((a == 0) & (b == 1)).sum())  # method1 fail, method2 success
    b10 = int(((a == 1) & (b == 0)).sum())  # method1 success, method2 fail
    n_discordant = b01 + b10
    p = stats.binomtest(min(b01, b10), n_discordant, p=0.5).pvalue if n_discordant > 0 else 1.0
    return {
        "method1_rate": float(a.mean()), "method2_rate": float(b.mean()),
        "discordant_1to2": b01, "discordant_2to1": b10, "mcnemar_exact_p": float(p),
    }


def main():
    samples = json.load(open(SAMPLES_FILE))

    exec1, exec2, valid1, valid2, cov1, cov2, full1, full2 = ([] for _ in range(8))
    for s in samples:
        m1, m2 = s.get("method1_fewshot", {}), s.get("method2_schema", {})
        for k in sorted(set(m1) & set(m2), key=int):
            r1, r2 = m1[k], m2[k]
            exec1.append(int(r1["exec_success"])); exec2.append(int(r2["exec_success"]))
            valid1.append(int(r1["has_results"])); valid2.append(int(r2["has_results"]))
            c1 = r1.get("gold_score", {}).get("coverage", 0.0) or 0.0
            c2 = r2.get("gold_score", {}).get("coverage", 0.0) or 0.0
            cov1.append(c1); cov2.append(c2)
            full1.append(int(c1 >= 0.999)); full2.append(int(c2 >= 0.999))

    n = len(exec1)
    results = {"n_paired_subquestions": n}

    results["exec_success"] = mcnemar(exec1, exec2)
    results["has_results"] = mcnemar(valid1, valid2)
    results["gold_full_match"] = mcnemar(full1, full2)

    cov1a, cov2a = np.array(cov1), np.array(cov2)
    diff = cov2a - cov1a
    t_stat, t_p = stats.ttest_rel(cov2a, cov1a)
    w_stat, w_p = stats.wilcoxon(cov2a, cov1a)
    pooled_std = diff.std(ddof=1)
    cohend = float(diff.mean() / pooled_std) if pooled_std > 0 else float("nan")
    ci = stats.t.interval(0.95, n - 1, loc=diff.mean(), scale=stats.sem(diff))
    results["gold_coverage"] = {
        "method1_mean": float(cov1a.mean()), "method2_mean": float(cov2a.mean()),
        "mean_diff": float(diff.mean()), "paired_ttest_t": float(t_stat), "paired_ttest_p": float(t_p),
        "wilcoxon_p": float(w_p), "cohens_d": cohend,
        "ci95_mean_diff": [float(ci[0]), float(ci[1])],
    }

    print(json.dumps(results, indent=2))
    with open(OUT_FILE, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {OUT_FILE}")


if __name__ == "__main__":
    main()
