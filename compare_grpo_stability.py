"""
compare_grpo_stability.py - Paired before/after statistical comparison of
decomp_v1 vs decomp_v1+GRPO at high temperature.

Loads two samples files produced by gen_decomp_v1_temp.py (same 120 questions,
same temperature, same N independent samples per question — only the model
differs) and reports, for each of the three axes the temp=1.0 baseline eval
flagged:
  1. Sub-question content semantic similarity (embedding cosine, Hungarian-matched)
  2. Sub-question count exact-match rate
  3. Self-consistency (pairwise cosine among the N samples' full text, per question)

For each axis: paired difference (post - pre) across the 120 questions,
paired t-test + Wilcoxon signed-rank, Cohen's d, and 95% CI on the mean
difference.

Usage:
    python compare_grpo_stability.py \
        --pre cypher_results/decomp_v1_temp1.0_samples.json \
        --post cypher_results/decomp_v1_temp1.0_grpo_samples.json
"""

import argparse
import json
import numpy as np
from scipy import stats

from evaluate import parse_output, get_openai_client, get_embeddings, compute_alignment, EMBEDDING_MODEL
from run_finetuned_pipeline_eval import truncate_decomposition
from run_cypher_qa_comparison import extract_all_subquestions


def prep(samples):
    for s in samples:
        ref = s["reference_output"]
        for r in s["runs"]:
            trunc = truncate_decomposition(r["raw_output"])
            r["truncated_output"] = trunc
            r["exact_match_ref"] = trunc.strip() == ref.strip()
            subqs = extract_all_subquestions(trunc)
            r["n_searching"] = sum(1 for x in subqs if x["type"] == "searching")
            r["n_total"] = len(subqs)
    return samples


def cos(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10))


def per_question_stats(samples, get_e):
    """Returns (mean_semantic_per_question, count_match_rate_per_question,
    selfcons_per_question) — each a list of length n_questions."""
    sem_pq, count_pq, selfcons_pq = [], [], []
    for s in samples:
        ref_subs = parse_output(s["reference_output"])
        r_e = np.stack([get_e(sq.sub_question) for sq in ref_subs]) if ref_subs else None
        run_sems, run_full_embs, run_count_matches = [], [], []
        for r in s["runs"]:
            pred_subs = parse_output(r["truncated_output"])
            if pred_subs and ref_subs:
                p_e = np.stack([get_e(sq.sub_question) for sq in pred_subs])
                res = compute_alignment(ref_subs, pred_subs, r_e, p_e)
                sem = float(np.mean(res["scores"])) if res["scores"] else 0.0
            else:
                sem = 0.0
            run_sems.append(sem)
            run_full_embs.append(get_e(r["truncated_output"]))
            run_count_matches.append(1 if r["n_searching"] == s["ref_n_searching"] else 0)
        sem_pq.append(float(np.mean(run_sems)))
        count_pq.append(float(np.mean(run_count_matches)))
        pw = []
        n = len(run_full_embs)
        for i in range(n):
            for j in range(i + 1, n):
                pw.append(cos(run_full_embs[i], run_full_embs[j]))
        selfcons_pq.append(float(np.mean(pw)) if pw else 1.0)
    return np.array(sem_pq), np.array(count_pq), np.array(selfcons_pq)


def report_paired(label, pre, post):
    diff = post - pre
    t_stat, p_two = stats.ttest_rel(post, pre)
    p_one = p_two / 2 if t_stat > 0 else 1 - p_two / 2  # one-sided: post > pre
    try:
        w_stat, w_p = stats.wilcoxon(diff)
    except ValueError:
        w_stat, w_p = float("nan"), float("nan")
    cohens_d = diff.mean() / diff.std(ddof=1) if diff.std(ddof=1) > 0 else float("nan")
    ci = stats.t.interval(0.95, len(diff) - 1, loc=diff.mean(), scale=stats.sem(diff)) \
        if diff.std(ddof=1) > 0 else (diff.mean(), diff.mean())
    print(f"=== {label} ===")
    print(f"  pre  mean={pre.mean():.4f}  sd={pre.std(ddof=1):.4f}")
    print(f"  post mean={post.mean():.4f}  sd={post.std(ddof=1):.4f}")
    print(f"  paired diff (post-pre): mean={diff.mean():+.4f}  95% CI=({ci[0]:+.4f}, {ci[1]:+.4f})")
    print(f"  paired t-test: t={t_stat:.3f}, one-sided p={p_one:.2e}  |  Wilcoxon p={w_p:.2e}")
    print(f"  Cohen's d = {cohens_d:.4f}")
    print()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pre", required=True)
    ap.add_argument("--post", required=True)
    args = ap.parse_args()

    pre_samples = prep(json.load(open(args.pre)))
    post_samples = prep(json.load(open(args.post)))
    assert len(pre_samples) == len(post_samples), "pre/post sample counts differ"
    for a, b in zip(pre_samples, post_samples):
        assert a["input"] == b["input"], "pre/post question order mismatch"

    client = get_openai_client()
    texts, t2i = [], {}
    def reg(t):
        if t not in t2i:
            t2i[t] = len(texts); texts.append(t)
    for samples in (pre_samples, post_samples):
        for s in samples:
            for sq in parse_output(s["reference_output"]):
                reg(sq.sub_question)
            for r in s["runs"]:
                for sq in parse_output(r["truncated_output"]):
                    reg(sq.sub_question)
                reg(r["truncated_output"])

    print(f"Embedding {len(texts)} unique texts...")
    embs = get_embeddings(client, texts, model=EMBEDDING_MODEL)
    get_e = lambda t: embs[t2i[t]] if t in t2i else np.zeros(embs.shape[1])

    sem_pre, count_pre, selfcons_pre = per_question_stats(pre_samples, get_e)
    sem_post, count_post, selfcons_post = per_question_stats(post_samples, get_e)

    report_paired("Sub-question semantic similarity (per-question mean across N samples)",
                   sem_pre, sem_post)
    report_paired("Sub-question count exact-match rate (per-question mean across N samples)",
                   count_pre, count_post)
    report_paired("Self-consistency (mean pairwise cosine among N samples, per question)",
                   selfcons_pre, selfcons_post)


if __name__ == "__main__":
    main()
