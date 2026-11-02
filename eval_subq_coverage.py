"""
Sub-question COVERAGE evaluation (redundancy-tolerant).

Per the revised eval spec:
  - Only the `Sub-question:` lines matter. <PLAN>, `Step N:` descriptions are ignored
    (parse_output already only matches Sub-question lines).
  - A reference sub-question is COVERED if some predicted sub-question aligns to it
    (Hungarian 1:1) with cosine >= threshold.
  - Extra / redundant predicted sub-questions are NOT penalised. Only a MISSING
    reference sub-question (no adequate partner) is a failure.
  - Type consistency = among covered pairs, does pred type == ref type.
  - A sample "fully contains the expected result" if ALL its reference
    sub-questions are covered.

Reuses evaluate.py's exact embedding + Hungarian-alignment machinery.

Usage:
    python eval_subq_coverage.py --pred_file <test_predictions.json> --label progressive
    python eval_subq_coverage.py --pred_file ... --label baseline --threshold 0.85
"""

import argparse
import json

import numpy as np

from evaluate import (
    get_openai_client, get_embeddings, compute_alignment, parse_output,
    truncate_output, EMBEDDING_MODEL,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred_file", required=True)
    ap.add_argument("--label", default="model")
    ap.add_argument("--threshold", type=float, default=0.85,
                    help="cosine >= this => reference sub-question is covered (default 0.85)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    raw = json.load(open(args.pred_file, encoding="utf-8"))
    samples = []
    for p in raw:
        pred_text = p.get("prediction") or truncate_output(p["prediction_raw"])
        samples.append({
            "input": p["input"],
            "ref_subs": parse_output(p["reference"]),
            "pred_subs": parse_output(pred_text),
        })

    # batch-embed every unique sub-question
    texts, t2i = [], {}
    for s in samples:
        for sq in s["ref_subs"] + s["pred_subs"]:
            if sq.sub_question not in t2i:
                t2i[sq.sub_question] = len(texts)
                texts.append(sq.sub_question)
    client = get_openai_client()
    print(f"Embedding {len(texts)} unique sub-questions ({EMBEDDING_MODEL})...")
    embs = get_embeddings(client, texts, model=EMBEDDING_MODEL)
    gete = lambda t: embs[t2i[t]]

    TH = args.threshold

    tot_ref = tot_ref_s = tot_ref_a = 0
    cov = cov_s = cov_a = 0
    type_ok = type_tot = 0
    cos_covered = []
    pred_counts, ref_counts = [], []
    n_full_cov = n_full_cov_type = 0
    n_exact_count = 0
    missing_examples = []

    for s in samples:
        ref_subs, pred_subs = s["ref_subs"], s["pred_subs"]
        nr, npd = len(ref_subs), len(pred_subs)
        ref_counts.append(nr); pred_counts.append(npd)
        tot_ref += nr
        tot_ref_s += sum(1 for q in ref_subs if q.sq_type == "searching")
        tot_ref_a += sum(1 for q in ref_subs if q.sq_type == "aggregation")
        if nr == npd:
            n_exact_count += 1

        covered_flags = [False] * nr
        if ref_subs and pred_subs:
            r_e = np.stack([gete(q.sub_question) for q in ref_subs])
            p_e = np.stack([gete(q.sub_question) for q in pred_subs])
            res = compute_alignment(ref_subs, pred_subs, r_e, p_e)
            for a in res["alignment"]:
                ri, pi, c = a["ref_idx"], a["pred_idx"], a["cosine"]
                if c >= TH:
                    covered_flags[ri] = True
                    cos_covered.append(c)
                    type_tot += 1
                    if pred_subs[pi].sq_type == ref_subs[ri].sq_type:
                        type_ok += 1

        for i, q in enumerate(ref_subs):
            if covered_flags[i]:
                cov += 1
                if q.sq_type == "searching": cov_s += 1
                else: cov_a += 1

        if nr and all(covered_flags):
            n_full_cov += 1
        elif nr:
            miss = [ref_subs[i].sub_question for i in range(nr) if not covered_flags[i]]
            if len(missing_examples) < 6:
                missing_examples.append({"input": s["input"], "missing": miss})

        # full coverage AND every covered pair's type matched
        if nr and all(covered_flags):
            # recompute type correctness for this sample
            if pred_subs:
                res2 = compute_alignment(ref_subs, pred_subs,
                                         np.stack([gete(q.sub_question) for q in ref_subs]),
                                         np.stack([gete(q.sub_question) for q in pred_subs]))
                ok = True
                for a in res2["alignment"]:
                    if a["cosine"] >= TH and pred_subs[a["pred_idx"]].sq_type != ref_subs[a["ref_idx"]].sq_type:
                        ok = False; break
                if ok:
                    n_full_cov_type += 1

    N = len(samples)
    over = sum(1 for pc, rc in zip(pred_counts, ref_counts) if pc > rc)
    under = sum(1 for pc, rc in zip(pred_counts, ref_counts) if pc < rc)

    def pct(a, b): return f"{a}/{b} ({a / max(b,1) * 100:.1f}%)"

    print(f"\n{'='*66}")
    print(f"  Sub-question coverage — {args.label}   (cover threshold cos >= {TH})")
    print(f"  {N} samples | ref sub-Qs = {tot_ref} | pred sub-Qs = {sum(pred_counts)}")
    print(f"{'='*66}")
    print(f"\n  COVERAGE (ref sub-Qs with an adequate predicted partner; extras ignored)")
    print(f"    Overall:      {pct(cov, tot_ref)}")
    print(f"    searching:    {pct(cov_s, tot_ref_s)}")
    print(f"    aggregation:  {pct(cov_a, tot_ref_a)}")
    print(f"\n  PER-SAMPLE — output fully contains the expected sub-questions")
    print(f"    Full coverage:                 {pct(n_full_cov, N)}")
    print(f"    Full coverage + all types ok:  {pct(n_full_cov_type, N)}")
    print(f"\n  TYPE CONSISTENCY (among covered pairs)")
    print(f"    pred type == ref type:  {pct(type_ok, type_tot)}")
    print(f"\n  SEMANTIC (covered aligned pairs)")
    print(f"    cosine mean:  {np.mean(cos_covered):.4f}  (n={len(cos_covered)})" if cos_covered else "    (none)")
    print(f"\n  COUNT (redundancy tolerated — 'over' is not a failure)")
    print(f"    exact count match:  {pct(n_exact_count, N)}")
    print(f"    avg ref / pred:     {np.mean(ref_counts):.2f} / {np.mean(pred_counts):.2f}")
    print(f"    over / under:       {over} / {under}   (under = missing sub-Qs = real failure)")

    if missing_examples:
        print(f"\n  Samples with an uncovered reference sub-question (first {len(missing_examples)}):")
        for m in missing_examples:
            print(f"    - {m['input'][:80]}")
            for q in m["missing"]:
                print(f"        MISSING: {q[:90]}")

    result = {
        "label": args.label, "threshold": TH, "n_samples": N,
        "coverage_overall_pct": cov / max(tot_ref,1) * 100,
        "coverage_searching_pct": cov_s / max(tot_ref_s,1) * 100,
        "coverage_aggregation_pct": cov_a / max(tot_ref_a,1) * 100,
        "full_coverage_samples_pct": n_full_cov / N * 100,
        "full_coverage_and_type_samples_pct": n_full_cov_type / N * 100,
        "type_consistency_pct": type_ok / max(type_tot,1) * 100,
        "covered_pair_cosine_mean": float(np.mean(cos_covered)) if cos_covered else 0.0,
        "count_exact_pct": n_exact_count / N * 100,
        "avg_ref_count": float(np.mean(ref_counts)),
        "avg_pred_count": float(np.mean(pred_counts)),
        "count_over": over, "count_under": under,
    }
    out = args.out or args.pred_file.rsplit("/", 1)[0] + "/subq_coverage.json"
    with open(out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved -> {out}")


if __name__ == "__main__":
    main()
