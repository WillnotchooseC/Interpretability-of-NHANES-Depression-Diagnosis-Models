"""
Sub-question-level match analysis: model output vs. reference, split by
coverage / type / semantic similarity.

evaluate.py's Dim 1 (semantic similarity) already Hungarian-aligns pred vs
ref sub-questions per sample, but only ever saves the aggregate cosine mean
and the raw alignment list (pred_idx/ref_idx/cosine) -- not which sub-Q type
each side of a pair actually is, and not a thresholded "was this ref
sub-question actually covered" verdict. This script re-embeds and re-aligns
the same way (reusing evaluate.py's exact machinery) but keeps the per-pair
type labels, so it can report:

  1. Coverage: % of reference sub-questions that have an aligned prediction
     partner with cosine >= threshold (thresholds reuse the codebase's
     existing ones: HIGH_SIM_THRESHOLD=0.85 from evaluate.py,
     HIT_THRESHOLD=0.90 from calc_hit_rate.py), overall and split by the
     reference sub-question's type (searching vs aggregation).
  2. Type accuracy: among aligned pairs (not positional index-based like
     evaluate.py's supplementary type_accuracy, which breaks down when
     pred/ref counts differ), % where the predicted sub-question's type
     label matches its aligned reference's type label.
  3. Semantic similarity: aligned-pair cosine mean, split by reference type.

Usage:
    python analyze_subquestion_match.py --run_name plan_composition_v2
    python analyze_subquestion_match.py --run_name plan_composition_v2 --threshold 0.90
"""

import argparse
import json
import os

import numpy as np

from evaluate import (
    get_openai_client, get_embeddings, compute_alignment, parse_output,
    truncate_output, EMBEDDING_MODEL, HIGH_SIM_THRESHOLD, PERFECT_SIM_THRESHOLD,
)

HIT_THRESHOLD = 0.90  # matches calc_hit_rate.py's HIT_THRESHOLD


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_name", type=str, required=True)
    parser.add_argument("--pred_file", type=str, default=None,
                         help="Defaults to nhanes_model/<run_name>/test_predictions.json")
    args = parser.parse_args()

    pred_file = args.pred_file or os.path.join("nhanes_model", args.run_name, "test_predictions.json")
    raw_predictions = json.load(open(pred_file, "r", encoding="utf-8"))
    print(f"Loaded {len(raw_predictions)} predictions from {pred_file}")

    samples = []
    for p in raw_predictions:
        pred_text = p.get("prediction") or truncate_output(p["prediction_raw"])
        ref_subs = parse_output(p["reference"])
        pred_subs = parse_output(pred_text)
        samples.append({"input": p["input"], "ref_subs": ref_subs, "pred_subs": pred_subs})

    # Batch-embed every unique sub-question text across the whole test set
    client = get_openai_client()
    texts, t2i = [], {}
    for s in samples:
        for sq in s["ref_subs"] + s["pred_subs"]:
            if sq.sub_question not in t2i:
                t2i[sq.sub_question] = len(texts)
                texts.append(sq.sub_question)
    print(f"Embedding {len(texts)} unique sub-questions ({EMBEDDING_MODEL})...")
    embs = get_embeddings(client, texts, model=EMBEDDING_MODEL) if texts else np.zeros((0, 3072))
    get_e = lambda t: embs[t2i[t]]

    # Per-sample Hungarian alignment, keeping type labels on both sides
    pairs = []       # every aligned (ref, pred) pair: {ref_type, pred_type, cosine}
    unmatched_ref = []   # ref sub-Qs with no aligned pred partner at all (count mismatch)
    unmatched_pred = []  # pred sub-Qs with no aligned ref partner
    for s in samples:
        ref_subs, pred_subs = s["ref_subs"], s["pred_subs"]
        if not ref_subs or not pred_subs:
            unmatched_ref.extend(ref_subs)
            unmatched_pred.extend(pred_subs)
            continue
        r_e = np.stack([get_e(sq.sub_question) for sq in ref_subs])
        p_e = np.stack([get_e(sq.sub_question) for sq in pred_subs])
        res = compute_alignment(ref_subs, pred_subs, r_e, p_e)
        for a in res["alignment"]:
            pairs.append({
                "ref_type": ref_subs[a["ref_idx"]].sq_type,
                "pred_type": pred_subs[a["pred_idx"]].sq_type,
                "cosine": a["cosine"],
            })
        unmatched_ref.extend(ref_subs[i] for i in res["unmatched_ref"])
        unmatched_pred.extend(pred_subs[i] for i in res["unmatched_pred"])

    total_ref = sum(len(s["ref_subs"]) for s in samples)
    total_pred = sum(len(s["pred_subs"]) for s in samples)

    def coverage(subset, threshold):
        n = len(subset)
        if n == 0:
            return 0.0, 0
        hit = sum(1 for p in subset if p["cosine"] >= threshold)
        return hit / n * 100, n

    def report_for_type(ref_type_filter, label):
        subset = [p for p in pairs if ref_type_filter(p["ref_type"])]
        n_ref_of_type = sum(1 for s in samples for sq in s["ref_subs"] if ref_type_filter(sq.sq_type))
        cov85, _ = coverage(subset, HIGH_SIM_THRESHOLD)
        cov90, _ = coverage(subset, HIT_THRESHOLD)
        cov92, _ = coverage(subset, PERFECT_SIM_THRESHOLD)
        # coverage denominator is ALL ref sub-Qs of this type (including those
        # that never even got a Hungarian partner, e.g. count_pred==0 samples)
        hit85 = sum(1 for p in subset if p["cosine"] >= HIGH_SIM_THRESHOLD)
        hit90 = sum(1 for p in subset if p["cosine"] >= HIT_THRESHOLD)
        hit92 = sum(1 for p in subset if p["cosine"] >= PERFECT_SIM_THRESHOLD)
        cos_mean = float(np.mean([p["cosine"] for p in subset])) if subset else 0.0
        type_acc = (sum(1 for p in subset if p["pred_type"] == p["ref_type"]) / len(subset) * 100) if subset else 0.0
        print(f"\n  --- {label} (n_ref={n_ref_of_type}) ---")
        print(f"  Coverage @0.85:           {hit85}/{n_ref_of_type} ({hit85/max(n_ref_of_type,1)*100:.1f}%)")
        print(f"  Coverage @0.90 (hit-rate):{hit90}/{n_ref_of_type} ({hit90/max(n_ref_of_type,1)*100:.1f}%)")
        print(f"  Coverage @0.92:           {hit92}/{n_ref_of_type} ({hit92/max(n_ref_of_type,1)*100:.1f}%)")
        print(f"  Aligned-pair cosine mean: {cos_mean:.4f}")
        print(f"  Type match rate (pred==ref, among aligned pairs): {type_acc:.1f}%")
        return {
            "n_ref": n_ref_of_type,
            "coverage_0.85_pct": hit85 / max(n_ref_of_type, 1) * 100,
            "coverage_0.90_pct": hit90 / max(n_ref_of_type, 1) * 100,
            "coverage_0.92_pct": hit92 / max(n_ref_of_type, 1) * 100,
            "aligned_cosine_mean": cos_mean,
            "type_match_rate_pct": type_acc,
        }

    print(f"\n{'='*70}")
    print(f"  Sub-question match analysis: {args.run_name}")
    print(f"  {len(samples)} samples | ref sub-Qs={total_ref} | pred sub-Qs={total_pred}")
    print(f"{'='*70}")

    result = {}
    result["overall"] = report_for_type(lambda t: True, "OVERALL")
    result["searching"] = report_for_type(lambda t: t == "searching", "TYPE = searching")
    result["aggregation"] = report_for_type(lambda t: t == "aggregation", "TYPE = aggregation")

    n_unmatched_ref = len(unmatched_ref)
    n_unmatched_pred = len(unmatched_pred)
    unmatched_ref_by_type = {
        "searching": sum(1 for sq in unmatched_ref if sq.sq_type == "searching"),
        "aggregation": sum(1 for sq in unmatched_ref if sq.sq_type == "aggregation"),
    }
    unmatched_pred_by_type = {
        "searching": sum(1 for sq in unmatched_pred if sq.sq_type == "searching"),
        "aggregation": sum(1 for sq in unmatched_pred if sq.sq_type == "aggregation"),
    }
    print(f"\n  --- Unmatched (no Hungarian partner at all; count_pred==0 or count_ref==0 samples) ---")
    print(f"  Ref sub-Qs never paired:  {n_unmatched_ref} (searching={unmatched_ref_by_type['searching']}, aggregation={unmatched_ref_by_type['aggregation']})")
    print(f"  Pred sub-Qs never paired: {n_unmatched_pred} (searching={unmatched_pred_by_type['searching']}, aggregation={unmatched_pred_by_type['aggregation']})")
    result["unmatched_ref"] = {"total": n_unmatched_ref, **unmatched_ref_by_type}
    result["unmatched_pred"] = {"total": n_unmatched_pred, **unmatched_pred_by_type}

    out_path = os.path.join("nhanes_model", args.run_name, "subquestion_match_analysis.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
