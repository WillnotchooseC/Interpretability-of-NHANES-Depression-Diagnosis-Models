"""
Export the N-rep sub-question outputs of one SFT run on
ood_data/curriculum_test_250.json, together with the reference decomposition,
for EXTERNAL judging (no GPT-4o here, no verdicts -- the judge model fills them in).

Reads nhanes_model/<run>/eval_curriculum_test_250_rep{N}/test_predictions.json
(from gen_test250_reps.py) and writes:
  cypher_results/test250_<run>_reps_for_judging.yaml   human/LLM-readable
  cypher_results/test250_<run>_reps_for_judging.json   machine-readable

Only `Sub-question:` lines are exported -- they are the unit that gets scored
and the only part the downstream KG pipeline consumes (<PLAN> / "Step N:" text
is auxiliary chain-of-thought that parse_output already discards). Raw
generations stay in the per-rep test_predictions.json if the parse itself
needs auditing.

Also prints a run-to-run determinism summary (identical predicted sub-question
strings vs rep1) and counts, so the statistical write-up can quote them.

Usage:
    python export_test250_reps_for_judging.py
    python export_test250_reps_for_judging.py --run ood_curriculum_baseline_mixed1076_corrected --reps 1 2 3
"""

import argparse
import json
import os
from collections import Counter

from evaluate import parse_output
from gen_test250_reps import out_path, DEFAULT_RUN, TEST_FILE

HEADER = """\
# Sub-question decomposition outputs for judging with an independent model.
#
# Task: an SFT model reads `question` and must decompose it into sub-questions.
# One model, run {n_reps} times at temperature=0 (SGLang batched inference is not
# bit-deterministic, so repetitions can differ). Each rep is to be judged
# INDEPENDENTLY against the same reference.
#
# `split`: seen   = question template seen during training
#          unseen = held-out template (the OOD generalisation test)
#
# METRICS TO JUDGE (fill in `covered` / `grounded` per item)
#   1. RECALL        : fraction of REFERENCE sub-questions whose information
#                      need the model's predicted set COLLECTIVELY expresses.
#                      The prediction may be more explicit (spelling out model
#                      names, the ratio, the patient id) and may split one
#                      reference sub-question across several predicted ones --
#                      still counts as covered. One predicted sub-question may
#                      also cover two reference sub-questions at once.
#                      Mark covered=false only if the information is genuinely absent.
#   2. HALLUCINATION : fraction of PREDICTED sub-questions that correspond to
#                      no reference sub-question's information need -- a
#                      different entity/attribute/relation, a fabricated
#                      detail, or content unrelated to answering the question
#                      as the reference scopes it. grounded=false => hallucinated.
#   3. FULL COVERAGE : a question is fully covered iff every one of its
#                      reference sub-questions is covered.
#
# Sub-question `type` is the model's own searching/aggregation label; it is NOT
# part of the three metrics above.
"""


def yaml_str(s):
    return json.dumps(s, ensure_ascii=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=DEFAULT_RUN)
    ap.add_argument("--reps", nargs="+", type=int, default=[1, 2, 3])
    ap.add_argument("--test_file", default=TEST_FILE)
    ap.add_argument("--out_dir", default="cypher_results")
    args = ap.parse_args()
    test_stem = os.path.splitext(os.path.basename(args.test_file))[0]

    preds = {}
    for rep in args.reps:
        p = out_path(args.run, rep, args.test_file)
        if not os.path.exists(p):
            raise SystemExit(f"missing rep{rep}: {p}")
        preds[rep] = json.load(open(p, encoding="utf-8"))
    n = len(preds[args.reps[0]])
    assert all(len(v) == n for v in preds.values())

    records = []
    for i in range(n):
        base = preds[args.reps[0]][i]
        rec = {
            "idx": i,
            "split": base.get("split"),
            "question": base["input"],
            "reference_subquestions": [
                {"text": q.sub_question, "type": q.sq_type} for q in parse_output(base["reference"])
            ],
            "reps": {},
        }
        for rep in args.reps:
            s = preds[rep][i]
            assert s["input"] == base["input"], f"rep{rep} sample order mismatch at {i}"
            rec["reps"][f"rep{rep}"] = {
                "predicted_subquestions": [
                    {"text": q["text"], "type": q["type"], "grounded": None}
                    for q in s["predicted_subquestions"]
                ],
                "reference_covered": [
                    {"ref_text": q["text"], "covered": None} for q in rec["reference_subquestions"]
                ],
            }
        records.append(rec)

    os.makedirs(args.out_dir, exist_ok=True)
    stem = os.path.join(args.out_dir, f"{test_stem.replace('curriculum_', '')}_{args.run}_reps_for_judging")
    with open(stem + ".json", "w", encoding="utf-8") as f:
        json.dump({"run": args.run, "test_file": args.test_file, "reps": args.reps,
                   "temperature": 0, "records": records}, f, indent=2, ensure_ascii=False)

    lines = [HEADER.format(n_reps=len(args.reps))]
    for rec in records:
        lines.append(f"- idx: {rec['idx']}")
        lines.append(f"  split: {rec['split']}")
        lines.append(f"  question: {yaml_str(rec['question'])}")
        lines.append(f"  reference_subquestions:")
        for q in rec["reference_subquestions"]:
            lines.append(f"    - text: {yaml_str(q['text'])}")
            lines.append(f"      type: {q['type']}")
        lines.append(f"  reps:")
        for rep_name, data in rec["reps"].items():
            lines.append(f"    {rep_name}:")
            lines.append(f"      predicted_subquestions:")
            for q in data["predicted_subquestions"]:
                lines.append(f"        - text: {yaml_str(q['text'])}")
                lines.append(f"          type: {q['type']}")
                lines.append(f"          grounded: null")
            lines.append(f"      reference_covered:")
            for c in data["reference_covered"]:
                lines.append(f"        - ref_text: {yaml_str(c['ref_text'])}")
                lines.append(f"          covered: null")
        lines.append("")
    with open(stem + ".yaml", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    # ---- summary stats (no judging) ----
    print(f"{args.run} | {args.test_file} | reps {args.reps} | {n} questions "
          f"({Counter(r['split'] for r in records)})")
    n_ref = sum(len(r["reference_subquestions"]) for r in records)
    print(f"  reference sub-Qs: {n_ref}")
    key = lambda s: " | ".join(q["text"] for q in s["predicted_subquestions"])
    base_keys = [key(s) for s in preds[args.reps[0]]]
    for rep in args.reps:
        ps = preds[rep]
        n_pred = sum(len(s["predicted_subquestions"]) for s in ps)
        n_none = sum(1 for s in ps if not s["predicted_subquestions"])
        types = Counter(q["type"] for s in ps for q in s["predicted_subquestions"])
        ident = sum(1 for s, k in zip(ps, base_keys) if key(s) == k)
        print(f"  rep{rep}: {n_pred} predicted sub-Qs ({dict(types)}), "
              f"{n_none} samples with none, identical-to-rep1: {ident}/{n}")
    print(f"  {stem}.yaml  ({os.path.getsize(stem + '.yaml')/1024:.0f} KB)")
    print(f"  {stem}.json  ({os.path.getsize(stem + '.json')/1024:.0f} KB)")


if __name__ == "__main__":
    main()
