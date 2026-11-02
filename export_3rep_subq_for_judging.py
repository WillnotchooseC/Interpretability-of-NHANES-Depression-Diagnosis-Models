"""
Export the 3-rep sub-question outputs of both *_corrected runs, together with
the reference decomposition and GPT-4o's current per-item verdicts, so the
whole judging step can be redone with a different (larger) model.

Writes two files:
  cypher_results/corrected_3rep_subq_for_judging.yaml
      Human/LLM-readable. One block per test question: the original question,
      the reference sub-questions, then each model's three repetitions. Every
      sub-question carries GPT-4o's verdict inline, so a new judge's
      disagreements with GPT-4o can be read off directly.
  cypher_results/corrected_3rep_subq_for_judging.json
      Same content, machine-readable, for scripted re-judging.

Only `Sub-question:` lines are exported -- they are the unit that gets scored
and the only part the downstream KG pipeline consumes (<PLAN> / "Step N:" text
is auxiliary chain-of-thought that parse_output already discards). The raw
generations remain at nhanes_model/<run>/eval_curriculum_test_120{,_rep2,_rep3}/
test_predictions.json if the parse itself needs auditing.

Verdict fields mirror eval_progressive_vs_baseline_subq_ci.judge_sample:
  reference sub-question .covered  -- did the predicted set collectively express
                                      this information need? (drives RECALL)
  predicted sub-question .grounded -- does this predicted sub-question correspond
                                      to some reference sub-question's information
                                      need? false => HALLUCINATION

Usage:
    python export_3rep_subq_for_judging.py
"""

import json
import os

from eval_progressive_vs_baseline_subq_ci import load_samples, UNSEEN_IDX
from eval_corrected_3rep_subq_ci import MODELS, REPS, pred_path

CACHE = "cypher_results/corrected_3rep_judge_cache.json"
OUT_YAML = "cypher_results/corrected_3rep_subq_for_judging.yaml"
OUT_JSON = "cypher_results/corrected_3rep_subq_for_judging.json"

HEADER = """\
# Sub-question decomposition outputs for re-judging with an independent model.
#
# Task: an SFT model reads `question` and must decompose it into sub-questions.
# Two models are compared, each run 3 times (temperature=0; SGLang batched
# inference is not bit-deterministic, so repetitions can differ):
#   progressive_corrected : curriculum / progressive SFT (stage1 -> stage2 -> stage3)
#   baseline_corrected    : single-file SFT on the same corrected data
#
# `split` marks whether the question's template was seen in training:
#   seen_80   (80 questions)  template seen during training
#   unseen_40 (40 questions)  held-out template = the OOD generalisation test
#
# THE THREE METRICS BEING JUDGED
#   1. RECALL          : fraction of REFERENCE sub-questions whose information
#                        need the model's predicted set COLLECTIVELY expresses.
#                        The prediction may be more explicit (spelling out model
#                        names, the ratio, the patient id) and may split one
#                        reference sub-question across several predicted ones --
#                        still counts as covered. One predicted sub-question may
#                        also cover two reference sub-questions at once.
#                        Only mark covered=false if the information is genuinely absent.
#   2. HALLUCINATION   : fraction of PREDICTED sub-questions that correspond to
#                        no reference sub-question's information need -- a
#                        different entity/attribute/relation, a fabricated
#                        detail, or content unrelated to answering the question
#                        as the reference scopes it. grounded=false => hallucinated.
#   3. FULL COVERAGE   : a question is fully covered iff every one of its
#                        reference sub-questions is covered.
#
# `gpt4o_covered` / `gpt4o_grounded` below are the CURRENT gpt-4o verdicts under
# exactly those definitions -- included so a new judge can be diffed against them.
# Sub-question `type` is the model's own searching/aggregation label; it is NOT
# part of the three metrics above.
"""


def yaml_str(s):
    """Quote a scalar safely for YAML (these strings contain ':' and quotes)."""
    return json.dumps(s, ensure_ascii=False)


def main():
    cache = json.load(open(CACHE))

    # samples[label][rep] -> parsed samples; reference is identical across all of them
    parsed = {}
    for label, run_name in MODELS.items():
        parsed[label] = {rep: load_samples(pred_path(run_name, rep)) for rep in REPS}

    ref_samples = parsed[list(MODELS)[0]][1]
    n = len(ref_samples)

    records = []
    for i in range(n):
        base = ref_samples[i]
        rec = {
            "idx": i,
            "split": "unseen_40" if i in UNSEEN_IDX else "seen_80",
            "question": base["input"],
            "reference_subquestions": [
                {"text": q.sub_question, "type": q.sq_type} for q in base["ref"]
            ],
            "models": {},
        }
        for label in MODELS:
            reps_out = {}
            for rep in REPS:
                s = parsed[label][rep][i]
                ref_verdict, pred_verdict = cache[f"{label}|rep{rep}|None"][i]
                assert len(ref_verdict) == len(s["ref"]), f"verdict/ref mismatch {label} rep{rep} #{i}"
                assert len(pred_verdict) == len(s["pred"]), f"verdict/pred mismatch {label} rep{rep} #{i}"
                reps_out[f"rep{rep}"] = {
                    "predicted_subquestions": [
                        {"text": q.sub_question, "type": q.sq_type,
                         "gpt4o_grounded": bool(g)}
                        for q, g in zip(s["pred"], pred_verdict)
                    ],
                    "gpt4o_reference_covered": [
                        {"ref_text": q.sub_question, "covered": bool(c)}
                        for q, c in zip(s["ref"], ref_verdict)
                    ],
                }
            rec["models"][label] = reps_out
        records.append(rec)

    os.makedirs("cypher_results", exist_ok=True)
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)

    lines = [HEADER]
    for rec in records:
        lines.append(f"- idx: {rec['idx']}")
        lines.append(f"  split: {rec['split']}")
        lines.append(f"  question: {yaml_str(rec['question'])}")
        lines.append(f"  reference_subquestions:")
        for q in rec["reference_subquestions"]:
            lines.append(f"    - text: {yaml_str(q['text'])}")
            lines.append(f"      type: {q['type']}")
        lines.append(f"  models:")
        for label, reps_out in rec["models"].items():
            lines.append(f"    {label}:")
            for rep_name, data in reps_out.items():
                lines.append(f"      {rep_name}:")
                lines.append(f"        predicted_subquestions:")
                for q in data["predicted_subquestions"]:
                    lines.append(f"          - text: {yaml_str(q['text'])}")
                    lines.append(f"            type: {q['type']}")
                    lines.append(f"            gpt4o_grounded: {str(q['gpt4o_grounded']).lower()}")
                lines.append(f"        gpt4o_reference_covered:")
                for c in data["gpt4o_reference_covered"]:
                    lines.append(f"          - ref_text: {yaml_str(c['ref_text'])}")
                    lines.append(f"            covered: {str(c['covered']).lower()}")
        lines.append("")

    with open(OUT_YAML, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    n_ref = sum(len(r["reference_subquestions"]) for r in records)
    n_pred = sum(len(d["predicted_subquestions"])
                 for r in records for m in r["models"].values() for d in m.values())
    print(f"{n} questions | {n_ref} reference sub-Qs | "
          f"{n_pred} predicted sub-Qs across {len(MODELS)} models x {len(REPS)} reps")
    print(f"  {OUT_YAML}  ({os.path.getsize(OUT_YAML)/1024:.0f} KB)")
    print(f"  {OUT_JSON}  ({os.path.getsize(OUT_JSON)/1024:.0f} KB)")


if __name__ == "__main__":
    main()
