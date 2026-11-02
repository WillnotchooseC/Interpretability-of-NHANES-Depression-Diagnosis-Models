"""
GPT-4o sub-question COVERAGE judge (redundancy-tolerant).

Revised eval spec:
  - Only `Sub-question:` lines matter (<PLAN> / `Step N:` text is ignored).
  - For every REFERENCE sub-question, decide whether the model's set of
    predicted sub-questions COLLECTIVELY covers it: expresses the same
    information need. Added specificity (naming the models/ratio explicitly,
    referencing "the retrieved probabilities", etc.) still counts as covered.
  - Extra / redundant predicted sub-questions are NOT penalised.
  - Also judge whether the covering predicted sub-question has the same
    type (searching / aggregation) as the reference.
  - A sample "fully contains the expected result" iff every reference
    sub-question is covered.

Embedding cosine (evaluate.py Dim 1 / analyze_subquestion_match.py) is unreliable
here: the reference aggregation sub-questions are terse and generic
("Which requested model gives the highest depression probability?") and score
~0.6 cosine against the models' more explicit phrasings despite being
semantically equivalent. Hence an LLM entailment judge.

Usage:
    python judge_subq_coverage.py --pred_file <test_predictions.json> --label progressive
"""

import argparse
import concurrent.futures
import json
import re
import time

from evaluate import get_openai_client, parse_output, truncate_output, GPT_MODEL

MAX_RETRIES = 4


def judge_sample(client, inp, ref_subs, pred_subs, model=GPT_MODEL):
    ref_list = "\n".join(f"  R{i+1} [{q.sq_type}]: {q.sub_question}" for i, q in enumerate(ref_subs))
    pred_list = "\n".join(f"  P{j+1} [{q.sq_type}]: {q.sub_question}" for j, q in enumerate(pred_subs)) or "  (none)"
    prompt = f"""You are grading a question-decomposition model.

Original question: "{inp}"

REFERENCE sub-questions (the expected decomposition):
{ref_list}

MODEL's predicted sub-questions:
{pred_list}

For EACH reference sub-question Ri, decide:
  - "covered": true if one or more predicted sub-questions together express the
    same information need as Ri. The model MAY be more explicit (spelling out
    model names, the training ratio, "the retrieved probabilities", the patient
    id) and MAY split Ri across several predicted sub-questions - still covered.
    Only mark false if the information Ri asks for is genuinely absent from the
    predictions.
  - "covering": list of predicted indices (e.g. ["P1","P2"]) that cover it, or [].
  - "type_match": true if at least one covering prediction has the SAME type
    label ([searching] / [aggregation]) as Ri. false if covered but only by a
    wrong-type prediction. null if not covered.

Respond JSON only, no markdown:
{{"results": [{{"ref": 1, "covered": true, "covering": ["P1"], "type_match": true}}]}}"""

    for attempt in range(MAX_RETRIES):
        try:
            resp = client.chat.completions.create(
                model=model, messages=[{"role": "user", "content": prompt}],
                temperature=0, max_tokens=1200,
            )
            raw = resp.choices[0].message.content.strip()
            data = json.loads(re.sub(r"```json\s*|\s*```", "", raw).strip())
            out = []
            res = data.get("results", [])
            for i in range(len(ref_subs)):
                r = res[i] if i < len(res) else {}
                out.append({
                    "covered": bool(r.get("covered", False)),
                    "type_match": r.get("type_match", None),
                })
            return out
        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                time.sleep(1.5 * (attempt + 1))
            else:
                print(f"  judge failed: {e}")
    return [{"covered": False, "type_match": None} for _ in ref_subs]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred_file", required=True)
    ap.add_argument("--label", default="model")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    raw = json.load(open(args.pred_file, encoding="utf-8"))
    samples = []
    for p in raw:
        pred_text = p.get("prediction") or truncate_output(p["prediction_raw"])
        samples.append({
            "input": p["input"],
            "ref": parse_output(p["reference"]),
            "pred": parse_output(pred_text),
        })

    client = get_openai_client()
    print(f"Judging {len(samples)} samples with {GPT_MODEL} ...")

    def work(idx):
        s = samples[idx]
        return idx, judge_sample(client, s["input"], s["ref"], s["pred"])

    verdicts = [None] * len(samples)
    t0 = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as ex:
        for idx, v in ex.map(work, range(len(samples))):
            verdicts[idx] = v
    print(f"  done in {time.time()-t0:.0f}s")

    tot = {"searching": 0, "aggregation": 0}
    cov = {"searching": 0, "aggregation": 0}
    tym = {"searching": 0, "aggregation": 0}   # covered AND type_match
    n_full = n_full_type = 0
    ref_c, pred_c = [], []
    missing = []

    for s, v in zip(samples, verdicts):
        ref, pred = s["ref"], s["pred"]
        ref_c.append(len(ref)); pred_c.append(len(pred))
        all_cov = True; all_typ = True
        miss_here = []
        for q, vi in zip(ref, v):
            tot[q.sq_type] += 1
            if vi["covered"]:
                cov[q.sq_type] += 1
                if vi["type_match"]:
                    tym[q.sq_type] += 1
                else:
                    all_typ = False
            else:
                all_cov = False; all_typ = False
                miss_here.append(q.sub_question)
        if ref and all_cov:
            n_full += 1
            if all_typ:
                n_full_type += 1
        if miss_here and len(missing) < 10:
            missing.append({"input": s["input"], "missing": miss_here})

    N = len(samples)
    T = tot["searching"] + tot["aggregation"]
    C = cov["searching"] + cov["aggregation"]
    TM = tym["searching"] + tym["aggregation"]
    over = sum(1 for a, b in zip(pred_c, ref_c) if a > b)
    under = sum(1 for a, b in zip(pred_c, ref_c) if a < b)
    exact = sum(1 for a, b in zip(pred_c, ref_c) if a == b)

    def pct(a, b): return f"{a}/{b} ({a/max(b,1)*100:.1f}%)"

    print(f"\n{'='*66}")
    print(f"  GPT-4o sub-question coverage — {args.label}")
    print(f"  {N} samples | ref sub-Qs={T} (search {tot['searching']} / agg {tot['aggregation']})")
    print(f"{'='*66}")
    print(f"\n  COVERAGE  (ref sub-Q's info need met by the predictions; extras OK)")
    print(f"    Overall:      {pct(C, T)}")
    print(f"    searching:    {pct(cov['searching'], tot['searching'])}")
    print(f"    aggregation:  {pct(cov['aggregation'], tot['aggregation'])}")
    print(f"\n  PER-SAMPLE  (output fully contains every expected sub-question)")
    print(f"    Full coverage:                {pct(n_full, N)}")
    print(f"    Full coverage + types match:  {pct(n_full_type, N)}")
    print(f"\n  TYPE CONSISTENCY  (among covered ref sub-Qs)")
    print(f"    Overall:      {pct(TM, C)}")
    print(f"    searching:    {pct(tym['searching'], cov['searching'])}")
    print(f"    aggregation:  {pct(tym['aggregation'], cov['aggregation'])}")
    print(f"\n  COUNT  (redundancy tolerated; 'under' = missing = the real failure)")
    print(f"    exact match:   {pct(exact, N)}")
    print(f"    avg ref/pred:  {sum(ref_c)/N:.2f} / {sum(pred_c)/N:.2f}")
    print(f"    over / under:  {over} / {under}")

    if missing:
        print(f"\n  Uncovered reference sub-questions (first {len(missing)} samples):")
        for m in missing:
            print(f"    - {m['input'][:78]}")
            for q in m["missing"]:
                print(f"        MISSING: {q[:88]}")

    result = {
        "label": args.label, "n_samples": N,
        "coverage_overall_pct": C / max(T, 1) * 100,
        "coverage_searching_pct": cov["searching"] / max(tot["searching"], 1) * 100,
        "coverage_aggregation_pct": cov["aggregation"] / max(tot["aggregation"], 1) * 100,
        "full_coverage_samples_pct": n_full / N * 100,
        "full_coverage_and_type_samples_pct": n_full_type / N * 100,
        "type_consistency_overall_pct": TM / max(C, 1) * 100,
        "type_consistency_searching_pct": tym["searching"] / max(cov["searching"], 1) * 100,
        "type_consistency_aggregation_pct": tym["aggregation"] / max(cov["aggregation"], 1) * 100,
        "count_exact_pct": exact / N * 100,
        "avg_ref_count": sum(ref_c) / N, "avg_pred_count": sum(pred_c) / N,
        "count_over": over, "count_under": under,
    }
    out = args.out or args.pred_file.rsplit("/", 1)[0] + "/subq_coverage_gpt.json"
    with open(out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved -> {out}")


if __name__ == "__main__":
    main()
