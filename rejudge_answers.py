"""
Re-run ONLY the GPT-4o answer-quality judge over an existing per_sample.json
(the in-pipeline judge hit the 30k TPM gpt-4o rate limit with 12 workers and
recorded score=0.0 for the calls that 429'd). This one uses low concurrency +
honours the 429 retry-after, and rewrites summary.json's judge_* fields.

Usage:
    python rejudge_answers.py --dir cypher_results_progressive_test120
"""
import argparse
import json
import os
import re
import time

import numpy as np

from evaluate import get_openai_client, GPT_MODEL

JUDGE_PROMPT = """You are grading a knowledge-graph QA system's final answer against a gold answer.

Question: {question}

Gold answer: {gold}

Model answer: {pred}

Grade the model answer on:
- "correct": do ALL the specific facts it states (numeric values, predicted
  labels, directions, which ratio/model is highest/lowest, rankings) agree with
  the gold answer? Minor rounding (0.19 vs 0.190) is fine. false if any stated
  fact is wrong or contradicts gold.
- "complete": does it answer every part the question asks for (all requested
  values AND any comparison / highest-lowest / ranking)? false if part is missing.
- "no_hallucination": true if it does NOT invent facts absent from the gold /
  evidence. false if it adds unsupported specific claims.
- "concise": true if it sticks to the asked-for facts without dumping redundant
  retrieved fields (patient id, model, ratio restated needlessly) or padding.
- "score": overall answer quality 0.0-1.0 (1.0 = fully correct, complete, clean).

Respond JSON only:
{{"correct": true, "complete": true, "no_hallucination": true, "concise": true, "score": 1.0, "note": "..."}}"""


def judge_one(client, q, g, p, max_retries=8):
    prompt = JUDGE_PROMPT.format(question=q, gold=g, pred=p)
    for attempt in range(max_retries):
        try:
            r = client.chat.completions.create(
                model=GPT_MODEL, messages=[{"role": "user", "content": prompt}],
                temperature=0, max_tokens=400,
            )
            raw = re.sub(r"```json\s*|\s*```", "", r.choices[0].message.content.strip())
            d = json.loads(raw)
            return {
                "correct": bool(d.get("correct", False)),
                "complete": bool(d.get("complete", False)),
                "no_hallucination": bool(d.get("no_hallucination", False)),
                "concise": bool(d.get("concise", False)),
                "score": float(d.get("score", 0.0)),
                "note": d.get("note", ""),
            }
        except Exception as e:
            msg = str(e)
            wait = 3.0 * (attempt + 1)
            m = re.search(r"try again in ([\d.]+)s", msg)
            if m:
                wait = float(m.group(1)) + 0.5
            if attempt < max_retries - 1:
                time.sleep(wait)
            else:
                print(f"    give up: {msg[:120]}")
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    args = ap.parse_args()

    ps_path = os.path.join(args.dir, "per_sample.json")
    sm_path = os.path.join(args.dir, "summary.json")
    per_sample = json.load(open(ps_path, encoding="utf-8"))
    summary = json.load(open(sm_path, encoding="utf-8"))
    client = get_openai_client()

    for mname, rows in per_sample.items():
        print(f"\n=== re-judging {mname} ({len(rows)} answers), serial, rate-limit aware ===")
        t0 = time.time()
        for i, r in enumerate(rows):
            v = judge_one(client, r["input"], r["gold_answer"], r["final_answer"])
            if v is not None:
                r["judge"] = v
            if (i + 1) % 20 == 0:
                print(f"  {i+1}/{len(rows)}  ({time.time()-t0:.0f}s)")
        N = len(rows)
        js = {k: sum(1 for r in rows if r["judge"].get(k)) for k in
              ["correct", "complete", "no_hallucination", "concise"]}
        scores = [r["judge"].get("score", 0.0) for r in rows]
        cc = sum(1 for r in rows if r["judge"].get("correct") and r["judge"].get("complete"))
        summary[mname].update({
            "judge_score_mean": round(float(np.mean(scores)), 4),
            "judge_correct_rate": round(js["correct"] / N, 4),
            "judge_complete_rate": round(js["complete"] / N, 4),
            "judge_no_hallucination_rate": round(js["no_hallucination"] / N, 4),
            "judge_concise_rate": round(js["concise"] / N, 4),
            "judge_correct_and_complete_rate": round(cc / N, 4),
        })
        print(f"  {mname}: score={summary[mname]['judge_score_mean']} "
              f"correct={summary[mname]['judge_correct_rate']} "
              f"complete={summary[mname]['judge_complete_rate']} "
              f"corr&comp={summary[mname]['judge_correct_and_complete_rate']} "
              f"no_halluc={summary[mname]['judge_no_hallucination_rate']} "
              f"concise={summary[mname]['judge_concise_rate']}")

    json.dump(per_sample, open(ps_path, "w"), indent=2, ensure_ascii=False, default=str)
    json.dump(summary, open(sm_path, "w"), indent=2)
    print(f"\nRewrote {ps_path} and {sm_path}")

    print(f"\n{'metric':<38}{'method1_fewshot':>17}{'method2_schema':>17}")
    for k, label in [
        ("graph_coverage_mean", "graph-info coverage (mean)"),
        ("answer_cosine_mean", "answer cosine vs gold (mean)"),
        ("answer_cosine_median", "answer cosine vs gold (median)"),
        ("judge_score_mean", "JUDGE quality score (mean)"),
        ("judge_correct_rate", "  judge: factually correct"),
        ("judge_complete_rate", "  judge: complete"),
        ("judge_correct_and_complete_rate", "  judge: correct AND complete"),
        ("judge_no_hallucination_rate", "  judge: no hallucination"),
        ("judge_concise_rate", "  judge: concise / selective"),
    ]:
        print(f"{label:<38}{str(summary['method1_fewshot'].get(k)):>17}{str(summary['method2_schema'].get(k)):>17}")


if __name__ == "__main__":
    main()
