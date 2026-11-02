"""
Full-pipeline comparison v2 — progressive model + few-shot vs GraphRAG, focused
on FINAL-ANSWER QUALITY, on ood_data/curriculum_test_120_gold.json.

Differences from run_progressive_pipeline_compare.py:
  - test set: ood_data/curriculum_test_120_gold.json (120 samples)
  - answer synthesis (still Qwen3-8B-Base) uses a SELECTIVITY prompt: the model
    is told the retrieved evidence may contain redundant / extra rows and
    columns, and to put only the facts that actually answer the question into
    the answer (monkey-patches run_cypher_qa_comparison.SYNTHESIS_PROMPT_HEAD).
  - answer quality is scored two ways:
      * embedding cosine to gold_answer (kept for continuity; deflated by the
        terse-gold / verbose-prose style gap)
      * GPT-4o judge: factual correctness, completeness, no-hallucination, and
        conciseness vs gold_answer -> a 0-1 quality score

Phases 1-2 (progressive decomposition -> both Cypher methods -> Neo4j exec ->
synthesis) and the graph-info coverage metric are unchanged.

Usage:
    python run_progressive_pipeline_compare_v2.py
    python run_progressive_pipeline_compare_v2.py --limit 8
    python run_progressive_pipeline_compare_v2.py --skip_decomp
"""

import argparse
import concurrent.futures
import json
import os
import re
import time

import numpy as np
import requests
from neo4j import GraphDatabase

import run_cypher_qa_comparison as rcq
from few_shot_cypher import launch_sglang_server, SGLANG_PORT
from graph_schema import build_schema_card
from entity_retrieval import build_vocab
from run_cypher_qa_comparison import (
    run_method, extract_all_subquestions,
    build_fewshot_cypher_prompt, build_grounded_cypher_prompt,
    SCHEMA_PROMPT_HEAD_TEMPLATE, NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD,
)
from run_finetuned_pipeline_eval import (
    truncate_decomposition, score_against_gold, build_echo_values,
    hard_shutdown_sglang,
)
from evaluate import get_openai_client, get_embeddings, EMBEDDING_MODEL, GPT_MODEL

DECOMP_MODEL = "./nhanes_model/ood_curriculum_progressive/merged_model"
BASE_MODEL = "./nhanes_model/base_model_cache/Qwen3-8B-Base"
TEST_FILE = "ood_data/curriculum_test_120_gold.json"
OUT_DIR = "cypher_results_progressive_test120"

SYSTEM_PROMPT = (
    "You are a question decomposition assistant. Given a complex question, "
    "break it down into a sequence of simpler sub-questions that can be answered independently. "
    "For each step, provide a clear sub-question and label its type as either 'searching' "
    "(retrieving specific information) or 'aggregation' (combining results from previous steps)."
)

# Selectivity-guiding synthesis prompt (replaces rcq.SYNTHESIS_PROMPT_HEAD).
SELECTIVE_SYNTHESIS_PROMPT = """You are a clinical data assistant. Answer the user's original question using ONLY the evidence retrieved from a Neo4j knowledge graph.

Original question:
{question}

Sub-questions, Cypher queries, and retrieved results:
{evidence}

How to use the evidence:
- The retrieved rows often contain MORE than the question asks for: extra columns
  (patient id, model name, training ratio that were only filter inputs), extra
  rows, or results from a sub-question whose value isn't actually needed in the
  final answer. Do NOT dump everything. Select only the specific fact(s) that
  directly answer the question.
- Do not restate filter values (patient number, model, ratio, biomarker name)
  unless the question is explicitly asking to confirm them.
- If several rows are relevant (e.g. one per ratio), report just the asked-for
  quantity for each, plus the comparison the question asks for.
- If a needed value is missing, errored, or empty, say it is unavailable — do
  not guess.
- Be concise and direct: state the numbers / labels / ranking asked for in one
  or two sentences. No preamble like "Based on the retrieved evidence".

Final answer:"""

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


def judge_answers(client, items, workers=12):
    """items: list of (question, gold, pred). Returns list of dict verdicts."""
    def one(i):
        q, g, p = items[i]
        prompt = JUDGE_PROMPT.format(question=q, gold=g, pred=p)
        for _ in range(4):
            try:
                r = client.chat.completions.create(
                    model=GPT_MODEL, messages=[{"role": "user", "content": prompt}],
                    temperature=0, max_tokens=400,
                )
                raw = re.sub(r"```json\s*|\s*```", "", r.choices[0].message.content.strip())
                d = json.loads(raw)
                return i, {
                    "correct": bool(d.get("correct", False)),
                    "complete": bool(d.get("complete", False)),
                    "no_hallucination": bool(d.get("no_hallucination", False)),
                    "concise": bool(d.get("concise", False)),
                    "score": float(d.get("score", 0.0)),
                    "note": d.get("note", ""),
                }
            except Exception as e:
                err = e
                time.sleep(1.5)
        print(f"  judge failed [{i}]: {err}")
        return i, {"correct": False, "complete": False, "no_hallucination": False,
                   "concise": False, "score": 0.0, "note": "judge failed"}

    out = [None] * len(items)
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
        for i, v in ex.map(one, range(len(items))):
            out[i] = v
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test_file", default=TEST_FILE)
    ap.add_argument("--out_dir", default=OUT_DIR)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--port", type=int, default=SGLANG_PORT)
    ap.add_argument("--skip_decomp", action="store_true")
    ap.add_argument("--max_new_tokens_decomp", type=int, default=320)
    ap.add_argument("--max_new_tokens_cypher", type=int, default=256)
    ap.add_argument("--max_new_tokens_answer", type=int, default=400)
    ap.add_argument("--repair_retries", type=int, default=1)
    args = ap.parse_args()

    # install the selectivity synthesis prompt used by run_method
    rcq.SYNTHESIS_PROMPT_HEAD = SELECTIVE_SYNTHESIS_PROMPT

    os.makedirs(args.out_dir, exist_ok=True)
    url = f"http://localhost:{args.port}"
    data = json.load(open(args.test_file, encoding="utf-8"))
    if args.limit:
        data = data[: args.limit]
    print(f"Loaded {len(data)} samples from {args.test_file}")

    # ── Phase 1: progressive decomposition ────────────────────────────────
    decomp_cache = os.path.join(args.out_dir, "progressive_decomp_raw.json")
    if args.skip_decomp and os.path.exists(decomp_cache):
        raw_outputs = json.load(open(decomp_cache))["raw_outputs"]
        assert len(raw_outputs) == len(data), "cache size mismatch; drop --skip_decomp"
        print(f"Loaded cached decomposition ({len(raw_outputs)}).")
    else:
        print(f"\n=== PHASE 1: decomposition via {DECOMP_MODEL} ===")
        proc = launch_sglang_server(DECOMP_MODEL, args.port)
        if proc is None:
            print("ERROR: decomp SGLang failed."); return
        try:
            msgs = [[{"role": "system", "content": SYSTEM_PROMPT},
                     {"role": "user", "content": d["input"]}] for d in data]

            def one(i):
                for _ in range(3):
                    try:
                        r = requests.post(f"{url}/v1/chat/completions", json={
                            "model": "default", "messages": msgs[i],
                            "max_tokens": args.max_new_tokens_decomp, "temperature": 0,
                            "chat_template_kwargs": {"enable_thinking": False},
                        }, timeout=180)
                        r.raise_for_status()
                        return i, r.json()["choices"][0]["message"]["content"]
                    except Exception as e:
                        err = e; time.sleep(2)
                print(f"  [{i}] decomp failed: {err}")
                return i, ""

            raw_outputs = [""] * len(data)
            t0 = time.time()
            with concurrent.futures.ThreadPoolExecutor(max_workers=16) as ex:
                for i, txt in ex.map(one, range(len(data))):
                    raw_outputs[i] = txt
            print(f"  decomposed {len(data)} in {time.time()-t0:.0f}s")
        finally:
            hard_shutdown_sglang(proc, args.port)
        json.dump({"raw_outputs": raw_outputs}, open(decomp_cache, "w"), ensure_ascii=False)

    base_samples = []
    for i, (d, raw) in enumerate(zip(data, raw_outputs)):
        subqs = extract_all_subquestions(truncate_decomposition(raw))
        base_samples.append({"index": i, "input": d["input"], "output": d["output"],
                             "subqs": subqs,
                             "n_searching": sum(1 for s in subqs if s["type"] == "searching")})
    print(f"  avg sub-questions/sample: {np.mean([len(s['subqs']) for s in base_samples]):.2f} "
          f"(searching {np.mean([s['n_searching'] for s in base_samples]):.2f})")

    # ── Phase 2: Cypher (both methods) + exec + selective synthesis ───────
    print(f"\n=== PHASE 2: Cypher + exec + selective synthesis ({BASE_MODEL}) ===")
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    driver.verify_connectivity()
    schema_card = build_schema_card(driver)
    schema_prefix = SCHEMA_PROMPT_HEAD_TEMPLATE.format(schema_card=schema_card)
    vocab = build_vocab(driver)
    print(f"Neo4j connected; schema card {len(schema_card)} chars")

    proc = launch_sglang_server(BASE_MODEL, args.port)
    if proc is None:
        print("ERROR: base SGLang failed."); driver.close(); return

    results = {}
    try:
        for mname, pfn in [
            ("method1_fewshot", build_fewshot_cypher_prompt),
            ("method2_schema", lambda q: build_grounded_cypher_prompt(q, schema_prefix, driver, vocab)),
        ]:
            samples, _ = run_method(
                mname, pfn, base_samples, url, driver,
                args.max_new_tokens_cypher, args.max_new_tokens_answer,
                execute=True, repair_retries=args.repair_retries,
            )
            results[mname] = samples
    finally:
        hard_shutdown_sglang(proc, args.port)
        driver.close()

    # ── Scoring ──────────────────────────────────────────────────────────
    client = get_openai_client()
    gold_answers = [d.get("gold_answer", "") for d in data]
    gold_emb = get_embeddings(client, gold_answers, model=EMBEDDING_MODEL)

    summary, per_sample = {}, {}
    for mname, samples in results.items():
        ans_texts = [s["final_answer"] for s in samples]
        ans_emb = get_embeddings(client, ans_texts, model=EMBEDDING_MODEL)
        print(f"\nJudging {mname} answers with {GPT_MODEL} ...")
        verdicts = judge_answers(client, [(data[i]["input"], gold_answers[i], ans_texts[i])
                                          for i in range(len(samples))])

        cov_list, ans_sim, rows = [], [], []
        exec_ok = has_res = n_search = 0
        js = {"correct": 0, "complete": 0, "no_hallucination": 0, "concise": 0}
        jscore = []
        for si, s in enumerate(samples):
            actual = []
            for sq in s["subqs"]:
                if sq["type"] != "searching":
                    continue
                n_search += 1
                e = sq.get("exec", {})
                exec_ok += bool(e.get("success"))
                has_res += bool(e.get("has_results"))
                actual.extend(e.get("rows_data", []) or [])
            sc = score_against_gold(actual, gold_rows_for(data[si]),
                                    build_echo_values(data[si]["input"], vocab))
            if sc["coverage"] is not None:
                cov_list.append(sc["coverage"])
            asim = cos(ans_emb[si], gold_emb[si]); ans_sim.append(asim)
            v = verdicts[si]
            for k in js:
                js[k] += int(v[k])
            jscore.append(v["score"])
            rows.append({
                "input": data[si]["input"], "gold_answer": gold_answers[si],
                "final_answer": s["final_answer"],
                "graph_coverage": sc["coverage"], "answer_cosine": round(asim, 4),
                "judge": v,
            })
        per_sample[mname] = rows
        N = len(samples)
        summary[mname] = {
            "n_samples": N, "searching_subqs": n_search,
            "cypher_exec_rate": round(exec_ok / max(n_search, 1), 4),
            "cypher_nonempty_rate": round(has_res / max(n_search, 1), 4),
            "graph_coverage_mean": round(float(np.mean(cov_list)), 4) if cov_list else None,
            "answer_cosine_mean": round(float(np.mean(ans_sim)), 4),
            "answer_cosine_median": round(float(np.median(ans_sim)), 4),
            "judge_score_mean": round(float(np.mean(jscore)), 4),
            "judge_correct_rate": round(js["correct"] / N, 4),
            "judge_complete_rate": round(js["complete"] / N, 4),
            "judge_no_hallucination_rate": round(js["no_hallucination"] / N, 4),
            "judge_concise_rate": round(js["concise"] / N, 4),
            "judge_correct_and_complete_rate": round(
                sum(1 for r in rows if r["judge"]["correct"] and r["judge"]["complete"]) / N, 4),
        }

    json.dump(summary, open(os.path.join(args.out_dir, "summary.json"), "w"), indent=2)
    json.dump(per_sample, open(os.path.join(args.out_dir, "per_sample.json"), "w"),
              indent=2, ensure_ascii=False, default=str)

    print(f"\n{'='*74}")
    print(f"  Progressive + few-shot vs GraphRAG — final-answer quality — {args.test_file}")
    print(f"{'='*74}")
    hdr = f"{'metric':<38}{'method1_fewshot':>17}{'method2_schema':>17}"
    print(hdr); print("-" * len(hdr))
    for k, label in [
        ("searching_subqs", "searching sub-questions"),
        ("cypher_exec_rate", "Cypher exec-success rate"),
        ("cypher_nonempty_rate", "Cypher non-empty rate"),
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
        print(f"{label:<38}{str(summary['method1_fewshot'][k]):>17}{str(summary['method2_schema'][k]):>17}")
    print(f"\nSaved -> {args.out_dir}/summary.json  +  per_sample.json")


if __name__ == "__main__":
    main()
