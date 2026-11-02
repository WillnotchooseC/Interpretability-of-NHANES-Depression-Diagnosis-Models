"""
Full-pipeline comparison for the retrained progressive model
(nhanes_model/ood_curriculum_progressive) on ood_data/curriculum_test_100_gold.json:

  Phase 1  progressive model decomposes each `input`  (its own SGLang server)
  Phase 2  each SEARCHING sub-question -> Cypher via TWO methods, executed on Neo4j,
           then a final answer is synthesised from the retrieved evidence
             method1_fewshot  = few_shot_cypher.py's fixed few-shot examples
             method2_schema   = graph_schema card + live entity-grounded retrieval (GraphRAG)
           (both via the base Qwen3-8B-Base, one shared SGLang server)

Scored on exactly two axes, per the revised spec:
  A. GRAPH-INFO COVERAGE  — per question, pool every row the method's Cypher
     retrieved and score value-set containment against the pooled `gold_results`
     rows (run_finetuned_pipeline_eval.score_against_gold; filter-echo values
     from the question are excluded so only actually-retrieved facts count).
  B. ANSWER SEMANTIC SIMILARITY — cosine(text-embedding-3-large) between the
     synthesised `final_answer` and the reference `gold_answer`.

Reuses run_cypher_qa_comparison.run_method for the (cypher -> exec -> repair ->
synthesis) pass and run_finetuned_pipeline_eval for gold scoring. Cypher is
generated with a flat cross-sample batch (no cross-step coreference threading);
this applies equally to both methods so the comparison stays fair.

Usage:
    python run_progressive_pipeline_compare.py
    python run_progressive_pipeline_compare.py --limit 10
    python run_progressive_pipeline_compare.py --skip_decomp   # reuse cached decomposition
"""

import argparse
import json
import os
import time

import numpy as np
from neo4j import GraphDatabase

from few_shot_cypher import launch_sglang_server, SGLANG_PORT
from graph_schema import build_schema_card
from entity_retrieval import build_vocab
from run_cypher_qa_comparison import (
    run_method, extract_all_subquestions, sglang_complete_batch,
    build_fewshot_cypher_prompt, build_grounded_cypher_prompt,
    SCHEMA_PROMPT_HEAD_TEMPLATE, NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD,
)
from run_finetuned_pipeline_eval import (
    truncate_decomposition, score_against_gold, build_echo_values,
    hard_shutdown_sglang,
)
from evaluate import get_openai_client, get_embeddings, EMBEDDING_MODEL

DECOMP_MODEL = "./nhanes_model/ood_curriculum_progressive/merged_model"
BASE_MODEL = "./nhanes_model/base_model_cache/Qwen3-8B-Base"
TEST_FILE = "ood_data/curriculum_test_100_gold.json"
OUT_DIR = "cypher_results_progressive_test100"

SYSTEM_PROMPT = (
    "You are a question decomposition assistant. Given a complex question, "
    "break it down into a sequence of simpler sub-questions that can be answered independently. "
    "For each step, provide a clear sub-question and label its type as either 'searching' "
    "(retrieving specific information) or 'aggregation' (combining results from previous steps)."
)


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--port", type=int, default=SGLANG_PORT)
    ap.add_argument("--skip_decomp", action="store_true")
    ap.add_argument("--max_new_tokens_decomp", type=int, default=320)
    ap.add_argument("--max_new_tokens_cypher", type=int, default=256)
    ap.add_argument("--max_new_tokens_answer", type=int, default=400)
    ap.add_argument("--repair_retries", type=int, default=1)
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    url = f"http://localhost:{args.port}"
    data = json.load(open(TEST_FILE, encoding="utf-8"))
    if args.limit:
        data = data[: args.limit]
    print(f"Loaded {len(data)} samples from {TEST_FILE}")

    # ── Phase 1: progressive-model decomposition ───────────────────────────
    decomp_cache = os.path.join(OUT_DIR, "progressive_decomp_raw.json")
    if args.skip_decomp and os.path.exists(decomp_cache):
        raw_outputs = json.load(open(decomp_cache))["raw_outputs"]
        print(f"Loaded cached decomposition ({len(raw_outputs)}) — Phase 1 SGLang skipped.")
        assert len(raw_outputs) == len(data), "cache size mismatch; drop --skip_decomp"
    else:
        print(f"\n=== PHASE 1: decomposition via {DECOMP_MODEL} ===")
        proc = launch_sglang_server(DECOMP_MODEL, args.port)
        if proc is None:
            print("ERROR: decomp SGLang failed to launch.")
            return
        try:
            prompts = [
                [{"role": "system", "content": SYSTEM_PROMPT},
                 {"role": "user", "content": d["input"]}]
                for d in data
            ]
            # run_cypher_qa_comparison.sglang_complete_batch is a raw /completions
            # helper; use the chat endpoint here to match how the model was trained.
            import concurrent.futures, requests
            def one(i):
                for _ in range(3):
                    try:
                        r = requests.post(f"{url}/v1/chat/completions", json={
                            "model": "default", "messages": prompts[i],
                            "max_tokens": args.max_new_tokens_decomp, "temperature": 0,
                            "chat_template_kwargs": {"enable_thinking": False},
                        }, timeout=180)
                        r.raise_for_status()
                        return i, r.json()["choices"][0]["message"]["content"]
                    except Exception as e:
                        err = e
                        time.sleep(2)
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
        dec = truncate_decomposition(raw)
        subqs = extract_all_subquestions(dec)
        base_samples.append({
            "index": i, "input": d["input"], "output": d["output"],
            "subqs": subqs,
            "n_searching": sum(1 for s in subqs if s["type"] == "searching"),
        })
    print(f"  avg sub-questions/sample: {np.mean([len(s['subqs']) for s in base_samples]):.2f} "
          f"(searching {np.mean([s['n_searching'] for s in base_samples]):.2f})")

    # ── Phase 2: Cypher (both methods) + exec + synthesis ─────────────────
    print(f"\n=== PHASE 2: Cypher + execution + synthesis ({BASE_MODEL}) ===")
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    driver.verify_connectivity()
    print(f"Neo4j connected: {NEO4J_URI}")
    schema_card = build_schema_card(driver)
    schema_prefix = SCHEMA_PROMPT_HEAD_TEMPLATE.format(schema_card=schema_card)
    vocab = build_vocab(driver)
    print(f"schema card: {len(schema_card)} chars")

    proc = launch_sglang_server(BASE_MODEL, args.port)
    if proc is None:
        print("ERROR: base-model SGLang failed to launch.")
        driver.close()
        return

    results = {}
    try:
        for mname, pfn in [
            ("method1_fewshot", build_fewshot_cypher_prompt),
            ("method2_schema", lambda q: build_grounded_cypher_prompt(q, schema_prefix, driver, vocab)),
        ]:
            samples, refs = run_method(
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

    summary = {}
    per_sample = {}
    for mname, samples in results.items():
        ans_texts = [s["final_answer"] for s in samples]
        ans_emb = get_embeddings(client, ans_texts, model=EMBEDDING_MODEL)

        cov_list, ans_sim_list, full_list = [], [], []
        exec_ok = has_res = n_search = 0
        rows = []
        for si, s in enumerate(samples):
            actual = []
            for sq in s["subqs"]:
                if sq["type"] != "searching":
                    continue
                n_search += 1
                e = sq.get("exec", {})
                if e.get("success"):
                    exec_ok += 1
                if e.get("has_results"):
                    has_res += 1
                actual.extend(e.get("rows_data", []) or [])
            g_rows = gold_rows_for(data[si])
            echo = build_echo_values(data[si]["input"], vocab)
            sc = score_against_gold(actual, g_rows, echo)
            covv = sc["coverage"]
            if covv is not None:
                cov_list.append(covv)
                full_list.append(1.0 if (sc["n_gold_rows"] and sc["full_matches"] == sc["n_gold_rows"]) else 0.0)
            asim = cos(ans_emb[si], gold_emb[si])
            ans_sim_list.append(asim)
            rows.append({
                "input": data[si]["input"],
                "n_searching_pred": s["n_searching"] if "n_searching" in s else sum(1 for x in s["subqs"] if x["type"] == "searching"),
                "graph_coverage": covv,
                "gold_full_match": (sc["n_gold_rows"] and sc["full_matches"] == sc["n_gold_rows"]),
                "answer_cosine": round(asim, 4),
                "final_answer": s["final_answer"],
                "gold_answer": gold_answers[si],
            })
        per_sample[mname] = rows
        summary[mname] = {
            "n_samples": len(samples),
            "searching_subqs": n_search,
            "cypher_exec_rate": round(exec_ok / max(n_search, 1), 4),
            "cypher_nonempty_rate": round(has_res / max(n_search, 1), 4),
            "graph_coverage_mean": round(float(np.mean(cov_list)), 4) if cov_list else None,
            "graph_coverage_scored_samples": len(cov_list),
            "gold_full_match_rate": round(float(np.mean(full_list)), 4) if full_list else None,
            "answer_cosine_mean": round(float(np.mean(ans_sim_list)), 4),
            "answer_cosine_median": round(float(np.median(ans_sim_list)), 4),
        }

    json.dump(summary, open(os.path.join(OUT_DIR, "summary.json"), "w"), indent=2)
    json.dump(per_sample, open(os.path.join(OUT_DIR, "per_sample.json"), "w"),
              indent=2, ensure_ascii=False, default=str)

    print(f"\n{'='*72}")
    print(f"  Progressive model — few-shot vs GraphRAG on {TEST_FILE}")
    print(f"{'='*72}")
    hdr = f"{'metric':<34}{'method1_fewshot':>18}{'method2_schema':>18}"
    print(hdr); print("-" * len(hdr))
    keys = [
        ("searching_subqs", "searching sub-questions"),
        ("cypher_exec_rate", "Cypher exec-success rate"),
        ("cypher_nonempty_rate", "Cypher non-empty rate"),
        ("graph_coverage_mean", "GRAPH-INFO COVERAGE (mean)"),
        ("gold_full_match_rate", "  full gold-row match rate"),
        ("answer_cosine_mean", "ANSWER SIMILARITY (mean)"),
        ("answer_cosine_median", "  answer similarity (median)"),
    ]
    for k, label in keys:
        v1, v2 = summary["method1_fewshot"][k], summary["method2_schema"][k]
        print(f"{label:<34}{str(v1):>18}{str(v2):>18}")
    print(f"\nSaved -> {OUT_DIR}/summary.json  +  per_sample.json")


if __name__ == "__main__":
    main()
