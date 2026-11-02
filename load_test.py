"""
Backend load test: simulate N concurrent users running the full QA pipeline
(decompose → Cypher gen → Neo4j → GPT synthesis), bypassing Gradio.

This is more aggressive than production because Gradio's queue caps parallel
pipelines at 6 — here we fire all N at once. If this passes, production passes.
"""

import concurrent.futures
import json
import os
import statistics
import sys
import time
import traceback
from datetime import datetime

import requests
from neo4j import GraphDatabase
from openai import OpenAI

# Reuse the project's own helpers without triggering app.py startup
sys.path.insert(0, os.path.dirname(__file__))
from few_shot_cypher import (
    FEW_SHOT_TEMPLATE,
    NEO4J_PASSWORD,
    NEO4J_URI,
    NEO4J_USER,
    extract_subquestions,
    generate_cypher_sglang,
    truncate_output,
)

# ─── Config (matches app.py) ─────────────────────────────────────────────────
SGLANG_URL = "http://localhost:30000"
OPENAI_API_KEY = (
    "sk-proj-GA8mdwp4r4-kxoXaw0IpGnQ7fTAbR8SEuw_V5zFqZNTBaD3sHAiNyHrbzfmrObKPNfx2KRk"
    "EPAT3BlbkFJPI3y88p2UBjLWtDwaKKcZcsAudm1iC12_sVmqw3an8dh2v5nQrdaLcTnU0ezx2GovPAY53JhQA"
)

DECOMP_SYSTEM_PROMPT = (
    "You are a question decomposition assistant. Given a complex question, "
    "decompose it into atomic sub-questions, each tagged as searching or aggregation."
)
DECOMP_USER_TEMPLATE = "Question: {q}\n\nDecompose into atomic sub-questions:"

SYNTHESIS_PROMPT = """You are a clinical reasoning assistant.
Original question: {q}

Evidence retrieved from the NHANES knowledge graph:
{evidence}

Synthesize a concise, factual answer using ONLY the evidence above.
Final answer:"""


# Test questions: 5 representative complex questions (cycled across users)
TEST_QUESTIONS = [
    "What's the likelihood of patient 113 being diagnosed with depression using logistic regression at 1:4?",
    "Get patient 34's blood mercury, then check if logistic regression (1:3) was accurate.",
    "What is patient 26's vitamin d2 level?",
    "What is the blood lead value of patient 10?",
    "Is patient 50 diagnosed with depression by random forest at 1:2?",
]


# ─── Pipeline helpers (mirror app.py) ────────────────────────────────────────


def decompose(question: str) -> str:
    payload = {
        "text": (
            f"<|im_start|>system\n{DECOMP_SYSTEM_PROMPT}<|im_end|>\n"
            f"<|im_start|>user\n{DECOMP_USER_TEMPLATE.format(q=question)}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        ),
        "sampling_params": {"max_new_tokens": 512, "temperature": 0.0},
    }
    r = requests.post(f"{SGLANG_URL}/generate", json=payload, timeout=120)
    r.raise_for_status()
    out = r.json().get("text", "")
    return truncate_output(out)


def run_cypher(neo4j_driver, cypher: str) -> str:
    if not cypher or not cypher.strip():
        return "EMPTY"
    try:
        with neo4j_driver.session() as s:
            records = [record.data() for record in s.run(cypher)]
        if not records:
            return "EMPTY"
        return json.dumps(records[:5], default=str)[:500]
    except Exception as e:
        return f"ERROR: {type(e).__name__}: {str(e)[:120]}"


def synthesize(oai_client, question: str, evidence: list) -> str:
    evidence_str = "\n".join(
        f"[{i+1}] {row[0]} → {row[2][:200]}" for i, row in enumerate(evidence)
    )
    resp = oai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "user", "content": SYNTHESIS_PROMPT.format(q=question, evidence=evidence_str)},
        ],
        temperature=0.0,
        max_tokens=400,
    )
    return resp.choices[0].message.content.strip()


def run_one_pipeline(user_id: int, question: str, neo4j_driver, oai_client) -> dict:
    """Return per-user metrics; never raises (errors captured in result)."""
    result = {
        "user_id": user_id,
        "question": question,
        "ok": False,
        "stage": "start",
        "decompose_s": None,
        "cypher_neo4j_s": None,
        "synthesize_s": None,
        "total_s": None,
        "num_subqs": 0,
        "neo4j_empty": 0,
        "neo4j_error": 0,
        "error": None,
    }
    t0 = time.time()
    try:
        # Stage 1: decompose
        result["stage"] = "decompose"
        decomp = decompose(question)
        t1 = time.time()
        result["decompose_s"] = round(t1 - t0, 2)
        subqs = extract_subquestions(decomp)
        result["num_subqs"] = len(subqs)
        if not subqs:
            result["error"] = "no_subqs"
            result["total_s"] = round(time.time() - t0, 2)
            return result

        # Stage 2: parallel cypher gen + sequential neo4j exec
        result["stage"] = "cypher+neo4j"
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(8, len(subqs))) as ex:
            cyphers = list(ex.map(
                lambda sq: generate_cypher_sglang(sq["question"], SGLANG_URL, max_new_tokens=128),
                subqs,
            ))
        rows = []
        for sq, cypher in zip(subqs, cyphers):
            res = run_cypher(neo4j_driver, cypher)
            if res == "EMPTY":
                result["neo4j_empty"] += 1
            elif res.startswith("ERROR"):
                result["neo4j_error"] += 1
            rows.append([sq["question"], cypher, res])
        t2 = time.time()
        result["cypher_neo4j_s"] = round(t2 - t1, 2)

        # Stage 3: synthesize
        result["stage"] = "synthesize"
        _ = synthesize(oai_client, question, rows)
        t3 = time.time()
        result["synthesize_s"] = round(t3 - t2, 2)

        result["stage"] = "done"
        result["ok"] = True
    except Exception as e:
        result["error"] = f"{type(e).__name__}: {str(e)[:200]}"
        result["traceback"] = traceback.format_exc()[-500:]
    finally:
        result["total_s"] = round(time.time() - t0, 2)
    return result


def percentile(xs, p):
    if not xs:
        return None
    xs = sorted(xs)
    k = int(round((p / 100) * (len(xs) - 1)))
    return xs[k]


def summarise(label, results, wall_time):
    ok = [r for r in results if r["ok"]]
    fail = [r for r in results if not r["ok"]]
    totals = [r["total_s"] for r in ok]
    decomps = [r["decompose_s"] for r in ok if r["decompose_s"]]
    cyph_neo = [r["cypher_neo4j_s"] for r in ok if r["cypher_neo4j_s"]]
    synth = [r["synthesize_s"] for r in ok if r["synthesize_s"]]

    print(f"\n{'=' * 70}")
    print(f"  {label}: {len(ok)}/{len(results)} succeeded · wall-time {wall_time:.1f}s")
    print(f"{'=' * 70}")
    if totals:
        print(f"  Total latency  p50={percentile(totals, 50):.1f}s  p95={percentile(totals, 95):.1f}s  max={max(totals):.1f}s")
        print(f"  ├─ decompose    mean={statistics.mean(decomps):.1f}s  max={max(decomps):.1f}s")
        print(f"  ├─ cypher+neo4j mean={statistics.mean(cyph_neo):.1f}s  max={max(cyph_neo):.1f}s")
        print(f"  └─ synthesize   mean={statistics.mean(synth):.1f}s  max={max(synth):.1f}s")
        print(f"  Throughput:    {len(ok) / wall_time:.2f} pipelines/s")
    if fail:
        print(f"\n  Failures ({len(fail)}):")
        from collections import Counter
        err_counts = Counter(r.get("error", "unknown") for r in fail)
        for err, count in err_counts.most_common(5):
            print(f"    × {count}× {err}")
    # Neo4j health
    empty = sum(r["neo4j_empty"] for r in ok)
    nerr = sum(r["neo4j_error"] for r in ok)
    total_q = sum(r["num_subqs"] for r in ok)
    if total_q:
        print(f"\n  Neo4j queries: {total_q} total · {empty} empty results · {nerr} errors")


def main():
    n_wave1 = int(os.getenv("WAVE1", "25"))
    n_wave2 = int(os.getenv("WAVE2", "15"))
    wave2_delay = float(os.getenv("WAVE2_DELAY", "5"))
    max_workers = int(os.getenv("MAX_WORKERS", "50"))

    print(f"\n>> Load test: wave1={n_wave1}, wave2={n_wave2} after {wave2_delay}s")

    # Connection sanity check
    print("  [check] SGLang health ...", end=" ", flush=True)
    try:
        r = requests.get(f"{SGLANG_URL}/health", timeout=5)
        print(f"OK ({r.status_code})")
    except Exception as e:
        print(f"FAIL — {e}")
        sys.exit(1)

    print(f"  [check] Neo4j at {NEO4J_URI} ...", end=" ", flush=True)
    neo4j_driver = GraphDatabase.driver(
        NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD),
        max_connection_pool_size=100,
    )
    neo4j_driver.verify_connectivity()
    print("OK")

    print("  [check] OpenAI client ...", end=" ", flush=True)
    oai_client = OpenAI(api_key=OPENAI_API_KEY)
    print("OK")

    # Tasks: cycle questions across users
    def task_for(uid):
        q = TEST_QUESTIONS[uid % len(TEST_QUESTIONS)]
        return lambda: run_one_pipeline(uid, q, neo4j_driver, oai_client)

    all_results = []
    t_start = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as ex:
        # Wave 1
        print(f"\n  >> Firing wave 1: {n_wave1} concurrent users at t=0s ...")
        futures1 = [ex.submit(task_for(i)) for i in range(n_wave1)]
        # Wave 2 after delay
        time.sleep(wave2_delay)
        print(f"  >> Firing wave 2: {n_wave2} more users at t={wave2_delay}s ...")
        futures2 = [ex.submit(task_for(i)) for i in range(n_wave1, n_wave1 + n_wave2)]

        # Collect wave 1 results
        wave1_results = []
        for fut in concurrent.futures.as_completed(futures1):
            wave1_results.append(fut.result())
        wave1_wall = time.time() - t_start

        # Collect wave 2 results
        wave2_results = []
        t_wave2_start = t_start + wave2_delay
        for fut in concurrent.futures.as_completed(futures2):
            wave2_results.append(fut.result())
        wave2_wall = time.time() - t_wave2_start

        all_results = wave1_results + wave2_results

    total_wall = time.time() - t_start
    neo4j_driver.close()

    # Report
    summarise(f"WAVE 1 ({n_wave1} concurrent)", wave1_results, wave1_wall)
    summarise(f"WAVE 2 ({n_wave2} concurrent, started at t={wave2_delay}s)", wave2_results, wave2_wall)
    summarise(f"COMBINED ({len(all_results)} total)", all_results, total_wall)

    # Persist raw results for inspection
    out_path = f"/tmp/load_test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n  Raw results: {out_path}")


if __name__ == "__main__":
    main()
