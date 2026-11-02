"""
Compare two independent Cypher-generation strategies for answering
data/test_dataset.json against the NHANES Neo4j knowledge graph, both driven
by the same local Qwen3-8B-Base model via in-context prompting only (no
fine-tuning, no OpenAI dependency):

  Method 1 (fewshot) — the fixed hand-written Question->Cypher few-shot
                        examples from few_shot_cypher.py's FEW_SHOT_TEMPLATE.
                        The model pattern-matches the new question against
                        ~20 memorized examples.
  Method 2 (schema)  — a live-introspected graph schema card
                        (graph_schema.py) plus generic Cypher syntax shapes.
                        The model must reason about which labels /
                        relationships / properties apply, instead of
                        recalling a memorized question->query mapping.

For each test sample: extract the searching/aggregation sub-questions from
its `output` decomposition, generate Cypher for the searching ones with the
method under test, execute against Neo4j, then synthesize a final answer to
`input` from the retrieved evidence (also via the local model). Both methods
run independently end-to-end and are scored on Cypher execution/validity
rate; predictions and a comparison report are written to cypher_results/.

Usage:
    python run_cypher_qa_comparison.py                 # full run, both methods
    python run_cypher_qa_comparison.py --limit 10       # smoke test on 10 samples
    python run_cypher_qa_comparison.py --no_execute     # Cypher generation only
"""

import argparse
import concurrent.futures
import json
import os
import re
import time

import requests
from neo4j import GraphDatabase

from few_shot_cypher import (
    FEW_SHOT_TEMPLATE,
    build_cypher_prompt,
    launch_sglang_server,
    shutdown_sglang_server,
    SGLANG_PORT,
)
from graph_schema import build_schema_card
from entity_retrieval import build_vocab, extract_entities, retrieve_grounding_facts

MODEL_PATH = "./nhanes_model/base_model_cache/Qwen3-8B-Base"
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "12345678")
TEST_FILE = "data/test_dataset.json"
DEFAULT_OUTPUT_DIR = "cypher_results"
ROW_LIMIT = 300  # cap rows kept as evidence per sub-question — Neo4j doesn't guarantee
# row order without an explicit ORDER BY, so a low cap on a genuinely multi-row query
# (e.g. one biomarker/model per row) can silently drop an entire group's rows rather
# than truncating evenly; 300 comfortably covers the largest realistic result here
# (a full 290-patient cross-patient aggregation) without meaningfully bloating storage.

SCHEMA_PROMPT_HEAD_TEMPLATE = """You are an expert in translating English questions into Cypher queries for a Neo4j database. Use ONLY the labels, relationship types, and properties listed in the schema below for this specific database - never invent one that is not listed.

{schema_card}

Cypher syntax reference (illustrative pattern shapes only, not tied to any specific label/property above):

Question: "What is <property> for the node identified by <id>?"
Cypher: MATCH (n:<Label>) WHERE n.id = <value> RETURN n.<property>

Question: "What is <rel_property> between <entity A> and <entity B>?"
Cypher: MATCH (a:<Label>)-[r:<REL_TYPE>]->(b:<Label>) WHERE a.<property> = <value> AND b.<property> = <value> RETURN r.<rel_property>

Question: "What are the top <n> <entities> ranked by <rel_property>?"
Cypher: MATCH (a)-[r:<REL_TYPE>]->(b) WHERE <filter> RETURN b.<property>, r.<rel_property> ORDER BY r.<rel_property> DESC LIMIT <n>

Now, based on the schema above, generate a single-line Cypher query for the following question. Output only the Cypher query on one line, nothing else."""

SYNTHESIS_PROMPT_HEAD = """You are a clinical data assistant. Answer the user's original question using only the evidence retrieved from a Neo4j knowledge graph.

Original question:
{question}

Sub-questions, Cypher queries, and retrieved results:
{evidence}

Guidelines:
- Base your answer strictly on the retrieved evidence above.
- If a sub-question returned an error or empty result, mention that it is unavailable.
- Be concise and direct. Do not fabricate data not present in the evidence.

Final answer:"""


# ─── Sub-question extraction (no category filtering — evaluate everything) ──

def extract_all_subquestions(output_text: str) -> list[dict]:
    results = []
    for m in re.finditer(r"Sub-question:\s*(.+?)\s*#\s*Type:\s*(searching|aggregation)", output_text):
        q, t = m.group(1).strip(), m.group(2).strip()
        results.append({"question": q, "type": t})
    return results


# ─── Generic SGLang batch text-completion helpers ───────────────────────────

def sglang_complete(prompt: str, url: str, max_tokens: int, stop=None, retries: int = 3) -> str:
    payload = {
        "model": "default",
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0,
        "top_p": 1.0,
        "repetition_penalty": 1.1,
    }
    if stop:
        payload["stop"] = stop
    for attempt in range(retries):
        try:
            resp = requests.post(f"{url}/v1/completions", json=payload, timeout=60)
            resp.raise_for_status()
            return resp.json()["choices"][0]["text"].strip()
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(2)
            else:
                print(f"    sglang_complete failed: {e}")
                return ""


def sglang_complete_batch(prompts: list[str], url: str, max_tokens: int, stop=None,
                           max_workers: int = 32, label: str = "") -> list[str]:
    results = [""] * len(prompts)
    if not prompts:
        return results

    def one(idx_p):
        idx, p = idx_p
        return idx, sglang_complete(p, url, max_tokens, stop=stop)

    t0 = time.time()
    done = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as ex:
        for idx, text in ex.map(one, enumerate(prompts)):
            results[idx] = text
            done += 1
            if done % 20 == 0 or done == len(prompts):
                elapsed = time.time() - t0
                speed = done / elapsed if elapsed else 0
                eta = (len(prompts) - done) / speed if speed else 0
                print(f"    [{label}] {done}/{len(prompts)}  {speed:.1f} q/s  ETA {eta:.0f}s")
    print(f"    [{label}] finished {len(prompts)} in {time.time()-t0:.1f}s")
    return results


# ─── Cypher prompt builders ──────────────────────────────────────────────────

def build_fewshot_cypher_prompt(question: str, upstream_rows=None) -> str:
    # This pipeline's Phase 2 doesn't thread real upstream rows between a
    # sample's searching steps (see few_shot_cypher.py's own
    # process_subquestion_group for that), so coreference sub-questions
    # ("those biomarkers") get an empty upstream_context here -- same as
    # this function's behavior before FEW_SHOT_TEMPLATE grew that slot.
    return build_cypher_prompt(question, upstream_rows=upstream_rows)


def build_schema_cypher_prompt(question: str, schema_prefix: str) -> str:
    return f'{schema_prefix}\n\nQuestion: "{question}"\nCypher:'


def build_grounded_cypher_prompt(question: str, schema_prefix: str, driver, vocab: dict,
                                  upstream_rows=None) -> str:
    """Method 2, retrieval-augmented: extract the entities this specific
    question mentions (patient id, biomarkers, algorithm families, training
    ratios) and run small live lookups against the graph for concrete
    matching facts (real property values, real relationship direction),
    then show those alongside the schema before asking for Cypher. This is
    what makes it actually retrieval-*augmented* generation rather than a
    single static schema description reused for every question.

    upstream_rows: the previous searching step's executed result rows
    within the same decomposition, if any. extract_entities only actually
    uses these when the question's own phrasing needs them (e.g. "those 2
    biomarkers", "that biomarker") -- passing them for a question that
    doesn't need them is harmless (they're simply ignored)."""
    entities = extract_entities(question, vocab, upstream_rows=upstream_rows)
    facts = retrieve_grounding_facts(driver, entities)
    if facts:
        facts_block = (
            "\n\nReal graph facts relevant to THIS question (retrieved just now — "
            "use them to get relationship direction and property values right; "
            "adapt the pattern to what's actually asked, don't just copy numbers). "
            "IMPORTANT: reuse these facts' *join structure* exactly — every variable "
            "that has an id/name filter above must keep that same filter in your query "
            "(don't drop a WHERE constraint when combining facts, e.g. when a question "
            "needs both a patient's prediction and a patient's local explanation, both "
            "matches need their own `.id = <patient>` filter, not just one of them), and "
            "a property shown attached to a relationship variable (e.g. `r.shapValue`) "
            "lives on that relationship, not on the nodes it connects — return it as "
            "`r.shapValue`, never `<node>.shapValue`. When a question asks for a SHAP "
            "value, always return shapValue, absShapValue, AND direction together (not "
            "just one or two of them), even if the question only explicitly names one. "
            "When a question asks about a model's prediction for a patient (probability, "
            "predicted label, or correctness), always return the predicted label, "
            "probability, AND correct together as one bundle, plus mr.modelFamily (the "
            "human-readable name, e.g. 'Decision Tree') rather than mr.modelType (the "
            "internal token, e.g. 'decisiontree') for the model name — even if the "
            "question only names one of the prediction fields. More generally: whenever a "
            "REAL_GRAPH_ROWS fact below shows several properties returned together for one "
            "entity (e.g. a rank alongside its importance score, or a label alongside its "
            "probability), return that same full set in your RETURN clause too — mirror the "
            "bundle actually shown in the facts, not just the one field the question's "
            "wording happens to name. Copy every source property's name with the EXACT "
            "casing shown on the schema card and in these facts (e.g. `predictedLabel`, "
            "never a guessed snake_case variant like `predicted_label`) — Neo4j property "
            "names are case-sensitive and a wrong-cased property silently returns null "
            "instead of erroring; the alias you give a returned value in `AS alias` can be "
            "worded however you like, but the source `node.property`/`rel.property` before "
            "`AS` must match the schema exactly. Only filter WHERE on fields the question "
            "actually constrains (patient id, model, ratio, biomarker name); never add an "
            "extra filter on an outcome field like `correct` unless correctness is "
            "explicitly what's being asked — an unrequested filter can silently zero out "
            "otherwise-correct results. Never copy a `$`-prefixed parameter name into your "
            "final query; any TEMPLATE_CYPHER shown below is illustrative only and its "
            "$names are not real Cypher — substitute literal values instead:\n  "
            + "\n  ".join(facts)
        )
    else:
        facts_block = ""
    return f'{schema_prefix}{facts_block}\n\nQuestion: "{question}"\nCypher:'


def clean_cypher(raw: str) -> str:
    text = raw.strip()
    if text.lower().startswith("cypher:"):
        text = text[len("cypher:"):].strip()
    return text.split("\n")[0].strip()


def build_repair_prompt(original_prompt: str, prev_cypher: str, exec_result: dict) -> str:
    """Append a failed attempt + its outcome to the original prompt and ask
    for a corrected query. Reuses the same schema/few-shot context as the
    first attempt, so the repair step is still grounded in it, not a blind
    retry."""
    if not exec_result.get("success"):
        issue = f"failed to execute with this error: {exec_result.get('error')}"
    elif exec_result.get("rows", 0) > 0:
        issue = (f"executed and returned {exec_result['rows']} row(s), but every returned "
                 "value was null — the query pattern matched something, but the wrong "
                 "something: check (a) that every variable with an id/name filter in the "
                 "question is actually constrained by a WHERE clause (an unconstrained "
                 "variable joined via a comma/cartesian product can bind to any node), and "
                 "(b) that RETURN reads each property off the correct variable — a property "
                 "shown on a relationship in the grounding facts (e.g. `r.shapValue`) is not "
                 "also a property of the nodes it connects")
    else:
        issue = ("executed without error but returned no results — the labels, "
                 "relationship types/directions, or property values likely don't "
                 "match this specific question")
    return (
        f"{original_prompt} {prev_cypher}\n\n"
        f"That query {issue}. Look at the schema again — labels, relationship "
        f"directions, and property values — and write a corrected single-line "
        f"Cypher query for the same question. Output only the corrected Cypher "
        f"query, nothing else.\n\nCorrected Cypher:"
    )


# ─── Neo4j execution (keeps row data, needed as answer-synthesis evidence) ──

def _has_any_value(v) -> bool:
    """True if v (a record leaf/dict/list from neo4j's .data()) contains at
    least one non-null scalar anywhere inside it. A query can execute
    successfully and return N rows that are all {prop: None, ...} — e.g. a
    dropped join filter that lets an unrelated node bind to the pattern, or
    a property read off the wrong variable (relationship vs. node) — and a
    plain `len(records) > 0` check treats that as a valid result. It isn't:
    it's a wrong-but-non-crashing query, and it should be repaired like one."""
    if isinstance(v, dict):
        return any(_has_any_value(x) for x in v.values())
    if isinstance(v, (list, tuple)):
        return any(_has_any_value(x) for x in v)
    return v is not None


def execute_cypher_with_data(driver, cypher: str, row_limit: int = ROW_LIMIT) -> dict:
    if not cypher:
        return {"success": False, "has_results": False, "error": "empty cypher", "rows": 0, "rows_data": []}
    try:
        with driver.session() as session:
            result = session.run(cypher)
            records = [r.data() for r in result]
            has_signal = any(_has_any_value(r) for r in records)
            return {
                "success": True,
                "has_results": len(records) > 0 and has_signal,
                "error": None,
                "rows": len(records),
                "rows_data": records[:row_limit],
            }
    except Exception as e:
        return {"success": False, "has_results": False, "error": str(e), "rows": 0, "rows_data": []}


def build_evidence_text(subqs_with_exec: list[dict]) -> str:
    lines = []
    for i, sq in enumerate(subqs_with_exec, 1):
        if sq["type"] == "searching":
            exec_result = sq.get("exec", {})
            if exec_result.get("success") and exec_result.get("rows_data"):
                result_str = json.dumps(exec_result["rows_data"], ensure_ascii=False, default=str)
            elif exec_result.get("success"):
                result_str = "(empty)"
            else:
                result_str = f"ERROR: {exec_result.get('error')}"
            lines.append(
                f"[{i}] Sub-question: {sq['question']}\n"
                f"    Cypher: {sq.get('cypher', '')}\n"
                f"    Result: {result_str}"
            )
        else:
            lines.append(f"[{i}] Sub-question (aggregation, reason over the above evidence): {sq['question']}")
    return "\n\n".join(lines) if lines else "(no evidence)"


# ─── Per-method pipeline ─────────────────────────────────────────────────────

def run_method(method_name, cypher_prompt_fn, base_samples, sglang_url, driver,
                max_new_tokens_cypher, max_new_tokens_answer, execute, repair_retries=0):
    """Run one full Cypher-generation + execution + answer-synthesis pass.

    If repair_retries > 0 and execute is True: after the first attempt, any
    sub-question whose query errored or returned no results is retried up to
    repair_retries times, feeding the failed query + its outcome back into
    the same prompt context and asking for a correction (self-healing).
    """
    samples = [
        {"index": s["index"], "input": s["input"], "output": s["output"],
         "subqs": [dict(sq) for sq in s["subqs"]]}
        for s in base_samples
    ]

    # Flat list of (sample_idx, subq_idx) for searching sub-questions
    searching_refs = []
    for si, s in enumerate(samples):
        for qi, sq in enumerate(s["subqs"]):
            if sq["type"] == "searching":
                searching_refs.append((si, qi))

    print(f"\n{'='*70}\n  [{method_name}] Cypher generation ({len(searching_refs)} searching sub-questions)\n{'='*70}")
    orig_prompts = [cypher_prompt_fn(samples[si]["subqs"][qi]["question"]) for si, qi in searching_refs]
    raw_cypher = sglang_complete_batch(
        orig_prompts, sglang_url, max_new_tokens_cypher, stop=["\n"], label=f"{method_name}-cypher")

    for (si, qi), raw in zip(searching_refs, raw_cypher):
        sq = samples[si]["subqs"][qi]
        sq["cypher"] = clean_cypher(raw)
        sq["repair_attempts"] = 0

    # Execute
    if execute:
        print(f"\n  [{method_name}] Executing {len(searching_refs)} Cypher queries against Neo4j...")
        for si, qi in searching_refs:
            sq = samples[si]["subqs"][qi]
            sq["exec"] = execute_cypher_with_data(driver, sq["cypher"])
    else:
        for si, qi in searching_refs:
            samples[si]["subqs"][qi]["exec"] = {"success": None, "has_results": None, "error": None, "rows": 0, "rows_data": []}

    # Repair loop: retry sub-questions that errored or came back empty
    if execute and repair_retries > 0:
        for round_i in range(repair_retries):
            needs_repair = [
                (si, qi) for si, qi in searching_refs
                if not samples[si]["subqs"][qi]["exec"].get("has_results")
            ]
            if not needs_repair:
                break
            print(f"\n  [{method_name}] Repair round {round_i+1}: retrying {len(needs_repair)} failed/empty queries...")
            idx_map = {ref: i for i, ref in enumerate(searching_refs)}
            repair_prompts = [
                build_repair_prompt(
                    orig_prompts[idx_map[(si, qi)]],
                    samples[si]["subqs"][qi]["cypher"],
                    samples[si]["subqs"][qi]["exec"],
                )
                for si, qi in needs_repair
            ]
            repaired_raw = sglang_complete_batch(
                repair_prompts, sglang_url, max_new_tokens_cypher, stop=["\n"],
                label=f"{method_name}-repair{round_i+1}")

            improved = 0
            for (si, qi), raw in zip(needs_repair, repaired_raw):
                sq = samples[si]["subqs"][qi]
                new_cypher = clean_cypher(raw)
                new_exec = execute_cypher_with_data(driver, new_cypher)
                sq["repair_attempts"] += 1
                # Keep the repaired version if it's at least as good (prefer
                # any non-empty result; otherwise prefer a query that at
                # least executes over one that errors).
                old_exec = sq["exec"]
                better = (new_exec.get("has_results") and not old_exec.get("has_results")) or \
                         (new_exec.get("success") and not old_exec.get("success"))
                if better:
                    sq["cypher"] = new_cypher
                    sq["exec"] = new_exec
                    improved += 1
            print(f"  [{method_name}] Repair round {round_i+1}: {improved}/{len(needs_repair)} improved")

    # Answer synthesis
    print(f"\n{'='*70}\n  [{method_name}] Answer synthesis ({len(samples)} samples)\n{'='*70}")
    answer_prompts = []
    for s in samples:
        evidence = build_evidence_text(s["subqs"])
        answer_prompts.append(SYNTHESIS_PROMPT_HEAD.format(question=s["input"], evidence=evidence))
    answers = sglang_complete_batch(
        answer_prompts, sglang_url, max_new_tokens_answer,
        stop=["\nOriginal question:", "\n\nQuestion:"], label=f"{method_name}-answer")
    for s, a in zip(samples, answers):
        s["final_answer"] = a.strip()

    return samples, searching_refs


def compute_method_metrics(samples, searching_refs, execute):
    total = len(searching_refs)
    if not execute or total == 0:
        return {"total_searching": total, "cypher_generated": sum(
            1 for si, qi in searching_refs if samples[si]["subqs"][qi]["cypher"])}

    exec_ok = 0
    has_res = 0
    for si, qi in searching_refs:
        e = samples[si]["subqs"][qi]["exec"]
        if e.get("success"):
            exec_ok += 1
        if e.get("has_results"):
            has_res += 1

    # Per-sample full coverage: every searching sub-question executed AND returned results
    full_cov = 0
    for s in samples:
        sq_searching = [sq for sq in s["subqs"] if sq["type"] == "searching"]
        if sq_searching and all(sq["exec"].get("has_results") for sq in sq_searching):
            full_cov += 1

    return {
        "total_samples": len(samples),
        "total_searching": total,
        "exec_success": exec_ok,
        "exec_rate": round(exec_ok / total, 4) if total else 0,
        "has_results": has_res,
        "valid_rate": round(has_res / total, 4) if total else 0,
        "full_coverage_samples": full_cov,
        "full_coverage_rate": round(full_cov / len(samples), 4) if samples else 0,
    }


# ─── Main ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Compare few-shot vs schema-grounded Cypher generation")
    parser.add_argument("--model", type=str, default=MODEL_PATH)
    parser.add_argument("--test_file", type=str, default=TEST_FILE)
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--limit", type=int, default=None, help="only process first N samples (smoke test)")
    parser.add_argument("--max_new_tokens_cypher", type=int, default=256,
                        help="Method 2's schema-composed queries run longer than Method 1's "
                             "memorized one-liners; 128 truncated some of them mid-clause.")
    parser.add_argument("--max_new_tokens_answer", type=int, default=220)
    parser.add_argument("--no_execute", action="store_true", help="skip Neo4j execution, Cypher generation only")
    parser.add_argument("--repair_retries", type=int, default=1,
                        help="Method 2 only: retry failed/empty queries this many times, "
                             "feeding the error/empty-result back to the model (0 disables).")
    parser.add_argument("--neo4j_uri", type=str, default=NEO4J_URI)
    parser.add_argument("--neo4j_user", type=str, default=NEO4J_USER)
    parser.add_argument("--neo4j_password", type=str, default=NEO4J_PASSWORD)
    parser.add_argument("--port", type=int, default=SGLANG_PORT)
    parser.add_argument("--sglang_url", type=str, default=None, help="use an already-running SGLang server")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    execute = not args.no_execute

    # ── Load test set ────────────────────────────────────────────────────
    with open(args.test_file) as f:
        raw_data = json.load(f)
    if args.limit:
        raw_data = raw_data[:args.limit]
    print(f"Loaded {len(raw_data)} samples from {args.test_file}")

    base_samples = []
    for i, item in enumerate(raw_data):
        base_samples.append({
            "index": i,
            "input": item["input"],
            "output": item["output"],
            "subqs": extract_all_subquestions(item["output"]),
        })
    n_searching = sum(1 for s in base_samples for sq in s["subqs"] if sq["type"] == "searching")
    n_agg = sum(1 for s in base_samples for sq in s["subqs"] if sq["type"] == "aggregation")
    print(f"Extracted {n_searching} searching + {n_agg} aggregation sub-questions")

    # ── Connect Neo4j (needed for both execution and the schema card) ────
    driver = GraphDatabase.driver(args.neo4j_uri, auth=(args.neo4j_user, args.neo4j_password))
    driver.verify_connectivity()
    print(f"Neo4j connected: {args.neo4j_uri}")

    print("\nBuilding live schema card for Method 2...")
    schema_card = build_schema_card(driver)
    schema_prefix = SCHEMA_PROMPT_HEAD_TEMPLATE.format(schema_card=schema_card)
    print(f"Schema card: {len(schema_card)} chars")
    with open(os.path.join(args.output_dir, "schema_card_used.txt"), "w") as f:
        f.write(schema_card)

    print("Building entity-matching vocabulary for Method 2's retrieval step...")
    vocab = build_vocab(driver)

    # ── Launch SGLang for Qwen3-8B-Base ───────────────────────────────────
    sglang_process = None
    if args.sglang_url:
        server_url = args.sglang_url
        resp = requests.get(f"{server_url}/health", timeout=5)
        resp.raise_for_status()
        print(f"Using existing SGLang server at {server_url}")
    else:
        server_url = f"http://localhost:{args.port}"
        print(f"\nLaunching SGLang server for {args.model} ...")
        sglang_process = launch_sglang_server(args.model, args.port)
        if sglang_process is None:
            print("ERROR: failed to launch SGLang server. Exiting.")
            driver.close()
            return

    try:
        # ── Method 1: fixed few-shot examples ────────────────────────────
        m1_samples, m1_refs = run_method(
            "method1_fewshot", build_fewshot_cypher_prompt, base_samples,
            server_url, driver, args.max_new_tokens_cypher, args.max_new_tokens_answer, execute)
        m1_metrics = compute_method_metrics(m1_samples, m1_refs, execute)

        # ── Method 2: schema-grounded prompting ──────────────────────────
        m2_samples, m2_refs = run_method(
            "method2_schema", lambda q: build_grounded_cypher_prompt(q, schema_prefix, driver, vocab), base_samples,
            server_url, driver, args.max_new_tokens_cypher, args.max_new_tokens_answer, execute,
            repair_retries=args.repair_retries)
        m2_metrics = compute_method_metrics(m2_samples, m2_refs, execute)
        m2_metrics["repair_retries_enabled"] = args.repair_retries
        m2_metrics["repaired_count"] = sum(
            1 for si, qi in m2_refs if m2_samples[si]["subqs"][qi].get("repair_attempts", 0) > 0
            and m2_samples[si]["subqs"][qi]["exec"].get("has_results"))
    finally:
        if sglang_process:
            shutdown_sglang_server(sglang_process)
        driver.close()

    # ── Save predictions ───────────────────────────────────────────────
    m1_out = os.path.join(args.output_dir, "method1_fewshot_predictions.json")
    m2_out = os.path.join(args.output_dir, "method2_schema_predictions.json")
    with open(m1_out, "w") as f:
        json.dump({"metrics": m1_metrics, "samples": m1_samples}, f, indent=2, ensure_ascii=False)
    with open(m2_out, "w") as f:
        json.dump({"metrics": m2_metrics, "samples": m2_samples}, f, indent=2, ensure_ascii=False)
    print(f"\nSaved: {m1_out}")
    print(f"Saved: {m2_out}")

    # ── Comparison report ────────────────────────────────────────────────
    report = {"method1_fewshot": m1_metrics, "method2_schema": m2_metrics}
    report_json = os.path.join(args.output_dir, "method_comparison_report.json")
    with open(report_json, "w") as f:
        json.dump(report, f, indent=2)

    report_md = os.path.join(args.output_dir, "method_comparison_report.md")
    with open(report_md, "w") as f:
        f.write("# Cypher Generation Method Comparison\n\n")
        f.write(f"Model: `{args.model}`  |  Test set: `{args.test_file}` ({len(base_samples)} samples)\n\n")
        f.write("| Metric | Method 1 (fewshot) | Method 2 (schema) |\n")
        f.write("|---|---|---|\n")
        keys = ["total_searching", "exec_success", "exec_rate", "has_results", "valid_rate",
                "full_coverage_samples", "full_coverage_rate", "repaired_count"]
        for k in keys:
            v1 = m1_metrics.get(k, "-")
            v2 = m2_metrics.get(k, "-")
            f.write(f"| {k} | {v1} | {v2} |\n")
        f.write("\n## Sample-level spot check (first 10)\n\n")
        for i in range(min(10, len(m1_samples))):
            f.write(f"### [{i}] {m1_samples[i]['input']}\n\n")
            f.write(f"**Method 1 answer:** {m1_samples[i]['final_answer']}\n\n")
            f.write(f"**Method 2 answer:** {m2_samples[i]['final_answer']}\n\n")
    print(f"Saved: {report_json}")
    print(f"Saved: {report_md}")

    print(f"\n{'='*70}\n  Summary\n{'='*70}")
    print(f"  {'Metric':<25}{'Method1 (fewshot)':<22}{'Method2 (schema)':<22}")
    for k in ["total_searching", "exec_rate", "valid_rate", "full_coverage_rate"]:
        print(f"  {k:<25}{str(m1_metrics.get(k, '-')):<22}{str(m2_metrics.get(k, '-')):<22}")


if __name__ == "__main__":
    main()
