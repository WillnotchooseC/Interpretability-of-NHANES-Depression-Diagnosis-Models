"""
Full pipeline evaluation: fine-tuned decomposition model -> Cypher generation
(both methods) -> gold-result comparison.

  Step 1  decomp_v1 (LoRA-tuned Qwen3-8B-Base, merged) generates a Step/
          Sub-question/Type decomposition from each `input` in
          data/test_dataset.json — NOT the reference `output`. This tests the
          fine-tuned model's real decomposition capability end-to-end.
  Step 2  For samples where the model's searching-sub-question COUNT matches
          the reference's (so positional alignment to the gold file is
          trustworthy), each searching sub-question is turned into Cypher by
          the SAME two methods compared earlier, run by base Qwen3-8B-Base:
            method1_fewshot — fixed few-shot examples
            method2_schema  — schema card + live entity-grounded retrieval
  Step 3  Execute each generated Cypher and compare the returned rows against
          test_dataset_gold_cypher_executable_goldresult_aligned.yaml's
          gold_result for that step, via column-name-agnostic value-set
          containment (different Cypher phrasing can legally return
          differently-named/ordered columns for the same underlying facts).

Usage:
    python run_finetuned_pipeline_eval.py
    python run_finetuned_pipeline_eval.py --limit 10
"""

import argparse
import concurrent.futures
import json
import os
import re
import subprocess
import time

import requests
import yaml
from neo4j import GraphDatabase

from few_shot_cypher import (
    FEW_SHOT_TEMPLATE, launch_sglang_server, shutdown_sglang_server, SGLANG_PORT,
    needs_upstream_context,
)
from graph_schema import build_schema_card
from entity_retrieval import build_vocab, extract_entities, retrieve_grounding_facts
from run_cypher_qa_comparison import (
    extract_all_subquestions, sglang_complete, sglang_complete_batch, build_fewshot_cypher_prompt,
    build_grounded_cypher_prompt, clean_cypher, execute_cypher_with_data,
    SCHEMA_PROMPT_HEAD_TEMPLATE, ROW_LIMIT,
)

DECOMP_MODEL_PATH = "./nhanes_model/decomp_v1/merged_model"
BASE_MODEL_PATH = "./nhanes_model/base_model_cache/Qwen3-8B-Base"
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "12345678")
TEST_FILE = "data/test_dataset.json"
GOLD_YAML = "test_dataset_gold_cypher_executable_goldresult_aligned.yaml"
OUTPUT_DIR = "cypher_results"

DECOMP_SYSTEM_PROMPT = (
    "You are a question decomposition assistant. Given a complex question, "
    "break it down into a sequence of simpler sub-questions that can be answered independently. "
    "For each step, provide a clear sub-question and label its type as either 'searching' "
    "(retrieving specific information) or 'aggregation' (combining results from previous steps)."
)


# ─── Truncation (mirrors few_shot_cypher.truncate_output) ───────────────────

def truncate_decomposition(text: str) -> str:
    garbage_patterns = ["�", "<|im_start|>", "\x00"]
    cut_pos = len(text)
    for pat in garbage_patterns:
        idx = text.find(pat)
        if 0 <= idx < cut_pos:
            cut_pos = idx
    user_turn = re.search(r"\nuser\n|\nassistant\n", text)
    if user_turn and user_turn.start() < cut_pos:
        cut_pos = user_turn.start()
    clean_text = text[:cut_pos]
    lines = clean_text.split("\n")

    # Under sampling (temperature > 0) the model sometimes doesn't stop after
    # a complete decomposition and loops into a second, near-duplicate (or
    # outright different) attempt starting again from "Step 1:". Cut at the
    # first such restart so the selected text is always the model's *first*
    # complete answer, never a looped re-attempt or hallucinated continuation
    # — greedy decoding (temperature=0) never triggers this, which is why it
    # went unnoticed until sampling was tried.
    loop_cut_idx = len(lines)
    last_step_num = 0
    for i, line in enumerate(lines):
        m = re.match(r"\s*Step\s+(\d+):", line)
        if m:
            step_num = int(m.group(1))
            if step_num <= last_step_num:
                loop_cut_idx = i
                break
            last_step_num = step_num
    lines = lines[:loop_cut_idx]

    last_valid_idx = -1
    for i, line in enumerate(lines):
        if re.search(r"#\s*Type:\s*(searching|aggregation)", line.strip()):
            last_valid_idx = i
    if last_valid_idx >= 0:
        return "\n".join(lines[: last_valid_idx + 1]).strip()
    return "\n".join(lines).strip()


# ─── SGLang chat-completion batch (for the decomposition model) ─────────────

def sglang_chat_complete(question: str, url: str, max_tokens: int, retries: int = 3) -> str:
    payload = {
        "model": "default",
        "messages": [
            {"role": "system", "content": DECOMP_SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ],
        "max_tokens": max_tokens,
        "temperature": 0,
        "top_p": 1.0,
        "stop": ["�", "\nStep by step:", "\nuser", "\nassistant", "<|im_start|>"],
        "chat_template_kwargs": {"enable_thinking": False},
    }
    for attempt in range(retries):
        try:
            resp = requests.post(f"{url}/v1/chat/completions", json=payload, timeout=60)
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"]
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(2)
            else:
                print(f"    sglang_chat_complete failed: {e}")
                return ""


def sglang_chat_complete_batch(questions, url, max_tokens, max_workers=16, label=""):
    results = [""] * len(questions)

    def one(idx_q):
        idx, q = idx_q
        return idx, sglang_chat_complete(q, url, max_tokens)

    t0 = time.time()
    done = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as ex:
        for idx, text in ex.map(one, enumerate(questions)):
            results[idx] = text
            done += 1
            if done % 20 == 0 or done == len(questions):
                print(f"    [{label}] {done}/{len(questions)}  ({time.time()-t0:.0f}s elapsed)")
    return results


# ─── Hard SGLang teardown ────────────────────────────────────────────────────
# few_shot_cypher.shutdown_sglang_server only signals the single tracked
# `launch_sglang_server` subprocess; SGLang's own scheduler/detokenizer
# workers are separate processes that don't reliably die with it (observed
# repeatedly today — they survive as orphans holding GPU memory). Launching
# a second SGLang server on the same GPU right after, without confirming
# those are actually gone, risks the second server inheriting a contended /
# corrupted CUDA state — the likely cause of a mid-run hang (0% GPU util,
# unresponsive health endpoint) that a single-server run never hit.

def hard_shutdown_sglang(process, port=SGLANG_PORT, wait_s=20):
    # NOTE: deliberately not os.killpg here — in this environment a
    # subprocess.Popen child shares the *caller's* process group (verified:
    # child pgid == our own pgid), so killing the group would kill this
    # script too. Stick to killing the tracked PID plus stray workers by
    # command-line pattern (safe — targets other processes, not our group).
    if process and process.poll() is None:
        process.kill()
        try:
            process.wait(timeout=10)
        except Exception:
            pass
    # Belt-and-suspenders: kill any stray sglang worker processes by name,
    # since scheduler/detokenizer don't reliably die with the launcher.
    subprocess.run(["pkill", "-9", "-f", "sglang.launch_server"], check=False)
    subprocess.run(["pkill", "-9", "-f", "sglang::scheduler"], check=False)
    subprocess.run(["pkill", "-9", "-f", "sglang::detokenizer"], check=False)

    # Confirm GPU memory actually comes back before letting the caller
    # launch the next server, instead of trusting the kill happened cleanly.
    t0 = time.time()
    while time.time() - t0 < wait_s:
        try:
            out = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                text=True,
            )
            if int(out.strip().splitlines()[0]) < 500:
                print(f"  GPU memory confirmed free ({time.time()-t0:.0f}s after kill).")
                return
        except Exception:
            pass
        time.sleep(1)
    print(f"  WARNING: GPU memory still not free {wait_s}s after shutdown attempt — proceeding anyway.")


# ─── Gold comparison: column-name-agnostic value-set containment ────────────

def _norm_val(v):
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return round(float(v), 4)
    if isinstance(v, str):
        return v.strip().lower()
    return v


def _flatten_leaves(v):
    """Yield hashable leaf values from a (possibly nested) Neo4j result
    value — a Cypher query can legally RETURN a whole node/map (dict) or a
    list, not just scalar properties, e.g. `RETURN p` or `RETURN p{.*}`."""
    if isinstance(v, dict):
        for vv in v.values():
            yield from _flatten_leaves(vv)
    elif isinstance(v, (list, tuple)):
        for vv in v:
            yield from _flatten_leaves(vv)
    elif v is not None:
        yield v


def _row_vals(row: dict) -> set:
    return {_norm_val(leaf) for v in row.values() for leaf in _flatten_leaves(v)}


FAMILY_DISPLAY = {
    "naivebayes": "naive bayes", "decisiontree": "decision tree",
    "logisticregression": "logistic regression", "randomforest": "random forest",
}


def build_echo_values(question: str, vocab: dict) -> set:
    """Values already given in the question itself (model family, ratio,
    biomarker name) are not "retrieved" facts — a correct Cypher answer is
    not obligated to redundantly echo them back as extra output columns.
    Gold rows often include them anyway (for readability), so they must be
    excluded before scoring or a perfectly correct answer that only returns
    the actually-asked-for value gets penalized for "missing" fields it was
    never asked to return."""
    entities = extract_entities(question, vocab)
    echoes = set()
    for fam in entities["model_types"]:
        echoes.add(_norm_val(FAMILY_DISPLAY.get(fam, fam)))
        echoes.add(_norm_val(fam))
    for r in entities["ratios"]:
        # entity_retrieval.extract_entities already returns "1:N"-formatted ratios.
        echoes.add(_norm_val(r))
    for display in entities["biomarkers"]:
        echoes.add(_norm_val(display))
    if entities["patient_ids"]:
        for pid in entities["patient_ids"]:
            echoes.add(_norm_val(float(pid)))
    return echoes


def score_against_gold(actual_rows: list, gold_rows: list, echo_values: set = None) -> dict:
    """Greedy row-containment match, robust to different column names/order
    and to extra columns in actual_rows. Filter-echo values (see
    build_echo_values) are excluded from both sides before comparing, so
    only the actually-retrieved facts are scored. Returns per-row coverage
    plus a strict (100% field overlap) and lenient (>=50%) match tally."""
    echo_values = echo_values or set()
    if not gold_rows:
        return {"coverage": None, "full_matches": None, "partial_matches": None, "n_gold_rows": 0}

    remaining = [(_row_vals(r) - echo_values, r) for r in actual_rows]
    coverages = []
    for gold_row in gold_rows:
        gvals = _row_vals(gold_row) - echo_values
        if not gvals:
            continue  # every field in this gold row was just a filter echo
        best_idx, best_score = None, -1.0
        for i, (avals, _raw) in enumerate(remaining):
            if not avals:
                continue
            score = len(gvals & avals) / len(gvals)
            if score > best_score:
                best_score, best_idx = score, i
        coverages.append(best_score if best_idx is not None else 0.0)
        if best_idx is not None and best_score > 0:
            del remaining[best_idx]

    full = sum(1 for c in coverages if c >= 0.999)
    partial = sum(1 for c in coverages if c >= 0.5)
    return {
        "coverage": round(sum(coverages) / len(coverages), 4) if coverages else None,
        "full_matches": full,
        "partial_matches": partial,
        "n_gold_rows": len(coverages),
    }


# ─── Per-sample sequential generation (threads coreference across steps) ────
# The flat cross-sample batch this replaced generated every searching step's
# Cypher independently, so a step phrased as "those top 2 biomarkers" (its
# ranking done by an *earlier* step in the same decomposition) always saw an
# empty upstream context and queried with no biomarker filter at all --
# returning whichever 2 biomarkers the database happened to return first,
# not the ones actually selected upstream. Samples are still processed
# concurrently for throughput; only steps *within* one sample are serialized
# (step N's prompt needs step N-1's executed rows).

def _repair_step(cy, ex, prompt, server_url, max_new_tokens, repair_retries, label):
    for _ in range(repair_retries):
        if ex.get("has_results"):
            break
        if not ex.get("success"):
            issue = f"failed to execute with this error: {ex.get('error')}"
        elif ex.get("rows", 0) > 0:
            issue = (f"executed and returned {ex['rows']} row(s), but every "
                     "returned value was null — check that every variable with an "
                     "id/name filter is actually constrained by WHERE (an "
                     "unconstrained variable joined via a comma/cartesian product "
                     "can bind to any node), and that RETURN reads each property "
                     "off the correct variable — a property shown on a relationship "
                     "in the grounding facts (e.g. `r.shapValue`) is not also a "
                     "property of the nodes it connects")
        else:
            issue = ("executed without error but returned no results — the labels, "
                     "relationship types/directions, or property values likely don't match")
        rep_prompt = (f"{prompt} {cy}\n\nThat query {issue}. Look at the schema again "
                       f"and write a corrected single-line Cypher query. Output only "
                       f"the corrected Cypher query.\n\nCorrected Cypher:")
        rep_raw = sglang_complete(rep_prompt, server_url, max_new_tokens, stop=["\n"])
        new_cy = clean_cypher(rep_raw)
        new_ex = execute_cypher_with_data(_repair_step.driver, new_cy)
        if new_ex.get("has_results") or (new_ex.get("success") and not ex.get("success")):
            cy, ex = new_cy, new_ex
    return cy, ex


def process_method_sequential(sample_groups, prompt_fn, driver, server_url,
                               max_new_tokens, repair_retries, max_workers, label):
    """sample_groups: list of (si, [(qi, question), ...]) for alignable samples.
    prompt_fn(question, upstream_rows) -> prompt string.
    Returns {(si, qi): (cypher, exec_result)}."""
    _repair_step.driver = driver  # small helper needs driver without threading it through every call
    results = {}

    def process_one_sample(si_group):
        si, qas = si_group
        local = {}
        upstream_rows = []
        for qi, q in qas:
            ctx = upstream_rows if needs_upstream_context(q) else []
            prompt = prompt_fn(q, ctx)
            raw = sglang_complete(prompt, server_url, max_new_tokens, stop=["\n"])
            cy = clean_cypher(raw)
            ex = execute_cypher_with_data(driver, cy)
            cy, ex = _repair_step(cy, ex, prompt, server_url, max_new_tokens, repair_retries, label)
            local[qi] = (cy, ex)
            if ex.get("rows_data"):
                upstream_rows = ex["rows_data"]
        return si, local

    t0 = time.time()
    done = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        for si, local in executor.map(process_one_sample, sample_groups):
            for qi, (cy, ex) in local.items():
                results[(si, qi)] = (cy, ex)
            done += 1
            if done % 10 == 0 or done == len(sample_groups):
                elapsed = time.time() - t0
                print(f"    [{label}] {done}/{len(sample_groups)} samples  ({elapsed:.0f}s elapsed)")
    print(f"    [{label}] finished {len(sample_groups)} samples in {time.time()-t0:.1f}s")
    return results


# ─── Main ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--decomp_model", type=str, default=DECOMP_MODEL_PATH)
    parser.add_argument("--base_model", type=str, default=BASE_MODEL_PATH)
    parser.add_argument("--max_new_tokens_decomp", type=int, default=320)
    parser.add_argument("--max_new_tokens_cypher", type=int, default=256)
    parser.add_argument("--repair_retries", type=int, default=1)
    parser.add_argument("--max_workers", type=int, default=24,
                        help="Concurrent SGLang requests for method1 (fewshot) Cypher gen/repair — "
                             "a short, fixed, fully-cacheable prompt; never implicated in a hang.")
    parser.add_argument("--max_workers_schema", type=int, default=3,
                        help="Concurrent SGLang requests for method2 (schema+retrieval) Cypher "
                             "gen/repair. Its prompts are much longer with a per-question-unique "
                             "suffix (only the schema portion is a shared cacheable prefix); every "
                             "hang seen so far — across different server-lifecycle and repair "
                             "configurations — correlated specifically with this method, so it "
                             "gets a much lower ceiling than --max_workers.")
    parser.add_argument("--output_dir", type=str, default=OUTPUT_DIR)
    parser.add_argument("--port", type=int, default=SGLANG_PORT)
    parser.add_argument("--decomp_cache", type=str, default=None,
                        help="Path to save/load decomp_v1's raw outputs (default: "
                             "<output_dir>/decomp_v1_raw_outputs.json). If it already exists "
                             "and matches this run's test_ids, Phase 1 is skipped entirely — no "
                             "SGLang launch, no regenerating decomp_v1's (deterministic) output. "
                             "This decouples decomposition generation from Cypher generation, so "
                             "a hang/crash while iterating on Cypher-generation doesn't force "
                             "redoing an unrelated, already-correct, already-finished step.")
    parser.add_argument("--regenerate_decomp", action="store_true",
                        help="Ignore --decomp_cache and regenerate decomp_v1's output from scratch.")
    parser.add_argument("--skip_phase2", action="store_true",
                        help="Stop after Phase 1 (decomposition + caching), skip Cypher generation.")
    parser.add_argument("--regenerate_cypher", action="store_true",
                        help="Ignore <output_dir>/method{1,2}_..._cypher_cache.json and regenerate "
                             "both methods' Cypher from scratch. Per-method caching means a method "
                             "that already finished successfully is skipped on rerun by default — "
                             "e.g. if method2 hangs, restarting doesn't redo method1's work too.")
    parser.add_argument("--sglang_extra_args", type=str, nargs="*", default=None,
                        help="Extra CLI flags passed to sglang.launch_server, e.g. "
                             "--sglang_extra_args --disable-overlap-schedule --watchdog-timeout 120 "
                             "to trade the overlap scheduler's speed for robustness against the "
                             "GPU-idle/unresponsive hangs seen under sustained concurrent load.")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    decomp_cache_path = args.decomp_cache or os.path.join(args.output_dir, "decomp_v1_raw_outputs.json")

    # ── Load test set + gold reference ───────────────────────────────────
    with open(TEST_FILE) as f:
        raw_data = json.load(f)
    if args.limit:
        raw_data = raw_data[: args.limit]
    gold_data = yaml.safe_load(open(GOLD_YAML))
    gold_by_id = {g["id"]: g for g in gold_data}
    test_ids = [f"test_{i+1:03d}" for i in range(len(raw_data))]
    print(f"Loaded {len(raw_data)} samples, {len(gold_data)} gold entries")

    # ═══ PHASE 1: decomp_v1 generates decomposition from raw `input` ═══════
    print(f"\n{'='*70}\n  PHASE 1: decomp_v1 decomposition generation\n{'='*70}")
    server_url = f"http://localhost:{args.port}"

    cached = None
    if not args.regenerate_decomp and os.path.exists(decomp_cache_path):
        with open(decomp_cache_path) as f:
            cached = json.load(f)
        if cached.get("test_ids") != test_ids:
            print(f"  Cache at {decomp_cache_path} doesn't match this run's samples — regenerating.")
            cached = None

    if cached:
        print(f"  Loaded cached decomp_v1 output from {decomp_cache_path} "
              f"({len(cached['raw_outputs'])} samples) — Phase 1 SGLang launch skipped.")
        raw_outputs = cached["raw_outputs"]
    else:
        print(f"Launching SGLang for {args.decomp_model} ...")
        proc = launch_sglang_server(args.decomp_model, args.port, extra_args=args.sglang_extra_args)
        if proc is None:
            print("ERROR: failed to launch decomp_v1 SGLang server.")
            return

        try:
            inputs = [item["input"] for item in raw_data]
            raw_outputs = sglang_chat_complete_batch(
                inputs, server_url, args.max_new_tokens_decomp, label="decomp")
        finally:
            print("Tearing down decomp_v1 SGLang server before launching the base-model one...")
            hard_shutdown_sglang(proc, args.port)

        with open(decomp_cache_path, "w") as f:
            json.dump({"test_ids": test_ids, "raw_outputs": raw_outputs}, f, ensure_ascii=False)
        print(f"  Saved decomp_v1 raw outputs to {decomp_cache_path} for reuse.")

    samples = []
    for i, (item, raw_out, tid) in enumerate(zip(raw_data, raw_outputs, test_ids)):
        model_output = truncate_decomposition(raw_out)
        model_subqs = extract_all_subquestions(model_output)
        ref_subqs = extract_all_subquestions(item["output"])
        model_searching = [sq for sq in model_subqs if sq["type"] == "searching"]
        ref_searching = [sq for sq in ref_subqs if sq["type"] == "searching"]
        gold_entry = gold_by_id.get(tid)
        gold_queries = gold_entry["queries"] if gold_entry else []
        alignable = gold_entry is not None and len(model_searching) == len(gold_queries)
        samples.append({
            "test_id": tid, "index": i, "input": item["input"],
            "reference_output": item["output"], "model_raw_output": raw_out,
            "model_output": model_output, "model_subqs": model_subqs,
            "model_n_searching": len(model_searching), "ref_n_searching": len(ref_searching),
            "gold_n_queries": len(gold_queries), "alignable": alignable,
            "gold_queries": gold_queries if alignable else [],
        })

    n_alignable = sum(1 for s in samples if s["alignable"])
    print(f"\nDecomposition fidelity: {n_alignable}/{len(samples)} samples have a searching-sub-question "
          f"count matching the reference (gold-alignable)")

    if args.skip_phase2:
        with open(os.path.join(args.output_dir, "decomp_only_samples.json"), "w") as f:
            json.dump(samples, f, indent=2, ensure_ascii=False, default=str)
        print(f"--skip_phase2 set: stopping after decomposition. Saved "
              f"{args.output_dir}/decomp_only_samples.json. Rerun without --skip_phase2 "
              f"(decomp_cache will be reused automatically) to continue into Cypher generation.")
        return

    # ═══ PHASE 2: Cypher generation (both methods) for alignable samples ═══
    print(f"\n{'='*70}\n  PHASE 2: Cypher generation for {n_alignable} alignable samples\n{'='*70}")
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    driver.verify_connectivity()
    print(f"Neo4j connected: {NEO4J_URI}")

    schema_card = build_schema_card(driver)
    schema_prefix = SCHEMA_PROMPT_HEAD_TEMPLATE.format(schema_card=schema_card)
    vocab = build_vocab(driver)
    print(f"Schema card: {len(schema_card)} chars")

    print(f"\nLaunching SGLang for {args.base_model} ...")
    proc = launch_sglang_server(args.base_model, args.port, extra_args=args.sglang_extra_args)
    if proc is None:
        print("ERROR: failed to launch base-model SGLang server.")
        driver.close()
        return

    # Flat list of (sample_idx, searching_subq_idx) across alignable samples
    # (kept for cache-signature / scoring order); sample_groups below is the
    # same data grouped per-sample for sequential-within-sample generation.
    refs = []
    sample_groups = []
    for si, s in enumerate(samples):
        if not s["alignable"]:
            continue
        si_searching = [sq for sq in s["model_subqs"] if sq["type"] == "searching"]
        group = []
        for qi, sq in enumerate(si_searching):
            refs.append((si, qi, sq["question"]))
            group.append((qi, sq["question"]))
        sample_groups.append((si, group))

    refs_signature = [q for _, _, q in refs]

    try:
        for method_name, prompt_fn in [
            ("method1_fewshot", lambda q, ur: build_fewshot_cypher_prompt(q, upstream_rows=ur)),
            ("method2_schema", lambda q, ur: build_grounded_cypher_prompt(
                q, schema_prefix, driver, vocab, upstream_rows=ur)),
        ]:
            method_cache_path = os.path.join(args.output_dir, f"{method_name}_cypher_cache.json")
            cached_method = None
            if not args.regenerate_cypher and os.path.exists(method_cache_path):
                with open(method_cache_path) as f:
                    candidate = json.load(f)
                if candidate.get("refs_signature") == refs_signature:
                    cached_method = candidate
                else:
                    print(f"  Cache at {method_cache_path} doesn't match this run's sub-questions — regenerating.")

            if cached_method:
                print(f"\n--- {method_name}: loaded {len(cached_method['cyphers'])} cached "
                      f"cypher+exec results from {method_cache_path} (skipping SGLang) ---")
                cyphers = cached_method["cyphers"]
                execs = cached_method["execs"]
            else:
                # method2's prompts embed the ~8.7KB schema card plus per-question
                # retrieved facts (only the schema portion is a shared, cacheable
                # prefix — the facts vary every request), unlike method1's fixed
                # ~4.8KB template reused verbatim for all 150 questions. Repeated
                # SGLang hangs correlated specifically with method2 regardless of
                # server-transition state or repair-round involvement, so it gets
                # a much lower concurrency ceiling.
                method_workers = args.max_workers if method_name == "method1_fewshot" else args.max_workers_schema
                print(f"\n--- {method_name}: generating {len(refs)} Cypher queries across "
                      f"{len(sample_groups)} samples (max_workers={method_workers}, "
                      f"sequential within each sample for cross-step coreference) ---")
                grouped = process_method_sequential(
                    sample_groups, prompt_fn, driver, server_url,
                    args.max_new_tokens_cypher, args.repair_retries, method_workers,
                    label=f"{method_name}")
                cyphers = [grouped[(si, qi)][0] for si, qi, _ in refs]
                execs = [grouped[(si, qi)][1] for si, qi, _ in refs]

                with open(method_cache_path, "w") as f:
                    json.dump({"refs_signature": refs_signature, "cyphers": cyphers, "execs": execs},
                               f, ensure_ascii=False, default=str)
                print(f"  Saved {method_name} cypher+exec results to {method_cache_path} for reuse.")

            # write back + score against gold
            for (si, qi, q_text), cy, ex in zip(refs, cyphers, execs):
                gold_row_set = samples[si]["gold_queries"][qi]["gold_result"] or []
                echo_values = build_echo_values(q_text, vocab)
                score = score_against_gold(ex.get("rows_data") or [], gold_row_set, echo_values)
                samples[si].setdefault(method_name, {})[qi] = {
                    "cypher": cy, "exec_success": ex.get("success"), "has_results": ex.get("has_results"),
                    "error": ex.get("error"), "rows_data": ex.get("rows_data"), "gold_score": score,
                }
    finally:
        hard_shutdown_sglang(proc, args.port)
        driver.close()

    # ═══ PHASE 3: aggregate + report ═══════════════════════════════════════
    def method_summary(method_name):
        total = exec_ok = has_res = full = partial = 0
        coverage_sum = 0.0
        for s in samples:
            if not s["alignable"]:
                continue
            for qi, res in s.get(method_name, {}).items():
                total += 1
                if res["exec_success"]:
                    exec_ok += 1
                if res["has_results"]:
                    has_res += 1
                sc = res["gold_score"]
                if sc["coverage"] is not None:
                    coverage_sum += sc["coverage"]
                    if sc["full_matches"] and sc["full_matches"] >= sc["n_gold_rows"]:
                        full += 1
                    elif sc["partial_matches"] and sc["partial_matches"] >= sc["n_gold_rows"]:
                        partial += 1
        return {
            "total_searching": total,
            "exec_rate": round(exec_ok / total, 4) if total else 0,
            "valid_rate": round(has_res / total, 4) if total else 0,
            "gold_full_match_rate": round(full / total, 4) if total else 0,
            "gold_partial_or_full_match_rate": round((full + partial) / total, 4) if total else 0,
            "avg_gold_coverage": round(coverage_sum / total, 4) if total else 0,
        }

    m1 = method_summary("method1_fewshot")
    m2 = method_summary("method2_schema")

    report = {
        "total_samples": len(samples),
        "gold_alignable_samples": n_alignable,
        "decomposition_fidelity_rate": round(n_alignable / len(samples), 4),
        "method1_fewshot": m1,
        "method2_schema": m2,
    }

    with open(os.path.join(args.output_dir, "finetuned_pipeline_samples.json"), "w") as f:
        json.dump(samples, f, indent=2, ensure_ascii=False, default=str)
    with open(os.path.join(args.output_dir, "finetuned_pipeline_report.json"), "w") as f:
        json.dump(report, f, indent=2)

    with open(os.path.join(args.output_dir, "finetuned_pipeline_report.md"), "w") as f:
        f.write("# Fine-tuned Decomposition -> Cypher -> Gold Comparison\n\n")
        f.write(f"Decomposition model: `{args.decomp_model}`  |  Cypher model: `{args.base_model}`\n\n")
        f.write(f"Total test samples: {len(samples)}  |  Gold-alignable "
                f"(model searching-sub-question count matches reference): {n_alignable} "
                f"({report['decomposition_fidelity_rate']:.1%})\n\n")
        f.write("| Metric | Method 1 (fewshot) | Method 2 (schema+retrieval) |\n|---|---|---|\n")
        for k in ["total_searching", "exec_rate", "valid_rate", "gold_full_match_rate",
                  "gold_partial_or_full_match_rate", "avg_gold_coverage"]:
            f.write(f"| {k} | {m1[k]} | {m2[k]} |\n")

    print(f"\n{'='*70}\n  SUMMARY\n{'='*70}")
    print(f"  Decomposition fidelity: {n_alignable}/{len(samples)} ({report['decomposition_fidelity_rate']:.1%})")
    print(f"  {'Metric':<32}{'Method1 (fewshot)':<20}{'Method2 (schema)':<20}")
    for k in ["total_searching", "exec_rate", "valid_rate", "gold_full_match_rate",
              "gold_partial_or_full_match_rate", "avg_gold_coverage"]:
        print(f"  {k:<32}{str(m1[k]):<20}{str(m2[k]):<20}")
    print(f"\nSaved: {args.output_dir}/finetuned_pipeline_report.md")


if __name__ == "__main__":
    main()
