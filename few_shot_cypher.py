"""
Few-shot English-to-Cypher evaluation using Qwen3-8B-Base.

Flow:
  1. Load decomposition prediction JSON files
  2. Truncate prediction garbage tokens
  3. Extract Sub-questions from reference & prediction
  4. Convert each Sub-question to Cypher via few-shot
  5. Execute Cypher against local Neo4j, check if it returns valid results
  6. Report execution success rate for reference (baseline) and each prediction file

Usage:
    python few_shot_cypher.py --files exp1_test.json exp2_test.json exp3_test.json
    python few_shot_cypher.py --files exp1_test.json --sglang              # auto-launch SGLang (fast)
    python few_shot_cypher.py --files exp1_test.json --sglang_url http://localhost:30000  # existing server
    python few_shot_cypher.py --files exp1_test.json --neo4j_password mypass
    python few_shot_cypher.py --files exp1_test.json --no_execute  # dry run, only generate Cypher
"""

import argparse
import json
import os
import re
import time
import subprocess
import concurrent.futures
import requests
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from neo4j import GraphDatabase

# ─── Configuration ───────────────────────────────────────────────────────────

MODEL_NAME = "Qwen/Qwen3-8B-Base"
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "12345678")

# ─── Few-shot template: graph schema + training-derived demonstrations ─────

FEW_SHOT_TEMPLATE = """You translate ONE searching sub-question into ONE executable Cypher query for Neo4j.

Use this graph schema exactly (labels and relationship types are case-sensitive):
  (:patient)-[:HAS_PREDICTION]->(:prediction)
  (:prediction)-[:GENERATED_BY]->(:model_run)
  (:prediction)-[:EXPLAINED_BY]->(:local_explanation)
  (:local_explanation)-[:GENERATED_BY]->(:model_run)
  (:prediction)-[pr:PREDICTS]->(:condition)              # pr.label, pr.probability
  (:patient)-[m:HAS_BIOMARKER_VALUE]->(:biomarker)      # m.value
  (:local_explanation)-[fi:FEATURE_IMPORTANCE]->(:biomarker)
                                                          # fi.scope='local', fi.shapValue, fi.absShapValue, fi.direction
  (:model_run)-[fi:FEATURE_IMPORTANCE]->(:biomarker)      # fi.scope='global', fi.influence
  (:model_run)-[s:SHAP_SUMMARY]->(:biomarker)             # s.rankByMeanAbsShap, s.meanAbsShap

model_run properties used to select a run: modelType, modelFamily, classRatio.

Hard rules:
1. Filter model runs with mr.modelType and mr.classRatio. Do NOT use dynamic per-run labels and do NOT use modelName.
2. Use toInteger(p.id) for patient ids.
3. Patient biomarker measurements are relationship property m.value, not flat patient properties.
4. Patient-specific SHAP uses the full patient -> prediction -> local_explanation -> model_run path and fi.scope = 'local'.
5. Global importance/rank uses SHAP_SUMMARY. Global increase-related tendency uses FEATURE_IMPORTANCE with fi.scope = 'global'.
6. Prediction label/probability uses prediction-[pr:PREDICTS]->condition; for depression use c.name = 'depression'.
7. Preserve ALL requested models and ALL requested ratios. Never truncate a comparison.
8. Match biomarker.name using the canonical value supplied by the question/context; do not invent unseen biomarker names.
9. Output one Cypher query only, with no explanation or markdown fence.

All demonstrations below are drawn from the training split's searching sub-questions.

# 1. Patient biomarker value
Question: "What is patient 6's vitamin D3 value?"
Cypher: MATCH (p:patient)-[m:HAS_BIOMARKER_VALUE]->(b:biomarker) WHERE toInteger(p.id) = 6 AND b.name = 'Vitamin D3' RETURN b.name AS biomarker, m.value AS value

# 2. Prediction probability
Question: "What depression probability does decision tree predict for patient 115 when trained with positive-to-negative ratios of 1:1, 1:2, 1:3, and 1:4?"
Cypher: MATCH (p:patient)-[:HAS_PREDICTION]->(pred:prediction)-[:GENERATED_BY]->(mr:model_run) MATCH (pred)-[pr:PREDICTS]->(c:condition) WHERE toInteger(p.id) = 115 AND mr.modelType IN ['decisiontree'] AND mr.classRatio IN ['1:1', '1:2', '1:3', '1:4'] AND c.name = 'depression' RETURN mr.modelFamily AS model, mr.classRatio AS ratio, pr.label AS predicted_label, pr.probability AS depression_probability, pred.correct AS correct ORDER BY mr.classRatio, model

# 3. Prediction correctness
Question: "Was the logistic regression prediction for patient 217, using a model trained with a 1:1 positive-to-negative ratio, correct?"
Cypher: MATCH (p:patient)-[:HAS_PREDICTION]->(pred:prediction)-[:GENERATED_BY]->(mr:model_run) MATCH (pred)-[pr:PREDICTS]->(c:condition) WHERE toInteger(p.id) = 217 AND mr.modelType IN ['logisticregression'] AND mr.classRatio IN ['1:1'] AND c.name = 'depression' RETURN mr.modelFamily AS model, mr.classRatio AS ratio, pr.label AS predicted_label, pr.probability AS depression_probability, pred.correct AS correct ORDER BY mr.classRatio, model

# 4. Local SHAP for a named biomarker
Question: "For patient 6 in naive bayes trained with a 1:1 positive-to-negative ratio, what is the local SHAP value of vitamin D3, and is its contribution positive or negative?"
Cypher: MATCH (p:patient)-[:HAS_PREDICTION]->(pred:prediction)-[:EXPLAINED_BY]->(le:local_explanation)-[:GENERATED_BY]->(mr:model_run) MATCH (le)-[fi:FEATURE_IMPORTANCE]->(b:biomarker) WHERE toInteger(p.id) = 6 AND mr.modelType IN ['naivebayes'] AND mr.classRatio IN ['1:1'] AND b.name IN ['Vitamin D3'] AND fi.scope = 'local' RETURN mr.modelFamily AS model, mr.classRatio AS ratio, b.name AS biomarker, fi.shapValue AS shap_value, fi.absShapValue AS abs_shap_value, fi.direction AS direction ORDER BY mr.classRatio, model, biomarker

# 5. Local SHAP for every available biomarker
Question: "For patient 97 in decision tree trained with a 1:1 positive-to-negative ratio, what local SHAP value does each available biomarker have?"
Cypher: MATCH (p:patient)-[:HAS_PREDICTION]->(pred:prediction)-[:EXPLAINED_BY]->(le:local_explanation)-[:GENERATED_BY]->(mr:model_run) MATCH (le)-[fi:FEATURE_IMPORTANCE]->(b:biomarker) WHERE toInteger(p.id) = 97 AND mr.modelType IN ['decisiontree'] AND mr.classRatio IN ['1:1'] AND fi.scope = 'local' RETURN mr.modelFamily AS model, mr.classRatio AS ratio, b.name AS biomarker, fi.shapValue AS shap_value, fi.absShapValue AS abs_shap_value, fi.direction AS direction ORDER BY mr.classRatio, model, biomarker

# 6. Largest positive local SHAP values
Question: "For patient 204 in decision tree trained with a 1:4 positive-to-negative ratio, which 3 biomarkers have the largest positive local SHAP values?"
Cypher: MATCH (p:patient)-[:HAS_PREDICTION]->(pred:prediction)-[:EXPLAINED_BY]->(le:local_explanation)-[:GENERATED_BY]->(mr:model_run) MATCH (le)-[fi:FEATURE_IMPORTANCE]->(b:biomarker) WHERE toInteger(p.id) = 204 AND mr.modelType = 'decisiontree' AND mr.classRatio = '1:4' AND fi.scope = 'local' AND fi.direction = 'positive' RETURN mr.modelFamily AS model, mr.classRatio AS ratio, b.name AS biomarker, fi.shapValue AS shap_value, fi.absShapValue AS abs_shap_value, fi.direction AS direction ORDER BY fi.shapValue DESC, b.name LIMIT 3

# 7. Largest absolute local SHAP value
Question: "For patient 155 in random forest trained with a 1:4 positive-to-negative ratio, which biomarker has the largest absolute local SHAP value?"
Cypher: MATCH (p:patient)-[:HAS_PREDICTION]->(pred:prediction)-[:EXPLAINED_BY]->(le:local_explanation)-[:GENERATED_BY]->(mr:model_run) MATCH (le)-[fi:FEATURE_IMPORTANCE]->(b:biomarker) WHERE toInteger(p.id) = 155 AND mr.modelType = 'randomforest' AND mr.classRatio = '1:4' AND fi.scope = 'local' RETURN mr.modelFamily AS model, mr.classRatio AS ratio, b.name AS biomarker, fi.shapValue AS shap_value, fi.absShapValue AS abs_shap_value, fi.direction AS direction ORDER BY fi.absShapValue DESC, b.name LIMIT 1

# 8. Global SHAP importance / rank
Question: "For random forest trained with a 1:2 positive-to-negative ratio, what are the top 3 biomarkers ranked by global SHAP importance, and what is the global SHAP importance of each?"
Cypher: MATCH (mr:model_run)-[s:SHAP_SUMMARY]->(b:biomarker) WHERE mr.modelType IN ['randomforest'] AND mr.classRatio IN ['1:2'] AND s.rankByMeanAbsShap <= 3 RETURN mr.modelFamily AS model, mr.classRatio AS ratio, b.name AS biomarker, s.rankByMeanAbsShap AS rank, s.meanAbsShap AS global_shap_importance ORDER BY mr.classRatio, model, rank, biomarker

# 9. Global increase-related tendency
Question: "For total cholesterol, what is the global SHAP tendency as the biomarker value increases in random forest, naive bayes, logistic regression, and decision tree at training ratios 1:1, 1:2, 1:3, 1:4, and 1:5?"
Cypher: MATCH (mr:model_run)-[fi:FEATURE_IMPORTANCE]->(b:biomarker) WHERE mr.modelType IN ['randomforest', 'naivebayes', 'logisticregression', 'decisiontree'] AND mr.classRatio IN ['1:1', '1:2', '1:3', '1:4', '1:5'] AND fi.scope = 'global' AND b.name IN ['Total Cholesterol'] RETURN mr.modelFamily AS model, mr.classRatio AS ratio, b.name AS biomarker, fi.influence AS increase_related_tendency ORDER BY mr.classRatio, model, biomarker

# 10. Global top-k rank + tendency in one searching step
Question: "For random forest and decision tree trained with a 1:1 positive-to-negative ratio, what are the top 3 biomarkers ranked by global SHAP importance, what is each biomarker's global SHAP importance, and what is its global SHAP tendency as the biomarker value increases?"
Cypher: MATCH (mr:model_run)-[s:SHAP_SUMMARY]->(b:biomarker) MATCH (mr)-[fi:FEATURE_IMPORTANCE]->(b) WHERE mr.modelType IN ['randomforest', 'decisiontree'] AND mr.classRatio IN ['1:1'] AND fi.scope = 'global' AND s.rankByMeanAbsShap <= 3 RETURN mr.modelFamily AS model, mr.classRatio AS ratio, b.name AS biomarker, s.rankByMeanAbsShap AS rank, s.meanAbsShap AS global_shap_importance, fi.influence AS increase_related_tendency ORDER BY mr.classRatio, model, rank, biomarker

# 11. Model accuracy
Question: "What is the accuracy of random forest trained at 1:2, 1:3, 1:4, and 1:5?"
Cypher: MATCH (pred:prediction)-[:GENERATED_BY]->(mr:model_run) WHERE mr.modelType = 'randomforest' AND mr.classRatio IN ['1:2', '1:3', '1:4', '1:5'] RETURN mr.classRatio AS ratio, avg(CASE WHEN pred.correct THEN 1.0 ELSE 0.0 END) AS accuracy, count(pred) AS prediction_count ORDER BY ratio

# 12. Biomarker clinical description
Question: "What is the clinical meaning of creatinine?"
Cypher: MATCH (b:biomarker) WHERE b.name = 'Creatinine' RETURN b.name AS biomarker, b.description AS description

Previous searching-step rows, if any: {upstream_context}
If the sub-question says "those biomarkers" or "that biomarker", bind the exact Biomarker.name values from these rows in the new Cypher WHERE clause.

Now generate Cypher for the following searching sub-question.
Question: "{input_text}"
Cypher:"""


def _training_searching_questions(train_dataset_path: str) -> set[str]:
    """Read only the training split and return its searching sub-questions."""
    with open(train_dataset_path, "r", encoding="utf-8") as f:
        items = json.load(f)
    questions = set()
    for item in items:
        text = item.get("output", "")
        for match in re.finditer(
            r"Sub-question:\s*(.+?)\s*#\s*Type:\s*searching",
            text,
        ):
            questions.add(match.group(1).strip())
    return questions


def audit_few_shot_provenance(train_dataset_path: str) -> dict:
    """Verify every static few-shot Question is present verbatim in training data.

    This audit intentionally reads only the supplied training split.
    """
    train_questions = _training_searching_questions(train_dataset_path)
    demo_questions = re.findall(r'^Question:\s*"(.+?)"\s*$', FEW_SHOT_TEMPLATE, re.M)
    demo_questions = [q for q in demo_questions if "{input_text}" not in q]
    missing = [q for q in demo_questions if q not in train_questions]
    return {
        "demo_count": len(demo_questions),
        "all_demos_in_training_split": not missing,
        "missing_demo_questions": missing,
    }


def clean_cypher_output(text: str) -> str:
    """Extract one complete Cypher statement without truncating multiline output."""
    text = (text or "").strip()
    if not text:
        return ""
    fence = re.search(r"```(?:cypher)?\s*(.*?)```", text, re.I | re.S)
    if fence:
        text = fence.group(1).strip()
    text = re.sub(r"^\s*Cypher\s*:\s*", "", text, flags=re.I)
    text = re.split(r"\n\s*(?:Question|Explanation|Answer)\s*:", text, maxsplit=1, flags=re.I)[0]
    return " ".join(text.split()).strip()


def build_cypher_prompt(question: str, upstream_rows=None) -> str:
    upstream_rows = upstream_rows or []
    compact = json.dumps(upstream_rows, ensure_ascii=False, default=str)
    return FEW_SHOT_TEMPLATE.format(input_text=question, upstream_context=compact)


def needs_upstream_context(question: str) -> bool:
    return bool(re.search(r"\b(those(?: top)? \d+ biomarkers|those biomarkers|that biomarker)\b", question, re.I))


# ─── Truncation ──────────────────────────────────────────────────────────────

def truncate_output(text: str) -> str:
    """
    Truncate prediction output at first garbage marker.
    Then keep up to the last valid '# Type:' line.
    This prevents extracting duplicate/hallucinated sub-questions from looping.
    """
    # Step 1: find earliest garbage position
    garbage_patterns = ['�', '<|im_start|>', '\x00']
    cut_pos = len(text)
    for pat in garbage_patterns:
        idx = text.find(pat)
        if 0 <= idx < cut_pos:
            cut_pos = idx

    # Also cut at model looping artifacts (new turn markers)
    user_turn = re.search(r'\nuser\n|\nassistant\n', text)
    if user_turn and user_turn.start() < cut_pos:
        cut_pos = user_turn.start()

    clean_text = text[:cut_pos]
    lines = clean_text.split("\n")

    # Step 2: under sampling the model sometimes doesn't stop after a complete
    # decomposition and loops into a second, near-duplicate (or outright
    # different) attempt starting again from "Step 1:". Cut at the first such
    # restart so the selected text is always the model's *first* complete
    # answer, never a looped re-attempt.
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

    # Step 3: keep up to the last valid # Type: line in the loop-free portion
    last_valid_idx = -1
    for i, line in enumerate(lines):
        if re.search(r"#\s*Type:\s*(searching|aggregation)", line.strip()):
            last_valid_idx = i

    if last_valid_idx >= 0:
        return "\n".join(lines[:last_valid_idx + 1]).strip()

    return "\n".join(lines).strip()


# ─── Sub-question extraction ────────────────────────────────────────────────

def extract_subquestions(text: str) -> list[dict]:
    """
    Extract searching-type Sub-questions from structured text.
    Skips aggregation type (not mappable to a single Cypher query).
    Returns list of {"question": str, "type": "searching"}.
    """
    results = []
    for match in re.finditer(
        r"Sub-question:\s*(.+?)\s*#\s*Type:\s*(searching|aggregation)",
        text
    ):
        q = match.group(1).strip()
        t = match.group(2).strip()
        if t == "aggregation":
            continue
        # Deduplicate exact repeats, but do not silently discard searching atoms.
        if not any(r["question"] == q for r in results):
            results.append({"question": q, "type": t})
    return results


# ─── SGLang server management ────────────────────────────────────────────────

SGLANG_PORT = 30000
SGLANG_URL = f"http://localhost:{SGLANG_PORT}"
SGLANG_TIMEOUT = 60
MAX_RETRIES = 3


def launch_sglang_server(model_path, port=SGLANG_PORT, extra_args=None):
    """Launch sglang server and wait until ready.

    extra_args: additional CLI flags appended verbatim, e.g.
    ["--disable-overlap-schedule", "--watchdog-timeout", "120"] to trade the
    overlap scheduler's speed for robustness against the GPU-idle/unresponsive
    hangs seen under sustained concurrent load (the overlap scheduler runs the
    CPU scheduler and GPU worker concurrently — a known deadlock-prone pattern
    that matches those symptoms) and to have the server crash instead of
    hanging silently if a batch does stall.
    """
    cmd = [
        "python", "-m", "sglang.launch_server",
        "--model-path", model_path,
        "--port", str(port),
        "--dtype", "bfloat16",
        "--host", "0.0.0.0",
        "--max-running-requests", "64",
        "--mem-fraction-static", "0.88",
    ]
    if extra_args:
        cmd += extra_args
    print(f"  Launching sglang server...")
    print(f"  Command: {' '.join(cmd)}")

    # IMPORTANT: stdout/stderr go to a real file, never subprocess.PIPE.
    # Nothing in this codebase drains a PIPE once the server is up (only the
    # startup-failure branch below ever reads it) — SGLang logs continuously
    # while serving, so an unread PIPE's small OS buffer (~64KB on Linux)
    # eventually fills, and the server's own print/logging calls then BLOCK
    # trying to write to it, freezing the whole process. This reproduces
    # exactly as "0% GPU utilization, health endpoint unresponsive" after
    # enough cumulative log volume — independent of request concurrency,
    # which is why lowering --max_workers never fixed it. A file has no such
    # backpressure.
    log_path = f"/tmp/sglang_server_{port}.log"
    log_file = open(log_path, "w")
    process = subprocess.Popen(cmd, stdout=log_file, stderr=subprocess.STDOUT, text=True)

    url = f"http://localhost:{port}/health"
    max_wait = 900
    start = time.time()
    print(f"  Waiting for server to start (max {max_wait}s)... (log: {log_path})")

    while time.time() - start < max_wait:
        try:
            resp = requests.get(url, timeout=3)
            if resp.status_code == 200:
                print(f"  Server ready! (took {time.time()-start:.0f}s)")
                return process
        except Exception:
            pass
        if process.poll() is not None:
            print(f"  ERROR: sglang server exited with code {process.returncode}")
            try:
                with open(log_path) as f:
                    print(f"  Output:\n{f.read()[-2000:]}")
            except Exception:
                pass
            return None
        time.sleep(3)

    print(f"  ERROR: Server did not start within {max_wait}s")
    process.terminate()
    return None


def shutdown_sglang_server(process):
    """Gracefully shutdown sglang server."""
    if process and process.poll() is None:
        print(f"  Shutting down sglang server...")
        process.terminate()
        try:
            process.wait(timeout=10)
        except Exception:
            process.kill()
        print(f"  Server stopped.")


# ─── Model loading & Cypher generation ──────────────────────────────────────

def load_model(model_name: str):
    print(f"Loading model: {model_name} ...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    print("Model loaded.")
    return model, tokenizer


def generate_cypher(model, tokenizer, question: str, max_new_tokens: int = 256, upstream_rows=None) -> str:
    prompt = build_cypher_prompt(question, upstream_rows=upstream_rows)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            repetition_penalty=1.05,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated_ids = outputs[0, inputs["input_ids"].shape[1]:]
    text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    return clean_cypher_output(text)


def generate_cypher_sglang(question: str, sglang_url: str, max_new_tokens: int = 256, upstream_rows=None) -> str:
    """Generate Cypher via SGLang's text completion API (for base model)."""
    prompt = build_cypher_prompt(question, upstream_rows=upstream_rows)
    payload = {
        "model": "default",
        "prompt": prompt,
        "max_tokens": max_new_tokens,
        "temperature": 0,
        "top_p": 1.0,
        "repetition_penalty": 1.05,
        "stop": ["\nQuestion:", "\nExplanation:", "\nAnswer:"],
    }
    for attempt in range(MAX_RETRIES):
        try:
            resp = requests.post(
                f"{sglang_url}/v1/completions",
                json=payload,
                timeout=SGLANG_TIMEOUT,
            )
            resp.raise_for_status()
            return clean_cypher_output(resp.json()["choices"][0]["text"])
        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                time.sleep(2)
            else:
                print(f"    generate_cypher_sglang failed: {e}")
                return ""


def generate_cypher_sglang_batch(subqs: list[dict], sglang_url: str,
                                  max_new_tokens: int = 128, max_workers: int = 32,
                                  label: str = "") -> list[str]:
    """Batch generate Cypher queries via SGLang with concurrent requests."""
    results = [""] * len(subqs)

    def infer_one(idx_sq):
        idx, sq = idx_sq
        return idx, generate_cypher_sglang(sq["question"], sglang_url, max_new_tokens)

    t_start = time.time()
    done = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = executor.map(infer_one, enumerate(subqs))
        for idx, cypher in futures:
            results[idx] = cypher
            done += 1
            if done % 20 == 0 or done == len(subqs):
                elapsed = time.time() - t_start
                speed = done / elapsed if elapsed > 0 else 0
                eta = (len(subqs) - done) / speed if speed > 0 else 0
                print(f"    [{label}] {done}/{len(subqs)} done, {speed:.1f} q/s, ETA {eta:.0f}s")

    elapsed = time.time() - t_start
    print(f"    [{label}] Finished {len(subqs)} queries in {elapsed:.1f}s")
    return results


# ─── Neo4j execution ────────────────────────────────────────────────────────

def execute_cypher(driver, cypher: str) -> dict:
    """
    Execute a Cypher query. Returns:
      {"success": bool, "has_results": bool, "error": str|None, "rows": int}
    """
    try:
        with driver.session() as session:
            result = session.run(cypher)
            records = [record.data() for record in result]
            return {
                "success": True,
                "has_results": len(records) > 0,
                "error": None,
                "rows": len(records),
                "records": records,
            }
    except Exception as e:
        return {
            "success": False,
            "has_results": False,
            "error": str(e),
            "rows": 0,
            "records": [],
        }


# ─── Core: process a list of sub-questions ──────────────────────────────────

def process_subquestions(subqs, model, tokenizer, driver, max_new_tokens, label="",
                        sglang_url=None):
    """
    For each sub-question: generate Cypher → execute → record result.
    If sglang_url is set, uses SGLang batch inference (much faster).
    Returns list of per-query results.
    """
    # Generate all Cypher queries
    if sglang_url:
        cypher_list = generate_cypher_sglang_batch(
            subqs, sglang_url, max_new_tokens, label=label)
    else:
        cypher_list = []
        for i, sq in enumerate(subqs):
            cypher_list.append(generate_cypher(model, tokenizer, sq["question"], max_new_tokens))
            if (i + 1) % 20 == 0:
                print(f"    [{label}] generated {i+1}/{len(subqs)}...")

    # Execute against Neo4j
    results = []
    for i, sq in enumerate(subqs):
        exec_result = {"success": None, "has_results": None, "error": None, "rows": 0}
        if driver is not None:
            exec_result = execute_cypher(driver, cypher_list[i])

        results.append({
            "question": sq["question"],
            "type": sq["type"],
            "cypher": cypher_list[i],
            "exec_success": exec_result["success"],
            "has_results": exec_result["has_results"],
            "error": exec_result["error"],
            "rows": exec_result["rows"],
        })

    return results


def process_subquestion_group(subqs, model, tokenizer, driver, max_new_tokens, label="", sglang_url=None):
    """Process one sample in decomposition order so later searching steps can use earlier rows.

    This is required for coreferential atoms such as "those top 3 biomarkers"
    and "that biomarker".  The old flattened batch pipeline made those queries
    fundamentally under-specified.
    """
    results = []
    upstream_rows = []
    for sq in subqs:
        context_rows = upstream_rows if needs_upstream_context(sq["question"]) else []
        if sglang_url:
            cypher = generate_cypher_sglang(
                sq["question"], sglang_url, max_new_tokens, upstream_rows=context_rows
            )
        else:
            cypher = generate_cypher(
                model, tokenizer, sq["question"], max_new_tokens, upstream_rows=context_rows
            )

        exec_result = {"success": None, "has_results": None, "error": None, "rows": 0, "records": []}
        if driver is not None:
            exec_result = execute_cypher(driver, cypher)

        results.append({
            "question": sq["question"],
            "type": sq["type"],
            "cypher": cypher,
            "used_upstream_rows": context_rows,
            "exec_success": exec_result["success"],
            "has_results": exec_result["has_results"],
            "error": exec_result["error"],
            "rows": exec_result["rows"],
            "records": exec_result.get("records", []),
        })

        # Only a successful non-empty searching step becomes the binding source
        # for the next searching step in the same decomposition.
        if exec_result.get("records"):
            upstream_rows = exec_result["records"]

    return results


def process_subquestion_groups(groups, model, tokenizer, driver, max_new_tokens, label="", sglang_url=None):
    """Correctness-first grouped processor; preserves sample and step order."""
    all_results = []
    for i, subqs in enumerate(groups):
        group_results = process_subquestion_group(
            subqs, model, tokenizer, driver, max_new_tokens,
            label=f"{label}:{i}", sglang_url=sglang_url,
        )
        all_results.extend(group_results)
        if (i + 1) % 20 == 0 or i + 1 == len(groups):
            print(f"    [{label}] processed {i+1}/{len(groups)} samples...")
    return all_results


def compute_metrics(query_results: list[dict]) -> dict:
    """Compute execution metrics from query results."""
    total = len(query_results)
    if total == 0:
        return {"total": 0, "exec_success": 0, "has_results": 0,
                "exec_rate": 0, "valid_rate": 0}

    exec_ok = sum(1 for r in query_results if r["exec_success"])
    has_res = sum(1 for r in query_results if r["has_results"])

    return {
        "total": total,
        "exec_success": exec_ok,
        "has_results": has_res,
        "exec_rate": round(exec_ok / total, 4),
        "valid_rate": round(has_res / total, 4),
    }


# ─── Main ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Few-shot Cypher evaluation")
    parser.add_argument("--files", type=str, nargs="+", required=True,
                        help="decomposition prediction JSON files")
    parser.add_argument("--model", type=str, default=MODEL_NAME,
                        help="Model name or local path")
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--neo4j_uri", type=str, default=NEO4J_URI)
    parser.add_argument("--neo4j_user", type=str, default=NEO4J_USER)
    parser.add_argument("--neo4j_password", type=str, default=NEO4J_PASSWORD)
    parser.add_argument("--no_execute", action="store_true",
                        help="Only generate Cypher, skip Neo4j execution")
    parser.add_argument("--output_dir", type=str, default="cypher_results",
                        help="Directory to save results")
    parser.add_argument("--sglang", action="store_true",
                        help="Use SGLang server for inference (much faster)")
    parser.add_argument("--sglang_url", type=str, default=None,
                        help="Use an already-running SGLang server at this URL")
    parser.add_argument("--port", type=int, default=SGLANG_PORT,
                        help="Port for auto-launched SGLang server")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # ── Load files ───────────────────────────────────────────────────────
    file_data = {}
    for fpath in args.files:
        if not os.path.exists(fpath):
            print(f"File not found: {fpath}, skipping.")
            continue
        # Use parent directory name to distinguish files with the same basename
        parent_dir = os.path.basename(os.path.dirname(os.path.abspath(fpath)))
        base_name = os.path.splitext(os.path.basename(fpath))[0]
        name = f"{parent_dir}_{base_name}" if parent_dir else base_name
        with open(fpath) as f:
            file_data[name] = json.load(f)
        print(f"Loaded {name}: {len(file_data[name])} samples")

    if not file_data:
        print("No data loaded. Exiting.")
        return

    first_data = list(file_data.values())[0]
    num_samples = len(first_data)

    # ── Extract reference sub-questions (shared baseline, once) ──────────
    print(f"\n{'='*70}")
    print("  Extracting reference sub-questions (baseline)")
    print(f"{'='*70}")

    ref_all_subqs = []  # flat list for batch processing
    ref_per_sample = []  # per-sample for detail output
    for sample in first_data:
        subqs = extract_subquestions(sample["reference"])
        ref_per_sample.append(subqs)
        ref_all_subqs.extend(subqs)

    print(f"  Total: {len(ref_all_subqs)} sub-questions from {num_samples} samples")
    print(f"  Types: {len(ref_all_subqs)} searching (aggregation skipped)")

    # ── Load model or launch SGLang ──────────────────────────────────────
    use_sglang = args.sglang or args.sglang_url
    sglang_process = None
    server_url = None
    model, tokenizer = None, None

    if use_sglang:
        server_url = args.sglang_url or f"http://localhost:{args.port}"
        # Check if server already running
        try:
            resp = requests.get(f"{server_url}/health", timeout=3)
            if resp.status_code == 200:
                print(f"  Using existing SGLang server at {server_url}")
        except Exception:
            if args.sglang_url:
                print(f"  ERROR: Cannot reach SGLang server at {server_url}")
                return
            print(f"  Auto-launching SGLang server for {args.model}...")
            sglang_process = launch_sglang_server(args.model, args.port)
            if sglang_process is None:
                print("  ERROR: Failed to launch SGLang server. Exiting.")
                return
    else:
        model, tokenizer = load_model(args.model)

    # ── Connect Neo4j ────────────────────────────────────────────────────
    driver = None
    if not args.no_execute:
        try:
            driver = GraphDatabase.driver(
                args.neo4j_uri, auth=(args.neo4j_user, args.neo4j_password))
            driver.verify_connectivity()
            print(f"Neo4j connected: {args.neo4j_uri}")
        except Exception as e:
            print(f"Neo4j connection failed: {e}")
            print("Falling back to --no_execute mode (Cypher generation only).")
            driver = None

    # ── Process reference baseline ───────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"  Processing reference baseline ({len(ref_all_subqs)} queries)")
    print(f"{'='*70}")
    ref_results = process_subquestion_groups(
        ref_per_sample, model, tokenizer, driver, args.max_new_tokens, label="ref",
        sglang_url=server_url)

    ref_metrics = compute_metrics(ref_results)
    print(f"\n  --- Reference Baseline ---")
    print(f"  Total queries:    {ref_metrics['total']}")
    if driver:
        print(f"  Exec success:     {ref_metrics['exec_success']}/{ref_metrics['total']} = {ref_metrics['exec_rate']:.1%}")
        print(f"  Valid (non-empty): {ref_metrics['has_results']}/{ref_metrics['total']} = {ref_metrics['valid_rate']:.1%}")

    # Save baseline
    baseline_out = os.path.join(args.output_dir, "reference_baseline.json")
    with open(baseline_out, "w") as f:
        json.dump({"metrics": ref_metrics, "queries": ref_results},
                  f, indent=2, ensure_ascii=False)
    print(f"  Saved: {baseline_out}")

    # ── Process each prediction file ─────────────────────────────────────
    all_metrics = {"reference": ref_metrics}

    for name, data in file_data.items():
        print(f"\n{'='*70}")
        print(f"  Processing prediction: {name}")
        print(f"{'='*70}")

        # Truncate + extract
        pred_all_subqs = []
        pred_per_sample = []
        for sample in data:
            truncated = truncate_output(sample["prediction"])
            subqs = extract_subquestions(truncated)
            pred_per_sample.append(subqs)
            pred_all_subqs.extend(subqs)

        print(f"  Total: {len(pred_all_subqs)} sub-questions (after truncation)")
        print(f"  Types: {len(pred_all_subqs)} searching (aggregation skipped)")

        # Generate Cypher + execute
        pred_results = process_subquestion_groups(
            pred_per_sample, model, tokenizer, driver, args.max_new_tokens, label=name,
            sglang_url=server_url)

        pred_metrics = compute_metrics(pred_results)
        all_metrics[name] = pred_metrics

        print(f"\n  --- {name} Results ---")
        print(f"  Total queries:    {pred_metrics['total']}")
        if driver:
            print(f"  Exec success:     {pred_metrics['exec_success']}/{pred_metrics['total']} = {pred_metrics['exec_rate']:.1%}")
            print(f"  Valid (non-empty): {pred_metrics['has_results']}/{pred_metrics['total']} = {pred_metrics['valid_rate']:.1%}")

        # Save per-file detail (with per-sample grouping for readability)
        detail_output = []
        idx = 0
        for i, sample in enumerate(data):
            n = len(pred_per_sample[i])
            detail_output.append({
                "index": i,
                "input": sample["input"],
                "prediction_truncated": truncate_output(sample["prediction"]),
                "num_subquestions": n,
                "queries": pred_results[idx:idx+n],
            })
            idx += n

        out_file = os.path.join(args.output_dir, f"{name}_cypher_eval.json")
        with open(out_file, "w") as f:
            json.dump({"metrics": pred_metrics, "details": detail_output},
                      f, indent=2, ensure_ascii=False)
        print(f"  Saved: {out_file}")

    # ── Summary table ────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("  Summary")
    print(f"{'='*70}")
    if driver:
        print(f"  {'Source':<25} {'Queries':<10} {'Exec OK':<12} {'Valid':<12}")
        print(f"  {'-'*59}")
        for name, m in all_metrics.items():
            print(f"  {name:<25} {m['total']:<10} "
                  f"{m['exec_rate']:<12.1%} {m['valid_rate']:<12.1%}")
    else:
        print(f"  {'Source':<25} {'Queries':<10}")
        print(f"  {'-'*35}")
        for name, m in all_metrics.items():
            print(f"  {name:<25} {m['total']:<10}")
        print("\n  (Neo4j not connected — run without --no_execute to test execution)")

    # Save combined
    combined_file = os.path.join(args.output_dir, "combined_metrics.json")
    with open(combined_file, "w") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"\nCombined metrics: {combined_file}")

    if driver:
        driver.close()

    if sglang_process:
        shutdown_sglang_server(sglang_process)


if __name__ == "__main__":
    main()
