"""
Few-shot English-to-Cypher evaluation using Qwen3-8B-Base.

Flow:
  1. Load test_predictions.json files
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
NEO4J_URI = "bolt://6.tcp.ngrok.io:17035"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "12345678"

# ─── Few-shot template ──────────────────────────────────────────────────────

FEW_SHOT_TEMPLATE = """You are an expert in translating English questions into Cypher queries for a Neo4j database.
If the question involves the ratio of the algorithm, then cypher will contain a suffix describing the ratio.
Like naive bayes with a 1:3 ratio should be naivebayes_1_3. Here are some examples of how to do this:

Question: "What is the blood lead value of patient 10 ?"
Cypher: match (p:patient) where p.id = 10 return p.blood_lead

Question: "Please tell me the albumin status of patient 22."
Cypher: match (p:patient) where p.id = 22 return p.albumin

Question: "Is the prediction outcome from random forest with ratio 1 to 1 for patient 16 correct?"
Cypher: match (p:patient)-[r:predicted_by]-(shap:randomforest_1_1) where p.id = 16 return r.correct

Question: "Is the prediction for patient 30 correct in the naive bayes algorithm with a 1:3 ratio ?"
Cypher: match (p:patient)-[r:predicted_by]-(shap:naivebayes_1_3) where p.id = 30 return r.correct

Question: "According to random forest with a 1:1 ratio, what is the impact of creatinine on patient 12 ?"
Cypher: match (shap:randomforest_1_1) where shap.id = 12 return shap.creatinine

Question: "Through the analysis of logistic regression algorithm with a 1:3 ratio, what is the shap value of direct hdl cholesterol on patient 32 ?"
Cypher: match (shap:logisticregression_1_3) where shap.id = 32 return shap.direct_hdl_cholesterol

Question: "What is the depression probability for patient 82 in decision tree at 1:4 ratio?"
Cypher: match (shap:decisiontree_1_2) where shap.id = 82 return shap.probability

Question: "Please show me the predicted probability of depression for patient 28 from the logistic regression model at a 1:3 ratio."
Cypher: match (shap:logisticregression_1_3) where shap.id = 28 return shap.probability

Question: "What did logistic regression predict for patient 12 at 1:2 ratio?"
Cypher: match (p:patient)-[r:predicted_by]-(shap:logisticregression_1_2) where p.id = 12 return r.predict_result

Question: "What is the prediction of the decision tree for patient 40 at a 1:5 ratio?"
Cypher: match (p:patient)-[r:predicted_by]-(shap:decisiontree_1_5) where p.id = 40 return r.predict_result

Question: "What is the definition of vitamin d3 ?"
Cypher: match (b:biomarker) where b.name = "vitamin d3" return b.description

Question: "What is the concept of creatinine ?"
Cypher: match (b:biomarker) where b.name = "creatinine" return b.description

Question: "In general, based on logistic regression with a 1:3 ratio, what is the intensity of direct hdl cholesterol?"
Cypher: match (b:biomarker)-[r:feature_importance]-(alg:algorithm) where b.name = "direct hdl-cholesterol" and alg.model_name = "logisticregression_1_3" return r.rank

Question: "Overall, based on naive bayes with a 1:2 ratio, what is the impact strength of blood lead ?"
Cypher: match (b:biomarker)-[r:feature_importance]-(alg:algorithm) where b.name = "blood lead" and alg.model_name = "naivebayes_1_2" return r.rank

Question: "As the value of vitamin d2 increases, how will it influence the prediction result of the random forest with a 1:1 ratio ?"
Cypher: match (b:biomarker)-[r:feature_importance]-(alg:algorithm) where b.name='vitamin d2' and alg.model_name='randomforest_1_1' return r.influence

Question: "When the value of total cholesterol increases, how will it affect the prediction result of decision tree with a 1:2 ratio ?"
Cypher: match (b:biomarker)-[r:feature_importance]-(alg:algorithm) where b.name='total cholesterol' and alg.model_name='decisiontree_1_2' return r.influence

Question: "What is the overall precision of decision tree with a 1:2 ratio ?"
Cypher: match (alg:algorithm) where alg.model_name = "decisiontree_1_2" return alg.precision_0, alg.precision_1

Question: "What is the overall f1 score of logistic regression with a 1:3 ratio ?"
Cypher: match (alg:algorithm) where alg.model_name = "logisticregression_1_3" return alg.f1_0, alg.f1_1

Question: "According to logistic regression with a 1:2 ratio, which features caused patient 108 to be predicted as depressed?"
Cypher: match (shap:logisticregression_1_2) where shap.id = 108 and any(key in keys(shap) where shap[key] > 0) with shap, [key in keys(shap)  where shap[key] > 0  and not key in ['id', 'predict_result', 'probability']] as positive_keys return positive_keys limit 1

Question: "Which features led the random forest model with a 1:4 ratio to predict patient 22 as depressed?"
Cypher: match (shap:randomforest_1_4) where shap.id = 22 and any(key in keys(shap) where shap[key] > 0) with shap, [key in keys(shap)  where shap[key] > 0  and not key in ['id', 'predict_result', 'probability']] as positive_keys return positive_keys limit 1

Now, based on the above examples, generate a Cypher query for the following question:

Question: "{input_text}"
Cypher:"""


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

    # Step 2: keep up to the last valid # Type: line
    lines = clean_text.split("\n")
    last_valid_idx = -1
    for i, line in enumerate(lines):
        if re.search(r"#\s*Type:\s*(searching|aggregation)", line.strip()):
            last_valid_idx = i

    if last_valid_idx >= 0:
        return "\n".join(lines[:last_valid_idx + 1]).strip()

    return clean_text.strip()


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
        # Deduplicate: skip if exact same question already extracted
        if not any(r["question"] == q for r in results):
            # Skip sub-questions in categories not covered by few-shot template
            q_lower = q.lower()
            if "actually have depression" in q_lower or "actually depressed" in q_lower:
                continue
            if re.search(r"which patient has the (highest|lowest)", q_lower):
                continue
            results.append({"question": q, "type": t})
    return results


# ─── SGLang server management ────────────────────────────────────────────────

SGLANG_PORT = 30000
SGLANG_URL = f"http://localhost:{SGLANG_PORT}"
SGLANG_TIMEOUT = 60
MAX_RETRIES = 3


def launch_sglang_server(model_path, port=SGLANG_PORT):
    """Launch sglang server and wait until ready."""
    cmd = [
        "python", "-m", "sglang.launch_server",
        "--model-path", model_path,
        "--port", str(port),
        "--dtype", "bfloat16",
        "--host", "0.0.0.0",
    ]
    print(f"  Launching sglang server...")
    print(f"  Command: {' '.join(cmd)}")

    process = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )

    url = f"http://localhost:{port}/health"
    max_wait = 300
    start = time.time()
    print(f"  Waiting for server to start (max {max_wait}s)...")

    while time.time() - start < max_wait:
        try:
            resp = requests.get(url, timeout=3)
            if resp.status_code == 200:
                print(f"  Server ready! (took {time.time()-start:.0f}s)")
                return process
        except Exception:
            pass
        if process.poll() is not None:
            output = process.stdout.read()
            print(f"  ERROR: sglang server exited with code {process.returncode}")
            print(f"  Output:\n{output[-2000:]}")
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


def generate_cypher(model, tokenizer, question: str, max_new_tokens: int = 128) -> str:
    prompt = FEW_SHOT_TEMPLATE.format(input_text=question)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=1.0,
            repetition_penalty=1.1,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated_ids = outputs[0, inputs["input_ids"].shape[1]:]
    cypher = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    # Take only the first line
    cypher = cypher.split("\n")[0].strip()
    return cypher


def generate_cypher_sglang(question: str, sglang_url: str, max_new_tokens: int = 128) -> str:
    """Generate Cypher via SGLang's text completion API (for base model)."""
    prompt = FEW_SHOT_TEMPLATE.format(input_text=question)
    payload = {
        "model": "default",
        "prompt": prompt,
        "max_tokens": max_new_tokens,
        "temperature": 0,
        "top_p": 1.0,
        "repetition_penalty": 1.1,
    }
    for attempt in range(MAX_RETRIES):
        try:
            resp = requests.post(
                f"{sglang_url}/v1/completions",
                json=payload,
                timeout=SGLANG_TIMEOUT,
            )
            resp.raise_for_status()
            text = resp.json()["choices"][0]["text"].strip()
            return text.split("\n")[0].strip()
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
            }
    except Exception as e:
        return {
            "success": False,
            "has_results": False,
            "error": str(e),
            "rows": 0,
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
                        help="test_predictions.json files")
    parser.add_argument("--model", type=str, default=MODEL_NAME,
                        help="Model name or local path")
    parser.add_argument("--max_new_tokens", type=int, default=128)
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
    ref_results = process_subquestions(
        ref_all_subqs, model, tokenizer, driver, args.max_new_tokens, label="ref",
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
        pred_results = process_subquestions(
            pred_all_subqs, model, tokenizer, driver, args.max_new_tokens, label=name,
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
