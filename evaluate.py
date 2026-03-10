"""
evaluate.py - Inference & Evaluation for SFT Experiments
=========================================================
1) Loads trained checkpoints and generates predictions on test set
2) Evaluates on 3 dimensions:
   - Semantic Similarity (text-embedding-3-large cosine + Hungarian matching)
   - Sub-question Count Accuracy
   - Redundancy Detection (embedding flagging + GPT verification)

Usage:
    python evaluate.py --exp 1                     # inference + evaluate
    python evaluate.py --exp 1 2 3 4 5             # all experiments + comparison
    python evaluate.py --exp 1 --skip_inference     # evaluate existing predictions
    python evaluate.py --exp 1 --api_base https://your-proxy.com/v1
"""

import os
import re
import json
import time
import argparse
import numpy as np
from typing import List, Dict
from dataclasses import dataclass, field
from scipy.optimize import linear_sum_assignment

# ====================================================================
# Configuration
# ====================================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
OUTPUT_ROOT = os.path.join(BASE_DIR, "nhanes_model")
TEST_FILE = os.path.join(DATA_DIR, "shared_test.json")

MODEL_NAME = "Qwen/Qwen3-8B-Base"

EMBEDDING_MODEL = "text-embedding-3-large"
GPT_MODEL = "gpt-4o"
OPENAI_API_KEY = "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"  # ← 换成你自己的key
OPENAI_API_BASE = None  # 如果用代理，改成 "https://your-proxy.com/v1"
MAX_RETRIES = 3
RETRY_DELAY = 5
EMBEDDING_BATCH_SIZE = 100

HIGH_SIM_THRESHOLD = 0.85
PERFECT_SIM_THRESHOLD = 0.92
REDUNDANCY_SIM_THRESHOLD = 0.90

SYSTEM_PROMPT = (
    "You are a question decomposition assistant. Given a complex question, "
    "break it down into a sequence of simpler sub-questions that can be answered independently. "
    "For each step, provide a clear sub-question and label its type as either 'searching' "
    "(retrieving specific information) or 'aggregation' (combining results from previous steps)."
)


# ====================================================================
# Data Structures
# ====================================================================

@dataclass
class SubQuestion:
    step_num: int
    description: str
    sub_question: str
    sq_type: str


@dataclass
class SampleEval:
    input_text: str
    ref_output: str
    pred_output: str
    ref_subs: List[SubQuestion] = field(default_factory=list)
    pred_subs: List[SubQuestion] = field(default_factory=list)
    count_ref: int = 0
    count_pred: int = 0
    count_match: bool = False
    alignment: List[Dict] = field(default_factory=list)
    semantic_scores: List[float] = field(default_factory=list)
    semantic_avg: float = 0.0
    unmatched_pred: List[int] = field(default_factory=list)
    unmatched_ref: List[int] = field(default_factory=list)
    redundant_pairs: List[Dict] = field(default_factory=list)
    redundancy_count: int = 0
    has_redundancy: bool = False
    format_valid: bool = False
    type_matches: int = 0
    type_total: int = 0


# ====================================================================
# Parsing
# ====================================================================

def parse_output(text: str) -> List[SubQuestion]:
    subs = []
    lines = text.strip().split("\n")
    step_num, desc = 0, ""
    for line in lines:
        line = line.strip()
        if not line:
            continue
        m = re.match(r"Step\s+(\d+):\s*(.+)", line)
        if m:
            step_num, desc = int(m.group(1)), m.group(2).strip()
            continue
        m = re.match(r"Sub-question:\s*(.+?)\s*#\s*Type:\s*(\w+)\s*$", line)
        if m:
            subs.append(SubQuestion(step_num, desc, m.group(1).strip(), m.group(2).strip().lower()))
            continue
        m = re.match(r"Sub-question:\s*(.+)", line)
        if m:
            subs.append(SubQuestion(step_num, desc, m.group(1).strip(), "unknown"))
    return subs


def is_format_valid(text: str) -> bool:
    return bool(re.search(r"Step\s+\d+:", text)) and "Sub-question:" in text and "# Type:" in text


# ====================================================================
# OpenAI API Helpers
# ====================================================================

def get_openai_client(api_base=None):
    from openai import OpenAI
    kwargs = {"api_key": OPENAI_API_KEY}
    if api_base:
        kwargs["base_url"] = api_base
    return OpenAI(**kwargs)


def get_embeddings(client, texts: List[str], model=EMBEDDING_MODEL) -> np.ndarray:
    all_embs = []
    for i in range(0, len(texts), EMBEDDING_BATCH_SIZE):
        batch = texts[i:i + EMBEDDING_BATCH_SIZE]
        for attempt in range(MAX_RETRIES):
            try:
                resp = client.embeddings.create(model=model, input=batch)
                all_embs.extend([item.embedding for item in resp.data])
                break
            except Exception as e:
                print(f"    Embedding failed (attempt {attempt+1}): {e}")
                if attempt < MAX_RETRIES - 1:
                    time.sleep(RETRY_DELAY * (attempt + 1))
                else:
                    all_embs.extend([np.zeros(3072).tolist()] * len(batch))
    return np.array(all_embs)


def cosine_sim_matrix(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    A_n = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-10)
    B_n = B / (np.linalg.norm(B, axis=1, keepdims=True) + 1e-10)
    return A_n @ B_n.T


# ====================================================================
# Dim 1: Semantic Similarity (Embedding + Hungarian)
# ====================================================================

def compute_alignment(ref_subs, pred_subs, ref_embs, pred_embs):
    n_r, n_p = len(ref_subs), len(pred_subs)
    if n_p == 0:
        return {"alignment": [], "scores": [], "unmatched_pred": [], "unmatched_ref": list(range(n_r))}
    if n_r == 0:
        return {"alignment": [], "scores": [], "unmatched_pred": list(range(n_p)), "unmatched_ref": []}

    sim = cosine_sim_matrix(pred_embs, ref_embs)
    n = max(n_p, n_r)
    cost = np.ones((n, n))
    cost[:n_p, :n_r] = 1.0 - sim
    ri, ci = linear_sum_assignment(cost)

    alignment, mp, mr = [], set(), set()
    for r, c in zip(ri, ci):
        if r < n_p and c < n_r:
            alignment.append({"pred_idx": int(r), "ref_idx": int(c), "cosine": float(sim[r, c])})
            mp.add(r); mr.add(c)

    return {
        "alignment": alignment,
        "scores": [a["cosine"] for a in alignment],
        "unmatched_pred": [i for i in range(n_p) if i not in mp],
        "unmatched_ref": [i for i in range(n_r) if i not in mr],
    }


# ====================================================================
# Dim 3: Redundancy (Embedding + GPT Verification)
# ====================================================================

def find_redundancy_candidates(pred_subs, pred_embs, threshold):
    n = len(pred_subs)
    if n <= 1:
        return []
    sim = cosine_sim_matrix(pred_embs, pred_embs)
    cands = []
    for i in range(n):
        for j in range(i + 1, n):
            if sim[i, j] >= threshold:
                cands.append({
                    "i": i, "j": j, "cosine": float(sim[i, j]),
                    "sub_i": pred_subs[i].sub_question, "sub_j": pred_subs[j].sub_question,
                })
    return cands


def gpt_verify_redundancy(client, question, candidates, pred_subs, model=GPT_MODEL):
    if not candidates:
        return []

    pairs_desc = "\n".join(
        f"  Pair {k+1}: P{c['i']+1} vs P{c['j']+1} (cosine={c['cosine']:.3f})\n"
        f"    P{c['i']+1}: [{pred_subs[c['i']].sq_type}] {c['sub_i']}\n"
        f"    P{c['j']+1}: [{pred_subs[c['j']].sq_type}] {c['sub_j']}"
        for k, c in enumerate(candidates)
    )
    prompt = f"""Analyze whether these sub-question pairs are truly redundant.

Original question: "{question}"

Candidate pairs (high embedding similarity):
{pairs_desc}

NOT redundant: same biomarker in different models/ratios; same entity different aspects (value vs SHAP); aggregation summarizing prior steps.
IS redundant: same question different wording; one is strict subset of another.

Respond JSON only (no markdown):
{{
  "results": [
    {{"pair": 1, "confirmed": false, "reason": "Different models"}},
    {{"pair": 2, "confirmed": true, "reason": "Duplicate"}}
  ]
}}"""

    for attempt in range(MAX_RETRIES):
        try:
            resp = client.chat.completions.create(
                model=model, messages=[{"role": "user", "content": prompt}],
                temperature=0, max_tokens=1500,
            )
            raw = resp.choices[0].message.content.strip()
            result = json.loads(re.sub(r"```json\s*|\s*```", "", raw).strip())
            verified = []
            for k, c in enumerate(candidates):
                r = result.get("results", [])
                conf = r[k].get("confirmed", False) if k < len(r) else False
                reason = r[k].get("reason", "") if k < len(r) else ""
                verified.append({**c, "confirmed": conf, "reason": reason})
            return verified
        except Exception as e:
            print(f"    GPT verify failed (attempt {attempt+1}): {e}")
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY * (attempt + 1))

    return [{**c, "confirmed": False, "reason": "GPT failed"} for c in candidates]


# ====================================================================
# Post-processing: truncate at first sign of garbage
# ====================================================================

def truncate_output(text: str) -> str:
    """
    Truncate generated output to keep only the first valid response.
    Stops at:
      - Replacement character (�) indicating decode garbage
      - A new user/assistant/system turn (model looping)
      - Any non-UTF8 or control characters
    Then finds the last valid '# Type:' line before that point.
    """
    # Step 1: Cut at first garbage marker
    garbage_patterns = ['�', '<|im_start|>', '\x00']
    cut_pos = len(text)
    for pat in garbage_patterns:
        idx = text.find(pat)
        if idx >= 0 and idx < cut_pos:
            cut_pos = idx

    # Also cut at lines starting with "user\n" or "assistant\n" (sglang artifact)
    user_turn = re.search(r'\nuser\n|\n�user\n|\nassistant\n', text)
    if user_turn and user_turn.start() < cut_pos:
        cut_pos = user_turn.start()

    clean_text = text[:cut_pos]

    # Step 2: Find last valid # Type: line in clean portion
    lines = clean_text.split("\n")
    last_valid_idx = -1

    for i, line in enumerate(lines):
        if re.search(r"#\s*Type:\s*(searching|aggregation)", line.strip()):
            last_valid_idx = i

    if last_valid_idx >= 0:
        return "\n".join(lines[:last_valid_idx + 1]).strip()

    # Fallback
    return clean_text.strip()


# ====================================================================
# Inference via SGLang
# ====================================================================

SGLANG_URL = "http://localhost:30000"
SGLANG_TIMEOUT = 60
SGLANG_PORT = 30000


def launch_sglang_server(model_path, lora_paths=None, port=SGLANG_PORT, dtype="bfloat16"):
    """
    Launch sglang server as a subprocess.
    Returns the process handle (call .terminate() to stop).
    """
    import subprocess

    cmd = [
        "python", "-m", "sglang.launch_server",
        "--model-path", model_path,
        "--port", str(port),
        "--dtype", dtype,
        "--host", "0.0.0.0",
    ]
    if lora_paths:
        cmd.append("--lora-paths")
        for name, path in lora_paths.items():
            cmd.append(f"{name}={path}")

    print(f"  Launching sglang server...")
    print(f"  Command: {' '.join(cmd)}")

    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    # Wait for server to be ready
    import requests
    url = f"http://localhost:{port}/health"
    max_wait = 300  # 5 minutes max
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

        # Check if process died
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


def run_inference_sglang(exp_id, sglang_url=SGLANG_URL, lora_name=None):
    """
    Run inference via sglang's OpenAI-compatible API.
    Much faster than HuggingFace generate (batch scheduling, paged attention, etc).
    """
    import requests

    test_data = json.load(open(TEST_FILE, "r", encoding="utf-8"))
    results = []

    # Determine model name for API call
    # If using LoRA: "default:lora_name", else just "default"
    model_name = f"default:{lora_name}" if lora_name else "default"

    print(f"  SGLang server: {sglang_url}")
    print(f"  Model: {model_name}")
    print(f"  Test samples: {len(test_data)}")

    # Check server health
    try:
        health = requests.get(f"{sglang_url}/health", timeout=5)
        print(f"  Server status: {'OK' if health.status_code == 200 else 'ERROR'}")
    except Exception as e:
        print(f"  WARNING: Cannot reach sglang server at {sglang_url}: {e}")
        print(f"  Please start it first. See instructions in evaluate.py.")
        return []

    # Send requests one by one (simple & reliable)
    # For even more speed, can batch with async requests
    print(f"\n  Generating...")
    t_start = time.time()

    for i, sample in enumerate(test_data):
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": sample["input"]},
        ]

        payload = {
            "model": model_name,
            "messages": messages,
            "max_tokens": 512,
            "temperature": 0,
            "top_p": 1.0,
            "chat_template_kwargs": {"enable_thinking": False},
        }

        for attempt in range(MAX_RETRIES):
            try:
                resp = requests.post(
                    f"{sglang_url}/v1/chat/completions",
                    json=payload,
                    timeout=SGLANG_TIMEOUT,
                )
                resp.raise_for_status()
                data = resp.json()
                pred_raw = data["choices"][0]["message"]["content"]
                pred = truncate_output(pred_raw)
                break
            except Exception as e:
                print(f"    [{i+1}] attempt {attempt+1} failed: {e}")
                if attempt < MAX_RETRIES - 1:
                    time.sleep(2)
                else:
                    pred = ""

        results.append({
            "input": sample["input"],
            "reference": sample["output"],
            "prediction": pred,
        })

        if (i + 1) % 10 == 0 or (i + 1) == len(test_data):
            elapsed = time.time() - t_start
            speed = (i + 1) / elapsed
            eta = (len(test_data) - i - 1) / speed if speed > 0 else 0
            print(f"    [{i+1}/{len(test_data)}] {speed:.1f} samples/s, ETA {eta:.0f}s")

    elapsed = time.time() - t_start
    print(f"\n  Done! {len(results)} samples in {elapsed:.1f}s ({len(results)/elapsed:.1f} samples/s)")

    return results


def run_inference_sglang_batch(exp_id, sglang_url=SGLANG_URL, lora_name=None):
    """
    Batch inference via sglang - send all requests concurrently for maximum throughput.
    """
    import concurrent.futures
    import requests

    test_data = json.load(open(TEST_FILE, "r", encoding="utf-8"))
    model_name = f"default:{lora_name}" if lora_name else "default"

    print(f"  SGLang batch inference: {len(test_data)} samples")

    def infer_one(idx_sample):
        idx, sample = idx_sample
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": sample["input"]},
        ]
        payload = {
            "model": model_name,
            "messages": messages,
            "max_tokens": 512,
            "temperature": 0,
            "top_p": 1.0,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        for attempt in range(MAX_RETRIES):
            try:
                resp = requests.post(
                    f"{sglang_url}/v1/chat/completions",
                    json=payload,
                    timeout=SGLANG_TIMEOUT,
                )
                resp.raise_for_status()
                pred_raw = resp.json()["choices"][0]["message"]["content"]
                return idx, truncate_output(pred_raw)
            except Exception:
                if attempt < MAX_RETRIES - 1:
                    time.sleep(1)
        return idx, ""

    t_start = time.time()
    results = [None] * len(test_data)

    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
        futures = executor.map(infer_one, enumerate(test_data))
        for idx, pred in futures:
            results[idx] = {
                "input": test_data[idx]["input"],
                "reference": test_data[idx]["output"],
                "prediction": pred,
            }

    elapsed = time.time() - t_start
    print(f"  Done! {len(results)} samples in {elapsed:.1f}s ({len(results)/elapsed:.1f} samples/s)")

    return results


# ====================================================================
# Evaluation Pipeline
# ====================================================================

def evaluate_experiment(exp_id, predictions, client, emb_model, gpt_model):
    print(f"\n{'='*70}")
    print(f"  Evaluating Experiment {exp_id} ({len(predictions)} samples)")
    print(f"{'='*70}")

    # Parse
    evals = []
    for p in predictions:
        se = SampleEval(input_text=p["input"], ref_output=p["reference"], pred_output=p["prediction"])
        se.ref_subs = parse_output(p["reference"])
        se.pred_subs = parse_output(p["prediction"])
        se.count_ref = len(se.ref_subs)
        se.count_pred = len(se.pred_subs)
        se.count_match = se.count_ref == se.count_pred
        se.format_valid = is_format_valid(p["prediction"])
        ml = min(se.count_ref, se.count_pred)
        se.type_total = ml
        se.type_matches = sum(1 for j in range(ml) if se.ref_subs[j].sq_type == se.pred_subs[j].sq_type)
        evals.append(se)

    # Batch embed
    print(f"  Embedding sub-questions ({emb_model})...")
    texts, t2i = [], {}
    for se in evals:
        for sq in se.ref_subs + se.pred_subs:
            if sq.sub_question not in t2i:
                t2i[sq.sub_question] = len(texts)
                texts.append(sq.sub_question)
    print(f"    Unique: {len(texts)}")

    embs = get_embeddings(client, texts, model=emb_model) if texts else np.zeros((0, 3072))
    get_e = lambda t: embs[t2i[t]] if t in t2i else np.zeros(embs.shape[1])

    # Semantic alignment
    print(f"  Semantic alignment (Hungarian)...")
    for se in evals:
        if se.count_pred == 0 or se.count_ref == 0:
            se.semantic_avg, se.semantic_scores = 0.0, []
            se.unmatched_pred = list(range(se.count_pred))
            se.unmatched_ref = list(range(se.count_ref))
            continue
        r_e = np.stack([get_e(sq.sub_question) for sq in se.ref_subs])
        p_e = np.stack([get_e(sq.sub_question) for sq in se.pred_subs])
        res = compute_alignment(se.ref_subs, se.pred_subs, r_e, p_e)
        se.alignment = res["alignment"]
        se.semantic_scores = res["scores"]
        se.semantic_avg = float(np.mean(res["scores"])) if res["scores"] else 0.0
        se.unmatched_pred = res["unmatched_pred"]
        se.unmatched_ref = res["unmatched_ref"]

    # Redundancy
    print(f"  Redundancy detection (threshold={REDUNDANCY_SIM_THRESHOLD})...")
    gpt_calls = 0
    for se in evals:
        if se.count_pred <= 1:
            se.has_redundancy, se.redundancy_count = False, 0
            continue
        p_e = np.stack([get_e(sq.sub_question) for sq in se.pred_subs])
        cands = find_redundancy_candidates(se.pred_subs, p_e, REDUNDANCY_SIM_THRESHOLD)
        if cands:
            gpt_calls += 1
            verified = gpt_verify_redundancy(client, se.input_text, cands, se.pred_subs, model=gpt_model)
            se.redundant_pairs = [v for v in verified if v.get("confirmed")]
            se.redundancy_count = len(se.redundant_pairs)
            se.has_redundancy = se.redundancy_count > 0
        else:
            se.has_redundancy, se.redundancy_count = False, 0
    print(f"    GPT verification calls: {gpt_calls}")

    metrics = aggregate_metrics(evals)
    return evals, metrics


def aggregate_metrics(evals):
    n = len(evals)
    all_sc = [s for e in evals for s in e.semantic_scores]
    sa = [e.semantic_avg for e in evals if e.count_pred > 0]
    cd = [e.count_pred - e.count_ref for e in evals]

    return {
        "n_samples": n,
        # Dim 1
        "semantic_mean": float(np.mean(all_sc)) if all_sc else 0,
        "semantic_std": float(np.std(all_sc)) if all_sc else 0,
        "semantic_sample_mean": float(np.mean(sa)) if sa else 0,
        "semantic_sample_std": float(np.std(sa)) if sa else 0,
        "semantic_perfect_rate": sum(1 for s in all_sc if s >= PERFECT_SIM_THRESHOLD) / max(len(all_sc), 1) * 100,
        "semantic_high_rate": sum(1 for s in all_sc if s >= HIGH_SIM_THRESHOLD) / max(len(all_sc), 1) * 100,
        "semantic_low_count": sum(1 for s in all_sc if s < 0.6),
        # Dim 2
        "count_exact_match": sum(1 for e in evals if e.count_match),
        "count_exact_rate": sum(1 for e in evals if e.count_match) / n * 100,
        "count_avg_ref": float(np.mean([e.count_ref for e in evals])),
        "count_avg_pred": float(np.mean([e.count_pred for e in evals])),
        "count_mean_diff": float(np.mean(cd)),
        "count_abs_diff": float(np.mean(np.abs(cd))),
        "count_over": sum(1 for d in cd if d > 0),
        "count_under": sum(1 for d in cd if d < 0),
        # Dim 3
        "redundancy_samples": sum(1 for e in evals if e.has_redundancy),
        "redundancy_rate": sum(1 for e in evals if e.has_redundancy) / n * 100,
        "redundancy_total": sum(e.redundancy_count for e in evals),
        "redundancy_per_sample": sum(e.redundancy_count for e in evals) / n,
        # Supplementary
        "format_valid": sum(1 for e in evals if e.format_valid),
        "format_rate": sum(1 for e in evals if e.format_valid) / n * 100,
        "type_accuracy": (sum(e.type_matches for e in evals) / max(sum(e.type_total for e in evals), 1) * 100),
    }


# ====================================================================
# Output
# ====================================================================

def print_metrics(m, exp_id):
    print(f"\n{'='*70}")
    print(f"  EXPERIMENT {exp_id} RESULTS")
    print(f"{'='*70}")
    print(f"\n  ┌─ Dim 1: Semantic Similarity (cosine, {EMBEDDING_MODEL})")
    print(f"  │  Per sub-Q mean:         {m['semantic_mean']:.4f} ± {m['semantic_std']:.4f}")
    print(f"  │  Per sample mean:        {m['semantic_sample_mean']:.4f} ± {m['semantic_sample_std']:.4f}")
    print(f"  │  ≥{PERFECT_SIM_THRESHOLD} (near-perfect):  {m['semantic_perfect_rate']:.1f}%")
    print(f"  │  ≥{HIGH_SIM_THRESHOLD} (high match):    {m['semantic_high_rate']:.1f}%")
    print(f"  │  <0.6 (low):             {m['semantic_low_count']}")
    print(f"  ├─ Dim 2: Count Accuracy")
    print(f"  │  Exact match:            {m['count_exact_match']}/{m['n_samples']} ({m['count_exact_rate']:.1f}%)")
    print(f"  │  Avg steps ref/pred:     {m['count_avg_ref']:.2f} / {m['count_avg_pred']:.2f}")
    print(f"  │  Mean diff:              {m['count_mean_diff']:+.2f}  |abs|={m['count_abs_diff']:.2f}")
    print(f"  │  Over/Under:             {m['count_over']} / {m['count_under']}")
    print(f"  ├─ Dim 3: Redundancy")
    print(f"  │  Samples w/ redundancy:  {m['redundancy_samples']}/{m['n_samples']} ({m['redundancy_rate']:.1f}%)")
    print(f"  │  Total pairs:            {m['redundancy_total']}  (avg {m['redundancy_per_sample']:.3f}/sample)")
    print(f"  └─ Supplementary")
    print(f"     Format valid:           {m['format_valid']}/{m['n_samples']} ({m['format_rate']:.1f}%)")
    print(f"     Type accuracy (S/A):    {m['type_accuracy']:.1f}%")
    print(f"{'='*70}")


def print_comparison(all_m):
    exps = sorted(all_m.keys())
    print(f"\n{'='*90}")
    print(f"  CROSS-EXPERIMENT COMPARISON")
    print(f"{'='*90}")
    h = f"  {'Metric':<35s}" + "".join(f"{'Exp'+str(e):>10s}" for e in exps)
    print(h); print("  " + "-" * (35 + 10 * len(exps)))

    rows = [
        ("Semantic Mean Cosine",      "semantic_sample_mean", ".4f", True),
        ("Semantic ≥0.85 (%)",        "semantic_high_rate",   ".1f", True),
        ("Semantic ≥0.92 (%)",        "semantic_perfect_rate",".1f", True),
        ("Count Exact Match (%)",     "count_exact_rate",     ".1f", True),
        ("Count |Diff|",              "count_abs_diff",       ".2f", False),
        ("Redundancy Rate (%)",       "redundancy_rate",      ".1f", False),
        ("Format Valid (%)",          "format_rate",          ".1f", True),
        ("Type Accuracy (%)",         "type_accuracy",        ".1f", True),
    ]
    for label, key, fmt, hb in rows:
        vals = [all_m[e].get(key, 0) for e in exps]
        best = max(vals) if hb else min(vals)
        row = f"  {label:<35s}"
        for e in exps:
            v = all_m[e].get(key, 0)
            s = f"*{v:{fmt}}" if v == best and len(exps) > 1 else f"{v:{fmt}}"
            row += f"{s:>10s}"
        print(row)
    print(f"\n  * = best")
    print(f"{'='*90}")


def save_results(exp_id, evals, metrics, predictions, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    detailed = [{
        "input": se.input_text, "reference": se.ref_output, "prediction": se.pred_output,
        "count_ref": se.count_ref, "count_pred": se.count_pred, "count_match": se.count_match,
        "alignment": se.alignment, "semantic_scores": se.semantic_scores, "semantic_avg": se.semantic_avg,
        "unmatched_pred": se.unmatched_pred, "unmatched_ref": se.unmatched_ref,
        "has_redundancy": se.has_redundancy, "redundancy_count": se.redundancy_count,
        "redundant_pairs": se.redundant_pairs,
        "format_valid": se.format_valid, "type_matches": se.type_matches, "type_total": se.type_total,
    } for se in evals]

    with open(os.path.join(out_dir, "detailed_results.json"), "w", encoding="utf-8") as f:
        json.dump(detailed, f, indent=2, ensure_ascii=False)
    print(f"  Saved to {out_dir}/")


# ====================================================================
# Main
# ====================================================================

def main():
    parser = argparse.ArgumentParser(description="Inference & Evaluation")
    parser.add_argument("--exp", type=int, nargs="+", required=True, help="Experiment IDs (1-5)")
    parser.add_argument("--inference_only", action="store_true", help="Only run inference, skip evaluation")
    parser.add_argument("--skip_inference", action="store_true", help="Use existing predictions, only evaluate")
    parser.add_argument("--pred_file", type=str, default=None, help="Custom predictions file")

    # SGLang options
    parser.add_argument("--model_path", type=str, default=MODEL_NAME, help="Base model path")
    parser.add_argument("--sglang_url", type=str, default=None,
                        help="If set, use an already-running sglang server instead of launching one")
    parser.add_argument("--port", type=int, default=SGLANG_PORT, help="Port for sglang server")
    parser.add_argument("--batch", action="store_true", help="Use concurrent batch inference (faster)")

    # Evaluation options
    parser.add_argument("--embedding_model", type=str, default=EMBEDDING_MODEL)
    parser.add_argument("--gpt_model", type=str, default=GPT_MODEL)
    parser.add_argument("--api_base", type=str, default=OPENAI_API_BASE, help="OpenAI API base URL")
    parser.add_argument("--redundancy_threshold", type=float, default=0.90)
    parser.add_argument("--no_truncate", action="store_true", help="Evaluate raw output without truncation")
    parser.add_argument("--both", action="store_true", help="Evaluate both raw and truncated, save both metrics")
    args = parser.parse_args()

    REDUNDANCY_SIM_THRESHOLD = args.redundancy_threshold

    # Determine which experiments need inference
    needs_inference = []
    if not args.skip_inference and not args.pred_file:
        for exp_id in args.exp:
            pred_path = os.path.join(OUTPUT_ROOT, f"exp{exp_id}", "test_predictions.json")
            if not os.path.exists(pred_path):
                needs_inference.append(exp_id)
            else:
                needs_inference.append(exp_id)  # re-generate

    # Auto-launch sglang server if needed
    sglang_process = None
    server_url = args.sglang_url or f"http://localhost:{args.port}"

    try:
        all_metrics = {}

        for exp_id in args.exp:
            assert exp_id in [1, 2, 3, 4, 5, 6]
            exp_dir = os.path.join(OUTPUT_ROOT, f"exp{exp_id}")
            os.makedirs(exp_dir, exist_ok=True)
            pred_path = os.path.join(exp_dir, "test_predictions.json")

            # --- Inference via SGLang ---
            if args.pred_file:
                predictions = json.load(open(args.pred_file, "r", encoding="utf-8"))
            elif args.skip_inference and os.path.exists(pred_path):
                print(f"\n  Loading existing predictions: {pred_path}")
                predictions = json.load(open(pred_path, "r", encoding="utf-8"))
            else:
                merged_dir = os.path.join(OUTPUT_ROOT, f"exp{exp_id}", "merged_model")
                if not os.path.exists(merged_dir):
                    print(f"\n  ERROR: Merged model not found at {merged_dir}")
                    print(f"  Run first: python merge_lora.py --exp {exp_id}")
                    continue

                # Launch sglang for this experiment
                if not args.sglang_url:
                    # Shutdown previous server if any
                    if sglang_process:
                        shutdown_sglang_server(sglang_process)
                        sglang_process = None

                    print(f"\n  Launching sglang for Exp{exp_id} (merged model)...")
                    sglang_process = launch_sglang_server(
                        model_path=merged_dir,
                        port=args.port,
                    )
                    if sglang_process is None:
                        print(f"  ERROR: Failed to launch sglang for Exp{exp_id}.")
                        continue

                print(f"\n  Running SGLang inference for Exp{exp_id}...")
                # No lora_name needed for merged model
                if args.batch:
                    predictions = run_inference_sglang_batch(exp_id, server_url, lora_name=None)
                else:
                    predictions = run_inference_sglang(exp_id, server_url, lora_name=None)

                if predictions:
                    with open(pred_path, "w", encoding="utf-8") as f:
                        json.dump(predictions, f, indent=2, ensure_ascii=False)
                    print(f"  Predictions saved: {pred_path}")
                else:
                    print(f"  ERROR: No predictions generated.")
                    continue

            # Re-truncate predictions (in case old truncate was buggy)
            # Keep raw predictions for --no_truncate or --both modes
            raw_predictions = [dict(p) for p in predictions]  # deep copy
            truncated_predictions = []
            for p in predictions:
                tp = dict(p)
                tp["prediction"] = truncate_output(p["prediction"])
                truncated_predictions.append(tp)

            # --- Inference only: stop here ---
            if args.inference_only:
                print(f"  Inference done. Predictions at: {pred_path}")
                continue

            # --- Evaluate ---
            client = get_openai_client(args.api_base)

            if args.both:
                # Evaluate BOTH raw and truncated
                print(f"\n  === Raw (no truncation) ===")
                evals_raw, metrics_raw = evaluate_experiment(
                    exp_id, raw_predictions, client,
                    emb_model=args.embedding_model, gpt_model=args.gpt_model,
                )
                print_metrics(metrics_raw, exp_id)
                save_results(exp_id, evals_raw, metrics_raw, raw_predictions,
                             os.path.join(exp_dir, "raw"))

                print(f"\n  === Truncated ===")
                evals_trunc, metrics_trunc = evaluate_experiment(
                    exp_id, truncated_predictions, client,
                    emb_model=args.embedding_model, gpt_model=args.gpt_model,
                )
                print_metrics(metrics_trunc, exp_id)
                save_results(exp_id, evals_trunc, metrics_trunc, truncated_predictions,
                             os.path.join(exp_dir, "truncated"))

                # Save combined comparison
                combined = {"raw": metrics_raw, "truncated": metrics_trunc}
                with open(os.path.join(exp_dir, "metrics_both.json"), "w") as f:
                    json.dump(combined, f, indent=2)

                all_metrics[exp_id] = metrics_trunc  # use truncated for cross-exp comparison

            elif args.no_truncate:
                # Evaluate raw only
                evals, metrics = evaluate_experiment(
                    exp_id, raw_predictions, client,
                    emb_model=args.embedding_model, gpt_model=args.gpt_model,
                )
                print_metrics(metrics, exp_id)
                save_results(exp_id, evals, metrics, raw_predictions, exp_dir)
                all_metrics[exp_id] = metrics

            else:
                # Default: evaluate truncated
                evals, metrics = evaluate_experiment(
                    exp_id, truncated_predictions, client,
                    emb_model=args.embedding_model, gpt_model=args.gpt_model,
                )
                print_metrics(metrics, exp_id)
                save_results(exp_id, evals, metrics, truncated_predictions, exp_dir)
                all_metrics[exp_id] = metrics

        # --- Comparison ---
        if len(args.exp) > 1 and not args.inference_only and all_metrics:
            print_comparison(all_metrics)
            with open(os.path.join(OUTPUT_ROOT, "comparison.json"), "w") as f:
                json.dump({f"exp{k}": v for k, v in all_metrics.items()}, f, indent=2)

    finally:
        # Always cleanup sglang server
        if sglang_process:
            shutdown_sglang_server(sglang_process)


if __name__ == "__main__":
    main()