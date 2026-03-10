"""
Calculate hit rate and execution rate for exp1, exp5, exp6.

Two-step funnel:
1. Hit Rate: how many reference searching sub-Qs are matched by a UNIQUE
   prediction sub-Q (cosine >= threshold, 1-to-1 Hungarian matching).
   Denominator = # ref searching sub-Qs.
2. Exec Rate: among the hit pred sub-Qs, how many have successful Cypher
   execution (has_results=true). Denominator = # hits.

Key: prediction sub-Qs are deduplicated first, and each pred sub-Q can only
match one ref sub-Q (no double counting).
"""

import json, re, time
import numpy as np
from dataclasses import dataclass
from typing import List
from scipy.optimize import linear_sum_assignment
from openai import OpenAI

EMBEDDING_MODEL = "text-embedding-3-large"
HIT_THRESHOLD = 0.90
BATCH_SIZE = 256

@dataclass
class SubQuestion:
    step: int
    description: str
    sub_question: str
    sq_type: str

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

def get_embeddings(client, texts, model=EMBEDDING_MODEL):
    if not texts:
        return np.array([])
    all_embs = []
    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i:i+BATCH_SIZE]
        for attempt in range(3):
            try:
                resp = client.embeddings.create(model=model, input=batch)
                all_embs.extend([item.embedding for item in resp.data])
                break
            except Exception as e:
                print(f"    Embedding failed (attempt {attempt+1}): {e}")
                if attempt < 2:
                    time.sleep(2 * (attempt + 1))
                else:
                    all_embs.extend([np.zeros(3072).tolist()] * len(batch))
    return np.array(all_embs)

def cosine_sim_matrix(A, B):
    A_n = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-10)
    B_n = B / (np.linalg.norm(B, axis=1, keepdims=True) + 1e-10)
    return A_n @ B_n.T

def dedup_pred_queries(pred_queries):
    """Deduplicate prediction queries by question text, keep first occurrence."""
    seen = set()
    unique = []
    for q in pred_queries:
        if q["question"] not in seen:
            seen.add(q["question"])
            unique.append(q)
    return unique

def main():
    client = OpenAI()

    experiments = {
        "Exp1": ("nhanes_model/exp1/test_predictions.json",
                 "cypher_results/exp1_test_predictions_cypher_eval.json"),
        "Exp5": ("nhanes_model/exp5/test_predictions.json",
                 "cypher_results/exp5_test_predictions_cypher_eval.json"),
        "Exp6": ("nhanes_model/exp6/test_predictions.json",
                 "cypher_results/exp6_test_predictions_cypher_eval.json"),
    }

    print(f"Two-step Funnel: Hit Rate → Exec Rate")
    print(f"Hit = cosine ≥ {HIT_THRESHOLD} | 1-to-1 Hungarian | dedup pred sub-Qs")
    print(f"Denominator = ref searching sub-Qs")
    print("=" * 70)

    all_results = {}

    for exp_name, (pred_path, cypher_path) in experiments.items():
        with open(pred_path) as f:
            predictions = json.load(f)
        with open(cypher_path) as f:
            cypher_eval = json.load(f)

        # Collect unique texts for batch embedding
        text_to_idx = {}
        texts = []

        def register(t):
            if t not in text_to_idx:
                text_to_idx[t] = len(texts)
                texts.append(t)

        sample_data = []
        total_pred_before_dedup = 0
        total_pred_after_dedup = 0

        for i, item in enumerate(predictions):
            ref_subs = [sq for sq in parse_output(item["reference"]) if sq.sq_type == "searching"]
            for sq in ref_subs:
                register(sq.sub_question)

            pred_queries_raw = cypher_eval["details"][i]["queries"]
            total_pred_before_dedup += len(pred_queries_raw)
            pred_queries = dedup_pred_queries(pred_queries_raw)
            total_pred_after_dedup += len(pred_queries)
            for q in pred_queries:
                register(q["question"])

            sample_data.append((ref_subs, pred_queries))

        print(f"\n{exp_name}: {len(predictions)} samples, {len(texts)} unique texts")
        print(f"  Pred sub-Qs: {total_pred_before_dedup} → {total_pred_after_dedup} after dedup")
        print(f"  Fetching embeddings...")
        embs = get_embeddings(client, texts)

        total_ref_searching = 0
        total_hit = 0
        total_hit_and_exec = 0

        for ref_subs, pred_queries in sample_data:
            if not ref_subs:
                continue

            n_ref = len(ref_subs)
            total_ref_searching += n_ref

            if not pred_queries:
                continue

            n_pred = len(pred_queries)
            ref_embs = np.stack([embs[text_to_idx[sq.sub_question]] for sq in ref_subs])
            pred_embs = np.stack([embs[text_to_idx[q["question"]]] for q in pred_queries])

            sim = cosine_sim_matrix(ref_embs, pred_embs)  # (n_ref, n_pred)

            # Hungarian 1-to-1 matching (minimize cost = 1 - sim)
            n = max(n_ref, n_pred)
            cost = np.ones((n, n))  # default cost = 1 (sim = 0)
            cost[:n_ref, :n_pred] = 1.0 - sim
            row_ind, col_ind = linear_sum_assignment(cost)

            for r, c in zip(row_ind, col_ind):
                if r < n_ref and c < n_pred:
                    match_sim = sim[r, c]
                    if match_sim >= HIT_THRESHOLD:
                        total_hit += 1
                        if pred_queries[c]["has_results"]:
                            total_hit_and_exec += 1

        hit_rate = total_hit / total_ref_searching if total_ref_searching > 0 else 0
        exec_rate = total_hit_and_exec / total_hit if total_hit > 0 else 0

        all_results[exp_name] = {
            "total_ref_searching": total_ref_searching,
            "total_hit": total_hit,
            "total_hit_and_exec": total_hit_and_exec,
            "hit_rate": hit_rate,
            "exec_rate": exec_rate,
        }

        print(f"  Ref searching sub-Qs:    {total_ref_searching}")
        print(f"  Hit (cosine ≥ {HIT_THRESHOLD}):      {total_hit}")
        print(f"  Hit + Exec:              {total_hit_and_exec}")
        print(f"  ───────────────────────────────────")
        print(f"  Hit Rate:                {hit_rate*100:.1f}%  ({total_hit}/{total_ref_searching})")
        print(f"  Exec Rate (among hits):  {exec_rate*100:.1f}%  ({total_hit_and_exec}/{total_hit})")

    # Summary table
    print("\n" + "=" * 70)
    print(f"{'Exp':<6} {'Ref(S)':<8} {'Hit':<6} {'Hit+Exec':<10} "
          f"{'Hit Rate':<12} {'Exec/Hit':<12}")
    print("-" * 70)
    for exp_name, r in all_results.items():
        print(f"{exp_name:<6} {r['total_ref_searching']:<8} {r['total_hit']:<6} "
              f"{r['total_hit_and_exec']:<10} "
              f"{r['hit_rate']*100:>5.1f}%{'':>5} "
              f"{r['exec_rate']*100:>5.1f}%")
    print("=" * 70)

if __name__ == "__main__":
    main()
