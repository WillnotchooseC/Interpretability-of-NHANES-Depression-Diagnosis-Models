"""
Score the exp1/exp2/exp3 GraphRAG-ablation outputs (results_exp{1,2,3}_.../progressive_decomp_for_experiments_exp*.json)
against ood_data/curriculum_test_120_gold.json on two axes:
  - graph-info coverage: value-set containment of all retrieved rows (pooled
    across a sample's searching steps) vs gold_results, reusing
    run_finetuned_pipeline_eval.score_against_gold (same methodology as the
    earlier run_progressive_pipeline_compare_v2.py scoring).
  - answer semantic similarity: text-embedding-3-large cosine of final_answer
    vs gold_answer.

Usage:
    python score_experiments.py
"""
import json
import numpy as np
from neo4j import GraphDatabase

from entity_retrieval import build_vocab
from run_finetuned_pipeline_eval import score_against_gold, build_echo_values
from evaluate import get_openai_client, get_embeddings, EMBEDDING_MODEL

NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "12345678"

GOLD_FILE = "ood_data/curriculum_test_120_gold.json"
EXPERIMENTS = {
    "exp1_baseline_fewshot": "results_exp1_baseline_fewshot/progressive_decomp_for_experiments_exp1_baseline_fewshot.json",
    "exp2_fewshot_graphrag": "results_exp2_fewshot_graphrag/progressive_decomp_for_experiments_exp2_fewshot_graphrag.json",
    "exp3_fewshot_graphrag_rescue": "results_exp3_fewshot_graphrag_rescue/progressive_decomp_for_experiments_exp3_fewshot_graphrag_rescue.json",
}

UNSEEN_IDX = set([10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89,
                  100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114,
                  115, 116, 117, 118, 119])


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
    gold = json.load(open(GOLD_FILE, encoding="utf-8"))
    n = len(gold)

    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    driver.verify_connectivity()
    vocab = build_vocab(driver)

    client = get_openai_client()
    gold_answers = [g.get("gold_answer", "") for g in gold]
    gold_emb = get_embeddings(client, gold_answers, model=EMBEDDING_MODEL)

    per_exp_rows = {}
    summaries = {}

    for exp_name, path in EXPERIMENTS.items():
        data = json.load(open(path, encoding="utf-8"))
        details = data["details"]
        assert len(details) == n, f"{exp_name}: {len(details)} details vs {n} gold samples"

        # A truly empty final_answer (failed generation) must still score near-
        # zero cosine against gold, but the OpenAI embeddings endpoint rejects
        # an empty string for ANY item in a batch request -- which after
        # MAX_RETRIES silently zeroes out the WHOLE batch (see evaluate.get_embeddings),
        # corrupting every other sample's score too. Substitute a placeholder
        # so only the genuinely-empty sample gets a low score.
        final_answers = [d.get("final_answer") or "" for d in details]
        empty_idx = [i for i, a in enumerate(final_answers) if not a.strip()]
        if empty_idx:
            print(f"  {exp_name}: {len(empty_idx)} empty final_answer(s) at {empty_idx} -> using placeholder for embedding")
        embed_inputs = [a if a.strip() else "(no answer was generated)" for a in final_answers]
        ans_emb = get_embeddings(client, embed_inputs, model=EMBEDDING_MODEL)

        rows = []
        for i, (g, d) in enumerate(zip(gold, details)):
            actual_rows = []
            for step in d.get("steps", []):
                actual_rows.extend(step.get("records") or [])
            gold_rows = gold_rows_for(g)
            echoes = build_echo_values(g["input"], vocab)
            sc = score_against_gold(actual_rows, gold_rows, echoes)
            asim = cos(ans_emb[i], gold_emb[i])
            rows.append({
                "index": i,
                "input": g["input"],
                "gold_answer": g.get("gold_answer", ""),
                "final_answer": final_answers[i],
                "graph_coverage": sc["coverage"],
                "answer_cosine": round(asim, 4),
                "n_searching_subquestions": d.get("num_searching_subquestions", 0),
            })
        per_exp_rows[exp_name] = rows

        def agg(idx_filter):
            sub = [r for r in rows if idx_filter(r["index"])]
            cov = [r["graph_coverage"] for r in sub if r["graph_coverage"] is not None]
            cos_ = [r["answer_cosine"] for r in sub]
            return {
                "n": len(sub),
                "graph_coverage_mean": round(float(np.mean(cov)), 4) if cov else None,
                "graph_coverage_n": len(cov),
                "answer_cosine_mean": round(float(np.mean(cos_)), 4) if cos_ else None,
            }

        summaries[exp_name] = {
            "all": agg(lambda i: True),
            "seen_80": agg(lambda i: i not in UNSEEN_IDX),
            "unseen_40": agg(lambda i: i in UNSEEN_IDX),
        }

    driver.close()

    json.dump(summaries, open("experiments_score_summary.json", "w"), indent=2, ensure_ascii=False)
    json.dump(per_exp_rows, open("experiments_score_per_sample.json", "w"), indent=2, ensure_ascii=False, default=str)

    print("\n" + "=" * 90)
    print(f"{'experiment':<32}{'subset':<12}{'n':>5}{'graph_coverage':>18}{'answer_cosine':>16}")
    print("-" * 90)
    for exp_name, s in summaries.items():
        for subset in ["all", "seen_80", "unseen_40"]:
            v = s[subset]
            print(f"{exp_name:<32}{subset:<12}{v['n']:>5}{str(v['graph_coverage_mean']):>18}{str(v['answer_cosine_mean']):>16}")
    print("\nSaved -> experiments_score_summary.json + experiments_score_per_sample.json")


if __name__ == "__main__":
    main()
