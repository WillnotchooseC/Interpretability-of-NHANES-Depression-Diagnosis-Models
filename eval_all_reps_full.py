"""
3-repetition statistical evaluation of the exp1_5 / exp2 / exp3 GraphRAG
ablation on ood_data/curriculum_test_120_gold.json.

exp1_5 = experiment1_5_baseline_original_input (few-shot that DOES see the
original question), exp2 = +GraphRAG grounding, exp3 = +neighborhood rescue.

Reps:  rep1 = results_<full>/           (pre-existing first run)
       rep2 = results_<full>_run2/
       rep3 = results_<full>_run3/
Decomp input for every rep: ood_data/progressive_decomp_for_experiments.json.

Four metrics, each broken out overall / seen_80 / unseen_40:
  1. Cypher execution failure   (per step;  fail = not exec_success)
  2. Gold recall                (per gold row; strict full-match via
                                 run_finetuned_pipeline_eval.score_against_gold)
  3. Irrelevant grounding       (per grounding row; GPT-4o judge; exp2/exp3 only)
  4. Final-answer correct       (per sample;  GPT-4o judge)

For each metric x split x method:
  - 3-rep POOLED point estimate + 95% CI (Wilson for proportions,
    bootstrap for the soft-coverage mean)
  - per-rep values + across-rep mean +/- std  (run-to-run stability)
Pairwise method comparison (exp1_5-vs-exp2, exp1_5-vs-exp3, exp2-vs-exp3),
per split:
  - rep-averaged per-sample paired vectors -> Wilcoxon signed-rank + Cohen's d
  - the same paired test re-run WITHIN each rep -> 3 p-values, so we can see
    whether the verdict is stable across reps
  - McNemar (rep-majority) for the binary final-answer-correct metric
  - irrelevant grounding (exp2 vs exp3): pooled 2-proportion test on rows

GPT-4o judge calls are cached by content hash (question+gold+answer, or
question+entities+rows) in cypher_results/all_reps_judge_cache.json, so
identical answers across reps are judged once.

Usage:
    python eval_all_reps_full.py            # all 120, workers=6
    python eval_all_reps_full.py --limit 10
"""
import argparse
import concurrent.futures
import hashlib
import json
import math
import os
import re
import threading
import time

import numpy as np
from scipy import stats
from neo4j import GraphDatabase

from entity_retrieval import build_vocab
from run_finetuned_pipeline_eval import score_against_gold, build_echo_values
from evaluate import get_openai_client, GPT_MODEL

NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "12345678"
GOLD_FILE = "ood_data/curriculum_test_120_gold.json"
INPUT_BASENAME = "progressive_decomp_for_experiments"
CACHE_FILE = "cypher_results/all_reps_judge_cache.json"

METHODS = {  # short label -> EXPERIMENT_NAME (== results dir stem & file suffix)
    "exp1_5": "exp1_5_baseline_original_input",
    "exp2":   "exp2_fewshot_graphrag",
    "exp3":   "exp3_fewshot_graphrag_rescue",
}
HAS_GROUNDING = {"exp1_5": False, "exp2": True, "exp3": True}
PAIRS = [("exp1_5", "exp2"), ("exp1_5", "exp3"), ("exp2", "exp3")]
REPS = [1, 2, 3]

UNSEEN_IDX = set([10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89,
                  100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114,
                  115, 116, 117, 118, 119])
SPLITS = {"overall": lambda i: True,
          "seen_80": lambda i: i not in UNSEEN_IDX,
          "unseen_40": lambda i: i in UNSEEN_IDX}

BIOMARKER_ALIASES = {"25-hydroxyvitamin d3": "vitamin d3", "25-hydroxyvitamin d2": "vitamin d2"}

ANSWER_JUDGE_PROMPT = """You are grading a knowledge-graph QA system's final answer against a gold answer.

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
- "score": overall answer quality 0.0-1.0 (1.0 = fully correct, complete, clean).

Respond JSON only:
{{"correct": true, "complete": true, "no_hallucination": true, "score": 1.0}}"""

GROUNDING_JUDGE_PROMPT = """You are auditing a GraphRAG retrieval step for irrelevant context.

Sub-question: "{question}"

The system extracted these entities from the sub-question: {entities}

Before generating Cypher, it looked up these EXACT grounding facts from the
knowledge graph (real property values / relationship directions for the
extracted entities):
{rows}

For EACH fact (by index), decide "relevant": true if it is about an entity,
attribute, or relation that this sub-question actually asks about (even if
just used to disambiguate an id/name/direction). false if it is about a
different entity/attribute the sub-question never asked about - i.e. noise
that doesn't belong in this step's Cypher-generation context.

Respond JSON only, no markdown:
{{"results": [{{"idx": 0, "relevant": true}}]}}"""


# ---------------------------------------------------------------- helpers
def _canon_row(row):
    if not isinstance(row, dict):
        return row
    out = {}
    for k, v in row.items():
        if isinstance(v, str) and v.strip().lower() in BIOMARKER_ALIASES:
            v = BIOMARKER_ALIASES[v.strip().lower()]
        out[k] = v
    return out


def _canon_rows(rows):
    return [_canon_row(r) for r in rows]


def gold_rows_for(sample):
    rows = []
    for g in sample.get("gold_results", []):
        r = g.get("result")
        if isinstance(r, list):
            rows.extend(x for x in r if isinstance(x, dict))
        elif isinstance(r, dict):
            rows.append(r)
    return rows


def wilson_ci(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, center - half) * 100, min(1.0, center + half) * 100)


def bootstrap_mean_ci(values, n_boot=5000, seed=0):
    if not values:
        return (0.0, 0.0, 0.0)
    arr = np.array(values, dtype=float)
    rng = np.random.RandomState(seed)
    means = [rng.choice(arr, size=len(arr), replace=True).mean() for _ in range(n_boot)]
    lo, hi = np.percentile(means, [2.5, 97.5])
    return (float(arr.mean()), float(lo), float(hi))


def cohens_d_paired(x, y):
    d = np.asarray(x, float) - np.asarray(y, float)
    sd = d.std(ddof=1)
    return float(d.mean() / sd) if sd > 0 else 0.0


def mean_std(vals):
    v = [x for x in vals if x is not None]
    if not v:
        return (None, None)
    return (round(float(np.mean(v)), 4), round(float(np.std(v, ddof=1)), 4) if len(v) > 1 else 0.0)


# ---------------------------------------------------------------- judge cache
_cache_lock = threading.Lock()
_cache = {}


def _load_cache():
    global _cache
    if os.path.exists(CACHE_FILE):
        try:
            _cache = json.load(open(CACHE_FILE))
        except Exception:
            _cache = {}


def _save_cache():
    os.makedirs(os.path.dirname(CACHE_FILE), exist_ok=True)
    with _cache_lock:
        json.dump(_cache, open(CACHE_FILE, "w"))


def _key(*parts):
    h = hashlib.sha1()
    for p in parts:
        h.update(str(p).encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()


def _chat_json(client, prompt, max_tokens, max_retries=8):
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=GPT_MODEL, messages=[{"role": "user", "content": prompt}],
                temperature=0, max_tokens=max_tokens,
            )
            raw = resp.choices[0].message.content.strip()
            return json.loads(re.sub(r"```json\s*|\s*```", "", raw).strip())
        except Exception as e:
            msg = str(e)
            wait = 2.0 * (attempt + 1)
            m = re.search(r"try again in ([\d.]+)s", msg)
            if m:
                wait = float(m.group(1)) + 0.5
            if attempt < max_retries - 1:
                time.sleep(wait)
            else:
                print(f"  judge failed: {msg[:150]}")
    return None


def judge_answer(client, question, gold, pred):
    if not pred.strip():
        return {"correct": False, "complete": False, "no_hallucination": False, "score": 0.0}
    ck = "ans:" + _key(question, gold, pred)
    with _cache_lock:
        if ck in _cache:
            return _cache[ck]
    d = _chat_json(client, ANSWER_JUDGE_PROMPT.format(question=question, gold=gold, pred=pred), 400)
    v = ({"correct": False, "complete": False, "no_hallucination": False, "score": 0.0}
         if d is None else
         {"correct": bool(d.get("correct", False)), "complete": bool(d.get("complete", False)),
          "no_hallucination": bool(d.get("no_hallucination", False)), "score": float(d.get("score", 0.0))})
    with _cache_lock:
        _cache[ck] = v
    return v


def judge_grounding(client, question, entities, rows):
    if not rows:
        return []
    ents_s = json.dumps(entities, sort_keys=True, default=str)
    rows_s = json.dumps(rows, sort_keys=True, default=str)
    ck = "grd:" + _key(question, ents_s, rows_s)
    with _cache_lock:
        if ck in _cache:
            return _cache[ck]
    row_list = "\n".join(f"  [{i}] {r}" for i, r in enumerate(rows))
    d = _chat_json(client, GROUNDING_JUDGE_PROMPT.format(
        question=question, entities=ents_s, rows=row_list), 800)
    if d is None:
        return [True] * len(rows)  # fail-open
    res = {r.get("idx"): bool(r.get("relevant", True)) for r in d.get("results", [])}
    v = [res.get(i, True) for i in range(len(rows))]
    with _cache_lock:
        _cache[ck] = v
    return v


# ---------------------------------------------------------------- load & score
def rep_path(full, rep):
    d = f"results_{full}" if rep == 1 else f"results_{full}_run{rep}"
    return f"{d}/{INPUT_BASENAME}_{full}.json"


def collect(details, gold, vocab):
    """Deterministic per-sample metrics + grounding jobs for one (method, rep)."""
    per_sample = []
    grounding_jobs = []  # (sample_idx, step_idx, question, entities, rows)
    for i, (g, d) in enumerate(zip(gold, details)):
        actual_rows, exec_flags = [], []
        for si, step in enumerate(d.get("steps", [])):
            exec_flags.append(bool(step.get("exec_success")))
            actual_rows.extend(step.get("records") or [])
            pg = step.get("primary_grounding_rows") or []
            if pg:
                grounding_jobs.append((i, si, step.get("question", ""), step.get("entities", {}), pg))
        gr = _canon_rows(gold_rows_for(g))
        ar = _canon_rows(actual_rows)
        sc = score_against_gold(ar, gr, build_echo_values(g["input"], vocab))
        per_sample.append({
            "idx": i,
            "split": "unseen_40" if i in UNSEEN_IDX else "seen_80",
            "n_steps": len(exec_flags),
            "exec_fail_steps": sum(1 for f in exec_flags if not f),
            "n_gold_rows": sc["n_gold_rows"] or 0,
            "full_matches": sc["full_matches"] or 0,
            "coverage": sc["coverage"],
            "gold_answer": g.get("gold_answer", ""),
            "final_answer": d.get("final_answer") or "",
        })
    return per_sample, grounding_jobs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()

    _load_cache()
    gold = json.load(open(GOLD_FILE, encoding="utf-8"))
    if args.limit:
        gold = gold[:args.limit]
    n = len(gold)

    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    driver.verify_connectivity()
    vocab = build_vocab(driver)
    driver.close()
    client = get_openai_client()

    # data[method][rep] = {"per_sample": [...], "grounding": {(i,si): [bool,...]}}
    data = {m: {} for m in METHODS}
    answer_jobs = []     # (method, rep, sample_idx)
    grounding_all = []    # (method, rep, i, si, q, ents, rows)

    for m, full in METHODS.items():
        for rep in REPS:
            p = rep_path(full, rep)
            try:
                det = json.load(open(p, encoding="utf-8"))["details"]
            except FileNotFoundError:
                print(f"MISSING {p} -- skipping {m} rep{rep}")
                continue
            if args.limit:
                det = det[:args.limit]
            assert len(det) == n, f"{m} rep{rep}: {len(det)} vs {n}"
            ps, gj = collect(det, gold, vocab)
            data[m][rep] = {"per_sample": ps, "grounding": {}}
            for (i, si, q, e, rows) in gj:
                grounding_all.append((m, rep, i, si, q, e, rows))
            for i in range(n):
                answer_jobs.append((m, rep, i))

    print(f"answer judgments: {len(answer_jobs)}  |  grounding steps: {len(grounding_all)}  "
          f"(cache has {len(_cache)} entries)")

    # ---- run judges concurrently (cache dedups identical content) ----
    def do_answer(job):
        m, rep, i = job
        s = data[m][rep]["per_sample"][i]
        v = judge_answer(client, gold[i]["input"], s["gold_answer"], s["final_answer"])
        return job, v

    def do_grounding(job):
        m, rep, i, si, q, e, rows = job
        return (m, rep, i, si), judge_grounding(client, q, e, rows)

    t0 = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as ex:
        for (m, rep, i, si), verdict in ex.map(do_grounding, grounding_all):
            data[m][rep]["grounding"][(i, si)] = verdict
    print(f"  grounding judge done ({time.time()-t0:.0f}s, cache {len(_cache)})")
    _save_cache()

    t0 = time.time()
    ans = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as ex:
        for job, v in ex.map(do_answer, answer_jobs):
            ans[job] = v
    for (m, rep, i), v in ans.items():
        s = data[m][rep]["per_sample"][i]
        s["answer_correct"] = int(v["correct"])
        s["answer_score"] = v["score"]
    print(f"  answer judge done ({time.time()-t0:.0f}s, cache {len(_cache)})")
    _save_cache()

    # ---------------------------------------------------------------- aggregate
    summary = {}
    for m in METHODS:
        if not data[m]:
            continue
        summary[m] = {}
        for sname, filt in SPLITS.items():
            per_rep = {"exec_fail_pct": [], "gold_recall_pct": [], "answer_correct_pct": [],
                       "coverage_mean_pct": [], "irrelevant_grounding_pct": []}
            # pooled tallies across the 3 reps
            P = {"fail": 0, "steps": 0, "gfull": 0, "grows": 0, "acorr": 0, "asamp": 0,
                 "girr": 0, "grow": 0}
            cov_pool = []
            for rep, blob in data[m].items():
                ps = [s for s in blob["per_sample"] if filt(s["idx"])]
                fail = sum(s["exec_fail_steps"] for s in ps)
                steps = sum(s["n_steps"] for s in ps)
                gfull = sum(s["full_matches"] for s in ps)
                grows = sum(s["n_gold_rows"] for s in ps)
                acorr = sum(s.get("answer_correct", 0) for s in ps)
                asamp = len(ps)
                covs = [s["coverage"] for s in ps if s["coverage"] is not None]
                per_rep["exec_fail_pct"].append(round(100 * fail / max(steps, 1), 3))
                per_rep["gold_recall_pct"].append(round(100 * gfull / max(grows, 1), 3))
                per_rep["answer_correct_pct"].append(round(100 * acorr / max(asamp, 1), 3))
                per_rep["coverage_mean_pct"].append(round(100 * float(np.mean(covs)), 3) if covs else None)
                P["fail"] += fail; P["steps"] += steps
                P["gfull"] += gfull; P["grows"] += grows
                P["acorr"] += acorr; P["asamp"] += asamp
                cov_pool.extend(covs)
                if HAS_GROUNDING[m]:
                    girr = grow = 0
                    for (i, si), v in blob["grounding"].items():
                        if not filt(i):
                            continue
                        grow += len(v)
                        girr += sum(1 for r in v if not r)
                    per_rep["irrelevant_grounding_pct"].append(round(100 * girr / max(grow, 1), 3))
                    P["girr"] += girr; P["grow"] += grow

            cov_m, cov_lo, cov_hi = bootstrap_mean_ci(cov_pool)
            entry = {
                "exec_fail": {
                    "pooled_pct": round(100 * P["fail"] / max(P["steps"], 1), 2),
                    "ci95": [round(x, 2) for x in wilson_ci(P["fail"], P["steps"])],
                    "pooled_k_n": [P["fail"], P["steps"]],
                    "per_rep_pct": per_rep["exec_fail_pct"],
                    "across_rep_mean_std": mean_std(per_rep["exec_fail_pct"]),
                },
                "gold_recall": {
                    "pooled_pct": round(100 * P["gfull"] / max(P["grows"], 1), 2),
                    "ci95": [round(x, 2) for x in wilson_ci(P["gfull"], P["grows"])],
                    "pooled_k_n": [P["gfull"], P["grows"]],
                    "per_rep_pct": per_rep["gold_recall_pct"],
                    "across_rep_mean_std": mean_std(per_rep["gold_recall_pct"]),
                },
                "gold_coverage_soft": {
                    "pooled_pct": round(100 * cov_m, 2),
                    "ci95": [round(100 * cov_lo, 2), round(100 * cov_hi, 2)],
                    "per_rep_pct": per_rep["coverage_mean_pct"],
                    "across_rep_mean_std": mean_std(per_rep["coverage_mean_pct"]),
                },
                "answer_correct": {
                    "pooled_pct": round(100 * P["acorr"] / max(P["asamp"], 1), 2),
                    "ci95": [round(x, 2) for x in wilson_ci(P["acorr"], P["asamp"])],
                    "pooled_k_n": [P["acorr"], P["asamp"]],
                    "per_rep_pct": per_rep["answer_correct_pct"],
                    "across_rep_mean_std": mean_std(per_rep["answer_correct_pct"]),
                },
            }
            if HAS_GROUNDING[m]:
                entry["irrelevant_grounding"] = {
                    "pooled_pct": round(100 * P["girr"] / max(P["grow"], 1), 2),
                    "ci95": [round(x, 2) for x in wilson_ci(P["girr"], P["grow"])],
                    "pooled_k_n": [P["girr"], P["grow"]],
                    "per_rep_pct": per_rep["irrelevant_grounding_pct"],
                    "across_rep_mean_std": mean_std(per_rep["irrelevant_grounding_pct"]),
                }
            summary[m][sname] = entry

    # ---------------------------------------------------------------- significance
    def rep_avg_vec(m, key):
        reps = data[m]
        out = []
        for i in range(n):
            vals = []
            for rep, blob in reps.items():
                s = blob["per_sample"][i]
                if key == "exec_fail_rate":
                    vals.append(s["exec_fail_steps"] / s["n_steps"] if s["n_steps"] else None)
                elif key == "gold_recall_rate":
                    vals.append(s["full_matches"] / s["n_gold_rows"] if s["n_gold_rows"] else None)
                elif key == "coverage":
                    vals.append(s["coverage"])
                elif key == "answer_correct":
                    vals.append(s.get("answer_correct"))
            vals = [v for v in vals if v is not None]
            out.append(float(np.mean(vals)) if vals else None)
        return out

    sig = {}
    for a, b in PAIRS:
        if not data[a] or not data[b]:
            continue
        pk = f"{a}__vs__{b}"
        sig[pk] = {}
        for sname, filt in SPLITS.items():
            e = {}
            for key in ("exec_fail_rate", "gold_recall_rate", "coverage", "answer_correct"):
                va, vb = rep_avg_vec(a, key), rep_avg_vec(b, key)
                pr = [(x, y) for i, (x, y) in enumerate(zip(va, vb))
                      if filt(i) and x is not None and y is not None]
                if len(pr) < 5:
                    e[key] = {"n_pairs": len(pr)}
                    continue
                xa = np.array([p[0] for p in pr]); xb = np.array([p[1] for p in pr])
                try:
                    w_p = float(stats.wilcoxon(xa, xb).pvalue)
                except ValueError:
                    w_p = 1.0
                # within-rep p-values
                per_rep_p = []
                for rep in sorted(set(data[a]) & set(data[b])):
                    pa = data[a][rep]["per_sample"]; pb = data[b][rep]["per_sample"]
                    prr = []
                    for i in range(n):
                        if not filt(i):
                            continue
                        if key == "exec_fail_rate":
                            xi = pa[i]["exec_fail_steps"] / pa[i]["n_steps"] if pa[i]["n_steps"] else None
                            yi = pb[i]["exec_fail_steps"] / pb[i]["n_steps"] if pb[i]["n_steps"] else None
                        elif key == "gold_recall_rate":
                            xi = pa[i]["full_matches"] / pa[i]["n_gold_rows"] if pa[i]["n_gold_rows"] else None
                            yi = pb[i]["full_matches"] / pb[i]["n_gold_rows"] if pb[i]["n_gold_rows"] else None
                        elif key == "coverage":
                            xi, yi = pa[i]["coverage"], pb[i]["coverage"]
                        else:
                            xi, yi = pa[i].get("answer_correct"), pb[i].get("answer_correct")
                        if xi is not None and yi is not None:
                            prr.append((xi, yi))
                    if len(prr) >= 5:
                        try:
                            per_rep_p.append(round(float(stats.wilcoxon(
                                [p[0] for p in prr], [p[1] for p in prr]).pvalue), 5))
                        except ValueError:
                            per_rep_p.append(1.0)
                e[key] = {
                    "n_pairs": len(pr),
                    "mean_a_pct": round(100 * float(xa.mean()), 2),
                    "mean_b_pct": round(100 * float(xb.mean()), 2),
                    "delta_b_minus_a_pct": round(100 * float(xb.mean() - xa.mean()), 2),
                    "wilcoxon_p": round(w_p, 6),
                    "cohens_d_paired": round(cohens_d_paired(xb, xa), 3),
                    "within_rep_wilcoxon_p": per_rep_p,
                    "stable": (all(p < 0.05 for p in per_rep_p) or all(p >= 0.05 for p in per_rep_p))
                              if per_rep_p else None,
                }
            # McNemar (rep-majority) for answer_correct
            fa, fb = rep_avg_vec(a, "answer_correct"), rep_avg_vec(b, "answer_correct")
            a_only = b_only = both = neither = 0
            for i in range(n):
                if not filt(i) or fa[i] is None or fb[i] is None:
                    continue
                ba, bb = (1 if fa[i] >= 0.5 else 0), (1 if fb[i] >= 0.5 else 0)
                if ba and bb: both += 1
                elif ba: a_only += 1
                elif bb: b_only += 1
                else: neither += 1
            disc = a_only + b_only
            mcp = float(stats.binomtest(min(a_only, b_only), disc, 0.5).pvalue) if disc else 1.0
            e["answer_correct_mcnemar"] = {"a_only": a_only, "b_only": b_only,
                                           "both": both, "neither": neither,
                                           "discordant": disc, "exact_p": round(mcp, 6)}
            sig[pk][sname] = e

    # irrelevant grounding: exp2 vs exp3 pooled 2-proportion (Fisher)
    if data.get("exp2") and data.get("exp3"):
        sig.setdefault("exp2__vs__exp3", {})
        for sname, filt in SPLITS.items():
            def tally(m):
                irr = tot = 0
                for rep, blob in data[m].items():
                    for (i, si), v in blob["grounding"].items():
                        if not filt(i):
                            continue
                        tot += len(v); irr += sum(1 for r in v if not r)
                return irr, tot
            i2, t2 = tally("exp2"); i3, t3 = tally("exp3")
            odds, fp = stats.fisher_exact([[i2, t2 - i2], [i3, t3 - i3]])
            sig["exp2__vs__exp3"].setdefault(sname, {})["irrelevant_grounding_pooled"] = {
                "exp2_pct": round(100 * i2 / max(t2, 1), 2), "exp2_k_n": [i2, t2],
                "exp3_pct": round(100 * i3 / max(t3, 1), 2), "exp3_k_n": [i3, t3],
                "fisher_p": round(float(fp), 6),
            }

    os.makedirs("cypher_results", exist_ok=True)
    json.dump(summary, open("cypher_results/all_reps_full_summary.json", "w"), indent=2, ensure_ascii=False)
    json.dump(sig, open("cypher_results/all_reps_full_significance.json", "w"), indent=2, ensure_ascii=False)
    json.dump({m: {str(r): b["per_sample"] for r, b in reps.items()} for m, reps in data.items()},
              open("cypher_results/all_reps_full_per_sample.json", "w"), indent=2, ensure_ascii=False, default=str)
    _save_cache()

    # ---------------------------------------------------------------- console
    METRIC_ROWS = [
        ("exec_fail", "Cypher exec failure  /step  (lower better)"),
        ("gold_recall", "Gold recall          /gold row"),
        ("irrelevant_grounding", "Irrelevant grounding /grounding row (lower better)"),
        ("answer_correct", "Final-answer correct /sample"),
    ]
    for sname in SPLITS:
        print("\n" + "=" * 104)
        print(f"  SPLIT = {sname}")
        print("=" * 104)
        print(f"  {'metric':<46}{'exp1_5':>18}{'exp2':>18}{'exp3':>18}   (pooled 3 reps, [95% CI])")
        for mk, label in METRIC_ROWS:
            cells = []
            for m in ("exp1_5", "exp2", "exp3"):
                d = summary.get(m, {}).get(sname, {}).get(mk)
                if not d:
                    cells.append(f"{'--':>18}")
                    continue
                cells.append(f"{d['pooled_pct']:>7.1f} [{d['ci95'][0]:.1f},{d['ci95'][1]:.1f}]".rjust(18))
            print(f"  {label:<46}" + "".join(cells))
        # per-rep spread line
        for mk, label in METRIC_ROWS:
            spread = []
            for m in ("exp1_5", "exp2", "exp3"):
                d = summary.get(m, {}).get(sname, {}).get(mk)
                if not d:
                    spread.append(f"{'--':>18}"); continue
                pr = d["per_rep_pct"]
                spread.append(("/".join(f"{x:.1f}" if x is not None else "-" for x in pr)).rjust(18))
            print(f"  {'  per-rep: ' + label:<46}" + "".join(spread))

    print("\n" + "=" * 104)
    print("  PAIRWISE (rep-averaged per-sample, paired Wilcoxon; 'stable' = same verdict in all 3 reps)")
    print("=" * 104)
    name = {"exec_fail_rate": "exec failure", "gold_recall_rate": "gold recall",
            "coverage": "soft coverage", "answer_correct": "answer correct"}
    for pk, subs in sig.items():
        print(f"\n  {pk}")
        for sname, e in subs.items():
            for key in ("exec_fail_rate", "gold_recall_rate", "coverage", "answer_correct"):
                d = e.get(key)
                if not d or "wilcoxon_p" not in d:
                    continue
                print(f"    {sname:<10} {name[key]:<15} Δ(b-a)={d['delta_b_minus_a_pct']:+6.2f}pp  "
                      f"p={d['wilcoxon_p']:.2g}  d={d['cohens_d_paired']:+.2f}  "
                      f"within-rep p={d['within_rep_wilcoxon_p']}  stable={d['stable']}")
            ig = e.get("irrelevant_grounding_pooled")
            if ig:
                print(f"    {sname:<10} irrel.grounding exp2={ig['exp2_pct']:.1f}% exp3={ig['exp3_pct']:.1f}%  "
                      f"fisher_p={ig['fisher_p']:.2g}")
    print("\nSaved -> cypher_results/all_reps_full_{summary,significance,per_sample}.json")


if __name__ == "__main__":
    main()
