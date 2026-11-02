"""
Few-shot (exp1.5) vs GraphRAG (exp2) Cypher-generation comparison on
ood_data/curriculum_test_200_gold.json, 3 repetitions each, strictly sequential.

Decomposition input (fixed across all 6 passes, same design as the 2026-09-01
test_120 eval): ood_data/progressive_corrected_decomp_test200.json = the
ood_curriculum_progressive_corrected model's rep1 greedy decomposition of
curriculum_test_200 (fields: idx, split, input, prediction -- no gold text, so
the experiment scripts cannot fall back to the reference).

Per pass: a FRESH SGLang server for the base Cypher model (Qwen/Qwen3-8B-Base)
is launched via evaluate.launch_sglang_server (sys.executable, process-group
shutdown, refuses to start if anything already listens on the port), the
experiment script runs against it with --sglang_url and --no_answer (no
final-answer synthesis, no OpenAI anywhere), then the server is torn down and
we wait until the port is closed before the next pass.

Pass order: fewshot rep1, rep2, rep3, then graphrag rep1, rep2, rep3.
Per-pass raw outputs: results_<exp>_test200_rep{k}/<decomp-stem>_<exp>.json
Merged (one file per arm, all 3 reps + gold):
  cypher_results/test200_fewshot_exp1_5_3reps.json
  cypher_results/test200_graphrag_exp2_3reps.json

Usage:
    python run_test200_fewshot_vs_graphrag_3reps.py            # run all 6 passes + merge
    python run_test200_fewshot_vs_graphrag_3reps.py --merge_only
    python run_test200_fewshot_vs_graphrag_3reps.py --arms graphrag --reps 2 3
"""

import argparse
import json
import os
import subprocess
import sys
import time

import requests

from evaluate import launch_sglang_server, shutdown_sglang_server

DECOMP_FILE = "ood_data/progressive_corrected_decomp_test200.json"
GOLD_FILE = "ood_data/curriculum_test_200_gold.json"
CYPHER_MODEL = "Qwen/Qwen3-8B-Base"
PORT = 30000
ARMS = {
    "fewshot": ("experiment1_5_baseline_original_input.py", "exp1_5_baseline_original_input"),
    "graphrag": ("experiment2_fewshot_graphrag.py", "exp2_fewshot_graphrag"),
    "graphrag_v2": ("experiment2_fewshot_graphrag_v2.py", "exp2_fewshot_graphrag_v2"),  # redacted grounding + one-RETURN rules + repair
    "graphrag_v2_rescue": ("experiment3_fewshot_graphrag_v2_rescue.py", "exp3_fewshot_graphrag_v2_rescue"),  # v2 + directed neighbourhood rescue (pre-gen + execution-feedback retry)
    "fewshot_notruncate": ("experiment1_6_baseline_notruncate.py", "exp1_6_baseline_notruncate"),  # exp1.5 + v2's local_max_abs no-LIMIT rule, still zero graph access
}
MERGED = {
    "fewshot": "cypher_results/test200_fewshot_exp1_5_3reps.json",
    "graphrag": "cypher_results/test200_graphrag_exp2_3reps.json",
    "graphrag_v2": "cypher_results/test200_graphrag_v2_exp2_3reps.json",
    "graphrag_v2_rescue": "cypher_results/test200_graphrag_v2_rescue_exp3_reps.json",
    "fewshot_notruncate": "cypher_results/test200_fewshot_notruncate_exp1_6_reps.json",
}


def out_dir(exp_name, rep):
    return f"results_{exp_name}_test200_rep{rep}"


def out_file(exp_name, rep):
    stem = os.path.splitext(os.path.basename(DECOMP_FILE))[0]
    return os.path.join(out_dir(exp_name, rep), f"{stem}_{exp_name}.json")


def port_closed():
    try:
        requests.get(f"http://localhost:{PORT}/health", timeout=2)
        return False
    except requests.RequestException:
        return True


def wait_port_closed(max_s=120):
    t0 = time.time()
    while time.time() - t0 < max_s:
        if port_closed() and subprocess.run(["pgrep", "-f", "sglang.launch_server"],
                                            capture_output=True).returncode != 0:
            return
        time.sleep(2)
    raise SystemExit(f"port {PORT} / sglang still up after {max_s}s -- aborting rather than risk reuse")


def run_pass(arm, rep):
    script, exp_name = ARMS[arm]
    target = out_file(exp_name, rep)
    if os.path.exists(target):
        print(f"\n### {arm} rep{rep}: exists, skipping ({target})")
        return
    print(f"\n{'#'*70}\n### {arm} rep{rep}  {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}\n{'#'*70}")
    wait_port_closed()
    proc = launch_sglang_server(model_path=CYPHER_MODEL, port=PORT)
    if proc is None:
        raise SystemExit("SGLang failed to start")
    try:
        cmd = [sys.executable, "-u", script, "--files", DECOMP_FILE,
               "--sglang_url", f"http://localhost:{PORT}", "--no_answer",
               "--output_dir", out_dir(exp_name, rep)]
        print("  $", " ".join(cmd))
        rc = subprocess.run(cmd).returncode
    finally:
        shutdown_sglang_server(proc)
        wait_port_closed()
    if rc != 0 or not os.path.exists(target):
        raise SystemExit(f"{arm} rep{rep} failed (rc={rc}); see output above")
    print(f"  -> {target}")


def dedupe_rows(rows):
    seen, out = set(), []
    for r in rows:
        key = json.dumps(r, sort_keys=True, default=str)
        if key not in seen:
            seen.add(key)
            out.append(r)
    return out


def merge(arm, reps, slim=False):
    _, exp_name = ARMS[arm]
    gold = json.load(open(GOLD_FILE, encoding="utf-8"))
    decomp = json.load(open(DECOMP_FILE, encoding="utf-8"))
    per_rep = {}
    for rep in reps:
        d = json.load(open(out_file(exp_name, rep), encoding="utf-8"))
        assert len(d["details"]) == len(gold), f"{arm} rep{rep}: {len(d['details'])} != {len(gold)}"
        per_rep[rep] = d
    records = []
    for i, (g, dc) in enumerate(zip(gold, decomp)):
        assert g["input"] == dc["input"]
        rec = {
            "idx": i, "split": g["split"], "input": g["input"],
            "decomposition": dc["prediction"],
            "gold_results": g["gold_results"], "gold_answer": g["gold_answer"],
            "reps": {},
        }
        for rep in reps:
            det = per_rep[rep]["details"][i]
            assert det["input"] == g["input"], f"{arm} rep{rep} order mismatch at {i}"
            steps = []
            for st in det["steps"]:
                s = {k: st.get(k) for k in ("question", "cypher", "exec_success", "has_results",
                                             "error", "rows", "records")}
                if slim:
                    # `rows` keeps the true Neo4j row count; `records` is de-duplicated
                    # (a cross-join can repeat the same fact hundreds of times).
                    s["records"] = dedupe_rows(st.get("records") or [])
                    s["rows_distinct"] = len(s["records"])
                if arm.startswith("graphrag"):
                    keys = ["query_type", "entities", "entity_provenance", "primary_path",
                            "primary_grounding_rows", "primary_grounding_identity_rows",
                            "answer_properties", "primary_grounding_error",
                            "repaired_concatenated_query", "rescue_triggered", "rescue_reasons",
                            "post_rescue_triggered", "post_rescue_reason", "cypher_before_rescue",
                            "exec_success_before_rescue", "rows_before_rescue", "post_rescue_improved"]
                    if slim:  # identity_rows is derivable from primary_grounding_rows
                        keys.remove("primary_grounding_identity_rows")
                    for k in keys:
                        if k in st:
                            s[k] = st.get(k)
                steps.append(s)
            rec["reps"][f"rep{rep}"] = {"num_searching_subquestions": det["num_searching_subquestions"],
                                        "steps": steps}
        records.append(rec)
    out = {
        "arm": arm, "experiment": exp_name,
        "use_graphrag": per_rep[reps[0]]["use_graphrag"],
        "use_neighborhood_rescue": per_rep[reps[0]]["use_neighborhood_rescue"],
        "decomposition_source": DECOMP_FILE,
        "decomposition_model": "ood_curriculum_progressive_corrected (curriculum_test_200 rep1, greedy)",
        "cypher_model": CYPHER_MODEL, "temperature": 0, "final_answer_synthesis": False,
        "gold_file": GOLD_FILE, "reps": reps, "slim": slim,
        "per_rep_metrics": {f"rep{r}": per_rep[r]["metrics"] for r in reps},
        "records": records,
    }
    os.makedirs(os.path.dirname(MERGED[arm]), exist_ok=True)
    json.dump(out, open(MERGED[arm], "w", encoding="utf-8"), indent=2, ensure_ascii=False, default=str)
    print(f"merged {arm}: {len(records)} records x reps {reps} -> {MERGED[arm]} "
          f"({os.path.getsize(MERGED[arm])/1024:.0f} KB)")
    for r in reps:
        print(f"  rep{r} metrics: {json.dumps(per_rep[r]['metrics'])}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+", default=["fewshot", "graphrag"], choices=list(ARMS))
    ap.add_argument("--reps", nargs="+", type=int, default=[1, 2, 3])
    ap.add_argument("--merge_only", action="store_true")
    ap.add_argument("--slim", action="store_true",
                    help="merged file: de-duplicate returned records (adds rows_distinct) and drop the "
                         "redundant primary_grounding_identity_rows field")
    args = ap.parse_args()

    if not args.merge_only:
        for arm in args.arms:
            for rep in args.reps:
                run_pass(arm, rep)
    for arm in args.arms:
        merge(arm, args.reps, slim=args.slim)
    print("\nALL DONE")


if __name__ == "__main__":
    main()
