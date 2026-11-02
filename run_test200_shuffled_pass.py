"""
One extra pass of a test_200 Cypher arm with the SAMPLE ORDER SHUFFLED (seed-based
permutation of ood_data/progressive_corrected_decomp_test200.json), to break the
"fresh server + identical prompt order = deterministic replay" property and expose
real greedy-decoding run-to-run variation (see CLAUDE.md Gotchas, 2026-09-13).

Everything else is identical to run_test200_fewshot_vs_graphrag_3reps.py: fresh
Qwen/Qwen3-8B-Base SGLang server, --sglang_url, --no_answer, sequential per-sample
execution. The output is un-shuffled back to the original idx order and compared
step-by-step with the arm's rep1 (Cypher text, returned rows, gold coverage).

Usage:
    python run_test200_shuffled_pass.py --arm fewshot --seed 42
    python run_test200_shuffled_pass.py --arm fewshot --seed 42 --compare_only
"""
import argparse, json, os, random, subprocess, sys
import numpy as np
from neo4j import GraphDatabase
from evaluate import launch_sglang_server, shutdown_sglang_server
from run_test200_fewshot_vs_graphrag_3reps import ARMS, MERGED, DECOMP_FILE, GOLD_FILE, CYPHER_MODEL, PORT, wait_port_closed
from run_finetuned_pipeline_eval import score_against_gold, build_echo_values
from entity_retrieval import build_vocab
from score_experiments import gold_rows_for


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="fewshot", choices=list(ARMS))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--compare_only", action="store_true")
    args = ap.parse_args()
    script, exp_name = ARMS[args.arm]

    base = json.load(open(DECOMP_FILE, encoding="utf-8"))
    order = list(range(len(base)))
    random.Random(args.seed).shuffle(order)
    shuf_file = DECOMP_FILE.replace(".json", f"_shuf{args.seed}.json")
    shuf = [dict(base[i], orig_idx=i) for i in order]
    json.dump(shuf, open(shuf_file, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    out_dir = f"results_{exp_name}_test200_shuf{args.seed}"
    out_file = os.path.join(out_dir, os.path.splitext(os.path.basename(shuf_file))[0] + f"_{exp_name}.json")
    print(f"shuffled input: {shuf_file} (first 10 orig idx: {order[:10]})")

    if not args.compare_only:
        wait_port_closed()
        proc = launch_sglang_server(model_path=CYPHER_MODEL, port=PORT)
        if proc is None:
            raise SystemExit("SGLang failed to start")
        try:
            cmd = [sys.executable, "-u", script, "--files", shuf_file, "--sglang_url", f"http://localhost:{PORT}",
                   "--no_answer", "--output_dir", out_dir]
            print("  $", " ".join(cmd))
            rc = subprocess.run(cmd).returncode
        finally:
            shutdown_sglang_server(proc)
            wait_port_closed()
        if rc != 0:
            raise SystemExit(f"pass failed rc={rc}")

    # ---- un-shuffle and compare with rep1 ----
    shuf_out = json.load(open(out_file, encoding="utf-8"))["details"]
    by_orig = {s["orig_idx"]: d for s, d in zip(shuf, shuf_out)}
    for s, d in zip(shuf, shuf_out):
        assert s["input"] == d["input"]
    rep1 = json.load(open(MERGED[args.arm], encoding="utf-8"))["records"]
    gold = json.load(open(GOLD_FILE, encoding="utf-8"))
    drv = GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", "12345678")); vocab = build_vocab(drv); drv.close()

    n_steps = diff_cypher = diff_rows = diff_nonempty = 0
    cov1, cov2, changed_samples, examples = [], [], [], []
    for r in rep1:
        i = r["idx"]; d = by_orig[i]
        s1 = r["reps"]["rep1"]["steps"]; s2 = d["steps"]
        assert len(s1) == len(s2)
        sample_changed = False
        for a, b in zip(s1, s2):
            n_steps += 1
            if a["cypher"] != b["cypher"]:
                diff_cypher += 1; sample_changed = True
                if len(examples) < 4:
                    examples.append((i, r["split"], a["question"][:90], a["cypher"], b["cypher"]))
            ra = json.dumps(a["records"], sort_keys=True, default=str); rb = json.dumps(b.get("records"), sort_keys=True, default=str)
            if ra != rb: diff_rows += 1
            if bool(a["has_results"]) != bool(b.get("has_results")): diff_nonempty += 1
        if sample_changed: changed_samples.append(i)
        echo = build_echo_values(r["input"], vocab); g = gold_rows_for(r)
        c1 = score_against_gold([x for st in s1 for x in (st.get("records") or [])], g, echo)["coverage"] or 0.0
        c2 = score_against_gold([x for st in s2 for x in (st.get("records") or [])], g, echo)["coverage"] or 0.0
        cov1.append(c1); cov2.append(c2)
    cov1, cov2 = np.array(cov1), np.array(cov2); splits = np.array([r["split"] for r in rep1])
    print(f"\n===== {args.arm}: rep1 (original order) vs shuffled pass (seed={args.seed}) =====")
    print(f"  steps: {n_steps} | Cypher text differs: {diff_cypher} | returned rows differ: {diff_rows} | non-empty flag flips: {diff_nonempty}")
    print(f"  samples with any changed Cypher: {len(changed_samples)}/200 -> {changed_samples}")
    for sp in ["overall", "seen", "unseen"]:
        m = np.ones(len(cov1), bool) if sp == "overall" else (splits == sp)
        d = cov2[m] - cov1[m]
        print(f"  graph_coverage {sp:8s}: rep1={cov1[m].mean():.4f}  shuffled={cov2[m].mean():.4f}  delta={d.mean():+.4f}  "
              f"(samples better/worse/same = {(d>0).sum()}/{(d<0).sum()}/{(d==0).sum()})")
    for i, sp, q, a, b in examples:
        print(f"\n  e.g. idx {i} [{sp}] {q}\n     rep1:     {a[:260]}\n     shuffled: {b[:260]}")
    json.dump({"arm": args.arm, "seed": args.seed, "order": order, "steps": n_steps, "diff_cypher": diff_cypher,
               "diff_rows": diff_rows, "changed_samples": changed_samples,
               "coverage_rep1": cov1.tolist(), "coverage_shuffled": cov2.tolist()},
              open(f"cypher_results/test200_{args.arm}_shuf{args.seed}_vs_rep1.json", "w"), indent=1)


if __name__ == "__main__":
    main()
