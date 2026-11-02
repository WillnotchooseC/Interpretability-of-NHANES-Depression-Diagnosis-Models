"""
Merge three few-shot/GraphRAG passes that used DIFFERENT sample orders into one
per-arm file, so the reps carry genuine run-to-run variation (see CLAUDE.md
Gotchas: identical-order reps are deterministic replays).

  rep1 = results_<exp>_test200_rep1/          (original order)
  rep2 = results_<exp>_test200_shuf<seedA>/   (order permuted with seedA)
  rep3 = results_<exp>_test200_shuf<seedB>/   (order permuted with seedB)

Every rep is un-shuffled back to the original idx order, so records[i] always
refers to gold sample i and the file has exactly the layout of
cypher_results/test200_<arm>_*_3reps.json (records[i].reps.rep{1,2,3}.steps[]
+ gold). Top-level `rep_orders` records which permutation each rep was run in.

Usage:
    python merge_test200_shuffled_reps.py --arm fewshot --seeds 42 43
"""
import argparse, json, os
from run_test200_fewshot_vs_graphrag_3reps import ARMS, DECOMP_FILE, GOLD_FILE, CYPHER_MODEL, out_file

OUT = {
    "fewshot": "cypher_results/test200_fewshot_exp1_5_3orders.json",
    "graphrag": "cypher_results/test200_graphrag_exp2_3orders.json",
    "graphrag_v2": "cypher_results/test200_graphrag_v2_exp2_3orders.json",
}
STEP_KEYS = ("question", "cypher", "exec_success", "has_results", "error", "rows", "records")
GRAPHRAG_KEYS = ("query_type", "entities", "entity_provenance", "primary_path",
                 "primary_grounding_rows", "answer_properties", "primary_grounding_error",
                 "repaired_concatenated_query")


def load_pass(arm, exp_name, seed):
    """Return details indexed by ORIGINAL idx, plus the order the pass ran in."""
    if seed is None:
        d = json.load(open(out_file(exp_name, 1), encoding="utf-8"))["details"]
        return {i: x for i, x in enumerate(d)}, list(range(len(d)))
    shuf_in = json.load(open(DECOMP_FILE.replace(".json", f"_shuf{seed}.json"), encoding="utf-8"))
    stem = os.path.splitext(os.path.basename(DECOMP_FILE))[0] + f"_shuf{seed}"
    path = os.path.join(f"results_{exp_name}_test200_shuf{seed}", f"{stem}_{exp_name}.json")
    d = json.load(open(path, encoding="utf-8"))["details"]
    assert len(d) == len(shuf_in)
    for s, x in zip(shuf_in, d):
        assert s["input"] == x["input"], f"seed {seed}: order mismatch"
    return {s["orig_idx"]: x for s, x in zip(shuf_in, d)}, [s["orig_idx"] for s in shuf_in]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="fewshot", choices=list(ARMS))
    ap.add_argument("--seeds", nargs=2, type=int, default=[42, 43])
    args = ap.parse_args()
    _, exp_name = ARMS[args.arm]
    gold = json.load(open(GOLD_FILE, encoding="utf-8"))
    decomp = json.load(open(DECOMP_FILE, encoding="utf-8"))

    passes = {"rep1": load_pass(args.arm, exp_name, None),
              "rep2": load_pass(args.arm, exp_name, args.seeds[0]),
              "rep3": load_pass(args.arm, exp_name, args.seeds[1])}
    rep_orders = {"rep1": {"description": "original sample order", "seed": None, "order": passes["rep1"][1]},
                  "rep2": {"description": f"sample order permuted with seed {args.seeds[0]}", "seed": args.seeds[0], "order": passes["rep2"][1]},
                  "rep3": {"description": f"sample order permuted with seed {args.seeds[1]}", "seed": args.seeds[1], "order": passes["rep3"][1]}}

    records = []
    for i, (g, dc) in enumerate(zip(gold, decomp)):
        rec = {"idx": i, "split": g["split"], "input": g["input"], "decomposition": dc["prediction"],
               "gold_results": g["gold_results"], "gold_answer": g["gold_answer"], "reps": {}}
        for rep, (by_idx, _) in passes.items():
            det = by_idx[i]
            assert det["input"] == g["input"]
            steps = []
            for st in det["steps"]:
                s = {k: st.get(k) for k in STEP_KEYS}
                if args.arm.startswith("graphrag"):
                    for k in GRAPHRAG_KEYS:
                        if k in st:
                            s[k] = st[k]
                steps.append(s)
            rec["reps"][rep] = {"num_searching_subquestions": det["num_searching_subquestions"], "steps": steps}
        records.append(rec)

    # rep-to-rep identity summary
    def key(rec, rep): return [s["cypher"] for s in rec["reps"][rep]["steps"]]
    ident = {f"rep1_vs_{r}": sum(1 for rec in records if key(rec, "rep1") == key(rec, r)) for r in ("rep2", "rep3")}
    ident["rep2_vs_rep3"] = sum(1 for rec in records if key(rec, "rep2") == key(rec, "rep3"))

    out = {"arm": args.arm, "experiment": exp_name, "decomposition_source": DECOMP_FILE,
           "decomposition_model": "ood_curriculum_progressive_corrected (curriculum_test_200 rep1, greedy)",
           "cypher_model": CYPHER_MODEL, "temperature": 0, "final_answer_synthesis": False,
           "gold_file": GOLD_FILE, "reps": [1, 2, 3],
           "note": "each rep = fresh SGLang server, sequential one-request-at-a-time generation; reps differ ONLY in the order the 200 samples were processed (rep_orders), which is what makes greedy outputs vary between reps",
           "rep_orders": rep_orders,
           "samples_with_identical_cypher": ident,
           "records": records}
    json.dump(out, open(OUT[args.arm], "w", encoding="utf-8"), indent=2, ensure_ascii=False, default=str)
    print(f"{args.arm}: {len(records)} records x 3 reps -> {OUT[args.arm]} ({os.path.getsize(OUT[args.arm])/1024:.0f} KB)")
    print("  samples with identical Cypher across reps:", ident)


if __name__ == "__main__":
    main()
