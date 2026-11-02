"""
Re-run ONLY the final-answer synthesis step for exp1/exp2/exp3, reusing each
experiment's already-executed Cypher/Neo4j steps (details[i]['steps']) cached
in its output JSON. Motivated by build_answer_prompt's new SHAP sign-convention
rule (positive shap_value -> toward depression, negative -> away) -- Cypher
generation and graph retrieval are unaffected by that change, so redoing the
full pipeline (SGLang decomp+cypher+exec) would just waste ~35min re-deriving
identical retrieval results. This only re-does the cheap synthesis call.

Usage:
    python resynthesize_answers.py
"""
import concurrent.futures
import importlib.util
import json
import re
import time

import requests

SGLANG_URL = "http://localhost:30000"
MAX_NEW_TOKENS_ANSWER = 384

EXPERIMENTS = {
    "exp1_baseline_fewshot": (
        "experiment1_baseline_fewshot.py",
        "results_exp1_baseline_fewshot/progressive_decomp_for_experiments_exp1_baseline_fewshot.json",
    ),
    "exp2_fewshot_graphrag": (
        "experiment2_fewshot_graphrag.py",
        "results_exp2_fewshot_graphrag/progressive_decomp_for_experiments_exp2_fewshot_graphrag.json",
    ),
    "exp3_fewshot_graphrag_rescue": (
        "experiment3_fewshot_graphrag_rescue.py",
        "results_exp3_fewshot_graphrag_rescue/progressive_decomp_for_experiments_exp3_fewshot_graphrag_rescue.json",
    ),
}


def load_module(script_path, mod_name):
    spec = importlib.util.spec_from_file_location(mod_name, script_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def generate_sglang(prompt, max_new_tokens, stop=None, retries=3):
    payload = {
        "model": "default", "prompt": prompt, "max_tokens": max_new_tokens,
        "temperature": 0, "top_p": 1.0, "repetition_penalty": 1.05,
    }
    if stop:
        payload["stop"] = stop
    for attempt in range(retries):
        try:
            r = requests.post(f"{SGLANG_URL}/v1/completions", json=payload, timeout=90)
            r.raise_for_status()
            return r.json()["choices"][0]["text"].strip()
        except Exception as e:
            if attempt + 1 == retries:
                print(f"  generation failed: {e}")
                return ""
            time.sleep(2)
    return ""


def main():
    for exp_name, (script_path, out_path) in EXPERIMENTS.items():
        print(f"\n=== {exp_name} ===")
        mod = load_module(script_path, exp_name)
        data = json.load(open(out_path, encoding="utf-8"))
        details = data["details"]

        def one(i):
            d = details[i]
            prompt = mod.build_answer_prompt(d["input"], d["steps"])
            raw = generate_sglang(
                prompt, MAX_NEW_TOKENS_ANSWER,
                stop=["\nQuestion:", "\nOriginal question:"],
            )
            ans = re.sub(r"^\s*(?:Final answer|Answer)\s*:\s*", "", raw, flags=re.I).strip()
            return i, ans

        t0 = time.time()
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as ex:
            for i, ans in ex.map(one, range(len(details))):
                details[i]["final_answer"] = ans
                if (i + 1) % 20 == 0:
                    print(f"  {i+1}/{len(details)} ({time.time()-t0:.0f}s)")
        empties = [i for i, d in enumerate(details) if not (d.get("final_answer") or "").strip()]
        print(f"  done in {time.time()-t0:.0f}s; empty final_answer at: {empties}")

        json.dump(data, open(out_path, "w"), indent=2, ensure_ascii=False, default=str)
        print(f"  overwrote {out_path}")


if __name__ == "__main__":
    main()
