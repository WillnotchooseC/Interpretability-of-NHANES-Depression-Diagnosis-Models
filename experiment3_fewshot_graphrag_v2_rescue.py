"""
Experiment 2 — few-shot + exact GraphRAG: same pipeline as Experiment 1, but the Cypher stage also uses the original input to recover entities omitted by decomposition, resolves simple dynamic references from previous graph rows, and retrieves exact live graph-path grounding before Cypher generation. No neighborhood rescue is used.

Experimental control
--------------------
All three scripts use the same:
- decomposition parsing;
- 12 training-derived few-shot demonstrations;
- Cypher LLM and decoding settings;
- sequential execution of searching sub-questions;
- Neo4j execution;
- final-answer synthesis prompt.

The ONLY intended experimental switches are:
    USE_GRAPHRAG = True
    USE_NEIGHBORHOOD_RESCUE = False

Input JSON should contain, per sample:
    input: original user question
    prediction: predicted decomposition text
`output` is accepted as a fallback for prediction so gold/reference files can
also be dry-run through the same pipeline.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import time
from typing import Any

import requests
import torch
from neo4j import GraphDatabase
from transformers import AutoModelForCausalLM, AutoTokenizer


EXPERIMENT_NAME = 'exp3_fewshot_graphrag_v2_rescue'
USE_GRAPHRAG = True
USE_NEIGHBORHOOD_RESCUE = True

MODEL_NAME = "Qwen/Qwen3-8B-Base"
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "12345678")
SGLANG_PORT = 30000
SGLANG_TIMEOUT = 90
MAX_RETRIES = 3
PRIMARY_ROW_LIMIT = 50
RESCUE_ROW_LIMIT = 40


# ---------------------------------------------------------------------------
# Static schema + exactly the same 12 few-shot demonstrations in all 3 groups.
# ---------------------------------------------------------------------------

STATIC_SCHEMA = """Graph schema (labels/relationship/property names are case-sensitive):
  (:patient)-[:HAS_PREDICTION]->(:prediction)
  (:prediction)-[:GENERATED_BY]->(:model_run)
  (:prediction)-[:EXPLAINED_BY]->(:local_explanation)
  (:local_explanation)-[:GENERATED_BY]->(:model_run)
  (:prediction)-[pr:PREDICTS]->(:condition)              # pr.label, pr.probability
  (:patient)-[m:HAS_BIOMARKER_VALUE]->(:biomarker)      # m.value
  (:local_explanation)-[fi:FEATURE_IMPORTANCE]->(:biomarker)
                                                            # fi.scope='local', fi.shapValue, fi.absShapValue, fi.direction
  (:model_run)-[fi:FEATURE_IMPORTANCE]->(:biomarker)     # fi.scope='global', fi.influence
  (:model_run)-[s:SHAP_SUMMARY]->(:biomarker)            # s.rankByMeanAbsShap, s.meanAbsShap

model_run selection properties: modelType, modelFamily, classRatio.
"""

FEW_SHOT_DEMOS = [
    ("What is patient 6's vitamin D3 value?",
     "MATCH (p:patient)-[m:HAS_BIOMARKER_VALUE]->(b:biomarker) WHERE toInteger(p.id) = 6 AND b.name = 'Vitamin D3' RETURN b.name AS biomarker, m.value AS value"),
    ("What depression probability does decision tree predict for patient 115 when trained with positive-to-negative ratios of 1:1, 1:2, 1:3, and 1:4?",
     "MATCH (p:patient)-[:HAS_PREDICTION]->(pred:prediction)-[:GENERATED_BY]->(mr:model_run) MATCH (pred)-[pr:PREDICTS]->(c:condition) WHERE toInteger(p.id) = 115 AND mr.modelType IN ['decisiontree'] AND mr.classRatio IN ['1:1', '1:2', '1:3', '1:4'] AND c.name = 'depression' RETURN mr.modelFamily AS model, mr.classRatio AS ratio, pr.label AS predicted_label, pr.probability AS depression_probability, pred.correct AS correct ORDER BY mr.classRatio, model"),
    ("Was the logistic regression prediction for patient 217, using a model trained with a 1:1 positive-to-negative ratio, correct?",
     "MATCH (p:patient)-[:HAS_PREDICTION]->(pred:prediction)-[:GENERATED_BY]->(mr:model_run) MATCH (pred)-[pr:PREDICTS]->(c:condition) WHERE toInteger(p.id) = 217 AND mr.modelType IN ['logisticregression'] AND mr.classRatio IN ['1:1'] AND c.name = 'depression' RETURN mr.modelFamily AS model, mr.classRatio AS ratio, pr.label AS predicted_label, pr.probability AS depression_probability, pred.correct AS correct ORDER BY mr.classRatio, model"),
    ("For patient 6 in naive bayes trained with a 1:1 positive-to-negative ratio, what is the local SHAP value of vitamin D3, and is its contribution positive or negative?",
     "MATCH (p:patient)-[:HAS_PREDICTION]->(pred:prediction)-[:EXPLAINED_BY]->(le:local_explanation)-[:GENERATED_BY]->(mr:model_run) MATCH (le)-[fi:FEATURE_IMPORTANCE]->(b:biomarker) WHERE toInteger(p.id) = 6 AND mr.modelType IN ['naivebayes'] AND mr.classRatio IN ['1:1'] AND b.name IN ['Vitamin D3'] AND fi.scope = 'local' RETURN mr.modelFamily AS model, mr.classRatio AS ratio, b.name AS biomarker, fi.shapValue AS shap_value, fi.absShapValue AS abs_shap_value, fi.direction AS direction ORDER BY mr.classRatio, model, biomarker"),
    ("For patient 97 in decision tree trained with a 1:1 positive-to-negative ratio, what local SHAP value does each available biomarker have?",
     "MATCH (p:patient)-[:HAS_PREDICTION]->(pred:prediction)-[:EXPLAINED_BY]->(le:local_explanation)-[:GENERATED_BY]->(mr:model_run) MATCH (le)-[fi:FEATURE_IMPORTANCE]->(b:biomarker) WHERE toInteger(p.id) = 97 AND mr.modelType IN ['decisiontree'] AND mr.classRatio IN ['1:1'] AND fi.scope = 'local' RETURN mr.modelFamily AS model, mr.classRatio AS ratio, b.name AS biomarker, fi.shapValue AS shap_value, fi.absShapValue AS abs_shap_value, fi.direction AS direction ORDER BY mr.classRatio, model, biomarker"),
    ("For patient 204 in decision tree trained with a 1:4 positive-to-negative ratio, which 3 biomarkers have the largest positive local SHAP values?",
     "MATCH (p:patient)-[:HAS_PREDICTION]->(pred:prediction)-[:EXPLAINED_BY]->(le:local_explanation)-[:GENERATED_BY]->(mr:model_run) MATCH (le)-[fi:FEATURE_IMPORTANCE]->(b:biomarker) WHERE toInteger(p.id) = 204 AND mr.modelType = 'decisiontree' AND mr.classRatio = '1:4' AND fi.scope = 'local' AND fi.direction = 'positive' RETURN mr.modelFamily AS model, mr.classRatio AS ratio, b.name AS biomarker, fi.shapValue AS shap_value, fi.absShapValue AS abs_shap_value, fi.direction AS direction ORDER BY fi.shapValue DESC, b.name LIMIT 3"),
    ("For patient 155 in random forest trained with a 1:4 positive-to-negative ratio, which biomarker has the largest absolute local SHAP value?",
     "MATCH (p:patient)-[:HAS_PREDICTION]->(pred:prediction)-[:EXPLAINED_BY]->(le:local_explanation)-[:GENERATED_BY]->(mr:model_run) MATCH (le)-[fi:FEATURE_IMPORTANCE]->(b:biomarker) WHERE toInteger(p.id) = 155 AND mr.modelType = 'randomforest' AND mr.classRatio = '1:4' AND fi.scope = 'local' RETURN mr.modelFamily AS model, mr.classRatio AS ratio, b.name AS biomarker, fi.shapValue AS shap_value, fi.absShapValue AS abs_shap_value, fi.direction AS direction ORDER BY fi.absShapValue DESC, b.name LIMIT 1"),
    ("For random forest trained with a 1:2 positive-to-negative ratio, what are the top 3 biomarkers ranked by global SHAP importance, and what is the global SHAP importance of each?",
     "MATCH (mr:model_run)-[s:SHAP_SUMMARY]->(b:biomarker) WHERE mr.modelType IN ['randomforest'] AND mr.classRatio IN ['1:2'] AND s.rankByMeanAbsShap <= 3 RETURN mr.modelFamily AS model, mr.classRatio AS ratio, b.name AS biomarker, s.rankByMeanAbsShap AS rank, s.meanAbsShap AS global_shap_importance ORDER BY mr.classRatio, model, rank, biomarker"),
    ("For total cholesterol, what is the global SHAP tendency as the biomarker value increases in random forest, naive bayes, logistic regression, and decision tree at training ratios 1:1, 1:2, 1:3, 1:4, and 1:5?",
     "MATCH (mr:model_run)-[fi:FEATURE_IMPORTANCE]->(b:biomarker) WHERE mr.modelType IN ['randomforest', 'naivebayes', 'logisticregression', 'decisiontree'] AND mr.classRatio IN ['1:1', '1:2', '1:3', '1:4', '1:5'] AND fi.scope = 'global' AND b.name IN ['Total Cholesterol'] RETURN mr.modelFamily AS model, mr.classRatio AS ratio, b.name AS biomarker, fi.influence AS increase_related_tendency ORDER BY mr.classRatio, model, biomarker"),
    ("For random forest and decision tree trained with a 1:1 positive-to-negative ratio, what are the top 3 biomarkers ranked by global SHAP importance, what is each biomarker's global SHAP importance, and what is its global SHAP tendency as the biomarker value increases?",
     "MATCH (mr:model_run)-[s:SHAP_SUMMARY]->(b:biomarker) MATCH (mr)-[fi:FEATURE_IMPORTANCE]->(b) WHERE mr.modelType IN ['randomforest', 'decisiontree'] AND mr.classRatio IN ['1:1'] AND fi.scope = 'global' AND s.rankByMeanAbsShap <= 3 RETURN mr.modelFamily AS model, mr.classRatio AS ratio, b.name AS biomarker, s.rankByMeanAbsShap AS rank, s.meanAbsShap AS global_shap_importance, fi.influence AS increase_related_tendency ORDER BY mr.classRatio, model, rank, biomarker"),
    ("What is the accuracy of random forest trained at 1:2, 1:3, 1:4, and 1:5?",
     "MATCH (pred:prediction)-[:GENERATED_BY]->(mr:model_run) WHERE mr.modelType = 'randomforest' AND mr.classRatio IN ['1:2', '1:3', '1:4', '1:5'] RETURN mr.classRatio AS ratio, avg(CASE WHEN pred.correct THEN 1.0 ELSE 0.0 END) AS accuracy, count(pred) AS prediction_count ORDER BY ratio"),
    ("What is the clinical meaning of creatinine?",
     "MATCH (b:biomarker) WHERE b.name = 'Creatinine' RETURN b.name AS biomarker, b.description AS description"),
]

HARD_RULES = """Hard rules:
1. Filter model runs using mr.modelType and mr.classRatio. Do NOT synthesize dynamic per-run labels and do NOT use modelName.
2. Use toInteger(p.id) for patient ids.
3. Patient biomarker measurements are relationship property m.value.
4. Patient-specific SHAP uses patient -> prediction -> local_explanation -> model_run and fi.scope='local'.
5. Global rank/importance uses SHAP_SUMMARY. Global tendency uses FEATURE_IMPORTANCE with fi.scope='global'.
6. Prediction label/probability uses prediction-[pr:PREDICTS]->condition; for depression use c.name='depression'.
7. Preserve every requested model and ratio unless a singular dynamic reference has been resolved from upstream evidence.
8. Use canonical biomarker names supplied in graph context; do not invent graph entities.
9. Output ONE executable Cypher query only, no markdown and no explanation.
10. ONE query means ONE chain of MATCH/WHERE clauses ending in exactly ONE RETURN. Never write a second
    MATCH after a RETURN and never emit two RETURN clauses. If the sub-question asks for two facts
    (e.g. a measurement AND a SHAP value), put both patterns as MATCH clauses of the same query and
    return both in the single RETURN.
11. HAS_BIOMARKER_VALUE, HAS_PREDICTION, PREDICTS, FEATURE_IMPORTANCE and SHAP_SUMMARY are relationship
    types. They are NEVER node labels: write -[fi:FEATURE_IMPORTANCE]->, never (fi:FEATURE_IMPORTANCE).
12. Answer-bearing properties (m.value, fi.shapValue, fi.absShapValue, fi.direction, fi.influence,
    pr.probability, pr.label, pred.correct, s.meanAbsShap) go in RETURN / ORDER BY. Put one in WHERE
    only when the CURRENT sub-question itself states that value or direction (e.g. "largest positive").
13. If a demonstration below has the same question pattern as the sub-question, reuse that
    demonstration's Cypher structure exactly and only substitute the entity values.
"""


def _render_demos() -> str:
    chunks = []
    for i, (q, c) in enumerate(FEW_SHOT_DEMOS, 1):
        chunks.append(f"# {i}\nQuestion: {json.dumps(q)}\nCypher: {c}")
    return "\n\n".join(chunks)


# ---------------------------------------------------------------------------
# Decomposition parsing (identical across all three groups).
# ---------------------------------------------------------------------------

def truncate_output(text: str) -> str:
    text = text or ""
    cut_pos = len(text)
    for marker in ["�", "<|im_start|>", "\x00"]:
        idx = text.find(marker)
        if 0 <= idx < cut_pos:
            cut_pos = idx
    m = re.search(r"\nuser\n|\nassistant\n", text)
    if m:
        cut_pos = min(cut_pos, m.start())
    lines = text[:cut_pos].splitlines()

    # Keep only first monotonic Step 1,2,... block.
    out = []
    last_step = 0
    for line in lines:
        sm = re.match(r"\s*Step\s+(\d+):", line)
        if sm:
            n = int(sm.group(1))
            if n <= last_step:
                break
            last_step = n
        out.append(line)

    last_type = -1
    for i, line in enumerate(out):
        if re.search(r"#\s*Type:\s*(searching|aggregation)", line):
            last_type = i
    if last_type >= 0:
        out = out[:last_type + 1]
    return "\n".join(out).strip()


def extract_subquestions(text: str) -> list[dict]:
    result = []
    for m in re.finditer(
        r"Sub-question:\s*(.+?)\s*#\s*Type:\s*(searching|aggregation)",
        text or "",
    ):
        q, t = m.group(1).strip(), m.group(2).strip()
        if t != "searching":
            continue
        if not any(x["question"] == q for x in result):
            result.append({"question": q, "type": t})
    return result


def get_decomposition_text(sample: dict) -> str:
    for key in ("prediction", "output", "reference"):
        if sample.get(key):
            return truncate_output(str(sample[key]))
    return ""


# ---------------------------------------------------------------------------
# LLM helpers.
# ---------------------------------------------------------------------------

_ENTITY_TOKENS_RE = re.compile(r"toInteger\(p\.id\)\s*=\s*\d+|modelType|classRatio|b\.name|c\.name", re.I)


def repair_concatenated_queries(cypher: str) -> tuple[str, bool]:
    """
    v1 failure mode: the model emitted two independent queries back to back
    ("MATCH ... RETURN ... MATCH ... RETURN ..."), a Neo4j syntax error. Split at
    every MATCH that directly follows a RETURN clause and keep the block that
    carries the most entity filters (patient id / modelType / classRatio /
    names) -- deterministic, no model call. Returns (cypher, repaired_flag).
    """
    if len(re.findall(r"\bRETURN\b", cypher, re.I)) < 2:
        return cypher, False
    blocks = re.split(r"(?<=\S)\s+(?=MATCH\b)", cypher, flags=re.I)
    # re-join pieces that do not themselves contain a RETURN (a query's inner MATCHes)
    merged = []
    for b in blocks:
        if merged and not re.search(r"\bRETURN\b", merged[-1], re.I):
            merged[-1] = merged[-1] + " " + b
        else:
            merged.append(b)
    full = [b for b in merged if re.search(r"\bRETURN\b", b, re.I)]
    if len(full) < 2:
        return cypher, False
    best = max(full, key=lambda b: (len(_ENTITY_TOKENS_RE.findall(b)), len(b)))
    return best.strip(), True


def clean_cypher_output(text: str) -> str:
    text = (text or "").strip()
    if not text:
        return ""
    fence = re.search(r"```(?:cypher)?\s*(.*?)```", text, re.I | re.S)
    if fence:
        text = fence.group(1).strip()
    text = re.sub(r"^\s*Cypher\s*:\s*", "", text, flags=re.I)
    text = re.split(r"\n\s*(?:Question|Explanation|Answer|Final answer)\s*:", text, maxsplit=1, flags=re.I)[0]
    return " ".join(text.split()).strip()


def load_model(model_name: str):
    print(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    return model, tokenizer


def generate_local(model, tokenizer, prompt: str, max_new_tokens: int) -> str:
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            repetition_penalty=1.05,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
        )
    ids = out[0, inputs["input_ids"].shape[1]:]
    return tokenizer.decode(ids, skip_special_tokens=True).strip()


def generate_sglang(prompt: str, sglang_url: str, max_new_tokens: int,
                     stop: list[str] | None = None) -> str:
    payload = {
        "model": "default",
        "prompt": prompt,
        "max_tokens": max_new_tokens,
        "temperature": 0,
        "top_p": 1.0,
        "repetition_penalty": 1.05,
    }
    if stop:
        payload["stop"] = stop
    for attempt in range(MAX_RETRIES):
        try:
            r = requests.post(f"{sglang_url}/v1/completions", json=payload, timeout=SGLANG_TIMEOUT)
            r.raise_for_status()
            return r.json()["choices"][0]["text"].strip()
        except Exception as e:
            if attempt + 1 == MAX_RETRIES:
                print(f"SGLang generation failed: {e}")
                return ""
            time.sleep(2)
    return ""


def generate_text(prompt: str, model, tokenizer, sglang_url: str | None,
                  max_new_tokens: int, stop: list[str] | None = None) -> str:
    if sglang_url:
        return generate_sglang(prompt, sglang_url, max_new_tokens, stop=stop)
    return generate_local(model, tokenizer, prompt, max_new_tokens)


def launch_sglang_server(model_path: str, port: int):
    log_path = f"/tmp/sglang_{EXPERIMENT_NAME}_{port}.log"
    log_file = open(log_path, "w")
    cmd = [
        "python", "-m", "sglang.launch_server",
        "--model-path", model_path,
        "--port", str(port),
        "--dtype", "bfloat16",
        "--host", "0.0.0.0",
        "--max-running-requests", "64",
        "--mem-fraction-static", "0.88",
    ]
    print("Launching SGLang:", " ".join(cmd))
    proc = subprocess.Popen(cmd, stdout=log_file, stderr=subprocess.STDOUT, text=True)
    health = f"http://localhost:{port}/health"
    start = time.time()
    while time.time() - start < 900:
        try:
            r = requests.get(health, timeout=3)
            if r.status_code == 200:
                print(f"SGLang ready at http://localhost:{port} (log: {log_path})")
                return proc
        except Exception:
            pass
        if proc.poll() is not None:
            print(f"SGLang exited early with code {proc.returncode}; log: {log_path}")
            return None
        time.sleep(3)
    proc.terminate()
    print("SGLang start timeout")
    return None


def shutdown_sglang_server(proc):
    if proc and proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except Exception:
            proc.kill()


# ---------------------------------------------------------------------------
# Neo4j execution.
# ---------------------------------------------------------------------------

def execute_cypher(driver, cypher: str, database: str | None = None,
                   params: dict | None = None) -> dict:
    if not cypher:
        return {"success": False, "has_results": False, "error": "empty cypher", "rows": 0, "records": []}
    try:
        with driver.session(database=database) as session:
            records = [r.data() for r in session.run(cypher, params or {})]
        return {
            "success": True,
            "has_results": bool(records),
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


# ---------------------------------------------------------------------------
# GraphRAG entity recovery / exact path grounding. In group 1 these functions
# are present but never called, keeping the execution scaffold otherwise equal.
# ---------------------------------------------------------------------------

def _norm(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", (text or "").lower())


def _lower_words(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", (text or "").lower())).strip()


def build_vocab(driver, database=None) -> dict:
    def run(q, params=None):
        with driver.session(database=database) as session:
            return [r.data() for r in session.run(q, params or {})]

    biomarkers = run("MATCH (b:biomarker) RETURN DISTINCT b.name AS name, b.aliases AS aliases, b.key AS key")
    model_runs = run("MATCH (mr:model_run) RETURN DISTINCT mr.modelType AS modelType, mr.modelFamily AS modelFamily, mr.classRatio AS classRatio")
    conditions = run("MATCH (c:condition) RETURN DISTINCT c.name AS name")

    biomarker_alias_to_name = {}
    for row in biomarkers:
        name = row.get("name")
        if not name:
            continue
        aliases = [name, row.get("key")]
        extra = row.get("aliases")
        if isinstance(extra, (list, tuple)):
            aliases.extend(extra)
        elif extra:
            aliases.append(extra)
        for alias in aliases:
            if alias:
                biomarker_alias_to_name[_norm(str(alias))] = str(name)

    model_alias_to_type = {}
    for row in model_runs:
        mt = row.get("modelType")
        mf = row.get("modelFamily")
        if mt:
            model_alias_to_type[_norm(str(mt))] = str(mt)
            if mf:
                model_alias_to_type[_norm(str(mf))] = str(mt)
    for mt, surfaces in {
        "naivebayes": ["naive bayes", "naivebayes"],
        "decisiontree": ["decision tree", "decisiontree"],
        "logisticregression": ["logistic regression", "logisticregression"],
        "randomforest": ["random forest", "randomforest"],
    }.items():
        if mt in set(model_alias_to_type.values()):
            for s in surfaces:
                model_alias_to_type[_norm(s)] = mt

    condition_alias_to_name = {}
    for row in conditions:
        if row.get("name"):
            condition_alias_to_name[_norm(str(row["name"]))] = str(row["name"])

    return {
        "biomarker_alias_to_name": biomarker_alias_to_name,
        "model_alias_to_type": model_alias_to_type,
        "condition_alias_to_name": condition_alias_to_name,
    }


def infer_query_type(question: str) -> str:
    q = _lower_words(question)
    if any(x in q for x in ["clinical meaning", "clinical concept", "definition of", "represent clinically"]):
        return "biomarker_description"
    if "accuracy" in q:
        return "model_accuracy"
    if ("largest positive" in q or "strongest positive" in q) and ("shap" in q or "contribution" in q):
        return "local_top_positive"
    if (("largest absolute" in q and "shap" in q) or
        ("strongest local" in q) or
        ("most strongly" in q and ("push" in q or "influence" in q))):
        return "local_max_abs"
    if "global shap" in q and any(x in q for x in ["top ", "ranked", "importance"]):
        if any(x in q for x in ["tendency", "as each", "increases"]):
            return "global_rank_plus_tendency"
        return "global_rank"
    if any(x in q for x in ["global shap tendency", "as the biomarker value increases", "as that biomarker increases", "as each biomarker"]):
        return "global_tendency"
    if "local shap" in q or ("biomarker" in q and any(x in q for x in ["push", "local influence", "contribution"] )):
        return "local_shap"
    if any(x in q for x in ["correct", "prediction label", "depression probability", "what prediction", "predict for patient", "predicted probability"]):
        return "prediction"
    if "patient" in q and any(x in q for x in [" value", "measurement", "recorded value", "recorded for patient"]):
        return "patient_biomarker_value"
    return "unknown"


def extract_top_k(question: str) -> int | None:
    for pat in [r"\btop\s+(\d+)\b", r"\bwhich\s+(\d+)\s+(?:biomarkers?|features?)\b", r"\b(\d+)\s+(?:strongest|largest|top)\b"]:
        m = re.search(pat, question or "", re.I)
        if m:
            return int(m.group(1))
    if re.search(r"\bwhich biomarker\b.*\b(strongest|largest|most)\b", question or "", re.I):
        return 1
    return None


def extract_entities(text: str, vocab: dict) -> dict:
    text = text or ""
    qn = _norm(text)
    patient_ids = sorted({int(x) for x in re.findall(r"\bpatient\s+(\d+)\b", text, re.I)})

    biomarkers = []
    for alias, canonical in sorted(vocab.get("biomarker_alias_to_name", {}).items(), key=lambda kv: len(kv[0]), reverse=True):
        if alias and alias in qn and canonical not in biomarkers:
            biomarkers.append(canonical)

    model_types = []
    for alias, canonical in sorted(vocab.get("model_alias_to_type", {}).items(), key=lambda kv: len(kv[0]), reverse=True):
        if alias and alias in qn and canonical not in model_types:
            model_types.append(canonical)

    ratios = []
    for m in re.finditer(r"\b1\s*(?::|-|to)\s*(\d+)\b", text, re.I):
        ratio = f"1:{m.group(1)}"
        if ratio not in ratios:
            ratios.append(ratio)

    conditions = []
    for alias, canonical in vocab.get("condition_alias_to_name", {}).items():
        if alias and alias in qn and canonical not in conditions:
            conditions.append(canonical)

    return {
        "patient_ids": patient_ids,
        "biomarkers": biomarkers,
        "model_types": model_types,
        "ratios": ratios,
        "conditions": conditions,
        "top_k": extract_top_k(text),
    }


def _unique(values):
    out = []
    for x in values or []:
        if x is not None and x not in out:
            out.append(x)
    return out


def _numeric(row: dict, keys: list[str]):
    for k in keys:
        v = row.get(k)
        if isinstance(v, (int, float)):
            return float(v)
        try:
            if v is not None:
                return float(v)
        except Exception:
            pass
    return None


def resolve_dynamic_reference(question: str, upstream_rows: list[dict]) -> dict:
    """Resolve dependent references from actual previous graph rows.

    Aggregation sub-questions are not sent to Cypher, so when the next searching
    atom says "the selected/highest/best ratio" we deterministically reproduce
    the simple selector over the previous rows. This is graph-side binding, not
    another LLM call.
    """
    q = (question or "").lower()
    rows = upstream_rows or []
    out = {"patient_ids": [], "biomarkers": [], "model_types": [], "ratios": []}
    if not rows:
        return out

    # Singular / plural biomarker coreference.
    if re.search(r"\b(that|selected|same)\s+(biomarker|feature)\b|\bthose(?: top)?\s*\d*\s*(biomarkers|features)\b", q):
        ranked = list(rows)
        if any(k in r for r in rows for k in ["abs_shap_value", "absolute_local_shap"]):
            ranked.sort(key=lambda r: _numeric(r, ["abs_shap_value", "absolute_local_shap"]) or float("-inf"), reverse=True)
        elif any(k in r for r in rows for k in ["global_shap_importance", "importance"]):
            ranked.sort(key=lambda r: _numeric(r, ["global_shap_importance", "importance"]) or float("-inf"), reverse=True)
        elif any(k in r for r in rows for k in ["shap_value", "local_shap"]):
            ranked.sort(key=lambda r: abs(_numeric(r, ["shap_value", "local_shap"]) or 0.0), reverse=True)
        k = extract_top_k(question) or (1 if re.search(r"\b(that|selected|same)\s+(biomarker|feature)\b", q) else len(ranked))
        out["biomarkers"] = _unique([
            r.get("biomarker") or r.get("name") or r.get("b.name") for r in ranked[:k]
        ])

    # Selected/best/highest/lowest ratio after a probability/accuracy retrieval.
    if re.search(r"\b(selected|that|best(?:-performing)?|highest(?:-probability)?|lowest(?:-probability)?)\s+ratio\b", q):
        scored = []
        for r in rows:
            ratio = r.get("ratio") or r.get("classRatio")
            score = _numeric(r, ["probability", "depression_probability", "accuracy"])
            if ratio is not None and score is not None:
                scored.append((score, str(ratio)))
        if scored:
            choose_min = "lowest" in q
            score, ratio = min(scored) if choose_min else max(scored)
            out["ratios"] = [ratio]

    if re.search(r"\b(selected|that|best|highest)\s+model\b", q):
        scored = []
        for r in rows:
            model = r.get("model_type") or r.get("model")
            score = _numeric(r, ["probability", "depression_probability", "accuracy", "abs_shap_value"])
            if model is not None and score is not None:
                scored.append((score, str(model)))
        if scored:
            out["model_types"] = [max(scored)[1]]

    return out


def merge_entities(subquestion: str, original_question: str, upstream_rows: list[dict], vocab: dict) -> tuple[dict, dict]:
    """Current sub-question is authoritative; graph bindings come next; original
    input only fills omitted slots. Returns merged entities + provenance."""
    sq = extract_entities(subquestion, vocab)
    oq = extract_entities(original_question, vocab)
    dyn = resolve_dynamic_reference(subquestion, upstream_rows)

    merged = {}
    provenance = {}
    for key in ["patient_ids", "biomarkers", "model_types", "ratios", "conditions"]:
        if sq.get(key):
            merged[key] = sq[key]
            provenance[key] = "subquestion"
        elif dyn.get(key):
            merged[key] = dyn[key]
            provenance[key] = "upstream_graph_rows"
        elif oq.get(key):
            merged[key] = oq[key]
            provenance[key] = "original_input"
        else:
            merged[key] = []
            provenance[key] = "missing"
    merged["top_k"] = sq.get("top_k") or oq.get("top_k")
    provenance["top_k"] = "subquestion" if sq.get("top_k") else ("original_input" if oq.get("top_k") else "missing")
    return merged, provenance


PRIMARY_PATHS = {
    "patient_biomarker_value": "(:patient)-[HAS_BIOMARKER_VALUE]->(:biomarker)",
    "prediction": "(:patient)-[:HAS_PREDICTION]->(:prediction)-[:GENERATED_BY]->(:model_run); (:prediction)-[:PREDICTS]->(:condition)",
    "local_shap": "(:patient)-[:HAS_PREDICTION]->(:prediction)-[:EXPLAINED_BY]->(:local_explanation)-[:GENERATED_BY]->(:model_run); (:local_explanation)-[:FEATURE_IMPORTANCE]->(:biomarker)",
    "local_top_positive": "same local-SHAP path, restricted to fi.scope='local' and positive direction",
    "local_max_abs": "same local-SHAP path, ordered by absShapValue",
    "global_rank": "(:model_run)-[:SHAP_SUMMARY]->(:biomarker)",
    "global_tendency": "(:model_run)-[:FEATURE_IMPORTANCE {scope:'global'}]->(:biomarker)",
    "global_rank_plus_tendency": "(:model_run)-[:SHAP_SUMMARY]->(:biomarker) plus (:model_run)-[:FEATURE_IMPORTANCE {scope:'global'}]->(:biomarker)",
    "model_accuracy": "(:prediction)-[:GENERATED_BY]->(:model_run)",
    "biomarker_description": "(:biomarker)",
    "unknown": "No exact atomic path confidently inferred.",
}


def _base_params(e: dict):
    return {
        "patient_ids": e.get("patient_ids") or [],
        "biomarkers": e.get("biomarkers") or [],
        "model_types": e.get("model_types") or [],
        "ratios": e.get("ratios") or [],
        "conditions": e.get("conditions") or [],
        "top_k": e.get("top_k"),
        "row_limit": PRIMARY_ROW_LIMIT,
    }


def primary_grounding_spec(query_type: str, e: dict) -> tuple[str, dict] | None:
    p = _base_params(e)
    if query_type == "patient_biomarker_value":
        return ("""
MATCH (p:patient)-[m:HAS_BIOMARKER_VALUE]->(b:biomarker)
WHERE (size($patient_ids)=0 OR toInteger(p.id) IN $patient_ids)
  AND (size($biomarkers)=0 OR b.name IN $biomarkers)
RETURN toInteger(p.id) AS patient_id, b.name AS biomarker, m.value AS value
ORDER BY patient_id, biomarker LIMIT $row_limit
""".strip(), p)
    if query_type == "biomarker_description":
        return ("""
MATCH (b:biomarker)
WHERE size($biomarkers)=0 OR b.name IN $biomarkers
RETURN b.name AS biomarker, b.description AS description
ORDER BY biomarker LIMIT $row_limit
""".strip(), p)
    if query_type == "prediction":
        return ("""
MATCH (p:patient)-[:HAS_PREDICTION]->(pred:prediction)-[:GENERATED_BY]->(mr:model_run)
MATCH (pred)-[pr:PREDICTS]->(c:condition)
WHERE (size($patient_ids)=0 OR toInteger(p.id) IN $patient_ids)
  AND (size($model_types)=0 OR mr.modelType IN $model_types)
  AND (size($ratios)=0 OR mr.classRatio IN $ratios)
  AND (size($conditions)=0 OR c.name IN $conditions OR c.name='depression')
RETURN toInteger(p.id) AS patient_id, mr.modelType AS model_type, mr.modelFamily AS model,
       mr.classRatio AS ratio, c.name AS condition, pr.label AS predicted_label,
       pr.probability AS depression_probability, pred.correct AS correct
ORDER BY patient_id, ratio, model_type LIMIT $row_limit
""".strip(), p)
    if query_type in {"local_shap", "local_top_positive", "local_max_abs"}:
        extra = ""
        order = "ORDER BY patient_id, ratio, model_type, biomarker"
        if query_type == "local_top_positive":
            extra = "AND fi.direction='positive'"
            order = "ORDER BY fi.shapValue DESC, biomarker"
        elif query_type == "local_max_abs":
            order = "ORDER BY fi.absShapValue DESC, biomarker"
            # Do not cap row_limit by top_k: a "top-K toward AND top-K away"
            # question needs every biomarker's signed SHAP value retrieved
            # before ranking/grouping by sign happens downstream. Capping to
            # ~top_k rows sorted by |value| alone can be dominated by one sign
            # and silently drop the other direction. Each patient/model/ratio
            # triple has ~12 biomarkers total, well under PRIMARY_ROW_LIMIT.
        return (f"""
MATCH (p:patient)-[:HAS_PREDICTION]->(pred:prediction)-[:EXPLAINED_BY]->(le:local_explanation)-[:GENERATED_BY]->(mr:model_run)
MATCH (le)-[fi:FEATURE_IMPORTANCE]->(b:biomarker)
WHERE (size($patient_ids)=0 OR toInteger(p.id) IN $patient_ids)
  AND (size($model_types)=0 OR mr.modelType IN $model_types)
  AND (size($ratios)=0 OR mr.classRatio IN $ratios)
  AND (size($biomarkers)=0 OR b.name IN $biomarkers)
  AND fi.scope='local' {extra}
RETURN toInteger(p.id) AS patient_id, mr.modelType AS model_type, mr.modelFamily AS model,
       mr.classRatio AS ratio, b.name AS biomarker, fi.shapValue AS shap_value,
       fi.absShapValue AS abs_shap_value, fi.direction AS direction
{order} LIMIT $row_limit
""".strip(), p)
    if query_type == "model_accuracy":
        return ("""
MATCH (pred:prediction)-[:GENERATED_BY]->(mr:model_run)
WHERE (size($model_types)=0 OR mr.modelType IN $model_types)
  AND (size($ratios)=0 OR mr.classRatio IN $ratios)
RETURN mr.modelType AS model_type, mr.modelFamily AS model, mr.classRatio AS ratio,
       avg(CASE WHEN pred.correct THEN 1.0 ELSE 0.0 END) AS accuracy,
       count(pred) AS prediction_count
ORDER BY ratio, model_type LIMIT $row_limit
""".strip(), p)
    if query_type == "global_rank":
        return ("""
MATCH (mr:model_run)-[s:SHAP_SUMMARY]->(b:biomarker)
WHERE (size($model_types)=0 OR mr.modelType IN $model_types)
  AND (size($ratios)=0 OR mr.classRatio IN $ratios)
  AND (size($biomarkers)=0 OR b.name IN $biomarkers)
  AND ($top_k IS NULL OR s.rankByMeanAbsShap <= $top_k)
RETURN mr.modelType AS model_type, mr.modelFamily AS model, mr.classRatio AS ratio,
       b.name AS biomarker, s.rankByMeanAbsShap AS rank,
       s.meanAbsShap AS global_shap_importance
ORDER BY ratio, model_type, rank, biomarker LIMIT $row_limit
""".strip(), p)
    if query_type == "global_tendency":
        return ("""
MATCH (mr:model_run)-[fi:FEATURE_IMPORTANCE]->(b:biomarker)
WHERE (size($model_types)=0 OR mr.modelType IN $model_types)
  AND (size($ratios)=0 OR mr.classRatio IN $ratios)
  AND (size($biomarkers)=0 OR b.name IN $biomarkers)
  AND fi.scope='global'
RETURN mr.modelType AS model_type, mr.modelFamily AS model, mr.classRatio AS ratio,
       b.name AS biomarker, fi.influence AS increase_related_tendency
ORDER BY ratio, model_type, biomarker LIMIT $row_limit
""".strip(), p)
    if query_type == "global_rank_plus_tendency":
        return ("""
MATCH (mr:model_run)-[s:SHAP_SUMMARY]->(b:biomarker)
MATCH (mr)-[fi:FEATURE_IMPORTANCE]->(b)
WHERE (size($model_types)=0 OR mr.modelType IN $model_types)
  AND (size($ratios)=0 OR mr.classRatio IN $ratios)
  AND (size($biomarkers)=0 OR b.name IN $biomarkers)
  AND ($top_k IS NULL OR s.rankByMeanAbsShap <= $top_k)
  AND fi.scope='global'
RETURN mr.modelType AS model_type, mr.modelFamily AS model, mr.classRatio AS ratio,
       b.name AS biomarker, s.rankByMeanAbsShap AS rank,
       s.meanAbsShap AS global_shap_importance,
       fi.influence AS increase_related_tendency
ORDER BY ratio, model_type, rank, biomarker LIMIT $row_limit
""".strip(), p)
    return None


# v2: the live grounding rows are reduced to ENTITY-IDENTITY fields only. Answer-bearing
# values (shap_value, direction, probability, accuracy, ...) are withheld and replaced by the
# NAMES of the schema properties that carry them on this path. In v1 the model sometimes copied
# a grounded answer value into the WHERE clause (e.g. `fi.direction = 'negative'`), turning the
# grounding into a wrong filter and returning 0 rows.
IDENTITY_KEYS = ("patient_id", "model_type", "model", "ratio", "biomarker", "condition")
ANSWER_PROPERTIES = {
    "patient_biomarker_value": ["m.value"],
    "prediction": ["pr.label", "pr.probability", "pred.correct"],
    "local_shap": ["fi.shapValue", "fi.absShapValue", "fi.direction"],
    "local_top_positive": ["fi.shapValue", "fi.absShapValue", "fi.direction"],
    "local_max_abs": ["fi.shapValue", "fi.absShapValue", "fi.direction"],
    "global_rank": ["s.rankByMeanAbsShap", "s.meanAbsShap"],
    "global_tendency": ["fi.influence"],
    "global_rank_plus_tendency": ["s.rankByMeanAbsShap", "s.meanAbsShap", "fi.influence"],
    "model_accuracy": ["avg(CASE WHEN pred.correct THEN 1.0 ELSE 0.0 END) AS accuracy", "count(pred)"],
    "biomarker_description": ["b.description"],
    "unknown": [],
}


def redact_grounding_rows(rows: list[dict]) -> list[dict]:
    """Keep identity fields only, de-duplicated, order preserved."""
    out, seen = [], set()
    for r in rows or []:
        ident = {k: r[k] for k in IDENTITY_KEYS if k in r}
        key = json.dumps(ident, sort_keys=True, default=str)
        if ident and key not in seen:
            seen.add(key)
            out.append(ident)
    return out


def get_primary_grounding(driver, database: str | None, query_type: str, entities: dict) -> dict:
    path = PRIMARY_PATHS.get(query_type, PRIMARY_PATHS["unknown"])
    answer_props = ANSWER_PROPERTIES.get(query_type, [])
    spec = primary_grounding_spec(query_type, entities)
    if spec is None:
        return {"path": path, "rows": [], "identity_rows": [], "answer_properties": answer_props, "error": None}
    cypher, params = spec
    res = execute_cypher(driver, cypher, database=database, params=params)
    rows = res.get("records", [])
    return {
        "path": path,
        "rows": rows,                                   # full rows kept for the output file / audit
        "identity_rows": redact_grounding_rows(rows),   # what the prompt actually sees
        "answer_properties": answer_props,
        "error": res.get("error"),
    }


REQUIRED_SLOTS = {
    "patient_biomarker_value": ["patient_ids", "biomarkers"],
    "prediction": ["patient_ids", "model_types", "ratios"],
    "local_shap": ["patient_ids", "model_types", "ratios"],
    "local_top_positive": ["patient_ids", "model_types", "ratios"],
    "local_max_abs": ["patient_ids", "model_types", "ratios"],
    "model_accuracy": ["model_types", "ratios"],
    "global_rank": ["model_types", "ratios"],
    "global_tendency": ["model_types", "ratios", "biomarkers"],
    "global_rank_plus_tendency": ["model_types", "ratios"],
    "biomarker_description": ["biomarkers"],
}


def missing_required_slots(query_type: str, entities: dict) -> list[str]:
    return [k for k in REQUIRED_SLOTS.get(query_type, []) if not entities.get(k)]


def redact_rescue_rows(rows: list[dict]) -> list[dict]:
    """v2 principle applied to rescue context: keep graph STRUCTURE (labels, ids,
    names, model/ratio identities, relationship types + direction) but replace each
    edge's property VALUES with the list of property NAMES, so answer values
    (shapValue, direction, probability, ...) can never leak into a WHERE clause."""
    out = []
    for r in rows:
        r2 = dict(r)
        if isinstance(r2.get("edges"), list):
            r2["edges"] = [
                {"rel_type": e.get("rel_type"), "from_labels": e.get("from_labels"),
                 "to_labels": e.get("to_labels"),
                 "property_names": sorted((e.get("props") or {}).keys())}
                for e in r2["edges"] if isinstance(e, dict)
            ]
        out.append(r2)
    return out


def retrieve_neighborhood_rescue(driver, database: str | None, entities: dict,
                                 total_limit: int = RESCUE_ROW_LIMIT) -> list[dict]:
    """Retrieve compact 1-2 hop neighborhoods around recovered entities.

    This is deliberately generic: it does not assume which exact relation the
    decomposition intended. It exposes nearby entity/relation evidence so the
    same Cypher LLM can recover when an atomic sub-question is incomplete.
    """
    rows = []
    per_anchor = max(6, total_limit // 3)

    queries = []
    if entities.get("patient_ids"):
        queries.append(("patient", """
MATCH path=(a:patient)-[*1..2]-(n)
WHERE toInteger(a.id) IN $patient_ids
RETURN 'patient' AS anchor_type, toInteger(a.id) AS anchor_patient,
       labels(n) AS neighbor_labels, n.id AS neighbor_id, n.name AS neighbor_name,
       n.modelType AS neighbor_model_type, n.modelFamily AS neighbor_model,
       n.classRatio AS neighbor_ratio,
       [r IN relationships(path) | {rel_type:type(r), from_labels:labels(startNode(r)), to_labels:labels(endNode(r)), props:properties(r)}] AS edges
LIMIT $limit
""".strip(), {"patient_ids": entities["patient_ids"], "limit": per_anchor}))

    if entities.get("biomarkers"):
        queries.append(("biomarker", """
MATCH path=(a:biomarker)-[*1..2]-(n)
WHERE a.name IN $biomarkers
RETURN 'biomarker' AS anchor_type, a.name AS anchor_biomarker,
       labels(n) AS neighbor_labels, n.id AS neighbor_id, n.name AS neighbor_name,
       n.modelType AS neighbor_model_type, n.modelFamily AS neighbor_model,
       n.classRatio AS neighbor_ratio,
       [r IN relationships(path) | {rel_type:type(r), from_labels:labels(startNode(r)), to_labels:labels(endNode(r)), props:properties(r)}] AS edges
LIMIT $limit
""".strip(), {"biomarkers": entities["biomarkers"], "limit": per_anchor}))

    if entities.get("model_types") or entities.get("ratios"):
        queries.append(("model_run", """
MATCH path=(a:model_run)-[*1..2]-(n)
WHERE (size($model_types)=0 OR a.modelType IN $model_types)
  AND (size($ratios)=0 OR a.classRatio IN $ratios)
RETURN 'model_run' AS anchor_type, a.modelType AS anchor_model_type,
       a.modelFamily AS anchor_model, a.classRatio AS anchor_ratio,
       labels(n) AS neighbor_labels, n.id AS neighbor_id, n.name AS neighbor_name,
       n.modelType AS neighbor_model_type, n.modelFamily AS neighbor_model,
       n.classRatio AS neighbor_ratio,
       [r IN relationships(path) | {rel_type:type(r), from_labels:labels(startNode(r)), to_labels:labels(endNode(r)), props:properties(r)}] AS edges
LIMIT $limit
""".strip(), {
            "model_types": entities.get("model_types") or [],
            "ratios": entities.get("ratios") or [],
            "limit": per_anchor,
        }))

    seen = set()
    for anchor_type, cypher, params in queries:
        res = execute_cypher(driver, cypher, database=database, params=params)
        for row in res.get("records", []):
            key = json.dumps(row, ensure_ascii=False, sort_keys=True, default=str)
            if key not in seen:
                seen.add(key)
                rows.append(row)
            if len(rows) >= total_limit:
                return redact_rescue_rows(rows)
    return redact_rescue_rows(rows)


# ---------------------------------------------------------------------------
# Prompt construction.
# ---------------------------------------------------------------------------

def build_baseline_cypher_prompt(subquestion: str, upstream_rows: list[dict]) -> str:
    upstream = json.dumps(upstream_rows or [], ensure_ascii=False, default=str)
    return f"""You translate ONE searching sub-question into ONE executable Cypher query for Neo4j.

{STATIC_SCHEMA}
{HARD_RULES}

All demonstrations below are from the training split.

{_render_demos()}

Previous searching-step rows, if any:
{upstream}
If the sub-question says "those biomarkers" or "that biomarker", bind names from previous rows when possible.

Now generate Cypher for this searching sub-question.
Question: {json.dumps(subquestion)}
Cypher:"""


def build_graphrag_cypher_prompt(subquestion: str, original_question: str,
                                upstream_rows: list[dict], query_type: str,
                                entities: dict, provenance: dict,
                                primary: dict, rescue_rows: list[dict] | None = None) -> str:
    rescue_rows = rescue_rows or []
    no_truncate_rule = ""
    if query_type == "local_max_abs":
        no_truncate_rule = f"""
IMPORTANT for this atomic query type (local_max_abs): the sub-question's own
top-K wording (e.g. "which {entities.get('top_k')} biomarkers...") may cover
only ONE side of what the ORIGINAL question actually needs (e.g. top-K
pushing toward AND top-K pushing away / positive AND negative). Do NOT add a
LIMIT that returns only the sub-question's literal K rows sorted by absolute
value -- that can silently drop an entire sign/direction. Return ALL
biomarkers on this path for this patient/model/ratio (same count as "live
rows from that path" above, no LIMIT needed) so the final-answer step can
select the correct top-K per direction itself.
"""
    return f"""You translate ONE searching sub-question into ONE executable Cypher query for Neo4j.

{STATIC_SCHEMA}
{HARD_RULES}

All demonstrations below are from the training split and are IDENTICAL to the baseline.

{_render_demos()}

ORIGINAL USER QUESTION (use only to recover details omitted by decomposition):
{original_question}

CURRENT SEARCHING SUB-QUESTION (authoritative when it explicitly states a value):
{subquestion}

PREVIOUS SEARCHING-STEP GRAPH ROWS:
{json.dumps(upstream_rows or [], ensure_ascii=False, default=str)}

GRAPHRAG EXACT GROUNDING (entity identities and schema facts only -- answer values are withheld):
- inferred atomic query type: {query_type}
- recovered canonical entities: {json.dumps(entities, ensure_ascii=False, default=str)}
- entity provenance: {json.dumps(provenance, ensure_ascii=False, default=str)}
- relevant graph path (arrow direction is authoritative): {primary.get('path')}
- entity bindings confirmed to exist on that path ({len(primary.get('identity_rows', []))} distinct): {json.dumps(primary.get('identity_rows', []), ensure_ascii=False, default=str)}
- answer-bearing properties to RETURN for this query type: {json.dumps(primary.get('answer_properties', []))}
- grounding error, if any: {primary.get('error')}

{('GRAPHRAG 1-2 HOP RESCUE CONTEXT:\n' + json.dumps(rescue_rows, ensure_ascii=False, default=str)) if rescue_rows else ''}
{no_truncate_rule}
How to use the context: pick the demonstration whose question pattern matches the CURRENT
sub-question and keep its Cypher structure; substitute the confirmed entity names, model types and
ratios above; use the path to get relationship direction right; RETURN the listed answer-bearing
properties. The grounding never adds WHERE conditions the sub-question did not ask for.
Write exactly ONE query with exactly ONE RETURN.
Cypher:"""


def _split_shap_by_sign(records: list[dict]) -> tuple[list[dict], list[dict]]:
    """Deterministically pre-split SHAP records by sign, strongest-first.

    A text rule telling the answer-synthesis LLM "positive shap = toward
    depression" was not reliably followed: the base model tended to just
    label whichever rows were sorted first by |shap_value| as "toward",
    regardless of their actual sign. Doing the split here in code (instead of
    asking the LLM to notice the `direction` field itself) removes that
    reasoning step entirely."""
    pos, neg = [], []
    for r in records:
        if not isinstance(r, dict):
            continue
        direction = r.get("direction")
        val = r.get("shap_value")
        is_pos = direction == "positive" or (direction is None and isinstance(val, (int, float)) and val > 0)
        is_neg = direction == "negative" or (direction is None and isinstance(val, (int, float)) and val < 0)
        if is_pos:
            pos.append(r)
        elif is_neg:
            neg.append(r)
    key = lambda r: abs(r.get("shap_value") or 0)
    pos.sort(key=key, reverse=True)
    neg.sort(key=key, reverse=True)
    return pos, neg


def build_answer_prompt(original_question: str, step_results: list[dict]) -> str:
    evidence = []
    for i, step in enumerate(step_results, 1):
        records = step.get("records", []) or []
        entry = {
            "step": i,
            "subquestion": step.get("question"),
            "cypher": step.get("cypher"),
            "exec_success": step.get("exec_success"),
            "records": records,
        }
        if any(isinstance(r, dict) and "direction" in r for r in records):
            pos, neg = _split_shap_by_sign(records)
            entry["records_PUSHING_TOWARD_depression_positive_shap_strongest_first"] = pos
            entry["records_PUSHING_AWAY_FROM_depression_negative_shap_strongest_first"] = neg
        evidence.append(entry)
    return f"""You are the final answer synthesizer for a graph-grounded QA system.

Original question:
{original_question}

Neo4j evidence produced by the searching sub-questions:
{json.dumps(evidence, ensure_ascii=False, default=str)}

Rules:
1. Answer the ORIGINAL question, not the sub-questions separately.
2. Use only the Neo4j evidence above for factual claims; do not invent missing values.
3. Perform any comparison, ranking, grouping, sign interpretation, or simple aggregation requested by the original question yourself.
4. SHAP sign convention: a POSITIVE shap_value pushes the prediction TOWARD
   depression (increases the predicted depression probability/label); a
   NEGATIVE shap_value pushes it AWAY FROM depression (decreases it). Where a
   step provides "records_PUSHING_TOWARD_depression_positive_shap_strongest_first"
   / "records_PUSHING_AWAY_FROM_depression_negative_shap_strongest_first",
   those are ALREADY correctly split and sorted by sign -- read the requested
   top-K directly off those two lists rather than re-deriving sign yourself
   from the raw "records" list.
5. If part of the requested information is not supported by the evidence, state exactly what is missing rather than guessing.
6. Be concise but include all requested models/ratios/biomarkers and numeric values when available.

Final answer:"""


# ---------------------------------------------------------------------------
# One sample end-to-end.
# ---------------------------------------------------------------------------

def process_sample(sample: dict, driver, database: str | None, vocab: dict | None,
                   model, tokenizer, sglang_url: str | None,
                   cypher_max_tokens: int, answer_max_tokens: int,
                   synthesize_answer: bool = True) -> dict:
    original = str(sample.get("input", ""))
    decomp = get_decomposition_text(sample)
    subqs = extract_subquestions(decomp)
    upstream_rows = []
    step_results = []

    for sq in subqs:
        q = sq["question"]
        graph_meta = {}
        rescue_rows, rescue_reasons = [], []
        query_type, entities, provenance, primary = "unknown", {}, {}, {"path": None, "rows": []}

        if USE_GRAPHRAG and driver is not None and vocab is not None:
            query_type = infer_query_type(q)
            entities, provenance = merge_entities(q, original, upstream_rows, vocab)
            primary = get_primary_grounding(driver, database, query_type, entities)

            if USE_NEIGHBORHOOD_RESCUE:
                missing = missing_required_slots(query_type, entities)
                if query_type == "unknown":
                    rescue_reasons.append("unknown_query_type")
                if missing:
                    rescue_reasons.append("missing_required_slots:" + ",".join(missing))
                if not primary.get("rows"):
                    rescue_reasons.append("empty_exact_grounding")
                # (v1's "entity_recovered_from_original_input" trigger is deliberately
                # NOT used: it fired on ~90% of steps and only added noise.)
                if rescue_reasons:
                    rescue_rows = retrieve_neighborhood_rescue(driver, database, entities)

            prompt = build_graphrag_cypher_prompt(
                q, original, upstream_rows, query_type, entities, provenance,
                primary, rescue_rows=rescue_rows,
            )
            graph_meta = {
                "query_type": query_type,
                "entities": entities,
                "entity_provenance": provenance,
                "primary_path": primary.get("path"),
                "primary_grounding_rows": primary.get("rows", []),
                "primary_grounding_identity_rows": primary.get("identity_rows", []),
                "answer_properties": primary.get("answer_properties", []),
                "primary_grounding_error": primary.get("error"),
                "rescue_triggered": bool(rescue_rows) if USE_NEIGHBORHOOD_RESCUE else False,
                "rescue_reasons": rescue_reasons if USE_NEIGHBORHOOD_RESCUE else [],
                "rescue_rows": rescue_rows if USE_NEIGHBORHOOD_RESCUE else [],
            }
        else:
            prompt = build_baseline_cypher_prompt(q, upstream_rows)

        raw = generate_text(
            prompt, model, tokenizer, sglang_url, cypher_max_tokens,
            stop=["\nQuestion:", "\nExplanation:", "\nAnswer:", "\nFinal answer:"],
        )
        cypher = clean_cypher_output(raw)
        cypher, repaired = repair_concatenated_queries(cypher)
        exec_res = (
            execute_cypher(driver, cypher, database=database)
            if driver is not None else
            {"success": None, "has_results": None, "error": None, "rows": 0, "records": []}
        )

        # ---- EXECUTION-FEEDBACK RESCUE (new in v2_rescue) ----
        # The pre-generation rescue above only fires when grounding is missing, which
        # on test_200 is rare (4/166 steps of imperfect samples). The failure mode that
        # actually remains is "query runs but is wrong": a syntax error or 0 rows. When
        # that happens (and the graph is available), retrieve the directed 1-2 hop
        # neighbourhood around the recovered entities and regenerate ONCE with the
        # failed query + Neo4j's verdict as feedback. Still temperature 0, one retry.
        post_rescue = {"post_rescue_triggered": False}
        if (USE_NEIGHBORHOOD_RESCUE and driver is not None and vocab is not None
                and (not exec_res.get("success") or not exec_res.get("records"))):
            if not rescue_rows:
                rescue_rows = retrieve_neighborhood_rescue(driver, database, entities)
            verdict = (f"Neo4j error: {exec_res.get('error')}" if not exec_res.get("success")
                       else "the query executed but returned 0 rows")
            base_prompt = build_graphrag_cypher_prompt(
                q, original, upstream_rows, query_type, entities, provenance,
                primary, rescue_rows=rescue_rows,
            )
            head, _tail = base_prompt.rsplit("\nCypher:", 1)   # only the FINAL "Cypher:" cue, not the demos'
            retry_prompt = head + f"""
PREVIOUS ATTEMPT (do not repeat it verbatim -- fix it):
{cypher}
RESULT OF PREVIOUS ATTEMPT: {verdict}
Use the rescue context above (labels, relationship types and their direction) to correct the
pattern, the property names, or the entity filters. Keep every rule above.
Cypher:"""
            raw2 = generate_text(
                retry_prompt, model, tokenizer, sglang_url, cypher_max_tokens,
                stop=["\nQuestion:", "\nExplanation:", "\nAnswer:", "\nFinal answer:"],
            )
            cypher2, repaired2 = repair_concatenated_queries(clean_cypher_output(raw2))
            exec2 = execute_cypher(driver, cypher2, database=database)
            post_rescue = {
                "post_rescue_triggered": True,
                "post_rescue_reason": "exec_error" if not exec_res.get("success") else "empty_result",
                "cypher_before_rescue": cypher,
                "exec_success_before_rescue": exec_res.get("success"),
                "rows_before_rescue": exec_res.get("rows", 0),
                "post_rescue_improved": bool(exec2.get("records")),
            }
            # keep the retry only if it actually produced rows (never trade a syntax
            # error for a different wrong-but-non-empty query blindly? -- we do accept
            # non-empty: an empty/failed first attempt scores 0 anyway)
            if exec2.get("records"):
                cypher, raw, repaired, exec_res = cypher2, raw2, repaired2, exec2
            graph_meta["rescue_triggered"] = True
            graph_meta["rescue_reasons"] = rescue_reasons + [post_rescue["post_rescue_reason"]]
            graph_meta["rescue_rows"] = rescue_rows

        step = {
            "question": q,
            "type": "searching",
            "cypher": cypher,
            "exec_success": exec_res.get("success"),
            "has_results": exec_res.get("has_results"),
            "error": exec_res.get("error"),
            "rows": exec_res.get("rows", 0),
            "records": exec_res.get("records", []),
            "repaired_concatenated_query": repaired,
            "raw_generation": raw,
        }
        step.update(graph_meta)
        step.update(post_rescue)
        step_results.append(step)

        # The next searching atom gets the last successful non-empty graph result.
        if exec_res.get("records"):
            upstream_rows = exec_res["records"]

    final_answer = ""
    if synthesize_answer:
        answer_prompt = build_answer_prompt(original, step_results)
        final_answer = generate_text(
            answer_prompt, model, tokenizer, sglang_url, answer_max_tokens,
            stop=["\nQuestion:", "\nOriginal question:"],
        ).strip()
        final_answer = re.sub(r"^\s*(?:Final answer|Answer)\s*:\s*", "", final_answer, flags=re.I)

    return {
        "input": original,
        "decomposition": decomp,
        "num_searching_subquestions": len(subqs),
        "steps": step_results,
        "final_answer": final_answer,
    }


def compute_metrics(details: list[dict]) -> dict:
    steps = [s for d in details for s in d.get("steps", [])]
    total = len(steps)
    exec_ok = sum(1 for s in steps if s.get("exec_success") is True)
    valid = sum(1 for s in steps if s.get("has_results") is True)
    all_valid_samples = sum(
        1 for d in details
        if d.get("steps") and all(s.get("has_results") is True for s in d["steps"])
    )
    m = {
        "experiment": EXPERIMENT_NAME,
        "samples": len(details),
        "searching_subquestions": total,
        "exec_success": exec_ok,
        "exec_rate": round(exec_ok / total, 4) if total else 0.0,
        "nonempty_results": valid,
        "valid_rate": round(valid / total, 4) if total else 0.0,
        "samples_all_searching_steps_nonempty": all_valid_samples,
        "sample_all_steps_rate": round(all_valid_samples / len(details), 4) if details else 0.0,
    }
    if USE_GRAPHRAG:
        recovered = sum(
            1 for s in steps
            if any(v == "original_input" for v in (s.get("entity_provenance") or {}).values())
        )
        m["steps_using_original_input_entity_recovery"] = recovered
    if USE_NEIGHBORHOOD_RESCUE:
        rescue = sum(1 for s in steps if s.get("rescue_triggered"))
        m["rescue_triggered_steps"] = rescue
        m["rescue_trigger_rate"] = round(rescue / total, 4) if total else 0.0
        m["post_rescue_triggered_steps"] = sum(1 for s in steps if s.get("post_rescue_triggered"))
        m["post_rescue_improved_steps"] = sum(1 for s in steps if s.get("post_rescue_improved"))
    return m


# ---------------------------------------------------------------------------
# CLI.
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=EXPERIMENT_NAME)
    parser.add_argument("--files", nargs="+", required=True,
                        help="JSON file(s) containing input + predicted decomposition")
    parser.add_argument("--model", default=MODEL_NAME)
    parser.add_argument("--cypher_max_tokens", type=int, default=256)
    parser.add_argument("--answer_max_tokens", type=int, default=384)
    parser.add_argument("--neo4j_uri", default=NEO4J_URI)
    parser.add_argument("--neo4j_user", default=NEO4J_USER)
    parser.add_argument("--neo4j_password", default=NEO4J_PASSWORD)
    parser.add_argument("--database", default=os.getenv("NEO4J_DATABASE") or None)
    parser.add_argument("--output_dir", default=f"results_{EXPERIMENT_NAME}")
    parser.add_argument("--sglang", action="store_true", help="auto-launch SGLang")
    parser.add_argument("--sglang_url", default=None, help="use existing SGLang server")
    parser.add_argument("--port", type=int, default=SGLANG_PORT)
    parser.add_argument("--no_execute", action="store_true")
    parser.add_argument("--no_answer", action="store_true", help="skip final-answer synthesis")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Model backend.
    proc = None
    model = tokenizer = None
    server_url = args.sglang_url
    if args.sglang or args.sglang_url:
        server_url = args.sglang_url or f"http://localhost:{args.port}"
        healthy = False
        try:
            healthy = requests.get(f"{server_url}/health", timeout=3).status_code == 200
        except Exception:
            healthy = False
        if not healthy:
            if args.sglang_url:
                raise RuntimeError(f"Cannot reach SGLang at {server_url}")
            proc = launch_sglang_server(args.model, args.port)
            if proc is None:
                raise RuntimeError("Failed to launch SGLang")
    else:
        model, tokenizer = load_model(args.model)

    # Neo4j.
    driver = None
    vocab = None
    if not args.no_execute:
        driver = GraphDatabase.driver(args.neo4j_uri, auth=(args.neo4j_user, args.neo4j_password))
        driver.verify_connectivity()
        print("Neo4j connected:", args.neo4j_uri)
        if USE_GRAPHRAG:
            vocab = build_vocab(driver, database=args.database)
            print("GraphRAG vocabulary built from live graph")

    try:
        for fpath in args.files:
            with open(fpath, "r", encoding="utf-8") as f:
                data = json.load(f)
            print(f"\n[{EXPERIMENT_NAME}] {fpath}: {len(data)} samples")

            details = []
            start = time.time()
            for i, sample in enumerate(data, 1):
                details.append(process_sample(
                    sample,
                    driver=driver,
                    database=args.database,
                    vocab=vocab,
                    model=model,
                    tokenizer=tokenizer,
                    sglang_url=server_url,
                    cypher_max_tokens=args.cypher_max_tokens,
                    answer_max_tokens=args.answer_max_tokens,
                    synthesize_answer=not args.no_answer,
                ))
                if i % 10 == 0 or i == len(data):
                    elapsed = time.time() - start
                    print(f"  processed {i}/{len(data)} samples ({elapsed:.1f}s)")

            metrics = compute_metrics(details)
            base = os.path.splitext(os.path.basename(fpath))[0]
            out_path = os.path.join(args.output_dir, f"{base}_{EXPERIMENT_NAME}.json")
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump({
                    "experiment": EXPERIMENT_NAME,
                    "use_graphrag": USE_GRAPHRAG,
                    "use_neighborhood_rescue": USE_NEIGHBORHOOD_RESCUE,
                    "metrics": metrics,
                    "details": details,
                }, f, indent=2, ensure_ascii=False, default=str)
            print(json.dumps(metrics, ensure_ascii=False, indent=2))
            print("Saved:", out_path)
    finally:
        if driver is not None:
            driver.close()
        if proc is not None:
            shutdown_sglang_server(proc)


if __name__ == "__main__":
    main()
