"""
Entity + intent retrieval for graph-grounded Cypher generation.

This module is designed to run on each *searching sub-question* produced by the
question decomposition stage.  It does three things:

1. Extract graph entities (patient ids, biomarker names, model families,
   training ratios, condition names) from live vocabulary.
2. Infer one atomic query type from the sub-question semantics. The supported
   families are patient value, prediction/probability, local SHAP,
   local top-positive, local max-|SHAP|, global rank, global tendency,
   global rank+tendency, accuracy, and biomarker description.
3. Retrieve a small, parameterised fact/path sample from the *same graph path*
   that the final Cypher should use.  This gives the generator concrete label,
   relationship direction, property, and value grounding without hardcoding
   patient/feature-specific Cypher strings into the prompt.

Important design choice: algorithm x ratio combinations are NOT truncated.
All requested combinations are expressed using IN filters in one Cypher query;
only returned grounding rows are bounded.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

DEFAULT_ROW_LIMIT = 60


def _norm(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", (text or "").lower())


def _lower_words(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", (text or "").lower())).strip()


def _safe_list(value):
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        return list(value)
    return [value]


def build_vocab(driver, database=None) -> dict:
    """Build entity vocabulary from the live graph.

    The graph itself is the source of truth. Model aliases come from both
    ModelRun.modelType and ModelRun.modelFamily, so natural-language surface
    forms resolve to the canonical modelType stored in the graph.
    """

    def run(query: str, params: dict | None = None):
        with driver.session(database=database) as session:
            return [r.data() for r in session.run(query, params or {})]

    biomarkers = run(
        "MATCH (b:biomarker) "
        "RETURN DISTINCT b.name AS name, b.aliases AS aliases, b.key AS key"
    )
    model_runs = run(
        "MATCH (mr:model_run) "
        "RETURN DISTINCT mr.modelType AS modelType, mr.modelFamily AS modelFamily, "
        "mr.classRatio AS classRatio"
    )
    conditions = run("MATCH (c:condition) RETURN DISTINCT c.name AS name")

    biomarker_alias_to_name: dict[str, str] = {}
    biomarker_alias_surface: dict[str, str] = {}
    for row in biomarkers:
        name = row.get("name")
        if not name:
            continue
        aliases = [name, row.get("key")] + _safe_list(row.get("aliases"))
        for alias in aliases:
            if not alias:
                continue
            key = _norm(str(alias))
            if key:
                biomarker_alias_to_name[key] = name
                biomarker_alias_surface[key] = str(alias)

    model_alias_to_type: dict[str, str] = {}
    model_type_to_family: dict[str, str] = {}
    ratios = set()
    for row in model_runs:
        model_type = row.get("modelType")
        family = row.get("modelFamily")
        ratio = row.get("classRatio")
        if ratio:
            ratios.add(str(ratio))
        if not model_type:
            continue
        model_type = str(model_type)
        if family:
            model_type_to_family[model_type] = str(family)
        for alias in [model_type, family]:
            if alias:
                model_alias_to_type[_norm(str(alias))] = model_type

    # These are generic orthographic variants of the canonical modelType
    # tokens, not graph facts.  Add them only if the corresponding modelType
    # actually exists in the graph.
    common_surfaces = {
        "naivebayes": ["naive bayes", "naivebayes"],
        "decisiontree": ["decision tree", "decisiontree"],
        "logisticregression": ["logistic regression", "logisticregression"],
        "randomforest": ["random forest", "randomforest"],
    }
    existing_types = set(model_type_to_family) | set(model_alias_to_type.values())
    for model_type, surfaces in common_surfaces.items():
        if model_type in existing_types:
            for surface in surfaces:
                model_alias_to_type[_norm(surface)] = model_type

    condition_alias_to_name = {}
    for row in conditions:
        name = row.get("name")
        if name:
            condition_alias_to_name[_norm(str(name))] = str(name)

    return {
        "biomarker_alias_to_name": biomarker_alias_to_name,
        "biomarker_alias_surface": biomarker_alias_surface,
        "model_alias_to_type": model_alias_to_type,
        "model_type_to_family": model_type_to_family,
        "ratios": sorted(ratios),
        "condition_alias_to_name": condition_alias_to_name,
    }


def infer_query_type(question: str) -> str:
    """Infer the atomic query family for one decomposition sub-question."""
    q = _lower_words(question)

    if any(x in q for x in ["clinical meaning", "clinical concept", "definition of", "what does"]):
        # Keep this narrow enough that "what does random forest predict" does
        # not become a biomarker-description query.
        if any(x in q for x in ["clinical meaning", "clinical concept", "definition of", "represent clinically"]):
            return "biomarker_description"

    if "accuracy" in q:
        return "model_accuracy"

    if "largest positive local shap" in q or "largest positive" in q and "shap" in q:
        return "local_top_positive"

    if ("largest absolute local shap" in q or "largest absolute" in q and "shap" in q or
            "most strongly" in q and "push" in q):
        return "local_max_abs"

    # "those top k biomarkers" / "that biomarker" is a dependent second
    # searching step: ranking was already done upstream, so this step only
    # needs the global tendency path.
    if re.search(r"\b(those(?: top)? \d+ biomarkers|that biomarker)\b", q):
        if "tendency" in q or "increases" in q:
            return "global_tendency"

    if "global shap" in q and any(x in q for x in ["as each biomarker", "tendency", "increases"]):
        if any(x in q for x in ["top ", "ranked", "importance"]):
            return "global_rank_plus_tendency"

    if any(x in q for x in ["global shap importance rank", "ranked by global shap", "global shap importance"]):
        return "global_rank"

    if any(x in q for x in ["global shap tendency", "as the biomarker value increases", "as each biomarker value increases"]):
        return "global_tendency"

    if "local shap" in q or ("contribution" in q and "positive or negative" in q):
        return "local_shap"

    if "correct" in q or "prediction label" in q or "depression probability" in q or "what prediction" in q or "predict for patient" in q:
        return "prediction"

    if "patient" in q and any(x in q for x in [" value", "measurement", "recorded for patient"]):
        return "patient_biomarker_value"

    return "unknown"


def _extract_top_k(question: str) -> int | None:
    patterns = [
        r"\btop\s+(\d+)\b",
        r"\bwhich\s+(\d+)\s+(?:biomarkers?|features?)\b",
        r"\b(\d+)\s+(?:biomarkers?|features?)\s+have the largest",
    ]
    for pattern in patterns:
        m = re.search(pattern, question, re.I)
        if m:
            return int(m.group(1))
    if re.search(r"\bwhich biomarker has the largest\b", question, re.I):
        return 1
    return None


def extract_entities(question: str, vocab: dict, upstream_rows: list[dict] | None = None) -> dict:
    """Extract entities from one atomic sub-question using live vocabulary."""
    q_norm = _norm(question)

    patient_ids = sorted({
        int(x) for x in re.findall(r"\bpatient\s+(\d+)\b", question, re.I)
    })

    # Longest aliases first prevents a shorter alias from winning when aliases
    # overlap.  Deduplicate by canonical Biomarker.name.
    biomarkers = []
    seen_biomarkers = set()
    aliases = sorted(
        vocab.get("biomarker_alias_to_name", {}).items(),
        key=lambda kv: len(kv[0]),
        reverse=True,
    )
    for alias_norm, canonical in aliases:
        if alias_norm and alias_norm in q_norm and canonical not in seen_biomarkers:
            biomarkers.append(canonical)
            seen_biomarkers.add(canonical)

    # Resolve decomposition coreference from previous searching-step results.
    # Typical decomposition coreference forms are "those 3 biomarkers",
    # "those top 5 biomarkers", and "that biomarker". Without this binding, the current
    # sub-question is under-specified and no generator can know the intended
    # Biomarker.name values.
    has_biomarker_coref = bool(re.search(
        r"\b(those(?: top)? \d+ biomarkers|those biomarkers|that biomarker)\b",
        question,
        re.I,
    ))
    if has_biomarker_coref and upstream_rows:
        for row in upstream_rows:
            candidate = row.get("biomarker") or row.get("b.name") or row.get("name")
            if candidate and candidate not in seen_biomarkers:
                biomarkers.append(str(candidate))
                seen_biomarkers.add(str(candidate))

    model_types = []
    seen_models = set()
    model_aliases = sorted(
        vocab.get("model_alias_to_type", {}).items(),
        key=lambda kv: len(kv[0]),
        reverse=True,
    )
    for alias_norm, model_type in model_aliases:
        if alias_norm and alias_norm in q_norm and model_type not in seen_models:
            model_types.append(model_type)
            seen_models.add(model_type)

    ratios = []
    seen_ratios = set()
    for m in re.finditer(r"\b1\s*(?::|-|to)\s*(\d+)\b", question, re.I):
        ratio = f"1:{m.group(1)}"
        if ratio not in seen_ratios:
            ratios.append(ratio)
            seen_ratios.add(ratio)

    conditions = []
    for alias_norm, canonical in vocab.get("condition_alias_to_name", {}).items():
        if alias_norm and alias_norm in q_norm:
            conditions.append(canonical)
    conditions = list(dict.fromkeys(conditions))

    q_low = question.lower()
    return {
        "query_type": infer_query_type(question),
        "patient_ids": patient_ids,
        "biomarkers": biomarkers,
        "model_types": model_types,
        "ratios": ratios,
        "conditions": conditions,
        "top_k": _extract_top_k(question),
        "all_biomarkers": bool(re.search(r"\b(each|all) available (biomarkers?|features?)\b", q_low)),
        "direction_only": "positive or negative" in q_low and "shap value" not in q_low,
        "has_biomarker_coreference": has_biomarker_coref,
    }


@dataclass
class RetrievalSpec:
    query_type: str
    path: str
    cypher: str
    params: dict[str, Any]


def _base_params(entities: dict, row_limit: int) -> dict:
    return {
        "patient_ids": entities.get("patient_ids") or [],
        "biomarkers": entities.get("biomarkers") or [],
        "model_types": entities.get("model_types") or [],
        "ratios": entities.get("ratios") or [],
        "conditions": entities.get("conditions") or [],
        "top_k": entities.get("top_k"),
        "row_limit": int(row_limit),
    }


def build_retrieval_spec(entities: dict, row_limit: int = DEFAULT_ROW_LIMIT) -> RetrievalSpec | None:
    """Build one parameterised retrieval query for the inferred atomic type."""
    t = entities.get("query_type") or "unknown"
    p = _base_params(entities, row_limit)

    if t == "patient_biomarker_value":
        return RetrievalSpec(
            t,
            "(:patient)-[m:HAS_BIOMARKER_VALUE]->(:biomarker)",
            """
MATCH (p:patient)-[m:HAS_BIOMARKER_VALUE]->(b:biomarker)
WHERE (size($patient_ids) = 0 OR toInteger(p.id) IN $patient_ids)
  AND (size($biomarkers) = 0 OR b.name IN $biomarkers)
RETURN toInteger(p.id) AS patient_id, b.name AS biomarker, m.value AS value
ORDER BY patient_id, biomarker
LIMIT $row_limit
""".strip(), p)

    if t == "biomarker_description":
        return RetrievalSpec(
            t,
            "(:biomarker)",
            """
MATCH (b:biomarker)
WHERE size($biomarkers) = 0 OR b.name IN $biomarkers
RETURN b.name AS biomarker, b.description AS description
ORDER BY biomarker
LIMIT $row_limit
""".strip(), p)

    if t == "prediction":
        return RetrievalSpec(
            t,
            "(:patient)-[:HAS_PREDICTION]->(:prediction)-[:GENERATED_BY]->(:model_run); "
            "(:prediction)-[pr:PREDICTS]->(:condition)",
            """
MATCH (p:patient)-[:HAS_PREDICTION]->(pred:prediction)-[:GENERATED_BY]->(mr:model_run)
OPTIONAL MATCH (pred)-[pr:PREDICTS]->(c:condition)
WHERE (size($patient_ids) = 0 OR toInteger(p.id) IN $patient_ids)
  AND (size($model_types) = 0 OR mr.modelType IN $model_types)
  AND (size($ratios) = 0 OR mr.classRatio IN $ratios)
  AND (size($conditions) = 0 OR c.name IN $conditions OR c IS NULL)
RETURN toInteger(p.id) AS patient_id, mr.modelType AS model_type,
       mr.modelFamily AS model, mr.classRatio AS ratio, pred.correct AS correct,
       c.name AS condition, pr.label AS predicted_label, pr.probability AS probability
ORDER BY patient_id, ratio, model_type
LIMIT $row_limit
""".strip(), p)

    if t in {"local_shap", "local_top_positive", "local_max_abs"}:
        extra = ""
        order = "ORDER BY patient_id, ratio, model_type, biomarker"
        if t == "local_top_positive":
            extra = "  AND fi.direction = 'positive'\n"
            order = "ORDER BY fi.shapValue DESC, biomarker"
            if p.get("top_k"):
                p["row_limit"] = min(p["row_limit"], int(p["top_k"]))
        elif t == "local_max_abs":
            order = "ORDER BY fi.absShapValue DESC, biomarker"
            p["row_limit"] = min(p["row_limit"], 5)

        return RetrievalSpec(
            t,
            "(:patient)-[:HAS_PREDICTION]->(:prediction)-[:EXPLAINED_BY]->"
            "(:local_explanation)-[:GENERATED_BY]->(:model_run); "
            "(:local_explanation)-[fi:FEATURE_IMPORTANCE {scope:'local'}]->(:biomarker)",
            f"""
MATCH (p:patient)-[:HAS_PREDICTION]->(pred:prediction)-[:EXPLAINED_BY]->(le:local_explanation)-[:GENERATED_BY]->(mr:model_run)
MATCH (le)-[fi:FEATURE_IMPORTANCE]->(b:biomarker)
WHERE (size($patient_ids) = 0 OR toInteger(p.id) IN $patient_ids)
  AND (size($model_types) = 0 OR mr.modelType IN $model_types)
  AND (size($ratios) = 0 OR mr.classRatio IN $ratios)
  AND (size($biomarkers) = 0 OR b.name IN $biomarkers)
  AND fi.scope = 'local'
{extra}RETURN toInteger(p.id) AS patient_id, mr.modelType AS model_type,
       mr.modelFamily AS model, mr.classRatio AS ratio, b.name AS biomarker,
       fi.shapValue AS shap_value, fi.absShapValue AS abs_shap_value,
       fi.direction AS direction
{order}
LIMIT $row_limit
""".strip(), p)

    if t == "model_accuracy":
        return RetrievalSpec(
            t,
            "(:prediction)-[:GENERATED_BY]->(:model_run)",
            """
MATCH (pred:prediction)-[:GENERATED_BY]->(mr:model_run)
WHERE (size($model_types) = 0 OR mr.modelType IN $model_types)
  AND (size($ratios) = 0 OR mr.classRatio IN $ratios)
RETURN mr.modelType AS model_type, mr.modelFamily AS model, mr.classRatio AS ratio,
       avg(CASE WHEN pred.correct THEN 1.0 ELSE 0.0 END) AS accuracy,
       count(pred) AS prediction_count
ORDER BY ratio, model_type
LIMIT $row_limit
""".strip(), p)

    if t == "global_rank":
        return RetrievalSpec(
            t,
            "(:model_run)-[s:SHAP_SUMMARY]->(:biomarker)",
            """
MATCH (mr:model_run)-[s:SHAP_SUMMARY]->(b:biomarker)
WHERE (size($model_types) = 0 OR mr.modelType IN $model_types)
  AND (size($ratios) = 0 OR mr.classRatio IN $ratios)
  AND (size($biomarkers) = 0 OR b.name IN $biomarkers)
  AND ($top_k IS NULL OR s.rankByMeanAbsShap <= $top_k)
RETURN mr.modelType AS model_type, mr.modelFamily AS model, mr.classRatio AS ratio,
       b.name AS biomarker, s.rankByMeanAbsShap AS rank,
       s.meanAbsShap AS global_shap_importance
ORDER BY ratio, model_type, rank, biomarker
LIMIT $row_limit
""".strip(), p)

    if t == "global_tendency":
        return RetrievalSpec(
            t,
            "(:model_run)-[fi:FEATURE_IMPORTANCE {scope:'global'}]->(:biomarker)",
            """
MATCH (mr:model_run)-[fi:FEATURE_IMPORTANCE]->(b:biomarker)
WHERE (size($model_types) = 0 OR mr.modelType IN $model_types)
  AND (size($ratios) = 0 OR mr.classRatio IN $ratios)
  AND (size($biomarkers) = 0 OR b.name IN $biomarkers)
  AND fi.scope = 'global'
RETURN mr.modelType AS model_type, mr.modelFamily AS model, mr.classRatio AS ratio,
       b.name AS biomarker, fi.influence AS increase_related_tendency
ORDER BY ratio, model_type, biomarker
LIMIT $row_limit
""".strip(), p)

    if t == "global_rank_plus_tendency":
        return RetrievalSpec(
            t,
            "(:model_run)-[s:SHAP_SUMMARY]->(:biomarker)<-[fi:FEATURE_IMPORTANCE {scope:'global'}]-(:model_run)",
            """
MATCH (mr:model_run)-[s:SHAP_SUMMARY]->(b:biomarker)
MATCH (mr)-[fi:FEATURE_IMPORTANCE]->(b)
WHERE (size($model_types) = 0 OR mr.modelType IN $model_types)
  AND (size($ratios) = 0 OR mr.classRatio IN $ratios)
  AND (size($biomarkers) = 0 OR b.name IN $biomarkers)
  AND ($top_k IS NULL OR s.rankByMeanAbsShap <= $top_k)
  AND fi.scope = 'global'
RETURN mr.modelType AS model_type, mr.modelFamily AS model, mr.classRatio AS ratio,
       b.name AS biomarker, s.rankByMeanAbsShap AS rank,
       s.meanAbsShap AS global_shap_importance,
       fi.influence AS increase_related_tendency
ORDER BY ratio, model_type, rank, biomarker
LIMIT $row_limit
""".strip(), p)

    # Fallback: the sub-question didn't match any of the fixed atomic types
    # above -- either a genuinely novel phrasing, or an incomplete/
    # underspecified decomposition step (the decomposition model dropped a
    # detail, or phrased the ask in a way infer_query_type doesn't
    # recognize). Previously this returned None here, so the caller got
    # zero grounding and method2 degraded to a schema-only guess -- no
    # better off than the fixed few-shot baseline it's meant to improve on.
    # Instead, retrieve a broad snapshot around whatever entities WERE
    # extracted -- something real to reason from beats nothing, even if
    # it's not the exact atomic answer. Ordered patient > biomarker >
    # model_run because a patient id is the most specific anchor when present.
    if entities.get("patient_ids"):
        return RetrievalSpec(
            "fallback_patient_snapshot",
            "(:patient); (:patient)-[:HAS_PREDICTION]->(:prediction)-[:GENERATED_BY]->(:model_run); "
            "(:patient)-[:HAS_BIOMARKER_VALUE]->(:biomarker)",
            """
MATCH (p:patient)
WHERE toInteger(p.id) IN $patient_ids
OPTIONAL MATCH (p)-[:HAS_PREDICTION]->(pred:prediction)-[:GENERATED_BY]->(mr:model_run)
WHERE (size($model_types) = 0 OR mr IS NULL OR mr.modelType IN $model_types)
  AND (size($ratios) = 0 OR mr IS NULL OR mr.classRatio IN $ratios)
OPTIONAL MATCH (p)-[m:HAS_BIOMARKER_VALUE]->(b:biomarker)
WHERE size($biomarkers) = 0 OR b IS NULL OR b.name IN $biomarkers
RETURN toInteger(p.id) AS patient_id, mr.modelFamily AS model, mr.classRatio AS ratio,
       pred.probability AS probability, pred.predictedLabel AS predicted_label,
       pred.correct AS correct, b.name AS biomarker, m.value AS biomarker_value
LIMIT $row_limit
""".strip(), p)

    if entities.get("biomarkers"):
        return RetrievalSpec(
            "fallback_biomarker_snapshot",
            "(:biomarker)",
            """
MATCH (b:biomarker)
WHERE b.name IN $biomarkers
RETURN b.name AS biomarker, b.description AS description, b.key AS key
LIMIT $row_limit
""".strip(), p)

    if entities.get("model_types") or entities.get("ratios"):
        return RetrievalSpec(
            "fallback_model_run_snapshot",
            "(:model_run)",
            """
MATCH (mr:model_run)
WHERE (size($model_types) = 0 OR mr.modelType IN $model_types)
  AND (size($ratios) = 0 OR mr.classRatio IN $ratios)
RETURN mr.modelFamily AS model, mr.classRatio AS ratio,
       mr.precision_0 AS precision_0, mr.precision_1 AS precision_1,
       mr.f1_0 AS f1_0, mr.f1_1 AS f1_1
LIMIT $row_limit
""".strip(), p)

    return None


def _format_row(row: dict) -> str:
    return json.dumps(row, ensure_ascii=False, default=str, sort_keys=True)


def retrieve_grounding_context(driver, question: str, vocab: dict | None = None,
                               database=None, row_limit: int = DEFAULT_ROW_LIMIT,
                               upstream_rows: list[dict] | None = None) -> dict:
    """Retrieve structured grounding for one searching sub-question."""
    vocab = vocab or build_vocab(driver, database=database)
    entities = extract_entities(question, vocab, upstream_rows=upstream_rows)
    spec = build_retrieval_spec(entities, row_limit=row_limit)

    if spec is None:
        return {
            "question": question,
            "entities": entities,
            "query_type": entities.get("query_type", "unknown"),
            "path": None,
            "cypher": None,
            "params": {},
            "rows": [],
        }

    with driver.session(database=database) as session:
        rows = [r.data() for r in session.run(spec.cypher, spec.params)]

    return {
        "question": question,
        "entities": entities,
        "query_type": spec.query_type,
        "path": spec.path,
        "cypher": spec.cypher,
        "params": spec.params,
        "rows": rows,
    }


def retrieve_grounding_facts(driver, entities_or_question, max_variants=None,
                             database=None, row_limit: int = DEFAULT_ROW_LIMIT) -> list[str]:
    """Backward-compatible text wrapper.

    ``max_variants`` is accepted for compatibility but intentionally ignored:
    variant truncation was a correctness bug for multi-model/multi-ratio
    questions.  Pass either the original sub-question string or an entity dict.
    """
    del max_variants

    if isinstance(entities_or_question, str):
        context = retrieve_grounding_context(
            driver, entities_or_question, database=database, row_limit=row_limit
        )
    else:
        entities = entities_or_question
        spec = build_retrieval_spec(entities, row_limit=row_limit)
        if spec is None:
            return []
        with driver.session(database=database) as session:
            rows = [r.data() for r in session.run(spec.cypher, spec.params)]
        context = {
            "query_type": spec.query_type,
            "path": spec.path,
            "cypher": spec.cypher,
            "params": spec.params,
            "rows": rows,
        }

    facts = [
        f"ATOMIC_QUERY_TYPE: {context['query_type']}",
        f"RELEVANT_GRAPH_PATH: {context['path']}",
        # Labeled TEMPLATE (not TEMPLATE_CYPHER) and paired with an explicit
        # warning: showing this $-parameterized query without one let the
        # generator copy it verbatim into its final answer, which then
        # failed at execution time with Neo.ClientError.Statement.ParameterMissing
        # (Neo4j has no bound values for $patient_ids etc. outside this
        # internal lookup) -- ~8% of method2 failures in one eval were this
        # exact bug. The real grounding is REAL_GRAPH_ROWS below; this line
        # only illustrates the traversal shape.
        "TEMPLATE_CYPHER_SHOWING_THE_TRAVERSAL_SHAPE (has $-parameters -- "
        "these are NOT real Cypher syntax, do not copy this query or its "
        "$names into your answer; write literal values instead): "
        + " ".join((context.get("cypher") or "").split()),
        "TEMPLATE_PARAMS (what the $-names above resolved to, for reference only): "
        + json.dumps(context.get("params") or {}, ensure_ascii=False, default=str),
    ]
    rows = context.get("rows") or []
    if rows:
        facts.append("REAL_GRAPH_ROWS:")
        facts.extend("  " + _format_row(row) for row in rows)
    else:
        facts.append("REAL_GRAPH_ROWS: []")
    return facts


def build_generation_context(driver, subquestion: str, vocab: dict | None = None,
                             database=None, row_limit: int = DEFAULT_ROW_LIMIT,
                             upstream_rows: list[dict] | None = None) -> str:
    """Convenience function: compact prompt context for a Cypher generator."""
    context = retrieve_grounding_context(
        driver, subquestion, vocab=vocab, database=database, row_limit=row_limit,
        upstream_rows=upstream_rows
    )
    lines = [
        f"Atomic query type: {context['query_type']}",
        f"Extracted entities: {json.dumps(context['entities'], ensure_ascii=False)}",
        f"Relevant graph path: {context['path']}",
        "A parameterized query that successfully follows this path:",
        " ".join((context.get("cypher") or "").split()),
        f"Parameters: {json.dumps(context.get('params') or {}, ensure_ascii=False, default=str)}",
    ]
    rows = context.get("rows") or []
    if rows:
        lines.append("Representative rows returned from the live graph:")
        lines.extend(_format_row(row) for row in rows)
    else:
        lines.append("Representative rows returned from the live graph: []")
    return "\n".join(lines)
