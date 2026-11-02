"""
Dynamic Neo4j schema/path introspection for decomposition-grounded Cypher generation.

Main changes from the previous version
--------------------------------------
1. Treat per-run labels such as ``naivebayes_1_3`` as legacy/dynamic labels.
   They are removed from the stable schema shown to the LLM.  If ``ModelRun``
   exposes ``modelType`` and ``classRatio``, the card explicitly recommends
   filtering on those properties instead of synthesising a label/modelName.
2. Build a structured schema index in addition to the text card.  The index can
   be used by retrieval code to select relevant graph paths for an atomic
   sub-question.
3. Keep relationship direction and relationship properties together.  This is
   important for ``FEATURE_IMPORTANCE`` and ``HAS_BIOMARKER_VALUE``, where the
   answer-bearing value lives on the relationship.
4. Enumerate only low-cardinality, query-useful STRING values to avoid prompt
   bloat.
5. Provide a bounded schema-level path finder.  It preserves edge direction in
   the returned path even when it traverses an edge backwards during search.
"""

from __future__ import annotations

import re
from collections import defaultdict, deque
from typing import Iterable

MODEL_VARIANT_RE = re.compile(
    r"^(naivebayes|decisiontree|logisticregression|randomforest)_\d+_\d+$",
    re.I,
)
ENUM_LIMIT = 30
MAX_ENUM_VALUE_LEN = 80

# Properties that are particularly useful as WHERE filters or semantic values.
# This is a preference, not a schema assumption: a property still has to be
# discovered by live introspection before it is shown.
_ENUM_HINTS = (
    "name", "type", "family", "ratio", "scope", "direction", "label",
    "condition", "method", "correct",
)


def _quote_ident(name: str) -> str:
    """Quote a Cypher identifier safely."""
    return "`" + str(name).replace("`", "``") + "`"


def _is_legacy_variant(label: str) -> bool:
    return bool(MODEL_VARIANT_RE.match(label or ""))


def _stable_labels(labels: Iterable[str]) -> tuple[str, ...]:
    """Drop only dynamic model-variant labels; preserve all stable labels."""
    return tuple(sorted({x for x in labels if x and not _is_legacy_variant(x)}))


def _should_enum(prop: str, ptype: str) -> bool:
    if "STRING" not in (ptype or "").upper():
        return False
    low = prop.lower()
    return any(hint in low for hint in _ENUM_HINTS)


def build_schema_index(driver, database=None, enum_limit: int = ENUM_LIMIT) -> dict:
    """Return a structured schema index built only from the connected graph.

    Returned keys:
      node_props: {label: {property: type}}
      node_values: {label: {property: [low-cardinality values]}}
      rel_props: {rel_type: {property: type}}
      patterns: [{start_labels, relationship, end_labels, constants}]
      legacy_variant_labels: [...]
    """

    def run(query: str, params: dict | None = None):
        with driver.session(database=database) as session:
            return [r.data() for r in session.run(query, params or {})]

    node_props: dict[str, dict[str, str]] = defaultdict(dict)
    legacy_variants: set[str] = set()

    for row in run("CALL db.schema.nodeTypeProperties()"):
        labels = row.get("nodeLabels") or []
        legacy_variants.update(x for x in labels if _is_legacy_variant(x))
        stable = _stable_labels(labels)
        if not stable:
            continue
        prop = row.get("propertyName")
        ptypes = row.get("propertyTypes") or ["?"]
        if prop:
            # A node can carry more than one stable label.  Recording the
            # property under each label makes the index useful for normal
            # single-label MATCH patterns and avoids forcing label combos.
            for label in stable:
                node_props[label][prop] = ptypes[0]
        else:
            for label in stable:
                node_props.setdefault(label, {})

    rel_props: dict[str, dict[str, str]] = defaultdict(dict)
    for row in run("CALL db.schema.relTypeProperties()"):
        rel_type = (row.get("relType") or "").strip(":`")
        if not rel_type:
            continue
        prop = row.get("propertyName")
        ptypes = row.get("propertyTypes") or ["?"]
        if prop:
            rel_props[rel_type][prop] = ptypes[0]
        else:
            rel_props.setdefault(rel_type, {})

    def distinct_node_values(label: str, prop: str):
        rows = run(
            f"MATCH (n:{_quote_ident(label)}) "
            f"WHERE n.{_quote_ident(prop)} IS NOT NULL "
            f"RETURN DISTINCT n.{_quote_ident(prop)} AS v LIMIT $limit",
            {"limit": enum_limit + 1},
        )
        vals = [r.get("v") for r in rows]
        if not vals or len(vals) > enum_limit:
            return None
        # Do not inject very long free-text values (e.g. descriptions) into the
        # schema card even if cardinality is low.
        if any(len(str(v)) > MAX_ENUM_VALUE_LEN for v in vals):
            return None
        return sorted(vals, key=lambda x: str(x))

    node_values: dict[str, dict[str, list]] = defaultdict(dict)
    for label, props in node_props.items():
        for prop, ptype in props.items():
            if not _should_enum(prop, ptype):
                continue
            vals = distinct_node_values(label, prop)
            if vals:
                node_values[label][prop] = vals

    raw_patterns = run(
        "MATCH (a)-[r]->(b) "
        "RETURN DISTINCT labels(a) AS al, type(r) AS rt, labels(b) AS bl"
    )

    pattern_keys: set[tuple[tuple[str, ...], str, tuple[str, ...]]] = set()
    for row in raw_patterns:
        al = _stable_labels(row.get("al") or [])
        bl = _stable_labels(row.get("bl") or [])
        rt = row.get("rt")
        if al and bl and rt:
            pattern_keys.add((al, rt, bl))

    def relationship_constants(start_labels: tuple[str, ...], rel_type: str,
                               end_labels: tuple[str, ...]) -> dict:
        # Only low-cardinality relationship string properties are candidates.
        candidates = [
            p for p, t in rel_props.get(rel_type, {}).items()
            if _should_enum(p, t)
        ]
        if not candidates:
            return {}

        start_anchor = "".join(f":{_quote_ident(x)}" for x in start_labels)
        end_anchor = "".join(f":{_quote_ident(x)}" for x in end_labels)
        constants = {}
        for prop in candidates:
            rows = run(
                f"MATCH (a{start_anchor})-[r:{_quote_ident(rel_type)}]->(b{end_anchor}) "
                f"WHERE r.{_quote_ident(prop)} IS NOT NULL "
                f"RETURN DISTINCT r.{_quote_ident(prop)} AS v LIMIT 2"
            )
            vals = [r.get("v") for r in rows]
            if len(vals) == 1:
                constants[prop] = vals[0]
        return constants

    patterns = []
    for al, rt, bl in sorted(pattern_keys):
        patterns.append({
            "start_labels": al,
            "relationship": rt,
            "end_labels": bl,
            "constants": relationship_constants(al, rt, bl),
        })

    return {
        "node_props": {k: dict(v) for k, v in node_props.items()},
        "node_values": {k: dict(v) for k, v in node_values.items()},
        "rel_props": {k: dict(v) for k, v in rel_props.items()},
        "patterns": patterns,
        "legacy_variant_labels": sorted(legacy_variants),
    }


def _fmt_label_combo(labels: Iterable[str]) -> str:
    return ":" + ":".join(labels)


def build_schema_card(driver, database=None, enum_limit: int = ENUM_LIMIT) -> str:
    """Render a compact LLM-oriented schema card from live introspection."""
    idx = build_schema_index(driver, database=database, enum_limit=enum_limit)
    node_props = idx["node_props"]
    node_values = idx["node_values"]
    rel_props = idx["rel_props"]

    lines = [
        "GRAPH SCHEMA (live Neo4j introspection; use exact label/relationship/property casing):",
        "",
        "Stable node labels and properties:",
    ]

    for label in sorted(node_props):
        parts = []
        for prop, ptype in sorted(node_props[label].items()):
            vals = node_values.get(label, {}).get(prop)
            if vals:
                parts.append(f"{prop} ({ptype}) [values: {', '.join(repr(v) for v in vals)}]")
            else:
                parts.append(f"{prop} ({ptype})")
        lines.append(f"  (:{label}) {{{', '.join(parts)}}}")

    lines.extend(["", "Relationship types and properties:"])
    for rel_type in sorted(rel_props):
        props = rel_props[rel_type]
        suffix = ""
        if props:
            suffix = " {" + ", ".join(
                f"{p} ({t})" for p, t in sorted(props.items())
            ) + "}"
        lines.append(f"  [:{rel_type}]{suffix}")

    lines.extend([
        "",
        "Graph structure (direction is significant; {...} are constant relationship values on that path):",
    ])
    for pat in idx["patterns"]:
        ann = pat.get("constants") or {}
        ann_text = ""
        if ann:
            ann_text = " {" + ", ".join(f"{k}: {v!r}" for k, v in sorted(ann.items())) + "}"
        lines.append(
            f"  ({_fmt_label_combo(pat['start_labels'])}) "
            f"-[:{pat['relationship']}{ann_text}]-> "
            f"({_fmt_label_combo(pat['end_labels'])})"
        )

    legacy = idx.get("legacy_variant_labels") or []
    if legacy:
        lines.extend([
            "",
            "Legacy/dynamic per-run labels were observed but intentionally omitted from the stable schema: "
            + ", ".join(legacy[:12])
            + (" ..." if len(legacy) > 12 else ""),
            "Do not construct Cypher by synthesising one of those labels unless a task explicitly requires it.",
        ])

    modelrun_props = node_props.get("ModelRun", {})
    if "modelType" in modelrun_props and "classRatio" in modelrun_props:
        lines.extend([
            "",
            "Model-run selection rule: identify an algorithm/training setting with "
            "(:ModelRun).modelType and (:ModelRun).classRatio. Prefer those properties over "
            "legacy modelName values or dynamic labels.",
        ])

    # Explicitly remind the generator that relationship values are not node properties.
    if any(rel_props.values()):
        lines.extend([
            "",
            "Relationship-property rule: when an answer field is listed on a relationship, bind that "
            "relationship variable and read the property from it (for example r.value or fi.shapValue).",
        ])

    return "\n".join(lines)


def find_schema_paths(schema_index: dict, source_labels: Iterable[str],
                      target_labels: Iterable[str], max_hops: int = 4,
                      limit: int = 8) -> list[list[dict]]:
    """Find bounded label-level paths while preserving each edge's true direction.

    Search itself is undirected so callers can connect two concepts regardless
    of which one appears first in the graph.  Each returned step includes
    ``traversal`` ("forward"/"reverse") plus the real ``start_labels`` and
    ``end_labels`` of the Neo4j relationship.
    """
    source = set(source_labels)
    target = set(target_labels)
    if not source or not target:
        return []

    adjacency: dict[str, list[tuple[str, dict, str]]] = defaultdict(list)
    for pat in schema_index.get("patterns", []):
        for a in pat["start_labels"]:
            for b in pat["end_labels"]:
                adjacency[a].append((b, pat, "forward"))
                adjacency[b].append((a, pat, "reverse"))

    results = []
    queue = deque((s, [], {s}) for s in source)
    shortest = None

    while queue and len(results) < limit:
        node, path, seen = queue.popleft()
        if shortest is not None and len(path) > shortest:
            continue
        if node in target and path:
            shortest = len(path)
            results.append(path)
            continue
        if len(path) >= max_hops:
            continue
        for nxt, pat, traversal in adjacency.get(node, []):
            if nxt in seen:
                continue
            step = {
                "start_labels": pat["start_labels"],
                "relationship": pat["relationship"],
                "end_labels": pat["end_labels"],
                "constants": pat.get("constants") or {},
                "traversal": traversal,
            }
            queue.append((nxt, path + [step], seen | {nxt}))

    return results


def format_schema_path(path: list[dict]) -> str:
    """Human-readable path preserving true relationship direction."""
    if not path:
        return ""
    parts = []
    for i, step in enumerate(path):
        if step["traversal"] == "forward":
            left = _fmt_label_combo(step["start_labels"])
            right = _fmt_label_combo(step["end_labels"])
            edge = f"-[:{step['relationship']}]->"
        else:
            left = _fmt_label_combo(step["end_labels"])
            right = _fmt_label_combo(step["start_labels"])
            edge = f"<-[:{step['relationship']}]-"
        if i == 0:
            parts.append(f"({left})")
        parts.append(edge)
        parts.append(f"({right})")
    return " ".join(parts)


if __name__ == "__main__":
    import os
    from neo4j import GraphDatabase

    uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    user = os.getenv("NEO4J_USER", "neo4j")
    password = os.getenv("NEO4J_PASSWORD", "12345678")
    database = os.getenv("NEO4J_DATABASE") or None

    driver = GraphDatabase.driver(uri, auth=(user, password))
    try:
        print(build_schema_card(driver, database=database))
    finally:
        driver.close()
