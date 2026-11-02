"""
Re-score gold-row retrieval for exp1_5 / exp2 / exp3 across 3 reps, removing
the two false-negative sources in run_finetuned_pipeline_eval.score_against_gold:

  (1) format/sign: gold stores signed shap_value (-0.090); Cypher legitimately
      returns abs_shap_value (0.09) or a rounded/stringified number -> strict
      set-intersection scores it as a miss.
  (2) superset gold: gold_results often holds the FULL ranked list while the
      question asks for top-k; a correct top-k retrieval is penalised for the
      rows it correctly omitted.

Reported per method x rep x split:
  strict_recall      -- original score_against_gold (control)
  tol_recall         -- matched gold rows / all gold rows, tolerant value match
                        (sign-agnostic, 1e-3 relative rounding tol, numeric-string
                         coercion, biomarker alias + token-subset for strings).
                        Conservative: still divides by the full (possibly superset) gold.
  tol_precision      -- matched retrieved rows / all retrieved rows, same matcher.
                        Correct top-k subset -> 1.0. Upper-ish bound.
The true retrieval quality is bracketed by [tol_recall, tol_precision].

Usage: python rescore_gold_recall.py
"""
import json
import re
import numpy as np
from neo4j import GraphDatabase

from entity_retrieval import build_vocab
from run_finetuned_pipeline_eval import score_against_gold, build_echo_values

NEO4J = ("bolt://localhost:7687", ("neo4j", "12345678"))
GOLD_FILE = "ood_data/curriculum_test_120_gold.json"
INPUT_BASENAME = "progressive_decomp_for_experiments"
METHODS = {
    "exp1_5": "exp1_5_baseline_original_input",
    "exp2": "exp2_fewshot_graphrag",
    "exp3": "exp3_fewshot_graphrag_rescue",
}
REPS = [1, 2, 3]
UNSEEN_IDX = set([10,11,12,13,14,15,16,17,18,19,80,81,82,83,84,85,86,87,88,89,
                  100,101,102,103,104,105,106,107,108,109,110,111,112,113,114,
                  115,116,117,118,119])
SPLITS = {"overall": lambda i: True,
          "seen_80": lambda i: i not in UNSEEN_IDX,
          "unseen_40": lambda i: i in UNSEEN_IDX}

BIOMARKER_ALIASES = {
    "25-hydroxyvitamin d3": "vitamin d3", "25-hydroxyvitamin d2": "vitamin d2",
    "25 hydroxyvitamin d3": "vitamin d3", "25 hydroxyvitamin d2": "vitamin d2",
}
_STOP = {"the", "a", "of", "and", "value", "level", "blood", "serum", "total", "direct"}


def gold_rows_for(sample):
    rows = []
    for g in sample.get("gold_results", []):
        r = g.get("result")
        if isinstance(r, list):
            rows.extend(x for x in r if isinstance(x, dict))
        elif isinstance(r, dict):
            rows.append(r)
    return rows


def _to_num(v):
    if isinstance(v, bool):
        return None
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, str):
        s = v.strip().replace(",", "")
        if re.fullmatch(r"-?\d+(\.\d+)?([eE]-?\d+)?", s):
            return float(s)
    return None


def _canon_str(s):
    s = str(s).strip().lower()
    s = BIOMARKER_ALIASES.get(s, s)
    return s


def _tokens(s):
    return {t for t in re.findall(r"[a-z0-9]+", _canon_str(s)) if t not in _STOP}


def val_match(a, b, sign_agnostic=True):
    an, bn = _to_num(a), _to_num(b)
    if an is not None and bn is not None:
        scale = max(abs(an), abs(bn), 1.0)
        if abs(an - bn) <= 1e-3 * scale:
            return True
        if sign_agnostic and abs(abs(an) - abs(bn)) <= 1e-3 * scale:
            return True
        return False
    if an is not None or bn is not None:
        return False
    sa, sb = _canon_str(a), _canon_str(b)
    if sa == sb:
        return True
    ta, tb = _tokens(sa), _tokens(sb)
    if ta and tb and (ta <= tb or tb <= ta):
        return len(ta & tb) / min(len(ta), len(tb)) >= 0.8
    return False


def _norm_leaves(v, out):
    if isinstance(v, dict):
        for vv in v.values():
            _norm_leaves(vv, out)
    elif isinstance(v, (list, tuple)):
        for vv in v:
            _norm_leaves(vv, out)
    elif v is not None:
        out.append(v)


def row_vals(row, echoes):
    leaves = []
    _norm_leaves(row, leaves)
    out = []
    for lf in leaves:
        n = _to_num(lf)
        key = round(n, 4) if n is not None else _canon_str(lf)
        if key in echoes:
            continue
        out.append(lf)
    return out


def tolerant_scores(actual_rows, gold_rows, echo_values):
    echoes = set()
    for e in (echo_values or set()):
        n = _to_num(e)
        echoes.add(round(n, 4) if n is not None else _canon_str(e))
    gold = [row_vals(r, echoes) for r in gold_rows]
    act = [row_vals(r, echoes) for r in actual_rows]
    gold = [g for g in gold if g]
    act_nonempty = [a for a in act if a]
    if not gold:
        return {"recall": None, "precision": None}

    # recall: each gold row's best containment by some actual row (greedy consume)
    remaining = list(range(len(act)))
    covs = []
    for gv in gold:
        best, best_j = 0.0, None
        for j in remaining:
            av = act[j]
            if not av:
                continue
            hit = sum(1 for x in gv if any(val_match(x, y) for y in av))
            sc = hit / len(gv)
            if sc > best:
                best, best_j = sc, j
        covs.append(best)
        if best_j is not None and best > 0:
            remaining.remove(best_j)
    recall = sum(covs) / len(covs)

    # precision: each retrieved row's best containment by some gold row
    if not act_nonempty:
        precision = 0.0
    else:
        rem_g = list(range(len(gold)))
        pcs = []
        for av in act_nonempty:
            best, best_k = 0.0, None
            for k in rem_g:
                gv = gold[k]
                hit = sum(1 for x in av if any(val_match(x, y) for y in gv))
                sc = hit / len(av)
                if sc > best:
                    best, best_k = sc, k
            pcs.append(best)
            if best_k is not None and best > 0:
                rem_g.remove(best_k)
        precision = sum(pcs) / len(pcs)
    return {"recall": recall, "precision": precision}


def rep_path(full, rep):
    d = f"results_{full}" if rep == 1 else f"results_{full}_run{rep}"
    return f"{d}/{INPUT_BASENAME}_{full}.json"


def main():
    gold = json.load(open(GOLD_FILE, encoding="utf-8"))
    n = len(gold)
    drv = GraphDatabase.driver(NEO4J[0], auth=NEO4J[1])
    drv.verify_connectivity()
    vocab = build_vocab(drv)
    drv.close()
    echoes_all = [build_echo_values(g["input"], vocab) for g in gold]
    gold_rows_all = [gold_rows_for(g) for g in gold]

    per = {m: {} for m in METHODS}
    for m, full in METHODS.items():
        for rep in REPS:
            try:
                det = json.load(open(rep_path(full, rep), encoding="utf-8"))["details"]
            except FileNotFoundError:
                continue
            rows = []
            for i, d in enumerate(det):
                actual = []
                for st in d.get("steps", []):
                    actual.extend(st.get("records") or [])
                strict = score_against_gold(actual, gold_rows_all[i], echoes_all[i])
                tol = tolerant_scores(actual, gold_rows_all[i], echoes_all[i])
                rows.append({"i": i, "strict": strict["coverage"],
                             "tol_recall": tol["recall"], "tol_prec": tol["precision"]})
            per[m][rep] = rows

    def agg(rows, key, filt):
        v = [r[key] for r in rows if filt(r["i"]) and r[key] is not None]
        return float(np.mean(v)) * 100 if v else None

    print(f"\n{'method':<10}{'split':<11}{'strict':>10}{'tol_recall':>13}{'tol_precision':>15}   (3-run mean, %)")
    print("-" * 74)
    summary = {}
    for m in METHODS:
        summary[m] = {}
        for sname, filt in SPLITS.items():
            trip = {"strict": [], "tol_recall": [], "tol_prec": []}
            for rep, rows in per[m].items():
                for k in trip:
                    val = agg(rows, k, filt)
                    if val is not None:
                        trip[k].append(val)
            row = {k: (round(np.mean(v), 1), round(np.std(v, ddof=1) if len(v) > 1 else 0.0, 1))
                   for k, v in trip.items() if v}
            summary[m][sname] = row
            def fmt(k):
                mm, ss = row.get(k, (None, None))
                return "--" if mm is None else f"{mm:.1f}±{ss:.1f}"
            print(f"{m:<10}{sname:<11}{fmt('strict'):>10}{fmt('tol_recall'):>13}{fmt('tol_prec'):>15}")

    # biggest strict->tol_recall jumps on unseen (sanity-check for over-generous matching)
    print("\nLargest strict -> tol_recall gains (exp1_5, unseen, rep1):")
    rows = per["exp1_5"][1]
    diffs = sorted(((r["tol_recall"] - r["strict"], r["i"]) for r in rows
                    if r["i"] in UNSEEN_IDX and r["strict"] is not None and r["tol_recall"] is not None),
                   reverse=True)[:10]
    for dlt, i in diffs:
        r = next(x for x in rows if x["i"] == i)
        print(f"  idx {i:3d}: strict {r['strict']:.2f} -> tol_recall {r['tol_recall']:.2f} "
              f"(tol_prec {r['tol_prec']:.2f})   {gold[i]['input'][:70]}")

    json.dump({m: {str(rep): rows for rep, rows in per[m].items()} for m in METHODS},
              open("cypher_results/rescore_gold_recall_per_sample.json", "w"), indent=1, default=str)
    json.dump(summary, open("cypher_results/rescore_gold_recall_summary.json", "w"), indent=2)
    print("\nSaved -> cypher_results/rescore_gold_recall_{summary,per_sample}.json")


if __name__ == "__main__":
    main()
