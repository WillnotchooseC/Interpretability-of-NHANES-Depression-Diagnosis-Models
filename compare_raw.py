"""
compare_raw.py - Compare RAW (no truncation) metrics across experiments
========================================================================
Reads metrics_both.json from each experiment folder.

Usage:
    python compare_raw.py                     # compare all 6
    python compare_raw.py --exp 1 6           # compare exp1 vs grpo
    python compare_raw.py --exp 1 2 3 4 5 6   # all
"""

import os
import json
import argparse

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_ROOT = os.path.join(BASE_DIR, "nhanes_model")

EXP_NAMES = {
    1: "SFT 1000 domain",
    2: "SFT Mix Replace",
    3: "SFT Domain Expand",
    4: "SFT SQA Expand",
    5: "SFT SQA Large",
    6: "SFT+GRPO 1000 domain",
}

METRICS_TABLE = [
    ("Semantic Mean",           "semantic_sample_mean",  ".4f", True),
    ("Semantic Std",            "semantic_sample_std",   ".4f", False),
    ("Semantic >=0.92 (%)",     "semantic_perfect_rate", ".1f", True),
    ("Semantic >=0.85 (%)",     "semantic_high_rate",    ".1f", True),
    ("Semantic <0.6 count",     "semantic_low_count",    "d",   False),
    ("Count Exact Match (%)",   "count_exact_rate",      ".1f", True),
    ("Count Avg Ref",           "count_avg_ref",         ".2f", None),
    ("Count Avg Pred",          "count_avg_pred",        ".2f", None),
    ("Count |Diff|",            "count_abs_diff",        ".2f", False),
    ("Over-decompose",          "count_over",            "d",   False),
    ("Under-decompose",         "count_under",           "d",   False),
    ("Redundancy Rate (%)",     "redundancy_rate",       ".1f", False),
    ("Redundancy Total Pairs",  "redundancy_total",      "d",   False),
    ("Format Valid (%)",        "format_rate",           ".1f", True),
    ("Type Accuracy (%)",       "type_accuracy",         ".1f", True),
]


def load_metrics(exp_id):
    # Try metrics_both.json first
    path = os.path.join(OUTPUT_ROOT, f"exp{exp_id}", "metrics_both.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)

    # Fallback: regular metrics.json (treat as truncated only)
    path2 = os.path.join(OUTPUT_ROOT, f"exp{exp_id}", "metrics.json")
    if os.path.exists(path2):
        with open(path2) as f:
            return {"truncated": json.load(f)}

    return None


def format_val(val, fmt, is_best=False):
    if isinstance(val, int) or fmt == "d":
        s = str(int(val))
    else:
        s = f"{val:{fmt}}"
    if is_best:
        s = f"*{s}"
    return s


def print_table(exps, all_data, mode, names):
    print(f"\n{'='*120}")
    print(f"  {mode.upper()} METRICS COMPARISON")
    print(f"{'='*120}")

    # Header with experiment names
    header = f"  {'Metric':<28s}"
    for e in exps:
        label = f"Exp{e}"
        header += f"{label:>14s}"
    print(header)

    name_row = f"  {'':28s}"
    for e in exps:
        name = names.get(e, "")
        if len(name) > 13:
            name = name[:12] + "."
        name_row += f"{name:>14s}"
    print(name_row)
    print("  " + "-" * (28 + 14 * len(exps)))

    for label, key, fmt, hb in METRICS_TABLE:
        vals = []
        for e in exps:
            d = all_data.get(e, {}).get(mode, {})
            vals.append(d.get(key, None))

        valid_vals = [v for v in vals if v is not None]
        if hb is True and valid_vals:
            best = max(valid_vals)
        elif hb is False and valid_vals:
            best = min(valid_vals)
        else:
            best = None

        row = f"  {label:<28s}"
        for v in vals:
            if v is None:
                row += f"{'N/A':>14s}"
            else:
                is_best = (best is not None and v == best and len(exps) > 1)
                row += f"{format_val(v, fmt, is_best):>14s}"
        print(row)

    print(f"\n  * = best value")
    print(f"{'='*120}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", type=int, nargs="+", default=[1, 2, 3, 4, 5, 6])
    args = parser.parse_args()

    all_data = {}
    for e in args.exp:
        data = load_metrics(e)
        if data:
            all_data[e] = data
            print(f"  Loaded Exp{e}: {EXP_NAMES.get(e, '')}")
        else:
            print(f"  WARNING: No metrics found for Exp{e}")

    if not all_data:
        print("  ERROR: No metrics files found.")
        return

    exps = sorted(all_data.keys())

    # Show RAW
    has_raw = any("raw" in all_data[e] for e in exps)
    if has_raw:
        print_table(exps, all_data, "raw", EXP_NAMES)
    else:
        print("  No raw metrics found. Run evaluate.py --both first.")


if __name__ == "__main__":
    main()