"""
compare.py - Cross-experiment comparison of raw & truncated metrics
====================================================================
Reads metrics_both.json from each experiment folder and generates
a comparison table.

Usage:
    python compare.py                          # compare all 5
    python compare.py --exp 1 2 3              # compare specific ones
    python compare.py --mode raw               # only raw
    python compare.py --mode truncated         # only truncated
    python compare.py --mode both              # side by side (default)
"""

import os
import json
import argparse

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_ROOT = os.path.join(BASE_DIR, "nhanes_model")

EXP_NAMES = {
    1: "Base (1000 domain)",
    2: "Mix Replace",
    3: "Domain Expand",
    4: "SQA Expand",
    5: "SQA Large",
}

METRICS_TABLE = [
    # (display name, key, format, higher_is_better)
    ("Semantic Mean",           "semantic_sample_mean",  ".4f", True),
    ("Semantic Std",            "semantic_sample_std",   ".4f", False),
    ("Semantic ≥0.92 (%)",      "semantic_perfect_rate", ".1f", True),
    ("Semantic ≥0.85 (%)",      "semantic_high_rate",    ".1f", True),
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
    path = os.path.join(OUTPUT_ROOT, f"exp{exp_id}", "metrics_both.json")
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)

    # Fallback: try regular metrics.json
    path2 = os.path.join(OUTPUT_ROOT, f"exp{exp_id}", "metrics.json")
    if os.path.exists(path2):
        with open(path2, "r") as f:
            m = json.load(f)
            return {"truncated": m}

    return None


def format_val(val, fmt, is_best=False):
    if isinstance(val, int) or fmt == "d":
        s = str(int(val))
    else:
        s = f"{val:{fmt}}"
    if is_best:
        s = f"*{s}"
    return s


def print_table(exps, all_data, mode):
    print(f"\n{'='*100}")
    print(f"  {mode.upper()} METRICS COMPARISON")
    print(f"{'='*100}")

    header = f"  {'Metric':<28s}" + "".join(f"{'Exp'+str(e):>14s}" for e in exps)
    print(header)
    print("  " + "-" * (28 + 14 * len(exps)))

    for label, key, fmt, hb in METRICS_TABLE:
        vals = []
        for e in exps:
            d = all_data.get(e, {}).get(mode, {})
            vals.append(d.get(key, None))

        # Find best
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
    print(f"{'='*100}")


def print_delta_table(exps, all_data):
    """Show improvement from truncation: truncated - raw"""
    print(f"\n{'='*100}")
    print(f"  TRUNCATION IMPACT (truncated - raw)")
    print(f"{'='*100}")

    header = f"  {'Metric':<28s}" + "".join(f"{'Exp'+str(e):>14s}" for e in exps)
    print(header)
    print("  " + "-" * (28 + 14 * len(exps)))

    delta_metrics = [
        ("Semantic Mean",          "semantic_sample_mean",  "+.4f"),
        ("Count Exact Match (%)",  "count_exact_rate",      "+.1f"),
        ("Count |Diff|",           "count_abs_diff",        "+.2f"),
        ("Redundancy Rate (%)",    "redundancy_rate",       "+.1f"),
        ("Type Accuracy (%)",      "type_accuracy",         "+.1f"),
    ]

    for label, key, fmt in delta_metrics:
        row = f"  {label:<28s}"
        for e in exps:
            raw = all_data.get(e, {}).get("raw", {}).get(key)
            trunc = all_data.get(e, {}).get("truncated", {}).get(key)
            if raw is not None and trunc is not None:
                delta = trunc - raw
                row += f"{delta:{fmt}:>14s}" if False else f"{'%s' % f'{delta:{fmt}}':>14s}"
            else:
                row += f"{'N/A':>14s}"
        print(row)

    print(f"{'='*100}")


def save_comparison(exps, all_data, output_path):
    result = {}
    for e in exps:
        result[f"exp{e}"] = all_data.get(e, {})
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n  Saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Compare experiment metrics")
    parser.add_argument("--exp", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    parser.add_argument("--mode", type=str, default="both",
                        choices=["raw", "truncated", "both"],
                        help="Which metrics to show")
    args = parser.parse_args()

    # Load all
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

    if args.mode == "raw":
        print_table(exps, all_data, "raw")
    elif args.mode == "truncated":
        print_table(exps, all_data, "truncated")
    else:
        # Show both + delta
        has_raw = any("raw" in all_data[e] for e in exps)
        has_trunc = any("truncated" in all_data[e] for e in exps)

        if has_raw:
            print_table(exps, all_data, "raw")
        if has_trunc:
            print_table(exps, all_data, "truncated")
        if has_raw and has_trunc:
            print_delta_table(exps, all_data)

    # Save
    save_comparison(exps, all_data, os.path.join(OUTPUT_ROOT, "comparison_all.json"))


if __name__ == "__main__":
    main()