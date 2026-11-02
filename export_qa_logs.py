"""
Export qa_log entries from ui_users.db into per-account JSONL files.

Usage:
    python export_qa_logs.py                    # export all users to ./qa_export/
    python export_qa_logs.py --out ./somedir    # custom output dir
    python export_qa_logs.py --user yao         # one specific user
    python export_qa_logs.py --since 2026-05-11 # only entries from this date
"""

import argparse
import json
import os
import sqlite3
import sys
from collections import Counter
from datetime import datetime


def export(db_path: str, out_dir: str, only_user: str | None, since: str | None):
    if not os.path.isfile(db_path):
        sys.exit(f"DB not found: {db_path}")
    os.makedirs(out_dir, exist_ok=True)

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    where, params = [], []
    if only_user:
        where.append("username = ?")
        params.append(only_user)
    if since:
        where.append("ts >= ?")
        params.append(since)
    where_sql = ("WHERE " + " AND ".join(where)) if where else ""

    # Per-user counts (for summary)
    user_rows = conn.execute(
        f"SELECT username, COUNT(*) AS n, MIN(ts) AS first_ts, MAX(ts) AS last_ts, "
        f"SUM(CASE WHEN error IS NULL THEN 1 ELSE 0 END) AS ok_n, "
        f"ROUND(AVG(duration_s), 2) AS avg_s "
        f"FROM qa_log {where_sql} GROUP BY username ORDER BY username",
        params,
    ).fetchall()

    if not user_rows:
        print("No matching qa_log entries.")
        return

    print(f"Exporting {sum(r['n'] for r in user_rows)} entries across {len(user_rows)} users\n")
    print(f"{'User':<25} {'#':>5}  {'OK':>5}  {'First':<19}  {'Last':<19}  {'avg_s':>8}")
    print("-" * 90)

    for user_row in user_rows:
        username = user_row["username"]
        rows = conn.execute(
            f"SELECT * FROM qa_log {where_sql} "
            f"AND username = ? ORDER BY ts" if where else
            f"SELECT * FROM qa_log WHERE username = ? ORDER BY ts",
            (*params, username) if where else (username,),
        ).fetchall()

        out_path = os.path.join(out_dir, f"{username}.jsonl")
        with open(out_path, "w", encoding="utf-8") as f:
            for row in rows:
                # Try to expand cypher_rows JSON for readability
                d = dict(row)
                if d.get("cypher_rows"):
                    try:
                        d["cypher_rows"] = json.loads(d["cypher_rows"])
                    except Exception:
                        pass
                f.write(json.dumps(d, ensure_ascii=False, default=str) + "\n")

        print(
            f"{username:<25} {user_row['n']:>5}  {user_row['ok_n']:>5}  "
            f"{user_row['first_ts']:<19}  {user_row['last_ts']:<19}  {user_row['avg_s'] or 0:>8.2f}"
        )

    # Also write a combined CSV summary
    summary_path = os.path.join(out_dir, "_summary.csv")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("username,total,ok,first_ts,last_ts,avg_duration_s\n")
        for r in user_rows:
            f.write(
                f"{r['username']},{r['n']},{r['ok_n']},{r['first_ts']},{r['last_ts']},{r['avg_s'] or 0}\n"
            )

    # Error stats (across all users)
    errs = conn.execute(
        f"SELECT error, COUNT(*) AS n FROM qa_log {where_sql} "
        f"AND error IS NOT NULL GROUP BY error ORDER BY n DESC LIMIT 10" if where else
        "SELECT error, COUNT(*) AS n FROM qa_log WHERE error IS NOT NULL GROUP BY error ORDER BY n DESC LIMIT 10",
        params,
    ).fetchall()
    if errs:
        print(f"\nTop errors:")
        for e in errs:
            print(f"  {e['n']:>4}× {e['error'][:80]}")

    conn.close()
    print(f"\nWrote {len(user_rows)} files + _summary.csv to {out_dir}/")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="./ui_users.db")
    ap.add_argument("--out", default="./qa_export")
    ap.add_argument("--user", default=None, help="Export only one user")
    ap.add_argument("--since", default=None, help="ISO date/datetime, e.g. 2026-05-11")
    args = ap.parse_args()
    export(args.db, args.out, args.user, args.since)


if __name__ == "__main__":
    main()
