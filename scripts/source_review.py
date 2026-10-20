"""List sources that are due for manual review or still unverified.

Usage:  python scripts/source_review.py [--today YYYY-MM-DD]

This script does not fetch anything. A person opens each URL, re-checks the facts,
then edits content/sources.yaml (facts, version, checked, status, review_by).
"""
import argparse
import datetime as dt
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.content import load_all  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--today", default=dt.date.today().isoformat())
    today = parser.parse_args().today
    sources = load_all()["sources"]
    due = [(sid, s) for sid, s in sources.items() if s["review_by"] <= today]
    unverified = [(sid, s) for sid, s in sources.items() if s["status"] == "unverified" or not s["checked"]]
    print(f"Review date: {today}\n")
    print(f"Due for review ({len(due)}):")
    for sid, s in sorted(due, key=lambda x: x[1]["review_by"]):
        print(f"  {s['review_by']}  {sid:28} {s['url']}")
    print(f"\nUnverified or never checked ({len(unverified)}):")
    for sid, s in unverified:
        print(f"  {sid:28} checked={s['checked']}  {s['url']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
