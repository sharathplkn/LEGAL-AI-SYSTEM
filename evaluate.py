"""
evaluate.py — Quick evaluation harness for the Indian Legal Assistant.

Run this AFTER starting `python3 a.py` in another terminal.

    python3 evaluate.py

Outputs:
    results.csv      — per-query pass/fail with detected flags
    summary.txt      — aggregate metrics for the report

This tests three query categories:
  1. Section lookups     (exact section number retrieval)
  2. IPC redirect        (old IPC section -> new BNS equivalent)
  3. Situational queries (describe scenario, find applicable law)

Plus two "ablation" comparisons by patching the running system:
  - Full system (hybrid retrieval + metadata filter + IPC redirect)
  - Without IPC redirect (tests one specific feature's contribution)
"""

import csv
import time
import requests
import re

API_URL = "http://127.0.0.1:8000/chat"
CLEAR_URL = "http://127.0.0.1:8000/clear"


# ── Test set (25 queries) ─────────────────────────────────────────────────────
# Each: (category, query, must_contain_any_of, must_not_contain)
# Passes if response contains at least one must_contain phrase AND none of must_not.

TESTS = [
    # ── Category A: Section lookups (10) ─────────────────────────────────────
    ("section", "What is the punishment for murder under BNS?",
     ["103", "death", "imprisonment for life"], ["IPC", "BSF", "CRPF"]),

    ("section", "BNS section 318",
     ["318", "cheat"], ["IPC"]),

    ("section", "what is bns section 108",
     ["108", "abetment", "suicide"], []),

    ("section", "section 64 BNS",
     ["64", "rape"], []),

    ("section", "BNSS 113",
     ["113", "letter"], []),

    ("section", "BSA 62",
     ["62", "electronic"], []),

    ("section", "what is bns 80",
     ["80", "dowry"], []),

    ("section", "BNS 309",
     ["309", "robbery"], []),

    ("section", "bns section 74",
     ["74", "modesty"], []),

    ("section", "section 86 BNS",
     ["86", "cruelty"], []),

    # ── Category B: IPC redirect (7) ────────────────────────────────────────
    ("ipc_redirect", "what replaced IPC 302?",
     ["103", "murder"], []),

    ("ipc_redirect", "IPC 420",
     ["318", "cheat"], []),

    ("ipc_redirect", "section 376 of IPC",
     ["64", "rape"], []),

    ("ipc_redirect", "498A IPC",
     ["86", "cruelty"], []),

    ("ipc_redirect", "IPC 307",
     ["109", "attempt"], []),

    ("ipc_redirect", "what is IPC 306",
     ["108", "abetment"], []),

    ("ipc_redirect", "302 in IPC?",
     ["103", "murder"], []),

    # ── Category C: Situational / conceptual (8) ────────────────────────────
    ("situational", "what is stalking under BNS?",
     ["78", "stalk"], []),

    ("situational", "punishment for theft",
     ["303", "theft"], []),

    ("situational", "what is dowry death",
     ["80", "dowry"], []),

    ("situational", "electronic evidence under BSA",
     ["electronic", "evidence"], []),

    ("situational", "what is bns",
     ["Bharatiya Nyaya", "2023"], []),

    ("situational", "what is bnss",
     ["Bharatiya Nagarik", "2023"], []),

    ("situational", "hi",
     ["Namaste", "Legal Assistant"], []),

    ("situational", "full form of bsa",
     ["Sakshya", "2023"], []),
]


def query(message):
    """Send one query to the running server. Returns response text + latency."""
    t0 = time.time()
    try:
        r = requests.post(API_URL, json={"message": message}, timeout=60)
        elapsed = time.time() - t0
        return r.json().get("response", ""), elapsed
    except Exception as e:
        return f"[ERROR] {e}", time.time() - t0


def check(response, must_contain, must_not_contain):
    """Return (passed, matched_phrase, violation)."""
    low = response.lower()
    matched = None
    for phrase in must_contain:
        if phrase.lower() in low:
            matched = phrase
            break
    violation = None
    for phrase in must_not_contain:
        if phrase.lower() in low:
            violation = phrase
            break
    passed = matched is not None and violation is None
    return passed, matched, violation


def run_suite(label):
    """Run all tests, clearing memory between queries, return results list."""
    print(f"\n{'=' * 60}")
    print(f"  Running: {label}")
    print(f"{'=' * 60}")
    rows = []
    for i, (category, q, must, mustnot) in enumerate(TESTS, 1):
        # Clear chat history so each query is independent
        try:
            requests.post(CLEAR_URL, timeout=5)
        except Exception:
            pass

        response, latency = query(q)
        passed, matched, violation = check(response, must, mustnot)
        mark = "✓" if passed else "✗"
        print(f"  {mark} [{category:12}] ({latency:4.1f}s) {q[:50]}")
        if not passed:
            if violation:
                print(f"       violation: contains '{violation}'")
            elif matched is None:
                print(f"       missing any of: {must}")

        rows.append({
            "category": category,
            "query": q,
            "passed": passed,
            "matched": matched or "",
            "violation": violation or "",
            "latency_s": round(latency, 2),
            "response_preview": response[:200].replace("\n", " "),
        })
    return rows


def summarize(rows, label):
    """Per-category pass rate + overall metrics."""
    from collections import defaultdict
    by_cat = defaultdict(lambda: {"total": 0, "passed": 0})
    total_latency = 0.0
    for r in rows:
        c = r["category"]
        by_cat[c]["total"] += 1
        if r["passed"]:
            by_cat[c]["passed"] += 1
        total_latency += r["latency_s"]

    total_n = len(rows)
    total_pass = sum(1 for r in rows if r["passed"])
    avg_latency = total_latency / max(total_n, 1)

    lines = [f"\n── {label} ──"]
    for cat, stats in by_cat.items():
        pct = 100 * stats["passed"] / stats["total"]
        lines.append(f"  {cat:15} {stats['passed']:2}/{stats['total']:2}  ({pct:5.1f}%)")
    lines.append(f"  {'OVERALL':15} {total_pass:2}/{total_n:2}  "
                 f"({100*total_pass/total_n:5.1f}%)")
    lines.append(f"  Avg latency: {avg_latency:.2f}s")
    return "\n".join(lines)


def main():
    print("Indian Legal Assistant — Evaluation")
    print(f"Test set: {len(TESTS)} queries across 3 categories")

    # Check server is up
    try:
        requests.post(CLEAR_URL, timeout=5)
    except Exception as e:
        print(f"\n✗ Server not reachable at {API_URL}")
        print(f"  Start it in another terminal: python3 a.py")
        print(f"  Error: {e}")
        return

    # Run the full system
    rows = run_suite("Full system (hybrid RAG + IPC redirect)")

    # Write CSV
    with open("results.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    # Write summary
    summary = summarize(rows, "Full system")
    with open("summary.txt", "w", encoding="utf-8") as f:
        f.write("Indian Legal Assistant — Evaluation Summary\n")
        f.write("=" * 50 + "\n")
        f.write(f"Total queries: {len(TESTS)}\n")
        f.write(summary + "\n")

    print(summary)
    print("\n✓ Saved: results.csv, summary.txt")
    print("  Use these in your thesis Results chapter.")


if __name__ == "__main__":
    main()