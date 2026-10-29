#!/usr/bin/env python3
"""Crash-isolated numpy-compatibility runner.

Runs every numpy test copied into this directory in its **own subprocess**, so a
C-extension abort / segfault in one test cannot take the whole run down (rsnumpy
is not yet as hard as numpy against every edge case).  Reports how
numpy-compatible rsnumpy is:

    passed   — rsnumpy handles this numpy case correctly
    failed   — a real rsnumpy gap (wrong result / missing API)
    skipped  — needs a numpy C-helper / feature rsnumpy lacks (see conftest)
    crashed  — rsnumpy C ext aborts / segfaults (robustness bug to track)

Usage:  python run_compat.py            (all files in this dir)
        python run_compat.py test_indexing.py   (a subset)
"""
import concurrent.futures as cf
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
# rsnumpy repo root (HERE is <root>/test/numpy) — pytest nodeids are relative to
# the rootdir (the repo root, where pyproject.toml lives), so run subprocesses
# from there too, otherwise per-test nodeids don't resolve.
ROOT = os.path.dirname(os.path.dirname(HERE))
RELPATH = "test/numpy"
PY = sys.executable
WORKERS = int(os.environ.get("COMPAT_WORKERS", "8"))


def collect(files):
    """Return the list of test nodeids for the given files (relative to ROOT)."""
    args = [PY, "-m", "pytest", "--collect-only", "-q", "-p", "no:cacheprovider"] + files
    r = subprocess.run(args, capture_output=True, text=True, cwd=ROOT)
    nodeids = []
    for line in r.stdout.splitlines():
        line = line.strip()
        # a nodeid line starts with a .py path and contains '::'
        if line.endswith(".py") is False and ".py::" in line and line.count("::") >= 1:
            nodeids.append(line.split(" ")[0])
    # de-dup, keep order
    seen, out = set(), []
    for n in nodeids:
        if n not in seen:
            seen.add(n)
            out.append(n)
    return out


def classify(r):
    out = (r.stdout or "") + (r.stderr or "")
    if re.search(r"Fatal Python error|Segmentation fault|\bAborted\b|core dumped|Floating point exception", out):
        return "crashed"
    if re.search(r"\b1 skipped\b|\bSKIPPED\b", out):
        return "skipped"
    if re.search(r"\b1 failed\b|\bFAILED\b", out):
        return "failed"
    if re.search(r"\b1 passed\b|\bpassed\b", out) or r.returncode == 0:
        return "passed"
    return f"error(rc={r.returncode})"


def run_one(nodeid):
    r = subprocess.run(
        [PY, "-m", "pytest", nodeid, "-q", "-p", "no:cacheprovider", "--no-header", "-x"],
        capture_output=True, text=True, cwd=ROOT, timeout=180,
    )
    return classify(r), nodeid


def main(files):
    if not files:
        files = sorted(f for f in os.listdir(HERE)
                       if f.startswith("test_") and f.endswith(".py"))
        files = [f"{RELPATH}/{f}" for f in files]
    else:
        files = [f if "/" in f else f"{RELPATH}/{f}" for f in files]
    print(f"collecting from: {files}", file=sys.stderr)
    nodeids = collect(files)
    print(f"collected {len(nodeids)} test nodeids", file=sys.stderr)
    tally = {"passed": 0, "failed": 0, "skipped": 0, "crashed": 0}
    detail = {"crashed": [], "failed": [], "error": []}
    with cf.ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futures = [ex.submit(run_one, n) for n in nodeids]
        for i, fut in enumerate(cf.as_completed(futures), 1):
            status, nodeid = fut.result()
            key = status if status in tally else "error"
            if status in tally:
                tally[status] += 1
            else:
                tally.setdefault(status, 0)
                tally[status] += 1
            if status == "crashed":
                detail["crashed"].append(nodeid)
            elif status == "failed":
                detail["failed"].append(nodeid)
            elif status not in ("passed", "skipped"):
                detail["error"].append((status, nodeid))
            if i % 50 == 0 or i == len(nodeids):
                print(f"  ...{i}/{len(nodeids)}", file=sys.stderr)

    total = sum(tally.values())
    print("\n================ numpy-compatibility report ================")
    for k in ("passed", "failed", "skipped", "crashed"):
        print(f"  {k:8}: {tally.get(k, 0)}")
    for k in tally:
        if k not in ("passed", "failed", "skipped", "crashed"):
            print(f"  {k:8}: {tally[k]}")
    print(f"  {'total':8}: {total}")
    if nodeids:
        print(f"  numpy-compatible (passed/total): {tally.get('passed', 0)}/{total} "
              f"= {100.0 * tally.get('passed', 0) / total:.1f}%")
    print("=============================================================\n")

    if detail["crashed"]:
        print(f"CRASHED tests ({len(detail['crashed'])}) — rsnumpy C-ext robustness bugs:")
        for n in detail["crashed"]:
            print(f"  {n}")
    if detail["error"]:
        print(f"\nERRORS ({len(detail['error'])}):")
        for s, n in detail["error"]:
            print(f"  [{s}] {n}")
    # failed tests: only show count + first 40 (can be many)
    if detail["failed"]:
        print(f"\nFAILED tests ({len(detail['failed'])}) — rsnumpy gaps (first 40 shown):")
        for n in detail["failed"][:40]:
            print(f"  {n}")
        if len(detail["failed"]) > 40:
            print(f"  ... and {len(detail['failed']) - 40} more")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
