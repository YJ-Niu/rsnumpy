"""Per-file test status collector -> tests/test_config.txt.

A tiny pytest plugin (loaded via ``-p _status_plugin``) that records, for every
test **file**, the worst outcome seen in that file:

    all tests passed            -> flag 1
    any failed / error / skipped -> flag 0

On ``sessionfinish`` (controller process only, so it also works under xdist)
it merges the results into the config file:

* ``RSNUMPY_TEST_CONFIG_FULL=1``  — full rebuild: every ``tests/**/test_*.py``
  candidate starts at 0 and only files that ran **and** fully passed become 1.
  Files with collection errors are unseen, therefore stay 0.
* otherwise (merge mode, used by ``--test debug``) — only files actually seen
  in this run are updated; entries for files that were not run are preserved.

Only writes anything when ``RSNUMPY_TEST_CONFIG`` is set, so plain
``pytest tests`` runs are unaffected.
"""
import os

_RANK = {"passed": 0, "skipped": 1, "failed": 2}
_status = {}  # relpath -> "passed" | "skipped" | "failed"


def _mark(fspath, outcome):
    if not fspath:
        return
    rel = os.path.relpath(str(fspath), os.getcwd()).replace(os.sep, "/")
    if not rel.endswith(".py"):
        return
    rank = _RANK.get(outcome, _RANK["failed"])
    cur = _status.get(rel)
    if cur is None or rank > _RANK[cur]:
        _status[rel] = outcome if outcome in _RANK else "failed"


def pytest_runtest_logreport(report):
    fspath = getattr(report, "fspath", None)
    if fspath is None:
        loc = getattr(report, "location", None)
        fspath = loc[0] if loc else None
    _mark(fspath, report.outcome)


def pytest_collectreport(report):
    # collection error -> the file never produced tests; force it to 0
    if report.outcome != "passed":
        _mark(getattr(report, "fspath", None), "failed")


def pytest_sessionfinish(session, exitstatus):
    # xdist workers also fire this hook; only the controller owns the merged
    # report state (worker processes have a `workerinput` attribute).
    if hasattr(session.config, "workerinput"):
        return
    cfg = os.environ.get("RSNUMPY_TEST_CONFIG")
    if not cfg:
        return
    cfg = os.path.abspath(cfg)
    full = os.environ.get("RSNUMPY_TEST_CONFIG_FULL") == "1"

    entries = {}
    if not full and os.path.exists(cfg):
        with open(cfg, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                if len(parts) == 2 and parts[1] in ("0", "1"):
                    entries[parts[0]] = parts[1]
    elif full:
        # full rebuild: every candidate test file starts at 0; only files that
        # ran and fully passed get promoted to 1 below.
        for root, dirs, files in os.walk("tests"):
            dirs[:] = [d for d in dirs if d != "__pycache__"]
            for fn in files:
                if fn.startswith("test_") and fn.endswith(".py"):
                    rel = os.path.join(root, fn).replace(os.sep, "/")
                    entries[rel] = "0"

    for rel, outcome in _status.items():
        entries[rel] = "1" if outcome == "passed" else "0"

    with open(cfg, "w", encoding="utf-8") as f:
        f.write(
            "# rsnumpy 测试配置: <测试文件> <标志>\n"
            "#   1 = 上次全量运行全部通过   0 = 存在 failed/skipped/error（或尚未跑通）\n"
            "# 由 ./build_wheel.sh --test all 自动重新生成；--test debug 只跑标志为 0 的文件。\n"
        )
        for rel in sorted(entries):
            f.write(f"{rel} {entries[rel]}\n")
