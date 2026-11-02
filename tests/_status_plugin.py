"""Per-file test status collector -> tests/test_config.txt（边测边更新）。

A tiny pytest plugin (loaded via ``-p _status_plugin``) that records, for every
test **file**, the worst outcome seen in that file:

    all tests passed            -> flag 1
    any failed / error / skipped -> flag 0

增量落盘（核心）：每跑完一个测试文件，立即把该文件标志原子写回配置文件
（临时文件 + os.replace），不必等整轮结束。中途 Ctrl+C / 崩溃退出时，
已跑完的文件标志都是新的——可直接按清单先修前面的错误。

模式：

* ``RSNUMPY_TEST_CONFIG_FULL=1``  —— 全量重建：sessionstart 先把
  ``tests/**/test_*.py`` 全部候选置 0 落盘（覆盖旧文件），之后边跑边把
  跑完的文件翻成实际结果；collection 错误文件立即置 0。
* 否则（merge 模式，``--test debug`` 用）—— 只更新本轮实际跑到的文件，
  其余条目保持原样；新出现的文件跑完后追加。

只有 controller 进程写盘（xdist worker 只上报报告，不直接写），
避免多进程并发写同一文件。

Only writes anything when ``RSNUMPY_TEST_CONFIG`` is set, so plain
``pytest tests`` runs are unaffected.
"""
import os

_RANK = {"passed": 0, "skipped": 1, "failed": 2}
_status = {}   # relpath -> "passed" | "skipped" | "failed"（该文件最坏结果）
_pending = {}  # relpath -> 该文件的测试项总数（controller 收集期统计）
_seen = {}     # relpath -> 已出过报告的 nodeid 集合（setup/call/teardown 去重）
_config = None  # pytest.Config（pytest_configure 里保存）

_HEADER = (
    "# rsnumpy 测试配置: <测试文件> <标志>\n"
    "#   1 = 上次全部通过   0 = 存在 failed/skipped/error（或尚未跑通）\n"
    "# 由 build_wheel.sh 的测试阶段自动维护：每跑完一个文件立即更新（边测边更新），\n"
    "# --test debug 只跑标志为 0 的文件；--test all 全量重建。\n"
)


def _is_controller():
    return _config is not None and not hasattr(_config, "workerinput")


def _rel(fspath):
    if not fspath:
        return None
    rel = os.path.relpath(str(fspath), os.getcwd()).replace(os.sep, "/")
    return rel if rel.endswith(".py") else None


def _mark(rel, outcome):
    if not rel:
        return
    rank = _RANK.get(outcome, _RANK["failed"])
    cur = _status.get(rel)
    if cur is None or rank > _RANK[cur]:
        _status[rel] = outcome if outcome in _RANK else "failed"


def _cfg_path():
    cfg = os.environ.get("RSNUMPY_TEST_CONFIG")
    return os.path.abspath(cfg) if cfg else None


def _read_entries(cfg):
    entries = {}
    if os.path.exists(cfg):
        with open(cfg, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                if len(parts) == 2 and parts[1] in ("0", "1"):
                    entries[parts[0]] = parts[1]
    return entries


def _write_all(cfg, entries):
    tmp = cfg + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(_HEADER)
        for rel in sorted(entries):
            f.write(f"{rel} {entries[rel]}\n")
    os.replace(tmp, cfg)


def _flush_file(rel):
    """立即把单个文件标志写盘（原子替换；中途退出也安全）。"""
    if not _is_controller() or rel not in _status:
        return
    cfg = _cfg_path()
    if not cfg:
        return
    entries = _read_entries(cfg)
    entries[rel] = "1" if _status[rel] == "passed" else "0"
    _write_all(cfg, entries)


def _flush_all():
    if not _is_controller():
        return
    cfg = _cfg_path()
    if not cfg:
        return
    entries = _read_entries(cfg)
    for rel, outcome in _status.items():
        entries[rel] = "1" if outcome == "passed" else "0"
    _write_all(cfg, entries)


def pytest_configure(config):
    global _config
    _config = config
    if hasattr(config, "workerinput"):
        return
    # FULL 模式：开场先把全部候选文件置 0 落盘——让"边测边更新"的清单从头完整，
    # 且中途退出后未跑的文件也是明确的 0（而不是旧文件里的过期标志）。
    if os.environ.get("RSNUMPY_TEST_CONFIG_FULL") != "1":
        return
    cfg = _cfg_path()
    if not cfg:
        return
    entries = {}
    for root, dirs, files in os.walk("tests"):
        dirs[:] = [d for d in dirs if d != "__pycache__"]
        for fn in files:
            if fn.startswith("test_") and fn.endswith(".py"):
                rel = os.path.join(root, fn).replace(os.sep, "/")
                entries[rel] = "0"
    _write_all(cfg, entries)


def pytest_collection_modifyitems(config, items):
    # xdist：完整收集在 controller 上进行，这里统计每个文件的测试项数。
    if hasattr(config, "workerinput"):
        return
    for item in items:
        rel = _rel(item.location[0])
        if rel:
            _pending[rel] = _pending.get(rel, 0) + 1


def pytest_runtest_logreport(report):
    fspath = getattr(report, "fspath", None)
    if fspath is None:
        loc = getattr(report, "location", None)
        fspath = loc[0] if loc else None
    rel = _rel(fspath)
    _mark(rel, report.outcome)
    if not _is_controller() or not rel or rel not in _pending:
        return
    # 同一 item 的 setup/call/teardown 多份报告按 nodeid 去重；
    # 该文件所有测试项都出过报告 => 文件跑完 => 立即落盘。
    _seen.setdefault(rel, set()).add(report.nodeid)
    if len(_seen[rel]) >= _pending[rel]:
        _flush_file(rel)


def pytest_collectreport(report):
    # collection error -> the file never produced tests; force it to 0 并立即落盘
    if report.outcome != "passed":
        rel = _rel(getattr(report, "fspath", None))
        _mark(rel, "failed")
        if _is_controller() and rel is not None and _pending.get(rel, 0) == 0:
            _flush_file(rel)


def pytest_sessionfinish(session, exitstatus):
    # xdist workers also fire this hook; only the controller owns the merged
    # report state (worker processes have a `workerinput` attribute)。
    # 兜底：把本轮全部结果再合并写一次（增量落盘已覆盖大多数情况）。
    if hasattr(session.config, "workerinput"):
        return
    if not os.environ.get("RSNUMPY_TEST_CONFIG"):
        return
    _flush_all()
