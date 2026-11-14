#!/usr/bin/env bash
set -euo pipefail

# Build Rust lib into Python wheel using maturin
# Usage: ./build_wheel.sh [--release|--debug] [--out-dir DIR] [--python PYTHON_EXEC] [--test all|debug]
#
# 跨平台：macOS / Linux 直接运行；Windows 请在 Git Bash 或 WSL 中运行
# （venv 布局会自动切换为 Scripts/python.exe）。
#
# 测试参数 --test（不给 = 快测：只跑核心门禁 + 冒烟测试，总时长 2 分钟内）：
#   debug 只跑 test_config.txt 里标志为 0 的文件（上次没全过的）
#   all   全量跑 ./tests 并重建 tests/test_config.txt
# 每个文件一个独立 pytest 子进程（文件内 xdist 并行）：崩溃只损失该文件，
# 跑完一个文件立即写回 test_config.txt（边测边更新）：
#   <测试文件> <标志 0/1> <passed>/<failed>/<skipped>（进程崩溃标 crash）

# ========== 从 Cargo.toml 读取元信息 ==========
NAME=$(grep -m1 '^name = ' Cargo.toml | sed 's/name = "\(.*\)"/\1/')
VERSION=$(grep -m1 '^version = ' Cargo.toml | sed 's/version = "\(.*\)"/\1/')

echo "Project: $NAME, Version: $VERSION"

# ========== 同步版本到 pyproject.toml ==========
sed -i.bak "s/^version = \".*\"/version = \"$VERSION\"/" pyproject.toml
rm -f pyproject.toml.bak
echo "  -> pyproject.toml version updated to $VERSION"

# ========== 同步版本到 python/rsnumpy/__init__.py ==========
sed -i.bak "s/^__version__ = \".*\"/__version__ = \"$VERSION\"/" python/rsnumpy/__init__.py 2>/dev/null || true
rm -f python/rsnumpy/__init__.py.bak
echo "  -> python/rsnumpy/__init__.py __version__ updated to $VERSION"

# ========== 检查 cargo 是否可用（maturin 构建需要） ==========
if ! command -v cargo; then
  echo "Error: 'cargo' not found in PATH." >&2
  echo "Please install Rust and Cargo from https://rustup.rs/" >&2
  echo "After installation, run: source \$HOME/.cargo/env" >&2
  exit 1
fi

# ========== 依据 rust-toolchain.toml 准备工具链与 clippy 组件 ==========
# 项目用 rust-toolchain.toml 固定了 Rust channel；rustup 会自动据此选择工具链，
# 但构建前的 clippy 检查依赖 clippy 组件，全新环境未必已安装，这里显式补齐。
if [[ -f rust-toolchain.toml ]]; then
  RUST_CHANNEL=$(grep -m1 '^channel *= *' rust-toolchain.toml | sed -E 's/.*"([^"]+)".*/\1/')
  if [[ -n "${RUST_CHANNEL:-}" ]]; then
    echo "Pinned Rust toolchain (rust-toolchain.toml): $RUST_CHANNEL"
    if command -v rustup; then
      # 幂等：已安装则为空操作；离线且缺失时失败被忽略，交由后续 clippy 步骤明确报错。
      rustup toolchain install "$RUST_CHANNEL" || true
      rustup component add clippy --toolchain "$RUST_CHANNEL" || true
    else
      echo "  -> rustup not found; using the cargo/clippy already on PATH." >&2
    fi
  fi
fi

RELEASE=true
OUT_DIR="wheelhouse"
# 无 --test 参数：快测模式（核心门禁 + 冒烟测试，2 分钟内）；
# --test debug|all：额外跑全量 compat 报告（见下方测试阶段）。
TEST_MODE="off"

# 如果 wheelhouse 目录存在, 则删除
if [[ -d "$OUT_DIR" ]]; then
  rm -rf "$OUT_DIR"
fi

# Default python executable; may be overridden by --python or positional arg
PYTHON_EXEC="python"
# Flag set when the user explicitly provided a Python executable
PYTHON_EXEC_SET=""

# ========== 平台探测：Windows(Git Bash/MSYS/Cygwin) 的 venv 是 Scripts/ 且解释器带 .exe ==========
case "$(uname -s 2>/dev/null || echo unknown)" in
  MINGW*|MSYS*|CYGWIN*) VENV_BIN=".venv/Scripts"; VENV_PY=".venv/Scripts/python.exe" ;;
  *)
    # 不同机器上本地 venv 目录名不同（.venv 或 venv）：存在哪个就用哪个。
    if [[ -d "venv/bin" && ! -d ".venv/bin" ]]; then
      VENV_BIN="venv/bin"; VENV_PY="venv/bin/python"
    else
      VENV_BIN=".venv/bin"; VENV_PY=".venv/bin/python"
    fi
    ;;
esac

# 若存在本地 venv 则激活（让其中的 maturin 进入 PATH）；不存在也不报错。
if [[ -f "$VENV_BIN/activate" ]]; then
  # shellcheck source=/dev/null
  source "$VENV_BIN/activate"
fi
while [[ $# -gt 0 ]]; do
  case $1 in
    --release) RELEASE=true; shift ;;
    --debug) RELEASE=false; shift ;;
    --out-dir) OUT_DIR="$2"; shift 2 ;;
    --python) PYTHON_EXEC="$2"; PYTHON_EXEC_SET=1; shift 2 ;;
    --test)
      case "$2" in
        all|debug) TEST_MODE="$2" ;;
        *) echo "Error: --test accepts 'all' or 'debug' (got '$2')." >&2; exit 1 ;;
      esac
      shift 2 ;;
    --test=*)
      case "${1#*=}" in
        all|debug) TEST_MODE="${1#*=}" ;;
        *) echo "Error: --test accepts 'all' or 'debug' (got '${1#*=}')." >&2; exit 1 ;;
      esac
      shift ;;
    *) PYTHON_EXEC="$1"; PYTHON_EXEC_SET=1; shift ;;
  esac
done

# If a local .venv exists and user didn't explicitly set Python, prefer it
if [[ -f "$VENV_PY" && -z "$PYTHON_EXEC_SET" ]]; then
  PYTHON_EXEC="$VENV_PY"
fi

# Ensure chosen Python executable exists or is runnable
if ! command -v "$PYTHON_EXEC" && [[ ! -x "$PYTHON_EXEC" && ! -f "$PYTHON_EXEC" ]]; then
  echo "Error: Python executable '$PYTHON_EXEC' not found or not executable." >&2
  exit 1
fi

# Prefer a maturin on PATH, otherwise try running maturin via the chosen Python
if command -v maturin; then
  MATURIN_MODE="path"
elif "$PYTHON_EXEC" -c "import maturin"; then
  MATURIN_MODE="python-module"
else
  echo "Error: maturin not found. Install it in your chosen Python (e.g. '$PYTHON_EXEC -m pip install maturin')." >&2
  exit 1
fi

# ========== 构建前的 Fmt 静态检查（-check：任何错误都当作错误） ==========
echo "Running fmt checks (cargo fmt --all -- --check) ..."
if ! cargo fmt --all -- --check; then
  echo "Error: fmt checks failed. Fix the warnings above before building." >&2
  exit 1
fi
echo "  -> fmt checks passed."

# ========== 构建前的 Clippy 静态检查（-D warnings：任何告警都当作错误） ==========
echo "Running clippy checks (cargo clippy --all-targets -- -D warnings) ..."
if ! cargo clippy --all-targets -- -D warnings; then
  echo "Error: clippy checks failed. Fix the warnings above before building." >&2
  exit 1
fi
echo "  -> clippy checks passed."

# maturin 无 --debug 标志：默认即 debug，--release 才是优化构建。用字符串（非数组）
# 规避 bash 3.2 + set -u 下空数组 "${BUILD_ARGS[@]}" 的 unbound variable 问题。
BUILD_ARGS=""
if $RELEASE; then BUILD_ARGS="--release"; fi

mkdir -p "$OUT_DIR"
echo "Building wheel into $OUT_DIR using $PYTHON_EXEC (release=$RELEASE)"
if [[ "$MATURIN_MODE" == "path" ]]; then
  maturin build $BUILD_ARGS -o "$OUT_DIR" -i "$PYTHON_EXEC"
else
  # run maturin as a module under the chosen Python
  "$PYTHON_EXEC" -m maturin build $BUILD_ARGS -o "$OUT_DIR" -i "$PYTHON_EXEC"
fi

# Locate the built wheel and install it into the local venv
WHEEL=$(ls -t "$OUT_DIR"/*.whl 2>/dev/null | head -n1 || true)
INSTALLED=0
if [[ -n "$WHEEL" && -f "$WHEEL" ]]; then
  echo "Wheel built: $WHEEL"
  if [[ -f "$VENV_PY" ]]; then
    echo "Installing into .venv ..."
    # --no-deps：本地 wheel 已含全部绑定代码，无需重新解析依赖。
    # 优先用 uv（显式 --python 指向本 venv，避免依赖是否已 activate）；无 uv 时回退到 pip。
    if command -v uv; then
      uv pip install --python "$VENV_PY" --reinstall --no-deps "$WHEEL"
    else
      "$VENV_PY" -m pip install --force-reinstall --no-deps "$WHEEL"
    fi
    echo "Installed into .venv successfully."
    INSTALLED=1
  else
    echo "Warning: .venv not found; skip install. Wheels available in $OUT_DIR" >&2
  fi
else
  echo "Warning: no .whl file found in $OUT_DIR; skip install." >&2
fi

# ========== 构建后核心测试门禁（对本次构建的 wheel 把关） ==========
# 每个成功安装的构建都必须通过 tests/ 下的核心 pytest 套件：buffer 协议、
# 逐元素算术、构造、索引、归约、比较，并用 xfail 钉住已知缺口（修好后自动翻绿）。
# 显式枚举核心测试文件（不跑 `pytest tests/` 整目录）——因为 tests/ 其余目录是
# 「NumPy 参照基准」harness，单独作为非门禁报告跑（见下），不并入核心门禁。
# SKIP_TESTS=1 可跳过（例如只要快速出 release wheel、不做验证时）。
CORE_TEST_FILES=(
  tests/test_buffer.py
  tests/test_arithmetic.py
  tests/test_creation.py
  tests/test_indexing.py
  tests/test_reduction.py
  tests/test_comparison.py
  tests/test_known_gaps.py
)
if [[ "$INSTALLED" -eq 1 && "${SKIP_TESTS:-0}" != "1" ]]; then
  echo "Running core test suite against the installed wheel ..."
  if command -v uv; then
    uv pip install --python "$VENV_PY" --quiet pytest
    # xdist 是可选的：装不上就回退单进程（构建不应因可选插件而失败）。
    uv pip install --python "$VENV_PY" --quiet pytest-xdist || true
  else
    "$VENV_PY" -m pip install --quiet pytest
    "$VENV_PY" -m pip install --quiet pytest-xdist || true
  fi
  if "$VENV_PY" -m pytest "${CORE_TEST_FILES[@]}" -q; then
    echo "  -> core test suite passed."
  else
    echo "Error: core test suite failed — the built wheel did not pass its tests." >&2
    exit 1
  fi

  # ========== NumPy 参照基准：验证 rsnumpy 兼容性（非门禁报告） ==========
  # 把 ./tests 下的全部测试（从 NumPy 官方测试树拷贝、导入已改为 rsnumpy）作为
  # 参照基准跑一遍——被测对象是 rsnumpy，这些测试只是 ground truth。
  # failed / error = rsnumpy 待修的 bug 或待补的 API，故只报告、不门禁；
  # 核心门禁仍是上面的 7 个核心文件。
  #   * --continue-on-collection-errors：rsnumpy 尚未实现的子模块（f2py、
  #     polynomial 各子模块、random.Generator 等）会让整个测试文件收集失败，
  #     不能因此中断全量报告。
  #   * RSNUMPY_COMPAT_SKIP_CRASHES=1：跳过已知会让 pytest 进程直接崩溃的用例；
  #     未在清单中的 crash 由 xdist worker 隔离（只损失该 worker 上的用例）。
  #
  # 测试模式（--test all|debug；本段仅在给出 --test 时到达）：
  #   debug —— 只跑 test_config.txt 中标志为 0 的文件（上次没全过的）。
  #            修 bug 时的快速回归回路。
  #   all   —— 全量跑 ./tests（核心 7 文件已由门禁跑过，候选列表里排除去重）：
  #            开场把全部候选文件置 0，随后每跑完一个文件翻成实际结果，重建
  #            tests/test_config.txt（全通过 → 1，存在 failed/skipped/error → 0）。
  #
  # 执行架构：每个测试文件一个独立 pytest 子进程（文件内 xdist 并行，默认 4 worker）：
  #   * 崩溃隔离 —— 用例把 rsnumpy C 扩展打崩（Rust panic -> abort / segfault）
  #     只损失该文件的进程，不再杀掉整份报告（旧的单个 xdist controller 在 worker
  #     或 controller 自身崩溃时会让整个运行被 abort 截断）。
  #   * 边测边更新 —— 每个文件跑完立即把计数（passed/failed/skipped）原子写回
  #     test_config.txt：跑的过程中就能实时看到进度，中途退出清单也已是最新。
  #   * 配置格式：<测试文件> <标志> <passed>/<failed>/<skipped>（进程崩溃时计数列为 crash）。
  # 仅显式 --test debug|all 时跑全量 compat 报告（默认 off：无参 = 快测，
  # 只有上方核心门禁 + 下方冒烟测试，总时长 2 分钟内）。
  if [[ "$TEST_MODE" != "off" ]]; then
  echo "Running rsnumpy-compatibility check over ./tests (non-gating report, mode=$TEST_MODE) ..."
  # 文件内并行度默认 4（COMPAT_WORKERS 可覆盖，设为 0 强制单进程）——按文件起子进程
  # 后总并行度仍是 4：每个用例都是数组的重计算，并行度过高机器会满载发烫。
  COMPAT_WORKERS_N="${COMPAT_WORKERS:-4}"
  if ! "$VENV_PY" -c "import xdist" >/dev/null 2>&1; then
    echo "  -> pytest-xdist unavailable; single-process per file (a crash may truncate that file's report)."
    COMPAT_WORKERS_N=0
  fi
  mkdir -p debug_test
  CONFIG_FILE="tests/test_config.txt"

  CFG_HEADER='# rsnumpy 测试配置: <测试文件> <标志> <passed>/<failed>/<skipped>
#   1 = 上次全部通过   0 = 存在 failed/skipped/error（或尚未跑通）
# 由 build_wheel.sh 的测试阶段自动维护：每个文件一个 pytest 子进程，跑完立即
# 写回结果（边测边更新；崩溃只影响该文件，计数列标 crash）。
# --test debug 只跑标志为 0 的文件；--test all 全量重建。
'

  # 全部候选文件（排除 __pycache__ 与上面刚门禁通过的核心 7 文件，避免跑两遍）。
  CANDIDATES=()
  while IFS= read -r _cf; do
    CANDIDATES+=("$_cf")
  done < <(find tests -name 'test_*.py' -not -path '*/__pycache__/*' | sort)
  RUN_CANDIDATES=()
  for _cf in "${CANDIDATES[@]}"; do
    _skip=0
    for _core in "${CORE_TEST_FILES[@]}"; do
      [[ "$_cf" == "$_core" ]] && _skip=1
    done
    [[ $_skip -eq 0 ]] && RUN_CANDIDATES+=("$_cf")
  done

  # 把一行结果原子写进 test_config.txt：保留头部注释与既有行，
  # 替换（或追加）该文件一行，临时文件 + os.replace。
  _update_cfg() {
    "$VENV_PY" - "$CONFIG_FILE" "$1" "$2" <<'PYEOF'
import os, sys
cfg, f, line = sys.argv[1], sys.argv[2], sys.argv[3]
entries = {}
if os.path.exists(cfg):
    with open(cfg, encoding="utf-8") as fh:
        for ln in fh:
            ln = ln.rstrip("\n")
            if not ln or ln.startswith("#"):
                continue
            key = ln.split()[0]
            if key not in entries:
                entries[key] = ln
entries[f] = line
tmp = cfg + ".tmp"
with open(tmp, "w", encoding="utf-8") as fh:
    if os.path.exists(cfg):
        with open(cfg, encoding="utf-8") as src:
            for ln in src:
                if ln.startswith("#"):
                    fh.write(ln)
                else:
                    break
    for k in sorted(entries):
        fh.write(entries[k] + "\n")
os.replace(tmp, cfg)
PYEOF
  }

  TARGETS=()
  if [[ ! -f "$CONFIG_FILE" || "$TEST_MODE" == "all" ]]; then
    # all 模式（含无配置文件时的首次引导）：先把全部候选置 0 落盘（未跑 = 明确的 0），
    # 随后全量跑 RUN_CANDIDATES（核心 7 文件已门禁通过，不重复跑）。
    {
      printf '%s' "$CFG_HEADER"
      for _cf in "${CANDIDATES[@]}"; do
        printf '%s 0 0/0/0\n' "$_cf"
      done
    } > "$CONFIG_FILE"
    TARGETS=("${RUN_CANDIDATES[@]}")
    echo "  -> ${#TARGETS[@]} candidate file(s); full run."
  else
    # debug 模式：从配置里挑出标志为 0 的文件
    while read -r _line_f _line_flag _line_rest; do
      [[ "$_line_f" == \#* || -z "${_line_f:-}" ]] && continue
      if [[ "$_line_flag" == "0" && -f "$_line_f" ]]; then
        TARGETS+=("$_line_f")
      fi
    done < "$CONFIG_FILE"
    if [[ ${#TARGETS[@]} -eq 0 ]]; then
      echo "  -> every test file is green in $CONFIG_FILE; nothing to debug."
      echo "     Run './build_wheel.sh --test all' to refresh the flags."
    else
      echo "  -> ${#TARGETS[@]} file(s) flagged 0 in $CONFIG_FILE:"
      printf '     %s\n' "${TARGETS[@]}"
    fi
  fi

  COMPAT_LOG="${COMPAT_LOG:-tests/${TEST_MODE}_report.log}"
  COMPAT_RC=0
  if [[ ${#TARGETS[@]} -gt 0 ]]; then
    RUN_DIR=$(mktemp -d "${TMPDIR:-/tmp}/rsnumpy_compat.XXXXXX")
    : > "$COMPAT_LOG"
    _N=${#TARGETS[@]}
    _idx=0
    for f in "${TARGETS[@]}"; do
      _idx=$((_idx + 1))
      if [[ "$COMPAT_WORKERS_N" -gt 0 ]]; then
        JFLAGS="-n $COMPAT_WORKERS_N"
      else
        JFLAGS=""
      fi
      echo "  [$_idx/$_N] $f"
      set +e
      # 边测边显示：输出流式打印到终端（tee 同时追加完整日志），另存本文件输出供解析。
      env RSNUMPY_COMPAT_SKIP_CRASHES=1 PYTHONUNBUFFERED=1 \
        "$VENV_PY" -m pytest "$f" \
        -q -p no:cacheprovider --continue-on-collection-errors $JFLAGS \
        2>&1 | tee -a "$COMPAT_LOG" > "$RUN_DIR/out.log"
      rc=${PIPESTATUS[0]}
      set -e
      if [[ $rc -ge 128 ]] || grep -q 'Fatal Python error' "$RUN_DIR/out.log"; then
        # 进程/worker 被信号杀死或致命错误（Rust panic -> abort / segfault 等）：
        # 只损失该文件的结果（xdist 下 worker 阵亡时 controller 可能以 rc=5 退出）。
        _status="CRASHED (rc=$rc); this file's result is lost"
        _update_cfg "$f" "$f 0 crash"
      else
        # 解析末行汇总 "N failed, M passed, K skipped ... in Xs ="（行尾 = 前有空格）
        _counts=$("$VENV_PY" - "$RUN_DIR/out.log" <<'PYEOF'
import re, sys
text = open(sys.argv[1], encoding="utf-8", errors="replace").read()
summ = None
for line in text.splitlines():
    if re.search(r" in \d+(?:\.\d+)?s", line) and re.search(
        r"\d+ (passed|failed|skipped|error)", line
    ):
        summ = line
p = fa = s = 0
if summ:
    for m in re.finditer(r"(\d+) (passed|failed|skipped|error)", summ):
        n, k = int(m.group(1)), m.group(2)
        if k == "passed":
            p = n
        elif k == "skipped":
            s = n
        else:
            fa += n
print(f"{p}/{fa}/{s}")
PYEOF
)
        if [[ "$_counts" == "0/0/0" && $rc -ne 0 ]]; then
          # 没有可解析的汇总行（collection error / 早期退出等）
          _status="no summary (rc=$rc)"
          _update_cfg "$f" "$f 0 0/0/0"
        else
          flag=0
          if [[ $_counts == */0/0 && ${_counts%%/*} -gt 0 ]]; then
            flag=1
          fi
          _status="$_counts (rc=$rc)"
          _update_cfg "$f" "$f $flag $_counts"
        fi
      fi
      echo "      -> $_status"
      [[ $rc -ne 0 ]] && COMPAT_RC=1
    done
    rm -rf "$RUN_DIR"
    echo
    echo "  -> full log: $COMPAT_LOG"
    echo "  -> rsnumpy compatibility report above (failed/error = rsnumpy bugs/gaps vs the numpy reference; non-gating, rc=$COMPAT_RC)."

    # all 模式：核心 7 文件不进本轮运行，但门禁已确认通过，配置里直接置 1
    # （否则 debug 模式会重复跑它们）。
    if [[ "$TEST_MODE" == "all" && -f "$CONFIG_FILE" ]]; then
      for _f in "${CORE_TEST_FILES[@]}"; do
        sed -i.bak -E "s|^$_f 0 |$_f 1 |" "$CONFIG_FILE"
        rm -f "$CONFIG_FILE.bak"
      done
    fi
    if [[ -f "$CONFIG_FILE" ]]; then
      _n1=$(awk '$1 !~ /^#/ && $2 == "1" {n++} END {print n+0}' "$CONFIG_FILE")
      _n0=$(awk '$1 !~ /^#/ && $2 == "0" {n++} END {print n+0}' "$CONFIG_FILE")
      echo "  -> $CONFIG_FILE updated: $_n1 file(s) green (1), $_n0 file(s) flagged 0."
    fi
  fi
  else
    echo "Skipping the full rsnumpy-compatibility report (no --test given)."
    echo "      './build_wheel.sh --test debug' 只重跑未通过文件；'--test all' 全量并重建清单。"
  fi

  # ========== 冒烟测试：run_test.py / run_test2.py（非门禁报告） ==========
  # 独立顺序执行脚本（非 pytest），验证 rsnumpy 核心功能无异常。
  # failed = rsnumpy 存在未修复的 bug，故只报告、不门禁。
  SMOKE_TESTS=(
    tests/run_test.py
    tests/run_test2.py
  )
  echo "Running rsnumpy smoke tests (non-gating report) ..."
  mkdir -p debug_test
  for SMOKE_TEST in "${SMOKE_TESTS[@]}"; do
    set +e
    SMOKE_OUT=$("$VENV_PY" "$SMOKE_TEST" 2>&1)
    SMOKE_RC=$?
    set -e
    if [[ $SMOKE_RC -eq 0 ]]; then
      echo "  -> $SMOKE_TEST passed."
    else
      echo "  -> $SMOKE_TEST FAILED (rc=$SMOKE_RC)"
      echo "$SMOKE_OUT" | tail -20
    fi
  done
fi

echo "Done."