#!/usr/bin/env bash
set -euo pipefail

# Build Rust lib into Python wheel using maturin
# Usage: ./build_wheel.sh [--release|--debug] [--out-dir DIR] [--python PYTHON_EXEC]
#
# 跨平台：macOS / Linux 直接运行；Windows 请在 Git Bash 或 WSL 中运行
# （venv 布局会自动切换为 Scripts/python.exe）。

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
  *)                    VENV_BIN=".venv/bin";     VENV_PY=".venv/bin/python" ;;
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
# 显式枚举核心测试文件（不跑 `pytest tests/` 整目录）——因为 tests/numpy/ 是
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
  else
    "$VENV_PY" -m pip install --quiet pytest
  fi
  if "$VENV_PY" -m pytest "${CORE_TEST_FILES[@]}" -q; then
    echo "  -> core test suite passed."
  else
    echo "Error: core test suite failed — the built wheel did not pass its tests." >&2
    exit 1
  fi

  # ========== NumPy 参照基准：验证 rsnumpy 兼容性（非门禁报告） ==========
  # 用 NumPy 官方测试（tests/test_rsnumpy_*.py，~3400 用例，numpy 改名为 rsnumpy）
  # 作为参照基准验证 rsnumpy 的 NumPy 兼容性——被测对象是 rsnumpy，这些测试只是
  # ground truth。failed = rsnumpy 待修的 bug 或待补的 API（见下），故只报告数字、
  # 不门禁；核心门禁是上面的 7 个核心文件。构建模式跳过已知 C 扩展崩溃测试
  # （RSNUMPY_COMPAT_SKIP_CRASHES=1）避免单进程 pytest 被 crash 中断；完整 crash
  # 清单由 ./tests/run_compat.py 产出。
  REFERENCE_TESTS=(
    tests/test_rsnumpy_array_coercion.py
    tests/test_rsnumpy_indexing.py
    tests/test_rsnumpy_numeric.py
    tests/test_rsnumpy_scalarmath.py
    tests/test_rsnumpy_defchararray.py
    tests/test_rsnumpy_function_base.py
    tests/test_rsnumpy_numerictypes.py
    tests/test_rsnumpy_records.py
    tests/test_rsnumpy_shape_base.py
  )
  echo "Running rsnumpy -compatibility check (non-gating report) ..."
  set +e
  COMPAT_OUT=$(RSNUMPY_COMPAT_SKIP_CRASHES=1 "$VENV_PY" -m pytest "${REFERENCE_TESTS[@]}" -q -p no:cacheprovider 2>&1)
  COMPAT_RC=$?
  set -e
  echo "$COMPAT_OUT" | tail -15
  echo "  -> rsnumpy compatibility report above (failed = rsnumpy bugs/gaps vs the numpy reference; non-gating, rc=$COMPAT_RC)."

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