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

# ========== 同步版本到 python/rsplotlib/__init__.py ==========
sed -i.bak "s/^__version__ = \".*\"/__version__ = \"$VERSION\"/" python/rsplotlib/__init__.py 2>/dev/null || true
rm -f python/rsplotlib/__init__.py.bak
echo "  -> python/rsplotlib/__init__.py __version__ updated to $VERSION"

# ========== 检查 cargo 是否可用（maturin 构建需要） ==========
if ! command -v cargo >/dev/null 2>&1; then
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
    if command -v rustup >/dev/null 2>&1; then
      # 幂等：已安装则为空操作；离线且缺失时失败被忽略，交由后续 clippy 步骤明确报错。
      rustup toolchain install "$RUST_CHANNEL" >/dev/null 2>&1 || true
      rustup component add clippy --toolchain "$RUST_CHANNEL" >/dev/null 2>&1 || true
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
if ! command -v "$PYTHON_EXEC" >/dev/null 2>&1 && [[ ! -x "$PYTHON_EXEC" && ! -f "$PYTHON_EXEC" ]]; then
  echo "Error: Python executable '$PYTHON_EXEC' not found or not executable." >&2
  exit 1
fi

# Prefer a maturin on PATH, otherwise try running maturin via the chosen Python
if command -v maturin >/dev/null 2>&1; then
  MATURIN_MODE="path"
elif "$PYTHON_EXEC" -c "import maturin" >/dev/null 2>&1; then
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

BUILD_ARGS=()
if $RELEASE; then BUILD_ARGS+=(--release); else BUILD_ARGS+=(--debug); fi

mkdir -p "$OUT_DIR"
echo "Building wheel into $OUT_DIR using $PYTHON_EXEC (release=$RELEASE)"
if [[ "$MATURIN_MODE" == "path" ]]; then
  maturin build "${BUILD_ARGS[@]}" -o "$OUT_DIR" -i "$PYTHON_EXEC"
else
  # run maturin as a module under the chosen Python
  "$PYTHON_EXEC" -m maturin build "${BUILD_ARGS[@]}" -o "$OUT_DIR" -i "$PYTHON_EXEC"
fi

# Locate the built wheel and install it into the local venv
WHEEL=$(ls -t "$OUT_DIR"/*.whl 2>/dev/null | head -n1 || true)
if [[ -n "$WHEEL" && -f "$WHEEL" ]]; then
  echo "Wheel built: $WHEEL"
  if [[ -f "$VENV_PY" ]]; then
    echo "Installing into .venv ..."
    # --no-deps：本地 wheel 已含全部绑定代码，无需重新解析依赖。
    # 优先用 uv（显式 --python 指向本 venv，避免依赖是否已 activate）；无 uv 时回退到 pip。
    if command -v uv >/dev/null 2>&1; then
      uv pip install --python "$VENV_PY" --reinstall --no-deps "$WHEEL"
    else
      "$VENV_PY" -m pip install --force-reinstall --no-deps "$WHEEL"
    fi
    echo "Installed into .venv successfully."
  else
    echo "Warning: .venv not found; skip install. Wheels available in $OUT_DIR" >&2
  fi
else
  echo "Warning: no .whl file found in $OUT_DIR; skip install." >&2
fi

echo "Done."