# rsnumpy

A high-performance, NumPy-compatible array library where every compute-heavy
operator is implemented in Rust and dispatched through PyO3 with the GIL
released. The Python layer is a thin parameter-forwarding shell, so
multi-threaded Python code actually runs in parallel.

---

## Highlights

- **NumPy-compatible API.** Drop-in for typical NumPy use cases. Covers array
  creation, math, statistics, `linalg`, `fft`, `random`, polynomial, file I/O
  (`.npy` / `.npz` / text), `datetime64` / `timedelta64`, masked and
  structured arrays. Aligned with the NumPy 2.5 behavior.
- **Rust-powered performance.** On macOS (Apple Silicon) for `n = 1,000,000`
  `float64`:
  - `sin` / `exp` — **5.2×–5.4×** faster than NumPy
  - `sort` — **~32×** faster
  - `cumsum` — **~1.9×** faster
  - `add` / `mul` / `div` / `sqrt` — **~1.6×–1.7×** faster
  - `matmul` 1024×1024 — on par with NumPy via system Accelerate BLAS
- **GIL-free parallelism.** All heavy operators (`sort`, `cumsum`, `matmul`,
  `inv`, element-wise, reductions) release the GIL via `py.detach` and are
  parallelized in Rust with rayon, behind cost-tiered thresholds so small
  arrays don't pay a thread-scheduling tax.
- **Zero-copy buffer protocol.** Implements PEP 3118 (`__getbuffer__` /
  `__releasebuffer__` in Rust) plus PEP 688 (`__buffer__` for Python 3.12+).
  Downstream libraries like NumPy or `rsplotlib` can read `float64` data
  without copying. Non-`float64` dtypes fall back to `__array_interface__`
  with dtype-precise `bytes`.
- **Variable-length elements.** `VarStringArray` is a faster drop-in for
  variable-length string data that would otherwise degrade to `object`
  dtype. `BigIntArray` uses Python `int` to hold integers that exceed `2^53`
  without losing precision.
- **Large-array tools.** `chunked_apply` for axis-chunked user functions,
  `mmap_array` for reading part of a binary file by `dtype` + `shape` +
  `offset`, `LazyArray` for deferred chained operations, and
  `memory_usage()` for inspecting array footprint.
- **Thread-safety helpers.** `threadsafe_copy` / `threadsafe_view` and a
  `thread_safe=True` switch on `ndarray.copy()` and `np.copy()` to make
  multi-threaded sharing safe without paying for a global lock on the hot
  path.
- **Smart dtype inference.** `array(['hello', 'world'])` infers `U5` instead
  of `object`; mixed-type and integer-overflow paths emit warnings and
  recommend better dtypes or `BigIntArray`.
- **Compact binary & fast import.** Release build uses `lto="fat"`,
  `codegen-units=1`, `panic="abort"`, and `strip="symbols"`, plus a
  `num_cpus`-free dependency tree — `num_core.cpython-*.so` is about **5.1 MB**.
  Circular imports are broken with lazy `_get_deps()`, so the first
  `import rsnumpy` finishes in roughly **7 ms**.
- **Thread-pool control.** `set_num_threads()`, `get_num_threads()`,
  `parallel_context()`, and `set_parallel_thresholds()` /
  `get_parallel_thresholds()` let callers steer the underlying rayon pool
  and per-operator cost thresholds at runtime.

---

## Installation

```bash
pip install rsnumpy
# or
uv pip install rsnumpy
```

Prebuilt wheels are published for **Python 3.10–3.14** on:

- Linux `x86_64` / `aarch64`
- macOS Apple Silicon / Intel
- Windows `x64`

---

## Quick Start

```python
import rsnumpy as np

# Basics
a = np.array([1.0, 2.0, 3.0, 4.0])
b = np.arange(12).reshape(3, 4)
print(np.sin(a).tolist())
print(np.exp(a).tolist())
print(np.sum(a), np.mean(a))

# Linear algebra
A = np.array([[1.0, 2.0], [3.0, 4.0]])
print(np.linalg.inv(A).tolist())
print(np.linalg.det(A))

# FFT
spectrum = np.fft.fft(a)

# Reproducible parallel random sampling
rng = np.random.default_rng(42)
print(rng.random(5).tolist())

# Variable-length strings (faster than object dtype)
vsa = np.VarStringArray(['hello', 'world', 'foo'])
print((vsa + '!').to_list())

# Big integers (no precision loss past 2^53)
big = np.BigIntArray([2**60, 2**70, 2**80])
print(big[0], big.sum())

# Large-array tools
large = np.arange(10_000_000)
np.chunked_apply(np.sum, large, chunk_size=1_000_000)

lazy = np.LazyArray(np.arange(1000.0))
result = lazy.sqrt().multiply(2).add(1).compute()

# Zero-copy interop with NumPy
import numpy as real_numpy
arr = real_numpy.asarray(a)  # no copy, dtype=float64
```

---

## Project Layout

```
rsnumpy/
├── src/                       # Rust source (22,000+ lines)
│   ├── lib.rs                 # core ndarray + PyO3 module
│   ├── indexing.rs            # multi-dim indexing & slicing
│   ├── fft.rs                 # FFT
│   ├── linalg.rs              # linear algebra
│   ├── random.rs              # random sampling
│   ├── threadpool.rs          # rayon pool + parallel thresholds
│   ├── arithmetic/            # + - * / % **
│   ├── bitwise/               # & | ^ ~
│   ├── buffer/                # bytes ⇄ array interop
│   ├── creation/              # zeros, ones, arange, linspace, ...
│   ├── formatting/            # repr / str formatting
│   ├── io/                    # .npy / .npz
│   ├── logic/                 # comparison & logical ops
│   ├── manipulation/          # reshape, transpose, concatenate, ...
│   ├── mathematics/           # sin, cos, exp, log, ...
│   ├── poly/                  # polynomial primitives
│   ├── searching/             # argmax, argmin, searchsorted, ...
│   ├── setops/                # unique, intersect1d, ...
│   ├── sorting/               # sort, argsort, partition, ...
│   └── statistics/            # sum, mean, std, var, ...
├── python/rsnumpy/            # thin Python wrappers (26,000+ lines)
│   ├── __init__.py
│   ├── array_methods.py
│   ├── array_ops.py
│   ├── _extra.py              # ~140 NumPy-compatible helpers
│   ├── _dtypes.py
│   ├── char.py, io.py, ma.py, rec.py
│   ├── math_functions.py, statistics.py
│   ├── linalg/, polynomial/, random/, fft/, lib/, typing/
├── test/                      # regression & compat tests
│   ├── run_test.py, run_test2.py
│   └── test_rf/               # networkx / scipy compatibility
├── debug/                     # benchmarks & debug scripts
├── Cargo.toml                 # Rust manifest & release profile
├── pyproject.toml             # Python build config (maturin)
├── build_wheel.sh             # one-shot build & install
└── README.md / RELEASE_NOTES.md / PYPI.md
```

---

## Building from Source

```bash
git clone https://github.com/YJ-Niu/rsnumpy.git
cd rsnumpy

# Create a virtualenv (uv recommended)
uv venv .venv --python 3.11
source .venv/bin/activate
uv pip install maturin

# Build & install (release)
bash build_wheel.sh
# bash build_wheel.sh --debug   # debug build with symbols
```

The script detects the active Python, runs `maturin build --release`,
places the wheel under `wheelhouse/`, and installs it with
`uv pip install --no-deps`. For a development build that picks up source
edits immediately:

```bash
maturin develop --release
```

---

## Dependencies

`rsnumpy` is **self-contained** at the Python level — there are no runtime
Python dependencies (it does not depend on NumPy). All computation lives in
the Rust extension.

Build-time only:

- `maturin >= 1.13, < 2.0`
- A Rust toolchain ≥ 1.75 (pinned via `rust-toolchain.toml`)

Runtime Rust crates bundled into the wheel:

- `pyo3 0.29` (extension-module)
- `ndarray 0.17.2` (with `rayon` and `matrixmultiply-threading`; on macOS
  also `blas` to dispatch to Accelerate)
- `rayon 1.12`
- `rand 0.10.2` / `rand_distr 0.6.0`
- `num-traits 0.2.19` / `num-complex 0.4.6`
- `rustfft 6.4.1`
- `zip 8.6.0` (with only `deflate-flate2-zlib-rs` enabled, for `.npz`
  read/write)
- `thiserror 2.0.19`
- `blas-src 0.14.0` (macOS, with `accelerate`)

`num_cpus` was **removed** in v1.2.3 — the project now uses
`std::thread::available_parallelism()` to keep the binary small.

---

## Version

**v1.2.3** — see [RELEASE_NOTES.md](RELEASE_NOTES.md) for the full changelog
(binary size −25%, import time −130×, dynamic thread pool, variable-length
elements, large-array tools, smart dtype inference, thread-safety helpers,
and several `sort` / `indexing` / `statistics` bug fixes).

### Previous Highlights

- **v1.2.1** — 2×2 batched `linalg.inv` parallel with rayon; 2-port S-parameter
  renormalization ~700× faster; closed-form S-parameter inversion ~8× faster.
- **v1.1.6** — GIL release for compute-heavy operators; cost-tiered parallel
  thresholds; in-place parallel `cumsum` / `cumprod`; macOS Accelerate BLAS
  backend.
- **v1.1.0** — ~140 NumPy-compatible functions added; aligned with NumPy 2.5.
- **v1.0.7** — `__array__` interface for direct matplotlib consumption.
- **v1.0.0** — Initial release with arrays, math, statistics, linalg, FFT,
  random, polynomial, and file I/O.

---

## License

MIT — see [LICENSE](LICENSE).

## Links

- [GitHub repository](https://github.com/YJ-Niu/rsnumpy)
- [README](https://github.com/YJ-Niu/rsnumpy/blob/main/README.md)
- [Release notes](https://github.com/YJ-Niu/rsnumpy/blob/main/RELEASE_NOTES.md)
- [Issue tracker](https://github.com/YJ-Niu/rsnumpy/issues)
