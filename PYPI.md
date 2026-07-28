# rsnumpy

A high-performance NumPy-compatible array library powered by Rust, with zero-copy buffer protocol support for downstream libraries like rsplotlib.

## Key Features

- **NumPy-Compatible API**: Drop-in replacement for most NumPy use cases. Low migration cost.
- **Rust-Powered Performance**: Core computation runs in Rust. Element-wise operations (`sin`/`exp`) up to **5.4x** faster than NumPy. `sort` up to **32x** faster. 2×2 batch matrix inversion parallelized with rayon.
- **Zero-Copy Buffer Protocol**: Implements PEP 3118 (`__buffer__`/`__release_buffer__`), allowing rsplotlib and other libraries to read float64 array data **without copying**.
- **GIL-Free Parallelism**: Computation-heavy operators (`sort`, `cumsum`, `matmul`, `inv`, element-wise ops) release the GIL for true multi-threaded parallelism.
- **BLAS Acceleration**: `matmul`/`dot` dispatch to system BLAS (Accelerate on macOS), matching NumPy performance.
- **S-Parameter Fast Paths**: 2-port network renormalization (scalar impedance) up to **700x** faster, S-parameter inversion up to **8x** faster.
- **Complete Feature Set**: Arrays, math functions, statistics, linear algebra, FFT, random numbers, polynomials, file I/O (`.npy`, `.npz`, text), datetime64, structured arrays, masked arrays.
- **Cross-Platform**: Linux (x86_64/aarch64), macOS (Apple Silicon/Intel), Windows (x64).

## Installation

```bash
pip install rsnumpy
```

Prebuilt wheels available for Python 3.10-3.14 across Linux, macOS, and Windows.

## Quick Start

```python
import rsnumpy as np

# Basic array operations
a = np.array([1, 2, 3, 4])
b = np.array([5, 6, 7, 8])
print((a + b).tolist())       # [6.0, 8.0, 10.0, 12.0]
print(np.sin(a).tolist())     # [0.8415, 0.9093, 0.1411, -0.7568]

# Zero-copy buffer protocol (for rsplotlib integration)
x = np.arange(0, 10, 0.1)
y = np.sin(x)
# rsplotlib can read x, y directly without copying

# Linear algebra
A = np.array([[1, 2], [3, 4]])
print(np.linalg.inv(A).tolist())
print(np.linalg.eigvals(A).tolist())

# FFT
spectrum = np.fft.fft(np.array([1.0, 0.0, 0.0, 0.0]))

# Random numbers
rng = np.random.default_rng(42)
print(rng.random(5).tolist())
```

### Buffer Protocol Usage

```python
import rsnumpy as np

# float64 arrays: zero-copy memoryview
arr = np.array([1.0, 2.0, 3.0])
mv = memoryview(arr)
print(mv.format)   # 'd'
print(mv.shape)    # (3,)

# numpy interop (zero-copy)
import numpy as np
np_arr = np.asarray(arr)  # dtype=float64, no copy
```

## Performance Highlights

Measured on macOS (Apple Silicon), comparing to NumPy:

| Operation            | vs NumPy                   |
| -------------------- | -------------------------- |
| `sin`/`exp`          | **5.2x–5.4x** faster       |
| `sort`               | **32x** faster             |
| `cumsum`             | **1.9x** faster            |
| `add`/`mul`/`div`    | **1.6x–1.7x** faster       |
| `matmul` (1024×1024) | **1.0x** (Accelerate BLAS) |

## Version v1.2.1

### Improvements

- **Matrix Inversion Acceleration**: 2×2 batch matrix inversion parallelized with rayon in Rust, supporting complex types. Python `linalg.inv` batch branch dispatches to parallel Rust.
- **S-Parameter Fast Paths**: 2-port network `renormalize_s` scalar impedance fast path avoids double matrix inversion (~700x faster). Closed-form S-parameter inversion (~8x faster).
- **Impedance Renormalization**: Refactored 2-port impedance renormalization to support per-frequency independent impedance parameters.
- **Bug Fixes**: Fixed S-parameter matrix dtype inference, suppressed CPW conductor loss low-frequency RuntimeWarning.

### Previous Versions

- v1.2.0: Zero-copy buffer protocol (PEP 3118), code cleanup, `recfunctions` fix, module restructuring.
- v1.1.6: GIL release for compute-heavy operators, cost-tiered parallel thresholds, `cumsum`/`cumprod` in-place scan optimization, macOS Accelerate BLAS backend.
- v1.1.5: String/void dtype support, masked arrays, structured arrays.
- v1.1.0: Added ~140 NumPy-compatible functions, aligned with NumPy 2.5.1.

## Documentation

- [GitHub Repository](https://github.com/YJ-Niu/rsnumpy)
- [README](https://github.com/YJ-Niu/rsnumpy/blob/main/README.md)
- [Release Notes](https://github.com/YJ-Niu/rsnumpy/blob/main/RELEASE_NOTES.md)

## License

MIT License
