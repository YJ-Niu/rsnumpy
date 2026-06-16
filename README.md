# rsnumpy

**A Rust-powered NumPy-compatible array library.**

---

## 中文文档

### 1. 项目简介

**rsnumpy** 是一个由 **Rust** 驱动的高性能多维数组库，提供与 **NumPy** 兼容的 API。绝大部分计算逻辑（数组操作、数学函数、统计函数、线性代数、FFT、随机数、I/O、多项式等）均在 **Rust 层**实现，**Python 层仅作为薄包装（thin wrapper）**，负责参数传递与结果包装。

**优势：**
- 性能：核心计算使用 Rust 实现，接近或超过 NumPy 的速度
- 类型安全：利用 Rust 的强类型系统避免运行时错误
- 内存安全：无数据竞争，无内存泄漏
- 完整 API：覆盖 NumPy 常用功能（数组操作、数学、统计、linalg、FFT、random、I/O、polynomial）

**项目结构：**
```
rsnumpy/
├── src/                       # Rust 源码
│   ├── lib.rs                 # 核心 ndarray 与通用函数
│   ├── fft.rs                 # 快速傅里叶变换
│   ├── linalg.rs              # 线性代数
│   └── random.rs              # 随机数生成
├── rsnumpy/                     # Python 薄包装
│   ├── __init__.py            # 主模块，整合所有 API
│   ├── array_methods.py       # ndarray 对象方法
│   ├── array_ops.py           # 数组操作函数
│   ├── math_functions.py      # 数学函数
│   ├── statistics.py          # 统计函数
│   ├── io.py                  # 文件 I/O
│   ├── linalg/                # 线性代数子模块
│   ├── polynomial/            # 多项式子模块
│   └── random/                # 随机数子模块
├── Cargo.toml                 # Rust 依赖
├── pyproject.toml             # Python 构建配置
├── build_wheel.sh             # 构建脚本
└── README.md                  # 本文件
```

---

### 2. 环境要求

| 工具       | 最低版本       | 说明                    |
|------------|---------------|-------------------------|
| Python     | ≥ 3.8         | 推荐 3.10+              |
| Rust       | ≥ 1.75        | `edition = "2024"`      |
| maturin    | ≥ 1.13, < 2.0 | Rust ↔ Python 绑定      |
| uv（可选） | 最新          | 快速创建 venv 与安装     |

---

### 3. 快速开始

#### 3.1 克隆项目

```bash
git clone <your-repo-url> rsnumpy
cd rsnumpy
```

#### 3.2 创建虚拟环境（推荐使用 uv）

```bash
# 使用 uv 创建 .venv
uv venv .venv --python 3.11
source .venv/bin/activate

# 安装 maturin 和 numpy
uv pip install maturin numpy
```

如果使用标准 `venv`：

```bash
python -m venv .venv
source .venv/bin/activate
pip install maturin numpy
```

#### 3.3 构建并安装

直接运行项目自带的构建脚本：

```bash
bash build_wheel.sh
```

脚本会自动：
1. 检测 `.venv` 中的 Python
2. 调用 `maturin build --release`
3. 生成 `.whl` 到 `wheelhouse/`
4. 使用 `uv pip install --no-deps` 安装到 `.venv`

构建完成后即可使用：

```python
python -c "import rsnumpy; print(rsnumpy.__doc__)"
```

---

### 4. 详细构建方式

#### 4.1 使用构建脚本（推荐）

```bash
# 默认 release 模式，构建到 .venv
bash build_wheel.sh

# 指定 Python 解释器
bash build_wheel.sh --python .venv/bin/python

# 输出到指定目录
bash build_wheel.sh --out-dir build/

# Debug 模式（更快但更慢）
bash build_wheel.sh --debug
```

#### 4.2 手动使用 maturin

```bash
# 开发模式：实时编辑，无需重新构建
maturin develop --release

# 仅构建 wheel
maturin build --release -o wheelhouse/

# 手动安装 wheel
uv pip install --no-deps wheelhouse/rsnumpy-*.whl
```

#### 4.3 清理构建产物

```bash
rm -rf target/ wheelhouse/
```

---

### 5. 使用示例

#### 5.1 基本数组操作

```python
import rsnumpy as np

# 创建数组
a = np.array([1.0, 2.0, 3.0])
b = np.zeros((2, 3))        # 2x3 全零
c = np.ones((2, 2))          # 2x2 全一
d = np.eye(3)                # 3x3 单位矩阵
e = np.arange(0, 10, 2)      # [0, 2, 4, 6, 8]
f = np.linspace(0, 1, 5)     # [0, 0.25, 0.5, 0.75, 1.0]

# 属性
print(a.shape)    # (3,)
print(a.ndim)     # 1
print(a.size)     # 3
```

#### 5.2 数学函数

```python
import rsnumpy as np

a = np.array([0.0, np.pi / 2, np.pi])
print(np.sin(a))         # [0.0, 1.0, 0.0]
print(np.cos(a))         # [1.0, 0.0, -1.0]
print(np.exp(a))         # 指数
print(np.sqrt(a))        # 平方根
print(np.log(a + 1))     # 自然对数
```

#### 5.3 数组操作

```python
import rsnumpy as np

a = np.array([[1, 2, 3], [4, 5, 6]])

# 变形
b = a.reshape((3, 2))
print(b.tolist())   # [[1, 2], [3, 4], [5, 6]]

# 转置
c = a.transpose()
print(c.tolist())   # [[1, 4], [2, 5], [3, 6]]

# 切片（多维 tuple 索引，由 Rust 实现）
print(a[0:2, 1:3].tolist())  # [[2, 3], [5, 6]]
print(a[:, 1].tolist())      # [[2], [5]]

# 拼接
d = np.concatenate([a, a], axis=0)
print(d.shape)   # (4, 3)

e = np.vstack([a, a])   # 垂直堆叠
f = np.hstack([a, a])   # 水平堆叠
```

#### 5.4 统计函数

```python
import rsnumpy as np

a = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

print(np.sum(a))         # 15
print(np.mean(a))        # 3.0
print(np.std(a))         # 标准差
print(np.var(a))         # 方差
print(np.max(a))         # 5
print(np.argmin(a))      # 0
```

#### 5.5 线性代数

```python
import rsnumpy as np

A = np.array([[1.0, 2.0], [3.0, 4.0]])

print(np.linalg.det(A))           # -2.0
print(np.linalg.inv(A).tolist())  # 逆矩阵
print(np.linalg.norm(A))          # 范数
print(np.linalg.solve(A, [1, 1])) # 求解线性方程组

# 矩阵分解
U, S, V = np.linalg.svd(A)
Q, R = np.linalg.qr(A)
```

#### 5.6 随机数

```python
import rsnumpy as np

# 新 API
rng = np.random.default_rng(seed=42)
print(rng.random(5).tolist())   # 均匀分布
print(rng.normal(0, 1, 5).tolist())  # 正态分布
print(rng.integers(0, 10, 5).tolist())  # 整数

# 旧 API
np.random.seed(0)
print(np.random.rand(3).tolist())
print(np.random.randn(3).tolist())
```

#### 5.7 FFT

```python
import rsnumpy as np

x = np.array([1.0, 0.0, 0.0, 0.0])
spectrum = np.fft.fft(x)
recovered = np.fft.ifft(spectrum)

# 实数输入
r_spectrum = np.fft.rfft(x)
recovered_r = np.fft.irfft(r_spectrum, n=4)
```

#### 5.8 多项式

```python
import rsnumpy as np

# 系数从高到低（NumPy 兼容）
p = np.polynomial.Poly([1, -3, 2])  # x^2 - 3x + 2
print(p(2))               # 0（在 x=2 处的值）
print(p.roots().tolist()) # [2.0, 1.0]

# 多项式运算
q = np.polynomial.Poly([1, -1])  # x - 1
print(p + q)              # Poly([1, -2, 1])
print(p * q)              # Poly([1, -4, 5, -2])

# 拟合
x = np.array([0, 1, 2, 3, 4])
y = np.array([1, 2, 5, 10, 17])  # y = x^2 + 1
coef = np.polynomial.polyfit(x, y, 2)
```

#### 5.9 文件 I/O

```python
import rsnumpy as np

# 保存 / 加载 .npy
a = np.array([1, 2, 3])
np.save('data.npy', a)
b = np.load('data.npy')

# 保存 / 加载文本
np.savetxt('data.txt', a)
data = np.loadtxt('data.txt')

# 保存 / 加载 .npz（多个数组）
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])
np.savez('multi.npz', a, b, c=a)
loaded = np.load_npz('multi.npz')
print(loaded['arr_0'].tolist())   # [1, 2, 3]
print(loaded['c'].tolist())       # [1, 2, 3]

# 从缓冲区创建
arr = np.frombuffer(bytes_data)
```

#### 5.10 判断函数与常量

```python
import rsnumpy as np

print(np.pi)           # 3.141592653589793
print(np.e)            # 2.718281828459045
print(np.inf)          # inf
print(np.nan)          # nan

a = np.array([1.0, np.nan, np.inf])
print(np.isnan(a))     # 逐元素判断
print(np.isinf(a))
print(np.isfinite(a))
```

---

### 6. 常见问题

#### Q1: `ModuleNotFoundError: No module named 'rsnumpy'`
A: 需要先构建并安装：`bash build_wheel.sh`

#### Q2: 编译报错 `error: linker not found`
A: 安装 Xcode Command Line Tools（macOS）：`xcode-select --install`

#### Q3: 编译报错 `pyo3` 版本冲突
A: 确保 Python ≥ 3.8，且 `pip install --upgrade maturin pyo3`

#### Q4: 性能是否优于 NumPy？
A: 取决于具体操作。Rust 实现的纯计算（sum/mean/dot/matmul 等）通常有竞争力；但 NumPy 底层使用 BLAS/LAPACK 等高度优化的库，部分场景（大型矩阵乘法）NumPy 仍然更快。

#### Q5: 是否支持 GPU？
A: 当前版本仅支持 CPU。

---

### 7. 开发提示

- **所有计算逻辑都应在 Rust 层实现**，Python 层只保留 `def f(x): return _core.f(x)` 这样的薄包装
- 新增 API 时，先在 `src/lib.rs` 实现，再用 `m.add_function(wrap_pyfunction!(name, m)?)?` 注册
- 修改后需要重新运行 `bash build_wheel.sh`
- 提交前请运行 `/Users/user/Desktop/rust_project/rsnumpy/.venv/bin/python /tmp/test_*.py` 验证

---

## English Documentation

### 1. Overview

**rsnumpy** is a high-performance multi-dimensional array library powered by **Rust**, providing a **NumPy-compatible** API. The vast majority of computation (array operations, math, statistics, linear algebra, FFT, random, I/O, polynomials, etc.) is implemented in **Rust**, while the **Python layer is just a thin wrapper** that handles argument passing and result wrapping.

**Advantages:**
- **Performance**: Core computations in Rust, comparable to or faster than NumPy
- **Type safety**: Strong typing from Rust eliminates runtime errors
- **Memory safety**: No data races, no memory leaks
- **Comprehensive API**: Covers all common NumPy functionality

**Project layout:**
```
rsnumpy/
├── src/                       # Rust source
│   ├── lib.rs                 # Core ndarray & general functions
│   ├── fft.rs                 # Fast Fourier Transform
│   ├── linalg.rs              # Linear algebra
│   └── random.rs              # Random number generation
├── rsnumpy/                     # Python thin wrappers
│   ├── __init__.py            # Main module, exports public API
│   ├── array_methods.py       # ndarray object methods
│   ├── array_ops.py           # Array manipulation functions
│   ├── math_functions.py      # Math functions
│   ├── statistics.py          # Statistics functions
│   ├── io.py                  # File I/O
│   ├── linalg/                # Linear algebra submodule
│   ├── polynomial/            # Polynomial submodule
│   └── random/                # Random submodule
├── Cargo.toml                 # Rust dependencies
├── pyproject.toml             # Python build config
├── build_wheel.sh             # Build script
└── README.md                  # This file
```

---

### 2. Requirements

| Tool         | Minimum Version | Notes                       |
|--------------|-----------------|-----------------------------|
| Python       | ≥ 3.8           | 3.10+ recommended           |
| Rust         | ≥ 1.75          | `edition = "2024"`          |
| maturin      | ≥ 1.13, < 2.0   | Rust ↔ Python binding       |
| uv (optional)| latest          | Fast venv & package manager |

---

### 3. Quick Start

#### 3.1 Clone the repository

```bash
git clone <your-repo-url> rsnumpy
cd rsnumpy
```

#### 3.2 Create a virtual environment (uv recommended)

```bash
# Create .venv with uv
uv venv .venv --python 3.11
source .venv/bin/activate

# Install maturin and numpy
uv pip install maturin numpy
```

Or with standard `venv`:

```bash
python -m venv .venv
source .venv/bin/activate
pip install maturin numpy
```

#### 3.3 Build & install

Just run the build script:

```bash
bash build_wheel.sh
```

The script will:
1. Detect Python from `.venv`
2. Run `maturin build --release`
3. Generate `.whl` in `wheelhouse/`
4. Install into `.venv` with `uv pip install --no-deps`

After build, verify:

```python
python -c "import rsnumpy; print(rsnumpy.__doc__)"
```

---

### 4. Detailed Build Options

#### 4.1 Using the build script (recommended)

```bash
# Default release mode, install to .venv
bash build_wheel.sh

# Specify Python interpreter
bash build_wheel.sh --python .venv/bin/python

# Output to a custom directory
bash build_wheel.sh --out-dir build/

# Debug mode (faster compile, slower runtime)
bash build_wheel.sh --debug
```

#### 4.2 Manual maturin commands

```bash
# Develop mode: live editing, no rebuild needed
maturin develop --release

# Build a wheel only
maturin build --release -o wheelhouse/

# Install the wheel manually
uv pip install --no-deps wheelhouse/rsnumpy-*.whl
```

#### 4.3 Clean build artifacts

```bash
rm -rf target/ wheelhouse/
```

---

### 5. Usage Examples

#### 5.1 Basic array operations

```python
import rsnumpy as np

# Array creation
a = np.array([1.0, 2.0, 3.0])
b = np.zeros((2, 3))        # 2x3 zeros
c = np.ones((2, 2))          # 2x2 ones
d = np.eye(3)                # 3x3 identity matrix
e = np.arange(0, 10, 2)      # [0, 2, 4, 6, 8]
f = np.linspace(0, 1, 5)     # [0, 0.25, 0.5, 0.75, 1.0]

# Attributes
print(a.shape)    # (3,)
print(a.ndim)     # 1
print(a.size)     # 3
```

#### 5.2 Math functions

```python
import rsnumpy as np

a = np.array([0.0, np.pi / 2, np.pi])
print(np.sin(a))         # [0.0, 1.0, 0.0]
print(np.cos(a))         # [1.0, 0.0, -1.0]
print(np.exp(a))         # exponential
print(np.sqrt(a))        # square root
print(np.log(a + 1))     # natural log
```

#### 5.3 Array manipulation

```python
import rsnumpy as np

a = np.array([[1, 2, 3], [4, 5, 6]])

# Reshape
b = a.reshape((3, 2))
print(b.tolist())   # [[1, 2], [3, 4], [5, 6]]

# Transpose
c = a.transpose()
print(c.tolist())   # [[1, 4], [2, 5], [3, 6]]

# Slicing (multi-dim tuple index, implemented in Rust)
print(a[0:2, 1:3].tolist())  # [[2, 3], [5, 6]]
print(a[:, 1].tolist())      # [[2], [5]]

# Concatenation
d = np.concatenate([a, a], axis=0)
print(d.shape)   # (4, 3)

e = np.vstack([a, a])   # vertical stack
f = np.hstack([a, a])   # horizontal stack
```

#### 5.4 Statistics

```python
import rsnumpy as np

a = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

print(np.sum(a))         # 15
print(np.mean(a))        # 3.0
print(np.std(a))         # standard deviation
print(np.var(a))         # variance
print(np.max(a))         # 5
print(np.argmin(a))      # 0
```

#### 5.5 Linear algebra

```python
import rsnumpy as np

A = np.array([[1.0, 2.0], [3.0, 4.0]])

print(np.linalg.det(A))           # -2.0
print(np.linalg.inv(A).tolist())  # inverse matrix
print(np.linalg.norm(A))          # Frobenius norm
print(np.linalg.solve(A, [1, 1])) # solve linear system

# Decompositions
U, S, V = np.linalg.svd(A)
Q, R = np.linalg.qr(A)
```

#### 5.6 Random numbers

```python
import rsnumpy as np

# New API
rng = np.random.default_rng(seed=42)
print(rng.random(5).tolist())        # uniform
print(rng.normal(0, 1, 5).tolist())  # normal
print(rng.integers(0, 10, 5).tolist())  # integers

# Legacy API
np.random.seed(0)
print(np.random.rand(3).tolist())
print(np.random.randn(3).tolist())
```

#### 5.7 FFT

```python
import rsnumpy as np

x = np.array([1.0, 0.0, 0.0, 0.0])
spectrum = np.fft.fft(x)
recovered = np.fft.ifft(spectrum)

# Real input
r_spectrum = np.fft.rfft(x)
recovered_r = np.fft.irfft(r_spectrum, n=4)
```

#### 5.8 Polynomials

```python
import rsnumpy as np

# Coefficients from high to low (NumPy-compatible)
p = np.polynomial.Poly([1, -3, 2])  # x^2 - 3x + 2
print(p(2))               # 0 (value at x=2)
print(p.roots().tolist()) # [2.0, 1.0]

# Polynomial arithmetic
q = np.polynomial.Poly([1, -1])  # x - 1
print(p + q)              # Poly([1, -2, 1])
print(p * q)              # Poly([1, -4, 5, -2])

# Curve fitting
x = np.array([0, 1, 2, 3, 4])
y = np.array([1, 2, 5, 10, 17])  # y = x^2 + 1
coef = np.polynomial.polyfit(x, y, 2)
```

#### 5.9 File I/O

```python
import rsnumpy as np

# Save / load .npy
a = np.array([1, 2, 3])
np.save('data.npy', a)
b = np.load('data.npy')

# Save / load text
np.savetxt('data.txt', a)
data = np.loadtxt('data.txt')

# Save / load .npz (multiple arrays)
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])
np.savez('multi.npz', a, b, c=a)
loaded = np.load_npz('multi.npz')
print(loaded['arr_0'].tolist())   # [1, 2, 3]
print(loaded['c'].tolist())       # [1, 2, 3]

# From buffer
arr = np.frombuffer(bytes_data)
```

#### 5.10 Constants & predicates

```python
import rsnumpy as np

print(np.pi)           # 3.141592653589793
print(np.e)            # 2.718281828459045
print(np.inf)          # inf
print(np.nan)          # nan

a = np.array([1.0, np.nan, np.inf])
print(np.isnan(a))     # element-wise
print(np.isinf(a))
print(np.isfinite(a))
```

---

### 6. FAQ

#### Q1: `ModuleNotFoundError: No module named 'rsnumpy'`
A: Build and install first: `bash build_wheel.sh`

#### Q2: Compilation error `error: linker not found`
A: Install Xcode Command Line Tools (macOS): `xcode-select --install`

#### Q3: Compilation error `pyo3` version conflict
A: Ensure Python ≥ 3.8 and `pip install --upgrade maturin pyo3`

#### Q4: Is rsnumpy faster than NumPy?
A: It depends. Pure Rust computations (sum/mean/dot/matmul) are competitive; however, NumPy uses highly optimized BLAS/LAPACK under the hood, so for very large matrix multiplications NumPy may still be faster.

#### Q5: GPU support?
A: Not in the current version (CPU only).

---

### 7. Development Notes

- **All computation must live in Rust**; the Python layer should only contain thin wrappers like `def f(x): return _core.f(x)`
- When adding new APIs, implement them in `src/lib.rs` first, then register with `m.add_function(wrap_pyfunction!(name, m)?)?`
- After modification, run `bash build_wheel.sh` to rebuild
- Before committing, run `python /tmp/test_*.py` to verify functionality

---

## License

MIT
