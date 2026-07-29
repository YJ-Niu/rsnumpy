# rsnumpy

**A Rust-powered NumPy-compatible array library.**

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python Version](https://img.shields.io/badge/python-%3E%3D%203.10-blue.svg)
![Rust Version](https://img.shields.io/badge/rust-%3E%3D%201.75-orange.svg)

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
│   ├── indexing.rs            # 多维索引与切片
│   ├── fft.rs                 # 快速傅里叶变换
│   ├── linalg.rs              # 线性代数
│   └── random.rs              # 随机数生成（可复现并行采样）
├── python/rsnumpy/            # Python 薄包装
│   ├── __init__.py            # 主模块，整合所有 API
│   ├── array_methods.py       # ndarray 对象方法
│   ├── array_ops.py           # 数组操作函数
│   ├── _extra.py              # 补充 API（别名、nan 系列、集合运算、窗函数等）
│   ├── math_functions.py      # 数学函数
│   ├── statistics.py          # 统计函数
│   ├── char.py                # 字符串数组函数
│   ├── matlib.py              # 矩阵便捷构造
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

| 工具       | 最低版本      | 说明                 |
| ---------- | ------------- | -------------------- |
| Python     | ≥ 3.9         | 推荐 3.10+           |
| Rust       | ≥ 1.75        | `edition = "2024"`   |
| maturin    | ≥ 1.13, < 2.0 | Rust ↔ Python 绑定   |
| uv（可选） | 最新          | 快速创建 venv 与安装 |

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

# 安装 maturin 和 rsnumpy
uv pip install maturin
```

如果使用标准 `venv`：

```bash
python -m venv .venv
source .venv/bin/activate
pip install maturin
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

> **可复现性**：相同 `seed` 保证生成相同的结果。大数组采样在 Rust 层并行执行，采用固定分块 + `splitmix64` 派生各分块的独立种子，因此**结果与线程数无关**，在不同机器上同样可复现。同一个 `Generator` 每次调用都会推进内部状态，连续调用不会产生重复序列。

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

#### 5.8 零拷贝缓冲协议

rsnumpy 支持 PEP 3118 缓冲协议，允许下游库（如 rsplotlib）零拷贝读取数组数据：

```python
import rsnumpy as np

# float64 数组支持 memoryview 零拷贝
a = np.array([1.0, 2.0, 3.0, 4.0])
mv = memoryview(a)
print(mv.format)   # 'd' (f64)
print(mv.shape)    # (4,)
print(mv.readonly) # True

# numpy 也能通过缓冲协议零拷贝读取
import numpy as np
arr = np.asarray(a)  # 零拷贝，dtype=float64

# int/bool 等非 float64 dtype 通过 __array_interface__ bytes 回退（保证 dtype 精度）
ia = np.array([1, 2, 3], dtype='int64')
arr_int = np.asarray(ia)  # dtype=int64，通过 bytes 副本
```

#### 5.9 多项式

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

#### 5.10 文件 I/O

```python
import rsnumpy as np

# 保存 / 加载 .npy
a = np.array([1, 2, 3])
np.save('data.npy', a)
b = np.load('data.npy')

# 保存 / 加载文本
# fmt 参数控制输出格式，delimiter 参数指定分隔符
np.savetxt('data.txt', a, fmt='%d', delimiter=',')  # 保存为整数，逗号分隔
data = np.loadtxt('data.txt', delimiter=',', dtype=int)  # 加载为整数类型
print(data)  # [[0 0 1 1 2]
             #  [2 3 3 4 4]
             #  [5 5 6 6 7]
             #  [7 8 8 9 9]]

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

#### 5.11 判断函数与常量

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

#### 5.12 日期时间与网格

```python
import rsnumpy as np

# 日期时间标量（兼容 numpy.datetime64）
d = np.datetime64('2024-01-01')
delta = np.timedelta64(7, 'D')          # 7 天
print(d + delta)                         # numpy.datetime64('2024-01-08')
print(np.datetime64('2024-01-08') - d)   # numpy.timedelta64(7,'D')

# 用 datetime64/timedelta64 生成日期序列
days = np.arange(np.datetime64('2024-01-01'),
                 np.datetime64('2024-01-05'),
                 np.timedelta64(1, 'D'))

# 坐标网格（默认 'xy' 索引，与 NumPy 一致）
x = np.array([1, 2, 3])
y = np.array([10, 20])
X, Y = np.meshgrid(x, y)
print(X.tolist())   # [[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]]
print(Y.tolist())   # [[10.0, 10.0, 10.0], [20.0, 20.0, 20.0]]
```

#### 5.12 补充 API 覆盖（对齐 NumPy 2.5.1）

在核心 Rust 原语之上，`_extra.py` 补充了约 140 个 NumPy 兼容函数，全部对照 NumPy 2.5.1 逐函数验证通过。计算密集/逐元素部分复用现有 Rust 原语，纯变形与组合类则以薄 Python 包装实现。

```python
import rsnumpy as np

# 三角/双曲别名（Array API 命名）
np.acos(1.0); np.asin(0.0); np.atan(1.0); np.atan2(1.0, 1.0)
np.radians(180.0); np.degrees(np.pi)

# 逐元素数学
np.hypot(3.0, 4.0)          # 5.0
np.maximum([1, 5], [3, 2])  # [3, 5]
np.fmax([np.nan, 2], [1, np.nan])  # 忽略 NaN
np.logaddexp(0.0, 0.0)      # 数值稳定 log(exp(a)+exp(b))
np.rint([0.5, 1.5, 2.5])    # 四舍六入五成双 -> [0, 2, 2]
np.gcd(12, 8); np.lcm(4, 6)
frac, whole = np.modf([1.5, 2.25])

# nan 系列归约
a = np.array([1.0, np.nan, 3.0])
np.nansum(a); np.nanmean(a); np.nanstd(a); np.nanmax(a)
np.nancumsum(a); np.nanargmax(a)
np.prod([1, 2, 3, 4])       # 24

# 集合运算
np.intersect1d([1, 2, 3], [2, 3, 4])  # [2, 3]
np.union1d([1, 2], [2, 3])            # [1, 2, 3]
np.setdiff1d([1, 2, 3], [2])          # [1, 3]
np.isin([1, 2, 5], [1, 5])            # [True, False, True]
uniq, counts = np.unique_counts([1, 1, 2, 3, 3, 3])

# 变形 / 组合
np.block([[np.eye(2), np.ones((2, 1))]])
np.pad([1, 2, 3], (1, 2), mode='reflect')
np.kron([1, 2], [1, 1])
np.tensordot(np.ones((2, 3)), np.ones((3, 4)), axes=1)

# 复数
z = np.array([1 + 2j, 3 - 1j])
np.real(z); np.imag(z); np.conjugate(z); np.angle(z)

# 信号 / 窗函数 / 插值
np.convolve([1, 2, 3], [1, 1])
np.interp(2.5, [1, 2, 3], [10, 20, 30])
np.hanning(8); np.hamming(8); np.blackman(8); np.bartlett(8)
np.vander([1, 2, 3], 3)
np.bincount([0, 1, 1, 2, 2, 2])
```

---

### 6. 常见问题

#### Q1: `ModuleNotFoundError: No module named 'rsnumpy'`

A: 需要先构建并安装：`bash build_wheel.sh`

#### Q2: 编译报错 `error: linker not found`

A: 安装 Xcode Command Line Tools（macOS）：`xcode-select --install`

#### Q3: 编译报错 `pyo3` 版本冲突

A: 确保 Python ≥ 3.10，且 `pip install --upgrade maturin pyo3`

#### Q4: 性能是否优于 NumPy？

A: 多数逐元素与归约操作（`sin`/`exp`/`sum`/`std`/`cumsum` 等）在中大规模数组上快于 NumPy（约 1.3x–5x）；`matmul`/`dot` 在 macOS 上通过 Accelerate BLAS 与 NumPy 基本持平。小数组因跨语言调用开销可能略慢。详见「性能对比」。

#### Q5: 是否支持 GPU？

A: 当前版本仅支持 CPU。

---

### 7. 开发提示

- **所有计算逻辑都应在 Rust 层实现**，Python 层只保留 `def f(x): return _core.f(x)` 这样的薄包装
- 新增 API 时，先在 `src/lib.rs` 实现，再用 `m.add_function(wrap_pyfunction!(name, m)?)?` 注册
- 修改后需要重新运行 `bash build_wheel.sh`
- 提交前请运行测试脚本验证：`.venv/bin/python test/run_test.py`

### 8. 性能对比

以下为在 macOS (Apple Silicon) 上以 NumPy 为基准的实测结果。每个算子取 20 次运行的最优耗时（`sort`/`matmul` 取 5 次），输入为 `arange` 生成的 `float64` 数组。「相对 NumPy」= NumPy 耗时 ÷ rsnumpy 耗时，> 1 表示 rsnumpy 更快。

**逐元素 / 归约（n = 1,000,000）**

| 操作       | rsnumpy  | NumPy    | 相对 NumPy |
| ---------- | -------- | -------- | ---------- |
| `add`      | 0.113 ms | 0.190 ms | **1.7x**   |
| `multiply` | 0.110 ms | 0.186 ms | **1.7x**   |
| `divide`   | 0.116 ms | 0.188 ms | **1.6x**   |
| `sin`      | 0.570 ms | 2.969 ms | **5.2x**   |
| `exp`      | 0.248 ms | 1.349 ms | **5.4x**   |
| `sqrt`     | 0.134 ms | 0.225 ms | **1.7x**   |
| `sum`      | 0.081 ms | 0.103 ms | **1.3x**   |
| `mean`     | 0.080 ms | 0.104 ms | **1.3x**   |
| `std`      | 0.271 ms | 0.438 ms | **1.6x**   |
| `cumsum`   | 1.030 ms | 1.908 ms | **1.9x**   |
| `sort`     | 0.741 ms | 24.03 ms | **32x**    |
| `interp`   | 0.227 ms | 0.687 ms | **3.0x**   |
| `i0`       | 4.459 ms | 16.41 ms | **3.7x**   |

**矩阵乘法（方阵，macOS 下经 Accelerate BLAS）**

| 规模      | rsnumpy  | NumPy    | 相对 NumPy |
| --------- | -------- | -------- | ---------- |
| 128×128   | 0.013 ms | 0.011 ms | 0.9x       |
| 512×512   | 0.607 ms | 0.605 ms | 1.0x       |
| 1024×1024 | 4.375 ms | 4.470 ms | 1.0x       |

> **说明**：小数组（约 n < 数万）因跨语言调用与线程调度开销，rsnumpy 可能略慢于 NumPy；随规模增大，Rust 层的并行与向量化优势显现。`matmul` 在 macOS 上分派到系统 Accelerate 框架，与 NumPy 基本持平。实际数字随机器与数据分布浮动。

#### 8.1 性能架构

rsnumpy 的性能来自以下几项底层设计：

- **主动释放 GIL**：计算密集算子（逐元素、归约、排序、`cumsum`、`matmul`、`inv` 等）在进入 Rust 前通过 `py.detach` 释放 GIL，使纯计算与其它 Python 线程真正并行。
- **成本分级并行阈值**：按算子的每元素成本设置不同的并行启用门槛——超越函数（`sin`/`exp` 等）阈值最低，访存密集型（`add`/`mul`/`div`）阈值最高——避免小数组因线程调度反而变慢。
- **BLAS 后端**：macOS 上 `matmul`/`dot` 经 ndarray 的 `blas` 特性分派到系统 Accelerate 框架；其它平台回退到多线程的纯 Rust `matrixmultiply` 内核。
- **2×2 矩阵求逆并行**：`linalg.inv` 的 2×2 批量分支在 Rust 层用 rayon 并行加速，支持复数类型，避免 Python 层逐个矩阵求逆。
- **连续内存与就地扫描**：`cumsum`/`cumprod` 等在单份 C 序缓冲上就地前缀扫描并按连续块并行，减少分配与拷贝。
- **精简依赖**：关闭 `zip` 等依赖中未使用的编解码器特性，配合 `lto = "fat"` 与符号裁剪，将扩展体积控制在约 5 MB。

### 9. CI/CD

项目使用 GitHub Actions 进行持续集成：

- **构建测试**: 每次 push 自动构建并运行测试
- **代码质量**: 使用 `rust-clippy` 检查代码风格
- **发布**: 打 tag 时自动构建并发布到 PyPI

相关配置文件：

- `.github/workflows/ci.yml` - 主 CI 流程
- `.github/workflows/rust-clippy.yml` - Rust 代码质量检查
- `.github/workflows/release.yml` - 发布流程

### 10. 贡献指南

欢迎贡献代码！请遵循以下流程：

1. **Fork 项目**并创建新分支
2. **编写代码**，确保遵循项目规范：
   - 所有计算逻辑在 Rust 层实现
   - Python 层只做参数传递
   - 添加适当的测试用例
3. **运行测试**：`.venv/bin/python test/run_test.py`
4. **提交 PR**，描述你的改动

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
│   ├── indexing.rs            # Multi-dimensional indexing & slicing
│   ├── fft.rs                 # Fast Fourier Transform
│   ├── linalg.rs              # Linear algebra
│   └── random.rs              # Random number generation (reproducible parallel sampling)
├── python/rsnumpy/            # Python thin wrappers
│   ├── __init__.py            # Main module, exports public API
│   ├── array_methods.py       # ndarray object methods
│   ├── array_ops.py           # Array manipulation functions
│   ├── _extra.py              # Supplementary API (aliases, nan-reductions, set ops, windows, ...)
│   ├── math_functions.py      # Math functions
│   ├── statistics.py          # Statistics functions
│   ├── char.py                # String array functions
│   ├── matlib.py              # Matrix construction helpers
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

| Tool          | Minimum Version | Notes                       |
| ------------- | --------------- | --------------------------- |
| Python        | ≥ 3.10          | 3.10+ recommended           |
| Rust          | ≥ 1.75          | `edition = "2024"`          |
| maturin       | ≥ 1.13, < 2.0   | Rust ↔ Python binding       |
| uv (optional) | latest          | Fast venv & package manager |

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

> **Reproducibility**: the same `seed` always produces the same result. Large-array sampling runs in parallel on the Rust side using fixed-size chunks with per-chunk seeds derived via `splitmix64`, so results are **independent of the thread count** and reproducible across machines. Each call on a `Generator` advances its internal state, so consecutive calls never repeat the same sequence.

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

#### 5.8 Zero-Copy Buffer Protocol

rsnumpy supports the PEP 3118 buffer protocol, allowing downstream libraries like rsplotlib to read array data without copying:

```python
import rsnumpy as np

# float64 arrays support zero-copy memoryview
a = np.array([1.0, 2.0, 3.0, 4.0])
mv = memoryview(a)
print(mv.format)   # 'd' (f64)
print(mv.shape)    # (4,)
print(mv.readonly) # True

# numpy can also read via buffer protocol (zero-copy)
import numpy as np
arr = np.asarray(a)  # zero-copy, dtype=float64

# Non-float64 dtypes (int/bool) fall back to __array_interface__ bytes (dtype fidelity)
ia = np.array([1, 2, 3], dtype='int64')
arr_int = np.asarray(ia)  # dtype=int64, via bytes copy
```

#### 5.9 Polynomials

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

#### 5.10 File I/O

```python
import rsnumpy as np

# Save / load .npy
a = np.array([1, 2, 3])
np.save('data.npy', a)
b = np.load('data.npy')

# Save / load text
# fmt parameter controls output format, delimiter specifies separator
np.savetxt('data.txt', a, fmt='%d', delimiter=',')  # Save as integers, comma-separated
data = np.loadtxt('data.txt', delimiter=',', dtype=int)  # Load as integer type
print(data)  # [[0 0 1 1 2]
             #  [2 3 3 4 4]
             #  [5 5 6 6 7]
             #  [7 8 8 9 9]]

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

#### 5.11 Constants & predicates

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

#### 5.12 Datetime & meshgrid

```python
import rsnumpy as np

# Datetime scalars (compatible with numpy.datetime64)
d = np.datetime64('2024-01-01')
delta = np.timedelta64(7, 'D')           # 7 days
print(d + delta)                          # numpy.datetime64('2024-01-08')
print(np.datetime64('2024-01-08') - d)    # numpy.timedelta64(7,'D')

# Generate a date range with datetime64/timedelta64
days = np.arange(np.datetime64('2024-01-01'),
                 np.datetime64('2024-01-05'),
                 np.timedelta64(1, 'D'))

# Coordinate grids (default 'xy' indexing, matching NumPy)
x = np.array([1, 2, 3])
y = np.array([10, 20])
X, Y = np.meshgrid(x, y)
print(X.tolist())   # [[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]]
print(Y.tolist())   # [[10.0, 10.0, 10.0], [20.0, 20.0, 20.0]]
```

#### 5.12 Supplementary API Coverage (aligned with NumPy 2.5.1)

On top of the core Rust primitives, `_extra.py` adds ~140 NumPy-compatible functions, each verified against NumPy 2.5.1. Compute-heavy / element-wise paths reuse existing Rust primitives, while pure reshaping and composition helpers are implemented as thin Python wrappers.

```python
import rsnumpy as np

# Trig/hyperbolic aliases (Array API names)
np.acos(1.0); np.asin(0.0); np.atan(1.0); np.atan2(1.0, 1.0)
np.radians(180.0); np.degrees(np.pi)

# Element-wise math
np.hypot(3.0, 4.0)          # 5.0
np.maximum([1, 5], [3, 2])  # [3, 5]
np.fmax([np.nan, 2], [1, np.nan])  # NaN-ignoring
np.logaddexp(0.0, 0.0)      # numerically stable log(exp(a)+exp(b))
np.rint([0.5, 1.5, 2.5])    # round-half-to-even -> [0, 2, 2]
np.gcd(12, 8); np.lcm(4, 6)
frac, whole = np.modf([1.5, 2.25])

# nan-reductions
a = np.array([1.0, np.nan, 3.0])
np.nansum(a); np.nanmean(a); np.nanstd(a); np.nanmax(a)
np.nancumsum(a); np.nanargmax(a)
np.prod([1, 2, 3, 4])       # 24

# Set operations
np.intersect1d([1, 2, 3], [2, 3, 4])  # [2, 3]
np.union1d([1, 2], [2, 3])            # [1, 2, 3]
np.setdiff1d([1, 2, 3], [2])          # [1, 3]
np.isin([1, 2, 5], [1, 5])            # [True, False, True]
uniq, counts = np.unique_counts([1, 1, 2, 3, 3, 3])

# Reshaping / composition
np.block([[np.eye(2), np.ones((2, 1))]])
np.pad([1, 2, 3], (1, 2), mode='reflect')
np.kron([1, 2], [1, 1])
np.tensordot(np.ones((2, 3)), np.ones((3, 4)), axes=1)

# Complex
z = np.array([1 + 2j, 3 - 1j])
np.real(z); np.imag(z); np.conjugate(z); np.angle(z)

# Signal / windows / interpolation
np.convolve([1, 2, 3], [1, 1])
np.interp(2.5, [1, 2, 3], [10, 20, 30])
np.hanning(8); np.hamming(8); np.blackman(8); np.bartlett(8)
np.vander([1, 2, 3], 3)
np.bincount([0, 1, 1, 2, 2, 2])
```

---

### 6. FAQ

#### Q1: `ModuleNotFoundError: No module named 'rsnumpy'`

A: Build and install first: `bash build_wheel.sh`

#### Q2: Compilation error `error: linker not found`

A: Install Xcode Command Line Tools (macOS): `xcode-select --install`

#### Q3: Compilation error `pyo3` version conflict

A: Ensure Python ≥ 3.10 and `pip install --upgrade maturin pyo3`

#### Q4: Is rsnumpy faster than NumPy?

A: For most element-wise and reduction operations (`sin`/`exp`/`sum`/`std`/`cumsum`, etc.) rsnumpy is faster than NumPy on medium-to-large arrays (roughly 1.3x–5x). `matmul`/`dot` on macOS dispatch to Accelerate BLAS and are on par with NumPy. Small arrays may be slightly slower due to cross-language call overhead. See "Performance Comparison" for numbers.

#### Q5: GPU support?

A: Not in the current version (CPU only).

---

### 7. Development Notes

- **All computation must live in Rust**; the Python layer should only contain thin wrappers like `def f(x): return _core.f(x)`
- When adding new APIs, implement them in `src/lib.rs` first, then register with `m.add_function(wrap_pyfunction!(name, m)?)?`
- After modification, run `bash build_wheel.sh` to rebuild
- Before committing, run `.venv/bin/python test/run_test.py` to verify functionality

### 8. Performance Comparison

Measured on macOS (Apple Silicon), using NumPy as the baseline. Each operator reports the best of 20 runs (`sort`/`matmul` best of 5); inputs are `float64` arrays from `arange`. "vs NumPy" = NumPy time ÷ rsnumpy time, so > 1 means rsnumpy is faster.

**Element-wise / reductions (n = 1,000,000)**

| Operation  | rsnumpy  | NumPy    | vs NumPy |
| ---------- | -------- | -------- | -------- |
| `add`      | 0.113 ms | 0.190 ms | **1.7x** |
| `multiply` | 0.110 ms | 0.186 ms | **1.7x** |
| `divide`   | 0.116 ms | 0.188 ms | **1.6x** |
| `sin`      | 0.570 ms | 2.969 ms | **5.2x** |
| `exp`      | 0.248 ms | 1.349 ms | **5.4x** |
| `sqrt`     | 0.134 ms | 0.225 ms | **1.7x** |
| `sum`      | 0.081 ms | 0.103 ms | **1.3x** |
| `mean`     | 0.080 ms | 0.104 ms | **1.3x** |
| `std`      | 0.271 ms | 0.438 ms | **1.6x** |
| `cumsum`   | 1.030 ms | 1.908 ms | **1.9x** |
| `sort`     | 0.741 ms | 24.03 ms | **32x**  |
| `interp`   | 0.227 ms | 0.687 ms | **3.0x** |
| `i0`       | 4.459 ms | 16.41 ms | **3.7x** |

**Matrix multiply (square, via Accelerate BLAS on macOS)**

| Size      | rsnumpy  | NumPy    | vs NumPy |
| --------- | -------- | -------- | -------- |
| 128×128   | 0.013 ms | 0.011 ms | 0.9x     |
| 512×512   | 0.607 ms | 0.605 ms | 1.0x     |
| 1024×1024 | 4.375 ms | 4.470 ms | 1.0x     |

> **Note**: for small arrays (roughly n < tens of thousands), cross-language call and thread-scheduling overhead can make rsnumpy slightly slower; the Rust-layer parallelism and vectorization pay off as sizes grow. `matmul` dispatches to the system Accelerate framework on macOS and is on par with NumPy. Absolute numbers vary with hardware and data distribution.

#### 8.1 Performance Architecture

rsnumpy's performance comes from a few low-level design choices:

- **Releasing the GIL**: compute-heavy operators (element-wise, reductions, sort, `cumsum`, `matmul`, `inv`, ...) release the GIL via `py.detach` before entering Rust, so pure computation runs truly in parallel with other Python threads.
- **Cost-tiered parallel thresholds**: the size at which parallelism kicks in varies by per-element cost — lowest for transcendental functions (`sin`/`exp`), highest for memory-bound ops (`add`/`mul`/`div`) — avoiding slowdowns from thread scheduling on small arrays.
- **BLAS backend**: on macOS, `matmul`/`dot` dispatch to the system Accelerate framework through ndarray's `blas` feature; other platforms fall back to the multi-threaded pure-Rust `matrixmultiply` kernel.
- **2×2 matrix inversion parallel**: `linalg.inv`'s 2×2 batch branch uses rayon parallelism in Rust, supporting complex types, avoiding per-matrix inversion in Python.
- **Contiguous memory & in-place scans**: `cumsum`/`cumprod` do an in-place prefix scan over a single C-order buffer, parallelized over contiguous blocks, minimizing allocation and copies.
- **Lean dependencies**: unused codec features (e.g. in `zip`) are disabled, and combined with `lto = "fat"` and symbol stripping the extension stays around 5 MB.
- **Zero-copy buffer protocol**: `rsnumpy.ndarray` implements PEP 3118 (`__buffer__`/`__release_buffer__` in Python 3.12+) and PEP 3118 (`__getbuffer__`/`__releasebuffer__` in Rust), allowing downstream libraries like rsplotlib to read float64 array data without copying. For non-float64 dtypes, `__array_interface__` with bytes fallback ensures dtype fidelity.

### 9. CI/CD

The project uses GitHub Actions for continuous integration:

- **Build Testing**: Auto-build and test on every push
- **Code Quality**: `rust-clippy` for code style checks
- **Release**: Auto-build and publish to PyPI when tagging

Related configuration files:

- `.github/workflows/ci.yml` - Main CI workflow
- `.github/workflows/rust-clippy.yml` - Rust code quality
- `.github/workflows/release.yml` - Release workflow

### 10. Contributing

Contributions are welcome! Please follow these steps:

1. **Fork the project** and create a new branch
2. **Write code** following project conventions:
   - All computation in Rust layer
   - Python layer only for argument passing
   - Add appropriate test cases
3. **Run tests**: `.venv/bin/python test/run_test.py`
4. **Submit a PR** describing your changes

---

## License

MIT
