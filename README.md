# rsnumpy

**A high-performance NumPy-compatible array library powered by Rust.**

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-≥3.10-blue.svg)](https://www.python.org)
[![Rust](https://img.shields.io/badge/rust-≥1.75-orange.svg)](https://www.rust-lang.org)
[![Platforms](https://img.shields.io/badge/platforms-Linux%20%7C%20macOS%20%7C%20Windows-lightgrey.svg)]()

---

rsnumpy 重新实现了 NumPy 的核心 API，**所有计算密集型逻辑（数组操作、数学函数、统计、线性代数、FFT、随机数、I/O、多项式等）均在 Rust 层完成**，Python 层只做参数转发与结果包装。计算过程主动释放 GIL 并按算子成本分级启用并行，让 Python 多线程调用也能真正并行加速。

---

## ✨ 特性亮点

- 🚀 **Rust 驱动的高性能** — `sin`/`exp` 比 NumPy 快 **5x**，`sort` 快 **32x**，逐元素运算快 **1.6–1.7x**
- 🧵 **GIL-Free 并行** — 计算密集算子通过 `py.detach` 释放 GIL，多线程 Python 真正并行
- 📊 **完整的 NumPy 兼容 API** — 涵盖数组、数学、统计、linalg、FFT、random、I/O、polynomial、datetime64、masked/structured array
- 🪟 **零拷贝缓冲协议** — 实现 PEP 3118（Rust 层 `__getbuffer__`）与 PEP 3118（Python 3.12+ `__buffer__`），可被 NumPy / rsplotlib 直接零拷贝消费
- 🎯 **智能 dtype 推断** — 自动将字符串推断为 `U{n}`，避免不必要的 `object` dtype 性能陷阱
- 🛠️ **大数组处理工具** — `chunked_apply` 分块接口、`mmap_array` 内存映射、`LazyArray` 惰性求值
- 💎 **变长元素支持** — `VarStringArray` 变长字符串、`BigIntArray` 任意精度整数
- 🔒 **线程安全工具** — `threadsafe_copy` / `threadsafe_view`，多线程共享安全且比加锁快 **100x**
- 🪶 **精简二进制** — 关闭未用编解码器特性 + `lto="fat"` + 符号裁剪，扩展体积仅 **~5 MB**
- ⚡ **快速 import** — 通过延迟导入解决循环依赖，`import rsnumpy` 启动时间 **~7ms**

---

## 📦 安装

```bash
pip install rsnumpy
# 或
uv pip install rsnumpy
```

预构建 wheel 覆盖 **Python 3.10–3.14**（Linux x86_64/aarch64、macOS Apple Silicon/Intel、Windows x64）。

---

## 🚀 快速开始

```python
import rsnumpy as np

# 基础数组
a = np.array([1.0, 2.0, 3.0, 4.0])
b = np.arange(12).reshape(3, 4)

# 数学
print(np.sin(a).tolist())          # 元素级 sin
print(np.exp(a).tolist())          # 指数
print(np.sum(a), np.mean(a))       # 归约

# 线性代数
A = np.array([[1.0, 2.0], [3.0, 4.0]])
print(np.linalg.inv(A).tolist())   # 矩阵求逆
print(np.linalg.det(A))            # 行列式

# FFT
spectrum = np.fft.fft(a)

# 随机数（线程数无关的可复现并行采样）
rng = np.random.default_rng(42)
print(rng.random(5).tolist())
```

更多示例请参考 [使用示例](#-使用示例) 章节。

---

## 📊 性能对比

> 测试环境：macOS (Apple Silicon)，NumPy 为基准，取最优耗时（20 次运行，`sort`/`matmul` 取 5 次）。
> 「相对 NumPy」= NumPy 耗时 ÷ rsnumpy 耗时，> 1 表示 rsnumpy 更快。

**逐元素 / 归约（n = 1,000,000，float64）**

| 操作 | rsnumpy | NumPy | 相对 NumPy |
|------|---------|-------|------------|
| `add` / `multiply` / `divide` | 0.11 ms | 0.19 ms | **1.6–1.7x** |
| `sqrt` | 0.13 ms | 0.23 ms | **1.7x** |
| `sum` / `mean` | 0.08 ms | 0.10 ms | **1.3x** |
| `std` | 0.27 ms | 0.44 ms | **1.6x** |
| `cumsum` | 1.03 ms | 1.91 ms | **1.9x** |
| `sort` | 0.74 ms | 24.0 ms | **32x** |
| `sin` | 0.57 ms | 2.97 ms | **5.2x** |
| `exp` | 0.25 ms | 1.35 ms | **5.4x** |
| `interp` | 0.23 ms | 0.69 ms | **3.0x** |
| `i0` | 4.46 ms | 16.4 ms | **3.7x** |

**矩阵乘法（macOS Accelerate BLAS）**

| 规模 | rsnumpy | NumPy | 相对 NumPy |
|------|---------|-------|------------|
| 128×128 | 0.013 ms | 0.011 ms | 0.9x |
| 512×512 | 0.607 ms | 0.605 ms | 1.0x |
| 1024×1024 | 4.38 ms | 4.47 ms | 1.0x |

> 小数组（n < 数万）受跨语言调用与线程调度影响，rsnumpy 可能略慢；规模增大后 Rust 层并行与向量化优势显现。

### 性能架构

- **GIL 释放**：所有计算密集算子（`sort`/`cumsum`/`matmul`/`inv`/逐元素/归约）通过 `py.detach` 释放 GIL
- **成本分级并行阈值**：按每元素成本设置不同并行门槛——超越函数（`sin`/`exp`）最低、访存密集（`add`/`mul`）最高——避免小数组的线程调度反噬
- **BLAS 分派**：macOS 上 `matmul`/`dot` 通过 ndarray 的 `blas` 特性分派到系统 Accelerate；其它平台回退到多线程 `matrixmultiply`
- **就地前缀扫描**：`cumsum`/`cumprod` 在 C 序缓冲上原地并行扫描，最小化分配与拷贝
- **2×2 矩阵求逆并行**：`linalg.inv` 的 2×2 批量分支用 rayon 并行，支持复数
- **零拷贝缓冲**：PEP 3118 让 rsplotlib 等下游库无需复制即可读取 float64 数据

---

## 🏗️ 项目结构

```
rsnumpy/
├── src/                       # Rust 源码（22,000+ 行）
│   ├── lib.rs                 # 核心 ndarray 与通用函数
│   ├── indexing.rs            # 多维索引与切片
│   ├── fft.rs                 # 快速傅里叶变换
│   ├── linalg.rs              # 线性代数
│   ├── random.rs              # 随机数（可复现并行采样）
│   ├── threadpool.rs          # 线程池与并行阈值
│   ├── arithmetic/            # 算术运算
│   ├── bitwise/               # 位运算
│   ├── buffer/                # 缓冲互操作
│   ├── creation/              # 数组创建
│   ├── formatting/            # 输出格式化
│   ├── io/                    # .npy/.npz 读写
│   ├── logic/                 # 逻辑运算
│   ├── manipulation/          # 数组操作
│   ├── mathematics/           # 数学函数
│   ├── poly/                  # 多项式
│   ├── searching/             # 搜索
│   ├── setops/                # 集合运算
│   ├── sorting/               # 排序
│   └── statistics/            # 统计
├── python/rsnumpy/            # Python 薄包装（26,000+ 行）
│   ├── __init__.py            # 主模块，公开 API
│   ├── array_methods.py       # ndarray 对象方法
│   ├── array_ops.py           # 数组操作函数
│   ├── _extra.py              # 补充 API（~140 个 NumPy 兼容函数）
│   ├── _dtypes.py             # dtype 定义
│   ├── char.py                # 字符串数组
│   ├── io.py                  # 文件 I/O
│   ├── ma.py                  # 掩码数组
│   ├── rec.py                 # 记录数组
│   ├── math_functions.py      # 数学函数
│   ├── statistics.py          # 统计函数
│   ├── linalg/                # 线性代数子模块
│   ├── polynomial/            # 多项式子模块
│   ├── random/                # 随机数子模块
│   ├── fft/                   # FFT 子模块
│   ├── lib/                   # recfunctions 等工具
│   └── typing/                # 类型定义
├── test/                      # 测试
│   ├── run_test.py            # 主测试
│   ├── run_test2.py           # 辅助测试
│   └── test_rf/               # 兼容层（networkx、scipy）
├── debug/                     # 调试与基准测试
├── Cargo.toml                 # Rust 依赖与构建配置
├── pyproject.toml             # Python 构建配置
├── build_wheel.sh             # 一键构建脚本
└── README.md                  # 本文件
```

---

## 🔧 从源码构建

```bash
# 克隆
git clone https://github.com/YJ-Niu/rsnumpy.git
cd rsnumpy

# 创建虚拟环境（推荐使用 uv）
uv venv .venv --python 3.11
source .venv/bin/activate
uv pip install maturin

# 构建并安装
bash build_wheel.sh           # release 模式
# bash build_wheel.sh --debug # debug 模式
```

构建脚本会自动：
1. 检测 `.venv` 中的 Python
2. 调用 `maturin build --release` 构建
3. 生成 `.whl` 到 `wheelhouse/`
4. 用 `uv pip install --no-deps` 安装到 `.venv`

### 手动使用 maturin

```bash
maturin develop --release         # 开发模式，实时生效
maturin build --release -o wheelhouse/
uv pip install --no-deps wheelhouse/rsnumpy-*.whl
```

### 清理

```bash
rm -rf target/ wheelhouse/
```

---

## 📖 使用示例

### 基础数组操作

```python
import rsnumpy as np

a = np.array([1.0, 2.0, 3.0])
b = np.zeros((2, 3))         # 2x3 全零
c = np.ones((2, 2))           # 2x2 全一
d = np.eye(3)                 # 3x3 单位矩阵
e = np.arange(0, 10, 2)       # [0, 2, 4, 6, 8]
f = np.linspace(0, 1, 5)      # [0, 0.25, 0.5, 0.75, 1.0]

print(a.shape, a.ndim, a.size, a.dtype)
```

### 数学函数

```python
a = np.array([0.0, np.pi / 2, np.pi])
print(np.sin(a))     # 三角函数
print(np.exp(a))     # 指数
print(np.log(a + 1)) # 对数
print(np.sqrt(a))    # 平方根

# nan-safe 归约
b = np.array([1.0, np.nan, 3.0])
print(np.nanmean(b)) # 2.0
```

### 数组操作

```python
a = np.array([[1, 2, 3], [4, 5, 6]])

# 变形
print(a.reshape((3, 2)).tolist())

# 转置
print(a.transpose().tolist())

# 多维切片
print(a[0:2, 1:3].tolist())  # [[2, 3], [5, 6]]

# 拼接
print(np.concatenate([a, a], axis=0).shape)  # (4, 3)
print(np.vstack([a, a]).shape)               # (4, 3)
```

### 统计函数

```python
a = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
print(np.sum(a), np.mean(a), np.std(a), np.var(a))
print(np.max(a), np.min(a), np.argmin(a), np.argmax(a))

# 分位数
print(np.percentile(a, [25, 50, 75]))
print(np.median(a))
```

### 线性代数

```python
A = np.array([[1.0, 2.0], [3.0, 4.0]])
B = np.array([[5.0, 6.0], [7.0, 8.0]])

print(A @ B)                      # 矩阵乘
print(np.linalg.det(A))           # 行列式
print(np.linalg.inv(A).tolist())  # 逆矩阵
print(np.linalg.solve(A, [1, 1])) # 求解 Ax = b
print(np.linalg.norm(A))          # Frobenius 范数

# 矩阵分解
U, S, V = np.linalg.svd(A)
Q, R = np.linalg.qr(A)
eigvals, eigvecs = np.linalg.eig(A)
```

### 随机数

```python
# 现代 API（推荐）
rng = np.random.default_rng(seed=42)
print(rng.random(5).tolist())         # 均匀分布 [0, 1)
print(rng.normal(0, 1, 5).tolist())   # 正态分布
print(rng.integers(0, 10, 5).tolist()) # 整数

# 经典 API
np.random.seed(0)
print(np.random.rand(3).tolist())
print(np.random.randn(3).tolist())
```

> **可复现性**：相同 `seed` 在任何机器、任何线程数下产生相同结果。Rust 层使用固定分块 + `splitmix64` 派生各分块独立种子。

### FFT

```python
x = np.array([1.0, 0.0, 0.0, 0.0])
spectrum = np.fft.fft(x)
recovered = np.fft.ifft(spectrum)

# 实数输入
r_spectrum = np.fft.rfft(x)
recovered_r = np.fft.irfft(r_spectrum, n=4)
```

### 零拷贝缓冲协议

```python
import rsnumpy as np

# float64 数组支持 memoryview 零拷贝
a = np.array([1.0, 2.0, 3.0, 4.0])
mv = memoryview(a)
print(mv.format)   # 'd' (f64)
print(mv.shape)    # (4,)
print(mv.readonly) # True

# numpy 也能通过缓冲协议零拷贝读取
import numpy as np_np
arr = np_np.asarray(a)  # 零拷贝，dtype=float64

# int/bool 等 dtype 通过 __array_interface__ bytes 回退
ia = np.array([1, 2, 3], dtype='int64')
arr_int = np_np.asarray(ia)  # dtype=int64，通过 bytes 副本
```

### 大数组处理工具

```python
import rsnumpy as np

# 分块处理：避免一次性加载全部数据导致 OOM
large = np.arange(10_000_000)
def my_func(chunk):
    return np.sum(chunk)
result = np.chunked_apply(my_func, large, chunk_size=1_000_000)

# 内存映射：从文件读取大型数组
with open('data.bin', 'wb') as f:
    f.write(np.arange(1_000_000).tobytes())
arr = np.mmap_array('data.bin', dtype='float64', shape=(1_000_000,))

# 惰性求值：链式操作延迟到需要时计算
lazy = np.LazyArray(np.arange(1000.0))
result = lazy.sqrt().multiply(2).add(1).compute()
```

### 变长元素

```python
import rsnumpy as np

# 变长字符串（比 object dtype 更高效）
vsa = np.VarStringArray(['hello', 'world', 'foo'])
print((vsa + '!').to_list())     # ['hello!', 'world!', 'foo!']
print(vsa.upper().to_list())     # ['HELLO', 'WORLD', 'FOO']
print(vsa.replace('o', '0').to_list())

# 大整数（任意精度，超出 f64 范围也不丢失）
big = np.BigIntArray([2**60, 2**70, 2**80])
print(big[0])                    # 1152921504606846976
print(big.sum())                 # 正确求和

# 智能整数数组：自动选择合适的存储方式
arr = np.smart_int_array([1, 2, 3])           # ndarray float64
arr = np.smart_int_array([1, 2**60, 3])       # BigIntArray（自动）
```

### 多项式

```python
import rsnumpy as np

# 系数从高到低（NumPy 兼容）
p = np.polynomial.Poly([1, -3, 2])  # x² - 3x + 2
print(p(2))               # 0
print(p.roots().tolist()) # [2.0, 1.0]

# 运算
q = np.polynomial.Poly([1, -1])  # x - 1
print(p + q)              # Poly([1, -2, 1])
print(p * q)              # Poly([1, -4, 5, -2])

# 拟合
x = np.array([0, 1, 2, 3, 4])
y = np.array([1, 2, 5, 10, 17])
coef = np.polynomial.polyfit(x, y, 2)
```

### 文件 I/O

```python
import rsnumpy as np

# .npy
a = np.array([1, 2, 3])
np.save('data.npy', a)
b = np.load('data.npy')

# 文本
np.savetxt('data.txt', a, fmt='%d', delimiter=',')
data = np.loadtxt('data.txt', delimiter=',', dtype=int)

# .npz（多个数组）
np.savez('multi.npz', a, b, c=a)
loaded = np.load_npz('multi.npz')
print(loaded['arr_0'].tolist())
print(loaded['c'].tolist())
```

### 线程安全工具

```python
import rsnumpy as np
import threading

# 线程安全副本（比加锁快 100x）
data = np.arange(1_000_000)
safe_copy = np.threadsafe_copy(data)

# 线程安全视图（带锁的共享包装）
safe_view = np.threadsafe_view(data)
with safe_view.locked() as arr:
    arr[0] = 100

# ndarray.copy 支持 thread_safe 参数
arr = data.copy(thread_safe=True)
```

### datetime64 / timedelta64

```python
import rsnumpy as np

d = np.datetime64('2024-01-01')
delta = np.timedelta64(7, 'D')
print(d + delta)  # numpy.datetime64('2024-01-08')

# 日期范围
days = np.arange(np.datetime64('2024-01-01'),
                 np.datetime64('2024-01-05'),
                 np.timedelta64(1, 'D'))
```

---

## ❓ 常见问题

**Q1: `ModuleNotFoundError: No module named 'rsnumpy'`**

A: 需要先构建并安装：`bash build_wheel.sh`

**Q2: 编译报错 `error: linker not found`**

A: 安装 Xcode Command Line Tools（macOS）：`xcode-select --install`

**Q3: 性能是否优于 NumPy？**

A: 多数逐元素与归约操作（`sin`/`exp`/`sum`/`std`/`cumsum` 等）在中大规模数组上快于 NumPy（1.3x–5x）；`matmul` 在 macOS 上分派到 Accelerate BLAS，与 NumPy 基本持平。小数组因跨语言调用开销可能略慢。

**Q4: 是否支持 GPU？**

A: 当前版本仅支持 CPU。

**Q5: 与 NumPy 100% 兼容吗？**

A: 已对齐 NumPy 2.5.1，覆盖 ~140 个常用 API。但一些高级或实验性 API 暂未实现。

---

## 🤝 贡献

欢迎贡献代码！请遵循：

1. Fork 项目并创建新分支
2. 所有计算逻辑应在 Rust 层实现（`src/`），Python 层只做参数转发（`python/rsnumpy/`）
3. 在 `debug/` 添加对应的调试或基准测试
4. 提交前运行：
   - `cargo fmt --all -- --check`
   - `cargo clippy --all-targets -- -D warnings`
   - `.venv/bin/python test/run_test.py`
5. 提交 PR

---

## 📄 许可

MIT License — 详见 [LICENSE](LICENSE)

---

## 🔗 链接

- [Release Notes](RELEASE_NOTES.md)
- [Optimization List](OPTIMIZATION_LIST.md)
- [Issue Tracker](https://github.com/YJ-Niu/rsnumpy/issues)
