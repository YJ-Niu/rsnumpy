# rsnumpy 项目开发规则

## 项目概述

rsnumpy 是一个使用 **Rust + PyO3** 开发的高性能 Python 数值计算库，提供与 **NumPy** 兼容的 API 接口。

- **Rust 层** (`src/`): 负责底层核心实现，包括数组操作、算术运算、线性代数、FFT、排序统计等，底层数据恒为 `Array<f64, IxDyn>`（ndarray 0.17.2）
- **Python 层** (`python/rsnumpy/`): 负责公开 API 接口，方法参数默认值要全面，代理调用 Rust 底层实现，包含 dtype 追踪、结构化数组、掩码数组等扩展功能
- **构建工具**: 使用 `maturin` 构建 wheel，通过 `build_wheel.sh` 脚本构建并安装到 `.venv`

## 开发环境

- Python 虚拟环境: `.venv` (使用 uv 创建)
- 构建命令: `./build_wheel.sh --debug` (调试模式) 或 `./build_wheel.sh` (release 模式)
- Rust 工具链: 由 `rust-toolchain.toml` 固定为 stable，当前为 edition 2024

## 关键约定

### 1. Debug 目录

- 所有调试、分析、验证、测试代码及生成的结果必须放在 `/Users/user/Desktop/rust_project/rsnumpy/debug/` 目录中
- 禁止在项目根目录或其他位置创建零散的测试文件
- 调试时，添加一些debug信息，如打印数组形状、dtype、内存布局等，也要判断数据是否正确加载

### 2. 代码修改范围

- **可以修改**: `rsnumpy` 项目内的代码 (`src/`, `python/rsnumpy/`, `pyproject.toml`, `Cargo.toml`, `test/test_rf/networkx`, `test/test_rf/scipy` 等)
- **禁止修改**: 用户测试代码、`.venv` 环境中的任何代码
- 修改后使用 `build_wheel.sh` 构建并安装到 `.venv` 进行测试

### 3. 修复优先级

- 优先在 **Rust 层** (`src/`) 实现底层修复
- 然后在 **Python 层** (`python/rsnumpy/`) 补全和优化方法接口
- 确保 Python 接口的默认参数全面，与 NumPy 行为一致

### 4. 依赖库处理

- **禁止**安装 `numpy` 或其他第三方 Python 数值库（如 `scipy` 的核心功能）
- 项目自身即为 numpy 兼容库，应完善自身实现而非引入 numpy
- `test/test_rf/` 目录下的 networkx/scipy 用于测试兼容层，可按需更新

### 5. 代码风格

- Rust 代码: 遵循 `rustfmt` 和 `clippy` 规范（构建脚本会自动检查）
- Python 代码: 遵循 PEP 8 规范
- 提交前确保 `cargo fmt --all -- --check` 和 `cargo clippy --all-targets -- -D warnings` 通过
- edition 2024 要求：`unsafe fn` 内的 unsafe 操作仍须显式 `unsafe {}` 块（unsafe_op_in_unsafe_fn）

## 6. 修复后的构建与测试

- 每次修改后，立即执行 `./build_wheel.sh` 重新构建并安装。
- 运行项目提供的测试用例（`python test/run_test.py`、`python test/run_test2.py`）确保修复有效。若测试失败，继续修改直到通过。

## 7. 长期优化原则（非强制）

- 当发现性能瓶颈来自 Python 层循环时，应评估将其迁移到 Rust 实现的可行性，但**本次修复不必强制执行**。
- 优先释放 GIL（`Python::allow_threads`）以提升并行计算性能。

## 8. 缓冲协议架构

rsnumpy 通过双层缓冲协议实现零拷贝数据共享，同时保证 dtype 精度：

- **内层 Rust 层** (`_core.ndarray`)：实现 PEP 3118 的 `__getbuffer__`/`__releasebuffer__`，直接暴露底层 `Array<f64, IxDyn>` 的连续内存，支持 shape/strides/format='d'/readonly。

- **外层 Python 层** (`rsnumpy.ndarray`)：实现 PEP 688 的 `__buffer__`/`__release_buffer__`（Python 3.12+），转发到内层 `_array` 的缓冲协议。**仅对 float64 dtype 暴露缓冲**——因为底层存储恒为 f64，若对 int/bool 等 dtype 也暴露 f64 缓冲，numpy 会优先按缓冲解读导致 dtype 失真。

- **回退路径**：int/bool/datetime64 等 dtype 不暴露缓冲，消费方自动回退到 `__array_interface__`（含 bytes 副本的 dtype 精确表示）。

- **兼容性**：Python 3.11 及更早版本不识别 `__buffer__`，`memoryview()` 会失败并自动回退到 `__array_interface__` bytes，行为安全。

## 9. 已知限制

- **f64 唯一存储**：底层数组恒为 `Array<f64, IxDyn>`，int64/bool/datetime64 等 dtype 仅在 Python 层通过 `_dtype` 追踪。
- **int64 精度丢失**：大于 `2^53` 的整数会因 f64 存储限制而丢失精度，即使 `typestr` 报 `<i8`。
- **datetime64/timedelta64**：内部存储为 f64 纪元值，`dtype.kind` 返回 `'f'`，日期零拷贝需未来引入独立整型存储。

## 项目结构

```
rsnumpy/
├── src/                    # Rust 源码
│   ├── lib.rs              # 库入口，PyO3 模块注册，NdArray 定义及缓冲协议
│   ├── arithmetic/         # 算术运算（加减乘除、幂、取模等）
│   ├── bitwise/            # 位运算
│   ├── buffer/             # 缓冲互操作（bytes↔数组、__array_interface__）
│   ├── creation/           # 数组创建（array、zeros、ones、empty 等）
│   ├── fft.rs              # FFT 变换
│   ├── formatting/         # 数组格式化输出
│   ├── indexing.rs         # 索引操作
│   ├── io.rs               # IO 操作（npz 读写）
│   ├── linalg.rs           # 线性代数（matmul、solve、inv、eig 等）
│   ├── logic/              # 逻辑运算
│   ├── manipulation/       # 数组操作（reshape、transpose、concatenate 等）
│   ├── mathematics/        # 数学函数（sin、cos、exp、log 等）
│   ├── poly/               # 多项式
│   ├── random.rs           # 随机数
│   ├── searching/          # 搜索操作
│   ├── setops/             # 集合操作
│   ├── sorting/            # 排序操作
│   └── statistics/         # 统计函数（mean、std、sum、min、max 等）
├── python/                 # Python 源码
│   └── rsnumpy/            # Python 包
│       ├── __init__.py     # 包入口，导出核心 API，公开 ndarray 类
│       ├── _core/          # 内部模块
│       ├── _dtypes.py      # dtype 定义
│       ├── _extra.py       # 额外功能
│       ├── array_methods.py # 数组方法
│       ├── array_ops.py    # 数组操作
│       ├── char.py         # 字符操作
│       ├── exceptions.py   # 异常定义
│       ├── io.py           # IO 接口
│       ├── ma.py           # 掩码数组
│       ├── math_functions.py # 数学函数
│       ├── matlib.py       # 矩阵库
│       ├── rec.py          # 记录数组
│       ├── statistics.py   # 统计接口
│       ├── fft/            # FFT 模块
│       ├── lib/            # 库模块（recfunctions）
│       ├── linalg/         # 线性代数模块
│       ├── polynomial/     # 多项式模块
│       ├── random/         # 随机数模块
│       └── typing/         # 类型定义
├── test/                   # 测试目录
│   ├── run_test.py         # 主测试脚本
│   ├── run_test2.py        # 辅助测试脚本
│   └── test_rf/            # 兼容测试（networkx、scipy）
├── build_wheel.sh          # 构建脚本（含 fmt/clippy 门禁）
├── pyproject.toml          # 项目配置
├── Cargo.toml              # Rust 配置（含 macOS Accelerate BLAS）
├── rust-toolchain.toml     # Rust 工具链固定
└── README.md               # 项目文档
```
