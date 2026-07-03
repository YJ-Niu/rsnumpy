# Release Notes

- python可以使用pip install rsnnumpy安装，也可以使用uv pip install rsnumpy安装

## v1.0.2 (2026-06-24)

### 性能优化

- **索引模块优化**：向量预分配，提升花式索引性能
- **数组拼接优化**：`concatenate`、`stack` 等函数性能提升
- **数组格式化输出优化**：预分配字符串缓冲区、简化切片遍历逻辑
- **工具函数优化**：`arange`、`reshape` 等函数的预分配和边界处理
- **IO 函数优化**：`loadtxt`、`savez` 等函数的预分配和遍历逻辑

### 模块重构

- **随机数模块重构**：使用 const 初始化线程局部 RNG、优化形状解析预分配、使用 `div_ceil` 计算分块大小、简化采样类型转换
- **线性代数模块重构**：将 Python 绑定函数移至 PyO3 detach 闭包，避免 GIL 持有过长时间
- **统计函数优化**：简化 `polyfit`、`cov` 等计算的循环逻辑

### Bug 修复

- 修复 `median`、`percentile` 等统计函数的偶数长度判断逻辑
- 修复部分边界检查的冗余代码

### 构建优化

- 新增 `build_wheel.sh` 自动同步版本号的逻辑
- 优化构建配置，二进制大小略有优化

## v1.0.1 (2026-06-24)

### 功能增强

- **新增 ArrayFlags 支持**：提供数组标志位查询功能
- **新增 like-series 数组创建函数**：支持根据已有数组的属性创建新数组
- **改进 logspace 函数**：优化对数空间生成算法

### 性能提升

- **并行加速矩阵运算**：利用多线程并行化矩阵运算
- **随机数生成优化**：并行加速随机数生成
- **构建配置优化**：调整编译参数，提升构建效率

### 代码重构

- **迁移矩阵和数组操作到 Rust 后端**：将大量矩阵和数组操作从 Python 层迁移到 Rust 层实现
- **重构 matlib.py 的矩阵函数实现**：优化矩阵函数的代码结构

### 文档更新

- 更新 README 中 Python 版本支持要求至 3.10+
- 更新 README 文档细节并修复格式问题

---

## v1.0.0 (Initial Release)

### 核心功能

- **高性能多维数组**：基于 Rust 的 ndarray 实现，提供 NumPy 兼容的 API
- **数学函数**：支持 sin、cos、exp、sqrt、log 等常见数学函数
- **数组操作**：支持 reshape、transpose、切片、拼接等操作
- **统计函数**：支持 sum、mean、std、var、max、min、argmin、argmax 等
- **线性代数**：支持 det、inv、norm、solve、svd、qr 等矩阵运算
- **FFT**：支持快速傅里叶变换及其逆变换
- **随机数**：支持 uniform、normal、integers 等分布，兼容新旧两种 API
- **多项式**：支持多项式求值、求根、运算和拟合
- **文件 I/O**：支持 .npy、.npz 和文本文件的读写

### 优势

- **性能**：核心计算使用 Rust 实现，部分操作（sum、mean、dot 等）比 NumPy 快 1.2x-2.0x
- **类型安全**：利用 Rust 的强类型系统避免运行时错误
- **内存安全**：无数据竞争，无内存泄漏
- **完整 API**：覆盖 NumPy 常用功能

### 支持平台

- Python ≥ 3.10
- Rust ≥ 1.75
- macOS、Linux、Windows
