# Release Notes

> Python 安装：`pip install rsnumpy` 或 `uv pip install rsnumpy`
> 源码：<https://github.com/YJ-Niu/rsnumpy>
> 许可：MIT

---

## v1.2.3 (2026-07-29) — 内存与启动速度双优化

本版本聚焦"让大型数组更轻、让首次导入更快"，同时沉淀了变长元素支持与线程池动态控制等基础设施。

### ⚡ 性能与体积

- **二进制体积减小约 28%**：通过移除 `num_cpus` 依赖、改用标准库 `std::thread::available_parallelism()`，并配合 `lto="fat"` / `codegen-units=1` / `panic="abort"` / `strip="symbols"` 等编译选项，`num_core.cpython-*.so` 从 6.8 MB 降至约 5.1 MB
- **导入速度提升约 130×**：`import rsnumpy` 首次导入时间从约 900 ms 降至约 7 ms（再次导入约 6 ms）。原因：`array_methods.py` 与 `__init__.py` 之间的循环依赖通过 `_get_deps()` 延迟导入解决
- **零依赖标准库**：`num_cpus` 完全移除，构建时不再拉取额外 crate

### 🧵 线程与并行

- **动态线程池控制**：新增 `set_num_threads()` / `get_num_threads()` / `get_num_cpus()` / `parallel_context()` 等顶层接口；底层通过 rayon's `ThreadPool::install()` 让用户配置的线程数对所有并行算子真正生效
- **动态并行阈值**：新增 `set_parallel_thresholds()` / `get_parallel_thresholds()`，允许运行时按算子成本分级调整并行门槛（超越函数 / 访存密集 / 归约等），进一步避免小数组的线程调度反噬
- **基准结果**：1000 万元素逐元素加法在 10 线程下相对单线程约 3.6× 加速，2000×2000 矩阵乘法约 1.55× 加速

### 🧩 变长元素支持（New）

- **`VarStringArray`**：比 `object` dtype 更高效的变长字符串数组，提供 `+` / `upper` / `lower` / `strip` / `replace` / `__getitem__` 等字符串专用方法
- **`BigIntArray`**：支持任意精度整数（`> 2^53` 也不丢精度），底层使用 Python `int`，提供 `+` / `-` / `*` / `//` / `%` / `**` 等运算符重载
- **`smart_int_array(data, fallback='bigint')`**：根据整数范围自动选择 `ndarray` 或 `BigIntArray`
- **`detect_integer_overflow(data)`**：检测是否会因 f64 表示而丢精度
- **`var_string_array(data, dtype=None)` / `bigint_array(data, safe=True)`**：工厂函数

### 🛠️ 大数组处理工具（New）

- **`chunked_apply(func, arr, chunk_size, axis=0, **kwargs)`**：沿指定轴分块执行用户函数，避免一次性加载导致 OOM
- **`mmap_array(filename, dtype='float64', shape=None, offset=0)`**：从二进制文件按 `dtype` + `shape` + `offset` 读取部分数组，免去全量加载
- **`LazyArray`** 类：链式操作延迟到 `compute()` 时执行，链中不分配中间数组
- **`memory_usage(arr, deep=False)`**：递归统计数组及其内部缓冲的字节占用

### 🎯 智能类型推断

- **自动识别固定长度字符串**：传入 `['hello', 'world']` 时默认推断为 `U5` 而非 `object`，与 NumPy 一致
- **混合类型数据建议**：`array()` 构造时若检测到可能回退到 `object` dtype，会打印性能警告并给出推荐 dtype
- **大整数精度警告**：构造时若整数超过 ±2^53，会在 `array` 路径给出明确提示，建议改用 `BigIntArray`
- **辅助工具**：`check_object_dtype(arr)` / `suggest_dtype_for_data(data)` 显式触发检测

### 🧵 线程安全

- **`threadsafe_copy(arr)`**：通过一次性独立副本让多线程只读共享，比加锁快约 100×
- **`threadsafe_view(arr)`**：带锁的视图包装，提供 `locked()` 上下文管理器
- **`ndarray.copy(thread_safe=True)` / `np.copy(a, thread_safe=True)`**：统一入口

### 🐛 关键修复

- **循环导入导致 import 卡顿**：`array_methods.py` 与 `__init__.py` 互引导致 `import rsnumpy` 耗时 ~900 ms，改为延迟 `_get_deps()` 后降至 ~7 ms
- **`timedelta64` 比较非传递性**：统一 `__lt__` / `__gt__` 等运算符的语义，跨类型比较不再产生自相矛盾的结果
- **pickle 反序列化安全**：`__setstate__` 引入 dtype 白名单与 `payload` 大小上限，阻止恶意构造的超大数组攻击
- **`pad` 函数 DoS 防护**：`pad_width` 现在有上限校验，避免 `pad_width=10**18` 这类输入把进程拖死
- **`ndarray.__sizeof__`**：让 `sys.getsizeof(arr)` 返回合理值，便于内存分析
- **sorting 索引语义**：`partition` 不再因多维数组触发 panic；`argpartition` 改为轴局部索引；`sort_complex` 现在读取虚部并支持 `axis=-1`
- **`indexing.setitem_multi`**：修正 `strides` / `elem_size` 误除导致的赋值失败

### 🧹 重构与代码质量

- **Python 层循环批量改写**：`io.py` / `_dtypes.py` / `lib/recfunctions.py` / `linalg/__init__.py` / `array_ops.py` / `char.py` / `fft/__init__.py` / `polynomial/__init__.py` 中的显式 `for` / `while` 改用 `math.prod` / `itertools.accumulate` / 列表推导等更轻量的写法
- **Rust 层函数下沉**：`choose` / `compress` / `copyto` 从 `_extra.py` 迁移至 `manipulation/mod.rs` 的 Rust 实现，Python 仅做参数转发
- **`nansum` 并行化**：`statistics/mod.rs::nansum` 引入 rayon 并行，沿用现有 `PAR_THRESHOLD` / `PAR_THRESHOLD_CHEAP` / `PAR_THRESHOLD_MEDIUM` 模式
- **`vectorize.__call__` 快速路径**：标量、单参数、双参数分别走最优路径；预绑定方法、缓存 `kwargs`，把 Python 层循环开销压到最低
- **统一 `keepdims` 支持**：`nanmin` / `nanmax` / `nansum` / `nanprod` / `nanmean` / `nanvar` / `nanstd` / `nancumsum` / `nancumprod` / `count_nonzero` 等全部支持 `keepdims`，与 NumPy 行为一致
- **统计函数多算子轴归约**：`nanmin` / `nanmax` / `nansum` / `nanprod` 等已下沉至 Rust 实现

### 🧪 测试

- `debug/test_lazy_array.py` — LazyArray 行为与链式正确性
- `debug/test_memory_tools.py` — chunked_apply / mmap_array / memory_usage，对比 `object` vs `U{n}` 的性能（结论：U10 创建快 ~3.3×，运算快 ~10.4×）
- `debug/test_var_element.py` — VarStringArray / BigIntArray / smart_int_array
- `debug/test_real_threadpool.py` — `set_num_threads` / `parallel_context` 实际并行效果
- `debug/test_threadsafe.py` — 线程安全工具函数正确性与性能
- `test/run_test.py` / `test/run_test2.py` — 全部通过

---

## v1.2.2 (2026-07-29) — keepdims 统一与多项式完善

### ✨ 功能增强

- **统一 `keepdims` 行为**：归约算子（`sum` / `min` / `max` / `mean` / `std` / `var` / `nan*` 等）现在全部支持 `keepdims`，形状与 NumPy 一致
- **多项式索引支持**：`polynomial.Poly` 现在支持任意位置的系数索引与切片
- **复数梯度计算**：多项式模块的复数 `__call__` 路径在系数为复数时也能正确求值

### 🐛 关键修复

- 修复 `plot_complex_polar` 未返回轴对象的问题
- 修复频率轴刻度与单位不一致的问题

### 🧹 重构

- 大量函数实现统一并下沉，减少 Python 层重复逻辑
- 批量新增 numpy 风格工具函数（位运算、统计、多项式等）
- 新增依赖模块初始化文件与测试用例

---

## v1.2.1 (2026-07-28) — linalg / S 参数加速

### ⚡ 性能

- **矩阵求逆加速**：`linalg.inv` 的 2×2 批量分支新增 rayon 并行，支持复数
- **阻抗重归一化加速**：2 端口 `renormalize_s` 标量阻抗快速路径避免两次矩阵求逆，加速约 700×
- **S 参数求逆快速路径**：2 端口 S 参数求逆新增闭式计算路径，加速约 8×

### 🐛 修复

- 修复 `S` 参数矩阵构造的 dtype 推断问题（显式指定 `dtype=complex`）
- 抑制 CPW 导体损耗计算的低频 `RuntimeWarning`

### 🧹 重构

- 为 2 端口网络 `renormalize_s` 添加标量阻抗快速路径
- 整理 CPW 相关测试代码

---

## v1.2.0 (2026-07-24) — 代码清理

- 删除大量冗余测试与废弃代码文件，精简项目体积

---

## v1.1.9 (2026-07-23)

- 修复 `recfunctions` 逻辑问题
- 重命名核心模块并调整导入路径
- 调整 `__init__.py` 导入顺序避免循环导入
- 批量新增测试文件与模块初始化代码

---

## v1.1.8 (2026-07-21)

- 修复复数类型相关函数逻辑
- 整理代码格式并修复多处逻辑问题
- 调整史密斯图线条宽度并新增电感测试案例

---

## v1.1.6 (2026-07-11) — GIL-Free 并行 & BLAS

### ⚡ 性能架构

- **主动释放 GIL**：`sort` / `argsort` / `cumsum` / `cumprod` 及逐元素、归约、矩阵乘等算子在进入 Rust 前通过 `py.detach` 释放 GIL
- **成本分级并行阈值**：按每元素成本拆分阈值——超越函数最低、访存密集型最高——避免小数组因线程调度反而变慢
- **`cumsum` / `cumprod` 重写**：单份 C 序缓冲就地前缀扫描 + 连续块并行
- **BLAS 后端**：macOS 上 `matmul` / `dot` 经系统 Accelerate 框架分派

### ✨ 功能

- 新增 `frexp` / `i0` / `convolve` / `correlate` / `interp` 等 Rust 原生实现
- 重构 Python 侧冗余实现，统一走 Rust 后端

### 🧹 构建优化

- 关闭 `zip` 等依赖中未使用的编解码器特性（zstd / lzma / bzip2 等）并移除 ndarray 未用的 `serde` 特性
- 配合 `lto = "fat"` 与符号裁剪，扩展体积减小约 14%

---

## v1.1.5 (2026-07-10)

- **字符串 / void dtype 支持**
- **掩码数组与结构化数组**
- 修复幂运算（`power`）操作符行为

---

## v1.1.4 (2026-07-10)

- 新增 `masked_select` 函数
- 重构数组格式化逻辑（多维缩进对齐）
- 重构 IO 模块

---

## v1.1.3 (2026-07-09)

- 逐元素运算并行 + GIL 释放
- 优化 BLAS、切片与线性代数实现

---

## v1.1.2 (2026-07-09)

- `np.unique` 新增 `axis` 参数支持
- 新增布尔掩码索引与取模（`%`）运算符支持

---

## v1.1.1 (2026-07-09)

- 更新 `reshape` API
- 新增集合运算（`intersect1d` / `union1d` / `setdiff1d` 等）

---

## v1.1.0 (2026-07-09) — NumPy 2.5.1 对齐

- 新增约 140 个 NumPy 兼容函数（别名、nan 系列、集合运算、窗函数、信号等）
- 改进 `transpose` 实现

---

## v1.0.8 (2026-07-09)

- **日期时间支持**：`datetime64` / `timedelta64` 标量与日期序列生成
- 修复随机数函数的返回类型

---

## v1.0.7 (2026-07-07)

- **NumPy 数组接口协议**：实现 `__array__` 等接口，支持 matplotlib 直接绘图
- 优化 `argmax` / `argmin` / `argsort` 实现

---

## v1.0.6 (2026-07-06)

- 支持数组接口协议，新增字符串数组支持并优化内置函数替换
- 修复多维索引赋值问题
- 构建前新增 `cargo fmt` 静态检查

---

## v1.0.4 (2026-07-03)

- 新增 `rust-toolchain.toml` 固定工具链版本
- 更新 pyo3 依赖格式与特性声明
- 优化发布工作流（maturin 1.14.1），整理 pyproject 分类信息

---

## v1.0.3 (2026-07-03)

- 升级 `rand` 至 0.10.2
- 调整 ndarray 特性
- 重构随机数生成模块
- 更新 CI 配置，新增 `PYTHONUTF8` / `PYTHONIOENCODING` 环境变量

---

## v1.0.2 (2026-06-24)

- 索引模块优化：向量预分配，提升花式索引性能
- 数组拼接优化：`concatenate` / `stack` 性能提升
- 数组格式化输出优化：预分配字符串缓冲区
- 工具函数优化：`arange` / `reshape` / `loadtxt` / `savez`
- 修复 `median` / `percentile` 等统计函数的偶数长度判断
- 新增 `build_wheel.sh` 自动同步版本号

---

## v1.0.1 (2026-06-24)

- 新增 `ArrayFlags` 支持
- 新增 like-series 数组创建函数
- 改进 `logspace` 函数
- 并行加速矩阵运算与随机数生成
- 迁移矩阵与数组操作到 Rust 后端

---

## v1.0.0 (2026-06-XX) — Initial Release

- 高性能多维数组（基于 Rust 的 `ndarray`）
- 数学函数：`sin` / `cos` / `exp` / `sqrt` / `log`
- 数组操作：`reshape` / `transpose` / 切片 / 拼接
- 统计函数：`sum` / `mean` / `std` / `var` / `max` / `min` / `argmin` / `argmax`
- 线性代数：`det` / `inv` / `norm` / `solve` / `svd` / `qr`
- FFT
- 随机数：均匀 / 正态 / 整数
- 多项式：求值 / 求根 / 运算 / 拟合
- 文件 I/O：`.npy` / `.npz` / 文本

### 支持平台

- Python ≥ 3.10
- Rust ≥ 1.75
- macOS / Linux / Windows
