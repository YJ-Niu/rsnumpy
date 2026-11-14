# rsnumpy 的 NumPy 兼容性验证（以 NumPy 官方测试为参照基准）

> **方向**：这里的**被测对象是 rsnumpy**；NumPy 的官方测试用例是**参照基准
> （ground truth）**。也就是说——「拿 NumPy 自己的测试跑一遍 rsnumpy，量出它
> 兼容到多少」，**不是**「在 rsnumpy 里测试 NumPy」。

rsnumpy 定位是「重新实现 NumPy 的 NumPy 兼容数组库」（见仓库 README：对齐 NumPy
2.5.1、覆盖 ~140 个常用 API）。要客观度量它到底兼容到多少，最有力的参照就是
**NumPy 自己的测试套件**。这里拷入 NumPy `_core/tests` 的 4 个核心测试文件，
通过一个垫片 conftest 把 `import numpy` 解析到 rsnumpy，于是 NumPy 的每条用例
都在**验证 rsnumpy** 的行为。

## 文件

| 文件 | 来源 | 作用 |
|------|------|------|
| `conftest.py` | 本仓库 | 垫片：让 NumPy 测试 verbatim 跑在 rsnumpy 上（见下） |
| `run_compat.py` | 本仓库 | 崩溃隔离 runner：每测试独立子进程，产出兼容性报告 |
| `test_array_coercion.py` | NumPy | 数组构造 / 类型转换（943 行） |
| `test_indexing.py` | NumPy | 索引 / 切片（1728 行） |
| `test_numeric.py` | NumPy | 数值运算（4331 行） |
| `test_scalarmath.py` | NumPy | 标量数学（1180 行） |

这 4 个文件是 NumPy `_core/tests` 里最核心、最偏「纯数组语义」的部分
（避开 memmap / dlpack / records / strings 等 rsnumpy 无对应实现、且依赖大量
NumPy 内部 C helper 的类别）。

## 垫片 conftest.py 做了什么

NumPy 测试假设一个完整的 NumPy 环境。垫片把这些假设映射到 rsnumpy，让用例
**逐字不改**就能跑在 rsnumpy 上：

1. **顶层别名**：`sys.modules["numpy"] = rsnumpy`，让 `import numpy as np` 解析到
   rsnumpy（rsnumpy 顶层名字全部可用）。
2. **真实子模块映射**：`numpy._core` / `numpy.random` / `numpy.ma` /
   `numpy.exceptions` 等指向**真实的** rsnumpy 子模块（避免按 `numpy.` 名重新导入
   导致相对导入断到桩上）。
3. **NumPy 专属子模块打桩**：`numpy.testing`（断言函数）、`numpy._utils._pep440`、
   `numpy._core._multiarray_umath` / `_multiarray_tests` / `_rational_tests`
   （C helper）、`numpy.testing._private.hypothesis_helpers`（hypothesis 未装 →
   相关测试 skip）等。每个桩都挂到真实父模块上，使 `import a.b.c` 与
   `import a.b.c as x` 都能解析。
4. **源码级跳过**：依赖 C helper（`ncu.` / `array_indexing` / `rational`）或 rsnumpy
   缺的特性（float16 / longdouble / datetime64 / object）的用例被 skip——这样剩下
   的失败都是**真实的 rsnumpy gap**，不是垫片限制。
5. **少量用例文件的模块级构造适配**（垫片覆盖不到的模块加载期构造）：
   - `test_array_coercion.py`：`scalar_instances()` 生成器改防御式（rsnumpy 建不出的
     标量跳过该 param）；删模块级 NaT timedelta/datetime parametrize。
   - `test_numeric.py`：`test_NaT_propagation` 改 skip（模块级 NaT 构造）。
   - `test_scalarmath.py`：`objecty_things` 的 object 数组构造改防御式。

这些适配都是**收窄到 rsnumpy 支持范围**，不改变用例本身的断言语义。

## runner run_compat.py

每个用例在**独立子进程**里跑（默认 8 路并行），这样某条用例让 rsnumpy C 扩展
abort / segfault 时只影响它自己，不会拖垮整个 run。分类：

- **passed** — rsnumpy 正确处理了这条 NumPy 用例
- **failed** — 真实的 rsnumpy gap（结果错 / 缺 API / 行为不符）
- **skipped** — 依赖 NumPy C helper 或 rsnumpy 缺的特性（见 conftest）
- **crashed** — rsnumpy C 扩展 abort / segfault（robustness bug，需单独跟踪）

### 运行

```bash
# 完整扫描（约 3-4 分钟）
python tests/numpy/run_compat.py

# 只跑部分文件
python tests/numpy/run_compat.py test_indexing.py test_numeric.py
```

报告写到 stdout（可重定向到 `compat_report.txt`）。

## 结果怎么读

`passed/total` 就是 rsnumpy 对这 4 类核心 NumPy 语义的**兼容率**。**failed**
列表是 rsnumpy 待补的 gap 清单（按文件/用例名），**crashed** 列表是 C 扩展健壮性
bug（比功能 gap 更严重——会直接崩进程）。随 rsnumpy 迭代，这两个列表应逐步缩短，
兼容率应上升。

> **与构建的关系**：这是**非门禁**的兼容性度量。rsnumpy 的**功能门禁**是仓库
> `tests/` 下的核心套件（buffer / arithmetic / creation / indexing / reduction /
> comparison，稳定全绿）。本目录（`tests/numpy/`）在构建里只作为 rsnumpy 兼容性
> 的报告产出，不作为通过/失败判定。
