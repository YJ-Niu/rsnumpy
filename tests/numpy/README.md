# numpy 测试套件跑在 rsnumpy 上（numpy-compatibility suite）

把 **numpy 官方测试**（`numpy/_core/tests` 的 4 个核心文件）拷进这里，直接
跑在 **rsnumpy** 上，用来度量 rsnumpy 的 numpy 兼容性——哪些 numpy 用例
rsnumpy 能正确通过、哪些是真实 gap、哪些会触发 C 扩展崩溃。

## 文件

| 文件 | 来源 | 说明 |
|------|------|------|
| `conftest.py` | 本仓库 | 垫片：让 numpy 测试文件 verbatim 可跑（见下） |
| `run_compat.py` | 本仓库 | 崩溃隔离 runner：每个测试独立子进程，产出兼容性报告 |
| `test_array_coercion.py` | numpy | 数组构造/类型转换（943 行） |
| `test_indexing.py` | numpy | 索引/切片（1728 行） |
| `test_numeric.py` | numpy | 数值运算（4331 行） |
| `test_scalarmath.py` | numpy | 标量数学（1180 行） |

这 4 个文件是 numpy `_core/tests` 里最核心、最偏「纯数组语义」的部分
（避开 memmap/dlpack/records/strings 等 rsnumpy 无对应实现、且依赖大量
C helper 的类别）。

## 垫片 conftest.py 做了什么

numpy 测试假设一个完整的 numpy。垫片把这些假设映射到 rsnumpy：

1. **顶层别名**：`sys.modules["numpy"] = rsnumpy`，让 `import numpy as np`
   解析到 rsnumpy（rsnumpy 的顶层名字全部可用）。
2. **真实子模块映射**：`numpy._core` / `numpy.random` / `numpy.ma` /
   `numpy.exceptions` 等指向 **真实的** rsnumpy 子模块（避免按 `numpy.` 名
   重新导入导致相对导入断到桩上）。
3. **numpy 专属子模块打桩**：`numpy.testing`（断言函数）、`numpy._utils._pep440`、
   `numpy._core._multiarray_umath`/`_multiarray_tests`/`_rational_tests`（C helper）、
   `numpy.testing._private.hypothesis_helpers`（hypothesis 未装→相关测试 skip）等。
   每个桩都挂到真实父模块上，使 `import a.b.c` 与 `import a.b.c as x` 都能解析。
4. **源码级跳过**：依赖 C helper（`ncu.`/`array_indexing`/`rational`）或 rsnumpy
   缺的特性（float16/longdouble/datetime64/object）的测试被 skip——这样剩下的
   失败都是**真实 rsnumpy gap**，不是垫片限制。
5. **少量测试文件适配**（非垫片能解决的模块级构造）：
   - `test_array_coercion.py`：`scalar_instances()` 生成器改防御式（rsnumpy 建不出的
     标量跳过该 param）；删模块级 NaT timedelta/datetime parametrize。
   - `test_numeric.py`：`test_NaT_propagation` 改 skip（模块级 NaT 构造）。
   - `test_scalarmath.py`：`objecty_things` 的 object 数组构造改防御式。

这些适配都是**收窄到 rsnumpy 支持范围**，不改变测试本身的断言语义。

## runner run_compat.py

每个测试在**独立子进程**里跑（8 路并行），这样某个测试让 rsnumpy C 扩展
abort / segfault 时只影响它自己，不会拖垮整个 run。分类：

- **passed** — rsnumpy 正确处理了这个 numpy 用例
- **failed** — 真实 rsnumpy gap（结果错 / 缺 API / 行为不符）
- **skipped** — 依赖 numpy C helper 或 rsnumpy 缺的特性（见 conftest）
- **crashed** — rsnumpy C 扩展 abort / segfault（robustness bug，需单独跟踪）

### 运行

```bash
# 完整扫描（约 3-4 分钟）
python test/numpy/run_compat.py

# 只跑部分文件
python test/numpy/run_compat.py test_indexing.py test_numeric.py
```

报告写到 stdout（或重定向到 `compat_report.txt`）。

## 结果怎么读

`passed/total` 就是 rsnumpy 对这 4 类 numpy 语义的兼容率。**failed** 列表是
待补的 gap 清单（按文件/测试名），**crashed** 列表是 C 扩展健壮性 bug
（比功能 gap 更严重——会直接崩进程）。随 rsnumpy 迭代，这两个列表应逐步缩短，
兼容率应上升。

> 注意：这是**非门禁**的兼容性报告（rsnumpy 尚在补 gap，大量 failed 属预期）。
> rsnumpy 的**功能门禁**用的是 `tests/` 里的核心套件（buffer/arithmetic/creation/
> indexing/reduction/comparison 等，稳定全绿）。
