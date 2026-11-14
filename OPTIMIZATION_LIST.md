# rsnumpy 兼容性优化路线图

> **事实源**：`tests/test_config.txt`（build_wheel.sh 每次测试自动回写
> 各文件 passed/failed/skipped）。本文件只维护**优先级与路线**，
> 具体缺口以 test_config.txt 实时数据为准。
>
> 历史文档「针对 NumPy 不足的增值优化清单」（线程安全/智能 dtype 推断等）
> 已全部完成，见 git 历史；当前阶段聚焦 **NumPy 语义兼容** 与 **scicore
> 集成**。

---

## 📍 当前位置（1.2.10，2026-11-13）

- 核心门禁 7 文件 50/50 全绿；scicore test_scirs 集成测试 4/4 通过
- 4 个已知算术 gap（stft/welch/cubic_spline/interp_decomp）已关闭
- random MT19937 legacy 对齐 Phase 1/2 完成（uniform/randint/normal/
  exponential/uniform 逐位对齐）
- CI 三档：门禁核心套件 + 非门禁 NumPy 参照基准 + 冒烟
- 许可：MIT（PEP 639）+ NOTICE（NumPy BSD-3 测试代码归属）
- P0 零失败文件收尾完成：12 用例修好（6 类根因）、0 回归、门禁 50/50
  全绿；polynomial 41/41 全过（0/0/0 为历史残留）

非门禁参照基准当前失败总量约 9000+（以 ma/strings/nanfunctions 三大簇
为主），按下列批次推进。

---

## P0 快赢批（小失败数 + 0 收集诊断）

- [ ] `linalg_tests/test_linalg.py` **0 收集**——诊断完成：非收集错误，
      521 用例正常收集并运行，但大面积失败（多根因大簇，非单一根因）；
      需按函数（svd/eig/cholesky/…）逐簇归因，另立批次
- [x] `polynomial_tests/test_polynomial.py` **0 收集**——诊断完成：
      41/41 全过，0/0/0 为历史崩溃残留，无需修复
- [x] 零失败文件收尾：12 用例修好（function_base 1→0、defchararray 1→0、
      matrixlib 1→0、numerictypes 5→1、shape_base 2→1、numeric 真实基线
      22→18，修 4；旧配置 6 已过期）；剩余失败见根因备忘
- [ ] 中簇：indexing 29、api 41、unicode 57、arrayobject 18、
      scalarmath 22（逐簇归因）

## P1 单根因大解锁

- [ ] **浮点格式化引擎**（Dragon4 最短 repr：`1.` vs `1.0`、
      test_dragon4）——解锁 scalarprint 5 + scalarmath repr 类 +
      print 相关；根因已定位，Python 层 `format_float_positional`
- [ ] **NEP 50 弱类型提升**（test_nep50_promotions 152）——NumPy 2.x
      核心语义，Rust dtype 提升路径
- [ ] **casting 规则**（test_casting_floatingpoint_errors 182）——
      与浮点根因部分同源
- [ ] **random 收尾**（test_random 128 + randomstate_regression 17）——
      MT19937 对齐剩余窄 dtype 缺口
- [ ] **nanfunctions 归因**（854）——先诊断是否单一根因再定方案

## P2 大模块（工作量大、自包含）

- [ ] **masked arrays**（ma_tests 合计 ~4300）——最大单簇，专项推进
- [ ] **字符串函数**（test_strings 751）——char_op.rs 近期刚起步
- [ ] **float16 真 dtype**（39 skip + 连带）——特性级工作
      （当前全 f64 存储）

## P3 健壮性（C 层）

- [ ] test_multiarray 单文件 OOM（3.4GB）
- [ ] test_regression rc=-6 崩溃
- [ ] test_nditer / test_records 超时（>300s）

---

## 已知根因备忘（避免重复排查）

- `arr[0]` 返回 0-d ndarray 而非标量 → `repr` 含裸 `array` →
  `eval(repr)` 缺命名空间时 NameError（scalarmath test_float_repr
  同源）；标量化是全局行为差异，改动波及面大，高风险
- `np.linalg.svd` 的 s 形状不一致（2x2 返回 2D 对角矩阵、3x3 返回
  1D 向量）；_mvn_sample 已鲁棒化
- astype int→int 存储约定：有符号目标补码、无符号 <2^63 存值本身、
  uint64 ≥2^63 回落补码负值（f64 越界保护）
- 测试套件有执行状态依赖：前面测试的崩溃模式会改变后续测试结果，
  回归 diff 须核对失败断言是否同源
- `_core_tests` 整目录单进程必 OOM；对照验证方法 = 按文件新进程
  扫描 + 分块电池
- P0 零失败文件收尾已修根因（2026-11-13）：
  - concat 族把裸 tuple 当 shape（ndarray 构造器 np.zeros((2,3)) 契约）
    → array_ops 增 _as_data 数据语义助手（concatenate/stack/vstack/hstack）
  - fromiter：参数 dtype 遮蔽模块级 dtype 类（旧检查 TypeError 被吞失效）；
    list 未归一化 iterator；结构化/子数组改走 array()（ndarray 构造器把
    tuple 行摊平成额外维度）；子数组项嵌套形状递归校验（元素个数对
    首维大小而非维数）
  - view(..., type=matrix) 绕过 matrix.__init__ 未强制 2-D
  - 零尺寸 reshape 丢目标形状（array_ops.reshape 与 array_methods.reshape
    两条路径，均改经 zeros 按目标形状构造并继承元数据）
  - 子数组 dtype .type = 基类型（numpy 语义 = np.void）
  - 0-d × Python 标量返回 (1,)（ravel().tolist() 标量包 1 元素列表且
    ndim>1 分支漏 0-d；__mul__/__floordiv__/__rfloordiv__/clip 四处）
- 架构性限制（本批定位、暂不修）：
  - operator.concat 对 Python 类：CPython 3.13 不查 __concat__，直接回退
    nb_add——ndarray Python 类架构不可修，记已知限制
  - TestBoolCmp ×2：切片 getitem 返回副本（链式 inf[::3][e]= 写入丢失），
    Rust 视图层
  - nonzero_large_dtype：V dtype 按 f64 槽存储（每元素恒 8B，不论声明
    itemsize）→ V≠V8 的 view/写回尺寸错误；结构化 dtype 存储是字节精确
    （matrixlib view 正常）
  - nonzero_twodim：flags.aligned 是 stub（恒 True）
  - TestTypes void promotion ×2：类内状态污染（单独跑全过；整类跑只挂
    首用例），污染源待二分
  - names_are_unique：long/longlong、ulong/ulonglong、double/longdouble、
    cdouble/clongdouble 四组别名被 dtype 架构统一（P1 NEP50 项）
  - test_numeric 真实基线 18：BoolCmp ×2、TestTypes ×2、Nonzero ×4、
    BinaryRepr ×2、BaseRepr ×1、none_compares ×1、clip out 重叠 ×1、
    isclose masked ×1、full ×1、filled_like ×1、keepdims raise ×1、
    astype ×1

## 验证约定

- 每批修完：跑受影响文件（新进程）对照 test_config.txt 基线，
  确认 0 新回归后回写计数并单独提交
- 大改动走 WT/HEAD 分块电池对照（方法见 git 历史与项目记忆）
