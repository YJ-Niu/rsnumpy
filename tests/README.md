# rsarray 测试套件

`tests/` 包含 rsarray 的核心测试套件（功能门禁）与冒烟测试脚本。

## 核心测试（CI 门禁，7 文件）

| 文件 | 覆盖 |
|------|------|
| `test_buffer.py` | PEP 3118 / PEP 688 缓冲协议 |
| `test_arithmetic.py` | 逐元素算术 |
| `test_creation.py` | 数组构造 |
| `test_indexing.py` | 索引 / 切片 |
| `test_reduction.py` | 归约 |
| `test_comparison.py` | 比较运算 |
| `test_known_gaps.py` | 已知缺口（xfail 钉住） |

```bash
python -m pytest tests/test_buffer.py tests/test_arithmetic.py \
  tests/test_creation.py tests/test_indexing.py tests/test_reduction.py \
  tests/test_comparison.py tests/test_known_gaps.py -q
```

## 冒烟测试

`run_test.py` 是独立顺序执行脚本（非 pytest），端到端验证 rsarray 核心功能：

```bash
python tests/run_test.py
```

## 状态清单

`test_config.txt` 由 build_wheel.sh 自动回写（每文件一个 pytest 子进程，
边跑边更新）：`<测试文件> <标志> <passed>/<failed>/<skipped>`。
`--test debug` 只跑标志为 0 的文件；`--test all` 全量重建。
