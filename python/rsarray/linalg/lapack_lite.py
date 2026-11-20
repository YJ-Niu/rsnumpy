# SPDX-License-Identifier: MIT
"""numpy.linalg.lapack_lite 兼容桩。

NumPy 中该模块暴露底层 LAPACK 例程；rsnumpy 无对应 C 层，仅提供
测试用到的最小接口（xerbla 抛 LinAlgError，与 NumPy 行为一致）。
"""


def xerbla(srname, info, **kwargs):
    """LAPACK 错误报告例程（NumPy 中直接抛 LinAlgError）。"""
    from rsnumpy.linalg import LinAlgError
    raise LinAlgError(
        f"{srname} (info={info})")
