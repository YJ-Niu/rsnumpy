# SPDX-License-Identifier: MIT
"""numpy.linalg._linalg 兼容实现（私有算法）。

multi_dot 的矩阵链乘法顺序由动态规划给出（Cormen 教材的标准
矩阵链乘 DP）。
"""


def _multi_dot_matrix_chain_order(arrays, return_costs=False):
    """用 DP 求最低成本的矩阵乘法顺序（numpy 同名函数兼容）。

    arrays: 2-d 数组序列。返回 (split_table, cost_table)（return_costs=True）：
    两表均为 n x n 下三角有效，split[i][j] 为 (i..j) 段的最优切分点
    （0-based，i <= k <= j；段内单矩阵/未计算处填 j）。
    """
    n = len(arrays)
    if n < 3:
        if return_costs:
            s = [[j for j in range(n)] for _ in range(n)]
            m = [[0] * n for _ in range(n)]
            return s, m
        return [[j for j in range(n)] for _ in range(n)]
    dims = [a.shape[0] for a in arrays] + [arrays[-1].shape[1]]
    m = [[0] * n for _ in range(n)]
    s = [[0] * n for _ in range(n)]
    for i in range(n):
        s[i][i] = i
    for l in range(2, n + 1):
        for i in range(n - l + 1):
            j = i + l - 1
            best = None
            for k in range(i, j):
                val = m[i][k] + m[k + 1][j] + dims[i] * dims[k + 1] * dims[j + 1]
                if best is None or val < best:
                    best = val
                    s[i][j] = k
            m[i][j] = best
    if return_costs:
        return s, m
    return s


def multi_dot(arrays, out=None):
    """按最优顺序连乘多个矩阵（numpy.linalg.multi_dot 兼容）。"""
    import rsnumpy as np
    if not isinstance(arrays, (list, tuple)) or len(arrays) < 2:
        raise ValueError("Expected an array-like of two or more matrices.")
    arrays = [np.asanyarray(a) for a in arrays]
    if any(a.ndim != 2 for a in arrays):
        raise ValueError("Each array must be 2-dimensional")
    for i in range(len(arrays) - 1):
        if arrays[i].shape[1] != arrays[i + 1].shape[0]:
            raise ValueError("Matrix multiplication dimension mismatch")
    s, _ = _multi_dot_matrix_chain_order(arrays, return_costs=True)

    def rec(i, j):
        if i == j:
            return arrays[i]
        return np.dot(rec(i, s[i][j]), rec(s[i][j] + 1, j))

    return rec(0, len(arrays) - 1)


__all__ = ['_multi_dot_matrix_chain_order', 'multi_dot']
