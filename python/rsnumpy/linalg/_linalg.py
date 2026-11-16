# SPDX-License-Identifier: MIT
"""numpy.linalg._linalg 兼容实现（私有算法）。

multi_dot 的矩阵链乘法顺序由动态规划给出（Cormen 教材的标准
矩阵链乘 DP）。
"""


def _multi_dot_matrix_chain_order(arrays, return_costs=False):
    """用 DP 求最低成本的矩阵乘法顺序（numpy 同名函数兼容）。

    arrays: 1-d/2-d 数组序列（1-d 仅允许在首/末位，成本计算按
    (1, n)/(n, 1) 处理）。return_costs=True 时返回 (split_table,
    cost_table)，均为 n×n ndarray；否则返回 split_table。
    split[i][j] 为段 (i..j) 的最优切分点 k（0-based，i <= k < j）；
    split[i][i] = i；严格上三角（i<j）有效。
    """
    import rsnumpy as np
    n = len(arrays)
    m = [[0] * n for _ in range(n)]
    s = [[i for i in range(n)] for _ in range(n)]
    if n >= 2:
        dims = [a.shape[0] if a.ndim == 2 else 1 for a in arrays]
        dims.append(arrays[-1].shape[-1])
        for l in range(2, n + 1):
            for i in range(n - l + 1):
                j = i + l - 1
                best = None
                best_k = i
                for k in range(i, j):
                    val = m[i][k] + m[k + 1][j] + \
                        dims[i] * dims[k + 1] * dims[j + 1]
                    if best is None or val < best:
                        best = val
                        best_k = k
                m[i][j] = best
                s[i][j] = best_k
    s_arr = np.array(s, dtype=np.int64)
    if return_costs:
        return s_arr, np.array(m)
    return s_arr


def multi_dot(arrays, out=None):
    """按最优顺序连乘多个矩阵（numpy.linalg.multi_dot 兼容）。

    arrays: list/tuple，至少 2 个（不足 → ValueError）；每个元素
    1-d/2-d，1-d 仅允许在首/末位；维数不匹配 → ValueError。
    out: 给定时原地写入并返回该对象本身（ret is out）。
    """
    import rsnumpy as np
    if not isinstance(arrays, (list, tuple)) or len(arrays) < 2:
        raise ValueError("Expected an array-like of two or more matrices.")
    arrays = [np.asanyarray(a) for a in arrays]
    nd = len(arrays)
    for i, a in enumerate(arrays):
        if a.ndim > 2:
            raise ValueError(
                "Each array must be 1-dimensional or 2-dimensional")
        if a.ndim == 1 and i not in (0, nd - 1):
            raise ValueError(
                "Each array must be 2-dimensional, except possibly the "
                "first and last, which may be 1-dimensional as well")
    for i in range(nd - 1):
        a = arrays[i]
        b = arrays[i + 1]
        n_a = a.shape[1] if a.ndim == 2 else a.shape[0]
        if n_a != b.shape[0]:
            raise ValueError("Matrix multiplication dimension mismatch")

    if nd == 2:
        res = np.dot(arrays[0], arrays[1])
    else:
        s, _ = _multi_dot_matrix_chain_order(arrays, return_costs=True)

        def rec(i, j):
            if i == j:
                return arrays[i]
            k = int(s[i, j])
            return np.dot(rec(i, k), rec(k + 1, j))

        res = rec(0, nd - 1)

    if out is not None:
        out_arr = out if hasattr(out, '_array') else np.asanyarray(out)
        out_arr[...] = res
        return out_arr
    return res


__all__ = ['_multi_dot_matrix_chain_order', 'multi_dot']
