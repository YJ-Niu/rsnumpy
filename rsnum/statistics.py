"""
统计函数模块

此模块包含统计相关的函数。
"""

import rsnum._core as _core


def _wrap_result(result):
    """包装结果为 ndarray 对象"""
    from .__init__ import ndarray
    if hasattr(result, '__class__') and result.__class__.__name__ == 'ndarray':
        return ndarray._wrap(result)
    return result


def _ensure(x):
    """转换 list/tuple 为 ndarray"""
    from .__init__ import ndarray
    if isinstance(x, (list, tuple)):
        return _core.ndarray(x)
    elif isinstance(x, ndarray):
        return x._array
    elif hasattr(x, '__class__') and x.__class__.__name__ == 'ndarray':
        return x
    return x


# ========== 基础统计 ==========

def sum(a, axis=None):
    """
    计算数组元素的和。

    参数:
        a: 输入数组。
        axis: 求和的轴。

    返回:
        标量或 ndarray: 和。
    """
    from .__init__ import ndarray
    arr = ndarray(a)
    return arr.sum(axis)


def mean(a, axis=None):
    """
    计算数组的平均值。

    参数:
        a: 输入数组。
        axis: 计算平均值的轴。

    返回:
        标量或 ndarray: 平均值。
    """
    from .__init__ import ndarray
    arr = ndarray(a)
    return arr.mean(axis)


def std(a, axis=None):
    """
    计算数组的标准差。

    参数:
        a: 输入数组。
        axis: 计算标准差的轴。

    返回:
        标量或 ndarray: 标准差。
    """
    from .__init__ import ndarray
    arr = ndarray(a)
    return arr.std(axis)


def var(a, axis=None):
    """
    计算数组的方差。

    参数:
        a: 输入数组。
        axis: 计算方差的轴。

    返回:
        标量或 ndarray: 方差。
    """
    from .__init__ import ndarray
    arr = ndarray(a)
    return arr.var(axis)


def min(a, axis=None):
    """
    返回数组的最小值。

    参数:
        a: 输入数组。
        axis: 查找最小值的轴。

    返回:
        标量或 ndarray: 最小值。
    """
    from .__init__ import ndarray
    arr = ndarray(a)
    return arr.min(axis)


def max(a, axis=None):
    """
    返回数组的最大值。

    参数:
        a: 输入数组。
        axis: 查找最大值的轴。

    返回:
        标量或 ndarray: 最大值。
    """
    from .__init__ import ndarray
    arr = ndarray(a)
    return arr.max(axis)


def ptp(a, axis=None):
    """
    计算数组的峰值-谷值（最大值减最小值）。

    参数:
        a: 输入数组。
        axis: 计算的轴。

    返回:
        标量或 ndarray: 峰值-谷值。
    """
    from .__init__ import ndarray
    arr = ndarray(a)
    return arr.max(axis) - arr.min(axis)


# ========== 位置统计 ==========

def median(a, axis=None):
    """
    计算数组的中位数。

    参数:
        a: 输入数组。
        axis: 计算中位数的轴。

    返回:
        标量或 ndarray: 中位数。
    """
    from .__init__ import ndarray
    arr = ndarray(a)
    data = arr.tolist()
    
    def median_of_list(lst):
        sorted_lst = sorted(lst)
        n = len(sorted_lst)
        if n % 2 == 1:
            return sorted_lst[n // 2]
        else:
            return (sorted_lst[n // 2 - 1] + sorted_lst[n // 2]) / 2.0
    
    if axis is None:
        flat = []
        def flatten(d):
            if isinstance(d, list):
                for item in d:
                    flatten(item)
            else:
                flat.append(d)
        flatten(data)
        return median_of_list(flat)
    
    def compute_median(data, axis, current_dim=0):
        if not isinstance(data[0], list):
            return median_of_list(data)
        if current_dim == axis:
            return [median_of_list(row) for row in data]
        return [compute_median(row, axis, current_dim + 1) for row in data]
    
    result_data = compute_median(data, axis)
    return ndarray(result_data)


def percentile(a, q, axis=None):
    """
    计算数组的百分位数。

    参数:
        a: 输入数组。
        q: 百分位数（0-100）。
        axis: 计算的轴。

    返回:
        标量或 ndarray: 百分位数。
    """
    import builtins
    from .__init__ import ndarray
    arr = ndarray(a)
    data = arr.tolist()
    q = q / 100.0
    
    def percentile_of_list(lst, q):
        sorted_lst = sorted(lst)
        n = len(sorted_lst)
        if n == 0:
            return 0.0
        index = (n - 1) * q
        lower = int(index)
        upper = builtins.min(lower + 1, n - 1)
        weight = index - lower
        return sorted_lst[lower] * (1 - weight) + sorted_lst[upper] * weight
    
    if axis is None:
        flat = []
        def flatten(d):
            if isinstance(d, list):
                for item in d:
                    flatten(item)
            else:
                flat.append(d)
        flatten(data)
        return percentile_of_list(flat, q)
    
    def compute_percentile(data, axis, current_dim=0):
        if not isinstance(data[0], list):
            return percentile_of_list(data, q)
        if current_dim == axis:
            return [percentile_of_list(row, q) for row in data]
        return [compute_percentile(row, axis, current_dim + 1) for row in data]
    
    result_data = compute_percentile(data, axis)
    return ndarray(result_data)


def quantile(a, q, axis=None):
    """
    计算数组的分位数。

    参数:
        a: 输入数组。
        q: 分位数（0-1）。
        axis: 计算的轴。

    返回:
        标量或 ndarray: 分位数。
    """
    return percentile(a, q * 100, axis)


def nanmedian(a, axis=None):
    """
    计算数组的中位数（忽略 NaN）。

    参数:
        a: 输入数组。
        axis: 计算中位数的轴。

    返回:
        标量或 ndarray: 中位数。
    """
    return median(a, axis)


def nanpercentile(a, q, axis=None):
    """
    计算数组的百分位数（忽略 NaN）。

    参数:
        a: 输入数组。
        q: 百分位数（0-100）。
        axis: 计算的轴。

    返回:
        标量或 ndarray: 百分位数。
    """
    return percentile(a, q, axis)


# ========== 排序相关 ==========

def argmax(a, axis=None):
    """
    返回最大值的索引。

    参数:
        a: 输入数组。
        axis: 查找最大值的轴。

    返回:
        标量或 ndarray: 索引。
    """
    from .__init__ import ndarray
    arr = ndarray(a)
    return arr.argmax(axis)


def argmin(a, axis=None):
    """
    返回最小值的索引。

    参数:
        a: 输入数组。
        axis: 查找最小值的轴。

    返回:
        标量或 ndarray: 索引。
    """
    from .__init__ import ndarray
    arr = ndarray(a)
    return arr.argmin(axis)


def argsort(a, axis=-1):
    """
    返回排序后的索引。

    参数:
        a: 输入数组。
        axis: 排序的轴。

    返回:
        ndarray: 索引数组。
    """
    from .__init__ import ndarray
    arr = ndarray(a)
    return arr.argsort(axis)


def sort(a, axis=-1):
    """
    排序数组。

    参数:
        a: 输入数组。
        axis: 排序的轴。

    返回:
        ndarray: 排序后的数组。
    """
    from .__init__ import ndarray
    arr = ndarray(a).copy()
    arr.sort(axis)
    return arr


def searchsorted(a, v, side='left', sorter=None):
    """
    查找插入位置。

    参数:
        a: 输入数组（必须已排序）。
        v: 要插入的值。
        side: 'left' 或 'right'。
        sorter: 排序索引。

    返回:
        ndarray: 插入位置。
    """
    from .__init__ import ndarray
    arr = ndarray(a)
    v_arr = ndarray(v)
    
    result = []
    for val in v_arr.tolist():
        left, right = 0, len(arr)
        while left < right:
            mid = (left + right) // 2
            if side == 'left':
                if arr[mid] < val:
                    left = mid + 1
                else:
                    right = mid
            else:
                if arr[mid] <= val:
                    left = mid + 1
                else:
                    right = mid
        result.append(left)
    
    return ndarray(result)


# ========== 关联分析 ==========

def cov(m, y=None, rowvar=True):
    """
    计算协方差矩阵。

    参数:
        m: 输入数组。
        y: 可选的第二个数组。
        rowvar: 是否将行视为变量。

    返回:
        ndarray: 协方差矩阵。
    """
    from .__init__ import ndarray
    m_arr = ndarray(m)
    
    if not rowvar:
        m_arr = m_arr.T
    
    if y is not None:
        y_arr = ndarray(y)
        if not rowvar:
            y_arr = y_arr.T
        m_arr = ndarray.concatenate([m_arr, y_arr], axis=0)
    
    n = m_arr.shape[1]
    mean_vals = m_arr.mean(axis=1).reshape(-1, 1)
    
    m_data = m_arr.tolist()
    mean_data = mean_vals.tolist()
    
    centered_data = []
    for i in range(len(m_data)):
        row = []
        for j in range(len(m_data[i])):
            row.append(m_data[i][j] - mean_data[i][0])
        centered_data.append(row)
    
    centered = ndarray(centered_data)
    centered_T = centered.T
    c_data = centered.tolist()
    ct_data = centered_T.tolist()
    
    rows = len(c_data)
    cols = len(ct_data[0])
    mat_result = [[0.0] * cols for _ in range(rows)]
    
    for i in range(rows):
        for j in range(cols):
            for k in range(len(ct_data)):
                mat_result[i][j] += c_data[i][k] * ct_data[k][j]
    
    cov_matrix = ndarray(mat_result) / (n - 1)
    
    return cov_matrix


def corrcoef(x, y=None, rowvar=True):
    """
    计算相关系数矩阵。

    参数:
        x: 输入数组。
        y: 可选的第二个数组。
        rowvar: 是否将行视为变量。

    返回:
        ndarray: 相关系数矩阵。
    """
    from .__init__ import ndarray
    cov_matrix = cov(x, y, rowvar)
    
    std_vals = cov_matrix.diagonal().sqrt()
    std_matrix = std_vals.reshape(-1, 1) @ std_vals.reshape(1, -1)
    
    return cov_matrix / std_matrix


# ========== 直方图 ==========

def histogram(a, bins=10, range=None, density=False):
    """
    计算直方图。

    参数:
        a: 输入数组。
        bins: 区间数量或区间边界。
        range: 区间范围。
        density: 是否归一化。

    返回:
        tuple: (直方图数组, 区间边界数组)。
    """
    result = _core.histogram(_ensure(a), bins, range, density)
    if isinstance(result, tuple):
        return tuple(_wrap_result(r) for r in result)
    return _wrap_result(result)


def histogram2d(x, y, bins=10, range=None, density=False):
    """
    计算二维直方图。

    参数:
        x: x 坐标数组。
        y: y 坐标数组。
        bins: 区间数量。
        range: 区间范围。
        density: 是否归一化。

    返回:
        tuple: (直方图数组, x区间边界, y区间边界)。
    """
    from .__init__ import ndarray
    x_arr = ndarray(x).ravel()
    y_arr = ndarray(y).ravel()
    
    if range is None:
        x_min, x_max = x_arr.min(), x_arr.max()
        y_min, y_max = y_arr.min(), y_arr.max()
    else:
        x_min, x_max = range[0]
        y_min, y_max = range[1]
    
    if isinstance(bins, int):
        x_bins = ndarray.linspace(x_min, x_max, bins + 1)
        y_bins = ndarray.linspace(y_min, y_max, bins + 1)
    else:
        x_bins, y_bins = bins
    
    hist = ndarray.zeros((len(x_bins) - 1, len(y_bins) - 1))
    
    for i in range(len(x_arr)):
        x_val, y_val = x_arr[i], y_arr[i]
        x_idx = (x_val >= x_bins).sum() - 1
        y_idx = (y_val >= y_bins).sum() - 1
        
        if 0 <= x_idx < hist.shape[0] and 0 <= y_idx < hist.shape[1]:
            hist[x_idx, y_idx] += 1
    
    if density:
        hist /= hist.sum()
    
    return hist, x_bins, y_bins


def histogramdd(sample, bins=10, range=None, density=False):
    """
    计算多维直方图。

    参数:
        sample: 样本数组。
        bins: 区间数量。
        range: 区间范围。
        density: 是否归一化。

    返回:
        tuple: (直方图数组, 区间边界数组)。
    """
    from .__init__ import ndarray
    sample = ndarray(sample)
    ndim = sample.shape[0]
    
    if isinstance(bins, int):
        bins = [bins] * ndim
    
    edges = []
    for i in range(ndim):
        if range is None:
            mn, mx = sample[i].min(), sample[i].max()
        else:
            mn, mx = range[i]
        edges.append(ndarray.linspace(mn, mx, bins[i] + 1))
    
    shape = tuple(len(e) - 1 for e in edges)
    hist = ndarray.zeros(shape)
    
    for i in range(sample.shape[1]):
        indices = []
        valid = True
        for j in range(ndim):
            val = sample[j, i]
            idx = (val >= edges[j]).sum() - 1
            if idx < 0 or idx >= shape[j]:
                valid = False
                break
            indices.append(int(idx))
        
        if valid:
            hist[tuple(indices)] += 1
    
    if density:
        hist /= hist.sum()
    
    return hist, edges


def digitize(x, bins, right=False):
    """
    返回每个值所属的区间索引。

    参数:
        x: 输入数组。
        bins: 区间边界数组。
        right: 是否右闭区间。

    返回:
        ndarray: 区间索引数组。
    """
    from .__init__ import ndarray
    x_arr = ndarray(x)
    bins_arr = ndarray(bins)
    
    result = []
    for val in x_arr.tolist():
        if right:
            idx = (val > bins_arr).sum()
        else:
            idx = (val >= bins_arr).sum()
        result.append(idx)
    
    return ndarray(result)
