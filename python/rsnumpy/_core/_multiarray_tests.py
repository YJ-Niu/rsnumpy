"""numpy._core._multiarray_tests 兼容实现（C 测试辅助的 Python 版）。"""

__all__ = ['array_indexing']


def _rs():
    import rsnumpy
    return rsnumpy


def array_indexing(op, array, *args):
    """C-API 索引辅助：op=0 取 ``array[args]``；
    op=1 赋值 ``array[args[:-1]] = args[-1]``。

    0-d 数组与越界索引抛 IndexError（与 C 实现一致）；
    op=1 缺少值时抛 ValueError（删除不可能）。
    """
    rs = _rs()
    arr = array if isinstance(array, rs.ndarray) else rs.asarray(array)
    if arr.ndim == 0:
        raise IndexError("0-d arrays cannot be indexed")
    if op == 0:
        if not args:
            raise IndexError("no indices given")
        return arr[args]
    if op == 1:
        value = args[-1]
        idx = args[:-1]
        if not idx:
            raise ValueError("cannot delete from a C array")
        arr[idx] = value
        return None
    raise ValueError(f"unknown op {op}")
