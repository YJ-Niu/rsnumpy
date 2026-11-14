# SPDX-License-Identifier: MIT
"""numpy._core.arrayprint 兼容实现。

_typelessdata：把数组值转成与 dtype 无关的字符串表示（repr 测试用）。
"""


def _typelessdata(a, typestr):
    """递归地把数组元素转成 typestr 风格的字符串（numpy 同名函数兼容）。"""
    import rsnumpy as rs
    if getattr(a, 'ndim', 0) > 0:
        return typestr + '[' + ', '.join(_typelessdata(v, typestr) for v in a) + ']'
    try:
        v = a.item() if hasattr(a, 'item') else a
    except Exception:
        v = a
    if typestr == 'S':
        if isinstance(v, bytes):
            return typestr + v.decode('ascii', 'replace')
        return typestr + str(v)
    return str(v)


def _reexports():
    import rsnumpy as rs
    d = {}
    if hasattr(rs, 'set_printoptions'):
        d['set_printoptions'] = rs.set_printoptions
    return d


for _n, _v in _reexports().items():
    globals()[_n] = _v
del _reexports, _n, _v

__all__ = ['_typelessdata', 'set_printoptions']
