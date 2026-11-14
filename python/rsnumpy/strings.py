"""numpy.strings 风格字符串 API（np.strings.*）。

与 numpy 2.x 的 ``np.strings`` 命名空间对齐：逐元素作用于
定宽字符串数组（U/S dtype），多数实现复用 ``rsnumpy.char`` 的
底层算子，其余为等价的 Python 实现。
"""

import builtins

from . import char as _char


def _py_str(fn, a, *args):
    """逐元素字符串映射（Python 实现兜底）。"""
    vals = _char._decode(_char._get_strings(a))
    out = []
    for v in vals:
        try:
            out.append(fn(v, *args))
        except Exception:
            out.append('')
    shape = _char._shape_of(a)
    if isinstance(a, (str, bytes)) or shape == ():
        return out[0] if out else ''
    return _char._wrap_str(out, shape, _char._a_was_bytes(a))


def _py_int(fn, a):
    """逐元素整数值（numpy 语义：标量输入 → 0-d int 数组）。"""
    vals = _char._decode(_char._get_strings(a))
    out = []
    for v in vals:
        try:
            out.append(fn(v))
        except Exception:
            out.append(-1)
    shape = _char._shape_of(a)
    from .__init__ import ndarray
    if isinstance(a, (str, bytes)) or shape == ():
        res = ndarray([float(out[0]) if out else -1], _dtype='int64')
        return res.reshape(())
    return ndarray([float(v) for v in out], _dtype='int64').reshape(shape)


def _py_bool(fn, a):
    """逐元素布尔（numpy 语义：标量输入 → 0-d bool 数组）。"""
    vals = _char._decode(_char._get_strings(a))
    out = []
    for v in vals:
        try:
            out.append(bool(fn(v)))
        except Exception:
            out.append(False)
    shape = _char._shape_of(a)
    from .__init__ import ndarray
    if isinstance(a, (str, bytes)) or shape == ():
        res = ndarray([1.0 if (out and out[0]) else 0.0])
        res._dtype = 'bool'
        return res.reshape(())
    res = ndarray([1.0 if v else 0.0 for v in out])
    res._dtype = 'bool'
    return res.reshape(shape)


# ==================== 委托 np.char（同语义） ====================

add = _char.add
multiply = _char.multiply
capitalize = _char.capitalize
title = _char.title
lower = _char.lower
upper = _char.upper
swapcase = _char.swapcase
center = _char.center
ljust = _char.ljust
rjust = _char.rjust
zfill = _char.zfill
strip = _char.strip
lstrip = _char.lstrip
rstrip = _char.rstrip
replace = _char.replace
expandtabs = _char.expandtabs
find = _char.find
rfind = _char.rfind
index = _char.index
rindex = _char.rindex
count = _char.count
startswith = _char.startswith
endswith = _char.endswith
equal = _char.equal
not_equal = _char.not_equal
greater = _char.greater
greater_equal = _char.greater_equal
less = _char.less
less_equal = _char.less_equal


# ==================== Python 实现 ====================

def str_len(a):
    """逐元素字符串长度（numpy.strings.str_len 兼容）。"""
    return _py_int(len, a)


def decode(a, encoding='utf-8', errors='strict'):
    """bytes 数组 → unicode 数组（numpy.strings.decode 兼容）。"""
    vals = _char._get_strings(a)
    was_bytes = _char._a_was_bytes(a)
    if was_bytes:
        out = []
        for v in vals:
            try:
                out.append(v.decode(encoding, errors) if isinstance(v, (bytes, bytearray))
                           else str(v))
            except Exception:
                out.append('')
    else:
        out = _char._decode(vals)
    shape = _char._shape_of(a)
    if isinstance(a, (str, bytes)) or shape == ():
        return out[0] if out else ''
    return _char._wrap_str(out, shape, False)


def encode(a, encoding='utf-8', errors='strict'):
    """unicode 数组 → bytes 数组（numpy.strings.encode 兼容）。"""
    vals = _char._decode(_char._get_strings(a))
    out = [v.encode(encoding, errors) for v in vals]
    shape = _char._shape_of(a)
    if isinstance(a, (str, bytes)) or shape == ():
        return out[0] if out else b''
    return _char._wrap_str(out, shape, True)


def _partition_impl(a, sep, right=False):
    """partition/rpartition 共用实现。

    numpy 语义：sep 支持标量/数组（逐元素广播）；ArrayLike 输入经
    __array__ 转换（恰好一次）；S dtype 按 bytes 语义往返。"""
    from rsnumpy import asarray as _asarray
    _a = _asarray(a) if not hasattr(a, '_raw_data') else a
    _s = _asarray(sep) if not hasattr(sep, '_raw_data') else sep
    vals = _char._decode(_char._get_strings(_a))
    a_shape = _char._shape_of(_a)
    seps_flat = _char._decode(_char._get_strings(_s))
    s_shape = _char._shape_of(_s)
    bshape = _char._bc_shapes([a_shape, s_shape])
    wb = _char._a_was_bytes(_a)
    heads, seps, tails = [], [], []
    for idx in _char._iter_indices(bshape):
        v = _char._bc_get(vals, a_shape, idx)
        s = _char._bc_get(seps_flat, s_shape, idx)
        h, m, t = v.rpartition(s) if right else v.partition(s)
        heads.append(h)
        seps.append(m)
        tails.append(t)
    if isinstance(a, (str, bytes)):
        # 仅裸标量输入返回裸字符串；0-d 数组输入返回数组（numpy 语义）
        return (heads[0] if heads else '', seps[0] if seps else '',
                tails[0] if tails else '')
    _r1 = _char._wrap_str(heads, bshape, wb)
    _r2 = _char._wrap_str(seps, bshape, wb)
    _r3 = _char._wrap_str(tails, bshape, wb)
    # T（StringDType 变长）标记传播：输入为 T 时结果保持 T dtype
    # （ArrayLike 经 __array__ 转换后才是 T 数组，须查转换后的 _a）
    if getattr(_a, '_strdtype_T', False):
        for _r in (_r1, _r2, _r3):
            if hasattr(_r, '_str_dtype'):
                _r._strdtype_T = True
    return (_r1, _r2, _r3)


def partition(a, sep=' '):
    """按 sep 拆三段，返回 (head, sep, tail) 数组元组。"""
    return _partition_impl(a, sep, right=False)


def rpartition(a, sep=' '):
    """从右侧按 sep 拆三段（numpy.strings.rpartition 兼容）。"""
    return _partition_impl(a, sep, right=True)


class _NoValue:
    """numpy._NoValue 等价哨兵：区分 stop 缺省与 None。"""
    def __repr__(self):
        return '_NoValue'


_NoValue = _NoValue()


def _slice_kind_of(a):
    """字符串输入种类（'S'/'U'）；非字符串输入返回 None（numpy: no-loop）。"""
    if isinstance(a, bytes):
        return 'S'
    if isinstance(a, str):
        return 'U'
    if isinstance(a, (builtins.list, builtins.tuple)):
        flat = _char._flatten(builtins.list(a), [])
        if not flat:
            return None
        if builtins.all(isinstance(v, bytes) for v in flat):
            return 'S'
        if builtins.all(isinstance(v, str) for v in flat):
            return 'U'
        return None
    k = getattr(getattr(a, 'dtype', None), 'kind', None)
    # T（StringDType 变长）按字符串输入处理
    return k if k in ('S', 'U', 'T') else None


def slice(a, start=None, stop=_NoValue, step=None, /):  # noqa: A001
    """逐元素切片（numpy.strings.slice 语义）：

    - 仅给一个参数时该参解释为 stop（同 slice 对象构造规则）；
    - start/stop/step 支持整型标量/数组（广播到 a），字符串输入 →
      "Cannot cast ufunc '_slice' input ..."（TypeError）；
    - None 按 step 符号逐元素展开为 intp 极值；step=0 → ValueError；
    - 非字符串输入 → no-loop 错误（TypeError）。
    """
    if _slice_kind_of(a) is None:
        from rsnumpy._core._exceptions import _UFuncNoLoopError
        raise _UFuncNoLoopError(
            "'_slice' ufunc did not contain a loop with signature matching "
            "types")
    if stop is _NoValue:
        stop = start
        start = None
    from rsnumpy._core import umath as _um
    return _um._slice(a, start, stop, step)


def translate(a, table):
    """逐元素字符映射（numpy.strings.translate 兼容）。"""
    def _t(v):
        try:
            return v.translate(table)
        except Exception:
            return v
    return _py_str(_t, a)


def mod(a, values):
    """逐元素 printf 风格格式化（numpy.strings.mod 兼容）。"""
    vals = _char._decode(_char._get_strings(a))
    if isinstance(values, (tuple, list)):
        vlist = list(values)
    else:
        try:
            vlist = list(values.tolist())
        except Exception:
            vlist = [values] * builtins_max(len(vals), 1)
    out = []
    for i, v in enumerate(vals):
        vv = vlist[i % len(vlist)] if vlist else ''
        try:
            out.append(v % (vv if isinstance(vv, tuple) else (vv,)))
        except Exception:
            out.append('')
    shape = _char._shape_of(a)
    if isinstance(a, (str, bytes)) or shape == ():
        return out[0] if out else ''
    return _char._wrap_str(out, shape, _char._a_was_bytes(a))


def builtins_max(x, y):
    return x if x > y else y


def _is(a, method):
    return _py_bool(getattr(str, method), a)


isalnum = lambda a: _is(a, 'isalnum')        # noqa: E731
isalpha = lambda a: _is(a, 'isalpha')        # noqa: E731
def isdecimal(a):
    # numpy unicode-only：S（bytes）输入无 loop → no-loop 错误
    _char._unicode_only_guard(a, 'isdecimal')
    return _is(a, 'isdecimal')
isdigit = lambda a: _is(a, 'isdigit')        # noqa: E731
islower = lambda a: _is(a, 'islower')        # noqa: E731
def isnumeric(a):
    # numpy unicode-only：S（bytes）输入无 loop → no-loop 错误
    _char._unicode_only_guard(a, 'isnumeric')
    return _is(a, 'isnumeric')
isspace = lambda a: _is(a, 'isspace')        # noqa: E731
istitle = lambda a: _is(a, 'istitle')        # noqa: E731
isupper = lambda a: _is(a, 'isupper')        # noqa: E731


__all__ = [
    'add', 'capitalize', 'center', 'count', 'decode', 'encode',
    'endswith', 'equal', 'expandtabs', 'find', 'greater',
    'greater_equal', 'index', 'isalnum', 'isalpha', 'isdecimal',
    'isdigit', 'islower', 'isnumeric', 'isspace', 'istitle',
    'isupper', 'less', 'less_equal', 'ljust', 'lower', 'lstrip',
    'mod', 'multiply', 'not_equal', 'partition', 'replace',
    'rfind', 'rindex', 'rjust', 'rpartition', 'rstrip', 'slice',
    'startswith', 'str_len', 'strip', 'swapcase', 'title',
    'translate', 'upper', 'zfill',
]


# ==================== numpy 协议分派（TestOverride） ====================
# numpy strings 函数对实现 __array_function__ / __array_ufunc__ 的对象
# 直接分派：function 组查 __array_function__，ufunc 组查 __array_ufunc__，
# 返回协议提供的值。ndarray 自带 __array_ufunc__（数值 ufunc 协议），
# 必须排除在分派之外，否则常规输入会被劫持。

def _protocol_wrap(orig, use_ufunc):
    def wrapper(a, *args, **kwargs):
        # ndarray 自带 __array_ufunc__（数值 ufunc 协议）：排除在分派之外
        _is_arr = hasattr(a, '_raw_data') or hasattr(a, '_array')
        if not _is_arr:
            _hook_name = '__array_ufunc__' if use_ufunc else '__array_function__'
            _hook = getattr(type(a), _hook_name, None)
            if _hook is not None:
                if use_ufunc:
                    return _hook(a, orig, 'call', a, *args, **kwargs)
                return _hook(a, orig, (a,) + args, kwargs)
        return orig(a, *args, **kwargs)
    wrapper.__name__ = getattr(orig, '__name__', 'wrapper')
    wrapper.__doc__ = getattr(orig, '__doc__', None)
    wrapper.__wrapped__ = orig
    return wrapper


def _apply_protocol_wraps():
    _func_names = (
        'center', 'capitalize', 'decode', 'encode', 'expandtabs', 'ljust',
        'lower', 'mod', 'multiply', 'partition', 'rjust', 'rpartition',
        'swapcase', 'title', 'translate', 'upper', 'zfill')
    _ufunc_names = (
        'add', 'lstrip', 'rstrip', 'strip', 'equal', 'not_equal',
        'greater_equal', 'less_equal', 'greater', 'less', 'count',
        'endswith', 'find', 'index', 'isalnum', 'isalpha', 'isdecimal',
        'isdigit', 'islower', 'isnumeric', 'isspace', 'istitle', 'isupper',
        'rfind', 'rindex', 'startswith', 'str_len')
    g = globals()
    for _nm in _func_names:
        if g.get(_nm) is not None:
            g[_nm] = _protocol_wrap(g[_nm], use_ufunc=False)
    for _nm in _ufunc_names:
        if g.get(_nm) is not None:
            g[_nm] = _protocol_wrap(g[_nm], use_ufunc=True)


_apply_protocol_wraps()
