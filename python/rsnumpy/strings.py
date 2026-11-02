"""numpy.strings 风格字符串 API（np.strings.*）。

与 numpy 2.x 的 ``np.strings`` 命名空间对齐：逐元素作用于
定宽字符串数组（U/S dtype），多数实现复用 ``rsnumpy.char`` 的
底层算子，其余为等价的 Python 实现。
"""

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


def partition(a, sep=' '):
    """按 sep 拆三段，返回 (head, sep, tail) 数组元组。"""
    vals = _char._decode(_char._get_strings(a))
    heads, seps, tails = [], [], []
    for v in vals:
        h, s, t = v.partition(str(sep))
        heads.append(h)
        seps.append(s)
        tails.append(t)
    shape = _char._shape_of(a)
    wb = _char._a_was_bytes(a)
    if isinstance(a, (str, bytes)) or shape == ():
        return (heads[0] if heads else '', seps[0] if seps else '',
                tails[0] if tails else '')
    return (_char._wrap_str(heads, shape, wb),
            _char._wrap_str(seps, shape, wb),
            _char._wrap_str(tails, shape, wb))


def rpartition(a, sep=' '):
    """从右侧按 sep 拆三段（numpy.strings.rpartition 兼容）。"""
    vals = _char._decode(_char._get_strings(a))
    heads, seps, tails = [], [], []
    for v in vals:
        h, s, t = v.rpartition(str(sep))
        heads.append(h)
        seps.append(s)
        tails.append(t)
    shape = _char._shape_of(a)
    wb = _char._a_was_bytes(a)
    if isinstance(a, (str, bytes)) or shape == ():
        return (heads[0] if heads else '', seps[0] if seps else '',
                tails[0] if tails else '')
    return (_char._wrap_str(heads, shape, wb),
            _char._wrap_str(seps, shape, wb),
            _char._wrap_str(tails, shape, wb))


def slice(a, start, stop, step):  # noqa: A001 (numpy 同名 API)
    """逐元素切片（numpy.strings.slice 兼容）。"""
    return _py_str(lambda v: v[start:stop:step], a)


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
isdecimal = lambda a: _is(a, 'isdecimal')    # noqa: E731
isdigit = lambda a: _is(a, 'isdigit')        # noqa: E731
islower = lambda a: _is(a, 'islower')        # noqa: E731
isnumeric = lambda a: _is(a, 'isnumeric')    # noqa: E731
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
