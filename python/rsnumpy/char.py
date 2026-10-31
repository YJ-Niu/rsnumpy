"""rsnumpy.char - 向量化字符串操作模块（Rust 层实现，本模块为薄包装）。

所有元素级字符串运算由 Rust 端 ``num_core.char_str_op/char_int_op/
char_bool_op/char_add/char_multiply`` 一次跨边界批量完成。
"""

import builtins

import rsnumpy.num_core as _core


def _get_strings(a):
    """从 ndarray 或列表中提取字符串列表（bytes 解码为 utf-8 str）。"""
    if isinstance(a, (str, bytes)):
        return [a]
    if hasattr(a, '_raw_data'):
        vals = builtins.list(a._raw_data)
    elif hasattr(a, '_inst'):
        rd = getattr(a._inst, '_raw_data', None)
        if rd is not None:
            vals = builtins.list(rd)
        else:
            vals = _flatten(a._inst.tolist(), [])
    elif hasattr(a, 'tolist'):
        vals = _flatten(a.tolist(), [])
    else:
        vals = _flatten(a, [])
    return vals


def _flatten(x, out):
    for v in x:
        if isinstance(v, (builtins.list, builtins.tuple)):
            _flatten(v, out)
        else:
            out.append(v)
    return out


def _decode(vals):
    """bytes 元素解码（surrogateescape 往返无损）。"""
    return [v.decode('utf-8', 'surrogateescape') if isinstance(v, bytes) else v
            for v in vals]


def _is_bytes(vals):
    return builtins.any(isinstance(v, bytes) for v in vals)


def _shape_of(a):
    try:
        return builtins.tuple(a.shape)
    except Exception:
        return ()


def _wrap_str(vals, shape):
    """把扁平结果列表按 shape 包装为字符串数组。"""
    from .__init__ import ndarray
    if not shape:
        return ndarray([vals[0] if vals else ''])
    nested = vals
    for dim in builtins.reversed(shape[1:]):
        nested = [nested[i:i + dim] for i in range(0, builtins.len(nested), dim)]
    return ndarray(nested)


def _str_op(op, a, arg1=None, arg2=None, arg3=None):
    """字符串算子薄包装：解码 → Rust → 编码 → 包装。"""
    vals = _decode(_get_strings(a))
    out = _core.char_str_op(op, vals,
                            None if arg1 is None else _as_str_arg(arg1),
                            None if arg2 is None else _as_str_arg(arg2),
                            None if arg3 is None else _as_str_arg(arg3))
    if _is_bytes(_get_strings(a)):
        out = [v.encode('utf-8', 'surrogateescape') for v in out]
    shape = _shape_of(a)
    if isinstance(a, (str, bytes)):
        return out[0] if out else ''
    if shape == ():
        return out[0] if out else ''
    return _wrap_str(out, shape)


def _as_str_arg(x):
    """算子参数转字符串（宽度/填充符等）。"""
    if isinstance(x, (str, bytes)):
        return x.decode('utf-8', 'surrogateescape') if isinstance(x, bytes) else x
    return builtins.str(x)


def _int_op(op, a, arg1=None, arg2=None, arg3=None):
    vals = _decode(_get_strings(a))
    out = _core.char_int_op(op, vals,
                            None if arg1 is None else _as_str_arg(arg1),
                            None if arg2 is None else _as_str_arg(arg2),
                            None if arg3 is None else _as_str_arg(arg3))
    shape = _shape_of(a)
    if isinstance(a, (str, bytes)) or shape == ():
        return out[0] if out else -1
    from .__init__ import ndarray
    res = ndarray([builtins.float(v) for v in out], _dtype='int64')
    return res.reshape(shape)


def _bool_op(op, a, arg1=None, other=None):
    vals = _decode(_get_strings(a))
    other_vals = None
    if other is not None:
        other_vals = _decode(_get_strings(other))
    out = _core.char_bool_op(op, vals,
                             None if arg1 is None else _as_str_arg(arg1),
                             other_vals)
    shape = _shape_of(a)
    if isinstance(a, (str, bytes)) or shape == ():
        return builtins.bool(out[0]) if out else False
    from .__init__ import ndarray
    res = ndarray([1.0 if v else 0.0 for v in out])
    res._dtype = 'bool'
    return res.reshape(shape)


# ==================== 模块级函数（np.char.*） ====================

def add(x1, x2):
    """逐元素字符串连接（np.char.add）。"""
    v1 = _decode(_get_strings(x1))
    v2 = _decode(_get_strings(x2))
    out = _core.char_add(v1, v2)
    if _is_bytes(_get_strings(x1)):
        out = [v.encode('utf-8', 'surrogateescape') for v in out]
    shape = _shape_of(x1)
    if shape == () and _shape_of(x2) == ():
        return out[0] if out else ''
    if shape == ():
        shape = _shape_of(x2)
    return _wrap_str(out, shape)


def multiply(a, i):
    """逐元素重复（np.char.multiply）。"""
    vals = _decode(_get_strings(a))
    if hasattr(i, 'tolist'):
        fl = []
        _flatten(i.tolist(), fl)
        ns = [builtins.int(v) for v in fl]
    elif isinstance(i, (builtins.list, builtins.tuple)):
        fl = []
        _flatten(builtins.list(i), fl)
        ns = [builtins.int(v) for v in fl]
    else:
        ns = [builtins.int(i)]
    out = _core.char_multiply(vals, ns)
    if _is_bytes(_get_strings(a)):
        out = [v.encode('utf-8', 'surrogateescape') for v in out]
    shape = _shape_of(a)
    if shape == ():
        return out[0] if out else ''
    return _wrap_str(out, shape)


def capitalize(a):
    """首字母大写，其余小写。"""
    return _str_op('capitalize', a)


def title(a):
    """numpy title 语义（非字母后首字母大写）。"""
    return _str_op('title', a)


def lower(a):
    """全部小写。"""
    return _str_op('lower', a)


def upper(a):
    """全部大写。"""
    return _str_op('upper', a)


def swapcase(a):
    """大小写互换。"""
    return _str_op('swapcase', a)


def strip(a, chars=None):
    """两侧去字符。"""
    return _str_op('strip', a, chars)


def lstrip(a, chars=None):
    return _str_op('lstrip', a, chars)


def rstrip(a, chars=None):
    return _str_op('rstrip', a, chars)


def center(a, width, fillchar=' '):
    """居中填充。"""
    return _str_op('center', a, width, fillchar)


def ljust(a, width, fillchar=' '):
    return _str_op('ljust', a, width, fillchar)


def rjust(a, width, fillchar=' '):
    return _str_op('rjust', a, width, fillchar)


def zfill(a, width):
    return _str_op('zfill', a, width)


def replace(a, old, new, count=None):
    """逐元素替换。"""
    return _str_op('replace', a, old, new, count)


def expandtabs(a, tabsize=8):
    return _str_op('expandtabs', a, tabsize)


def join(sep, a):
    """以 sep 连接序列元素。"""
    return _str_op('join', a, sep)


def find(a, sub, start=0, end=None):
    """逐元素 str.find（未找到返回 -1）。"""
    return _int_op('find', a, sub, start, end)


def rfind(a, sub, start=0, end=None):
    return _int_op('rfind', a, sub, start, end)


def index(a, sub, start=0, end=None):
    return _int_op('index', a, sub, start, end)


def rindex(a, sub, start=0, end=None):
    return _int_op('rindex', a, sub, start, end)


def count(a, sub, start=0, end=None):
    return _int_op('count', a, sub)


def startswith(a, prefix, start=0, end=None):
    return _bool_op('startswith', a, prefix)


def endswith(a, suffix, start=0, end=None):
    return _bool_op('endswith', a, suffix)


def equal(x1, x2):
    """逐元素 ==。"""
    return _bool_op('equal', x1, other=x2)


def not_equal(x1, x2):
    return _bool_op('not_equal', x1, other=x2)


def greater(x1, x2):
    return _bool_op('greater', x1, other=x2)


def greater_equal(x1, x2):
    return _bool_op('greater_equal', x1, other=x2)


def less(x1, x2):
    return _bool_op('less', x1, other=x2)


def less_equal(x1, x2):
    return _bool_op('less_equal', x1, other=x2)


def _is_method(a, op):
    return _bool_op(op, a)


def isalnum(a):
    return _is_method(a, 'isalnum')


def isalpha(a):
    return _is_method(a, 'isalpha')


def isdecimal(a):
    return _is_method(a, 'isdecimal')


def isdigit(a):
    return _is_method(a, 'isdigit')


def islower(a):
    return _is_method(a, 'islower')


def isnumeric(a):
    return _is_method(a, 'isnumeric')


def isspace(a):
    return _is_method(a, 'isspace')


def istitle(a):
    return _is_method(a, 'istitle')


def isupper(a):
    return _is_method(a, 'isupper')


def split(a, sep=None, maxsplit=None):
    """逐元素分割（Rust 层暂不支持，回退 Python）。

    numpy 的 split 返回 ragged object 数组，无法映射为规则字符串算子。
    """
    s = _get_strings(a)
    result = [v.split(sep, maxsplit) if maxsplit is not None
              else v.split(sep) for v in s]
    if isinstance(a, (str, bytes)):
        return result[0]
    return _make_result(result)


def rsplit(a, sep=None, maxsplit=None):
    s = _get_strings(a)
    result = [v.rsplit(sep, maxsplit) if maxsplit is not None
              else v.rsplit(sep) for v in s]
    if isinstance(a, (str, bytes)):
        return result[0]
    return _make_result(result)


def splitlines(a, keepends=False):
    s = _get_strings(a)
    result = [v.splitlines(keepends) for v in s]
    if isinstance(a, (str, bytes)):
        return result[0]
    return _make_result(result)


def decode(a, encoding='utf-8', errors='strict'):
    """数组元素依次调用 bytes.decode。"""
    s = _get_strings(a)
    result = [x.decode(encoding, errors) if isinstance(x, bytes) else x for x in s]
    if isinstance(a, (str, bytes)):
        return result[0]
    return _make_result(result)


def encode(a, encoding='utf-8', errors='strict'):
    """数组元素依次调用 str.encode。"""
    s = _get_strings(a)
    result = [x.encode(encoding, errors) if isinstance(x, str) else x for x in s]
    if isinstance(a, (str, bytes)):
        return result[0]
    return _make_result(result)


def partition(a, sep):
    s = _get_strings(a)
    result = [v.partition(sep) for v in s]
    if isinstance(a, (str, bytes)):
        return result[0]
    return _make_result(result)


def rpartition(a, sep):
    s = _get_strings(a)
    result = [v.rpartition(sep) for v in s]
    if isinstance(a, (str, bytes)):
        return result[0]
    return _make_result(result)


def mod(a, values):
    """逐元素 % 格式化。"""
    s = _get_strings(a)
    if hasattr(values, 'tolist'):
        values = _flatten(values.tolist(), [])
    if not isinstance(values, (builtins.list, builtins.tuple)):
        values = [values] * builtins.len(s)
    result = [v % x for v, x in builtins.zip(s, values)]
    if isinstance(a, (str, bytes)):
        return result[0]
    return _make_result(result)


def _make_result(strings):
    """将字符串列表包装为 ndarray（ragged 场景 object 数组）。"""
    from .__init__ import ndarray
    return ndarray._wrap_raw(strings)


# ==================== chararray（组合代理，方法委托 Rust） ====================

_RUST_STR_OPS = {
    'capitalize': ('capitalize',), 'lower': ('lower',), 'upper': ('upper',),
    'title': ('title',), 'swapcase': ('swapcase',), 'strip': ('strip',),
    'lstrip': ('lstrip',), 'rstrip': ('rstrip',), 'expandtabs': ('expandtabs',),
}
_RUST_ARG_OPS = {
    'center': ('center', 2), 'ljust': ('ljust', 2), 'rjust': ('rjust', 2),
    'zfill': ('zfill', 1), 'replace': ('replace', None),
}
_RUST_INT_OPS = {'find': 1, 'rfind': 1, 'index': 1, 'rindex': 1, 'count': 1}
_RUST_BOOL_OPS = {'startswith': 1, 'endswith': 1, 'isalnum': 0, 'isalpha': 0,
                  'isdecimal': 0, 'isdigit': 0, 'islower': 0, 'isnumeric': 0,
                  'isspace': 0, 'istitle': 0, 'isupper': 0}


def _nested_shape(data):
    shape = []
    while isinstance(data, (builtins.list, builtins.tuple)):
        shape.append(builtins.len(data))
        data = data[0] if data else None
    return builtins.tuple(shape) or (0,)


class chararray:
    """字符串/字节 ndarray（numpy chararray 语义，方法委托 Rust 算子）。

    组合代理：内部持有真实 ndarray（_inst），ndarray 协议经 __getattr__
    代理，元素级字符串方法经 _core.char_* 算子在 Rust 层完成。
    """

    def __new__(cls, obj, itemsize=None, copy=True, unicode=None, order=None):
        from .__init__ import ndarray, array as _array
        _ = copy, order
        if isinstance(obj, chararray):
            inst = ndarray(obj._inst._array.copy())
            inst._dtype = getattr(obj._inst, '_dtype', 'string_')
            inst._raw_data = builtins.list(
                getattr(obj._inst, '_raw_data', None) or [])
            inst._str_dtype = getattr(obj._inst, '_str_dtype', None)
            return _as_chararray(inst)
        if hasattr(obj, '_array'):
            inst = obj.copy()
        elif isinstance(obj, (str, bytes)):
            inst = ndarray([obj])
            inst = inst.reshape(())
        else:
            flat0 = None
            if isinstance(obj, (builtins.list, builtins.tuple)) and obj:
                fl = []
                _flatten(builtins.list(obj), fl)
                flat0 = fl[0] if fl else None
            if unicode is True or (unicode is None and
                                   isinstance(flat0, str)):
                inst = _array(obj, dtype='U')
            elif unicode is False:
                inst = _array(obj, dtype='S')
            else:
                inst = _array(obj)
        return _as_chararray(inst)

    def __getattr__(self, name):
        # ndarray 协议代理：shape/dtype/tolist/ndim/size/reshape/astype/...
        return getattr(self._inst, name)

    def _vals(self):
        rd = getattr(self._inst, '_raw_data', None)
        if rd is not None:
            return _flatten(builtins.list(rd), [])
        return _flatten(self._inst.tolist(), [])

    def _apply(self, op, args):
        """元素级方法委托 Rust 算子。"""
        vals = _decode(self._vals())
        was_bytes = _is_bytes(self._vals())
        shape = _shape_of(self._inst)
        str_args = [_as_str_arg(x) if not isinstance(x, (builtins.list, builtins.tuple))
                    and not hasattr(x, 'tolist') else None for x in args]
        # 数组参数：取标量语义（chararray 方法多数为标量参数）
        str_args = [_as_str_arg(x) if v is None else v for x, v in
                    builtins.zip(args, str_args)]
        if op in _RUST_STR_OPS or op in _RUST_ARG_OPS:
            a1 = str_args[0] if builtins.len(str_args) > 0 else None
            a2 = str_args[1] if builtins.len(str_args) > 1 else None
            a3 = str_args[2] if builtins.len(str_args) > 2 else None
            if op == 'replace' and builtins.len(args) < 3:
                a3 = None
            out = _core.char_str_op(op, vals, a1, a2, a3)
        elif op in _RUST_INT_OPS:
            out = _core.char_int_op(op, vals,
                                    str_args[0] if str_args else None)
            return self._pack(out, shape, as_int=True)
        elif op in _RUST_BOOL_OPS:
            out = _core.char_bool_op(op, vals,
                                     str_args[0] if str_args else None)
            return self._pack(out, shape, as_bool=True)
        else:
            raise AttributeError(op)
        if was_bytes:
            out = [v.encode('utf-8', 'surrogateescape') for v in out]
        return self._pack(out, shape)

    def _pack(self, vals, shape, as_int=False, as_bool=False):
        from .__init__ import ndarray
        if as_int:
            res = ndarray([builtins.float(v) for v in vals], _dtype='int64')
            return res.reshape(shape) if shape else res
        if as_bool:
            res = ndarray([1.0 if v else 0.0 for v in vals])
            res._dtype = 'bool'
            return res.reshape(shape) if shape else res
        return _wrap_str(vals, shape)

    def capitalize(self):
        return self._apply('capitalize', [])

    def lower(self):
        return self._apply('lower', [])

    def upper(self):
        return self._apply('upper', [])

    def title(self):
        return self._apply('title', [])

    def swapcase(self):
        return self._apply('swapcase', [])

    def strip(self, chars=None):
        return self._apply('strip', [chars])

    def lstrip(self, chars=None):
        return self._apply('lstrip', [chars])

    def rstrip(self, chars=None):
        return self._apply('rstrip', [chars])

    def center(self, width, fillchar=' '):
        return self._apply('center', [width, fillchar])

    def ljust(self, width, fillchar=' '):
        return self._apply('ljust', [width, fillchar])

    def rjust(self, width, fillchar=' '):
        return self._apply('rjust', [width, fillchar])

    def zfill(self, width):
        return self._apply('zfill', [width])

    def expandtabs(self, tabsize=8):
        return self._apply('expandtabs', [tabsize])

    def replace(self, old, new, count=None):
        return self._apply('replace', [old, new, 0 if count is None else count])

    def find(self, sub, start=0, end=None):
        return self._apply('find', [sub])

    def rfind(self, sub, start=0, end=None):
        return self._apply('rfind', [sub])

    def index(self, sub, start=0, end=None):
        return self._apply('index', [sub])

    def rindex(self, sub, start=0, end=None):
        return self._apply('rindex', [sub])

    def count(self, sub, start=0, end=None):
        return self._apply('count', [sub])

    def startswith(self, prefix, start=0, end=None):
        return self._apply('startswith', [prefix])

    def endswith(self, suffix, start=0, end=None):
        return self._apply('endswith', [suffix])

    def isalnum(self):
        return self._apply('isalnum', [])

    def isalpha(self):
        return self._apply('isalpha', [])

    def isdecimal(self):
        return self._apply('isdecimal', [])

    def isdigit(self):
        return self._apply('isdigit', [])

    def islower(self):
        return self._apply('islower', [])

    def isnumeric(self):
        return self._apply('isnumeric', [])

    def isspace(self):
        return self._apply('isspace', [])

    def istitle(self):
        return self._apply('istitle', [])

    def isupper(self):
        return self._apply('isupper', [])

    def join(self, seq):
        return self._apply('join', [seq])

    def mod(self, values):
        return mod(self, values)

    def multiply(self, i):
        return multiply(self, i)

    def split(self, sep=None, maxsplit=None):
        return split(self, sep, maxsplit)

    def rsplit(self, sep=None, maxsplit=None):
        return rsplit(self, sep, maxsplit)

    def splitlines(self, keepends=False):
        return splitlines(self, keepends)

    def decode(self, encoding='utf-8', errors='strict'):
        return decode(self, encoding, errors)

    def encode(self, encoding='utf-8', errors='strict'):
        return encode(self, encoding, errors)

    def partition(self, sep):
        return partition(self, sep)

    def rpartition(self, sep):
        return rpartition(self, sep)

    def __len__(self):
        return builtins.len(self._inst)

    def __iter__(self):
        return iter(self._inst)

    def __getitem__(self, key):
        return self._inst[key]

    def __setitem__(self, key, value):
        self._inst[key] = value

    def __add__(self, other):
        return add(self, other)

    def __radd__(self, other):
        return add(other, self)

    def __mul__(self, other):
        return multiply(self, other)

    def __rmul__(self, other):
        return multiply(self, other)

    def _cmp(self, other, op):
        return _bool_op(op, self, other=other)

    def __eq__(self, other):
        return self._cmp(other, 'equal')

    def __ne__(self, other):
        return self._cmp(other, 'not_equal')

    def __gt__(self, other):
        return self._cmp(other, 'greater')

    def __lt__(self, other):
        return self._cmp(other, 'less')

    def __ge__(self, other):
        return self._cmp(other, 'greater_equal')

    def __le__(self, other):
        return self._cmp(other, 'less_equal')

    def __bool__(self):
        v = self._vals()
        return builtins.bool(v[0]) if builtins.len(v) == 1 else builtins.bool(self._inst.size)


def _as_chararray(inst):
    """把 ndarray 实例包装为 chararray（组合代理）。"""
    wrapper = object.__new__(chararray)
    wrapper._inst = inst
    return wrapper


def array(obj, itemsize=None, copy=True, unicode=None, order=None):
    """创建 chararray（np.char.array）。"""
    return chararray(obj, itemsize=itemsize, copy=copy, unicode=unicode,
                     order=order)


def asarray(obj, itemsize=None, unicode=None):
    """创建 chararray（np.char.asarray）。"""
    return chararray(obj, itemsize=itemsize, unicode=unicode)
