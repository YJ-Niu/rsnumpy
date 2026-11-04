"""numpy._core.numerictypes 兼容实现（标量类型表与转换工具）。

对齐 NumPy 2.x 语义；与更新版 NumPy 行为不一致处（如 ``sctype2char(1)``
抛 ValueError）以 rsnumpy 测试契约为准。
"""

__all__ = [
    'sctypes', 'sctypeDict', 'typeDict', 'typecodes',
    'issctype', 'obj2sctype', 'sctype2char',
    'bool', 'int', 'float', 'complex', 'str', 'bytes', 'object',
]

from builtins import bytes, complex, float, int, object, str  # noqa: F401
# 注意：bool 不从此处导入——nt.bool 必须指向 rsnumpy 标量类（numpy
# numerictypes 语义），由模块 __getattr__ 回落到 rsnumpy.bool。

_TYPECODES = {
    'Character': 'c',
    'Integer': 'bhilqnp',
    'UnsignedInteger': 'BHILQNP',
    'Float': 'efdg',
    'Complex': 'FDG',
    'AllInteger': 'bBhHiIlLqQnNpP',
    'AllFloat': 'efdgFDG',
    'Datetime': 'Mm',
    'All': '?bhilqnpBHILQNPefdgFDGSUVOMm',
}


def _rs():
    import rsnumpy
    return rsnumpy


def _concrete_scalar_types():
    """全部具体标量类型类（不含抽象基类）。"""
    rs = _rs()
    names = (
        'bool', 'int8', 'int16', 'int32', 'int64',
        'uint8', 'uint16', 'uint32', 'uint64',
        'float16', 'float32', 'float64', 'longdouble',
        'complex64', 'complex128', 'clongdouble',
    )
    types_ = [getattr(rs, n) for n in names if hasattr(rs, n)]
    for extra in ('object_', 'unicode_', 'bytes_', 'void'):
        t = getattr(rs, extra, None)
        if t is not None:
            types_.append(t)
    seen = set()
    out = []
    for t in types_:
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out


def _safe_itemsize(t):
    rs = _rs()
    try:
        return rs.dtype(t).itemsize
    except Exception:
        return 0


_TYPE_CHAR_FALLBACK = {'void': 'V', 'unicode_': 'U', 'bytes_': 'S',
                       'object_': 'O'}


def _type_char(t):
    rs = _rs()
    try:
        return rs.dtype(t).char
    except Exception:
        return _TYPE_CHAR_FALLBACK.get(t.__name__, 'O')


def _sctypes_table():
    """kind 分组 → 标量类型列表（numpy ``sctypes`` 语义）。"""
    rs = _rs()
    groups = (
        ('int', rs.signedinteger),
        ('uint', rs.unsignedinteger),
        ('float', rs.floating),
        ('complex', rs.complexfloating),
        ('others', rs.generic),
    )
    table = {key: [] for key, _ in groups}
    for t in _concrete_scalar_types():
        # object_ 不可带参构造（dtype('O').type(0) 会崩），不入 sctypes
        # 'others'（numpy sctypes 语义下 object 亦非数值分组成员）。
        if getattr(t, '__name__', '') in ('object_', 'object'):
            continue
        for key, abstract in groups:
            if issubclass(t, abstract):
                table[key].append(t)
                break
    for key in table:
        table[key].sort(key=lambda x: (_safe_itemsize(x), x.__name__))
    return table


def _sctype_dict():
    """名称 → 标量类型类（numpy ``sctypeDict`` 语义，含平台别名）。"""
    rs = _rs()
    pairs = (
        ('bool', 'bool'),
        ('int8', 'int8'), ('int16', 'int16'), ('int32', 'int32'),
        ('int64', 'int64'),
        ('uint8', 'uint8'), ('uint16', 'uint16'), ('uint32', 'uint32'),
        ('uint64', 'uint64'),
        ('float16', 'float16'), ('float32', 'float32'),
        ('float64', 'float64'), ('longdouble', 'longdouble'),
        ('complex64', 'complex64'), ('complex128', 'complex128'),
        ('clongdouble', 'clongdouble'),
        ('long', 'long'), ('ulong', 'ulong'),
        ('intc', 'intc'), ('uintc', 'uintc'), ('intp', 'intp'),
        ('uintp', 'uintp'),
        ('object', 'object_'), ('str_', 'unicode_'), ('bytes_', 'bytes_'),
        ('void', 'void'),
    )
    out = {}
    for key, attr in pairs:
        t = getattr(rs, attr, None)
        if t is not None:
            out[key] = t
    return out


def _module_attr(name):
    """PEP 562：按需暴露类型表与全部标量类型（numpy numerictypes 兼容）。"""
    if name == 'sctypes':
        return _sctypes_table()
    if name in ('sctypeDict', 'typeDict'):
        return _sctype_dict()
    if name == 'typecodes':
        return _TYPECODES
    rs = _rs()
    obj = getattr(rs, name, None)
    if obj is not None and not name.startswith('_'):
        return obj
    raise AttributeError(
        f"module 'rsnumpy._core.numerictypes' has no attribute {name!r}")


def __getattr__(name):
    return _module_attr(name)


def issctype(rep):
    """判断对象是否代表标量数据型（numpy ``issctype`` 兼容）。"""
    rs = _rs()
    if not isinstance(rep, (type, rs.dtype)):
        return False
    try:
        res = obj2sctype(rep)
    except Exception:
        return False
    if res is None:
        return False
    return res is not rs.object_


def obj2sctype(rep, default=None):
    """返回对象对应的标量类型（numpy ``obj2sctype`` 兼容）。"""
    rs = _rs()
    try:
        if isinstance(rep, type) and issubclass(rep, rs.generic):
            return rep
        if isinstance(rep, rs.ndarray):
            return rep.dtype.type
        if isinstance(rep, type) and issubclass(rep,
                (list, tuple, dict, set, bytearray, rs.ndarray)):
            return rs.object_
        dt = getattr(rep, 'dtype', None)
        if isinstance(dt, rs.dtype):
            t = type(rep)
            if t in _sctype_dict().values():
                return t
        return rs.dtype(rep).type
    except Exception:
        return default


def sctype2char(sctype):
    """返回标量类型对应的单字符表示（numpy ``sctype2char`` 兼容）。

    - 数组/标量实例 → 取其 dtype；
    - 抽象 generic 子类、第三方标量类型 → KeyError；
    - 无法推断的非类型值（如裸 int）→ ValueError。
    """
    rs = _rs()
    d = _sctype_dict()
    if isinstance(sctype, rs.ndarray):
        return rs.dtype(sctype.dtype).char
    if isinstance(sctype, rs.dtype):
        return sctype.char
    if isinstance(sctype, str):
        try:
            return rs.dtype(sctype).char
        except Exception:
            raise ValueError(f"invalid type string {sctype!r}") from None
    if not isinstance(sctype, type):
        dt = getattr(sctype, 'dtype', None)
        if dt is not None:
            sc = type(sctype)
            if sc not in d.values():
                raise KeyError(sc)
            return rs.dtype(sc).char
        raise ValueError(
            "sctype2char only accepts scalar types and dtype objects, "
            f"got {sctype!r}")
    if issubclass(sctype, rs.generic):
        sc = sctype
    else:
        try:
            sc = rs.dtype(sctype).type
        except Exception:
            if issubclass(sctype,
                          (list, tuple, dict, set, bytearray, rs.ndarray)):
                sc = rs.object_
            else:
                dt = getattr(sctype, 'dtype', None)
                if dt is not None and not isinstance(dt, rs.dtype):
                    # 第三方/测试标量类型（如 rational）：不在 sctypeDict 中
                    sc = sctype
                else:
                    raise ValueError(
                        f"sctype2char: unrecognized type {sctype!r}") from None
    if sc not in d.values():
        raise KeyError(sc)
    return _type_char(sc)
