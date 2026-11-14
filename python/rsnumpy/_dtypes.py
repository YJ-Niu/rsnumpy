# SPDX-License-Identifier: MIT
"""完整的 dtype 语义实现（解析、属性、str/repr），逐字节对齐 numpy。

DType 采用富数据模型，支持标量、结构化、子数组、标题、对齐等全部语法，
并复刻 numpy 的 dtype str/repr 生成规则。
"""

import builtins
import math as _math
import sys as _sys

_NATIVE = '<' if _sys.byteorder == 'little' else '>'

# name -> (kind, itemsize, alignment)
_SCALAR_INFO = {
    'bool': ('b', 1, 1),
    'int8': ('i', 1, 1), 'int16': ('i', 2, 2), 'int32': ('i', 4, 4), 'int64': ('i', 8, 8),
    'uint8': ('u', 1, 1), 'uint16': ('u', 2, 2), 'uint32': ('u', 4, 4), 'uint64': ('u', 8, 8),
    'float16': ('f', 2, 2), 'float32': ('f', 4, 4), 'float64': ('f', 8, 8),
    'complex64': ('c', 8, 4), 'complex128': ('c', 16, 8),
    'object': ('O', 8, 8),
    'datetime64': ('M', 8, 8), 'timedelta64': ('m', 8, 8),
}

# 单字符类型码（大小写敏感）
_CHARCODE = {
    '?': 'bool',
    'b': 'int8', 'B': 'uint8',
    'h': 'int16', 'H': 'uint16',
    'i': 'int32', 'I': 'uint32',
    'l': 'int64', 'L': 'uint64',
    'q': 'int64', 'Q': 'uint64',
    'e': 'float16', 'f': 'float32', 'd': 'float64', 'g': 'float64',
    'F': 'complex64', 'D': 'complex128', 'G': 'complex128',
    'O': 'object',
    'p': 'int64', 'P': 'uint64',
    'M': 'datetime64', 'm': 'timedelta64',
}

# kind+size
_KINDSIZE = {
    'i1': 'int8', 'i2': 'int16', 'i4': 'int32', 'i8': 'int64',
    'u1': 'uint8', 'u2': 'uint16', 'u4': 'uint32', 'u8': 'uint64',
    'f2': 'float16', 'f4': 'float32', 'f8': 'float64',
    'c8': 'complex64', 'c16': 'complex128',
    'b1': 'bool',
}

# 全名/别名 -> 规范名
_CANON_FULL = {
    'int': 'int64', 'uint': 'uint64',
    'intp': 'int64', 'uintp': 'uint64', 'intc': 'int32', 'uintc': 'uint32',
    'long': 'int64', 'longlong': 'int64', 'ulong': 'uint64', 'ulonglong': 'uint64',
    'short': 'int16', 'ushort': 'uint16', 'byte': 'int8', 'ubyte': 'uint8',
    'float': 'float64', 'double': 'float64', 'single': 'float32',
    'half': 'float16', 'longdouble': 'float64', 'float_': 'float64',
    'complex': 'complex128', 'cdouble': 'complex128', 'cfloat': 'complex128',
    'csingle': 'complex64', 'clongdouble': 'complex128', 'complex_': 'complex128',
    'bool': 'bool', 'bool_': 'bool',
    'object': 'object', 'object_': 'object',
}


def _round_up(offset, align):
    return ((offset + align - 1) // align) * align


# Hook：rsnumpy 包注册解析器后，DType.type 返回 rsnumpy 标量类（numpy 语义：
# np.dtype('int64').type 即 int64 标量类型）。未注册时 .type 回退 Python builtin。
_TYPE_HOOK = None


def _set_type_hook(fn):
    global _TYPE_HOOK
    _TYPE_HOOK = fn


class _DTypeTypeDesc:
    """metaclass 级数据描述符：仅在**类访问**（DType.type）时生效，
    返回可解析的泛型字符串类型（与 np.empty(shape, dtype=DType) 的产物
    dtype |S0 一致，供兼容套件 dtype 类分支做 dtype 往返与填充值构造）。
    实例访问（d.type）走类体 @property type，按 kind 分派标量类。

    必须是数据描述符（定义 __set__/__delete__）：否则 metaclass 属性在
    某些解释器路径下不可靠；类体 property 已优先接管实例访问。"""

    def __get__(self, inst, owner):
        try:
            from .__init__ import string_ as _st
            return _st
        except Exception:
            return bytes

    def __set__(self, inst, value):
        raise AttributeError("can't set DType.type")

    def __delete__(self, inst):
        raise AttributeError("can't delete DType.type")


class _DTypeMeta(type):
    type = _DTypeTypeDesc()


class DType(metaclass=_DTypeMeta):
    """元素数据类型，兼容 numpy.dtype 的属性与 str/repr。

    【关于 object dtype】
    ⚠️ 性能警告：object dtype 会完全失去向量化的速度优势！

    当 dtype='object' 时，数组存储的是 Python 对象的引用，而不是数值数据。
    这会导致以下性能问题：
    - 无法利用 SIMD 指令加速
    - 每次 Python 循环都会产生类型检查和函数调用开销
    - 内存占用更高（存储对象引用 + Python 对象本身）
    - 多线程性能下降（GIL 无法释放）

    【推荐替代方案】
    1. 变长字符串：使用固定长度字符串 dtype（如 'U10'）
    2. 大整数：如果范围在 ±2^53 内，使用 int64
    3. 混合类型：重新设计数据结构，使用结构化数组
    4. Python 对象：考虑使用列表或字典，而不是数组

    【使用示例】
    >>> import rsnumpy as np
    >>> # 不推荐：使用 object dtype（性能低）
    >>> arr = np.array([1, 'text', 3.14], dtype=object)  # 性能陷阱
    >>>
    >>> # 推荐：使用固定类型数组
    >>> int_arr = np.array([1, 2, 3])  # int64，高性能
    >>> str_arr = np.array(['a', 'b', 'c'])  # 自动推断为字符串
    """

    __slots__ = ('_kind', '_itemsize', '_byteorder', '_alignment', '_typename',
                 '_names', '_fields', '_aligned', '_subdtype', '_titles',
                 '_dt_unit', '_type_override')

    # 单例缓存：np.dtype('uint64') is np.dtype('ulong')（numpy 语义——
    # dtype 对象不可变，同类型同配置返回同一对象）
    _scalar_cache = {}

    @classmethod
    def _cache_key(cls, src):
        return (src._typename, src._itemsize, src._names,
                src._subdtype[1] if src._subdtype else None,
                src._byteorder)

    @property
    def _is_numeric(self):
        # numpy 语义：数值类 dtype（b/i/u/f/c）为 True，其余（S/U/V/O/M/T）False
        return self._kind in ('b', 'i', 'u', 'f', 'c')

    def __new__(cls, name=None, fields=None, align=False):
        if isinstance(name, str) and fields is None and not align:
            try:
                _k = cls._cache_key(_parse_any_dtype(name, False))
            except Exception:
                _k = None
            if _k is not None:
                inst = cls._scalar_cache.get(_k)
                if inst is not None:
                    return inst
        return super().__new__(cls)

    def __init__(self, name=None, fields=None, align=False):
        # 向后兼容：DType('int64') / DType(<其他>) 构造标量。
        # align=True：numpy 语义的对齐结构体（字段偏移按对齐要求填充）。
        self._kind = None
        self._itemsize = 0
        self._byteorder = '|'
        self._alignment = 1
        self._typename = None
        self._names = None
        self._fields = None
        self._aligned = False
        self._subdtype = None
        self._titles = None
        self._dt_unit = None
        self._type_override = None
        # numpy (type, descr) 形式：np.dtype((np.record, dt)) → 结构化
        # dtype 的 record 标记副本（.type = record）
        if (isinstance(name, tuple) and len(name) == 2
                and isinstance(name[0], type)):
            _tp, _dc = name
            if _tp.__name__ in ('record', '_Record'):
                _base = _dc if isinstance(_dc, DType) else DType(_dc)
                self._copy_from(_base)
                if getattr(_base, '_names', None) is not None:
                    try:
                        import rsnumpy as _rn_rt
                        self._type_override = _rn_rt._Record
                    except Exception:
                        pass
                return
        if name is None and fields is None:
            # numpy 语义：dtype(None) / dtype() 无参 → object
            name = 'O'
        if name is not None:
            src = _parse_any_dtype(name, False)
            self._copy_from(src)
        else:
            src = None
        if align:
            self._aligned = True
            if self._names:
                # numpy 语义：对齐结构体——每个字段偏移向上取整到自身
                # 对齐要求，itemsize 补齐到最大字段对齐数
                _max_align = 1
                _new_fields = {}
                _off = 0
                for _n in self._names:
                    _fdt = self._fields[_n][0]
                    _fa = max(1, getattr(_fdt, '_alignment', 1) or 1)
                    _max_align = max(_max_align, _fa)
                    _off = _round_up(_off, _fa)
                    _rest = tuple(self._fields[_n][2:])
                    _new_fields[_n] = (_fdt, _off) + _rest
                    _off += _fdt._itemsize
                self._itemsize = _round_up(_off, _max_align)
                self._fields = _new_fields
                self._alignment = _max_align
        if src is not None and fields is None and not align:
            try:
                DType._scalar_cache[DType._cache_key(src)] = self
            except Exception:
                pass

    def _copy_from(self, o):
        self._kind = o._kind
        self._itemsize = o._itemsize
        self._byteorder = o._byteorder
        self._alignment = o._alignment
        self._typename = o._typename
        self._names = o._names
        self._fields = o._fields
        self._aligned = o._aligned
        self._subdtype = o._subdtype
        self._titles = o._titles
        self._dt_unit = o._dt_unit
        self._type_override = getattr(o, '_type_override', None)

    # ---------- 属性 ----------
    @property
    def kind(self):
        return self._kind

    @property
    def itemsize(self):
        return self._itemsize

    @property
    def type(self):
        """该 dtype 对应的标量类型，对齐 `numpy.dtype.type`。

        rsnumpy 注册 hook 后返回 rsnumpy 标量类（numpy 语义；标量类仍是
        builtin 子类型，`np.iscomplexobj(dtype.type(1))` 这类嗅探写法不受
        影响）；未注册 hook 时回退到可调用的 Python builtin 类型。

        【使用示例】
        >>> import rsnumpy as np
        >>> np.dtype('float64').type
        <class 'rsnumpy.float64'>
        """
        # numpy 语义：子数组 dtype 本身是 void dtype（携带 .shape），
        # .type = np.void（如 ('i4',(2,)) → void，而非 int32）
        if self._subdtype is not None:
            try:
                import rsnumpy
                return getattr(rsnumpy, 'void', bytes)
            except Exception:
                return bytes
        # 结构化 dtype（V）：record-view（np.dtype((record,spec)) / .view(recarray)
        # 标记）的 .type=record；普通结构化 .type=void（numpy 语义）。优先于
        # hook——hook 对 V 返回字符串标量类是错的。
        if self._kind == 'V':
            ov = getattr(self, '_type_override', None)
            if ov is not None:
                return ov
            try:
                import rsnumpy
                return getattr(rsnumpy, 'void', bytes)
            except Exception:
                return bytes
        if _TYPE_HOOK is not None:
            try:
                t = _TYPE_HOOK(self)
            except Exception:
                t = None
            if t is not None:
                return t
        kind = self._kind
        if kind == 'b':
            return bool
        if kind == 'c':
            return complex
        if kind == 'f':
            return float
        if kind in ('i', 'u'):
            return int
        if kind == 'O':
            return object
        if kind == 'S':
            try:
                from .__init__ import bytes_ as _bytes_t
                return _bytes_t
            except Exception:
                return bytes
        try:
            from .__init__ import str_ as _str_t
            return _str_t
        except Exception:
            return str  # 'U' 与其余字符串类

    @property
    def alignment(self):
        return self._alignment

    @property
    def byteorder(self):
        return self._byteorder

    @property
    def isnative(self):
        """该 dtype 是否为原生字节序，对齐 `numpy.dtype.isnative`。"""
        import sys
        bo = self._byteorder
        if bo in ('|', '=', '@', 'n', 'native'):
            return True
        native = '<' if sys.byteorder == 'little' else '>'
        return bo == native

    def newbyteorder(self, order='S'):
        """Return a new dtype with a different byte order.

        Parameters
        ----------
        order : str, optional
            Byte order to force.  One of 'S', '<', '>', '=', '|',
            'native', 'I', 'L', 'B'.  Defaults to swapping the current
            byte order ('S').
        """
        if order == 'S' or order == 'native':
            # Swap byte order（numpy 语义：native 返回对侧字节序）
            if self._byteorder == '<':
                new_order = '>'
            elif self._byteorder == '>':
                new_order = '<'
            elif self._byteorder in ('|', '=', '@', 'native', 'n', None):
                import sys as _sys_bo
                new_order = '>' if _sys_bo.byteorder == 'little' else '<'
            else:
                new_order = self._byteorder
        elif order in ('<', '>', '|', '=', 'I', 'L', 'B', 'n'):
            new_order = order
        else:
            raise ValueError(f"Invalid byte order: {order!r}")
        new_dtype = DType()
        new_dtype._copy_from(self)
        new_dtype._byteorder = new_order
        return new_dtype

    @property
    def names(self):
        return self._names

    @names.setter
    def names(self, value):
        # numpy 语义：dt.names = ('k', 'v') 重命名字段（给定的名字依次
        # 对应现有字段；超出给定数的字段被隐藏——align=True 的填充字段
        # 即由此剔除）。
        new_names = tuple(value)
        if len(new_names) > len(self._names):
            raise ValueError(
                f"there are already {len(self._names)} fields, "
                f"cannot set {len(new_names)} names")
        old_names = list(self._names)
        new_fields = {}
        for i, nn in enumerate(new_names):
            new_fields[nn] = self._fields[old_names[i]]
        self._names = new_names
        self._fields = new_fields

    @property
    def isalignedstruct(self):
        return bool(self._aligned)

    @property
    def subdtype(self):
        return self._subdtype

    @property
    def shape(self):
        if self._subdtype is not None:
            return self._subdtype[1]
        return ()

    @property
    def base(self):
        if self._subdtype is not None:
            return self._subdtype[0]
        return self

    @property
    def fields(self):
        if self._names is None:
            return None
        # 字典推导代替显式 for 循环
        out = {n: self._fields[n] for n in self._names}
        # 二次字典推导：补全带标题的字段
        out.update({
            self._fields[n][2]: self._fields[n]
            for n in self._names
            if len(self._fields[n]) > 2 and self._fields[n][2] is not None
        })
        return out

    @property
    def str(self):
        return _typestr(self)

    @property
    def name(self):
        if self._names is not None:
            return 'void%d' % self._itemsize
        if self._subdtype is not None:
            return 'void%d' % self._itemsize
        if self._kind == 'S':
            # numpy 2.x 语义：固定长字节串 name 为 str{itemsize}
            return 'str%d' % self._itemsize
        if self._kind == 'U':
            return 'str%d' % self._itemsize
        if self._kind == 'V':
            return 'void%d' % self._itemsize
        return self._typename

    @property
    def char(self):
        """单字符类型码（numpy dtype.char 兼容）。"""
        mapping = {
            'bool': '?', 'bool_': '?',
            'int8': 'b', 'int16': 'h', 'int32': 'i', 'int64': 'l',
            'uint8': 'B', 'uint16': 'H', 'uint32': 'I', 'uint64': 'L',
            'float16': 'e', 'float32': 'f', 'float64': 'd', 'float128': 'g',
            'complex64': 'F', 'complex128': 'D',
        }
        n = self.name
        if n in mapping:
            return mapping[n]
        kind = self._kind
        if kind in ('b', 'i', 'u', 'f', 'c', 'S', 'U', 'V', 'O', 'M', 'm'):
            return kind
        if n:
            return n[0]
        return '?'

    @property
    def descr(self):
        if self._names is not None:
            # 抽取字段描述逻辑为闭包，便于列表推导复用
            def _field_descr(n):
                fdt = self._fields[n][0]
                if fdt._subdtype is not None:
                    b, sh = fdt._subdtype
                    return (n, _typestr(b), sh)
                elif fdt._names is not None:
                    # numpy 语义：嵌套结构化字段的 descr 全量保留子字段
                    return (n, fdt.descr)
                else:
                    return (n, _typestr(fdt))
            # 使用列表推导代替显式 for 循环
            out = [_field_descr(n) for n in self._names]
            # numpy 语义：字段偏移乱序（字段选择重排）时 descr 无定义
            _offs = [self._fields[n][1] for n in self._names]
            if any(_offs[i] > _offs[i + 1] for i in range(len(_offs) - 1)):
                raise ValueError(
                    "dtype.descr is not defined for types with overlapping "
                    "or out-of-order fields")
            # numpy 语义：字段选择后的尾部未用字节以 V 填充表达
            try:
                _total = self._itemsize
                _last_n = self._names[-1]
                _used = _offs[-1] + self._fields[_last_n][0]._itemsize
                if _used < _total:
                    out.append(('', '|V%d' % (_total - _used)))
            except Exception:
                pass
            return out
        return [('', _typestr(self))]

    @property
    def hasobject(self):
        return _has_object(self)

    # ---------- 索引 ----------
    def __getitem__(self, key):
        if self._names is None:
            raise KeyError(key)
        if isinstance(key, str):
            return self._fields[key][0]
        if isinstance(key, int):
            return self._fields[self._names[key]][0]
        if isinstance(key, list):
            # 一次性取出所有字段的 (v0, v1, v2) 三元组，避免重复索引
            vs = [self._fields[k] for k in key]
            new_names = list(key)
            new_formats = [v[0] for v in vs]
            new_offsets = [v[1] for v in vs]
            new_titles = [v[2] if len(v) > 2 else None for v in vs]
            return _build_struct(new_names, new_formats, new_offsets,
                                 self._itemsize, new_titles, self._aligned)
        raise KeyError(key)

    def __len__(self):
        if self._names is None:
            raise TypeError("len() of unsized object")
        return len(self._names)

    # ---------- str / repr ----------
    def __str__(self):
        if self._names is not None:
            return _struct_str(self, True)
        if self._subdtype is not None:
            return _subarray_str(self)
        return _scalar_str(self)

    def __repr__(self):
        if self._names is not None:
            arg = _struct_str(self, False)
            if self._aligned:
                arg += ", align=True"
            return "dtype(%s)" % arg
        if self._subdtype is not None:
            return "dtype(%s)" % _subarray_str(self)
        return "dtype(%s)" % _scalar_repr_arg(self)

    # ---------- 相等 / hash ----------
    def _eqkey(self):
        if self._names is not None:
            fk = tuple((n, self._fields[n][0]._eqkey(), self._fields[n][1])
                       for n in self._names)
            return ('struct', fk, self._itemsize, self._aligned)
        if self._subdtype is not None:
            return ('sub', self._subdtype[0]._eqkey(), self._subdtype[1])
        # _typename 参与等值：区分 M8[Y] vs M8[D] 等同 itemsize 的 datetime 单位
        return ('scalar', self._kind, self._itemsize, _norm_bo(self),
                self._typename)

    def __eq__(self, other):
        if isinstance(other, DType):
            return self._eqkey() == other._eqkey()
        if other is None:
            return False
        if isinstance(other, type):
            # dtype 对象 == 标量类（numpy 语义：np.dtype('f4') == np.float32）
            _nm = getattr(other, '__name__', None)
            if _nm:
                try:
                    return self._eqkey() == dtype(_nm)._eqkey()
                except Exception:
                    return NotImplemented
            return NotImplemented
        # np.dtypes 命名空间的动态 dtype 类实例（ObjectDType/Float64DType/…，
        # 仅带 _dt_name 属性的裸类）：按 dtype 名称比较（numpy 语义：
        # np.dtype('O') == np.dtypes.ObjectDType()）
        _onm = getattr(other, '_dt_name', None)
        if _onm is not None:
            try:
                return self._eqkey() == dtype(_onm)._eqkey()
            except Exception:
                return NotImplemented
        try:
            return self._eqkey() == dtype(other)._eqkey()
        except Exception:
            return NotImplemented

    def __ne__(self, other):
        r = self.__eq__(other)
        if r is NotImplemented:
            return r
        return not r

    def __hash__(self):
        return hash(self._eqkey())


# ---------- 构造工厂 ----------
def _new():
    d = DType.__new__(DType)
    d._kind = None
    d._itemsize = 0
    d._byteorder = '|'
    d._alignment = 1
    d._typename = None
    d._names = None
    d._fields = None
    d._aligned = False
    d._subdtype = None
    d._titles = None
    d._dt_unit = None
    return d


def _clone_with_byteorder(dt, bo):
    """克隆新 DType 实例并设 byteorder(不走单例缓存)。

    DType 标量实例按 (typename, itemsize, names, subdtype, byteorder)
    缓存为单例;直接改单例的 _byteorder 会污染同类型的所有数组
    (后续同 dtype 构造会把 byteorder 重置回 native)。凡需给数组挂
    非 native byteorder 的 _dtype_obj,必须用本助手克隆新实例。
    """
    d = _new()
    d._copy_from(dt)
    d._byteorder = bo
    return d


def _make_scalar(name, bo):
    kind, itemsize, align = _SCALAR_INFO[name]
    if itemsize == 1 or kind == 'O':
        byteorder = '|'
    elif bo == '>':
        byteorder = '>'
    else:
        byteorder = '='
    d = _new()
    d._kind = kind
    d._itemsize = itemsize
    d._byteorder = byteorder
    d._alignment = align
    d._typename = name
    return d


def _make_flexible(kind, width, bo):
    d = _new()
    d._kind = kind
    if kind == 'U':
        d._itemsize = width * 4
        d._alignment = 4
        d._byteorder = '>' if bo == '>' else '='
    elif kind == 'T':
        # numpy 2.x StringDType（变长字符串）：itemsize 恒 1，无宽度
        d._itemsize = 1
        d._alignment = 1
        d._byteorder = '|'
    else:  # S / V
        d._itemsize = width
        d._alignment = 1
        d._byteorder = '|'
    return d


def _make_datetime(kind, unit, bo):
    """datetime64(M)/timedelta64(m) 伪 dtype：itemsize 8，单位存 _dt_unit。"""
    if kind == 'm':
        # rsnumpy 的 timedelta 仅支持 generic 单位(numpy 语义:按单位
        # 求值已弃用)
        import warnings as _w
        _w.warn("The 'generic' unit for rsnumpy timedelta is deprecated",
                DeprecationWarning, stacklevel=2)
    d = _new()
    d._kind = kind
    d._itemsize = 8
    d._alignment = 8
    # datetime 始终 native（'='），仅允许显式大端
    d._byteorder = '>' if bo == '>' else '='
    d._typename = '%s8[%s]' % (kind, unit)
    d._dt_unit = unit
    return d


def _make_subarray(base, shape):
    # 使用 math.prod 代替显式循环求积
    prod = _math.prod(shape) if shape else 1
    d = _new()
    d._kind = 'V'
    d._itemsize = base._itemsize * prod
    d._byteorder = '|'
    d._alignment = base._alignment
    d._subdtype = (base, tuple(shape))
    return d


def _layout_fields(formats, align):
    offset = 0
    offsets = []
    max_align = 1
    for f in formats:
        a = f._alignment if align else 1
        offset = _round_up(offset, a)
        offsets.append(offset)
        offset += f._itemsize
        if a > max_align:
            max_align = a
    if align:
        itemsize = _round_up(offset, max_align)
    else:
        itemsize = offset
    return offsets, itemsize


def _build_struct(names, formats, offsets, itemsize, titles, aligned):
    names = list(names)
    formats = list(formats)
    if titles is None:
        titles = [None] * len(names)
    else:
        titles = list(titles)
    if offsets is None:
        offsets, computed = _layout_fields(formats, aligned)
    else:
        offsets = list(offsets)
        computed = max((o + f._itemsize for o, f in zip(offsets, formats)), default=0)
    if itemsize is None:
        itemsize = computed
    # 字典推导代替显式 for 循环
    fields = {
        name: ((fdt, off, title) if title is not None else (fdt, off))
        for name, fdt, off, title in zip(names, formats, offsets, titles)
    }
    d = _new()
    d._kind = 'V'
    d._itemsize = itemsize
    d._byteorder = '|'
    d._alignment = max((f._alignment for f in formats), default=1)
    d._names = tuple(names)
    d._fields = fields
    d._aligned = bool(aligned)
    d._titles = titles
    return d


# ---------- 解析 ----------
def _split_top_commas(s):
    parts = []
    depth = 0
    cur = []
    for ch in s:
        if ch in '([{':
            depth += 1
            cur.append(ch)
        elif ch in ')]}':
            depth -= 1
            cur.append(ch)
        elif ch == ',' and depth == 0:
            parts.append(''.join(cur))
            cur = []
        else:
            cur.append(ch)
    parts.append(''.join(cur))
    return parts


def _parse_shape(s):
    s = s.strip()
    if not s:
        return ()
    parts = [p.strip() for p in s.split(',') if p.strip() != '']
    return tuple(int(p) for p in parts)


def _parse_scalar_token(s):
    s = s.strip()
    bo = None
    if s and s[0] in '<>=|':
        bo = s[0]
        s = s[1:]
    # void 前缀名（'void'/'void3'/'void0'）须先于 flexible 字母分支处理：
    # 否则 'v3' 会把 V 当 flexible 字母解析成 float64。
    if s.lower().startswith('void'):
        rest = s[4:]
        if rest == '' or rest.isdigit():
            return _make_flexible('V', int(rest) if rest else 0, bo)
    # numpy 2.x StringDType：'T'（变长字符串，无宽度；'T<N>' 非 numpy 记法）
    if s == 'T':
        return _make_flexible('T', 0, bo)
    # flexible with width: S3, U10, V3, a3, or bare S/U/V/a
    if s and s[0] in 'SUVa':
        rest = s[1:]
        if rest == '' or rest.isdigit():
            k = 'S' if s[0] == 'a' else s[0]
            width = int(rest) if rest else 0
            # numpy 语义：S/U itemsize 不得超 intc max（超 → TypeError
            # "data type ... not understood"；void 不受此限）
            if k in ('S', 'U') and \
                    width * (4 if k == 'U' else 1) > 2147483647:
                raise TypeError("data type %r not understood" % s)
            return _make_flexible(k, width, bo)
    if len(s) == 1 and s in _CHARCODE:
        return _make_scalar(_CHARCODE[s], bo)
    # datetime64/timedelta64 伪类型：M8[unit] / m8[unit]（itemsize 8，
    # 不参与实际数值构造，仅供 dtype 解析/提升/比较）。
    if (len(s) > 4 and s[1] == '8' and s[2] == '[' and s[-1] == ']'
            and s[0] in 'Mm' and s[3:-1].isalpha()):
        return _make_datetime(s[0], s[3:-1], bo)
    low = s.lower()
    if s in _KINDSIZE:
        return _make_scalar(_KINDSIZE[s], bo)
    if low in _KINDSIZE:
        return _make_scalar(_KINDSIZE[low], bo)
    if low in _SCALAR_INFO:
        return _make_scalar(low, bo)
    if low in _CANON_FULL:
        return _make_scalar(_CANON_FULL[low], bo)
    # 规范化名 str<N>/bytes<N>：历史实现写 N=itemsize，numpy 规范为 N=itemsize*8
    # （str4=U1、bytes3=S3 两种约定都要能往返，故双约定接受）。
    if low.startswith('str') and low[3:].isdigit():
        n = int(low[3:])
        if n % 32 == 0:
            return _make_flexible('U', n // 32, bo)
        if n % 4 == 0:
            return _make_flexible('U', n // 4, bo)
    if low.startswith('bytes') and low[5:].isdigit():
        n = int(low[5:])
        if n % 8 == 0:
            return _make_flexible('S', n // 8, bo)
        return _make_flexible('S', n, bo)
    if low in ('str', 'str_', 'unicode', 'unicode_'):
        return _make_flexible('U', 0, bo)
    if low in ('bytes', 'bytes_', 'string', 'string_'):
        return _make_flexible('S', 0, bo)
    if low in ('void', 'void_'):
        return _make_flexible('V', 0, bo)
    # void<N>（DType.name 对 V 类型的输出，N=itemsize）
    if low.startswith('void') and low[4:].isdigit():
        return _make_flexible('V', int(low[4:]), bo)
    raise TypeError("data type %r not understood" % s)


def _parse_field_type(tok):
    tok = tok.strip()
    if not tok:
        raise TypeError("empty dtype token")
    # 剥离字节序前缀（<, >, =, |）再检测子数组形式（leading-digit / 括号），
    # 否则 '<0i4' 这类"前缀+子数组计数"会漏检落到 _parse_scalar_token 报错。
    # base 还原前缀以保留字节序。
    _bo = ''
    if tok[0] in '<>=|':
        _bo = tok[0]
        tok = tok[1:]
    if tok[0] == '(':
        depth = 0
        end = 0
        for i, ch in enumerate(tok):
            if ch == '(':
                depth += 1
            elif ch == ')':
                depth -= 1
                if depth == 0:
                    end = i
                    break
        shape = _parse_shape(tok[1:end])
        rest = tok[end + 1:].strip()
        base = _parse_scalar_token(_bo + rest)
        return _make_subarray(base, shape)
    j = 0
    while j < len(tok) and tok[j].isdigit():
        j += 1
    if 0 < j < len(tok):
        cnt = int(tok[:j])
        base = _parse_scalar_token(_bo + tok[j:].strip())
        return _make_subarray(base, (cnt,))
    return _parse_scalar_token(_bo + tok)


def _parse_string_dtype(s, align):
    parts = _split_top_commas(s)
    if len(parts) == 1:
        return _parse_field_type(parts[0].strip())
    formats = [_parse_field_type(p.strip()) for p in parts]
    names = ['f%d' % i for i in range(len(formats))]
    return _build_struct(names, formats, None, None, None, align)


def _parse_list_dtype(lst, align):
    # 列表推导：一次提取 (name, fdt, title) 三元组
    def _parse_item(item):
        name_spec = item[0]
        typ = item[1]
        shape = item[2] if len(item) > 2 else None
        if isinstance(name_spec, (tuple, list)):
            title, name = name_spec[0], name_spec[1]
        else:
            name, title = name_spec, None
        fdt = _parse_any_dtype(typ, align)
        if shape is not None and shape != ():
            sh = (shape,) if isinstance(shape, int) else tuple(shape)
            fdt = _make_subarray(fdt, sh)
        return name, fdt, title

    parsed = [_parse_item(item) for item in lst]
    names = [p[0] for p in parsed]
    formats = [p[1] for p in parsed]
    titles = [p[2] for p in parsed]
    return _build_struct(names, formats, None, None, titles, align)


def _parse_tuple_dtype(obj, align):
    base = _parse_any_dtype(obj[0], align)
    shp = obj[1]
    # legacy 嵌套形式 (base, dtype字符串)（如 ('i8', 'i4,i4')）：外层 base
    # 决定存储类型（numpy 语义），字符串不可当子数组 shape 拆字符。
    if isinstance(shp, str):
        return base
    # numpy legacy 形式：S/U 柔性 dtype 的 (type, int) = (type, 字符数)
    # （np.dtype(('S', 5)) → S5、('U', 4) → U4）；负数或 itemsize 超 intc
    # max → ValueError "invalid itemsize in generic type tuple"。其余类型
    # 的 int 第二项仍按子数组 shape 处理。
    if isinstance(shp, int) and not isinstance(shp, bool) and \
            getattr(base, '_kind', None) in ('S', 'U'):
        _bk = base._kind
        if shp < 0 or shp * (4 if _bk == 'U' else 1) > 2147483647:
            raise ValueError("invalid itemsize in generic type tuple")
        return _make_flexible(_bk, shp, getattr(base, '_byteorder', None))
    shape = (shp,) if isinstance(shp, int) else tuple(shp)
    return _make_subarray(base, shape)


def _parse_dict_dtype(d, align):
    if 'names' in d and 'formats' in d:
        names = list(d['names'])
        formats = [_parse_any_dtype(f, align) for f in d['formats']]
        offsets = d.get('offsets', None)
        itemsize = d.get('itemsize', None)
        titles = d.get('titles', None)
        aligned = align or bool(d.get('aligned', False))
        return _build_struct(names, formats, offsets, itemsize, titles, aligned)

    # 字典推导：一次提取 (name, typ, offset, title) 四元组
    def _parse_dict_item(name, spec):
        typ = spec[0]
        offset = spec[1]
        title = spec[2] if len(spec) > 2 else None
        return name, _parse_any_dtype(typ, align), offset, title

    parsed = [_parse_dict_item(name, spec) for name, spec in d.items()]
    names = [p[0] for p in parsed]
    formats = [p[1] for p in parsed]
    offsets = [p[2] for p in parsed]
    titles = [p[3] for p in parsed]
    return _build_struct(names, formats, offsets, None, titles, align)


def _resolve_scalar_obj(obj):
    if obj is int:
        return _make_scalar('int64', None)
    if obj is float:
        return _make_scalar('float64', None)
    if obj is complex:
        return _make_scalar('complex128', None)
    if obj is bool:
        return _make_scalar('bool', None)
    if obj is bytes:
        return _make_flexible('S', 0, None)
    if obj is str:
        return _make_flexible('U', 0, None)
    if obj is object:
        return _make_scalar('object', None)
    if isinstance(obj, type):
        # 协议：类级 __rsnumpy_dtype__ 声明该类对应的 dtype（rsnumpy 无
        # 用户自定义 C 标量注册机制，测试标量类用它声明近似 dtype）。
        _rdt = getattr(obj, '__rsnumpy_dtype__', None)
        if _rdt is not None:
            return _parse_any_dtype(_rdt, False)
    name = getattr(obj, '__name__', None)
    if name is None:
        name = getattr(obj, 'name', None)
    if name is None:
        name = str(obj)
    return _parse_scalar_token(name)


def _parse_any_dtype(obj, align):
    if isinstance(obj, DType):
        return obj
    # ndarray 输入：np.dtype(arr) = arr.dtype（numpy 语义）
    if not isinstance(obj, (str, bytes, type, list, tuple, dict)) \
            and hasattr(obj, '_array'):
        return dtype(obj.dtype, align)
    if isinstance(obj, str):
        return _parse_string_dtype(obj.strip(), align)
    if isinstance(obj, list):
        return _parse_list_dtype(obj, align)
    if isinstance(obj, tuple):
        return _parse_tuple_dtype(obj, align)
    if isinstance(obj, dict):
        return _parse_dict_dtype(obj, align)
    if obj is None:
        # numpy 语义：dtype(None) = object dtype
        return _make_scalar('object', None)
    # numpy dtype-like
    if type(obj).__name__ == 'dtype' and hasattr(obj, 'names') and obj.names:
        # 兼容 numpy 结构化 dtype（罕见路径）
        return _parse_string_dtype(str(obj), align)
    return _resolve_scalar_obj(obj)


def dtype(obj, align=False, copy=False):
    """创建 dtype 对象（兼容 numpy.dtype 构造语义）。"""
    if isinstance(obj, DType):
        if align and obj._names is not None and not obj._aligned:
            names = list(obj._names)
            formats = [obj._fields[n][0] for n in names]
            titles = [obj._fields[n][2] if len(obj._fields[n]) > 2 else None for n in names]
            return _build_struct(names, formats, None, None, titles, True)
        return obj
    return _parse_any_dtype(obj, align)


# ---------- str/repr 辅助 ----------
def _norm_bo(dt):
    b = dt._byteorder
    if b == '|':
        return '|'
    if b == '=':
        return _NATIVE
    return b


def _is_native(dt):
    return dt._byteorder in ('|', '=') or dt._byteorder == _NATIVE


def _byteorder_prefix(dt):
    b = dt._byteorder
    if b == '|':
        return ''
    if b == '=':
        return _NATIVE
    return b


def _typestr(dt):
    """array-protocol typestr，如 '<i4' / '|S3' / '<U10'。"""
    if dt._byteorder == '|':
        p = '|'
    elif dt._byteorder == '=':
        p = _NATIVE
    else:
        p = dt._byteorder
    k = dt._kind
    if k == 'b':
        return p + 'b1'
    if k == 'O':
        return p + 'O'
    if k == 'S':
        return p + 'S%d' % dt._itemsize
    if k == 'U':
        return p + 'U%d' % (dt._itemsize // 4)
    if k == 'T':
        return p + 'T'
    if k == 'V':
        return p + 'V%d' % dt._itemsize
    if k in ('M', 'm'):
        # datetime64/timedelta64：typestr 含单位，如 '<M8[ns]'
        return p + (dt._typename or (k + '8'))
    return p + '%s%d' % (k, dt._itemsize)


def _scalar_typestr_short(dt):
    """短类型码：'<i4' / 'i1' / 'S3' / '?' / 'O'。"""
    k = dt._kind
    if k == 'b':
        return '?'
    if k == 'O':
        return 'O'
    if k == 'T':
        return 'T'
    p = _byteorder_prefix(dt)
    if k == 'S':
        return 'S%d' % dt._itemsize
    if k == 'U':
        return '%sU%d' % (p, dt._itemsize // 4)
    if k == 'V':
        return 'V%d' % dt._itemsize
    return '%s%s%d' % (p, k, dt._itemsize)


def _scalar_short_quoted(dt):
    return "'%s'" % _scalar_typestr_short(dt)


def _scalar_str(dt):
    """str(dtype) 用于标量。"""
    k = dt._kind
    if k in ('S', 'U', 'V', 'T') or not _is_native(dt):
        return _typestr(dt)
    return dt.name


def _scalar_repr_arg(dt):
    """repr(dtype) 内部参数（引号内）用于标量。"""
    k = dt._kind
    if k == 'b':
        return "'bool'"
    if k == 'O':
        return "'O'"
    if k == 'S':
        return "'S'" if dt._itemsize == 0 else "'S%d'" % dt._itemsize
    if k == 'U':
        p = _byteorder_prefix(dt)
        return "'%sU'" % p if dt._itemsize == 0 else "'%sU%d'" % (p, dt._itemsize // 4)
    if k == 'V':
        return "'V'" if dt._itemsize == 0 else "'V%d'" % dt._itemsize
    if k == 'T':
        return "'T'"
    if _is_native(dt):
        return "'%s'" % dt._typename
    return "'%s'" % _scalar_typestr_short(dt)


def _shape_repr(shape):
    return repr(tuple(shape))


def _unpack_field(v):
    if len(v) > 2:
        return v[0], v[1], v[2]
    return v[0], v[1], None


def _type_repr_in_list(fdt):
    if fdt._names is not None:
        return _struct_str(fdt, True)
    if fdt._subdtype is not None:
        base, shape = fdt._subdtype
        return "%s, %s" % (_type_repr_in_list(base), _shape_repr(shape))
    return _scalar_short_quoted(fdt)


def _struct_list_str(dt):
    # 列表推导代替显式 for 循环
    def _entry(name):
        fdt, _off, title = _unpack_field(dt._fields[name])
        namestr = repr((title, name)) if title is not None else repr(name)
        return "(%s, %s)" % (namestr, _type_repr_in_list(fdt))
    parts = [_entry(name) for name in dt._names]
    return "[" + ", ".join(parts) + "]"


def _fmt_format_entry(fdt):
    if fdt._names is not None:
        return _struct_str(fdt, True)
    if fdt._subdtype is not None:
        base, shape = fdt._subdtype
        return "(%s, %s)" % (_fmt_format_entry(base), _shape_repr(shape))
    return _scalar_short_quoted(fdt)


def _struct_dict_str(dt, include_align):
    names = dt._names
    # 一次性解包所有字段，使用列表推导代替显式 for 循环
    unpacked = [_unpack_field(dt._fields[name]) for name in names]
    fld_dtypes = [u[0] for u in unpacked]
    offsets = [u[1] for u in unpacked]
    titles = [u[2] for u in unpacked]
    ret = "{'names': [" + ", ".join(repr(n) for n in names) + "]"
    ret += ", 'formats': [" + ", ".join(_fmt_format_entry(f) for f in fld_dtypes) + "]"
    ret += ", 'offsets': [" + ", ".join("%d" % o for o in offsets) + "]"
    if builtins.any(t is not None for t in titles):
        ret += ", 'titles': [" + ", ".join(repr(t) for t in titles) + "]"
    ret += ", 'itemsize': %d" % dt._itemsize
    if include_align and dt._aligned:
        ret += ", 'aligned': True"
    ret += "}"
    return ret


def _is_packed(dt):
    formats = [dt._fields[n][0] for n in dt._names]
    exp_offsets, exp_itemsize = _layout_fields(formats, dt._aligned)
    actual_offsets = [dt._fields[n][1] for n in dt._names]
    return exp_offsets == actual_offsets and exp_itemsize == dt._itemsize


def _struct_str(dt, include_align):
    if not (include_align and dt._aligned) and _is_packed(dt):
        return _struct_list_str(dt)
    return _struct_dict_str(dt, include_align)


def _subarray_str(dt):
    base, shape = dt._subdtype
    if base._names is not None:
        base_str = _struct_str(base, True)
    elif base._subdtype is not None:
        base_str = _subarray_str(base)
    else:
        base_str = _scalar_short_quoted(base)
    return "(%s, %s)" % (base_str, _shape_repr(shape))


def _has_object(dt):
    if dt._kind == 'O':
        return True
    if dt._subdtype is not None:
        return _has_object(dt._subdtype[0])
    if dt._names is not None:
        return any(_has_object(dt._fields[n][0]) for n in dt._names)
    return False
