"""完整的 dtype 语义实现（解析、属性、str/repr），逐字节对齐 numpy。

DType 采用富数据模型，支持标量、结构化、子数组、标题、对齐等全部语法，
并复刻 numpy 的 dtype str/repr 生成规则。
"""

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


class DType:
    """元素数据类型，兼容 numpy.dtype 的属性与 str/repr。"""

    __slots__ = ('_kind', '_itemsize', '_byteorder', '_alignment', '_typename',
                 '_names', '_fields', '_aligned', '_subdtype', '_titles')

    def __init__(self, name=None, fields=None):
        # 向后兼容：DType('int64') / DType(<其他>) 构造标量。
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
        if name is not None:
            src = _parse_any_dtype(name, False)
            self._copy_from(src)

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

    # ---------- 属性 ----------
    @property
    def kind(self):
        return self._kind

    @property
    def itemsize(self):
        return self._itemsize

    @property
    def alignment(self):
        return self._alignment

    @property
    def byteorder(self):
        return self._byteorder

    @property
    def names(self):
        return self._names

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
            return 'void%d' % (self._itemsize * 8)
        if self._subdtype is not None:
            return 'void%d' % (self._itemsize * 8)
        if self._kind == 'S':
            return 'bytes%d' % (self._itemsize * 8)
        if self._kind == 'U':
            return 'str%d' % (self._itemsize * 8)
        if self._kind == 'V':
            return 'void%d' % (self._itemsize * 8)
        return self._typename

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
                    return (n, fdt.descr)
                else:
                    return (n, _typestr(fdt))
            # 使用列表推导代替显式 for 循环
            return [_field_descr(n) for n in self._names]
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
        return ('scalar', self._kind, self._itemsize, _norm_bo(self))

    def __eq__(self, other):
        if isinstance(other, DType):
            return self._eqkey() == other._eqkey()
        if other is None:
            return False
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
    else:  # S / V
        d._itemsize = width
        d._alignment = 1
        d._byteorder = '|'
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
    # flexible with width: S3, U10, V3, a3, or bare S/U/V/a
    if s and s[0] in 'SUVa':
        rest = s[1:]
        if rest == '' or rest.isdigit():
            k = 'S' if s[0] == 'a' else s[0]
            width = int(rest) if rest else 0
            return _make_flexible(k, width, bo)
    if len(s) == 1 and s in _CHARCODE:
        return _make_scalar(_CHARCODE[s], bo)
    low = s.lower()
    if s in _KINDSIZE:
        return _make_scalar(_KINDSIZE[s], bo)
    if low in _KINDSIZE:
        return _make_scalar(_KINDSIZE[low], bo)
    if low in _SCALAR_INFO:
        return _make_scalar(low, bo)
    if low in _CANON_FULL:
        return _make_scalar(_CANON_FULL[low], bo)
    if low in ('str', 'str_', 'unicode', 'unicode_'):
        return _make_flexible('U', 0, bo)
    if low in ('bytes', 'bytes_', 'string', 'string_'):
        return _make_flexible('S', 0, bo)
    raise TypeError("data type %r not understood" % s)


def _parse_field_type(tok):
    tok = tok.strip()
    if not tok:
        raise TypeError("empty dtype token")
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
        base = _parse_scalar_token(rest)
        return _make_subarray(base, shape)
    j = 0
    while j < len(tok) and tok[j].isdigit():
        j += 1
    if 0 < j < len(tok):
        cnt = int(tok[:j])
        base = _parse_scalar_token(tok[j:].strip())
        return _make_subarray(base, (cnt,))
    return _parse_scalar_token(tok)


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
    name = getattr(obj, '__name__', None)
    if name is None:
        name = getattr(obj, 'name', None)
    if name is None:
        name = str(obj)
    return _parse_scalar_token(name)


def _parse_any_dtype(obj, align):
    if isinstance(obj, DType):
        return obj
    if isinstance(obj, str):
        return _parse_string_dtype(obj.strip(), align)
    if isinstance(obj, list):
        return _parse_list_dtype(obj, align)
    if isinstance(obj, tuple):
        return _parse_tuple_dtype(obj, align)
    if isinstance(obj, dict):
        return _parse_dict_dtype(obj, align)
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
    if k == 'V':
        return p + 'V%d' % dt._itemsize
    return p + '%s%d' % (k, dt._itemsize)


def _scalar_typestr_short(dt):
    """短类型码：'<i4' / 'i1' / 'S3' / '?' / 'O'。"""
    k = dt._kind
    if k == 'b':
        return '?'
    if k == 'O':
        return 'O'
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
    if k in ('S', 'U', 'V') or not _is_native(dt):
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
    if any(t is not None for t in titles):
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
