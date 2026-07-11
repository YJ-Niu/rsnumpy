"""I/O 模块 - 所有实现位于 Rust，这里仅保留薄包装。"""

import rsnumpy._core as _core


def _get_ndarray():
    from .__init__ import ndarray as _n
    return _n


def _ensure_raw(a):
    if hasattr(a, '_array'):
        return a._array
    return _core.ndarray(a)


def _wrap(result):
    return _get_ndarray()(result)


def save(file, arr):
    """将数组保存为二进制文件 (.npy)。"""
    _core.save_npy(file, _ensure_raw(arr))


class NpzFile:
    """用于表示加载的 .npz 文件的类。"""
    def __init__(self, data):
        self._data = data
    
    @property
    def files(self):
        """返回文件中所有数组的名称列表。"""
        return list(self._data.keys())
    
    def __getitem__(self, key):
        """通过键名获取数组。"""
        return self._data[key]


def load(file, mmap_mode=None, allow_pickle=False, fix_imports=True, encoding='ASCII'):
    """从二进制文件加载数组 (.npy 或 .npz)。"""
    if file.endswith('.npz'):
        return NpzFile(load_npz(file))
    return _wrap(_core.load_npy(file))


def savetxt(fname, X, fmt='%.18e', delimiter=' ', newline='\n', header='',
            footer='', comments='# ', encoding=None):
    """将数组保存为文本文件。"""
    _core.save_text(fname, _ensure_raw(X), fmt, delimiter)


def loadtxt(fname, dtype=float, comments='#', delimiter=None, converters=None,
            skiprows=0, usecols=None, unpack=False, ndmin=0, encoding='bytes',
            max_rows=None, *, like=None):
    """从文本文件加载数据。"""
    if delimiter is None:
        delimiter = ""
    result = _wrap(_core.load_text(fname, delimiter, skiprows))
    if dtype == int:
        result = result.astype("int")
    elif dtype == float:
        pass
    elif isinstance(dtype, type):
        result = result.astype(dtype.__name__)
    else:
        result = result.astype(dtype)
    if len(result) == 1:
        return result[0]
    else:
        return result


def _is_string_dtype(dt):
    """判断 dtype 是否为字符串类型（如 str、'|U5'、'<U10'、'S20'）。"""
    if dt is str or dt is bytes:
        return True
    if isinstance(dt, str):
        s = dt.lstrip('|<>=')
        if s and s[0] in ('U', 'S'):
            return True
        if dt in ('str', 'unicode', 'bytes', 'str_', 'bytes_'):
            return True
    return False


def _string_width(dt):
    """从字符串 dtype 中解析固定宽度（如 '|U5' -> 5），无宽度返回 None。"""
    if not isinstance(dt, str):
        return None
    s = dt.lstrip('|<>=')
    if s and s[0] in ('U', 'S') and s[1:].isdigit():
        return int(s[1:])
    return None


def _field_code_from_type(t):
    """将字段类型解析为结构化字段码：'i8'（整型/布尔）或 'f8'（其余）。"""
    from .__init__ import _resolve_type_name
    name = _resolve_type_name(t)
    if isinstance(name, str):
        low = name.lower()
        if low.startswith('int') or low.startswith('uint') or low.startswith('bool'):
            return 'i8'
    return 'f8'


def _classify_gft_dtype(dtype):
    """分类 genfromtxt 的 dtype 参数。

    返回 (kind, names, codes)：
    - kind='struct'：结构化（names 为字段名列表或 None，codes 为每列字段码）。
    - kind='scalar'：标量 dtype，names=None，codes 为单个字段码字符串。
    """
    if isinstance(dtype, list):
        names = [f[0] for f in dtype]
        codes = [_field_code_from_type(f[1]) for f in dtype]
        return ('struct', names, codes)
    if isinstance(dtype, tuple):
        return ('struct', None, [_field_code_from_type(t) for t in dtype])
    return ('scalar', None, _field_code_from_type(dtype))


def _split_fields(line, delimiter):
    """按 numpy LineSplitter 规则将一行拆成字段。

    - delimiter=None：先去除首尾换行，再按空白拆分。
    - 整数：定宽切片（空格有意义，仅去除换行）。
    - 序列：按累积宽度切片（仅去除换行）。
    - 字符串：去除首尾空白与换行后按分隔符拆分。
    空行返回 []。
    """
    if delimiter is None:
        line = line.strip("\r\n")
        return line.split() if line else []
    if isinstance(delimiter, int):
        line = line.strip("\r\n")
        if not line:
            return []
        return [line[i:i + delimiter] for i in range(0, len(line), delimiter)]
    if isinstance(delimiter, (list, tuple)):
        line = line.strip("\r\n")
        if not line:
            return []
        parts = []
        pos = 0
        for w in delimiter:
            parts.append(line[pos:pos + w])
            pos += w
        return parts
    line = line.strip(" \r\n")
    return line.split(delimiter) if line else []


def genfromtxt(fname, dtype=float, comments='#', delimiter=None, skip_header=0,
               skip_footer=0, converters=None, missing_values=None,
               filling_values=None, usecols=None, names=None, excludelist=None,
               deletechars=None, replace_space='_', autostrip=False,
               case_sensitive=True, defaultfmt='f%i', unpack=None, usemask=False,
               loose=True, invalid_raise=True, max_rows=None, encoding=None):
    """从文本文件或类文件对象加载数据，缺失值填充为 NaN（兼容 numpy.genfromtxt）。"""
    if hasattr(fname, 'read'):
        content = fname.read()
        if isinstance(content, bytes):
            content = content.decode('utf-8')
    else:
        with open(fname, 'r') as fh:
            content = fh.read()

    lines = content.splitlines()
    if skip_header:
        lines = lines[skip_header:]
    if skip_footer:
        lines = lines[:len(lines) - skip_footer]

    string_dtype = _is_string_dtype(dtype)
    str_width = _string_width(dtype) if string_dtype else None

    # 解析 names：字符串 "a, b, c"、序列，或 True（从首行读取，去除前导注释符）。
    name_list = None
    if isinstance(names, str):
        name_list = [n.strip() for n in names.split(',')]
    elif isinstance(names, (list, tuple)):
        name_list = [str(n).strip() for n in names]
    elif names is True:
        for idx, line in enumerate(lines):
            hl = line
            if comments and hl.lstrip().startswith(comments):
                hl = hl.lstrip()[len(comments):]
            header_parts = _split_fields(hl, delimiter)
            if header_parts:
                name_list = [p.strip() for p in header_parts]
                lines = lines[idx + 1:]
                break

    # dtype 分类（字符串 dtype 单独处理，不在此结构化）。
    if string_dtype:
        dt_kind, dt_names, dt_codes = ('scalar', None, None)
    else:
        dt_kind, dt_names, dt_codes = _classify_gft_dtype(dtype)
    is_struct = (not string_dtype) and (dt_kind == 'struct' or name_list is not None)

    if usecols is not None and not isinstance(usecols, (list, tuple)):
        usecols = (usecols,)

    # 第一遍：按行拆成字符串字段（去注释、拆分）。
    str_rows = []
    count = 0
    for line in lines:
        if comments:
            ci = line.find(comments)
            if ci != -1:
                line = line[:ci]
        parts = _split_fields(line, delimiter)
        if not parts:
            continue
        str_rows.append(parts)
        count += 1
        if max_rows is not None and count >= max_rows:
            break

    ndarray_cls = _get_ndarray()
    total_ncols = len(str_rows[0]) if str_rows else 0

    # 完整字段名：提供的 names 填充前若干列，其余用 defaultfmt 编号（从 0 计未命名列）。
    full_names = None
    if name_list is not None:
        full_names = []
        k = 0
        for i in range(total_ncols):
            if i < len(name_list):
                full_names.append(name_list[i])
            else:
                full_names.append(defaultfmt % k)
                k += 1
    elif dt_names is not None:
        full_names = list(dt_names)
    elif is_struct:
        full_names = [defaultfmt % i for i in range(total_ncols)]

    # 完整字段码（整型/浮点）。
    if dt_kind == 'struct' and dt_codes is not None:
        full_codes = list(dt_codes)
    elif is_struct:
        full_codes = ['f8'] * total_ncols
    else:
        full_codes = None

    # 解析 usecols（字符串列名→索引，负索引归一）。
    if usecols is not None:
        sel_indices = []
        for c in usecols:
            if isinstance(c, str):
                sel_indices.append(full_names.index(c))
            elif c < 0:
                sel_indices.append(total_ncols + c)
            else:
                sel_indices.append(c)
    else:
        sel_indices = list(range(total_ncols))

    # converters：键（列名或索引）映射到完整列索引。
    conv_by_idx = {}
    if converters:
        for key, fn in converters.items():
            if isinstance(key, str):
                conv_by_idx[full_names.index(key)] = fn
            elif key < 0:
                conv_by_idx[total_ncols + key] = fn
            else:
                conv_by_idx[key] = fn

    # 字符串 dtype 分支：保留字段文本。
    if string_dtype:
        rows = []
        for parts in str_rows:
            sel = [parts[i] for i in sel_indices]
            if autostrip:
                sel = [p.strip() for p in sel]
            if str_width is not None:
                sel = [p[:str_width] for p in sel]
            rows.append(sel)
        if rows and all(len(r) == 1 for r in rows):
            data = [r[0] for r in rows]
        elif len(rows) == 1:
            data = rows[0]
        else:
            data = rows
        return ndarray_cls(data)

    def parse_val(raw, ci, code):
        if autostrip:
            raw = raw.strip()
        if ci in conv_by_idx:
            v = conv_by_idx[ci](raw)
        else:
            s = raw.strip()
            if s == '':
                v = float('nan')
            else:
                try:
                    v = float(s)
                except ValueError:
                    v = float('nan')
        if code == 'i8':
            try:
                return int(round(float(v)))
            except (ValueError, OverflowError):
                return v
        return float(v)

    # 结构化数组分支。
    if is_struct:
        out_names = [full_names[i] for i in sel_indices]
        out_codes = [full_codes[i] for i in sel_indices]
        fields = list(zip(out_names, out_codes))
        raw = [
            tuple(parse_val(parts[ci], ci, out_codes[p])
                  for p, ci in enumerate(sel_indices))
            for parts in str_rows
        ]
        return ndarray_cls(raw, _dtype='void', _fields=fields, _raw_data=raw)

    # 非结构化（标量 dtype）：先按浮点解析，必要时 astype。
    rows = [
        [parse_val(parts[ci], ci, 'f8') for ci in sel_indices]
        for parts in str_rows
    ]
    # 单列数据压成一维，与 numpy 一致（如逐行单值 -> [0. 1. 2. ...]）。
    if rows and all(len(r) == 1 for r in rows):
        data = [r[0] for r in rows]
    elif len(rows) == 1:
        data = rows[0]
    else:
        data = rows

    result = ndarray_cls(data)
    if dtype == int or dtype == 'int':
        result = result.astype('int')
    elif dtype not in (float, 'float', None):
        result = result.astype(dtype.__name__ if isinstance(dtype, type) else dtype)

    if unpack:
        return result.T
    return result


def savez(file, *args, **kwds):
    """将多个数组保存为未压缩的 .npz 文件。"""
    arrays = []
    names = []
    for name in sorted(kwds.keys()):
        arrays.append(_ensure_raw(kwds[name]))
        names.append(name)
    for i, arg in enumerate(args):
        arrays.append(_ensure_raw(arg))
        names.append('arr_%d' % i)
    _core.savez_npz(file, arrays, names)


def _is_integer_array(arr):
    """检查数组是否应该是整数类型。"""
    raw_list = arr.tolist()
    
    def check_nested(data):
        if isinstance(data, list):
            return all(check_nested(x) for x in data)
        return abs(data - round(data)) < 1e-10
    
    return check_nested(raw_list)


def load_npz(file):
    """从 .npz 文件加载数组（返回 dict）。"""
    pairs = _core.load_npz(file)
    result = {}
    ndarray_cls = _get_ndarray()
    for key, raw in pairs:
        if _is_integer_array(raw):
            result[key] = ndarray_cls._wrap(raw, _dtype='int64')
        else:
            result[key] = ndarray_cls._wrap(raw, _dtype='float64')
    return result
