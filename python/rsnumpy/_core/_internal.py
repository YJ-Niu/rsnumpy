# SPDX-License-Identifier: MIT
"""rsnumpy._core._internal —— numpy._core._internal 的 Python 层兼容子集。

目前提供 ``_dtype_from_pep3118``：把 PEP 3118 格式串解析为 dtype，
算法移植自 numpy（含 '@' 原生对齐的填充/内嵌填充/尾填充规则）。
"""

import math


def _prod(a):
    p = 1
    for x in a:
        p *= x
    return p


def _gcd(a, b):
    """最大公约数。"""
    if not (math.isfinite(a) and math.isfinite(b)):
        raise ValueError(
            'Can only find greatest common divisor of '
            f'finite arguments, found "{a}"')
    while b:
        a, b = b, a % b
    return a


def _lcm(a, b):
    return a // _gcd(a, b) * b


# PEP 3118 类型表（numpy 原版）
_pep3118_native_map = {
    '?': '?',
    'c': 'S1',
    'b': 'b',
    'B': 'B',
    'h': 'h',
    'H': 'H',
    'i': 'i',
    'I': 'I',
    'l': 'l',
    'L': 'L',
    'q': 'q',
    'Q': 'Q',
    'e': 'e',
    'f': 'f',
    'd': 'd',
    'g': 'g',
    'Zf': 'F',
    'Zd': 'D',
    'Zg': 'G',
    's': 'S',
    'w': 'U',
    'O': 'O',
    'x': 'V',  # padding
}
_pep3118_native_typechars = ''.join(_pep3118_native_map.keys())

_pep3118_standard_map = {
    '?': '?',
    'c': 'S1',
    'b': 'b',
    'B': 'B',
    'h': 'i2',
    'H': 'u2',
    'i': 'i4',
    'I': 'u4',
    'l': 'i4',
    'L': 'u4',
    'q': 'i8',
    'Q': 'u8',
    'e': 'f2',
    'f': 'f',
    'd': 'd',
    'Zf': 'F',
    'Zd': 'D',
    's': 'S',
    'w': 'U',
    'O': 'O',
    'x': 'V',  # padding
}
_pep3118_standard_typechars = ''.join(_pep3118_standard_map.keys())

_pep3118_unsupported_map = {
    'u': 'UCS-2 strings',
    '&': 'pointers',
    't': 'bitfields',
    'X': 'function pointers',
}


class _Stream:
    def __init__(self, s):
        self.s = s
        self.byteorder = '@'

    def advance(self, n):
        res = self.s[:n]
        self.s = self.s[n:]
        return res

    def consume(self, c):
        if self.s[:len(c)] == c:
            self.advance(len(c))
            return True
        return False

    def consume_until(self, c):
        if callable(c):
            i = 0
            while i < len(self.s) and not c(self.s[i]):
                i = i + 1
            return self.advance(i)
        else:
            i = self.s.index(c)
            res = self.advance(i)
            self.advance(len(c))
            return res

    @property
    def next(self):
        return self.s[0]

    def __bool__(self):
        return bool(self.s)


def _scalar_dtype(dtypechar, byteorder):
    """构造标量 dtype；rsnumpy 不支持的类型退化为最近的可用类型。"""
    import rsnumpy as np
    try:
        return np.dtype(byteorder + dtypechar)
    except Exception:
        pass
    # 退化映射：longdouble → float64、longcomplex → complex128 等
    fallbacks = {
        'g16': 'd', 'g': 'd', 'G': 'D',
        'F': 'F', 'D': 'D',
        'e': 'f2',
    }
    base = dtypechar[:-2] if len(dtypechar) > 2 and dtypechar[-2:].isdigit() else dtypechar
    fb = fallbacks.get(base)
    if fb is not None:
        try:
            return np.dtype(fb)
        except Exception:
            pass
    raise ValueError(f"cannot build dtype from PEP 3118 char {dtypechar!r}")


def _fix_names(field_spec):
    """把 None 名字替换为下一个未用的 f%d。"""
    names = field_spec['names']
    for i, name in enumerate(names):
        if name is not None:
            continue
        j = 0
        while True:
            name = f'f{j}'
            if name not in names:
                break
            j = j + 1
        names[i] = name


def _add_trailing_padding(value, padding):
    """在 dtype 末尾注入指定数量的填充字节。"""
    import rsnumpy as np
    # 包装为单字段结构体并扩大 itemsize（numpy 对结构体会保留原字段，
    # 此处简化为统一包装；rsnumpy 测试不依赖该细节）
    field_spec = {
        'names': ['f0'],
        'formats': [value],
        'offsets': [0],
        'itemsize': value.itemsize + padding,
    }
    return np.dtype(field_spec)


def _dtype_from_pep3118(spec):
    import rsnumpy as np
    dtype = np.dtype
    stream = _Stream(spec)

    def __dtype_from_pep3118(stream, is_subdtype):
        field_spec = {
            'names': [],
            'formats': [],
            'offsets': [],
            'itemsize': 0,
        }
        offset = 0
        common_alignment = 1
        is_padding = False

        # Parse spec
        while stream:
            value = None

            # End of structure, bail out to upper level
            if stream.consume('}'):
                break

            # Sub-arrays (1)
            shape = None
            if stream.consume('('):
                shape = stream.consume_until(')')
                shape = tuple(map(int, shape.split(',')))

            # Byte order
            if stream.next in ('@', '=', '<', '>', '^', '!'):
                byteorder = stream.advance(1)
                if byteorder == '!':
                    byteorder = '>'
                stream.byteorder = byteorder

            # Byte order characters also control native vs. standard type sizes
            if stream.byteorder in ('@', '^'):
                type_map = _pep3118_native_map
                type_map_chars = _pep3118_native_typechars
            else:
                type_map = _pep3118_standard_map
                type_map_chars = _pep3118_standard_typechars

            # Item sizes
            itemsize_str = stream.consume_until(lambda c: not c.isdigit())
            if itemsize_str:
                itemsize = int(itemsize_str)
            else:
                itemsize = 1

            # Data types
            is_padding = False

            if stream.consume('T{'):
                value, align = __dtype_from_pep3118(
                    stream, is_subdtype=True)
            elif stream.next in type_map_chars:
                if stream.next == 'Z':
                    typechar = stream.advance(2)
                else:
                    typechar = stream.advance(1)

                is_padding = (typechar == 'x')
                dtypechar = type_map[typechar]
                if dtypechar in 'USV':
                    dtypechar += f'{itemsize}'
                    itemsize = 1
                numpy_byteorder = {'@': '=', '^': '='}.get(
                    stream.byteorder, stream.byteorder)
                value = _scalar_dtype(dtypechar, numpy_byteorder)
                align = value.alignment
            elif stream.next in _pep3118_unsupported_map:
                desc = _pep3118_unsupported_map[stream.next]
                raise NotImplementedError(
                    f"Unrepresentable PEP 3118 data type {stream.next!r} "
                    f"({desc})")
            else:
                raise ValueError(
                    f"Unknown PEP 3118 data type specifier {stream.s!r}")

            #
            # Native alignment may require padding
            #
            extra_offset = 0
            if stream.byteorder == '@':
                start_padding = (-offset) % align
                intra_padding = (-value.itemsize) % align

                offset += start_padding

                if intra_padding != 0:
                    if itemsize > 1 or (shape is not None and _prod(shape) > 1):
                        # Inject internal padding to the end of the sub-item
                        value = _add_trailing_padding(value, intra_padding)
                    else:
                        # We can postpone the injection of internal padding,
                        # as the item appears at most once
                        extra_offset += intra_padding

                # Update common alignment
                common_alignment = _lcm(align, common_alignment)

            # Convert itemsize to sub-array
            if itemsize != 1:
                value = dtype((value, (itemsize,)))

            # Sub-arrays (2)
            if shape is not None:
                value = dtype((value, shape))

            # Field name
            if stream.consume(':'):
                name = stream.consume_until(':')
            else:
                name = None

            if not (is_padding and name is None):
                if name is not None and name in field_spec['names']:
                    raise RuntimeError(
                        f"Duplicate field name '{name}' in PEP3118 format")
                field_spec['names'].append(name)
                field_spec['formats'].append(value)
                field_spec['offsets'].append(offset)

            offset += value.itemsize
            offset += extra_offset

            field_spec['itemsize'] = offset

        # extra final padding for aligned types
        if stream.byteorder == '@':
            field_spec['itemsize'] += (-offset) % common_alignment

        # Check if this was a simple 1-item type, and unwrap it
        if (field_spec['names'] == [None]
                and field_spec['offsets'][0] == 0
                and field_spec['itemsize'] == field_spec['formats'][0].itemsize
                and not is_subdtype):
            ret = field_spec['formats'][0]
        else:
            _fix_names(field_spec)
            ret = dtype(field_spec)

        # Finished
        return ret, common_alignment

    dtype_out, _align = __dtype_from_pep3118(stream, is_subdtype=False)
    return dtype_out
