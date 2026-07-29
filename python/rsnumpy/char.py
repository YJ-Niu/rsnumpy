"""rsnumpy.char - 向量化字符串操作模块

提供对字符串数组进行向量化操作的函数。
"""


def _get_strings(a):
    """从 ndarray 或列表中提取字符串列表。"""
    if isinstance(a, str):
        return [a]
    if isinstance(a, bytes):
        return [a]
    if hasattr(a, '_raw_data'):
        return list(a._raw_data)
    if hasattr(a, '_array') or hasattr(a, 'tolist'):
        return list(a.tolist())
    return list(a)


def _make_result(strings):
    """将字符串列表包装为 ndarray。"""
    from .__init__ import ndarray
    return ndarray._wrap_raw(strings)


def add(x1, x2):
    """对两个数组的逐个字符串元素进行连接。"""
    s1 = _get_strings(x1)
    s2 = _get_strings(x2)
    result = [a + b for a, b in zip(s1, s2)]
    if isinstance(x1, str) and isinstance(x2, str):
        return result[0]
    return _make_result(result)


def multiply(a, i):
    """返回按元素多重连接后的字符串。"""
    s = _get_strings(a)
    result = [x * i for x in s]
    if isinstance(a, str):
        return result[0]
    return _make_result(result)


def center(a, width, fillchar=' '):
    """居中字符串。"""
    s = _get_strings(a)
    result = [x.center(width, fillchar) for x in s]
    if isinstance(a, str):
        return result[0]
    return _make_result(result)


def capitalize(a):
    """将字符串第一个字母转换为大写。"""
    s = _get_strings(a)
    result = [x.capitalize() for x in s]
    if isinstance(a, str):
        return result[0]
    return _make_result(result)


def title(a):
    """将字符串的每个单词的第一个字母转换为大写。"""
    s = _get_strings(a)
    result = [x.title() for x in s]
    if isinstance(a, str):
        return result[0]
    return _make_result(result)


def lower(a):
    """数组元素转换为小写。"""
    s = _get_strings(a)
    result = [x.lower() for x in s]
    if isinstance(a, str):
        return result[0]
    return _make_result(result)


def upper(a):
    """数组元素转换为大写。"""
    s = _get_strings(a)
    result = [x.upper() for x in s]
    if isinstance(a, str):
        return result[0]
    return _make_result(result)


def split(a, sep=None, maxsplit=None):
    """指定分隔符对字符串进行分割，并返回数组列表。"""
    s = _get_strings(a)
    if maxsplit is not None:
        result = [x.split(sep, maxsplit) for x in s]
    else:
        result = [x.split(sep) for x in s]
    if isinstance(a, str):
        return result[0]
    return _make_result(result)


def splitlines(a, keepends=False):
    """返回元素中的行列表，以换行符分割。"""
    s = _get_strings(a)
    result = [x.splitlines(keepends) for x in s]
    if isinstance(a, str):
        return result[0]
    return _make_result(result)


def strip(a, chars=None):
    """移除元素开头或者结尾处的特定字符。"""
    s = _get_strings(a)
    result = [x.strip(chars) for x in s]
    if isinstance(a, str):
        return result[0]
    return _make_result(result)


def join(sep, a):
    """通过指定分隔符来连接数组中的元素或字符串。"""
    seps = _get_strings(sep)
    s = _get_strings(a)
    n_seps = len(seps)

    def _join_one(i, x):
        current_sep = seps[i % n_seps]
        if isinstance(x, (list, tuple)):
            return current_sep.join(x)
        if isinstance(x, str):
            return current_sep.join(x)
        return current_sep.join(str(c) for c in x)

    # 列表推导代替显式 for 循环
    result = [_join_one(i, x) for i, x in enumerate(s)]
    if isinstance(a, str) and isinstance(sep, str):
        return result[0]
    return _make_result(result)


def replace(a, old, new, count=None):
    """使用新字符串替换字符串中的所有子字符串。"""
    s = _get_strings(a)
    if count is not None:
        result = [x.replace(old, new, count) for x in s]
    else:
        result = [x.replace(old, new) for x in s]
    if isinstance(a, str):
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
