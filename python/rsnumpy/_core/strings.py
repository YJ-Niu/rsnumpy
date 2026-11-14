"""numpy._core.strings 兼容门面：strings 底层 ufunc 的 Python 等价物。

NumPy 中这是 C 层 ufunc 模块；rsnumpy 的 ufunc 是 Python 层可调用
对象。test_strings 的 gh-28829 回归测试依赖
``np._core.strings._expandtabs_length.reduce(...)`` 对非字符串输入抛
``_UFuncNoLoopError``（消息含 "did not contain a loop"）。
"""

import builtins

from rsnumpy._core._exceptions import _UFuncNoLoopError


def _no_loop_error(ufunc_name):
    return _UFuncNoLoopError(
        f"'{ufunc_name}' ufunc did not contain a loop with signature "
        "matching types")


class _StringsUFunc:
    """strings ufunc 占位：按输入类型做 no-loop 检查，字符串输入走等价实现。"""

    def __init__(self, name, impl):
        self._name = name
        self._impl = impl
        self.__name__ = name

    def __call__(self, *args, **kwargs):
        a = args[0]
        if not _is_string_input(a):
            raise _no_loop_error(self._name)
        return self._impl(*args, **kwargs)

    def reduce(self, a, *args, **kwargs):
        # strings ufunc 无 reduce 语义：numpy 直接按 no-loop 报错
        if not _is_string_input(a):
            raise _no_loop_error(self._name)
        raise _no_loop_error(self._name)


def _is_string_input(x):
    if isinstance(x, (str, bytes)):
        return True
    k = getattr(getattr(x, 'dtype', None), 'kind', None)
    return k in ('S', 'U')


def _expandtabs_length_impl(a, tabsize=8):
    from rsnumpy import char as _char
    vals = _char._decode(_char._get_strings(a))
    ts = int(tabsize)
    out = []
    for v in vals:
        new_len = 0
        line_pos = 0
        for ch in v:
            if ch == '\t':
                if ts > 0:
                    new_len += ts - (line_pos % ts)
                    line_pos += ts - (line_pos % ts)
            else:
                line_pos += 1
                new_len += 1
                if ch in '\n\r':
                    line_pos = 0
        out.append(new_len)
    from rsnumpy import ndarray as _nd
    shape = _char._shape_of(a)
    if isinstance(a, (str, bytes)) or shape == ():
        res = _nd([builtins.float(out[0]) if out else 0], _dtype='int64')
        return res.reshape(())
    res = _nd([builtins.float(v) for v in out], _dtype='int64')
    return res.reshape(shape)


_expandtabs_length = _StringsUFunc(
    '_expandtabs_length', _expandtabs_length_impl)
