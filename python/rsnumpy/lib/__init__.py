"""rsnumpy.lib - 兼容 numpy.lib 的子模块集合。"""

from . import recfunctions
from . import stride_tricks
from . import _index_tricks_impl

__all__ = ['recfunctions', 'stride_tricks', 'add_newdoc', 'NumpyVersion',
           '_index_tricks_impl']


def add_newdoc(where, obj, doc):
    """Stub: record a docstring for ``where.obj`` (no-op in rsnumpy)."""
    pass


class NumpyVersion:
    """NumPy 版本字符串比较器（numpy.lib.NumpyVersion 兼容）。

    支持 '1.21.0'、'1.21.0rc1'、'1.21.0.dev0+git...' 等形式；
    与字符串或其他 NumpyVersion 比较（<, <=, >, >=, ==, !=）。
    """

    def __init__(self, version, strict=False):
        self.vstring = str(version)
        self.strict = strict
        v = self.vstring
        # 去掉 git/构建后缀
        for sep in ('+', '.', '-'):
            if sep in v and not v.count(sep) == 2 and sep == '+':
                v = v.split('+', 1)[0]
        # 开发/预发布标记：'1.21.0rc1' → (1,21,0, 'rc1')
        import re
        m = re.match(r'^(\d+)(?:\.(\d+))?(?:\.(\d+))?(.*)$', v)
        if not m:
            raise ValueError(f"invalid version string: {version!r}")
        self._parts = (int(m.group(1)),
                       int(m.group(2)) if m.group(2) else 0,
                       int(m.group(3)) if m.group(3) else 0)
        self._tail = m.group(4) or ''

    def _key(self):
        # 预发布/开发版排在正式版之前：'rc1'/'dev0' → 0，无尾缀 → 1
        pre = 0 if self._tail else 1
        return (self._parts, pre, self._tail)

    def __str__(self):
        return self.vstring

    def __repr__(self):
        return f'<numpy.lib.NumpyVersion {self.vstring!r}>'

    def __eq__(self, other):
        return self._compare(other) == 0

    def __lt__(self, other):
        return self._compare(other) < 0

    def __le__(self, other):
        return self._compare(other) <= 0

    def __gt__(self, other):
        return self._compare(other) > 0

    def __ge__(self, other):
        return self._compare(other) >= 0

    def __ne__(self, other):
        return self._compare(other) != 0

    def __hash__(self):
        return hash(self.vstring)

    def _compare(self, other):
        if isinstance(other, str):
            other = NumpyVersion(other)
        if not isinstance(other, NumpyVersion):
            return NotImplemented
        return (self._key() > other._key()) - (self._key() < other._key())


def __getattr__(name):
    # 惰性导入私有 impl 模块（避免顶层循环导入）
    if name in ('_function_base_impl', '_nanfunctions_impl',
                '_stride_tricks_impl'):
        from importlib import import_module
        return import_module(f'.{name}', __name__)
    raise AttributeError(f"module 'rsnumpy.lib' has no attribute {name!r}")
