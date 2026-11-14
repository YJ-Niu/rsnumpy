# SPDX-License-Identifier: MIT
"""rsnumpy.lib - 兼容 numpy.lib 的子模块集合。"""

from . import recfunctions
from . import stride_tricks
from . import _index_tricks_impl

# numpy.lib._shape_base_impl 的兼容引用(suppress_warnings 等按模块过滤用)
from .. import _extra as _shape_base_impl

__all__ = ['recfunctions', 'stride_tricks', 'add_newdoc', 'NumpyVersion',
           '_index_tricks_impl', '_shape_base_impl']


def add_newdoc(where, obj, doc):
    """Stub: record a docstring for ``where.obj`` (no-op in rsnumpy)."""
    pass


class NumpyVersion:
    """NumPy 版本字符串比较器（numpy.lib.NumpyVersion 兼容）。

    支持 '1.21.0'、'1.21.0rc1'、'1.21.0.dev0+git...'、'1.21.0.dev-<hash>'
    等形式；与字符串或其他 NumpyVersion 比较（<, <=, >, >=, ==, !=）。
    """

    def __init__(self, version, strict=False):
        import re
        self.vstring = str(version)
        self.strict = strict
        # 官方语义：x.y.z 三段基线必须完整，可选预发布（aN/bN/rcN）、
        # dev 后缀（.dev0 / .dev-<hash>）与本地后缀（+<...>）；
        # '1.9'、'1,9.0'、'1.7.x' 等非法形式抛 ValueError。
        m = re.match(
            r'^(\d+)\.(\d+)\.(\d+)'
            r'(?:(a|b|rc)(\d+))?'
            r'((?:\.dev[-\w]*)?)'
            r'(?:\+.*)?$', self.vstring)
        if not m:
            raise ValueError(f"invalid version string: {version!r}")
        self._parts = (int(m.group(1)), int(m.group(2)), int(m.group(3)))
        self._pre = m.group(4)              # None / 'a' / 'b' / 'rc'
        self._pre_num = int(m.group(5)) if m.group(5) else 0
        self._is_dev = bool(m.group(6))

    def _key(self):
        # 次序（同一 x.y.z 内）：
        #   final.dev < aN.dev < aN < aN+1.dev < ... < bN.dev < bN <
        #   rcN.dev < rcN < final
        # - dev 版本比其所属基线低半步；
        # - final 基线的 dev 版本低于所有预发布（官方 1.0.dev0 < 1.0a1）；
        # - dev 后缀（hash 等）不参与比较（'1.9.0.dev-a' == '1.9.0.dev-b'，
        #   官方语义）。
        if self._pre is None:
            base = (4, 0)
        else:
            base = ({'a': 1, 'b': 2, 'rc': 3}[self._pre], self._pre_num)
        if self._is_dev:
            base = (0, 0) if self._pre is None else (base[0], base[1] - 0.5)
        return (self._parts, base)

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
