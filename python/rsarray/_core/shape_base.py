# SPDX-License-Identifier: MIT
"""rsarray._core.shape_base 内部辅助兼容模块。

rsarray 的 ``np.block`` 不经过 内部的 ``_block_*`` 入口；这些
辅助保留"调用即 skip"行为（与旧 conftest 桩一致），使移植测试的
依赖可解析且不改变测试结果。
"""

import pytest

__all__ = ['_block_setup', '_block_slicing', '_block_dispatcher',
           '_block_concatenate']


def _skip_c_helper(*args, **kwargs):
    raise pytest.skip(
        "rsarray 不暴露 rsarray._core.shape_base 内部 C 辅助的等价实现")


_block_setup = _skip_c_helper
_block_slicing = _skip_c_helper
_block_dispatcher = _skip_c_helper
_block_concatenate = _skip_c_helper
