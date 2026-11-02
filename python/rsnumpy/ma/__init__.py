"""掩码数组（numpy.ma 兼容层）。

子模块布局与 NumPy 一致：

- ``rsnumpy.ma.core``：核心（MaskedArray / 构造器 / 掩码辅助）
- ``rsnumpy.ma.extras``：统计与集合扩展
- ``rsnumpy.ma.mrecords``：掩码记录数组
"""

from . import core
from . import extras
from . import mrecords
from .core import (  # noqa: F401
    MAError,
    MaskError,
    MaskType,
    MaskedArray,
    MaskedArrayFutureWarning,
    anom,
    array,
    asarray,
    common_fill_value,
    concatenate,
    count,
    default_fill_value,
    empty,
    empty_like,
    fix_invalid,
    flatten_mask,
    flatten_structured_array,
    filled,
    full,
    getdata,
    getmask,
    getmaskarray,
    isMaskedArray,
    make_mask,
    make_mask_descr,
    masked,
    masked_array,
    masked_equal,
    masked_greater,
    masked_greater_equal,
    masked_inside,
    masked_invalid,
    masked_less,
    masked_less_equal,
    masked_not_equal,
    masked_object,
    masked_outside,
    masked_print_option,
    masked_values,
    masked_where,
    mask_or,
    maximum_fill_value,
    minimum_fill_value,
    minmax,
    mvoid,
    nomask,
    ones,
    ones_like,
    zeros,
    zeros_like,
)
from .mrecords import (  # noqa: F401
    MaskedRecords,
    mrecarray,
)

# numpy.ma 顶层暴露 core 的全部公共名（test_old_ma 按此导入）
for _n in dir(core):
    if not _n.startswith('_') and _n not in globals():
        globals()[_n] = getattr(core, _n)
del _n

# numpy.ma 顶层亦暴露 extras 的常用名
for _n in ('atleast_1d', 'atleast_2d', 'atleast_3d', 'average', 'clump',
           'clump_masked', 'clump_unmasked', 'compress_nd', 'compress_rowcols',
           'corrcoef', 'cov', 'diagflat', 'ediff1d', 'flatnotmasked_contiguous',
           'in1d', 'intersect1d', 'isin', 'masked_all', 'masked_all_like',
           'mask_rowcols', 'mask_rows', 'mask_cols', 'median', 'mr_', 'ndenumerate', 'notmasked_contiguous',
           'notmasked_edges', 'polyfit', 'setdiff1d', 'setxor1d', 'stack',
           'union1d', 'unique', 'unwrap', 'vstack', 'apply_along_axis',
           'apply_over_axes'):
    if hasattr(extras, _n):
        globals()[_n] = getattr(extras, _n)
del _n

__all__ = ['core', 'extras', 'mrecords', 'MaskedArray', 'MaskedRecords',
           'masked', 'nomask', 'array', 'asarray', 'masked_array', 'mrecarray',
           'MAError', 'MaskError', 'MaskType', 'MaskedArrayFutureWarning']
