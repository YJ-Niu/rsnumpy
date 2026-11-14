# SPDX-License-Identifier: MIT
"""rsnumpy.random - 随机数生成模块"""

import rsnumpy.num_core as _core
import random as _py_random
import rsnumpy as np

from ._distributions import Samplers, _POISSON_LAM_MAX, _broadcast_shapes, _zeta


# _core.random.Generator may be Python's built-in typing.Generator, not a
# real RNG.  Provide a working fallback.
if not hasattr(_core.random.Generator, 'integers'):
    class _PyGenerator:
        """Python-based random number generator fallback."""

        def __init__(self, seed=None):
            self._rng = _py_random.Random(seed)

        def integers(self, low, high=None, size=None, dtype=None, endpoint=False):
            if high is None:
                low, high = 0, low
            if size is None:
                return self._rng.randint(low, high)
            return np.asarray([self._rng.randint(low, high) for _ in range(size)])

        def uniform(self, low=0.0, high=1.0, size=None):
            import rsnumpy as np
            if size is None:
                return self._rng.uniform(low, high)
            return np.asarray([self._rng.uniform(low, high) for _ in range(size)])

        def normal(self, loc=0.0, scale=1.0, size=None):
            import rsnumpy as np
            if size is None:
                return self._rng.gauss(loc, scale)
            return np.asarray([self._rng.gauss(loc, scale) for _ in range(size)])

        def random(self, size=None):
            import rsnumpy as np
            if size is None:
                return self._rng.random()
            return np.asarray([self._rng.random() for _ in range(size)])

        def standard_normal(self, size=None):
            return self.normal(0.0, 1.0, size)

        def vonmises(self, kappa, size=None):
            import rsnumpy as np
            import math
            if size is None:
                size = ()
            n_samples = int(np.prod(size)) if isinstance(size, tuple) else int(size)
            result = []
            for _ in range(n_samples):
                while True:
                    u1 = self._rng.random()
                    u2 = self._rng.random()
                    z = math.cos(math.pi * u1)
                    if u2 < math.exp(kappa * (z - 1)):
                        break
                angle = math.acos(max(-1.0, min(1.0, z)))
                if self._rng.random() < 0.5:
                    angle = -angle
                result.append(angle)
            return np.asarray(result).reshape(size)

        def multinomial(self, n, pvals, size=None):
            import rsnumpy as np
            # Simple categorical sampling
            pvals = np.asarray(pvals)
            pvals = pvals / pvals.sum()
            cdf = np.cumsum(pvals)
            if size is None:
                size = ()
            result = []
            for _ in range(np.prod(size) if isinstance(size, tuple) else size):
                u = self._rng.random()
                idx = int(cdf.searchsorted(u))
                result.append(idx)
            return np.asarray(result).reshape(size)

    _core.random.Generator = _PyGenerator


class _MT19937:
    """Minimal stand-in for NumPy's MT19937 bit generator.

    Only the attributes that downstream code (e.g. rsscipy's ``_util``)
    inspects are provided: ``_seed_seq`` and ``state``.
    """

    def __init__(self, seed=None):
        self._seed_seq = None
        self._seed = seed

    @property
    def state(self):
        return {'bit_generator': 'MT19937', 'state': {'key': [0]}, 'pos': 0}

    def spawn(self, n_children):
        return [type(self)(None) for _ in range(n_children)]


def _is_arraylike(v):
    return isinstance(v, (list, tuple, np.ndarray)) or hasattr(v, 'shape')


def _norm_size(size):
    if size is None:
        return ()
    if isinstance(size, float):
        raise TypeError("size must be an integer or tuple of integers")
    if isinstance(size, (list, tuple)):
        return tuple(int(x) for x in size)
    if _is_arraylike(size):
        return tuple(int(x) for x in np.asarray(size).ravel().tolist())
    return (int(size),)


def _sample_dist(sampler, name, params, size, int_dist=False):
    """按 numpy 语义：数组参数广播 → 前置 size → 逐点采样 → 封装
    （标量 or ndarray）。params 为采样器位置参数序列（标量或数组）。"""
    param_vals = []
    arrs = []
    for p in params:
        if _is_arraylike(p):
            pa = np.asarray(p)
            if pa.ndim == 0:
                # 0-d：numpy 以 __float__/__int__ 提取标量，子类覆盖须被尊重
                # （test_scalar_exception_propagation）；整数分布的整数 dtype
                # 参数走 __int__（尊重 __index__ 覆盖），浮点参数保持 float
                _dt = str(pa.dtype)
                if int_dist and _dt not in ('float16', 'float32', 'float64',
                                            'complex64', 'complex128'):
                    param_vals.append(('s', int(p)))
                else:
                    param_vals.append(('s', float(p)))
                continue
            a = np.asarray(pa, dtype='float64')
            param_vals.append(('a', a))
            arrs.append(a)
        else:
            param_vals.append(('s', float(p)))
    ssize = _norm_size(size)
    if arrs:
        bcast = np.broadcast_arrays(*arrs)
        bshape = tuple(bcast[0].shape)
        # numpy 语义：size 与参数广播形一起（右对齐）广播，结果须等于 size
        # （test_n_zero_stream：binomial([[0],[10]], .25, size=(2,10)) → (2,10)）；
        # 不兼容时 _broadcast_shapes 抛 ValueError（与 numpy 一致）。
        bshape = _broadcast_shapes(ssize, bshape)
        # 每个参数数组扩到最终形状（如 (2,1) → (2,10)），否则扁平索引错位。
        bcast = [np.broadcast_to(b, bshape) for b in bcast]
        idx = 0
        flat_params = []
        for kind, p in param_vals:
            if kind == 's':
                flat_params.append(('s', p))
            else:
                flat_params.append(('a', bcast[idx].ravel().tolist()))
                idx += 1
    else:
        bshape = ()
        flat_params = param_vals
    out_shape = ssize + bshape if not arrs else bshape
    total = 1
    for d in out_shape:
        total *= d
    bn = 1
    for d in bshape:
        bn *= d
    fn = getattr(sampler, name)
    res = []
    for i in range(total):
        bidx = i % bn
        vals = [d if k == 's' else d[bidx] for k, d in flat_params]
        res.append(fn(*vals))
    if out_shape == ():
        return int(res[0]) if int_dist else res[0]
    return np.asarray(res, dtype='int64' if int_dist else 'float64').reshape(out_shape)


def _multinomial_impl(sampler, n, pvals, size):
    """numpy legacy multinomial：d-1 次 legacy_random_binomial 循环 + 余数
    （legacy-distributions.c L621-644），与 binomial() 共享同一 binomial 缓存。"""
    if isinstance(pvals, (str, bytes)):
        raise TypeError("pvals must be a 1-d sequence")
    parr = np.asarray(pvals, dtype='float64')
    if parr.ndim == 0:
        raise TypeError("pvals must be a 1-d sequence")
    if parr.ndim != 1:
        raise ValueError("pvals must be a 1-d sequence")
    d = int(parr.size)
    pix = parr.ravel().tolist()
    # CONS_BOUNDED_0_1：只拒绝 x<0 或 x>1（NaN 放行，由 binomial 捕获）
    for pv in pix:
        if pv < 0.0 or pv > 1.0:
            raise ValueError("pvals < 0, pvals > 1 or pvals contains NaNs")
    # sum(pvals[:-1]) > 1.0（末元素视为剩余概率，不参与求和）
    if d and sum(pix[:-1]) > (1.0 + 1e-12):
        raise ValueError("sum(pvals[:-1]) > 1.0")
    if size is None:
        shape = (d,)
    else:
        shape = _norm_size(size) + (d,)
    ni = int(n)
    if ni < 0:
        raise ValueError("n must be non-negative")
    if d == 0:
        return np.zeros(shape, dtype='int64')
    sz = 1
    for _dd in shape:
        sz *= _dd
    niter = sz // d if d else 0
    out = []
    for _i in range(niter):
        dn = ni
        row = [0] * d
        remaining_p = 1.0
        for j in range(d - 1):
            row[j] = int(sampler.binomial(dn, pix[j] / remaining_p))
            dn -= row[j]
            if dn <= 0:
                break
            remaining_p -= pix[j]
        if dn > 0:
            row[d - 1] = dn
        out.extend(row)
    return np.asarray(out, dtype='int64').reshape(shape)


# legacy 分布名 → Samplers 方法名（uniform/normal/standard_normal 走 Rust 内核，
# 不在此列，保持既有流不变）
_RS_DIST_DISPATCH = {
    'beta': 'beta',
    'binomial': 'binomial',
    'chisquare': 'chisquare',
    'exponential': 'exponential',
    'f': 'f',
    'gamma': 'gamma',
    'geometric': 'geometric',
    'gumbel': 'gumbel',
    'hypergeometric': 'hypergeometric',
    'laplace': 'laplace',
    'logistic': 'logistic',
    'lognormal': 'lognormal',
    'logseries': 'logseries',
    'negative_binomial': 'negative_binomial',
    'noncentral_chisquare': 'noncentral_chisquare',
    'noncentral_f': 'noncentral_f',
    'pareto': 'pareto',
    'poisson': 'poisson',
    'power': 'power',
    'rayleigh': 'rayleigh',
    'standard_cauchy': 'standard_cauchy',
    'standard_exponential': 'standard_exponential',
    'standard_gamma': 'standard_gamma',
    'standard_t': 'standard_t',
    't': 'standard_t',
    'triangular': 'triangular',
    'triangle': 'triangular',
    'weibull': 'weibull',
    'wald': 'wald',
    'zipf': 'zipf',
}
_RS_INT_DISTS = {'binomial', 'geometric', 'hypergeometric', 'logseries',
                 'negative_binomial', 'poisson', 'zipf'}

# 每个分布的规范位置参数名（用于把关键字参数映射到正确槽位）
_RS_DIST_PARAMS = {
    'beta': ('a', 'b'),
    'binomial': ('n', 'p'),
    'chisquare': ('df',),
    'exponential': ('scale',),
    'f': ('dfnum', 'dfden'),
    'gamma': ('shape', 'scale'),
    'geometric': ('p',),
    'gumbel': ('loc', 'scale'),
    'hypergeometric': ('ngood', 'nbad', 'nsample'),
    'laplace': ('loc', 'scale'),
    'logistic': ('loc', 'scale'),
    'lognormal': ('mean', 'sigma'),
    'logseries': ('p',),
    'negative_binomial': ('n', 'p'),
    'noncentral_chisquare': ('df', 'nonc'),
    'noncentral_f': ('dfnum', 'dfden', 'nonc'),
    'pareto': ('a',),
    'poisson': ('lam',),
    'power': ('a',),
    'rayleigh': ('scale',),
    'standard_cauchy': (),
    'standard_exponential': (),
    'standard_gamma': ('shape',),
    'standard_t': ('df',),
    't': ('df',),
    'triangular': ('left', 'mode', 'right'),
    'triangle': ('left', 'mode', 'right'),
    'weibull': ('a',),
    'wald': ('mean', 'scale'),
    'zipf': ('a',),
}

# numpy legacy 分布缺省参数（与 mtrand 签名一致，参数缺失时填充）
_RS_DIST_DEFAULTS = {
    ('exponential', 'scale'): 1.0,
    ('gamma', 'shape'): 1.0, ('gamma', 'scale'): 1.0,
    ('gumbel', 'loc'): 0.0, ('gumbel', 'scale'): 1.0,
    ('laplace', 'loc'): 0.0, ('laplace', 'scale'): 1.0,
    ('logistic', 'loc'): 0.0, ('logistic', 'scale'): 1.0,
    ('lognormal', 'mean'): 0.0, ('lognormal', 'sigma'): 1.0,
    ('poisson', 'lam'): 1.0,
    ('rayleigh', 'scale'): 1.0,
    ('triangular', 'left'): 0.0, ('triangular', 'mode'): 0.5,
    ('triangular', 'right'): 1.0,
    ('triangle', 'left'): 0.0, ('triangle', 'mode'): 0.5,
    ('triangle', 'right'): 1.0,
}

_MISSING = object()


def _fold_seed(seed):
    """任意 Python 种子 → Rust 可接受的 u64（新 API Generator 用，宽松）。"""
    if seed is None:
        return None
    if isinstance(seed, _SeedSequence):
        seed = seed.entropy
    if not isinstance(seed, int) and hasattr(seed, 'item'):
        seed = int(seed)
    if not isinstance(seed, int):
        if isinstance(seed, (list, tuple)):
            acc = 0
            for x in seed:
                acc = (acc * 1000003 + int(x)) & ((1 << 64) - 1)
            seed = acc
        else:
            try:
                seed = int(seed)
            except (TypeError, ValueError):
                raise TypeError(f"cannot convert {type(seed).__name__} to a seed")
    return seed & ((1 << 64) - 1)


def _legacy_validate_seed(seed):
    """numpy legacy RandomState 种子校验（严格）。

    合法：None、单个 [0, 2**32) 整数、1-d 非空整数数组（元素同范围）。
    越界/非整数 → ValueError/TypeError，与 numpy 对齐。
    返回：None / int（标量种子）/ list[int]（array 种子——numpy 对其走
    mt19937_init_by_array，不折叠为标量；0-d 整数数组按标量处理）。
    """
    if seed is None:
        return None
    if isinstance(seed, _SeedSequence):
        seed = seed.entropy
    if isinstance(seed, int) and not isinstance(seed, bool):
        if seed < 0 or seed >= 2 ** 32:
            raise ValueError("Seed must be in [0, 2**32)")
        return seed
    if isinstance(seed, float) or isinstance(seed, bool):
        raise TypeError("Seed must be a non-negative integer")
    if not isinstance(seed, (list, tuple, np.ndarray)) and hasattr(seed, 'item'):
        try:
            iv = int(seed)
        except (TypeError, ValueError):
            raise TypeError("Seed must be a non-negative integer")
        if iv < 0 or iv >= 2 ** 32:
            raise ValueError("Seed must be in [0, 2**32)")
        return iv
    try:
        arr = np.asarray(seed)
    except Exception:
        raise TypeError("Seed must be a non-negative integer or array of integers")
    if arr.ndim == 0:
        # numpy 语义：squeeze 后 0-d 整数数组按标量种子处理（operator.index 路径）
        v = arr.ravel().tolist()[0]
        if isinstance(v, float):
            raise TypeError("Seed must be a non-negative integer")
        iv = int(v)
        if iv != v:
            raise TypeError("Seed must be a non-negative integer")
        if iv < 0 or iv >= 2 ** 32:
            raise ValueError("Seed must be in [0, 2**32)")
        return iv
    if arr.ndim != 1:
        raise ValueError("Seed must be a 1-d array of integers or None")
    if arr.size == 0:
        raise ValueError("Seed must be a non-empty array or None")
    out = []
    for v in arr.ravel().tolist():
        if isinstance(v, float):
            raise TypeError("Seed elements must be integers")
        try:
            iv = int(v)
        except (TypeError, ValueError):
            raise TypeError("Seed elements must be integers")
        if iv != v:
            raise TypeError("Seed elements must be integers")
        if iv < 0 or iv >= 2 ** 32:
            raise ValueError("Seed elements must be in [0, 2**32)")
        out.append(iv)
    return out


def _parse_int_dtype(name):
    """'int32'/'i4'/'uint64'/'u8'/'bool'/'int' → ('int'|'uint', bits) / ('bool', None)。"""
    name = name.strip()
    if name in ('bool', '?'):
        return 'bool', None
    if name in ('int', 'l', 'long'):
        return 'int', 64
    if name.startswith('uint') and name[4:].isdigit():
        return 'uint', int(name[4:])
    if name.startswith('int') and name[3:].isdigit():
        return 'int', int(name[3:])
    # 单字母+数字简写是字节数（i4 = int32 / u8 = uint64），须 ×8 换成位宽
    if name.startswith('u') and name[1:].isdigit():
        return 'uint', int(name[1:]) * 8
    if name.startswith('i') and name[1:].isdigit():
        return 'int', int(name[1:]) * 8
    return None


def _python_shuffle(next_j, x, axis=0, warn_1d_object=False):
    """Fisher-Yates 洗牌：list / ndarray（含 object/结构化/memoryview）。

    next_j(i) 用 numpy legacy rk_interval（掩码拒绝）抽 j ∈ [0, i]，与
    numpy random_interval 流逐位一致（int(next_u()*(i+1)) 缩放不匹配）。
    object 数组逐元素搬运 Python 对象引用（不复制，保持对象身份）；
    数值数组按行搬运（行 view 需 copy 后再写回）。
    """
    if isinstance(x, list):
        n = len(x)
        for i in range(n - 1, 0, -1):
            j = next_j(i)
            x[i], x[j] = x[j], x[i]
        return
    # MaskedArray（纯 Python 对象，非 ndarray 子类）：data/mask 行同步交换
    # （gh-3263）。MA.__setitem__ 走数值 cast 无法写入 MA 行；mask getter 每次
    # 重建临时数组（_mask 是扁平 list），须交换后经 mask setter 写回。
    import rsnumpy.ma.core as _macore
    if isinstance(x, _macore.MaskedArray):
        d, m = x.data, x.mask
        n = d.shape[0]
        for i in range(n - 1, 0, -1):
            j = next_j(i)
            if j == i:
                continue
            di, dj = d[i].copy(), d[j].copy()
            mi, mj = m[i].copy(), m[j].copy()
            d[i], d[j] = dj, di
            m[i], m[j] = mj, mi
        x.mask = m
        return
    if hasattr(x, 'shape'):
        if hasattr(x, 'flags') and not x.flags.writeable:
            raise ValueError("read-only array -- cannot shuffle in-place")
        is_obj = False
        try:
            is_obj = str(x.dtype).strip() == 'object'
        except Exception:
            pass
        if warn_1d_object and is_obj and len(x.shape) == 1:
            import warnings
            warnings.warn(
                "Shuffling a one dimensional array of objects is a common "
                "use case, but the result is a new array of the shuffled "
                "objects; use a list instead.", UserWarning, stacklevel=3)
        n = x.shape[0]
        for i in range(n - 1, 0, -1):
            j = next_j(i)
            if j == i:
                continue
            el = x[i]
            tmp = el if (is_obj or not hasattr(el, 'copy')) else el.copy()
            x[i] = x[j]
            x[j] = tmp
        return
    import warnings
    warnings.warn(
        f"you are shuffling a '{type(x).__name__}' object, which is "
        "treated as a sequence by rsnumpy.random.shuffle", UserWarning,
        stacklevel=3)
    n = len(x)
    for i in range(n - 1, 0, -1):
        j = next_j(i)
        x[i], x[j] = x[j], x[i]


class _MT19937Legacy:
    """numpy legacy ``RandomState`` 的 MT19937 位级一致流。

    种子算法 = 标准 MT19937：标量种子走 ``init_genrand``；array 种子走
    ``mt19937_init_by_array``（numpy ``_legacy_seeding`` 语义，已与
    numpy 2.5.4 逐位对照：RandomState(range(10)) / [0] / [4294967295]
    的 randint(1000) = 468/973/265）。
    uniform = ``res53``（两个 32 位输出合成 53 位 double）；
    randint = ``rk_interval``（掩码拒绝采样，无偏）。
    """
    __slots__ = ('_mt', '_mti', '_gauss_cache')

    def _init_state(self, seed):
        if seed is None:
            import time
            seed = (time.time_ns() ^ id(self)) & 0xffffffff
        if isinstance(seed, (list, tuple)):
            self._init_by_array(seed)
            return
        seed = int(seed) & 0xffffffff
        mt = [0] * 624
        mt[0] = seed
        for i in range(1, 624):
            mt[i] = (1812433253 * (mt[i - 1] ^ (mt[i - 1] >> 30)) + i) & 0xffffffff
        self._mt = mt
        self._mti = 624
        self._gauss_cache = None

    def _init_by_array(self, key):
        """numpy ``mt19937_init_by_array``（legacy array 种子）。

        常量与 numpy ``numpy/random/src/mt19937/mt19937.c`` 一致：
        init_genrand(19650218) + 1664525 / 1566083941 两轮非线性扰动。
        """
        key = [int(k) & 0xffffffff for k in key]
        key_len = len(key)
        mt = [0] * 624
        mt[0] = 19650218
        for i in range(1, 624):
            mt[i] = (1812433253 * (mt[i - 1] ^ (mt[i - 1] >> 30)) + i) & 0xffffffff
        i, j = 1, 0
        k = 624 if 624 > key_len else key_len
        for _ in range(k):
            mt[i] = (mt[i] ^ ((mt[i - 1] ^ (mt[i - 1] >> 30)) * 1664525)) + key[j] + j
            mt[i] &= 0xffffffff
            i += 1
            j += 1
            if i >= 624:
                mt[0] = mt[623]
                i = 1
            if j >= key_len:
                j = 0
        for _ in range(623, 0, -1):
            mt[i] = (mt[i] ^ ((mt[i - 1] ^ (mt[i - 1] >> 30)) * 1566083941)) - i
            mt[i] &= 0xffffffff
            i += 1
            if i >= 624:
                mt[0] = mt[623]
                i = 1
        mt[0] = 0x80000000
        self._mt = mt
        self._mti = 624
        self._gauss_cache = None

    def __init__(self, seed):
        self._init_state(seed)

    def reseed(self, seed):
        self._init_state(seed)

    def _twist(self):
        mt = self._mt
        for i in range(624):
            y = (mt[i] & 0x80000000) + (mt[(i + 1) % 624] & 0x7fffffff)
            v = mt[(i + 397) % 624] ^ (y >> 1)
            if y & 1:
                v ^= 0x9908b0df
            mt[i] = v & 0xffffffff
        self._mti = 0

    def u32(self):
        if self._mti >= 624:
            self._twist()
        x = self._mt[self._mti]
        self._mti += 1
        x ^= (x >> 11)
        x ^= (x << 7) & 0x9d2c5680
        x ^= (x << 15) & 0xefc60000
        x ^= (x >> 18)
        return x & 0xffffffff

    def random_sample(self):
        a = self.u32() >> 5
        b = self.u32() >> 6
        return (a * 67108864.0 + b) * (1.0 / 9007199254740992.0)

    def interval(self, maxv):
        """numpy ``rk_interval(maxv)``：返回 [0, maxv] 的无偏整数。"""
        if maxv == 0:
            return 0
        mask = (1 << maxv.bit_length()) - 1
        while True:
            v = self.u32() & mask
            if v <= maxv:
                return v

    def _gen_mask(self, maxv):
        """numpy ``gen_mask``：最小位掩码 ≥ maxv（顶位以下全置 1）。"""
        mask = maxv
        mask |= mask >> 1
        mask |= mask >> 2
        mask |= mask >> 4
        mask |= mask >> 8
        mask |= mask >> 16
        mask |= mask >> 32
        return mask

    def intervals_u8(self, rngs):
        """numpy legacy _rand_uint8 掩码路径（distributions.c
        random_buffered_bounded_uint8）：每 32-bit 字出 4 个 8-bit 值（LSB
        起）、掩码拒绝；单次 randint 调用内 buf 跨元素共享，跨调用全新
        （numpy 中为函数局部变量）。rngs: 逐元素闭包上界（hi-1-lo）。"""
        buf = 0
        bcnt = 0
        out = []
        for rng in rngs:
            if rng == 0:
                out.append(0)
                continue
            mask = self._gen_mask(rng)
            while True:
                if bcnt == 0:
                    buf = self.u32()
                    bcnt = 3
                else:
                    buf >>= 8
                    bcnt -= 1
                val = buf & 0xFF & mask
                if val <= rng:
                    break
            out.append(val)
        return out

    def intervals_u16(self, rngs):
        """numpy legacy _rand_uint16 掩码路径：每 32-bit 字出 2 个 16-bit
        值（低半字先）、掩码拒绝；buf 语义同 intervals_u8。"""
        buf = 0
        bcnt = 0
        out = []
        for rng in rngs:
            if rng == 0:
                out.append(0)
                continue
            mask = self._gen_mask(rng)
            while True:
                if bcnt == 0:
                    buf = self.u32()
                    bcnt = 1
                else:
                    buf >>= 16
                    bcnt -= 1
                val = buf & 0xFFFF & mask
                if val <= rng:
                    break
            out.append(val)
        return out

    def intervals_bool(self, rngs):
        """numpy legacy _rand_bool（buffered_bounded_bool）：每 32-bit 字出
        32 个 bit（LSB 起）、无拒绝；rng==0 元素不消耗随机数。"""
        buf = 0
        bcnt = 0
        out = []
        for rng in rngs:
            if rng == 0:
                out.append(0)
                continue
            if bcnt == 0:
                buf = self.u32()
                bcnt = 31
            else:
                buf >>= 1
                bcnt -= 1
            out.append(1 if (buf & 1) else 0)
        return out

    def standard_normal(self):
        """numpy legacy ``rk_gauss``（极坐标 Box-Muller，带缓存）。

        numpy 先返回 v2*r（第二个 res53 对应值），缓存 v1*r（第一个），
        与 numpy 逐位一致（小 size 与标量已验证；大 size 有 C/Python 位级差）。
        """
        if self._gauss_cache is not None:
            x = self._gauss_cache
            self._gauss_cache = None
            return x
        import math
        while True:
            v1 = 2.0 * self.random_sample() - 1.0
            v2 = 2.0 * self.random_sample() - 1.0
            s = v1 * v1 + v2 * v2
            if 0.0 < s < 1.0:
                break
        r = math.sqrt(-2.0 * math.log(s) / s)
        self._gauss_cache = v1 * r
        return v2 * r


class _RandomStateWrapper:
    """Python wrapper around the core RandomState.

    Adds the legacy attributes SciPy expects (``_bit_generator``) and
    distributions the Rust core does not implement (``multivariate_normal``
    以及 _distributions 补齐的完整 legacy 分布集)。
    """

    def __init__(self, seed=None):
        seed = _legacy_validate_seed(seed)
        self._seed = seed
        self._mt = _MT19937Legacy(seed)
        self._sampler = Samplers(self._next_u, self._next_normal)
        # Rust 核流（ChaCha8）只收 u64 标量种子；array 种子折叠处理。
        # legacy 分布全在 _mt 上，_inner 仅供未处理属性回退。
        self._inner = _core.random.RandomState(_fold_seed(seed))

    def __deepcopy__(self, memo):
        # The Rust core object is not picklable.  Re-seed from the original
        # seed so the copy is independent of (and initially identical to)
        # this wrapper, matching numpy's deepcopy semantics for unused
        # generators.
        return _RandomStateWrapper(self._seed)

    def __copy__(self, memo=None):
        return _RandomStateWrapper(self._seed)

    def _rs_f64(self, vals):
        """确保输出是带 _dtype 的正规 rsnumpy ndarray（Rust 原生对象转换）。"""
        import rsnumpy as _np
        if hasattr(vals, 'tolist') and not hasattr(vals, '_dtype'):
            vals = vals.tolist()
        return _np.asarray(vals, dtype='float64')

    def rand(self, *args):
        """生成均匀分布的随机数（MT19937 res53 流）。"""
        # 兼容 rand((2,3)) / rand([2,3]) 单容器参数（视为 shape）
        if len(args) == 1 and isinstance(args[0], (tuple, list)):
            args = tuple(int(d) for d in args[0])
        shape = tuple(int(d) for d in args) if args else (1,)
        total = 1
        for d in shape:
            total *= d
        vals = [self._mt.random_sample() for _ in range(total)]
        return self._rs_f64(vals).reshape(shape)

    def random(self, *args, **kwargs):
        """numpy legacy random_sample：[0,1) 均匀（MT19937 res53）。"""
        if not args and 'size' in kwargs:
            size = kwargs['size']
        elif len(args) == 1:
            size = args[0]
        elif args:
            size = args
        else:
            size = None
        if size is None:
            return self._mt.random_sample()
        if isinstance(size, tuple):
            return self.rand(*size)
        return self.rand(int(size))

    def random_sample(self, size=None):
        """numpy legacy random_sample：[0,1) 均匀（MT19937 res53）。"""
        return self.random(size)

    def randn(self, *args):
        """生成标准正态分布的随机数。"""
        return self._rs_f64(self._inner.randn(1) if not args
                            else self._inner.randn(*args))

    def _next_u(self):
        return self._mt.random_sample()

    def _next_normal(self):
        # MT19937 Box-Muller（numpy legacy rk_gauss，带缓存）
        return self._mt.standard_normal()

    def normal(self, *args, **kwargs):
        """正态分布（MT19937 Box-Muller，numpy legacy rk_gauss 流）。"""
        import rsnumpy as _np
        dtype = kwargs.pop('dtype', None)
        size = kwargs.pop('size', None)
        loc = kwargs.pop('loc', None)
        scale = kwargs.pop('scale', None)
        if len(args) >= 1 and loc is None:
            loc = args[0]
        if len(args) >= 2 and scale is None:
            scale = args[1]
        if len(args) >= 3 and size is None:
            size = args[2]
        loc = 0.0 if loc is None else loc
        scale = 1.0 if scale is None else scale

        def _is_seq(x):
            try:
                len(x)
            except (TypeError, AttributeError):
                return False
            return not isinstance(x, (str, bytes))

        if _is_seq(loc) or _is_seq(scale):
            # 广播路径：loc/scale 为序列，按 numpy 1 维广播规则
            def _flat(x):
                if hasattr(x, 'tolist'):
                    x = x.tolist()
                if isinstance(x, (list, tuple)):
                    return list(x)
                return [float(x)]
            loc_f = _flat(loc)
            scale_f = _flat(scale)
            if len(loc_f) == 1 and len(scale_f) != 1:
                loc_f = loc_f * len(scale_f)
            elif len(scale_f) == 1 and len(loc_f) != 1:
                scale_f = scale_f * len(loc_f)
            if any(s < 0.0 for s in scale_f):
                raise ValueError("scale < 0")
            z = [self._next_normal() for _ in range(len(loc_f))]
            vals = [loc_f[i] + scale_f[i] * z[i] for i in range(len(loc_f))]
            result = _np.asarray(vals, dtype='float64')
            if dtype is not None and hasattr(result, 'astype'):
                result = result.astype(dtype)
            return result

        # 标量路径
        loc = float(loc)
        scale = float(scale)
        if scale < 0.0:
            raise ValueError("scale < 0")
        if size is None:
            val = loc if scale == 0.0 else loc + scale * self._next_normal()
            return val
        if isinstance(size, (tuple, list)):
            shape = tuple(int(d) for d in size)
        else:
            shape = (int(size),)
        n = 1
        for d in shape:
            n *= d
        if scale == 0.0:
            vals = [loc] * n
        else:
            vals = [loc + scale * self._next_normal() for _ in range(n)]
        result = _np.asarray(vals, dtype='float64')
        if len(shape) > 1:
            try:
                result = result.reshape(shape)
            except Exception:
                pass
        if dtype is not None and hasattr(result, 'astype'):
            result = result.astype(dtype)
        return result

    def standard_normal(self, size=None, **kwargs):
        """标准正态分布（MT19937 Box-Muller）。"""
        dtype = kwargs.pop('dtype', None)
        return self.normal(0.0, 1.0, size=size, dtype=dtype)

    def randn(self, *args):
        """标准正态分布 randn(d0, d1, ..., dn)（MT19937 Box-Muller）。"""
        return self.normal(0.0, 1.0, size=(args if args else None))

    def uniform(self, low=0.0, high=1.0, size=None, dtype=None):
        """均匀分布（MT19937 res53，numpy legacy；low/high 支持广播）。"""
        result = self._sample_dist('uniform', (low, high), size)
        if dtype is not None and hasattr(result, 'astype'):
            result = result.astype(dtype)
        return result

    def __getattr__(self, name):
        if name == 'multivariate_normal':
            def mvn(mean, cov, size=None, check_valid='warn', tol=1e-8):
                return _mvn_sample(self._next_normal, mean, cov, size,
                                   check_valid=check_valid, tol=tol)
            return mvn
        if name == 'choice':
            def choice(a, size=None, replace=True, p=None):
                # numpy legacy mtrand.choice：uniform replace 用 randint
                # （掩码拒绝）；uniform noreplace 用 permutation；weighted
                # replace 用 cdf.searchsorted(right)；weighted noreplace 是
                # “批量抽取 + 置零 + 首次出现去重”循环（mtrand.pyx L1030-1049）。
                a_arr = np.asarray(a)
                if a_arr.ndim == 0:
                    # rsnumpy 0-d item() 恒为 float（f64 存储），用 dtype 判定整数性
                    dt = str(a_arr.dtype)
                    if dt not in ('bool', 'int64', 'int32', 'int16', 'int8',
                                  'uint64', 'uint32', 'uint16', 'uint8'):
                        raise ValueError(
                            "a must be 1-dimensional or an integer")
                    pop = int(a_arr.item())
                    arr0 = None
                elif a_arr.ndim != 1:
                    raise ValueError("a must be 1-dimensional")
                else:
                    arr0 = a_arr
                    pop = a_arr.shape[0]

                def _sz_prod(sz):
                    if sz is None:
                        return 1
                    if isinstance(sz, (tuple, list)):
                        r = 1
                        for d in sz:
                            r *= int(d)
                        return r
                    return int(sz)

                if pop <= 0 and _sz_prod(size) != 0:
                    if arr0 is None:
                        raise ValueError(
                            "a must be greater than 0 unless no samples "
                            "are taken")
                    raise ValueError(
                        "'a' cannot be empty unless no samples are taken")

                if p is not None:
                    parr = np.asarray(p, dtype='float64')
                    if parr.ndim != 1:
                        raise ValueError("'p' must be 1-dimensional")
                    if parr.shape[0] != pop:
                        raise ValueError("'a' and 'p' must have same size")
                    pl = parr.ravel().tolist()
                    if str(parr.dtype) != 'float64' or any(
                            not isinstance(v, (int, float)) for v in pl):
                        # numpy 把 None 等无法转 float 的项按 NaN 处理
                        raise ValueError("probabilities contain NaN")
                    for v in pl:
                        if v != v:
                            raise ValueError("probabilities contain NaN")
                    for v in pl:
                        if v < 0:
                            raise ValueError(
                                "probabilities are not non-negative")
                    if abs(sum(pl) - 1.0) > 1.4901161193847656e-08:
                        raise ValueError("probabilities do not sum to 1")

                is_scalar = size is None
                if is_scalar:
                    shape = ()
                    nsize = 1
                else:
                    if isinstance(size, (tuple, list)):
                        shape = tuple(int(d) for d in size)
                    else:
                        shape = (int(size),)
                    nsize = 1
                    for d in shape:
                        nsize *= d

                def _ss_right(cdf, u):
                    # cdf.searchsorted(u, side='right')：首个 cdf[i] > u 的 i
                    for i in range(len(cdf)):
                        if cdf[i] > u:
                            return i
                    return len(cdf) - 1

                def _norm_cdf(probs):
                    total = 0.0
                    for v in probs:
                        total += v
                    cum = np.cumsum(np.asarray(probs, dtype='float64')) \
                        .ravel().tolist()
                    return [c / total for c in cum]

                if replace:
                    if p is not None:
                        cdf = _norm_cdf(pl)
                        out = [_ss_right(cdf, self._next_u())
                               for _ in range(nsize)]
                    else:
                        # randint(0, pop, size=shape)：rk_interval 掩码拒绝
                        out = [self._mt.interval(pop - 1)
                               for _ in range(nsize)]
                else:
                    if nsize > pop:
                        raise ValueError(
                            "Cannot take a larger sample than population "
                            "when 'replace=False'")
                    if nsize < 0:
                        raise ValueError(
                            "Negative dimensions are not allowed")
                    if p is not None:
                        if sum(1 for v in pl if v > 0) < nsize:
                            raise ValueError(
                                "Fewer non-zero entries in p than size")
                        pcur = list(pl)
                        found = []
                        while len(found) < nsize:
                            x = [self._next_u()
                                 for _ in range(nsize - len(found))]
                            if found:
                                for f in found:
                                    pcur[f] = 0.0
                            cdf = _norm_cdf(pcur)
                            for v in (_ss_right(cdf, u) for u in x):
                                if v not in found:
                                    found.append(v)
                        out = found
                    else:
                        # permutation(pop)[:nsize]：Fisher-Yates + interval
                        idx = list(range(pop))
                        for i in range(pop - 1, 0, -1):
                            j = self._mt.interval(i)
                            idx[i], idx[j] = idx[j], idx[i]
                        out = idx[:nsize]

                if is_scalar:
                    if arr0 is None:
                        return out[0]
                    if str(arr0.dtype) == 'object':
                        return arr0[out[0]]
                    return arr0.ravel().tolist()[out[0]]
                if nsize == 0:
                    if arr0 is None:
                        return np.zeros(shape, dtype='int64')
                    return np.empty(shape, dtype=arr0.dtype)
                if arr0 is None:
                    res = np.asarray(out)
                elif str(arr0.dtype) == 'object':
                    # 直接 getitem 保持对象引用身份（ravel/tolist 会转换值）
                    res = np.array([arr0[i] for i in out], dtype='object')
                else:
                    res = np.take(arr0, np.asarray(out))
                return res.reshape(shape)
            return choice
        if name == 'permutation':
            def permutation(x):
                # numpy legacy：int → 洗乱 arange(n)；array → 洗乱副本
                # （仅第一轴）。Fisher-Yates + interval，与 numpy 逐位一致。
                if isinstance(x, int):
                    n = x
                    idx = list(range(n))
                    for i in range(n - 1, 0, -1):
                        j = self._mt.interval(i)
                        idx[i], idx[j] = idx[j], idx[i]
                    if n == 0:
                        return np.zeros(0, dtype='int64')
                    return np.asarray(idx)
                arr = np.asarray(x)
                if arr.ndim < 1:
                    raise IndexError(
                        "x must be an integer or at least 1-dimensional")
                n = arr.shape[0]
                idx = list(range(n))
                for i in range(n - 1, 0, -1):
                    j = self._mt.interval(i)
                    idx[i], idx[j] = idx[j], idx[i]
                if n == 0:
                    return np.empty(arr.shape, dtype=arr.dtype)
                return np.take(arr, np.asarray(idx))
            return permutation
        if name == 'get_state':
            def get_state():
                # numpy legacy 布局: ('MT19937', keys[624], pos, has_gauss, cached)
                mt = self._mt
                keys = list(mt._mt)
                pos = mt._mti
                has_gauss = 1 if mt._gauss_cache is not None else 0
                cached = mt._gauss_cache if mt._gauss_cache is not None else 0.0
                return ('MT19937', keys, pos, has_gauss, cached)
            return get_state
        if name == 'set_state':
            def set_state(state):
                # 恢复完整 MT19937 缓冲 + 位置 + 高斯缓存（numpy legacy，逐位精确）。
                # 兼容 3 元素旧格式（无 has_gauss/cached）。
                keys = [int(k) & 0xffffffff for k in state[1]]
                pos = int(state[2])
                self._mt._mt = keys
                self._mt._mti = pos
                if len(state) >= 5 and state[3]:
                    self._mt._gauss_cache = float(state[4])
                else:
                    self._mt._gauss_cache = None
            return set_state
        if name == 'standard_normal':
            def standard_normal(size=None):
                return self._inner.randn(*( (size,) if isinstance(size, int)
                                            else tuple(size) if size else (1,)))
            return standard_normal
        if name == 'multinomial':
            def multinomial(n, pvals, size=None):
                return self._multinomial_impl(n, pvals, size)
            return multinomial
        if name == 'dirichlet':
            def dirichlet(alpha, size=None):
                return self._dirichlet_impl(alpha, size)
            return dirichlet
        if name == 'vonmises':
            def vonmises(*args, **kwargs):
                return self._vonmises_impl(*args, **kwargs)
            return vonmises
        if name in _RS_DIST_DISPATCH:
            _sampler_name = _RS_DIST_DISPATCH[name]
            _int = name in _RS_INT_DISTS
            def _dist_method(*args, _sampler_name=_sampler_name, _int=_int,
                             **kwargs):
                return self._dist_call(_sampler_name, args, kwargs, _int)
            return _dist_method
        return getattr(self._inner, name)

    def _dist_call(self, sampler_name, args, kwargs, int_dist):
        """位置 + 关键字参数 → 规范顺序；size 取 kwargs 或多余位置参数。"""
        names = _RS_DIST_PARAMS.get(sampler_name, ())
        size = kwargs.pop('size', None)
        pos = list(args)
        if size is None and len(pos) > len(names):
            extra = pos[len(names):]
            pos = pos[:len(names)]
            size = tuple(extra) if len(extra) > 1 else extra[0]
        vals = list(pos)
        for i, nm in enumerate(names):
            while len(vals) <= i:
                vals.append(_MISSING)
            if vals[i] is _MISSING and nm in kwargs:
                vals[i] = kwargs.pop(nm)
        for i, nm in enumerate(names):
            if vals[i] is _MISSING:
                vals[i] = _RS_DIST_DEFAULTS.get((sampler_name, nm))
        while vals and vals[-1] is _MISSING:
            vals.pop()
        for v in vals:
            if v is _MISSING:
                raise TypeError(f"missing required argument for {sampler_name}")
        if kwargs:
            raise TypeError(
                f"unexpected keyword argument {next(iter(kwargs))!r}")
        return self._sample_dist(sampler_name, tuple(vals), size,
                                 int_dist=int_dist)

    # ------------------------------------------------------------------
    # 纯 Python 补齐的 legacy 方法（统计正确；逐位兼容流需 MT19937 内核）
    # ------------------------------------------------------------------
    def _sample_dist(self, name, params, size, int_dist=False):
        return _sample_dist(self._sampler, name, params, size, int_dist)

    def _multinomial_impl(self, n, pvals, size):
        return _multinomial_impl(self._sampler, n, pvals, size)

    def _dirichlet_impl(self, alpha, size):
        if isinstance(alpha, (str, bytes)) or (
                not isinstance(alpha, (list, tuple, np.ndarray))
                and not hasattr(alpha, 'shape')):
            alpha = [float(alpha)]
        arr = np.asarray(alpha, dtype='float64')
        if arr.ndim != 1:
            raise ValueError("alpha must be a 1-d sequence")
        a_list = arr.ravel().tolist()
        k = len(a_list)
        ssize = _norm_size(size)
        out_shape = ssize + (k,)
        total = 1
        for d in ssize:
            total *= d
        res = []
        for _ in range(total):
            res.extend(self._sampler.dirichlet(a_list))
        if out_shape == ():
            return res
        return np.asarray(res, dtype='float64').reshape(out_shape)

    def _vonmises_impl(self, *args, **kwargs):
        """numpy legacy API：vonmises(mu=0.0, kappa=1.0, size=None)；
        位置参数按 (mu, kappa, [size]) 顺序，mu/kappa 支持数组广播。"""
        size = kwargs.pop('size', None)
        mu = kwargs.pop('mu', None)
        kappa = kwargs.pop('kappa', None)
        if args:
            if mu is None:
                mu = args[0]
            if len(args) >= 2 and kappa is None:
                kappa = args[1]
            if len(args) >= 3 and size is None:
                size = args[2]
        if kappa is None:
            raise TypeError("kappa is required")
        if kwargs:
            raise TypeError(
                f"unexpected keyword argument {next(iter(kwargs))!r}")
        mu_a = np.asarray(0.0 if mu is None else mu, dtype='float64')
        k_a = np.asarray(kappa, dtype='float64')
        # numpy CONS_NON_NEGATIVE：负 kappa → ValueError（NaN 放行到采样器）
        if any(k < 0.0 for k in k_a.ravel().tolist()):
            raise ValueError("kappa must be non-negative")
        mu_b, k_b = np.broadcast_arrays(mu_a, k_a)
        bshape = tuple(mu_b.shape)
        ssize = _norm_size(size)
        if bshape:
            # numpy 语义：size 与参数广播形一起（右对齐）广播；不兼容时
            # ValueError（与 _sample_dist 同一规则，已实测对齐）
            bshape = _broadcast_shapes(ssize, bshape)
            mu_b = np.broadcast_to(mu_b, bshape)
            k_b = np.broadcast_to(k_b, bshape)
        out_shape = bshape if bshape else ssize
        mflat = mu_b.ravel().tolist()
        kflat = k_b.ravel().tolist()
        bn = 1
        for d in bshape:
            bn *= d
        total = 1
        for d in out_shape:
            total *= d
        if total == 0:
            return np.empty(out_shape)
        res = [self._sampler.vonmises(mflat[i % bn], kflat[i % bn])
               for i in range(total)]
        return np.asarray(res, dtype='float64').reshape(out_shape)

    def randint(self, low, high=None, size=None, dtype='int64', endpoint=False):
        if high is None:
            low, high = 0, int(low)
        # dtype 解析：兼容 'int32'/'i4'/'uint64'/'u8'/Python 类型；
        # 非原生字节序（'>i4' 等）按 numpy 语义发 DeprecationWarning。
        raw = str(dtype)
        if any(ch in raw for ch in '<>=|'):
            import warnings
            warnings.warn("non-native byteorder is not supported",
                          DeprecationWarning)
        name = raw.lstrip('<>=| ')
        parsed = _parse_int_dtype(name)
        if parsed is None:
            name = str(np.dtype(dtype)).lstrip('<>=| ')
            parsed = _parse_int_dtype(name)
        if parsed is None:
            raise TypeError(f"dtype {dtype!r} is not a valid integer dtype")
        kind, bits = parsed
        is_bool = (kind == 'bool')
        signed = (kind == 'int')
        if is_bool:
            dmin, dmax = 0, 1
        else:
            dmin = 0 if not signed else -(1 << (bits - 1))
            dmax = (1 << (bits - 1)) - 1 if signed else (1 << bits) - 1
        # low/high 可为标量或数组（按 numpy 语义广播）。
        # 0-d 标量（rsnumpy 标量 / 0-d 数组）走 int() 取精确值：asarray 路径
        # 在 f64 2^63 边界丢精度（int64.max → 2^63 → tolist 环绕成 -2^63），
        # 而 int() 对标量返回精确值。无 ndim 属性的 list 等仍走 asarray 路径。
        def _is_scalar0(x):
            return _is_arraylike(x) and getattr(x, 'ndim', None) == 0
        lo_arr = _is_arraylike(low) and not _is_scalar0(low)
        hi_arr = _is_arraylike(high) and not _is_scalar0(high)
        if lo_arr or hi_arr:
            la = np.asarray(low, dtype='int64') if lo_arr else np.asarray(low)
            ha = np.asarray(high, dtype='int64') if hi_arr else np.asarray(high)
            la, ha = np.broadcast_arrays(la, ha)
            los = la.ravel().tolist()
            his = ha.ravel().tolist()
            bshape = tuple(la.shape)
        else:
            los = [int(low)]
            his = [int(high)]
            bshape = ()
        ssize = _norm_size(size)
        out_shape = ssize + bshape
        total = 1
        for d in out_shape:
            total *= d
        # numpy 语义：零长度 size 直接短路（不做 low/high 校验，
        # randint(0, -10, size=0) → shape (0,)）
        if total == 0:
            return np.zeros(out_shape, dtype=name)
        # 检查结果值域宽度是否放得进 dtype（numpy 语义：high 本身可越 dtype
        # 上界，如 randint(2**63-1, 2**63, dtype='int64')——int64/uint64 无
        # 公共整数超集，按值域宽度判，不按 high 判）
        maxwidth = dmax - dmin + 1 if dmax >= dmin else 0
        for lo, hi in zip(los, his):
            if hi <= lo or hi - lo > maxwidth:
                raise ValueError(
                    f"low/high ({lo}, {hi}) out of range for dtype {name}")

        bn = 1
        for d in bshape:
            bn *= d
        # numpy legacy randint 按宽度采样（_bounded_integers 掩码路径，
        # distributions.c）：u8=4 值/字、u16=2 值/字、bool=32 bit/字
        # （均 LSB 起 + 掩码拒绝），u32/u64=1 字/值（interval，无偏拒绝）。
        # 同宽度 signed 与 unsigned 流相同（仅二补体重解释）；单次调用内
        # buf 跨元素共享、跨调用全新。
        rngs = [his[i % bn] - 1 - los[i % bn] for i in range(total)]
        if is_bool:
            uint_vals = self._mt.intervals_bool(rngs)
        elif bits == 8:
            uint_vals = self._mt.intervals_u8(rngs)
        elif bits == 16:
            uint_vals = self._mt.intervals_u16(rngs)
        else:
            uint_vals = [self._mt.interval(r) for r in rngs]
        vals = [los[i % bn] + v for i, v in enumerate(uint_vals)]
        if total == 1 and ssize == () and bshape == ():
            v = vals[0]
            # 显式 Python 类型 → 返回 Python 标量（无 .dtype）
            if dtype is bool:
                return bool(v)
            if dtype is int:
                return v
            if is_bool:
                return np.asarray(1 if v else 0, dtype='bool')
            if signed and bits < 64:
                m = 1 << bits
                v = ((v + (1 << (bits - 1))) % m) - (1 << (bits - 1))
            return np.asarray(v, dtype=name)
        if is_bool:
            arr = np.asarray([1 if v else 0 for v in vals], dtype='bool')
        elif signed and bits < 64:
            m = 1 << bits
            vals = [((v + (1 << (bits - 1))) % m) - (1 << (bits - 1))
                    for v in vals]
            arr = np.asarray(vals, dtype='int64').astype(name)
        elif signed:
            arr = np.asarray(vals, dtype='int64')
        else:
            if bits < 64:
                m = 1 << bits
                vals = [v % m for v in vals]
            arr = np.asarray(vals, dtype='uint64' if bits == 64 else 'int64')
            if bits < 64:
                arr = arr.astype(name)
        return arr.reshape(out_shape)

    def random_integers(self, low, high=None, size=None):
        import warnings
        warnings.warn("random_integers is deprecated; use randint instead",
                      DeprecationWarning)
        if high is None:
            high = low
        return self.randint(low, high + 1, size=size)

    def bytes(self, length=4):
        # numpy legacy：randint(0, 2**32, size=n_uint32, dtype='uint32')
        # 的原生（小端）字节串截断——直接取 u32 流，非 res53 缩放。
        n = int(length)
        n_uint32 = (n - 1) // 4 + 1
        return b''.join(self._mt.interval(0xffffffff).to_bytes(4, 'little')
                        for _ in range(n_uint32))[:n]

    def tomaxint(self, type='int32'):
        import re
        m = re.match(r'[<>=|]?int(\d+)', str(type))
        bits = int(m.group(1)) if m else 32
        return (1 << (bits - 1)) - 1

    def shuffle(self, x, axis=0):
        if axis != 0 and hasattr(x, 'shape'):
            x = np.moveaxis(x, axis, 0)
        _python_shuffle(lambda i: self._mt.interval(i), x, 0,
                        warn_1d_object=True)

    def seed(self, s=None):
        s = _legacy_validate_seed(s)
        self._seed = s
        self._mt.reseed(s)
        self._inner = _core.random.RandomState(_fold_seed(s))

    @property
    def _bit_generator(self):
        return _MT19937(None)

    @property
    def _poisson_lam_max(self):
        """Knuth 采样上限；超过后切换正态近似（测试读取此常数）。"""
        return _POISSON_LAM_MAX


def _mvn_sample(next_standard_normal, mean, cov, size,
                check_valid='warn', tol=1e-8):
    """Draw multivariate normal samples via the SVD factor of ``cov``."""
    import rsnumpy as np
    mean = np.asarray(mean, dtype='float64').reshape(-1).tolist()
    cov = np.asarray(cov, dtype='float64')
    d = len(mean)
    _, s, vt = np.linalg.svd(cov, full_matrices=False)
    s_arr = np.asarray(s, dtype='float64')
    if s_arr.ndim == 2:
        # rsnumpy svd 对 2x2 返回 2D 对角矩阵（非常规）→ 取对角线奇异值
        s_vals = [s_arr[i, i] for i in range(min(s_arr.shape))]
    else:
        s_vals = s_arr.ravel().tolist()
    sl = [max(float(x), 0.0) ** 0.5 for x in s_vals]
    vtl = [[float(x) for x in row] for row in vt.tolist()]
    if check_valid != 'ignore':
        # numpy 语义（mtrand.pyx）：psd = allclose(V·diag(s)·Vt, cov,
        # rtol=tol, atol=tol)；非对称/非半正定时 V≠U，重构与 cov 偏离。
        if check_valid not in ('warn', 'raise'):
            raise ValueError(
                "check_valid must equal 'warn', 'raise', or 'ignore'")
        _covf = cov.reshape(-1).tolist()
        _psd = True
        for _a in range(d):
            for _c in range(d):
                _acc = 0.0
                for _b in range(d):
                    _acc += vtl[_b][_a] * s_vals[_b] * vtl[_b][_c]
                _ref = _covf[_a * d + _c]
                if abs(_acc - _ref) > tol + tol * abs(_ref):
                    _psd = False
                    break
            if not _psd:
                break
        if not _psd:
            if check_valid == 'warn':
                import warnings
                warnings.warn(
                    "covariance is not symmetric positive-semidefinite.",
                    RuntimeWarning)
            else:
                raise ValueError(
                    "covariance is not symmetric positive-semidefinite.")
    shape = _norm_size(size)
    n = int(np.prod(shape)) if shape else 1
    out = []
    for _ in range(n):
        z = [next_standard_normal() for _ in range(d)]
        row = []
        for i in range(d):
            acc = mean[i]
            for j in range(d):
                acc += vtl[j][i] * sl[j] * z[j]
            row.append(acc)
        out.append(row)
    return np.asarray(out).reshape(shape + (d,))


def _coerce_size_arg(args, kwargs):
    """size 参数数组化 → int/tuple（numpy 接受 array-like size，Rust 后端不接受）。"""

    def _conv(v):
        if isinstance(v, bool) or isinstance(v, int):
            return v
        if isinstance(v, (tuple, list)):
            return tuple(int(x) for x in v)
        try:
            arr = np.asarray(v)
            if arr.ndim == 0:
                return int(arr.item())
            return tuple(int(x) for x in arr.reshape((-1,)).tolist())
        except Exception:
            return v

    out_args = list(args)
    if out_args:
        out_args[0] = _conv(out_args[0])
    if 'size' in kwargs:
        kwargs['size'] = _conv(kwargs['size'])
    return tuple(out_args), kwargs


class _WrappedGenerator:
    """Generator 的薄包装，把核心返回的数组提升为 rsnumpy 数组。

    核心生成器（`_core.random.Generator`）返回的是 Rust 侧的 `ndarray`，它和
    `rsnumpy.ndarray` 不是同一个类：后者才有那些补齐了 NumPy 语义的 Python 层
    方法。差别最明显的是 `astype` —— 核心版只接受字符串，而 NumPy 语义要求也能
    吃 `dtype` 实例，因此诸如

        a = rng.random(shape).astype(np.dtype('float64'))

    的写法（numpy 约定 scipy 测试里到处在用）在裸核心对象上会直接报错。

    这里只做一件事：转发所有属性，并把方法返回值里的核心数组换成 `rsnumpy.ndarray`。
    """

    __slots__ = ('_inner', '_seed')

    def __init__(self, inner=None):
        # Accept NumPy-style construction: Generator(seed) or Generator()
        self._seed = inner
        if isinstance(inner, _WrappedGenerator):
            self._seed = inner._seed
            inner = inner._inner
        elif not hasattr(inner, 'random'):
            try:
                inner = _core.random.Generator(inner)
            except OverflowError:
                # NumPy accepts arbitrarily large seeds (SeedSequence);
                # fold them into the range the core understands.
                inner = _core.random.Generator(int(inner) % (2**63))
        self._inner = inner

    def __deepcopy__(self, memo):
        return _WrappedGenerator(self._seed)

    def __copy__(self):
        return _WrappedGenerator(self._seed)

    def _u(self):
        """One uniform float in [0, 1) from the inner generator."""
        v = self._inner.random()
        if hasattr(v, 'item'):
            v = v.item()
        return float(v)

    def shuffle(self, x, axis=0):
        """新 API 洗牌：object 数组不告警、保持对象身份。

        有界抽样走 Rust 内核 integers(0, i+1)。注：新 API 内核是 StdRng，
        与 numpy PCG64 流不同，位级精确需另行移植 PCG64（见独立任务）。
        """
        if axis != 0 and hasattr(x, 'shape'):
            x = np.moveaxis(x, axis, 0)

        def next_j(i):
            v = self._inner.integers(0, i + 1)
            if hasattr(v, 'item'):
                v = v.item()
            return int(v)
        _python_shuffle(next_j, x, 0)

    def _make_choice(self):
        """NumPy-compatible ``choice`` (the Rust core only takes array `a`
        and no ``p``; int populations and weights are handled here)."""
        def choice(a, size=None, replace=True, p=None):
            import rsnumpy as np
            if hasattr(a, 'shape') or isinstance(a, (list, tuple)):
                values = np.asarray(a)
                pop = values.shape[0] if values.ndim > 0 else values.size
            else:
                values = None
                pop = int(a)
            if size is None:
                sz_shape = ()
                n = 1
            elif isinstance(size, (int,)):
                sz_shape = (int(size),)
                n = int(size)
            else:
                sz_shape = tuple(int(x) for x in size)
                n = 1
                for x in sz_shape:
                    n *= x
            if not replace and pop < n:
                raise ValueError(
                    "Cannot take a larger sample than population when "
                    "'replace=false'")
            if p is not None:
                pw = np.asarray(p, dtype='float64').reshape(-1).tolist()
                total = float(sum(pw))
                cum = []
                acc = 0.0
                for x in pw:
                    acc += float(x) / total
                    cum.append(acc)
                cum[-1] = 1.0

                def draw():
                    u = self._u()
                    for i, c in enumerate(cum):
                        if u <= c:
                            return i
                    return len(cum) - 1
            else:
                def draw():
                    return int(self._u() * pop)
            idx = [draw() for _ in range(n)]
            if not replace:
                seen = set()
                fixed = []
                for v in idx:
                    while v in seen:
                        v = draw()
                    seen.add(v)
                    fixed.append(v)
                idx = fixed
            if values is None:
                res = np.asarray(idx)
            else:
                res = np.take(values, np.asarray(idx))
            if size is None:
                return res.reshape(sz_shape)[()]
            return res.reshape(sz_shape)
        return choice

    def _make_shuffle(self):
        """NumPy-compatible in-place ``shuffle`` (the Rust core only accepts
        its own ndarray type, so shuffle at the Python layer)."""
        def shuffle(x, axis=0):
            import rsnumpy as np
            arr = x if axis == 0 else np.moveaxis(x, axis, 0)
            n = arr.shape[0]
            for i in range(n - 1, 0, -1):
                j = int(self.integers(i + 1))
                if j == i:
                    continue
                tmp = arr[i].copy()
                arr[i] = arr[j]
                arr[j] = tmp
        return shuffle

    def __getattr__(self, name):
        if name == 'choice':
            return self._make_choice()
        if name == 'shuffle':
            return self._make_shuffle()
        attr = getattr(self._inner, name, None)
        if attr is None:
            # Fallback for methods not available on the inner generator
            if name == 'vonmises':
                def vonmises_wrapper(kappa, size=None, mu=0.0, dtype=None):
                    import rsnumpy as np
                    import math
                    # Broadcast mu/kappa to a common shape, then optionally to
                    # ``size``.  This lets callers pass array parameters (as
                    # rsscipy's distribution layer does) as well as scalars.
                    mu_b, kappa_b = np.broadcast_arrays(
                        np.asarray(mu, dtype='float64'),
                        np.asarray(kappa, dtype='float64'))
                    if size is None:
                        shape = tuple(mu_b.shape)
                    else:
                        shape = (tuple(size) if isinstance(size, (tuple, list))
                                 else (int(size),))
                        mu_b = np.broadcast_to(mu_b, shape)
                        kappa_b = np.broadcast_to(kappa_b, shape)
                    flat_mu = mu_b.reshape(-1).tolist()
                    flat_k = kappa_b.reshape(-1).tolist()
                    result = []
                    for m, k in zip(flat_mu, flat_k):
                        m = float(m)
                        k = float(k)
                        while True:
                            u1 = self._inner.random().item()
                            u2 = self._inner.random().item()
                            z = math.cos(math.pi * u1)
                            if u2 < math.exp(k * (z - 1)):
                                break
                        angle = math.acos(max(-1.0, min(1.0, z)))
                        if self._inner.random().item() < 0.5:
                            angle = -angle
                        result.append(m + angle)
                    result_arr = np.asarray(result).reshape(shape)
                    if dtype is not None and hasattr(result_arr, 'astype'):
                        result_arr = result_arr.astype(dtype)
                    return result_arr
                return vonmises_wrapper
            if name == 'multivariate_normal':
                def mvn_wrapper(mean, cov, size=None):
                    def _std_normal():
                        v = self._inner.standard_normal()
                        if hasattr(v, 'item'):
                            v = v.item()
                        return float(v)
                    return _mvn_sample(_std_normal, mean, cov, size)
                return mvn_wrapper
            if name == 'permutation':
                def permutation_wrapper(x, axis=0):
                    import rsnumpy as np
                    arr = np.asarray(x)
                    idx = self._inner.permutation(arr.shape[axis])
                    return np.take(arr, idx, axis=axis)
                return permutation_wrapper
            if name == 'permutated':
                def permutated_wrapper(x, axis=0, *args, **kwargs):
                    import rsnumpy as np
                    arr = np.asarray(x)
                    idx = self._inner.permutation(arr.shape[axis])
                    return np.take(arr, idx, axis=axis)
                return permutated_wrapper
            raise AttributeError(f"'{type(self._inner).__name__}' object has no attribute '{name}'")
        if callable(attr):
            def wrapper(*args, **kwargs):
                # Strip dtype from kwargs - rsnumpy core doesn't support it
                dtype = kwargs.pop('dtype', None)
                result = attr(*args, **kwargs)
                import rsnumpy as np
                # Wrap builtins.ndarray results into rsnumpy ndarray
                if type(result).__module__ == 'builtins' and type(result).__name__ == 'ndarray':
                    result = np.asarray(result)
                # Convert dtype if specified
                if dtype is not None and hasattr(result, 'astype'):
                    result = result.astype(dtype)
                return result
            return wrapper
        return attr

    def _convert(self, value):
        if type(value).__module__ == 'builtins' and \
                type(value).__name__ == 'ndarray':
            # `_core.ndarray`：Rust 侧类型，包一层即可获得 Python 层 API。
            import rsnumpy as np
            return np.asarray(value)
        return value

    def random(self, *args, **kwargs):
        dtype = kwargs.pop('dtype', None)
        args, kwargs = _coerce_size_arg(args, kwargs)
        # Rust Generator.random 不接受位置 size，统一改走 size= 关键字
        if args:
            kwargs.setdefault('size', args[0])
            args = args[1:]
        result = self._convert(self._inner.random(*args, **kwargs))
        if dtype is not None and hasattr(result, 'astype'):
            result = result.astype(dtype)
        return result

    def standard_normal(self, *args, **kwargs):
        dtype = kwargs.pop('dtype', None)
        result = self._convert(self._inner.standard_normal(*args, **kwargs))
        if dtype is not None and hasattr(result, 'astype'):
            result = result.astype(dtype)
        return result

    def normal(self, *args, **kwargs):
        dtype = kwargs.pop('dtype', None)
        result = self._convert(self._inner.normal(*args, **kwargs))
        if dtype is not None and hasattr(result, 'astype'):
            result = result.astype(dtype)
        return result

    def uniform(self, *args, **kwargs):
        dtype = kwargs.pop('dtype', None)
        result = self._convert(self._inner.uniform(*args, **kwargs))
        if dtype is not None and hasattr(result, 'astype'):
            result = result.astype(dtype)
        return result

    def integers(self, *args, **kwargs):
        dtype = kwargs.pop('dtype', None)
        result = self._convert(self._inner.integers(*args, **kwargs))
        if dtype is not None and hasattr(result, 'astype'):
            result = result.astype(dtype)
        return result

    def multinomial(self, n, pvals, size=None):
        import rsnumpy as np
        pvals = np.asarray(pvals, dtype=float)
        pvals = pvals / pvals.sum()
        cdf = np.cumsum(pvals)
        if size is None:
            size = ()
        n_samples = int(np.prod(size)) if isinstance(size, tuple) else int(size)
        # Count occurrences of each category
        result = np.zeros(len(pvals), dtype=int)
        for _ in range(n_samples):
            u = float(np.asarray(self._inner.random()).tolist())
            if isinstance(u, list):
                u = u[0]
            idx = int(np.asarray(cdf).searchsorted(u) if hasattr(cdf, 'searchsorted') else 0)
            idx = min(idx, len(pvals) - 1)
            result[idx] += 1
        return np.asarray(result)

    def permuted(self, x, axis=0, *args, **kwargs):
        import rsnumpy as np
        arr = np.asarray(x)
        idx = self._inner.permutation(arr.shape[axis])
        return np.take(arr, idx, axis=axis)


class mtrand_module:
    """mtrand 模块，提供全局随机数生成器"""
    def __init__(self):
        self._rand = _RandomStateWrapper(None)

    def __getattr__(self, name):
        # np.random.mtrand.<分布>/方法 全部委托给 legacy RandomState
        return getattr(self._rand, name)


class _SeedSequence:
    """Minimal stand-in for ``np.random.SeedSequence``.

    Only the attributes downstream code inspects (``entropy``) are kept.
    """

    def __init__(self, entropy=None):
        # 官方语义：entropy 为 int 或 1-d int 序列；嵌套序列（含 SeedSequence
        # 自身与循环引用）拒绝——numpy 抛 TypeError。
        if isinstance(entropy, _SeedSequence):
            raise TypeError("SeedSequence does not accept nested sequences.")
        if isinstance(entropy, (list, tuple)):
            for _item in entropy:
                if isinstance(_item, (list, tuple, _SeedSequence)):
                    raise TypeError(
                        "SeedSequence does not accept nested sequences.")
        self.entropy = entropy


class random_module:
    """随机数生成模块"""

    def __init__(self):
        self._random = _core.random
        self.mtrand = mtrand_module()

    @property
    def Generator(self):
        # Expose the wrapper class so ``isinstance(x, np.random.Generator)``
        # works for generators produced by ``default_rng``.
        return _WrappedGenerator

    @property
    def RandomState(self):
        return _RandomStateWrapper

    @property
    def SeedSequence(self):
        return _SeedSequence

    @property
    def BitGenerator(self):
        return _MT19937

    def default_rng(self, seed=None):
        """创建默认的随机数生成器（新 API）。

        参数:
            seed: 随机种子。

        返回:
            Generator: 随机数生成器。
        """
        # If seed is already a wrapped generator, return it directly
        if isinstance(seed, _WrappedGenerator):
            return seed
        return _WrappedGenerator(seed)

    def seed(self, s):
        """设置随机种子。"""
        return self._random.seed(s)
    
    def _rand_arr(self, vals):
        """确保输出是带 _dtype 的正规 rsnumpy ndarray（Rust 原生对象转换）。"""
        from ..__init__ import ndarray
        if hasattr(vals, 'tolist') and not hasattr(vals, '_dtype'):
            vals = vals.tolist()
        return ndarray(vals)

    def rand(self, *args):
        """生成均匀分布的随机数。"""
        if len(args) == 0:
            return self._rand_arr(self._random.rand(1))
        return self._rand_arr(self._random.rand(*args))

    def randn(self, *args):
        """生成标准正态分布的随机数。"""
        if len(args) == 0:
            return self._rand_arr(self._random.randn(1))
        return self._rand_arr(self._random.randn(*args))
    
    def randint(self, low, high=None, size=None):
        """生成整数随机数。"""
        from ..__init__ import ndarray
        return ndarray._wrap(self._random.randint(low, high, size), "int64")
    
    def uniform(self, low=0.0, high=1.0, size=None):
        """生成均匀分布的随机数。"""
        from ..__init__ import ndarray
        return ndarray(self._random.uniform(low, high, size))
    
    def random(self, size=None):
        """生成 [0.0, 1.0) 区间均匀分布的随机数。"""
        from ..__init__ import ndarray
        return ndarray(self._random.uniform(0.0, 1.0, size))
    
    def normal(self, loc=0.0, scale=1.0, size=None):
        """生成正态分布的随机数。"""
        from ..__init__ import ndarray
        if size is None:
            size = 1
        return ndarray(self._random.normal(loc, scale, size))
    
    def randn_like(self, a):
        """生成与输入数组形状相同的标准正态分布随机数。"""
        from ..__init__ import ndarray
        arr = ndarray(a)
        return ndarray(self._random.randn(*arr.shape))
    
    def rand_like(self, a):
        """生成与输入数组形状相同的均匀分布随机数。"""
        from ..__init__ import ndarray
        arr = ndarray(a)
        return ndarray(self._random.rand(*arr.shape))


# ---------------------------------------------------------------------------
# numpy 兼容的模块级 API：numpy 里 ``np.random`` 是模块，``from numpy.random
# import rand`` 是合法用法。以下模块级函数绑定同一个默认实例，且
# ``rsnumpy.random``（见 python/rsnumpy/__init__.py）指向本模块本身，
# 两种访问形式共享同一份全局状态。
# ---------------------------------------------------------------------------
_default_rng = random_module()
Generator = _default_rng.Generator
RandomState = _default_rng.RandomState
SeedSequence = _default_rng.SeedSequence
BitGenerator = _default_rng.BitGenerator
mtrand = _default_rng.mtrand
default_rng = _default_rng.default_rng

def rand(*args):
    # MT19937 全局流（与 randint/uniform 共用 _get_legacy_state，匹配 numpy）
    return _get_legacy_state().rand(*args)

randn = _default_rng.randn

def random(*args, **kwargs):
    # numpy legacy random_sample：MT19937 全局流
    return _get_legacy_state().random(*args, **kwargs)

random_sample = random
randn_like = _default_rng.randn_like
rand_like = _default_rng.rand_like

_legacy_state = None


def _get_legacy_state():
    global _legacy_state
    if _legacy_state is None:
        _legacy_state = _RandomStateWrapper(None)
    return _legacy_state


def seed(s=None):
    """设置全局随机种子（legacy 状态 + Rust 全局流，共享）。"""
    s = _legacy_validate_seed(s)
    if s is not None:
        try:
            _default_rng._random.seed(_fold_seed(s))
        except Exception:
            pass
    _get_legacy_state().seed(s)


def randint(low, high=None, size=None, dtype='int64'):
    """legacy randint：完整 dtype 语义（校验/边界/位宽截断）。"""
    return _get_legacy_state().randint(low, high, size=size, dtype=dtype)


def uniform(low=0.0, high=1.0, size=None):
    return _get_legacy_state()._sample_dist('uniform', (low, high), size)


def normal(loc=0.0, scale=1.0, size=None):
    return _get_legacy_state()._sample_dist('normal', (loc, scale), size)


_legacy_func_cache = {}


def __getattr__(name):
    """PEP 562：补齐遗留 np.random.* 函数名（分布/permutation/shuffle 等）。

    结果缓存：保证 ``np.random.geometric in (np.random.geometric,)`` 之类的
    函数身份比较成立（tests 用 ``func in probfuncs`` 判定）。
    """
    if name in _RS_DIST_DISPATCH or name in (
            'multinomial', 'dirichlet', 'vonmises', 'permutation', 'shuffle',
            'choice', 'multivariate_normal', 'random_integers', 'bytes',
            'tomaxint', 'standard_normal'):
        f = _legacy_func_cache.get(name)
        if f is None:
            f = getattr(_get_legacy_state(), name)
            _legacy_func_cache[name] = f
        return f
    raise AttributeError(f"module 'rsnumpy.random' has no attribute {name!r}")
