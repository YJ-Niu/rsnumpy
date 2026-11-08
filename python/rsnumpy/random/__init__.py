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
            a = np.asarray(p, dtype='float64')
            if a.ndim == 0:
                # 0-d：numpy 以 __float__ 提取标量，子类覆盖须被尊重
                # （test_scalar_exception_propagation）
                param_vals.append(('s', float(p)))
                continue
            param_vals.append(('a', a))
            arrs.append(a)
        else:
            param_vals.append(('s', float(p)))
    if arrs:
        bcast = np.broadcast_arrays(*arrs)
        bshape = tuple(bcast[0].shape)
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
    ssize = _norm_size(size)
    out_shape = ssize + bshape
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


def _multinomial_impl(draw_u, n, pvals, size):
    """numpy 语义 multinomial：pvals 必须 1-d；空 pvals → 尾维 0。"""
    if isinstance(pvals, (str, bytes)) or (
            not isinstance(pvals, (list, tuple, np.ndarray))
            and not hasattr(pvals, 'shape')):
        raise TypeError("pvals must be a 1-d sequence of probabilities")
    parr = np.asarray(pvals, dtype='float64')
    if parr.ndim != 1:
        raise ValueError("pvals must be a 1-d sequence of probabilities")
    k = int(parr.size)
    pvals_list = parr.ravel().tolist()
    if _is_arraylike(n):
        n_arr = np.asarray(n, dtype='int64')
        n_shape = tuple(n_arr.shape)
        n_flat = n_arr.ravel().tolist()
    else:
        n_shape = ()
        n_flat = [int(n)]
    ssize = _norm_size(size)
    draws_shape = ssize + n_shape
    out_shape = draws_shape + (k,)
    total = 1
    for d in draws_shape:
        total *= d
    if k == 0:
        return np.zeros(out_shape)
    psum = sum(pvals_list)
    if psum <= 0:
        raise ValueError("pvals must sum to a positive value")
    cdf = []
    acc = 0.0
    for pv in pvals_list:
        acc += pv / psum
        cdf.append(acc)
    cdf[-1] = 1.0
    bn_n = 1
    for d in n_shape:
        bn_n *= d
    res = []
    for i in range(total):
        n_i = int(n_flat[i % bn_n])
        counts = [0] * k
        for _ in range(max(0, n_i)):
            u = draw_u()
            for ci, c in enumerate(cdf):
                if u <= c:
                    counts[ci] += 1
                    break
            else:
                counts[k - 1] += 1
        res.extend(counts)
    return np.asarray(res, dtype='int64').reshape(out_shape)


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
    'wald': ('mean', 'gamma'),
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
    if arr.ndim != 1:
        raise ValueError("Seed must be a 1-d array of integers or None")
    if arr.size == 0:
        raise ValueError("Seed must be a non-empty array or None")
    out = 0
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
        out = (out * 1000003 + iv) % 2 ** 32
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


def _python_shuffle(next_u, x, axis=0, warn_1d_object=False):
    """Fisher-Yates 洗牌：list / ndarray（含 object/结构化/memoryview）。

    object 数组逐元素搬运 Python 对象引用（不复制，保持对象身份）；
    数值数组按行搬运（行 view 需 copy 后再写回）。
    """
    if isinstance(x, list):
        n = len(x)
        for i in range(n - 1, 0, -1):
            j = int(next_u() * (i + 1))
            x[i], x[j] = x[j], x[i]
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
            j = int(next_u() * (i + 1))
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
        j = int(next_u() * (i + 1))
        x[i], x[j] = x[j], x[i]


class _RandomStateWrapper:
    """Python wrapper around the core RandomState.

    Adds the legacy attributes SciPy expects (``_bit_generator``) and
    distributions the Rust core does not implement (``multivariate_normal``
    以及 _distributions 补齐的完整 legacy 分布集)。
    """

    def __init__(self, seed=None):
        seed = _legacy_validate_seed(seed)
        self._seed = seed
        self._sampler = Samplers(self._next_u, self._next_normal)
        self._inner = _core.random.RandomState(seed)

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
        """生成均匀分布的随机数。"""
        return self._rs_f64(self._inner.rand(1) if not args
                            else self._inner.rand(*args))

    def randn(self, *args):
        """生成标准正态分布的随机数。"""
        return self._rs_f64(self._inner.randn(1) if not args
                            else self._inner.randn(*args))

    def _next_u(self):
        return float(self._inner.random(1).tolist()[0])

    def _next_normal(self):
        return float(self._inner.randn(1).tolist()[0])

    def __getattr__(self, name):
        if name == 'multivariate_normal':
            def mvn(mean, cov, size=None):
                return _mvn_sample(self._next_normal, mean, cov, size)
            return mvn
        if name == 'choice':
            def choice(a, size=None, replace=True, p=None):
                import rsnumpy as np
                if isinstance(a, int):
                    if a < 0:
                        raise ValueError("'a' must be a non-negative integer")
                    pop = a
                else:
                    arr0 = np.asarray(a)
                    if arr0.ndim != 1:
                        raise ValueError(
                            "'a' must be a 1-d array or a non-negative integer")
                    pop = int(arr0.shape[0])
                    if pop == 0:
                        raise ValueError(
                            "'a' must be a non-empty 1-d array or a non-negative integer")
                if p is not None:
                    parr = np.asarray(p, dtype='float64')
                    if parr.ndim != 1:
                        raise ValueError("'a' and 'p' must have same size")
                    pl = parr.ravel().tolist()
                    if len(pl) != pop:
                        raise ValueError("'a' and 'p' must have same size")
                    bad = False
                    for v in pl:
                        if v != v or v < 0:
                            bad = True
                            break
                    if bad:
                        raise ValueError("probabilities contain NaN or are negative")
                    if abs(sum(pl) - 1.0) > 1e-8:
                        raise ValueError("probabilities do not sum to 1")
                    p = pl
                nsize = 1
                if size is not None:
                    if isinstance(size, (tuple, list)):
                        sz_shape = tuple(int(d) for d in size)
                        for d in sz_shape:
                            nsize *= d
                    else:
                        nsize = int(size)
                        sz_shape = (nsize,)
                if not replace and nsize > pop:
                    raise ValueError(
                        "Cannot take a larger sample than population "
                        "when 'replace=False'")
                out = [self._draw_index(pop, p) for _ in range(nsize)]
                if not replace:
                    seen = set()
                    fixed = []
                    for v in out:
                        while v in seen:
                            v = self._draw_index(pop, p)
                        seen.add(v)
                        fixed.append(v)
                    out = fixed
                if isinstance(a, int):
                    res = np.asarray(out)
                else:
                    res = np.take(np.asarray(a), np.asarray(out))
                # 标量分支：rsnumpy 一维数组不支持 [()] 取元，取首元素
                return res.reshape(sz_shape) if size is not None else res[0]
            return choice
        if name == 'permutation':
            def permutation(x):
                import rsnumpy as np
                n = x if isinstance(x, int) else np.asarray(x).shape[0]
                idx = list(range(n))
                for i in range(n - 1, 0, -1):
                    j = int(self._next_u() * (i + 1))
                    idx[i], idx[j] = idx[j], idx[i]
                if isinstance(x, int):
                    return np.asarray(idx)
                return np.take(np.asarray(x), np.asarray(idx))
            return permutation
        if name == 'shuffle':
            def shuffle(x):
                n = x.shape[0]
                for i in range(n - 1, 0, -1):
                    j = int(self._next_u() * (i + 1))
                    x[i], x[j] = x[j], x[i]
            return shuffle
        if name == 'get_state':
            def get_state():
                # Snapshot tuple mirroring NumPy's legacy layout.  The Rust
                # core does not expose the MT19937 buffer, so report the
                # seed plus an empty internal buffer.
                return ('MT19937', [self._seed], 0, 0, 0.0)
            return get_state
        if name == 'set_state':
            def set_state(state):
                # 用记录的关键字头重新播种；核心无法恢复任意 MT19937
                # 缓冲（逐位精确流需 MT19937 内核支持）。
                seed = _legacy_validate_seed(state[1][0])
                self._inner = _core.random.RandomState(seed)
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
        return _multinomial_impl(self._next_u, n, pvals, size)

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
        """legacy (kappa, size) 与新 API 关键字 (mu=, kappa=, size=) 兼容。"""
        size = kwargs.pop('size', None)
        mu = kwargs.pop('mu', 0.0)
        if 'kappa' in kwargs:
            kappa = float(kwargs.pop('kappa'))
        elif args:
            kappa = float(args[0])
            if len(args) >= 2 and size is None:
                size = args[1]
        else:
            raise TypeError("kappa is required")
        if kwargs:
            raise TypeError("unexpected keyword arguments")
        mu = float(mu)
        ssize = _norm_size(size)
        total = 1
        for d in ssize:
            total *= d

        def _one():
            return mu + self._sampler.vonmises(kappa)

        if ssize == ():
            return _one()
        res = [_one() for _ in range(total)]
        return np.asarray(res, dtype='float64').reshape(ssize)

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
            dmax = (1 << bits) - 1
        # low/high 可为标量或数组（按 numpy 语义广播）
        lo_arr = _is_arraylike(low)
        hi_arr = _is_arraylike(high)
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
        for lo, hi in zip(los, his):
            if hi <= lo or lo < dmin or hi - 1 > dmax:
                raise ValueError(
                    f"low/high ({lo}, {hi}) out of range for dtype {name}")

        def _one(lo, hi):
            return lo + int(self._next_u() * (hi - lo))

        ssize = _norm_size(size)
        out_shape = ssize + bshape
        total = 1
        for d in out_shape:
            total *= d
        bn = 1
        for d in bshape:
            bn *= d
        if total == 1 and ssize == () and bshape == ():
            v = _one(los[0], his[0])
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
        vals = []
        for i in range(total):
            bidx = i % bn
            vals.append(_one(los[bidx], his[bidx]))
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
        return bytes(int(self._next_u() * 256) for _ in range(int(length)))

    def tomaxint(self, type='int32'):
        import re
        m = re.match(r'[<>=|]?int(\d+)', str(type))
        bits = int(m.group(1)) if m else 32
        return (1 << (bits - 1)) - 1

    def shuffle(self, x, axis=0):
        if axis != 0 and hasattr(x, 'shape'):
            x = np.moveaxis(x, axis, 0)
        _python_shuffle(self._next_u, x, 0, warn_1d_object=True)

    def seed(self, s=None):
        s = _fold_seed(s)
        self._seed = s
        self._inner = _core.random.RandomState(s)

    def _draw_index(self, pop, p):
        if p is None:
            idx = int(self._next_u() * pop)
        else:
            import rsnumpy as np
            cdf = np.cumsum(np.asarray(p, dtype='float64'))
            u = self._next_u()
            idx = int(np.searchsorted(cdf, u))
        if idx >= pop:
            idx = pop - 1
        return idx

    @property
    def _bit_generator(self):
        return _MT19937(None)

    @property
    def _poisson_lam_max(self):
        """Knuth 采样上限；超过后切换正态近似（测试读取此常数）。"""
        return _POISSON_LAM_MAX


def _mvn_sample(next_standard_normal, mean, cov, size):
    """Draw multivariate normal samples via the SVD factor of ``cov``."""
    import rsnumpy as np
    mean = np.asarray(mean, dtype='float64').reshape(-1).tolist()
    cov = np.asarray(cov, dtype='float64')
    d = len(mean)
    _, s, vt = np.linalg.svd(cov, full_matrices=False)
    sl = [max(float(x), 0.0) ** 0.5 for x in s.tolist()]
    vtl = [[float(x) for x in row] for row in vt.tolist()]
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
        """新 API 洗牌：object 数组不告警、保持对象身份。"""
        if axis != 0 and hasattr(x, 'shape'):
            x = np.moveaxis(x, axis, 0)
        _python_shuffle(self._u, x, 0)

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
rand = _default_rng.rand
randn = _default_rng.randn
random = _default_rng.random
random_sample = _default_rng.random
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
            _default_rng._random.seed(s)
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
