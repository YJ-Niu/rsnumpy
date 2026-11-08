"""纯 Python 分布采样器（legacy RandomState / 模块级函数补齐用）。

设计：无状态。每个采样器由 ``next_u``（返回 [0,1) 均匀浮点）与
``next_normal``（返回标准正态浮点）驱动，返回**单个 Python 标量**。
数组参数广播 / size 前缀 / 结果封装由 ``random.__init__`` 的
``_sample_dist`` 完成，本文件只负责单点采样的数学。

边界语义以 numpy 2.5.4 为 oracle（逐条探测对齐）：
  * 参数为 ``-0.``（负零符号位）或负数 → ValueError；
  * 参数为 ``+0`` → 返回 0（lognormal 为 exp(mean)=1），不报错。
数值算法对齐 numpy 语义（逆变换 / Marsaglia-Tsang gamma /
拒绝采样等），统计性质正确；但不追求与 numpy 逐位一致
（那需要 MT19937/PCG64 的 legacy 精确流，见 linalg 同级的独立任务）。
"""

import math

# numpy 2.5.4 RandomState()._poisson_lam_max 的 oracle 值
_POISSON_LAM_MAX = 9.223372006484771e+18

_ZETA_CACHE = {}


def _is_neg_zero(x):
    return x == 0.0 and math.copysign(1.0, x) < 0.0


def _zeta(a):
    """Riemann zeta(a)，a>1（带缓存，尾部积分界截断）。"""
    if a in _ZETA_CACHE:
        return _ZETA_CACHE[a]
    s = 0.0
    k = 1
    term = 1.0
    while True:
        term = k ** (-a)
        s += term
        # 尾部积分界：sum_{j>k} j^-a <= (k+1)^(1-a)/(a-1)
        if term < 1e-15 * s:
            break
        k += 1
        if k > 10 ** 7:
            break
    _ZETA_CACHE[a] = s
    return s


def _broadcast_shapes(a, b):
    out = []
    la, lb = list(reversed(a)), list(reversed(b))
    for i in range(max(len(la), len(lb))):
        da = la[i] if i < len(la) else 1
        db = lb[i] if i < len(lb) else 1
        if da == 1:
            out.append(db)
        elif db == 1:
            out.append(da)
        elif da == db:
            out.append(da)
        else:
            raise ValueError(
                "shape mismatch: cannot broadcast %s and %s" % (tuple(a), tuple(b)))
    return tuple(reversed(out))


class Samplers:
    """单点采样器集合。所有方法返回单个 Python 标量。"""

    def __init__(self, next_u, next_normal):
        self._u = next_u
        self._n = next_normal

    # ------------------------------------------------------------------
    # 连续分布
    # ------------------------------------------------------------------
    def standard_normal(self):
        return self._n()

    def normal(self, loc=0.0, scale=1.0):
        if scale < 0.0 or _is_neg_zero(scale):
            raise ValueError("scale < 0")
        if scale == 0.0:
            return float(loc)
        return loc + scale * self._n()

    def uniform(self, low=0.0, high=1.0):
        if not (low < high):
            raise ValueError("high <= low")
        return low + (high - low) * self._u()

    def standard_exponential(self):
        u = self._u()
        return -math.log1p(-u)

    def exponential(self, scale=1.0):
        if scale < 0.0 or _is_neg_zero(scale):
            raise ValueError("scale < 0")
        if scale == 0.0:
            return 0.0
        return scale * self.standard_exponential()

    def standard_gamma(self, shape):
        if shape < 0.0 or _is_neg_zero(shape):
            raise ValueError("shape < 0")
        if shape == 0.0:
            return 0.0
        return self._gamma_mt(shape)

    def gamma(self, shape, scale=1.0):
        if shape < 0.0 or _is_neg_zero(shape):
            raise ValueError("shape < 0")
        if scale < 0.0 or _is_neg_zero(scale):
            raise ValueError("scale < 0")
        if shape == 0.0:
            return 0.0
        return scale * self._gamma_mt(shape)

    def chisquare(self, df):
        if df < 0.0 or _is_neg_zero(df):
            raise ValueError("df < 0")
        if df == 0.0:
            return 0.0
        return 2.0 * self._gamma_mt(df / 2.0)

    def _gamma_mt(self, alpha):
        """Marsaglia-Tsang，shape >= 1；shape < 1 用 boost 归约。"""
        if alpha < 1.0:
            u = self._u()
            while u <= 0.0:
                u = self._u()
            return self._gamma_mt(1.0 + alpha) * (u ** (1.0 / alpha))
        d = alpha - 1.0 / 3.0
        c = 1.0 / math.sqrt(9.0 * d)
        while True:
            while True:
                x = self._n()
                v = 1.0 + c * x
                if v > 0.0:
                    break
            v = v * v * v
            u = self._u()
            if u < 1.0 - 0.0331 * (x * x * x * x):
                return d * v
            if math.log(u) < 0.5 * x * x + d * (1.0 - v + math.log(v)):
                return d * v

    def standard_cauchy(self):
        return math.tan(math.pi * (self._u() - 0.5))

    def cauchy(self, loc=0.0, scale=1.0):
        return loc + scale * self.standard_cauchy()

    def gumbel(self, loc=0.0, scale=1.0):
        if scale < 0.0 or _is_neg_zero(scale):
            raise ValueError("scale < 0")
        if scale == 0.0:
            return float(loc)
        u = self._u()
        while u <= 0.0 or u >= 1.0:
            u = self._u()
        return loc - scale * math.log(-math.log1p(-u))

    def laplace(self, loc=0.0, scale=1.0):
        if scale < 0.0 or _is_neg_zero(scale):
            raise ValueError("scale < 0")
        if scale == 0.0:
            return float(loc)
        u = self._u()
        if u < 0.5:
            return loc - scale * math.log1p(-2.0 * u)
        return loc + scale * math.log1p(2.0 * u - 2.0)

    def logistic(self, loc=0.0, scale=1.0):
        if scale < 0.0 or _is_neg_zero(scale):
            raise ValueError("scale < 0")
        if scale == 0.0:
            return float(loc)
        u = self._u()
        while u <= 0.0 or u >= 1.0:
            u = self._u()
        return loc + scale * math.log(u / (1.0 - u))

    def weibull(self, a):
        if a < 0.0 or _is_neg_zero(a):
            raise ValueError("a < 0")
        if a == 0.0:
            return 0.0
        u = self._u()
        return (-math.log1p(-u)) ** (1.0 / a)

    def power(self, a):
        if not (a > 0.0):
            raise ValueError("a <= 0")
        u = self._u()
        return 1.0 - (1.0 - u) ** (1.0 / a)

    def rayleigh(self, scale=1.0):
        if scale < 0.0 or _is_neg_zero(scale):
            raise ValueError("scale < 0")
        if scale == 0.0:
            return 0.0
        u = self._u()
        return scale * math.sqrt(-2.0 * math.log1p(-u))

    def pareto(self, a):
        if not (a > 0.0):
            raise ValueError("a <= 0")
        u = self._u()
        return (1.0 - u) ** (-1.0 / a)

    # ------------------------------------------------------------------
    # f / t / 非中心族（经 gamma/normal 组合）
    # ------------------------------------------------------------------
    def f(self, dfnum, dfden):
        if not (dfnum > 0.0) or not (dfden > 0.0):
            raise ValueError("dfnum/dfden must be > 0")
        return (self._gamma_mt(dfnum / 2.0) / (dfnum / 2.0)
                / (self._gamma_mt(dfden / 2.0) / (dfden / 2.0)))

    def standard_t(self, df):
        if not (df > 0.0):
            raise ValueError("df must be > 0")
        return self._n() / math.sqrt(self._gamma_mt(df / 2.0) / df)

    def t(self, df):
        return self.standard_t(df)

    def noncentral_chisquare(self, df, nc):
        if not (df > 0.0) or not (nc >= 0.0):
            raise ValueError("df must be > 0, nc >= 0")
        if nc == 0:
            return self.chisquare(df)
        # (Z0 + sqrt(nc))^2 + chi2(df-1)，df>=1
        z0 = self._n() + math.sqrt(nc)
        if df <= 1:
            return z0 * z0
        return z0 * z0 + self.chisquare(df - 1.0)

    def noncentral_f(self, dfnum, dfden, nc):
        if not (dfnum > 0.0) or not (dfden > 0.0):
            raise ValueError("dfnum/dfden must be > 0")
        return (self.noncentral_chisquare(dfnum, nc) / dfnum
                / (self._gamma_mt(dfden / 2.0) / dfden))

    def beta(self, a, b):
        if not (a > 0.0) or not (b > 0.0):
            raise ValueError("a/b must be > 0")
        ga = self._gamma_mt(a)
        gb = self._gamma_mt(b)
        s = ga + gb
        if s == 0.0:
            # 极小 shape 下两 gamma 均下溢为 0 → 避免 0/0 NaN
            return 0.5
        return ga / s

    def lognormal(self, mean=0.0, sigma=1.0):
        if sigma < 0.0 or _is_neg_zero(sigma):
            raise ValueError("sigma < 0")
        if sigma == 0.0:
            return math.exp(mean)
        return math.exp(self.normal(mean, sigma))

    def triangular(self, left=0.0, mode=0.5, right=1.0):
        if not (left < right):
            raise ValueError("need left < right")
        if not (left <= mode <= right):
            raise ValueError("need left <= mode <= right")
        u = self._u()
        c = (mode - left) / (right - left)
        if u < c:
            return left + math.sqrt(u * (right - left) * (mode - left))
        return right - math.sqrt((1.0 - u) * (right - left) * (right - mode))

    def triangle(self, left=0.0, mode=0.5, right=1.0):
        return self.triangular(left, mode, right)

    def vonmises(self, kappa):
        # 拒绝采样（Balaji-Neeman）；结果落在 [-pi, pi]
        k = abs(kappa)
        while True:
            u1 = self._u()
            z = math.cos(math.pi * u1)
            if self._u() < math.exp(k * (z - 1.0)):
                break
        angle = math.acos(max(-1.0, min(1.0, z)))
        if self._u() < 0.5:
            angle = -angle
        return angle

    def wald(self, mean, gamma):
        """inverse Gaussian（Wichura AS 205 变体）。"""
        if not (mean > 0.0) or not (gamma > 0.0):
            raise ValueError("mean/gamma must be > 0")
        while True:
            u1 = self._u()
            u2 = self._u()
            h = 2.0 * gamma / u1 - mean * mean
            if h > 0.0:
                x = mean + mean * mean * (h - math.sqrt(2.0 * gamma * mean * h)) \
                    / (2.0 * gamma)
            else:
                x = mean - mean * mean * (h + math.sqrt(-2.0 * gamma * mean * h)) \
                    / (2.0 * gamma)
            if x > 0.0 and u2 * (x - mean) <= x:
                return x

    # ------------------------------------------------------------------
    # 离散分布
    # ------------------------------------------------------------------
    def geometric(self, p):
        if not (0.0 < p <= 1.0):
            raise ValueError("p must be in (0, 1]")
        u = self._u()
        # 首次成功所需试验数（>=1）
        return max(1, math.ceil(math.log1p(-u) / math.log1p(-p)))

    def binomial(self, n, p):
        n = int(n)
        if n < 0:
            raise ValueError("n must be >= 0")
        if not (0.0 <= p <= 1.0) or _is_neg_zero(p):
            raise ValueError("p must be in [0, 1]")
        if n == 0 or p == 0.0:
            return 0
        if p == 1.0:
            return n
        s = 0
        for _ in range(n):
            if self._u() < p:
                s += 1
        return s

    def negative_binomial(self, n, p):
        n = int(n)
        if n < 0:
            raise ValueError("n must be >= 0")
        if not (0.0 < p <= 1.0):
            raise ValueError("p must be in (0, 1]")
        if p == 1.0:
            return 0
        lp = math.log1p(-p)
        s = 0
        for _ in range(n):
            u = self._u()
            s += max(0, math.floor(math.log1p(-u) / lp))
        return s

    def poisson(self, lam):
        if not (0.0 <= lam <= _POISSON_LAM_MAX):
            raise ValueError("lam out of range")
        if lam == 0:
            return 0
        if lam < 30.0:
            L = math.exp(-lam)
            k = 0
            pp = 1.0
            while True:
                pp *= self._u()
                k += 1
                if pp <= L:
                    return k - 1
        # 大 lam：正态近似
        return max(0, int(round(self.normal(lam, math.sqrt(lam)))))

    def hypergeometric(self, ngood, nbad, nsample):
        ngood = int(ngood)
        nbad = int(nbad)
        nsample = int(nsample)
        if ngood < 0 or nbad < 0 or nsample < 0:
            raise ValueError("parameters must be >= 0")
        total = ngood + nbad
        if nsample > total:
            raise ValueError("nsample must be <= ngood + nbad")
        if nsample == 0:
            return 0
        if ngood == 0:
            return 0
        if nbad == 0 or nsample == total:
            return ngood
        # 大参数：正态近似（快）；小参数：逐次无放回（精确）
        if total > 100000:
            mean = nsample * ngood / total
            var = mean * (nbad / total) * (total - nsample) / (total - 1)
            if var <= 0:
                return max(0, min(nsample, int(round(mean))))
            x = self.normal(mean, math.sqrt(var))
            return max(0, min(nsample, int(round(x))))
        good, bad, tot = ngood, nbad, total
        cnt = 0
        for _ in range(nsample):
            if self._u() < good / tot:
                cnt += 1
                good -= 1
            else:
                bad -= 1
            tot -= 1
        return cnt

    def zipf(self, a, zeta=None):
        if not (a > 1.0):
            raise ValueError("a must be > 1")
        if zeta is None:
            zeta = _zeta(a)
        u = self._u()
        target = u * zeta
        c = 0.0
        k = 1
        while True:
            c += k ** (-a)
            if c >= target:
                return k
            k += 1
            if k > 10 ** 7:
                return k

    def logseries(self, p):
        if not (0.0 < p < 1.0):
            raise ValueError("p must be in (0, 1)")
        u = self._u()
        target = u * (-math.log1p(-p))
        c = 0.0
        k = 1
        pk = p
        while True:
            c += pk / k
            if c >= target:
                return k
            k += 1
            pk *= p
            if k > 10 ** 7:
                return k

    def dirichlet(self, alpha):
        """alpha: 1-d 序列；返回 list（和为 1）。"""
        for x in alpha:
            if not (x > 0.0):
                raise ValueError("alpha <= 0")
        gs = [self._gamma_mt(float(x)) for x in alpha]
        s = sum(gs)
        return [g / s for g in gs]
