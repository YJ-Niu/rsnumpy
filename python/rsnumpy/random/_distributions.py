# SPDX-License-Identifier: MIT
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


def _loggam(x):
    """numpy random_loggam（SPECFUN 级数）。用于 poisson PTRS / 超几何 HRUA。"""
    a = (8.333333333333333e-02, -2.777777777777778e-03,
         7.936507936507937e-04, -5.952380952380952e-04,
         8.417508417508418e-04, -1.917526917526918e-03,
         6.410256410256410e-03, -2.955065359477124e-02,
         1.796443723688307e-01, -1.39243221690590e+00)
    if (x == 1.0) or (x == 2.0):
        return 0.0
    n = int(7 - x) if x < 7.0 else 0
    x0 = x + n
    x2 = (1.0 / x0) * (1.0 / x0)
    lg2pi = 1.8378770664093453e+00
    gl0 = a[9]
    for k in range(8, -1, -1):
        gl0 *= x2
        gl0 += a[k]
    gl = gl0 / x0 + 0.5 * lg2pi + (x0 - 0.5) * math.log(x0) - x0
    if x < 7.0:
        for _k in range(1, n + 1):
            gl -= math.log(x0 - 1.0)
            x0 -= 1.0
    return gl


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
        # binomial 缓存（numpy binomial_t，跨调用持久）
        self._bin_has = 0
        self._bin_nsave = None
        self._bin_psave = None
        self._bin_q = 0.0
        self._bin_r = 0.0
        self._bin_c = 0.0
        self._bin_m = 0
        self._bin_fm = 0.0
        self._bin_p1 = 0.0
        self._bin_xm = 0.0
        self._bin_xl = 0.0
        self._bin_xr = 0.0
        self._bin_laml = 0.0
        self._bin_lamr = 0.0
        self._bin_p2 = 0.0
        self._bin_p3 = 0.0
        self._bin_p4 = 0.0

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
        if not math.isfinite(high - low):
            raise OverflowError("uniform range overflows float64")
        return low + (high - low) * self._u()

    def standard_exponential(self):
        # numpy legacy: -log(1-U)，U in [0,1)
        u = self._u()
        return -math.log(1.0 - u)

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
        """numpy legacy standard_gamma：shape==1 走指数；shape<1 Ahrens-Dieter；
        shape>=1 Marsaglia-Tsang（含 gauss 缓存）。"""
        if alpha == 1.0:
            return self.standard_exponential()
        if alpha == 0.0:
            return 0.0
        if alpha < 1.0:
            while True:
                u = self._u()
                v = self.standard_exponential()
                if u <= 1.0 - alpha:
                    x = u ** (1.0 / alpha)
                    if x <= v:
                        return x
                else:
                    y = -math.log((1.0 - u) / alpha)
                    x = (1.0 - alpha + alpha * y) ** (1.0 / alpha)
                    if x <= (v + y):
                        return x
        b = alpha - 1.0 / 3.0
        c = 1.0 / math.sqrt(9.0 * b)
        while True:
            while True:
                x = self._n()
                v = 1.0 + c * x
                if v > 0.0:
                    break
            v = v * v * v
            u = self._u()
            if u < 1.0 - 0.0331 * (x * x) * (x * x):
                return b * v
            if math.log(u) < 0.5 * x * x + b * (1.0 - v + math.log(v)):
                return b * v

    def standard_cauchy(self):
        # numpy legacy: gauss / gauss（比值，不是 tan 逆 CDF）
        return self._n() / self._n()

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
            return loc + scale * math.log(2.0 * u)
        return loc - scale * math.log(2.0 * (1.0 - u))

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
        # numpy legacy: pow(standard_exponential, 1/a)
        return self.standard_exponential() ** (1.0 / a)

    def power(self, a):
        if not (a > 0.0):
            raise ValueError("a <= 0")
        # numpy legacy: pow(1 - exp(-standard_exponential), 1/a)
        return (1.0 - math.exp(-self.standard_exponential())) ** (1.0 / a)

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
        # numpy legacy: exp(standard_exponential / a) - 1
        return math.exp(self.standard_exponential() / a) - 1.0

    # ------------------------------------------------------------------
    # f / t / 非中心族（经 gamma/normal 组合）
    # ------------------------------------------------------------------
    def f(self, dfnum, dfden):
        if not (dfnum > 0.0) or not (dfden > 0.0):
            raise ValueError("dfnum/dfden must be > 0")
        # numpy legacy: (chi2(dfnum)*dfden) / (chi2(dfden)*dfnum)
        return ((self.chisquare(dfnum) * dfden)
                / (self.chisquare(dfden) * dfnum))

    def standard_t(self, df):
        if not (df > 0.0):
            raise ValueError("df must be > 0")
        # numpy legacy: sqrt(df/2) * gauss / sqrt(gamma(df/2))
        num = self._n()
        denom = self._gamma_mt(df / 2.0)
        return math.sqrt(df / 2.0) * num / math.sqrt(denom)

    def t(self, df):
        return self.standard_t(df)

    def noncentral_chisquare(self, df, nc):
        if not (df > 0.0) or not (nc >= 0.0):
            raise ValueError("df must be > 0, nc >= 0")
        if nc == 0:
            return self.chisquare(df)
        if 1 < df:
            # numpy legacy: chisquare 先生成，再 gauss
            chi2 = self.chisquare(df - 1.0)
            n = self._n() + math.sqrt(nc)
            return chi2 + n * n
        i = self.poisson(nc / 2.0)
        return self.chisquare(df + 2.0 * i)

    def noncentral_f(self, dfnum, dfden, nc):
        if not (dfnum > 0.0) or not (dfden > 0.0):
            raise ValueError("dfnum/dfden must be > 0")
        # numpy legacy: t / (chisquare(dfden)*dfnum)
        t = self.noncentral_chisquare(dfnum, nc) * dfden
        return t / (self.chisquare(dfden) * dfnum)

    def beta(self, a, b):
        if not (a > 0.0) or not (b > 0.0):
            raise ValueError("a/b must be > 0")
        if (a <= 1.0) and (b <= 1.0):
            # numpy legacy: Johnk's algorithm (a,b <= 1)
            while True:
                u = self._u()
                v = self._u()
                x = u ** (1.0 / a)
                y = v ** (1.0 / b)
                if (x + y) <= 1.0:
                    if (x + y) > 0:
                        return x / (x + y)
                    else:
                        logx = math.log(u) / a
                        logy = math.log(v) / b
                        logm = logx if logx > logy else logy
                        logx -= logm
                        logy -= logm
                        return math.exp(
                            logx - math.log(math.exp(logx) + math.exp(logy)))
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

    def vonmises(self, mu, kappa):
        # numpy legacy_vonmises（legacy-distributions.c L646-702）逐位对齐。
        # mu 在采样内部加入，随后 wrap 到 [-pi, pi]（保号 fmod）。
        if math.isnan(kappa):
            return math.nan
        if kappa < 1e-8:
            # numpy 特例：近均匀，不应用 mu/wrap
            return math.pi * (2.0 * self._u() - 1.0)
        if kappa < 1e-5:
            # kappa=0 的二阶泰勒展开（double 精度下 rho 在 1.4e-8 前恒为 0）
            s = 1.0 / kappa + kappa
        else:
            # 1e-5 <= kappa <= 1e6
            r = 1.0 + math.sqrt(1.0 + 4.0 * kappa * kappa)
            rho = (r - math.sqrt(2.0 * r)) / (2.0 * kappa)
            s = (1.0 + rho * rho) / (2.0 * rho)
        while True:
            U = self._u()
            Z = math.cos(math.pi * U)
            W = (1.0 + s * Z) / (s + Z)
            Y = kappa * (s - W)
            V = self._u()
            # Y>=0 恒接受、Y<0 恒拒绝（V==0.0 安全）
            if (Y * (2.0 - Y) - V) >= 0.0 or (math.log(Y / V) + 1.0 - Y) >= 0.0:
                break
        U = self._u()
        result = math.acos(W)
        if U < 0.5:
            result = -result
        result += mu
        neg = result < 0
        mod = abs(result)
        mod = math.fmod(mod + math.pi, 2.0 * math.pi) - math.pi
        if neg:
            mod = -mod
        return mod

    def wald(self, mean, scale):
        """inverse Gaussian（numpy legacy，Wichura 1977）。"""
        if not (mean > 0.0) or not (scale > 0.0):
            raise ValueError("mean/scale must be > 0")
        mu_2l = mean / (2.0 * scale)
        y = self._n()
        y = mean * y * y
        x = mean + mu_2l * (y - math.sqrt(4.0 * scale * y + y * y))
        u = self._u()
        if u <= mean / (mean + x):
            return x
        else:
            return mean * mean / x

    # ------------------------------------------------------------------
    # 离散分布
    # ------------------------------------------------------------------
    def geometric(self, p):
        if not (0.0 < p <= 1.0):
            raise ValueError("p must be in (0, 1]")
        if p >= 0.333333333333333333333333:
            # numpy random_geometric_search
            x = 1
            s = p
            prod = p
            q = 1.0 - p
            u = self._u()
            while u > s:
                prod *= q
                s += prod
                x += 1
            return x
        # numpy legacy_geometric_inversion
        return int(math.ceil(math.log1p(-self._u()) / math.log(1.0 - p)))

    def binomial(self, n, p):
        n = int(n)
        if n < 0:
            raise ValueError("n must be >= 0")
        if not (0.0 <= p <= 1.0) or _is_neg_zero(p):
            raise ValueError("p must be in [0, 1]")
        if n == 0 or p == 0.0:
            # numpy legacy 流语义：退化参数也消耗 1 个 double（gh-14522 回归测试固定）
            self._u()
            return 0
        if p == 1.0:
            self._u()
            return n
        # numpy legacy: p<=0.5 用 p，否则对称用 q=1-p
        if p <= 0.5:
            if p * n <= 30.0:
                return self._bin_inversion(n, p)
            return self._bin_btpe(n, p)
        q = 1.0 - p
        if q * n <= 30.0:
            return n - self._bin_inversion(n, q)
        return n - self._bin_btpe(n, q)

    def _bin_inversion(self, n, p):
        # numpy legacy_random_binomial_inversion（有状态缓存）
        if (not self._bin_has) or (self._bin_nsave != n) \
                or (self._bin_psave != p):
            self._bin_nsave = n
            self._bin_psave = p
            self._bin_has = 1
            self._bin_q = q = 1.0 - p
            self._bin_r = qn = math.exp(n * math.log(q))
            self._bin_c = np_ = n * p
            self._bin_m = bound = min(n, int(np_ + 10.0 * math.sqrt(np_ * q + 1)))
        else:
            q = self._bin_q
            qn = self._bin_r
            np_ = self._bin_c
            bound = self._bin_m
        x = 0
        px = qn
        u = self._u()
        while u > px:
            x += 1
            if x > bound:
                x = 0
                px = qn
                u = self._u()
            else:
                u -= px
                px = ((n - x + 1) * p * px) / (x * q)
        return x

    def _bin_btpe(self, n, p):
        # numpy legacy_random_binomial_btpe（有状态缓存；legacy 用 13680. 且末两项加号）
        if (not self._bin_has) or (self._bin_nsave != n) \
                or (self._bin_psave != p):
            self._bin_nsave = n
            self._bin_psave = p
            self._bin_has = 1
            self._bin_r = r = min(p, 1.0 - p)
            self._bin_q = q = 1.0 - r
            self._bin_fm = fm = n * r + r
            self._bin_m = m = int(math.floor(fm))
            self._bin_p1 = p1 = math.floor(2.195 * math.sqrt(n * r * q) - 4.6 * q) + 0.5
            self._bin_xm = xm = m + 0.5
            self._bin_xl = xl = xm - p1
            self._bin_xr = xr = xm + p1
            self._bin_c = c = 0.134 + 20.5 / (15.3 + m)
            a = (fm - xl) / (fm - xl * r)
            self._bin_laml = laml = a * (1.0 + a / 2.0)
            a = (xr - fm) / (xr * q)
            self._bin_lamr = lamr = a * (1.0 + a / 2.0)
            self._bin_p2 = p2 = p1 * (1.0 + 2.0 * c)
            self._bin_p3 = p3 = p2 + c / laml
            self._bin_p4 = p4 = p3 + c / lamr
        else:
            r = self._bin_r
            q = self._bin_q
            fm = self._bin_fm
            m = self._bin_m
            p1 = self._bin_p1
            xm = self._bin_xm
            xl = self._bin_xl
            xr = self._bin_xr
            c = self._bin_c
            laml = self._bin_laml
            lamr = self._bin_lamr
            p2 = self._bin_p2
            p3 = self._bin_p3
            p4 = self._bin_p4
        while True:
            # Step10
            nrq = n * r * q
            u = self._u() * p4
            v = self._u()
            if u <= p1:
                y = int(math.floor(xm - p1 * v + u))
                # Step60
                if p > 0.5:
                    y = n - y
                return y
            # u > p1 → Step20
            if u <= p2:
                x = xl + (u - p1) / c
                v = v * c + 1.0 - abs(m - x + 0.5) / p1
                if v > 1.0:
                    continue  # Step10
                y = int(math.floor(x))
            else:
                # u > p2 → Step30
                if u <= p3:
                    y = int(math.floor(xl + math.log(v) / laml))
                    if (y < 0) or (v == 0.0):
                        continue  # Step10
                    v = v * (u - p2) * laml
                else:
                    # u > p3 → Step40
                    y = int(math.floor(xr - math.log(v) / lamr))
                    if (y > n) or (v == 0.0):
                        continue  # Step10
                    v = v * (u - p3) * lamr
            # Step50
            k = abs(y - m)
            if (k > 20) and (k < (nrq / 2.0 - 1)):
                # Step52
                rho = (k / nrq) * ((k * (k / 3.0 + 0.625)
                                    + 0.16666666666666666) / nrq + 0.5)
                t = -k * k / (2 * nrq)
                A = math.log(v)
                if A < (t - rho):
                    if p > 0.5:
                        y = n - y
                    return y
                if A > (t + rho):
                    continue  # Step10
                x1 = y + 1.0
                f1 = m + 1.0
                z = n + 1.0 - m
                w = n - y + 1.0
                x2 = x1 * x1
                f2 = f1 * f1
                z2 = z * z
                w2 = w * w
                if A > (xm * math.log(f1 / x1)
                        + (n - m + 0.5) * math.log(z / w)
                        + (y - m) * math.log(w * r / (x1 * q))
                        + (13680. - (462. - (132. - (99. - 140. / f2) / f2) / f2) / f2)
                        / f1 / 166320.
                        + (13680. - (462. - (132. - (99. - 140. / z2) / z2) / z2) / z2)
                        / z / 166320.
                        + (13680. - (462. - (132. - (99. - 140. / x2) / x2) / x2) / x2)
                        / x1 / 166320.
                        + (13680. - (462. - (132. - (99. - 140. / w2) / w2) / w2) / w2)
                        / w / 166320.):
                    continue  # Step10
                if p > 0.5:
                    y = n - y
                return y
            else:
                s = r / q
                a = s * (n + 1)
                F = 1.0
                if m < y:
                    for i in range(m + 1, y + 1):
                        F *= (a / i - s)
                elif m > y:
                    for i in range(y + 1, m + 1):
                        F /= (a / i - s)
                if v > F:
                    continue  # Step10
                if p > 0.5:
                    y = n - y
                return y

    def negative_binomial(self, n, p):
        if n < 0:
            raise ValueError("n must be >= 0")
        if not (0.0 < p <= 1.0):
            raise ValueError("p must be in (0, 1]")
        # numpy legacy: poisson(gamma(n, (1-p)/p))
        y = self.gamma(n, (1.0 - p) / p)
        return self.poisson(y)

    def poisson(self, lam):
        if not (0.0 <= lam <= _POISSON_LAM_MAX):
            raise ValueError("lam out of range")
        if lam == 0:
            return 0
        # numpy legacy: lam>=10 用 PTRS，否则 Knuth 乘法
        if lam >= 10:
            return self._poisson_ptrs(lam)
        enlam = math.exp(-lam)
        x = 0
        prod = 1.0
        while True:
            u = self._u()
            prod *= u
            if prod > enlam:
                x += 1
            else:
                return x

    def _poisson_ptrs(self, lam):
        # numpy random_poisson_ptrs（Hoermann transformed rejection）
        slam = math.sqrt(lam)
        loglam = math.log(lam)
        b = 0.931 + 2.53 * slam
        a = -0.059 + 0.02483 * b
        invalpha = 1.1239 + 1.1328 / (b - 3.4)
        vr = 0.9277 - 3.6224 / (b - 2)
        while True:
            u = self._u() - 0.5
            v = self._u()
            us = 0.5 - abs(u)
            k = int(math.floor((2 * a / us + b) * u + lam + 0.43))
            if (us >= 0.07) and (v <= vr):
                return k
            if (k < 0) or ((us < 0.013) and (v > us)):
                continue
            if (math.log(v) + math.log(invalpha) - math.log(a / (us * us) + b)
                    <= (-lam + k * loglam - _loggam(k + 1))):
                return k

    def hypergeometric(self, ngood, nbad, nsample):
        ngood = int(ngood)
        nbad = int(nbad)
        nsample = int(nsample)
        # numpy legacy 验证
        if ngood + nbad < nsample:
            raise ValueError("ngood + nbad < nsample")
        if nsample < 1:
            raise ValueError("nsample < 1 or nsample is NaN")
        # numpy legacy: sample>10 → HRUA, sample>0 → HYP, else 0
        if nsample > 10:
            return self._hgeo_hrua(ngood, nbad, nsample)
        elif nsample > 0:
            return self._hgeo_hyp(ngood, nbad, nsample)
        else:
            return 0

    def _hgeo_hyp(self, good, bad, sample):
        # numpy random_hypergeometric_hyp
        d1 = bad + good - sample
        d2 = min(bad, good)
        y = float(d2)
        k = sample
        while y > 0.0:
            u = self._u()
            y -= math.floor(u + y / (d1 + k))
            k -= 1
            if k == 0:
                break
        z = int(d2 - y)
        if good > bad:
            z = sample - z
        return z

    def _hgeo_hrua(self, good, bad, sample):
        # numpy random_hypergeometric_hrua
        mingoodbad = min(good, bad)
        popsize = good + bad
        maxgoodbad = max(good, bad)
        m = min(sample, popsize - sample)
        d4 = mingoodbad / popsize
        d5 = 1.0 - d4
        d6 = m * d4 + 0.5
        d7 = math.sqrt((popsize - m) * sample * d4 * d5 / (popsize - 1) + 0.5)
        d8 = 1.7155277699214135 * d7 + 0.8989161620588988
        d9 = int(math.floor((m + 1) * (mingoodbad + 1) / (popsize + 2)))
        d10 = (_loggam(d9 + 1) + _loggam(mingoodbad - d9 + 1)
               + _loggam(m - d9 + 1) + _loggam(maxgoodbad - m + d9 + 1))
        d11 = min(min(m, mingoodbad) + 1.0, math.floor(d6 + 16 * d7))
        while True:
            x = self._u()
            y = self._u()
            w = d6 + d8 * (y - 0.5) / x
            if (w < 0.0) or (w >= d11):
                continue
            z = int(math.floor(w))
            t = d10 - (_loggam(z + 1) + _loggam(mingoodbad - z + 1)
                       + _loggam(m - z + 1) + _loggam(maxgoodbad - m + z + 1))
            if (x * (4.0 - x) - 3.0) <= t:
                break
            if x * (x - t) >= 1:
                continue
            if 2.0 * math.log(x) <= t:
                break
        if good > bad:
            z = m - z
        if m < sample:
            z = good - z
        return z

    def zipf(self, a, zeta=None):
        if not (a > 1.0):
            raise ValueError("a must be > 1")
        # numpy legacy: rejection (deville)
        am1 = a - 1.0
        b = 2.0 ** am1
        while True:
            u = 1.0 - self._u()
            v = self._u()
            x = math.floor(u ** (-1.0 / am1))
            if x > 9223372036854775807 or x < 1:
                continue
            t = (1.0 + 1.0 / x) ** am1
            if v * x * (t - 1.0) / (b - 1.0) <= t / b:
                return int(x)

    def logseries(self, p):
        if not (0.0 < p < 1.0):
            raise ValueError("p must be in (0, 1)")
        # numpy legacy
        r = math.log(1.0 - p)
        while True:
            v = self._u()
            if v >= p:
                return 1
            u = self._u()
            q = 1.0 - math.exp(r * u)
            if v <= q * q:
                result = math.floor(1 + math.log(v) / math.log(q))
                if (result < 1) or (v == 0.0):
                    continue
                else:
                    return int(result)
            if v >= q:
                return 1
            return 2

    def dirichlet(self, alpha):
        """alpha: 1-d 序列；返回 list（和为 1）。"""
        for x in alpha:
            if not (x > 0.0):
                raise ValueError("alpha <= 0")
        gs = [self._gamma_mt(float(x)) for x in alpha]
        s = sum(gs)
        return [g / s for g in gs]
