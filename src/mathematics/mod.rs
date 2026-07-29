use crate::*;

fn unary_math_op(py: Python<'_>, x: &NdArray, threshold: usize, op: fn(f64) -> f64) -> NdArray {
    let data = &x.data;
    // 纯计算，主动释放 GIL；小数组走串行避免线程调度开销。
    let out = py.detach(|| {
        if data.len() >= threshold {
            crate::threadpool::with_pool(|| Zip::from(data).par_map_collect(|&v| op(v)))
        } else {
            data.mapv(op)
        }
    });
    NdArray {
        imag: None,
        data: out,
    }
}

/// 复数逐元素一元运算：对 (实部, 虚部) 对应用 op，返回实数数组（如 abs 求模）。
fn unary_complex_to_real_op(py: Python<'_>, x: &NdArray, op: fn(f64, f64) -> f64) -> NdArray {
    let re = &x.data;
    let zeros;
    let im = match &x.imag {
        Some(i) => i,
        None => {
            zeros = Array::zeros(re.raw_dim());
            &zeros
        }
    };
    let out = py.detach(|| Zip::from(re).and(im).map_collect(|&r, &i| op(r, i)));
    NdArray {
        imag: None,
        data: out,
    }
}

/// 复数逐元素一元运算：对 (实部, 虚部) 对应用 op，返回复数数组。
fn unary_complex_op(py: Python<'_>, x: &NdArray, op: fn(f64, f64) -> (f64, f64)) -> NdArray {
    let re = &x.data;
    let zeros;
    let im = match &x.imag {
        Some(i) => i,
        None => {
            zeros = Array::zeros(re.raw_dim());
            &zeros
        }
    };
    let (ro, io) = py.detach(|| {
        let pairs = Zip::from(re).and(im).map_collect(|&r, &i| op(r, i));
        (pairs.mapv(|p| p.0), pairs.mapv(|p| p.1))
    });
    NdArray::from_complex(ro, io)
}

// 复数逐元素超越函数（返回 (实部, 虚部)）。
#[inline]
fn c_exp(r: f64, i: f64) -> (f64, f64) {
    let e = r.exp();
    (e * i.cos(), e * i.sin())
}
#[inline]
fn c_ln(r: f64, i: f64) -> (f64, f64) {
    (0.5 * (r * r + i * i).ln(), i.atan2(r))
}
#[inline]
fn c_log10(r: f64, i: f64) -> (f64, f64) {
    let (lr, li) = c_ln(r, i);
    (lr / std::f64::consts::LN_10, li / std::f64::consts::LN_10)
}
#[inline]
fn c_sqrt(r: f64, i: f64) -> (f64, f64) {
    let m = (r * r + i * i).sqrt();
    let re = ((m + r) * 0.5).sqrt();
    let im0 = ((m - r) * 0.5).sqrt();
    (re, if i < 0.0 { -im0 } else { im0 })
}
#[inline]
fn c_sin(r: f64, i: f64) -> (f64, f64) {
    (r.sin() * i.cosh(), r.cos() * i.sinh())
}
#[inline]
fn c_cos(r: f64, i: f64) -> (f64, f64) {
    (r.cos() * i.cosh(), -(r.sin() * i.sinh()))
}
#[inline]
fn c_sinh(r: f64, i: f64) -> (f64, f64) {
    (r.sinh() * i.cos(), r.cosh() * i.sin())
}
#[inline]
fn c_cosh(r: f64, i: f64) -> (f64, f64) {
    (r.cosh() * i.cos(), r.sinh() * i.sin())
}
#[inline]
fn c_div(ar: f64, ai: f64, br: f64, bi: f64) -> (f64, f64) {
    let d = br * br + bi * bi;
    ((ar * br + ai * bi) / d, (ai * br - ar * bi) / d)
}
#[inline]
fn c_tan(r: f64, i: f64) -> (f64, f64) {
    let (sr, si) = c_sin(r, i);
    let (cr, ci) = c_cos(r, i);
    c_div(sr, si, cr, ci)
}
#[inline]
fn c_tanh(r: f64, i: f64) -> (f64, f64) {
    let (sr, si) = c_sinh(r, i);
    let (cr, ci) = c_cosh(r, i);
    c_div(sr, si, cr, ci)
}

// 计算密集（transcendental）逐元素函数：每元素工作量大，较低规模并行即可回本。
macro_rules! define_math_func {
    ($name:ident, $op:expr) => {
        #[pyfunction]
        fn $name(py: Python<'_>, x: &NdArray) -> PyResult<NdArray> {
            Ok(unary_math_op(py, x, PAR_THRESHOLD, $op))
        }
    };
}

// 复数感知的超越函数：含虚部走复数路径，否则复用实数快速路径。
macro_rules! define_math_func_c {
    ($name:ident, $rop:expr, $cop:expr) => {
        #[pyfunction]
        fn $name(py: Python<'_>, x: &NdArray) -> PyResult<NdArray> {
            if x.has_imag() {
                return Ok(unary_complex_op(py, x, $cop));
            }
            Ok(unary_math_op(py, x, PAR_THRESHOLD, $rop))
        }
    };
}

// 访存密集（sqrt/abs/floor…）逐元素函数：每元素工作量极小，
// 仅在大数组上并行才划算，故用更高阈值避免中等规模被线程开销拖慢。
macro_rules! define_cheap_math_func {
    ($name:ident, $op:expr) => {
        #[pyfunction]
        fn $name(py: Python<'_>, x: &NdArray) -> PyResult<NdArray> {
            Ok(unary_math_op(py, x, PAR_THRESHOLD_CHEAP, $op))
        }
    };
}

define_math_func_c!(sin, |v| v.sin(), c_sin);

define_math_func_c!(cos, |v| v.cos(), c_cos);

define_math_func_c!(tan, |v| v.tan(), c_tan);

// sqrt 比纯加乘略重，用中等阈值：中等规模即可从并行获益，但仍避开小数组的线程开销。
#[pyfunction]
fn sqrt(py: Python<'_>, x: &NdArray) -> PyResult<NdArray> {
    if x.has_imag() {
        return Ok(unary_complex_op(py, x, c_sqrt));
    }
    Ok(unary_math_op(py, x, PAR_THRESHOLD_MEDIUM, |v| v.sqrt()))
}

define_math_func_c!(exp, |v| v.exp(), c_exp);

define_math_func_c!(log, |v| v.ln(), c_ln);

define_math_func_c!(log10, |v| v.log10(), c_log10);

define_math_func!(log2, |v| v.log2());

define_math_func!(log1p, |v| v.ln_1p());

// abs 复数感知：含虚部时返回模 hypot(re,im)（与 numpy npy_cabs 一致），否则走实数快速路径。
#[pyfunction]
fn abs(py: Python<'_>, x: &NdArray) -> PyResult<NdArray> {
    if x.has_imag() {
        return Ok(unary_complex_to_real_op(py, x, |r, i| r.hypot(i)));
    }
    Ok(unary_math_op(py, x, PAR_THRESHOLD_CHEAP, |v| v.abs()))
}

define_math_func_c!(cosh, |v| v.cosh(), c_cosh);

define_math_func_c!(sinh, |v| v.sinh(), c_sinh);

define_math_func_c!(tanh, |v| v.tanh(), c_tanh);

define_math_func!(acosh, |v| v.acosh());

define_math_func!(asinh, |v| v.asinh());

define_math_func!(atanh, |v| v.atanh());

define_math_func!(asin, |v| v.asin());

define_math_func!(acos, |v| v.acos());

define_math_func!(atan, |v| v.atan());

define_math_func!(expm1, |v| v.exp_m1());

#[pyfunction]
fn clip(a: &NdArray, a_min: f64, a_max: f64) -> PyResult<NdArray> {
    a.clip(a_min, a_max)
}

#[pyfunction]
fn cross(a: &NdArray, b: &NdArray) -> PyResult<NdArray> {
    let a_vec: Vec<f64> = a.data.iter().copied().collect();
    let b_vec: Vec<f64> = b.data.iter().copied().collect();
    if a_vec.len() != 3 || b_vec.len() != 3 {
        return Err(PyValueError::new_err(
            "Cross product requires 3-element vectors",
        ));
    }
    let result = vec![
        a_vec[1] * b_vec[2] - a_vec[2] * b_vec[1],
        a_vec[2] * b_vec[0] - a_vec[0] * b_vec[2],
        a_vec[0] * b_vec[1] - a_vec[1] * b_vec[0],
    ];
    let arr = Array::from_shape_vec(IxDyn(&[3]), result)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(NdArray {
        imag: None,
        data: arr,
    })
}

#[pyfunction]
fn floor(py: Python<'_>, x: &NdArray) -> NdArray {
    unary_math_op(py, x, PAR_THRESHOLD_CHEAP, |v| v.floor())
}

#[pyfunction]
fn ceil(py: Python<'_>, x: &NdArray) -> NdArray {
    unary_math_op(py, x, PAR_THRESHOLD_CHEAP, |v| v.ceil())
}

#[pyfunction]
fn round(x: &NdArray, ndigits: Option<i32>) -> NdArray {
    let data = x.data.mapv(|v| match ndigits {
        Some(n) => {
            let factor = 10f64.powi(n);
            (v * factor).round() / factor
        }
        None => v.round(),
    });
    NdArray { data, imag: None }
}

define_cheap_math_func!(trunc, |v| v.trunc());

define_cheap_math_func!(fix, |v| v.trunc());

define_cheap_math_func!(square, |v| v * v);

define_math_func!(cbrt, |v| v.cbrt());

define_cheap_math_func!(sign, |v| if v > 0.0 {
    1.0
} else if v < 0.0 {
    -1.0
} else {
    0.0
});

define_cheap_math_func!(reciprocal, |v| 1.0 / v);

#[pyfunction]
fn arctan2(y: &NdArray, x: &NdArray) -> PyResult<NdArray> {
    let result = broadcast_binary_op(&y.data, &x.data, |a, b| a.atan2(b))?;
    Ok(NdArray {
        imag: None,
        data: result,
    })
}

#[pyfunction]
fn deg2rad(py: Python<'_>, x: &NdArray) -> NdArray {
    unary_math_op(py, x, PAR_THRESHOLD_CHEAP, |v| v.to_radians())
}

#[pyfunction]
fn rad2deg(py: Python<'_>, x: &NdArray) -> NdArray {
    unary_math_op(py, x, PAR_THRESHOLD_CHEAP, |v| v.to_degrees())
}

#[pyfunction]
fn hypot(x1: &NdArray, x2: &NdArray) -> PyResult<NdArray> {
    let result = broadcast_binary_op(&x1.data, &x2.data, |a, b| a.hypot(b))?;
    Ok(NdArray {
        imag: None,
        data: result,
    })
}

fn gcd_i64(a: i64, b: i64) -> i64 {
    let (mut a, mut b) = (a.abs(), b.abs());
    while b != 0 {
        let t = b;
        b = a % b;
        a = t;
    }
    a
}

#[pyfunction]
fn gcd(x1: &NdArray, x2: &NdArray) -> PyResult<NdArray> {
    let result = broadcast_binary_op(&x1.data, &x2.data, |a, b| {
        gcd_i64(a as i64, b as i64) as f64
    })?;
    Ok(NdArray {
        imag: None,
        data: result,
    })
}

#[pyfunction]
fn lcm(x1: &NdArray, x2: &NdArray) -> PyResult<NdArray> {
    let result = broadcast_binary_op(&x1.data, &x2.data, |a, b| {
        let (ai, bi) = (a as i64, b as i64);
        if ai == 0 || bi == 0 {
            0.0
        } else {
            ((ai / gcd_i64(ai, bi)) * bi).abs() as f64
        }
    })?;
    Ok(NdArray {
        imag: None,
        data: result,
    })
}

#[pyfunction]
fn nextafter(x1: &NdArray, x2: &NdArray) -> PyResult<NdArray> {
    let result = broadcast_binary_op(&x1.data, &x2.data, |a, b| {
        if a.is_nan() || b.is_nan() {
            f64::NAN
        } else if a < b {
            a.next_up()
        } else if a > b {
            a.next_down()
        } else {
            b
        }
    })?;
    Ok(NdArray {
        imag: None,
        data: result,
    })
}

#[pyfunction]
fn copysign(x1: &NdArray, x2: &NdArray) -> PyResult<NdArray> {
    let result = broadcast_binary_op(&x1.data, &x2.data, |a, b| a.copysign(b))?;
    Ok(NdArray {
        imag: None,
        data: result,
    })
}

#[pyfunction]
fn ldexp(x1: &NdArray, x2: &NdArray) -> PyResult<NdArray> {
    let result = broadcast_binary_op(&x1.data, &x2.data, |a, b| a * (2.0_f64).powi(b as i32))?;
    Ok(NdArray {
        imag: None,
        data: result,
    })
}

#[pyfunction]
fn signbit(x: &NdArray) -> NdArray {
    NdArray {
        imag: None,
        data: x
            .data
            .mapv(|v| if v.is_sign_negative() { 1.0 } else { 0.0 }),
    }
}

#[pyfunction]
fn rint(x: &NdArray) -> NdArray {
    NdArray {
        imag: None,
        data: x.data.mapv(|v| v.round_ties_even()),
    }
}

#[pyfunction]
fn spacing(x: &NdArray) -> NdArray {
    NdArray {
        imag: None,
        data: x.data.mapv(|v| v.next_up() - v),
    }
}

#[pyfunction]
fn sinc(x: &NdArray) -> NdArray {
    let pi = std::f64::consts::PI;
    NdArray {
        imag: None,
        data: x.data.mapv(|v| {
            if v == 0.0 {
                1.0
            } else {
                (pi * v).sin() / (pi * v)
            }
        }),
    }
}

#[pyfunction]
fn heaviside(x: &NdArray, h0: f64) -> NdArray {
    NdArray {
        imag: None,
        data: x.data.mapv(|v| {
            if v > 0.0 {
                1.0
            } else if v < 0.0 {
                0.0
            } else {
                h0
            }
        }),
    }
}

/// 拆分为尾数与二进制指数：v = mantissa * 2^exp，且 0.5 <= |mantissa| < 1（或 v 为 0）。
/// 语义与 C 库 frexp / numpy.frexp 一致。
fn frexp_f64(v: f64) -> (f64, i64) {
    if v == 0.0 || v.is_nan() || v.is_infinite() {
        return (v, 0);
    }
    let bits = v.to_bits();
    let raw_exp = ((bits >> 52) & 0x7ff) as i64;
    if raw_exp == 0 {
        // 次正规数：先放大到正规范围再修正指数。
        let (m, e) = frexp_f64(v * 18446744073709551616.0); // 2^64
        return (m, e - 64);
    }
    let e = raw_exp - 1022;
    // 将偏置指数改写为 1022，使尾数落入 [0.5, 1)。
    let m = f64::from_bits((bits & !(0x7ffu64 << 52)) | (1022u64 << 52));
    (m, e)
}

/// 逐元素返回 (尾数, 指数)。指数以 f64 承载（Python 侧再包装为 int64），两者均保持输入形状。
#[pyfunction]
fn frexp(py: Python<'_>, x: &NdArray) -> (NdArray, NdArray) {
    let data = &x.data;
    let (mant, expo) = py.detach(|| {
        let mut mant = Vec::with_capacity(data.len());
        let mut expo = Vec::with_capacity(data.len());
        for &v in data.iter() {
            let (m, e) = frexp_f64(v);
            mant.push(m);
            expo.push(e as f64);
        }
        let shape = data.shape();
        (
            Array::from_shape_vec(IxDyn(shape), mant).unwrap(),
            Array::from_shape_vec(IxDyn(shape), expo).unwrap(),
        )
    });
    (
        NdArray {
            imag: None,
            data: mant,
        },
        NdArray {
            imag: None,
            data: expo,
        },
    )
}

/// 第一类零阶修正贝塞尔函数 I0，级数展开（与 numpy.i0 精度一致）。
fn bessel_i0(v: f64) -> f64 {
    let mut total = 1.0_f64;
    let mut term = 1.0_f64;
    let half = v / 2.0;
    for k in 1..40 {
        let r = half / k as f64;
        term *= r * r;
        total += term;
        if term < 1e-18 * total {
            break;
        }
    }
    total
}

#[pyfunction]
fn i0(py: Python<'_>, x: &NdArray) -> NdArray {
    let data = &x.data;
    let out = py.detach(|| {
        if data.len() >= PAR_THRESHOLD {
            crate::threadpool::with_pool(|| Zip::from(data).par_map_collect(|&v| bessel_i0(v)))
        } else {
            data.mapv(bessel_i0)
        }
    });
    NdArray {
        imag: None,
        data: out,
    }
}

/// 一维线性插值：xp 必须单调递增；越界返回 left/right（默认端点值）。输出与 x 同形状。
fn interp_one(xi: f64, xp: &[f64], fp: &[f64], lo: f64, hi: f64) -> f64 {
    if xi <= xp[0] {
        return lo;
    }
    let last = xp.len() - 1;
    if xi >= xp[last] {
        return hi;
    }
    // xp[j] <= xi < xp[j+1]，用二分定位区间起点。
    let j = xp.partition_point(|&t| t <= xi) - 1;
    let (x0, x1) = (xp[j], xp[j + 1]);
    let (y0, y1) = (fp[j], fp[j + 1]);
    if x1 == x0 {
        return y0;
    }
    y0 + (y1 - y0) * (xi - x0) / (x1 - x0)
}

#[pyfunction]
#[pyo3(signature = (x, xp, fp, left=None, right=None))]
fn interp(
    py: Python<'_>,
    x: &NdArray,
    xp: &NdArray,
    fp: &NdArray,
    left: Option<f64>,
    right: Option<f64>,
) -> PyResult<NdArray> {
    let xpv: Vec<f64> = xp.data.iter().copied().collect();
    let fpv: Vec<f64> = fp.data.iter().copied().collect();
    if xpv.is_empty() || fpv.is_empty() {
        return Err(PyValueError::new_err("array of sample points is empty"));
    }
    let lo = left.unwrap_or(fpv[0]);
    let hi = right.unwrap_or(fpv[fpv.len() - 1]);
    let data = &x.data;
    let out = py.detach(|| {
        if data.len() >= PAR_THRESHOLD {
            crate::threadpool::with_pool(|| {
                Zip::from(data).par_map_collect(|&xi| interp_one(xi, &xpv, &fpv, lo, hi))
            })
        } else {
            data.mapv(|xi| interp_one(xi, &xpv, &fpv, lo, hi))
        }
    });
    Ok(NdArray {
        imag: None,
        data: out,
    })
}

/// 一维离散卷积（full），再按 mode 截取，语义与 numpy.convolve 一致。
fn convolve_modes(x: &[f64], h: &[f64], mode: &str) -> Result<Vec<f64>, String> {
    let (n, m) = (x.len(), h.len());
    if n == 0 || m == 0 {
        return Ok(vec![]);
    }
    let mut full = vec![0.0_f64; n + m - 1];
    for (i, &xi) in x.iter().enumerate() {
        for (j, &hj) in h.iter().enumerate() {
            full[i + j] += xi * hj;
        }
    }
    match mode {
        "full" => Ok(full),
        "same" => {
            let start = (m - 1) / 2;
            Ok(full[start..start + n].to_vec())
        }
        "valid" => {
            let length = n.max(m) - n.min(m) + 1;
            let start = n.min(m) - 1;
            Ok(full[start..start + length].to_vec())
        }
        other => Err(format!("unsupported mode: {other}")),
    }
}

#[pyfunction]
#[pyo3(signature = (a, v, mode="full"))]
fn convolve(py: Python<'_>, a: &NdArray, v: &NdArray, mode: &str) -> PyResult<NdArray> {
    let x: Vec<f64> = a.data.iter().copied().collect();
    let h: Vec<f64> = v.data.iter().copied().collect();
    let out = py
        .detach(|| convolve_modes(&x, &h, mode))
        .map_err(PyValueError::new_err)?;
    let arr = Array::from_shape_vec(IxDyn(&[out.len()]), out)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(NdArray {
        imag: None,
        data: arr,
    })
}

#[pyfunction]
#[pyo3(signature = (a, v, mode="valid"))]
fn correlate(py: Python<'_>, a: &NdArray, v: &NdArray, mode: &str) -> PyResult<NdArray> {
    let x: Vec<f64> = a.data.iter().copied().collect();
    let mut h: Vec<f64> = v.data.iter().copied().collect();
    h.reverse();
    let out = py
        .detach(|| convolve_modes(&x, &h, mode))
        .map_err(PyValueError::new_err)?;
    let arr = Array::from_shape_vec(IxDyn(&[out.len()]), out)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(NdArray {
        imag: None,
        data: arr,
    })
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(sin, m)?)?;
    m.add_function(wrap_pyfunction!(cos, m)?)?;
    m.add_function(wrap_pyfunction!(tan, m)?)?;
    m.add_function(wrap_pyfunction!(sqrt, m)?)?;
    m.add_function(wrap_pyfunction!(exp, m)?)?;
    m.add_function(wrap_pyfunction!(log, m)?)?;
    m.add_function(wrap_pyfunction!(log10, m)?)?;
    m.add_function(wrap_pyfunction!(log2, m)?)?;
    m.add_function(wrap_pyfunction!(log1p, m)?)?;
    m.add_function(wrap_pyfunction!(abs, m)?)?;
    m.add_function(wrap_pyfunction!(cosh, m)?)?;
    m.add_function(wrap_pyfunction!(sinh, m)?)?;
    m.add_function(wrap_pyfunction!(tanh, m)?)?;
    m.add_function(wrap_pyfunction!(acosh, m)?)?;
    m.add_function(wrap_pyfunction!(asinh, m)?)?;
    m.add_function(wrap_pyfunction!(atanh, m)?)?;
    m.add_function(wrap_pyfunction!(asin, m)?)?;
    m.add_function(wrap_pyfunction!(acos, m)?)?;
    m.add_function(wrap_pyfunction!(atan, m)?)?;
    m.add_function(wrap_pyfunction!(expm1, m)?)?;
    m.add_function(wrap_pyfunction!(trunc, m)?)?;
    m.add_function(wrap_pyfunction!(fix, m)?)?;
    m.add_function(wrap_pyfunction!(square, m)?)?;
    m.add_function(wrap_pyfunction!(cbrt, m)?)?;
    m.add_function(wrap_pyfunction!(sign, m)?)?;
    m.add_function(wrap_pyfunction!(reciprocal, m)?)?;
    m.add_function(wrap_pyfunction!(floor, m)?)?;
    m.add_function(wrap_pyfunction!(ceil, m)?)?;
    m.add_function(wrap_pyfunction!(round, m)?)?;
    m.add_function(wrap_pyfunction!(arctan2, m)?)?;
    m.add_function(wrap_pyfunction!(deg2rad, m)?)?;
    m.add_function(wrap_pyfunction!(rad2deg, m)?)?;
    m.add_function(wrap_pyfunction!(hypot, m)?)?;
    m.add_function(wrap_pyfunction!(gcd, m)?)?;
    m.add_function(wrap_pyfunction!(lcm, m)?)?;
    m.add_function(wrap_pyfunction!(nextafter, m)?)?;
    m.add_function(wrap_pyfunction!(copysign, m)?)?;
    m.add_function(wrap_pyfunction!(ldexp, m)?)?;
    m.add_function(wrap_pyfunction!(signbit, m)?)?;
    m.add_function(wrap_pyfunction!(rint, m)?)?;
    m.add_function(wrap_pyfunction!(spacing, m)?)?;
    m.add_function(wrap_pyfunction!(sinc, m)?)?;
    m.add_function(wrap_pyfunction!(heaviside, m)?)?;
    m.add_function(wrap_pyfunction!(clip, m)?)?;
    m.add_function(wrap_pyfunction!(cross, m)?)?;
    m.add_function(wrap_pyfunction!(frexp, m)?)?;
    m.add_function(wrap_pyfunction!(i0, m)?)?;
    m.add_function(wrap_pyfunction!(interp, m)?)?;
    m.add_function(wrap_pyfunction!(convolve, m)?)?;
    m.add_function(wrap_pyfunction!(correlate, m)?)?;
    m.add_function(wrap_pyfunction!(unwrap, m)?)?;
    m.add_function(wrap_pyfunction!(base_repr, m)?)?;
    m.add_function(wrap_pyfunction!(fmax, m)?)?;
    m.add_function(wrap_pyfunction!(fmin, m)?)?;
    m.add_function(wrap_pyfunction!(polymul, m)?)?;
    Ok(())
}

/// 相位解卷绕，沿指定轴
#[pyfunction]
#[pyo3(signature = (p, discont=None, axis=-1, period=6.283185307179587))]
fn unwrap(p: &NdArray, discont: Option<f64>, axis: isize, period: f64) -> NdArray {
    let shape = p.data.shape();
    let ndim = shape.len();
    if ndim == 0 {
        return NdArray {
            imag: p.imag.clone(),
            data: p.data.clone(),
        };
    }
    let ax = if axis < 0 {
        (ndim as isize + axis) as usize
    } else {
        axis as usize
    };
    let discont_val = discont.unwrap_or(period / 2.0);

    let flat: Vec<f64> = p.data.iter().copied().collect();
    let mut result = flat.clone();

    let outer_size: usize = shape[..ax].iter().product();
    let inner_size: usize = shape[ax + 1..].iter().product();
    let ax_len = shape[ax];
    let outer_strides = inner_size * ax_len;
    let inner_strides = inner_size;

    // NumPy 算法：基于原始值计算 delta，用 cumsum 累加修正
    for outer in 0..outer_size {
        for inner in 0..inner_size {
            let start = outer * outer_strides + inner;
            let mut cumulative = 0.0_f64;
            for i in 1..ax_len {
                let idx_prev = start + (i - 1) * inner_strides;
                let idx_curr = start + i * inner_strides;
                // 使用原始值计算 delta
                let delta = flat[idx_curr] - flat[idx_prev];
                // 仅在超过 discont 时修正
                if delta > discont_val {
                    cumulative -= period;
                } else if delta < -discont_val {
                    cumulative += period;
                }
                result[idx_curr] += cumulative;
            }
        }
    }

    NdArray {
        imag: None,
        data: Array::from_shape_vec(IxDyn(shape), result).expect("形状不变"),
    }
}

/// 将整数转换为给定进制的字符串
#[pyfunction]
#[pyo3(signature = (number, base=2, padding=0))]
fn base_repr(number: i64, base: u32, padding: usize) -> String {
    if !(2..=36).contains(&base) {
        return String::new();
    }
    let digits = b"0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ";
    let neg = number < 0;
    let mut num = number.unsigned_abs();
    let mut out = Vec::new();
    if num == 0 {
        out.push(b'0');
    }
    while num > 0 {
        out.push(digits[(num % base as u64) as usize]);
        num /= base as u64;
    }
    // 前置 padding 个零（reverse 前追加，等价于在最终字符串前面补零）
    out.extend(std::iter::repeat_n(b'0', padding));
    out.reverse();
    let mut s = String::from_utf8(out).unwrap_or_default();
    if neg {
        s.insert(0, '-');
    }
    s
}

/// fmax: 逐元素取较大值，NaN 视作缺失（与 NumPy 一致：仅当两者皆 NaN 才返回 NaN）。
#[pyfunction]
fn fmax(x1: &NdArray, x2: &NdArray) -> PyResult<NdArray> {
    let result = broadcast_binary_op(&x1.data, &x2.data, |a, b| {
        if a.is_nan() {
            b
        } else if b.is_nan() || a >= b {
            a
        } else {
            b
        }
    })?;
    Ok(NdArray {
        imag: None,
        data: result,
    })
}

/// fmin: 逐元素取较小值，NaN 视作缺失（与 NumPy 一致：仅当两者皆 NaN 才返回 NaN）。
#[pyfunction]
fn fmin(x1: &NdArray, x2: &NdArray) -> PyResult<NdArray> {
    let result = broadcast_binary_op(&x1.data, &x2.data, |a, b| {
        if a.is_nan() {
            b
        } else if b.is_nan() || a <= b {
            a
        } else {
            b
        }
    })?;
    Ok(NdArray {
        imag: None,
        data: result,
    })
}

/// 多项式乘法（系数按幂次降序），等价于离散卷积。
#[pyfunction]
fn polymul(a1: &NdArray, a2: &NdArray) -> PyResult<NdArray> {
    let x: Vec<f64> = a1.data.iter().copied().collect();
    let y: Vec<f64> = a2.data.iter().copied().collect();
    if x.is_empty() || y.is_empty() {
        return Ok(NdArray {
            imag: None,
            data: Array::from_shape_vec(IxDyn(&[0]), Vec::new())
                .map_err(|e| PyValueError::new_err(e.to_string()))?,
        });
    }
    let n = x.len() + y.len() - 1;
    let mut res = vec![0.0_f64; n];
    for (i, &xi) in x.iter().enumerate() {
        if xi == 0.0 {
            continue;
        }
        for (j, &yj) in y.iter().enumerate() {
            res[i + j] += xi * yj;
        }
    }
    Ok(NdArray {
        imag: None,
        data: Array::from_shape_vec(IxDyn(&[n]), res)
            .map_err(|e| PyValueError::new_err(e.to_string()))?,
    })
}
