use crate::*;

fn unary_math_op(x: &NdArray, op: fn(f64) -> f64) -> NdArray {
    let data = x.data.clone();
    let result_vec: Vec<f64> = data.into_par_iter().map(|v| op(*v)).collect();
    NdArray {
        data: Array::from_shape_vec(IxDyn(x.data.shape()), result_vec)
            .unwrap_or_else(|_| x.data.mapv(op)),
    }
}

macro_rules! define_math_func {
    ($name:ident, $op:expr) => {
        #[pyfunction]
        fn $name(x: &NdArray) -> PyResult<NdArray> {
            Ok(unary_math_op(x, $op))
        }
    };
}

define_math_func!(sin, |v| v.sin());

define_math_func!(cos, |v| v.cos());

define_math_func!(tan, |v| v.tan());

define_math_func!(sqrt, |v| v.sqrt());

define_math_func!(exp, |v| v.exp());

define_math_func!(log, |v| v.ln());

define_math_func!(log10, |v| v.log10());

define_math_func!(log2, |v| v.log2());

define_math_func!(log1p, |v| v.ln_1p());

define_math_func!(abs, |v| v.abs());

define_math_func!(cosh, |v| v.cosh());

define_math_func!(sinh, |v| v.sinh());

define_math_func!(tanh, |v| v.tanh());

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
    Ok(NdArray { data: arr })
}

#[pyfunction]
fn floor(x: &NdArray) -> NdArray {
    unary_math_op(x, |v| v.floor())
}

#[pyfunction]
fn ceil(x: &NdArray) -> NdArray {
    unary_math_op(x, |v| v.ceil())
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
    NdArray { data }
}

define_math_func!(trunc, |v| v.trunc());

define_math_func!(fix, |v| v.trunc());

define_math_func!(square, |v| v * v);

define_math_func!(cbrt, |v| v.cbrt());

define_math_func!(sign, |v| if v > 0.0 {
    1.0
} else if v < 0.0 {
    -1.0
} else {
    0.0
});

define_math_func!(reciprocal, |v| 1.0 / v);

#[pyfunction]
fn arctan2(y: &NdArray, x: &NdArray) -> PyResult<NdArray> {
    let result = broadcast_binary_op(&y.data, &x.data, |a, b| a.atan2(b))?;
    Ok(NdArray { data: result })
}

#[pyfunction]
fn deg2rad(x: &NdArray) -> NdArray {
    unary_math_op(x, |v| v.to_radians())
}

#[pyfunction]
fn rad2deg(x: &NdArray) -> NdArray {
    unary_math_op(x, |v| v.to_degrees())
}

#[pyfunction]
fn hypot(x1: &NdArray, x2: &NdArray) -> PyResult<NdArray> {
    let result = broadcast_binary_op(&x1.data, &x2.data, |a, b| a.hypot(b))?;
    Ok(NdArray { data: result })
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
    Ok(NdArray { data: result })
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
    Ok(NdArray { data: result })
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
    Ok(NdArray { data: result })
}

#[pyfunction]
fn copysign(x1: &NdArray, x2: &NdArray) -> PyResult<NdArray> {
    let result = broadcast_binary_op(&x1.data, &x2.data, |a, b| a.copysign(b))?;
    Ok(NdArray { data: result })
}

#[pyfunction]
fn ldexp(x1: &NdArray, x2: &NdArray) -> PyResult<NdArray> {
    let result = broadcast_binary_op(&x1.data, &x2.data, |a, b| a * (2.0_f64).powi(b as i32))?;
    Ok(NdArray { data: result })
}

#[pyfunction]
fn signbit(x: &NdArray) -> NdArray {
    NdArray {
        data: x
            .data
            .mapv(|v| if v.is_sign_negative() { 1.0 } else { 0.0 }),
    }
}

#[pyfunction]
fn rint(x: &NdArray) -> NdArray {
    NdArray {
        data: x.data.mapv(|v| v.round_ties_even()),
    }
}

#[pyfunction]
fn spacing(x: &NdArray) -> NdArray {
    NdArray {
        data: x.data.mapv(|v| v.next_up() - v),
    }
}

#[pyfunction]
fn sinc(x: &NdArray) -> NdArray {
    let pi = std::f64::consts::PI;
    NdArray {
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
    Ok(())
}
