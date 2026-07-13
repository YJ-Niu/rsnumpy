use crate::*;

#[pyfunction]
fn all(a: &NdArray) -> bool {
    a.all()
}

#[pyfunction]
fn any(a: &NdArray) -> bool {
    a.any()
}

#[pyfunction]
fn isnan(x: &NdArray) -> NdArray {
    NdArray {
        imag: None,
        data: x.data.mapv(|v| if v.is_nan() { 1.0 } else { 0.0 }),
    }
}

#[pyfunction]
fn isinf(x: &NdArray) -> NdArray {
    NdArray {
        imag: None,
        data: x.data.mapv(|v| if v.is_infinite() { 1.0 } else { 0.0 }),
    }
}

#[pyfunction]
fn isfinite(x: &NdArray) -> NdArray {
    NdArray {
        imag: None,
        data: x.data.mapv(|v| if v.is_finite() { 1.0 } else { 0.0 }),
    }
}

#[pyfunction]
fn maximum(x1: &NdArray, x2: &NdArray) -> PyResult<NdArray> {
    let result = broadcast_binary_op(&x1.data, &x2.data, |a, b| a.max(b))?;
    Ok(NdArray {
        imag: None,
        data: result,
    })
}

#[pyfunction]
fn minimum(x1: &NdArray, x2: &NdArray) -> PyResult<NdArray> {
    let result = broadcast_binary_op(&x1.data, &x2.data, |a, b| a.min(b))?;
    Ok(NdArray {
        imag: None,
        data: result,
    })
}

#[pyfunction]
fn allclose(a: &NdArray, b: &NdArray, rtol: f64, atol: f64) -> bool {
    if a.data.len() != b.data.len() {
        return false;
    }
    for (va, vb) in a.data.iter().zip(b.data.iter()) {
        if (va - vb).abs() > atol + rtol * vb.abs() {
            return false;
        }
    }
    true
}

#[pyfunction]
fn array_equal(a: &NdArray, b: &NdArray) -> bool {
    if a.data.shape() != b.data.shape() {
        return false;
    }
    a.data.iter().zip(b.data.iter()).all(|(x, y)| x == y)
}

#[pyfunction]
fn greater(x1: &NdArray, x2: &NdArray) -> PyResult<NdArray> {
    let result = broadcast_binary_op(&x1.data, &x2.data, |a, b| if a > b { 1.0 } else { 0.0 })?;
    Ok(NdArray {
        imag: None,
        data: result,
    })
}

#[pyfunction]
fn less(x1: &NdArray, x2: &NdArray) -> PyResult<NdArray> {
    let result = broadcast_binary_op(&x1.data, &x2.data, |a, b| if a < b { 1.0 } else { 0.0 })?;
    Ok(NdArray {
        imag: None,
        data: result,
    })
}

#[pyfunction]
fn equal(x1: &NdArray, x2: &NdArray) -> PyResult<NdArray> {
    let result = broadcast_binary_op(&x1.data, &x2.data, |a, b| {
        if (a - b).abs() < 1e-12 { 1.0 } else { 0.0 }
    })?;
    Ok(NdArray {
        imag: None,
        data: result,
    })
}

#[pyfunction]
fn not_equal(x1: &NdArray, x2: &NdArray) -> PyResult<NdArray> {
    let result = broadcast_binary_op(&x1.data, &x2.data, |a, b| {
        if (a - b).abs() >= 1e-12 { 1.0 } else { 0.0 }
    })?;
    Ok(NdArray {
        imag: None,
        data: result,
    })
}

#[pyfunction]
fn greater_equal(x1: &NdArray, x2: &NdArray) -> PyResult<NdArray> {
    let result = broadcast_binary_op(&x1.data, &x2.data, |a, b| if a >= b { 1.0 } else { 0.0 })?;
    Ok(NdArray {
        imag: None,
        data: result,
    })
}

#[pyfunction]
fn less_equal(x1: &NdArray, x2: &NdArray) -> PyResult<NdArray> {
    let result = broadcast_binary_op(&x1.data, &x2.data, |a, b| if a <= b { 1.0 } else { 0.0 })?;
    Ok(NdArray {
        imag: None,
        data: result,
    })
}

#[pyfunction]
fn logical_and(x1: &NdArray, x2: &NdArray) -> PyResult<NdArray> {
    let result = broadcast_binary_op(&x1.data, &x2.data, |a, b| {
        if a != 0.0 && b != 0.0 { 1.0 } else { 0.0 }
    })?;
    Ok(NdArray {
        imag: None,
        data: result,
    })
}

#[pyfunction]
fn logical_or(x1: &NdArray, x2: &NdArray) -> PyResult<NdArray> {
    let result = broadcast_binary_op(&x1.data, &x2.data, |a, b| {
        if a != 0.0 || b != 0.0 { 1.0 } else { 0.0 }
    })?;
    Ok(NdArray {
        imag: None,
        data: result,
    })
}

#[pyfunction]
fn logical_xor(x1: &NdArray, x2: &NdArray) -> PyResult<NdArray> {
    let result = broadcast_binary_op(&x1.data, &x2.data, |a, b| {
        if (a != 0.0) != (b != 0.0) { 1.0 } else { 0.0 }
    })?;
    Ok(NdArray {
        imag: None,
        data: result,
    })
}

#[pyfunction]
#[pyo3(signature = (a, b, rtol=1e-05, atol=1e-08))]
fn isclose(a: &NdArray, b: &NdArray, rtol: f64, atol: f64) -> PyResult<NdArray> {
    let tol_atol = atol;
    let tol_rtol = rtol;
    let result = broadcast_binary_op(&a.data, &b.data, move |x, y| {
        if (x - y).abs() <= tol_atol + tol_rtol * y.abs() {
            1.0
        } else {
            0.0
        }
    })?;
    Ok(NdArray {
        imag: None,
        data: result,
    })
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(greater, m)?)?;
    m.add_function(wrap_pyfunction!(less, m)?)?;
    m.add_function(wrap_pyfunction!(equal, m)?)?;
    m.add_function(wrap_pyfunction!(not_equal, m)?)?;
    m.add_function(wrap_pyfunction!(greater_equal, m)?)?;
    m.add_function(wrap_pyfunction!(less_equal, m)?)?;
    m.add_function(wrap_pyfunction!(logical_and, m)?)?;
    m.add_function(wrap_pyfunction!(logical_or, m)?)?;
    m.add_function(wrap_pyfunction!(logical_xor, m)?)?;
    m.add_function(wrap_pyfunction!(isclose, m)?)?;
    m.add_function(wrap_pyfunction!(allclose, m)?)?;
    m.add_function(wrap_pyfunction!(array_equal, m)?)?;
    m.add_function(wrap_pyfunction!(all, m)?)?;
    m.add_function(wrap_pyfunction!(any, m)?)?;
    m.add_function(wrap_pyfunction!(isnan, m)?)?;
    m.add_function(wrap_pyfunction!(isinf, m)?)?;
    m.add_function(wrap_pyfunction!(isfinite, m)?)?;
    m.add_function(wrap_pyfunction!(maximum, m)?)?;
    m.add_function(wrap_pyfunction!(minimum, m)?)?;
    Ok(())
}
