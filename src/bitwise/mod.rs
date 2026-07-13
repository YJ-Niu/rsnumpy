use crate::*;

#[pyfunction]
fn invert(x: &NdArray) -> NdArray {
    NdArray {
        imag: None, data: x.data.mapv(|v| if v == 0.0 { 1.0 } else { 0.0 }),
    }
}

#[pyfunction]
fn bitwise_and(x1: &NdArray, x2: &NdArray) -> PyResult<NdArray> {
    let result = broadcast_binary_op(&x1.data, &x2.data, |a, b| {
        let bits = (a as i64) & (b as i64);
        bits as f64
    })?;
    Ok(NdArray { imag: None, data: result })
}

#[pyfunction]
fn bitwise_or(x1: &NdArray, x2: &NdArray) -> PyResult<NdArray> {
    let result = broadcast_binary_op(&x1.data, &x2.data, |a, b| {
        let bits = (a as i64) | (b as i64);
        bits as f64
    })?;
    Ok(NdArray { imag: None, data: result })
}

#[pyfunction]
fn bitwise_xor(x1: &NdArray, x2: &NdArray) -> PyResult<NdArray> {
    let result = broadcast_binary_op(&x1.data, &x2.data, |a, b| {
        let bits = (a as i64) ^ (b as i64);
        bits as f64
    })?;
    Ok(NdArray { imag: None, data: result })
}

#[pyfunction]
fn left_shift(x1: &NdArray, x2: &NdArray) -> PyResult<NdArray> {
    let result = broadcast_binary_op(&x1.data, &x2.data, |a, b| {
        let bits = (a as i64) << (b as i64);
        bits as f64
    })?;
    Ok(NdArray { imag: None, data: result })
}

#[pyfunction]
fn right_shift(x1: &NdArray, x2: &NdArray) -> PyResult<NdArray> {
    let result = broadcast_binary_op(&x1.data, &x2.data, |a, b| {
        let bits = (a as i64) >> (b as i64);
        bits as f64
    })?;
    Ok(NdArray { imag: None, data: result })
}

#[pyfunction]
fn bitwise_not(x: &NdArray) -> NdArray {
    NdArray {
        imag: None, data: x.data.mapv(|v| {
            let bits = !(v as i64);
            bits as f64
        }),
    }
}

#[pyfunction]
fn binary_repr(num: i64, width: Option<usize>) -> String {
    let w = if let Some(w) = width {
        w
    } else {
        let bits = 64 - num.abs().leading_zeros();
        if bits == 0 { 1 } else { bits as usize }
    };

    if num >= 0 {
        format!("{num:0w$b}")
    } else {
        let val = (1 << w) + num;
        format!("{val:0w$b}")
    }
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(invert, m)?)?;
    m.add_function(wrap_pyfunction!(bitwise_and, m)?)?;
    m.add_function(wrap_pyfunction!(bitwise_or, m)?)?;
    m.add_function(wrap_pyfunction!(bitwise_xor, m)?)?;
    m.add_function(wrap_pyfunction!(bitwise_not, m)?)?;
    m.add_function(wrap_pyfunction!(left_shift, m)?)?;
    m.add_function(wrap_pyfunction!(right_shift, m)?)?;
    m.add_function(wrap_pyfunction!(binary_repr, m)?)?;
    Ok(())
}
