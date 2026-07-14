use crate::*;

#[pyfunction]
fn add(x1: &NdArray, x2: &NdArray) -> PyResult<NdArray> {
    if x1.has_imag() || x2.has_imag() {
        return complex_binop_nd(x1, x2, CBinOp::Add);
    }
    let result = broadcast_binary_op(&x1.data, &x2.data, |a, b| a + b)?;
    Ok(NdArray::from_real(result))
}

#[pyfunction]
fn subtract(x1: &NdArray, x2: &NdArray) -> PyResult<NdArray> {
    if x1.has_imag() || x2.has_imag() {
        return complex_binop_nd(x1, x2, CBinOp::Sub);
    }
    let result = broadcast_binary_op(&x1.data, &x2.data, |a, b| a - b)?;
    Ok(NdArray::from_real(result))
}

#[pyfunction]
fn multiply(x1: &NdArray, x2: &NdArray) -> PyResult<NdArray> {
    if x1.has_imag() || x2.has_imag() {
        return complex_binop_nd(x1, x2, CBinOp::Mul);
    }
    let result = broadcast_binary_op(&x1.data, &x2.data, |a, b| a * b)?;
    Ok(NdArray::from_real(result))
}

#[pyfunction]
fn divide(x1: &NdArray, x2: &NdArray) -> PyResult<NdArray> {
    if x1.has_imag() || x2.has_imag() {
        return complex_binop_nd(x1, x2, CBinOp::Div);
    }
    let result = broadcast_binary_op(&x1.data, &x2.data, |a, b| a / b)?;
    Ok(NdArray::from_real(result))
}

#[pyfunction]
fn power(x1: &NdArray, x2: &NdArray) -> PyResult<NdArray> {
    let result = broadcast_binary_op(&x1.data, &x2.data, |a, b| a.powf(b))?;
    Ok(NdArray::from_real(result))
}

#[pyfunction]
fn mod_(x1: &NdArray, x2: &NdArray) -> PyResult<NdArray> {
    let result = broadcast_binary_op(&x1.data, &x2.data, |a, b| a % b)?;
    Ok(NdArray::from_real(result))
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(add, m)?)?;
    m.add_function(wrap_pyfunction!(subtract, m)?)?;
    m.add_function(wrap_pyfunction!(multiply, m)?)?;
    m.add_function(wrap_pyfunction!(divide, m)?)?;
    m.add_function(wrap_pyfunction!(power, m)?)?;
    m.add_function(wrap_pyfunction!(mod_, m)?)?;
    Ok(())
}
