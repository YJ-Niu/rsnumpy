use pyo3::prelude::*;
use pyo3::types::{PyFloat, PyList, PyTuple};
use pyo3::exceptions::PyTypeError;
use rustfft::{FftPlanner, num_complex::Complex};
use crate::NdArray;

fn fft_1d(a: &[f64]) -> Vec<(f64, f64)> {
    let n = a.len();
    if n == 0 {
        return Vec::new();
    }
    
    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_forward(n);
    
    let mut buffer: Vec<Complex<f64>> = a.iter().map(|&x| Complex::new(x, 0.0)).collect();
    fft.process(&mut buffer);
    
    buffer.into_iter().map(|c| (c.re, c.im)).collect()
}

fn ifft_1d(a: &[(f64, f64)]) -> Vec<(f64, f64)> {
    let n = a.len();
    if n == 0 {
        return Vec::new();
    }
    
    let mut planner = FftPlanner::new();
    let ifft = planner.plan_fft_inverse(n);
    
    let mut buffer: Vec<Complex<f64>> = a.iter().map(|&(re, im)| Complex::new(re, im)).collect();
    ifft.process(&mut buffer);
    
    let scale = 1.0 / n as f64;
    buffer.into_iter().map(|c| (c.re * scale, c.im * scale)).collect()
}

fn rfft_1d(a: &[f64]) -> Vec<(f64, f64)> {
    let n = a.len();
    if n == 0 {
        return Vec::new();
    }
    
    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_forward(n);
    
    let mut buffer: Vec<Complex<f64>> = a.iter().map(|&x| Complex::new(x, 0.0)).collect();
    fft.process(&mut buffer);
    
    let output_len = n / 2 + 1;
    buffer.into_iter().take(output_len).map(|c| (c.re, c.im)).collect()
}

fn irfft_1d(a: &[(f64, f64)], n: Option<usize>) -> Vec<f64> {
    let input_len = a.len();
    if input_len == 0 {
        return Vec::new();
    }
    
    let output_len = n.unwrap_or_else(|| (input_len - 1) * 2);
    
    let mut full_spectrum: Vec<Complex<f64>> = Vec::with_capacity(output_len);
    
    for i in 0..output_len {
        if i < input_len {
            let &(re, im) = &a[i];
            full_spectrum.push(Complex::new(re, im));
        } else {
            let mirror_idx = output_len - i;
            if mirror_idx < input_len {
                let &(re, im) = &a[mirror_idx];
                full_spectrum.push(Complex::new(re, -im));
            } else {
                full_spectrum.push(Complex::new(0.0, 0.0));
            }
        }
    }
    
    let mut planner = FftPlanner::new();
    let ifft = planner.plan_fft_inverse(output_len);
    
    let mut buffer = full_spectrum;
    ifft.process(&mut buffer);
    
    let scale = 1.0 / output_len as f64;
    buffer.into_iter().map(|c| c.re * scale).collect()
}

#[pyfunction]
pub fn py_fft<'py>(py: Python<'py>, a: &Bound<'_, PyAny>) -> PyResult<Bound<'py, PyList>> {
    let mut vec: Vec<f64> = Vec::new();
    if let Ok(list) = a.cast::<PyList>() {
        for item in list.iter() {
            vec.push(item.extract::<f64>()?);
        }
    } else if let Ok(val) = a.extract::<f64>() {
        vec.push(val);
    } else {
        return Err(PyTypeError::new_err("Expected list or scalar"));
    }
    
    let result = fft_1d(&vec);
    let py_list = PyList::empty(py);
    for (re, im) in result {
        let re_py = pyo3::IntoPyObject::into_pyobject(re, py)?;
        let im_py = pyo3::IntoPyObject::into_pyobject(im, py)?;
        let tuple = PyTuple::new(py, &[re_py.into_any(), im_py.into_any()])?;
        py_list.append(tuple)?;
    }
    Ok(py_list)
}

#[pyfunction]
pub fn py_ifft<'py>(py: Python<'py>, a: &Bound<'_, PyAny>) -> PyResult<Bound<'py, PyList>> {
    let mut vec: Vec<(f64, f64)> = Vec::new();
    if let Ok(list) = a.cast::<PyList>() {
        for item in list.iter() {
            if let Ok(tup) = item.cast::<PyTuple>() {
                if tup.len() == 2 {
                    let re = tup.get_item(0)?.extract::<f64>()?;
                    let im = tup.get_item(1)?.extract::<f64>()?;
                    vec.push((re, im));
                }
            }
        }
    } else {
        return Err(PyTypeError::new_err("Expected list of tuples"));
    }
    
    let result = ifft_1d(&vec);
    let py_list = PyList::empty(py);
    for (re, im) in result {
        let re_py = pyo3::IntoPyObject::into_pyobject(re, py)?;
        let im_py = pyo3::IntoPyObject::into_pyobject(im, py)?;
        let tuple = PyTuple::new(py, &[re_py.into_any(), im_py.into_any()])?;
        py_list.append(tuple)?;
    }
    Ok(py_list)
}

#[pyfunction]
pub fn py_rfft<'py>(py: Python<'py>, a: &Bound<'_, PyAny>) -> PyResult<Bound<'py, PyList>> {
    let mut vec: Vec<f64> = Vec::new();
    if let Ok(list) = a.cast::<PyList>() {
        for item in list.iter() {
            vec.push(item.extract::<f64>()?);
        }
    } else if let Ok(val) = a.extract::<f64>() {
        vec.push(val);
    } else {
        return Err(PyTypeError::new_err("Expected list or scalar"));
    }
    
    let result = rfft_1d(&vec);
    let py_list = PyList::empty(py);
    for (re, im) in result {
        let re_py = pyo3::IntoPyObject::into_pyobject(re, py)?;
        let im_py = pyo3::IntoPyObject::into_pyobject(im, py)?;
        let tuple = PyTuple::new(py, &[re_py.into_any(), im_py.into_any()])?;
        py_list.append(tuple)?;
    }
    Ok(py_list)
}

#[pyfunction]
pub fn py_irfft<'py>(py: Python<'py>, a: &Bound<'_, PyAny>, n: Option<usize>) -> PyResult<Bound<'py, PyList>> {
    let mut vec: Vec<(f64, f64)> = Vec::new();
    if let Ok(list) = a.cast::<PyList>() {
        for item in list.iter() {
            if let Ok(tup) = item.cast::<PyTuple>() {
                if tup.len() == 2 {
                    let re = tup.get_item(0)?.extract::<f64>()?;
                    let im = tup.get_item(1)?.extract::<f64>()?;
                    vec.push((re, im));
                }
            }
        }
    } else {
        return Err(PyTypeError::new_err("Expected list of tuples"));
    }
    
    let result = irfft_1d(&vec, n);
    let py_list = PyList::empty(py);
    for val in result {
        py_list.append(val)?;
    }
    Ok(py_list)
}

// ===== NdArray 版本：直接从 NdArray 取数据，避免 Python 层 .tolist() 循环 =====

#[pyfunction]
pub fn py_fft_ndarray<'py>(py: Python<'py>, a: &NdArray) -> PyResult<Bound<'py, PyList>> {
    let vec: Vec<f64> = a.data.iter().copied().collect();
    let result = fft_1d(&vec);
    let py_list = PyList::empty(py);
    for (re, im) in result {
        let re_py = PyFloat::new(py, re).into_any();
        let im_py = PyFloat::new(py, im).into_any();
        let tuple = PyTuple::new(py, &[re_py, im_py]).map_err(|e| -> PyErr { e.into() })?;
        py_list.append(tuple).map_err(|e| -> PyErr { e.into() })?;
    }
    Ok(py_list)
}

#[pyfunction]
pub fn py_ifft_ndarray<'py>(py: Python<'py>, a: &NdArray) -> PyResult<Bound<'py, PyList>> {
    let flat_data: Vec<f64> = a.data.iter().copied().collect();
    // NdArray 存储为交错格式 (real0, imag0, real1, imag1, ...)
    let mut vec = Vec::with_capacity(flat_data.len() / 2);
    for chunk in flat_data.chunks(2) {
        if chunk.len() == 2 {
            vec.push((chunk[0], chunk[1]));
        }
    }
    let result = ifft_1d(&vec);
    let py_list = PyList::empty(py);
    for (re, im) in result {
        let re_py = PyFloat::new(py, re).into_any();
        let im_py = PyFloat::new(py, im).into_any();
        let tuple = PyTuple::new(py, &[re_py, im_py]).map_err(|e| -> PyErr { e.into() })?;
        py_list.append(tuple).map_err(|e| -> PyErr { e.into() })?;
    }
    Ok(py_list)
}

#[pyfunction]
pub fn py_rfft_ndarray<'py>(py: Python<'py>, a: &NdArray) -> PyResult<Bound<'py, PyList>> {
    let vec: Vec<f64> = a.data.iter().copied().collect();
    let result = rfft_1d(&vec);
    let py_list = PyList::empty(py);
    for (re, im) in result {
        let re_py = PyFloat::new(py, re).into_any();
        let im_py = PyFloat::new(py, im).into_any();
        let tuple = PyTuple::new(py, &[re_py, im_py]).map_err(|e| -> PyErr { e.into() })?;
        py_list.append(tuple).map_err(|e| -> PyErr { e.into() })?;
    }
    Ok(py_list)
}

#[pyfunction]
pub fn py_irfft_ndarray<'py>(py: Python<'py>, a: &NdArray, n: Option<usize>) -> PyResult<Bound<'py, PyList>> {
    let flat_data: Vec<f64> = a.data.iter().copied().collect();
    let mut vec = Vec::with_capacity(flat_data.len() / 2);
    for chunk in flat_data.chunks(2) {
        if chunk.len() == 2 {
            vec.push((chunk[0], chunk[1]));
        }
    }
    let result = irfft_1d(&vec, n);
    let py_list = PyList::empty(py);
    for val in result {
        py_list.append(PyFloat::new(py, val)).map_err(|e| -> PyErr { e.into() })?;
    }
    Ok(py_list)
}
