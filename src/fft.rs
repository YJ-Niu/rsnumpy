use ndarray::{Array, IxDyn};
use pyo3::prelude::*;
use rustfft::{FftPlanner, num_complex::Complex as FftComplex};
use std::cell::RefCell;

use crate::NdArray;

thread_local! {
    static FFT_PLANNER: RefCell<FftPlanner<f64>> = RefCell::new(FftPlanner::new());
    static IFFT_PLANNER: RefCell<FftPlanner<f64>> = RefCell::new(FftPlanner::new());
}

fn fft_1d(a: &[f64]) -> Vec<FftComplex<f64>> {
    let n = a.len();
    if n == 0 {
        return Vec::new();
    }

    let mut buffer: Vec<FftComplex<f64>> = Vec::with_capacity(n);
    buffer.extend(a.iter().map(|&x| FftComplex::new(x, 0.0)));

    FFT_PLANNER.with(|planner| {
        let fft = planner.borrow_mut().plan_fft_forward(n);
        fft.process(&mut buffer);
    });

    buffer
}

fn ifft_1d(a: &[FftComplex<f64>]) -> Vec<FftComplex<f64>> {
    let n = a.len();
    if n == 0 {
        return Vec::new();
    }

    let mut buffer = a.to_vec();

    IFFT_PLANNER.with(|planner| {
        let ifft = planner.borrow_mut().plan_fft_inverse(n);
        ifft.process(&mut buffer);
    });

    let scale = 1.0 / n as f64;
    buffer.iter_mut().for_each(|c| {
        *c = FftComplex::new(c.re * scale, c.im * scale);
    });

    buffer
}

fn rfft_1d(a: &[f64]) -> Vec<FftComplex<f64>> {
    let n = a.len();
    if n == 0 {
        return Vec::new();
    }

    let mut buffer: Vec<FftComplex<f64>> = Vec::with_capacity(n);
    buffer.extend(a.iter().map(|&x| FftComplex::new(x, 0.0)));

    FFT_PLANNER.with(|planner| {
        let fft = planner.borrow_mut().plan_fft_forward(n);
        fft.process(&mut buffer);
    });

    let output_len = n / 2 + 1;
    buffer.truncate(output_len);
    buffer
}

fn irfft_1d(a: &[FftComplex<f64>], n: Option<usize>) -> Vec<f64> {
    let input_len = a.len();
    if input_len == 0 {
        return Vec::new();
    }

    let output_len = n.unwrap_or_else(|| (input_len - 1) * 2);

    let mut full_spectrum: Vec<FftComplex<f64>> = Vec::with_capacity(output_len);

    for i in 0..output_len {
        if i < input_len {
            full_spectrum.push(a[i]);
        } else {
            let mirror_idx = output_len - i;
            if mirror_idx < input_len {
                full_spectrum.push(FftComplex::new(a[mirror_idx].re, -a[mirror_idx].im));
            } else {
                full_spectrum.push(FftComplex::new(0.0, 0.0));
            }
        }
    }

    IFFT_PLANNER.with(|planner| {
        let ifft = planner.borrow_mut().plan_fft_inverse(output_len);
        ifft.process(&mut full_spectrum);
    });

    let scale = 1.0 / output_len as f64;
    full_spectrum.into_iter().map(|c| c.re * scale).collect()
}

fn complex_vec_to_ndarray(result: Vec<FftComplex<f64>>) -> NdArray {
    let result_len = result.len();
    let mut re_data: Vec<f64> = Vec::with_capacity(result_len);
    let mut im_data: Vec<f64> = Vec::with_capacity(result_len);
    for c in result {
        re_data.push(c.re);
        im_data.push(c.im);
    }
    let re = Array::from_shape_vec(IxDyn(&[result_len]), re_data).unwrap();
    let im = Array::from_shape_vec(IxDyn(&[result_len]), im_data).unwrap();
    NdArray::from_complex(re, im)
}

#[pyfunction]
pub fn py_fft_ndarray(_py: Python<'_>, a: &NdArray) -> PyResult<NdArray> {
    _py.detach(move || {
        let result = if a.has_imag() {
            let re_vec: Vec<f64> = a.data.iter().copied().collect();
            let im_vec: Vec<f64> = a.imag.as_ref().unwrap().iter().copied().collect();
            let complex_input: Vec<FftComplex<f64>> = re_vec
                .iter()
                .zip(im_vec.iter())
                .map(|(&r, &i)| FftComplex::new(r, i))
                .collect();
            let mut buffer = complex_input;
            FFT_PLANNER.with(|planner| {
                let fft = planner.borrow_mut().plan_fft_forward(buffer.len());
                fft.process(&mut buffer);
            });
            buffer
        } else {
            let vec: Vec<f64> = a.data.iter().copied().collect();
            fft_1d(&vec)
        };
        Ok(complex_vec_to_ndarray(result))
    })
}

#[pyfunction]
pub fn py_ifft_ndarray(_py: Python<'_>, a: &NdArray) -> PyResult<NdArray> {
    _py.detach(move || {
        let complex_vec: Vec<FftComplex<f64>> = if a.has_imag() {
            let re_vec: Vec<f64> = a.data.iter().copied().collect();
            let im_vec: Vec<f64> = a.imag.as_ref().unwrap().iter().copied().collect();
            re_vec
                .iter()
                .zip(im_vec.iter())
                .map(|(&r, &i)| FftComplex::new(r, i))
                .collect()
        } else {
            let vec: Vec<f64> = a.data.iter().copied().collect();
            vec.iter().map(|&v| FftComplex::new(v, 0.0)).collect()
        };
        let result = ifft_1d(&complex_vec);
        Ok(complex_vec_to_ndarray(result))
    })
}

#[pyfunction]
pub fn py_rfft_ndarray(_py: Python<'_>, a: &NdArray) -> PyResult<NdArray> {
    _py.detach(move || {
        let result = if a.has_imag() {
            let re_vec: Vec<f64> = a.data.iter().copied().collect();
            let im_vec: Vec<f64> = a.imag.as_ref().unwrap().iter().copied().collect();
            let complex_input: Vec<FftComplex<f64>> = re_vec
                .iter()
                .zip(im_vec.iter())
                .map(|(&r, &i)| FftComplex::new(r, i))
                .collect();
            let n = complex_input.len();
            let mut buffer = complex_input;
            FFT_PLANNER.with(|planner| {
                let fft = planner.borrow_mut().plan_fft_forward(n);
                fft.process(&mut buffer);
            });
            let output_len = n / 2 + 1;
            buffer.truncate(output_len);
            buffer
        } else {
            let vec: Vec<f64> = a.data.iter().copied().collect();
            rfft_1d(&vec)
        };
        Ok(complex_vec_to_ndarray(result))
    })
}

#[pyfunction]
pub fn py_irfft_ndarray(_py: Python<'_>, a: &NdArray, n: Option<usize>) -> PyResult<NdArray> {
    _py.detach(move || {
        let complex_vec: Vec<FftComplex<f64>> = if a.has_imag() {
            let re_vec: Vec<f64> = a.data.iter().copied().collect();
            let im_vec: Vec<f64> = a.imag.as_ref().unwrap().iter().copied().collect();
            re_vec
                .iter()
                .zip(im_vec.iter())
                .map(|(&r, &i)| FftComplex::new(r, i))
                .collect()
        } else {
            let vec: Vec<f64> = a.data.iter().copied().collect();
            vec.iter().map(|&v| FftComplex::new(v, 0.0)).collect()
        };
        let result = irfft_1d(&complex_vec, n);
        Ok(NdArray {
            imag: None,
            data: Array::from_shape_vec(IxDyn(&[result.len()]), result).unwrap(),
        })
    })
}
