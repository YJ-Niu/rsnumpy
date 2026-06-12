use ndarray::{Array, IxDyn};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyTuple;

use crate::NdArray;

static RNG: std::sync::Mutex<Option<::rand::rngs::StdRng>> = std::sync::Mutex::new(None);

fn get_rng() -> std::sync::MutexGuard<'static, Option<::rand::rngs::StdRng>> {
    RNG.lock().unwrap()
}

/// 设置随机种子
#[pyfunction]
fn seed(val: u64) {
    let mut guard = get_rng();
    *guard = Some(<::rand::rngs::StdRng as ::rand::SeedableRng>::seed_from_u64(val));
}

fn ensure_rng() -> ::rand::rngs::StdRng {
    let mut guard = get_rng();
    if guard.is_none() {
        *guard = Some(<::rand::rngs::StdRng as ::rand::SeedableRng>::from_entropy());
    }
    guard.as_ref().unwrap().clone()
}

fn parse_shape_from_args(args: &Bound<'_, PyTuple>) -> Vec<usize> {
    let mut shape = Vec::new();
    for item in args.iter() {
        if let Ok(val) = item.extract::<usize>() {
            shape.push(val);
        }
    }
    shape
}

fn parse_size_arg(size: Option<&Bound<'_, PyAny>>) -> PyResult<Vec<usize>> {
    match size {
        None => Ok(vec![]),
        Some(s) => {
            if let Ok(val) = s.extract::<usize>() {
                Ok(vec![val])
            } else if let Ok(tup) = s.cast::<PyTuple>() {
                let mut result = Vec::new();
                for item in tup.iter() {
                    result.push(item.extract::<usize>()?);
                }
                Ok(result)
            } else if let Ok(list) = s.cast::<pyo3::types::PyList>() {
                let mut result = Vec::new();
                for item in list.iter() {
                    result.push(item.extract::<usize>()?);
                }
                Ok(result)
            } else {
                Err(PyValueError::new_err("size must be an integer or tuple"))
            }
        }
    }
}

/// 均匀分布 [0, 1) 随机数
#[pyfunction]
#[pyo3(signature = (*args), name = "rand")]
fn random_rand(args: &Bound<'_, PyTuple>) -> PyResult<NdArray> {
    let shape = parse_shape_from_args(args);
    if shape.is_empty() {
        let mut rng = ensure_rng();
        return Ok(NdArray {
            data: Array::from_elem(IxDyn(&[]), ::rand::Rng::r#gen(&mut rng)),
        });
    }
    let mut rng = ensure_rng();
    let total: usize = shape.iter().product();
    let values: Vec<f64> = (0..total).map(|_| ::rand::Rng::r#gen(&mut rng)).collect();
    let arr = Array::from_shape_vec(IxDyn(&shape), values)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(NdArray { data: arr })
}

/// 标准正态分布随机数
#[pyfunction]
#[pyo3(signature = (*args))]
fn randn(args: &Bound<'_, PyTuple>) -> PyResult<NdArray> {
    let shape = parse_shape_from_args(args);
    if shape.is_empty() {
        let mut rng = ensure_rng();
        let normal = ::rand_distr::Normal::new(0.0, 1.0)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        return Ok(NdArray {
            data: Array::from_elem(IxDyn(&[]), ::rand::distributions::Distribution::sample(&normal, &mut rng)),
        });
    }
    let mut rng = ensure_rng();
    let normal = ::rand_distr::Normal::new(0.0, 1.0)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let total: usize = shape.iter().product();
    let values: Vec<f64> = (0..total).map(|_| ::rand::distributions::Distribution::sample(&normal, &mut rng)).collect();
    let arr = Array::from_shape_vec(IxDyn(&shape), values)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(NdArray { data: arr })
}

/// 随机整数
#[pyfunction]
#[pyo3(signature = (low, high, size=None), name = "randint")]
fn random_randint(low: i64, high: i64, size: Option<&Bound<'_, PyAny>>) -> PyResult<NdArray> {
    let shape = parse_size_arg(size)?;
    let mut rng = ensure_rng();
    let dist = ::rand_distr::Uniform::new(low, high);
    if shape.is_empty() {
        return Ok(NdArray {
            data: Array::from_elem(IxDyn(&[]), ::rand::distributions::Distribution::sample(&dist, &mut rng) as f64),
        });
    }
    let total: usize = shape.iter().product();
    let values: Vec<f64> = (0..total).map(|_| ::rand::distributions::Distribution::sample(&dist, &mut rng) as f64).collect();
    let arr = Array::from_shape_vec(IxDyn(&shape), values)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(NdArray { data: arr })
}

/// 均匀分布随机数
#[pyfunction]
#[pyo3(signature = (low=0.0, high=1.0, size=None))]
fn uniform(low: f64, high: f64, size: Option<&Bound<'_, PyAny>>) -> PyResult<NdArray> {
    let shape = parse_size_arg(size)?;
    let mut rng = ensure_rng();
    let dist = ::rand_distr::Uniform::new(low, high);
    if shape.is_empty() {
        return Ok(NdArray {
            data: Array::from_elem(IxDyn(&[]), ::rand::distributions::Distribution::sample(&dist, &mut rng)),
        });
    }
    let total: usize = shape.iter().product();
    let values: Vec<f64> = (0..total).map(|_| ::rand::distributions::Distribution::sample(&dist, &mut rng)).collect();
    let arr = Array::from_shape_vec(IxDyn(&shape), values)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(NdArray { data: arr })
}

pub fn init_module(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(seed, m)?)?;
    m.add_function(wrap_pyfunction!(random_rand, m)?)?;
    m.add_function(wrap_pyfunction!(randn, m)?)?;
    m.add_function(wrap_pyfunction!(random_randint, m)?)?;
    m.add_function(wrap_pyfunction!(uniform, m)?)?;
    Ok(())
}