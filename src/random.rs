use ndarray::{Array, IxDyn};
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyTuple;
use rand::SeedableRng;
use rand_distr::Distribution;
use rayon::prelude::*;
use std::hash::{Hash, Hasher};

use crate::NdArray;

thread_local! {
    static THREAD_RNG: std::cell::RefCell<Option<::rand::rngs::StdRng>> = std::cell::RefCell::new(None);
}

static GLOBAL_SEED: std::sync::OnceLock<u64> = std::sync::OnceLock::new();

fn with_thread_rng<F, R>(f: F) -> R
where
    F: FnOnce(&mut ::rand::rngs::StdRng) -> R,
{
    THREAD_RNG.with(|cell| {
        let mut rng = cell.borrow_mut();
        if rng.is_none() {
            let seed = GLOBAL_SEED.get().copied().unwrap_or(0xdeadbeef);
            let mut hasher = std::collections::hash_map::DefaultHasher::new();
            std::thread::current().id().hash(&mut hasher);
            let thread_seed = seed.wrapping_add(hasher.finish().wrapping_mul(0x9e3779b97f4a7c15));
            *rng = Some(::rand::rngs::StdRng::seed_from_u64(thread_seed));
        }
        f(rng.as_mut().unwrap())
    })
}

fn new_rng(seed: Option<u64>) -> ::rand::rngs::StdRng {
    ::rand::rngs::StdRng::seed_from_u64(seed.unwrap_or_else(rand::random))
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

fn make_ndarray_parallel(
    shape: &[usize],
    dist: impl rand_distr::Distribution<f64> + Sync + Send + Clone + 'static,
) -> NdArray {
    let total: usize = shape.iter().product();
    
    let mut values = Vec::with_capacity(total);
    values.resize(total, 0.0);
    
    let num_chunks = rayon::current_num_threads();
    let chunk_size = (total + num_chunks - 1) / num_chunks;
    
    let seeds: Vec<u64> = (0..num_chunks).map(|_| rand::random()).collect();
    
    values.par_chunks_mut(chunk_size).enumerate().for_each(|(i, chunk)| {
        let mut local_rng = ::rand::rngs::StdRng::seed_from_u64(seeds[i]);
        for elem in chunk {
            *elem = dist.sample(&mut local_rng);
        }
    });
    
    let arr = Array::from_shape_vec(IxDyn(shape), values).unwrap();
    NdArray { data: arr }
}

fn make_ndarray_single(rng: &mut impl rand::Rng, shape: &[usize], dist: impl rand_distr::Distribution<f64>) -> NdArray {
    if shape.is_empty() {
        return NdArray {
            data: Array::from_elem(IxDyn(&[]), dist.sample(rng)),
        };
    }
    let total: usize = shape.iter().product();
    
    let mut values = Vec::with_capacity(total);
    values.resize(total, 0.0);
    
    for elem in values.iter_mut() {
        *elem = dist.sample(rng);
    }
    
    let arr = Array::from_shape_vec(IxDyn(shape), values).unwrap();
    NdArray { data: arr }
}

#[pyfunction]
fn seed(val: u64) {
    let _ = GLOBAL_SEED.set(val);
    THREAD_RNG.with(|rng| {
        *rng.borrow_mut() = Some(::rand::rngs::StdRng::seed_from_u64(val));
    });
}

#[pyfunction]
fn get_state() -> u64 {
    GLOBAL_SEED.get().copied().unwrap_or(0xdeadbeef)
}

#[pyfunction]
#[pyo3(signature = (*args), name = "rand")]
fn random_rand(_py: Python<'_>, args: &Bound<'_, PyTuple>) -> PyResult<NdArray> {
    let shape = parse_shape_from_args(args);
    let dist = ::rand_distr::Uniform::<f64>::new(0.0, 1.0)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    
    if shape.is_empty() {
        with_thread_rng(|rng| Ok(make_ndarray_single(rng, &shape, dist)))
    } else {
        Ok(_py.detach(move || make_ndarray_parallel(&shape, dist)))
    }
}

#[pyfunction]
#[pyo3(signature = (*args))]
fn randn(_py: Python<'_>, args: &Bound<'_, PyTuple>) -> PyResult<NdArray> {
    let shape = parse_shape_from_args(args);
    let normal = ::rand_distr::Normal::new(0.0, 1.0)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    
    if shape.is_empty() {
        with_thread_rng(|rng| Ok(make_ndarray_single(rng, &shape, normal)))
    } else {
        Ok(_py.detach(move || make_ndarray_parallel(&shape, normal)))
    }
}

#[pyfunction]
#[pyo3(signature = (low, high, size=None), name = "randint")]
fn random_randint(low: i64, high: i64, size: Option<&Bound<'_, PyAny>>) -> PyResult<NdArray> {
    let shape = parse_size_arg(size)?;
    let dist = ::rand_distr::Uniform::new(low, high)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    with_thread_rng(|rng| {
        if shape.is_empty() {
            return Ok(NdArray {
                data: Array::from_elem(IxDyn(&[]), dist.sample(rng) as f64),
            });
        }
        let total: usize = shape.iter().product();
        let values: Vec<f64> = (0..total).map(|_| dist.sample(rng) as f64).collect();
        let arr = Array::from_shape_vec(IxDyn(&shape), values)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(NdArray { data: arr })
    })
}

#[pyclass(name = "Generator", from_py_object)]
#[derive(Clone)]
struct PyGenerator {
    seed: Option<u64>,
}

#[pymethods]
impl PyGenerator {
    #[new]
    fn new(seed: Option<u64>) -> Self {
        PyGenerator { seed }
    }

    #[pyo3(signature = (size=None))]
    fn random(&self, _py: Python<'_>, size: Option<&Bound<'_, PyAny>>) -> PyResult<NdArray> {
        let shape = parse_size_arg(size)?;
        let dist = ::rand_distr::Uniform::<f64>::new(0.0, 1.0)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        
        if shape.is_empty() {
            let mut rng = new_rng(self.seed);
            Ok(make_ndarray_single(&mut rng, &shape, dist))
        } else {
            Ok(_py.detach(move || make_ndarray_parallel(&shape, dist)))
        }
    }

    #[pyo3(signature = (size=None))]
    fn standard_normal(&self, _py: Python<'_>, size: Option<&Bound<'_, PyAny>>) -> PyResult<NdArray> {
        let shape = parse_size_arg(size)?;
        let normal = ::rand_distr::Normal::new(0.0, 1.0)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        
        if shape.is_empty() {
            let mut rng = new_rng(self.seed);
            Ok(make_ndarray_single(&mut rng, &shape, normal))
        } else {
            Ok(_py.detach(move || make_ndarray_parallel(&shape, normal)))
        }
    }

    #[pyo3(signature = (low, high=None, size=None, endpoint=false))]
    fn integers(&self, low: i64, high: Option<i64>, size: Option<&Bound<'_, PyAny>>, endpoint: bool) -> PyResult<NdArray> {
        let hi = high.unwrap_or(low);
        let actual_low = if high.is_some() { low } else { 0 };
        let actual_high = if endpoint { hi + 1 } else { hi };
        if actual_high <= actual_low {
            return Err(PyValueError::new_err("high must be > low"));
        }
        let shape = parse_size_arg(size)?;
        let dist = ::rand_distr::Uniform::new(actual_low, actual_high)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        let mut rng = new_rng(self.seed);
        if shape.is_empty() {
            return Ok(NdArray {
                data: Array::from_elem(IxDyn(&[]), dist.sample(&mut rng) as f64),
            });
        }
        let total: usize = shape.iter().product();
        let values: Vec<f64> = (0..total).map(|_| dist.sample(&mut rng) as f64).collect();
        let arr = Array::from_shape_vec(IxDyn(&shape), values)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(NdArray { data: arr })
    }

    #[pyo3(signature = (a, size=None, replace=true))]
    fn choice<'py>(&self, py: Python<'py>, a: &Bound<'_, PyAny>, size: Option<usize>, replace: bool) -> PyResult<Bound<'py, PyAny>> {
        let mut rng = new_rng(self.seed);
        let vals: Vec<f64> = if let Ok(arr) = a.extract::<NdArray>() {
            arr.data.iter().copied().collect()
        } else if let Ok(list) = a.cast::<pyo3::types::PyList>() {
            let mut v = Vec::new();
            for item in list.iter() {
                v.push(item.extract::<f64>()?);
            }
            v
        } else {
            return Err(PyTypeError::new_err("a must be array-like"));
        };

        if vals.is_empty() {
            return Err(PyValueError::new_err("a must be non-empty"));
        }

        let n = size.unwrap_or(1);
        if !replace && n > vals.len() {
            return Err(PyValueError::new_err("Cannot take a larger sample than population when 'replace=false'"));
        }

        let dist = ::rand_distr::Uniform::new(0usize, vals.len())
            .map_err(|e| PyValueError::new_err(e.to_string()))?;

        let result: Vec<f64> = if replace {
            (0..n).map(|_| vals[dist.sample(&mut rng)]).collect()
        } else {
            let mut indices: Vec<usize> = (0..vals.len()).collect();
            for i in (1..vals.len()).rev() {
                let j_dist = ::rand_distr::Uniform::new(0usize, i + 1)
                .map_err(|e| PyValueError::new_err(e.to_string()))?;
                let j = j_dist.sample(&mut rng);
                indices.swap(i, j);
            }
            indices.truncate(n);
            indices.into_iter().map(|i| vals[i]).collect()
        };

        if n == 1 && size.is_none() {
            return Ok(result[0].into_pyobject(py).unwrap().into_any());
        }

        let arr = Array::from_shape_vec(IxDyn(&[n]), result)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(Bound::new(py, NdArray { data: arr })?.into_any())
    }

    fn shuffle(&self, a: &Bound<'_, PyAny>) -> PyResult<()> {
        let mut rng = new_rng(self.seed);
        if let Ok(mut arr) = a.extract::<NdArray>() {
            if arr.data.ndim() != 1 {
                return Err(PyValueError::new_err("shuffle only works for 1D arrays"));
            }
            let len = arr.data.len();
            let mut result: Vec<f64> = arr.data.iter().copied().collect();
            for i in (1..len).rev() {
                let j_dist = ::rand_distr::Uniform::new(0usize, i + 1)
                    .map_err(|e| PyValueError::new_err(e.to_string()))?;
                let j = j_dist.sample(&mut rng);
                result.swap(i, j);
            }
            arr.data = Array::from_shape_vec(IxDyn(&[len]), result).unwrap();
            Ok(())
        } else {
            Err(PyTypeError::new_err("Expected ndarray"))
        }
    }

    #[pyo3(signature = (a))]
    fn permutation<'py>(&self, py: Python<'py>, a: &Bound<'_, PyAny>) -> PyResult<Bound<'py, PyAny>> {
        let mut rng = new_rng(self.seed);
        if let Ok(n) = a.extract::<usize>() {
            let mut indices: Vec<f64> = (0..n).map(|i| i as f64).collect();
            for i in (1..n).rev() {
                let j_dist = ::rand_distr::Uniform::new(0usize, i + 1)
                    .map_err(|e| PyValueError::new_err(e.to_string()))?;
                let j = j_dist.sample(&mut rng);
                indices.swap(i, j);
            }
            let arr = Array::from_shape_vec(IxDyn(&[n]), indices)
                .map_err(|e| PyValueError::new_err(e.to_string()))?;
            Ok(Bound::new(py, NdArray { data: arr })?.into_any())
        } else if let Ok(arr) = a.extract::<NdArray>() {
            let vals: Vec<f64> = arr.data.iter().copied().collect();
            let n = vals.len();
            let mut indices: Vec<usize> = (0..n).collect();
            for i in (1..n).rev() {
                let j_dist = ::rand_distr::Uniform::new(0usize, i + 1)
                    .map_err(|e| PyValueError::new_err(e.to_string()))?;
                let j = j_dist.sample(&mut rng);
                indices.swap(i, j);
            }
            let result: Vec<f64> = indices.into_iter().map(|i| vals[i]).collect();
            let result_arr = Array::from_shape_vec(arr.data.dim(), result)
                .map_err(|e| PyValueError::new_err(e.to_string()))?;
            Ok(Bound::new(py, NdArray { data: result_arr.into_dyn() })?.into_any())
        } else {
            Err(PyTypeError::new_err("Expected integer or ndarray"))
        }
    }

    #[pyo3(signature = (low=0.0, high=1.0, size=None))]
    fn uniform(&self, _py: Python<'_>, low: f64, high: f64, size: Option<&Bound<'_, PyAny>>) -> PyResult<NdArray> {
        let shape = parse_size_arg(size)?;
        let dist = ::rand_distr::Uniform::new(low, high)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        
        if shape.is_empty() {
            let mut rng = new_rng(self.seed);
            Ok(make_ndarray_single(&mut rng, &shape, dist))
        } else {
            Ok(_py.detach(move || make_ndarray_parallel(&shape, dist)))
        }
    }

    #[pyo3(signature = (loc=0.0, scale=1.0, size=None))]
    fn normal(&self, _py: Python<'_>, loc: f64, scale: f64, size: Option<&Bound<'_, PyAny>>) -> PyResult<NdArray> {
        let shape = parse_size_arg(size)?;
        let dist = ::rand_distr::Normal::new(loc, scale)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        
        if shape.is_empty() {
            let mut rng = new_rng(self.seed);
            Ok(make_ndarray_single(&mut rng, &shape, dist))
        } else {
            Ok(_py.detach(move || make_ndarray_parallel(&shape, dist)))
        }
    }

    #[pyo3(signature = (a, b, size=None))]
    fn beta(&self, _py: Python<'_>, a: f64, b: f64, size: Option<&Bound<'_, PyAny>>) -> PyResult<NdArray> {
        let shape = parse_size_arg(size)?;
        let dist = ::rand_distr::Beta::new(a, b)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        
        if shape.is_empty() {
            let mut rng = new_rng(self.seed);
            Ok(make_ndarray_single(&mut rng, &shape, dist))
        } else {
            Ok(_py.detach(move || make_ndarray_parallel(&shape, dist)))
        }
    }

    #[pyo3(signature = (shape_param, scale=1.0, size=None))]
    fn gamma(&self, _py: Python<'_>, shape_param: f64, scale: f64, size: Option<&Bound<'_, PyAny>>) -> PyResult<NdArray> {
        let shape = parse_size_arg(size)?;
        let dist = ::rand_distr::Gamma::new(shape_param, scale)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        
        if shape.is_empty() {
            let mut rng = new_rng(self.seed);
            Ok(make_ndarray_single(&mut rng, &shape, dist))
        } else {
            Ok(_py.detach(move || make_ndarray_parallel(&shape, dist)))
        }
    }

    #[pyo3(signature = (scale=1.0, size=None))]
    fn exponential(&self, _py: Python<'_>, scale: f64, size: Option<&Bound<'_, PyAny>>) -> PyResult<NdArray> {
        let shape = parse_size_arg(size)?;
        let dist = ::rand_distr::Exp::new(1.0 / scale)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        
        if shape.is_empty() {
            let mut rng = new_rng(self.seed);
            Ok(make_ndarray_single(&mut rng, &shape, dist))
        } else {
            Ok(_py.detach(move || make_ndarray_parallel(&shape, dist)))
        }
    }

    #[pyo3(signature = (n, p, size=None))]
    fn binomial(&self, n: u64, p: f64, size: Option<&Bound<'_, PyAny>>) -> PyResult<NdArray> {
        let shape = parse_size_arg(size)?;
        let dist = ::rand_distr::Binomial::new(n, p)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        let mut rng = new_rng(self.seed);
        if shape.is_empty() {
            return Ok(NdArray {
                data: Array::from_elem(IxDyn(&[]), dist.sample(&mut rng) as f64),
            });
        }
        let total: usize = shape.iter().product();
        let values: Vec<f64> = (0..total).map(|_| dist.sample(&mut rng) as f64).collect();
        let arr = Array::from_shape_vec(IxDyn(&shape), values)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(NdArray { data: arr })
    }

    #[pyo3(signature = (lam, size=None))]
    fn poisson(&self, lam: f64, size: Option<&Bound<'_, PyAny>>) -> PyResult<NdArray> {
        let shape = parse_size_arg(size)?;
        let dist = ::rand_distr::Poisson::new(lam)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        let mut rng = new_rng(self.seed);
        if shape.is_empty() {
            return Ok(NdArray {
                data: Array::from_elem(IxDyn(&[]), dist.sample(&mut rng) as f64),
            });
        }
        let total: usize = shape.iter().product();
        let values: Vec<f64> = (0..total).map(|_| dist.sample(&mut rng) as f64).collect();
        let arr = Array::from_shape_vec(IxDyn(&shape), values)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(NdArray { data: arr })
    }

    #[pyo3(signature = (a, scale=1.0, size=None))]
    fn weibull(&self, _py: Python<'_>, a: f64, scale: f64, size: Option<&Bound<'_, PyAny>>) -> PyResult<NdArray> {
        let shape = parse_size_arg(size)?;
        let dist = ::rand_distr::Weibull::new(a, scale)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        
        if shape.is_empty() {
            let mut rng = new_rng(self.seed);
            Ok(make_ndarray_single(&mut rng, &shape, dist))
        } else {
            Ok(_py.detach(move || make_ndarray_parallel(&shape, dist)))
        }
    }

    #[pyo3(signature = (loc=0.0, scale=1.0, size=None))]
    fn logistic(&self, loc: f64, scale: f64, size: Option<&Bound<'_, PyAny>>) -> PyResult<NdArray> {
        let shape = parse_size_arg(size)?;
        let uniform = ::rand_distr::Uniform::new(0.0, 1.0)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        let mut rng = new_rng(self.seed);
        if shape.is_empty() {
            let u = uniform.sample(&mut rng) as f64;
            let val = loc + scale * (u / (1.0 - u)).ln();
            return Ok(NdArray {
                data: Array::from_elem(IxDyn(&[]), val),
            });
        }
        let total: usize = shape.iter().product();
        let values: Vec<f64> = (0..total).map(|_| {
            let u = uniform.sample(&mut rng) as f64;
            loc + scale * (u / (1.0 - u)).ln()
        }).collect();
        let arr = Array::from_shape_vec(IxDyn(&shape), values)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(NdArray { data: arr })
    }

    #[pyo3(signature = (loc=0.0, scale=1.0, size=None))]
    fn cauchy(&self, _py: Python<'_>, loc: f64, scale: f64, size: Option<&Bound<'_, PyAny>>) -> PyResult<NdArray> {
        let shape = parse_size_arg(size)?;
        let dist = ::rand_distr::Cauchy::new(loc, scale)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        
        if shape.is_empty() {
            let mut rng = new_rng(self.seed);
            Ok(make_ndarray_single(&mut rng, &shape, dist))
        } else {
            Ok(_py.detach(move || make_ndarray_parallel(&shape, dist)))
        }
    }

    #[pyo3(signature = (loc=0.0, scale=1.0, size=None))]
    fn laplace(&self, loc: f64, scale: f64, size: Option<&Bound<'_, PyAny>>) -> PyResult<NdArray> {
        let shape = parse_size_arg(size)?;
        let uniform = ::rand_distr::Uniform::new(0.0, 1.0)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        let mut rng = new_rng(self.seed);
        if shape.is_empty() {
            let u: f64 = uniform.sample(&mut rng);
            let val = if u < 0.5 {
                loc + scale * (2.0 * u).ln()
            } else {
                loc - scale * (2.0 * (1.0 - u)).ln()
            };
            return Ok(NdArray {
                data: Array::from_elem(IxDyn(&[]), val),
            });
        }
        let total: usize = shape.iter().product();
        let values: Vec<f64> = (0..total).map(|_| {
            let u: f64 = uniform.sample(&mut rng);
            if u < 0.5 {
                loc + scale * (2.0 * u).ln()
            } else {
                loc - scale * (2.0 * (1.0 - u)).ln()
            }
        }).collect();
        let arr = Array::from_shape_vec(IxDyn(&shape), values)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(NdArray { data: arr })
    }
}

#[pyfunction]
#[pyo3(signature = (low=0.0, high=1.0, size=None))]
fn uniform(_py: Python<'_>, low: f64, high: f64, size: Option<&Bound<'_, PyAny>>) -> PyResult<NdArray> {
    let shape = parse_size_arg(size)?;
    let dist = ::rand_distr::Uniform::new(low, high)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    
    if shape.is_empty() {
        with_thread_rng(|rng| Ok(make_ndarray_single(rng, &shape, dist)))
    } else {
        Ok(_py.detach(move || make_ndarray_parallel(&shape, dist)))
    }
}

#[pyfunction]
#[pyo3(signature = (loc=0.0, scale=1.0, size=None))]
fn normal(_py: Python<'_>, loc: f64, scale: f64, size: Option<&Bound<'_, PyAny>>) -> PyResult<NdArray> {
    let shape = parse_size_arg(size)?;
    let dist = ::rand_distr::Normal::new(loc, scale)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    
    if shape.is_empty() {
        with_thread_rng(|rng| Ok(make_ndarray_single(rng, &shape, dist)))
    } else {
        Ok(_py.detach(move || make_ndarray_parallel(&shape, dist)))
    }
}

#[pyfunction]
#[pyo3(signature = (a, b, size=None))]
fn beta(_py: Python<'_>, a: f64, b: f64, size: Option<&Bound<'_, PyAny>>) -> PyResult<NdArray> {
    let shape = parse_size_arg(size)?;
    let dist = ::rand_distr::Beta::new(a, b)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    
    if shape.is_empty() {
        with_thread_rng(|rng| Ok(make_ndarray_single(rng, &shape, dist)))
    } else {
        Ok(_py.detach(move || make_ndarray_parallel(&shape, dist)))
    }
}

#[pyfunction]
#[pyo3(signature = (shape_param, scale=1.0, size=None))]
fn gamma(_py: Python<'_>, shape_param: f64, scale: f64, size: Option<&Bound<'_, PyAny>>) -> PyResult<NdArray> {
    let shape = parse_size_arg(size)?;
    let dist = ::rand_distr::Gamma::new(shape_param, scale)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    
    if shape.is_empty() {
        with_thread_rng(|rng| Ok(make_ndarray_single(rng, &shape, dist)))
    } else {
        Ok(_py.detach(move || make_ndarray_parallel(&shape, dist)))
    }
}

#[pyfunction]
#[pyo3(signature = (scale=1.0, size=None))]
fn exponential(_py: Python<'_>, scale: f64, size: Option<&Bound<'_, PyAny>>) -> PyResult<NdArray> {
    let shape = parse_size_arg(size)?;
    let dist = ::rand_distr::Exp::new(1.0 / scale)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    
    if shape.is_empty() {
        with_thread_rng(|rng| Ok(make_ndarray_single(rng, &shape, dist)))
    } else {
        Ok(_py.detach(move || make_ndarray_parallel(&shape, dist)))
    }
}

pub fn init_module(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(seed, m)?)?;
    m.add_function(wrap_pyfunction!(random_rand, m)?)?;
    m.add_function(wrap_pyfunction!(randn, m)?)?;
    m.add_function(wrap_pyfunction!(random_randint, m)?)?;
    m.add_function(wrap_pyfunction!(uniform, m)?)?;
    m.add_function(wrap_pyfunction!(normal, m)?)?;
    m.add_function(wrap_pyfunction!(beta, m)?)?;
    m.add_function(wrap_pyfunction!(gamma, m)?)?;
    m.add_function(wrap_pyfunction!(exponential, m)?)?;
    m.add_function(wrap_pyfunction!(get_state, m)?)?;
    m.add_class::<PyGenerator>()?;
    Ok(())
}