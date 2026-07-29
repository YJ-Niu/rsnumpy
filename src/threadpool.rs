//! 线程池控制模块 - 真正的 rayon 线程池管理
//!
//! 【设计要点】
//! - 维护一个全局 Arc<rayon::ThreadPool>，由 set_num_threads / parallel_context 动态重建
//! - 提供 with_pool!(<closure>) 宏，所有并行计算统一走 pool.install(closure)
//! - 这样 rayon 的 par_iter / par_map_collect 等都会在我们指定线程数的池上执行
//! - rayon 自带 work-stealing，无需手动调度；小数组走串行分支，避免调度开销
//!
//! 【为什么不用 rayon::current_thread_pool / build_global】
//! - build_global 只能调用一次，无法动态改线程数
//! - 线程数改了必须重建 ThreadPool，所以用 OnceLock<Arc<...>> 加 RwLock 包装
//! - RwLock 读多写少（设置线程数是低频操作），性能没问题

use pyo3::Bound;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use rayon::{ThreadPool, ThreadPoolBuilder};
use std::sync::Arc;
use std::sync::RwLock;
use std::sync::atomic::{AtomicUsize, Ordering};

/// 获取逻辑 CPU 核心数（优先使用标准库，失败时回退到 1）
fn num_cpus_get() -> usize {
    std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1)
}

/// 全局 rayon 线程池（通过 RwLock 包装，支持动态重建）
static GLOBAL_POOL: RwLock<Option<Arc<ThreadPool>>> = RwLock::new(None);

/// 当前线程数（用于 Python 层查询，保持与 ThreadPool 实际线程数一致）
static CURRENT_NUM_THREADS: AtomicUsize = AtomicUsize::new(0);

/// 内部获取线程数（默认为 CPU 核心数）
fn default_num_threads() -> usize {
    if let Ok(threads_str) = std::env::var("RSNUMPY_NUM_THREADS")
        && let Ok(threads) = threads_str.parse::<usize>()
        && threads > 0
    {
        return threads.min(num_cpus_get());
    }
    num_cpus_get()
}

/// 确保全局线程池已初始化（如未初始化则用 default_num_threads 构建）
fn ensure_pool() -> Arc<ThreadPool> {
    // 先读锁快速路径
    {
        let guard = GLOBAL_POOL.read().expect("GLOBAL_POOL poisoned");
        if let Some(pool) = guard.as_ref() {
            return pool.clone();
        }
    }
    // 未初始化 -> 写锁创建
    let mut guard = GLOBAL_POOL.write().expect("GLOBAL_POOL poisoned");
    // 双重检查：另一个线程可能已经创建过了
    if guard.is_none() {
        let n = default_num_threads();
        let pool = ThreadPoolBuilder::new()
            .num_threads(n)
            .thread_name(|i| format!("rsnumpy-wk-{}", i))
            .build()
            .expect("Failed to create rayon thread pool");
        CURRENT_NUM_THREADS.store(n, Ordering::SeqCst);
        *guard = Some(Arc::new(pool));
    }
    guard.as_ref().unwrap().clone()
}

/// 重建全局线程池（set_num_threads / parallel_context 内部使用）
fn rebuild_pool(num_threads: usize) -> Arc<ThreadPool> {
    let n = num_threads.max(1).min(num_cpus_get());
    let pool = ThreadPoolBuilder::new()
        .num_threads(n)
        .thread_name(|i| format!("rsnumpy-wk-{}", i))
        .build()
        .expect("Failed to create rayon thread pool");
    let arc_pool = Arc::new(pool);
    let mut guard = GLOBAL_POOL.write().expect("GLOBAL_POOL poisoned");
    *guard = Some(arc_pool.clone());
    CURRENT_NUM_THREADS.store(n, Ordering::SeqCst);
    arc_pool
}

/// 对外导出：在当前 rayon 线程池上执行闭包并返回结果
/// 所有模块的并行计算统一调用这个函数。
pub fn with_pool<F, R>(f: F) -> R
where
    F: FnOnce() -> R + Send,
    R: Send,
{
    let pool = ensure_pool();
    pool.install(f)
}

// ========== Python 层接口 ==========

/// 设置线程数（Python 接口）
///
/// 【参数】
/// - `num_threads`: 线程数，0 表示自动检测（CPU 核心数）
///
/// 【使用示例】
/// ```python
/// import rsnumpy as np
/// np.set_num_threads(8)          # 设为 8 线程
/// np.set_num_threads(0)          # 恢复自动检测
/// print(np.get_num_threads())    # 查看当前线程数
/// ```
#[pyfunction]
#[pyo3(signature = (num_threads=0))]
pub fn set_num_threads(num_threads: usize) -> PyResult<()> {
    let actual = if num_threads == 0 {
        default_num_threads()
    } else {
        num_threads.max(1).min(num_cpus_get())
    };
    rebuild_pool(actual);
    Ok(())
}

/// 获取当前线程数（Python 接口）
#[pyfunction]
pub fn get_num_threads() -> usize {
    let n = CURRENT_NUM_THREADS.load(Ordering::SeqCst);
    if n == 0 {
        let _ = ensure_pool();
        CURRENT_NUM_THREADS.load(Ordering::SeqCst)
    } else {
        n
    }
}

/// 获取 CPU 核心数（Python 接口）
#[pyfunction]
pub fn get_num_cpus() -> usize {
    num_cpus_get()
}

/// 获取并行阈值（Python 接口）
/// - cheap: 访存密集型（加/减/乘/除等）阈值
/// - medium: 中等代价（sqrt/reciprocal 等）阈值
/// - expensive: 计算密集型（sin/exp/log 等）阈值
#[pyfunction]
pub fn get_parallel_thresholds(py: Python<'_>) -> PyResult<Bound<'_, PyDict>> {
    let dict = PyDict::new(py);
    dict.set_item("cheap", crate::par_threshold_cheap())?;
    dict.set_item("medium", crate::par_threshold_medium())?;
    dict.set_item("expensive", crate::par_threshold())?;
    Ok(dict)
}

/// 设置并行阈值（Python 接口）
///
/// 【参数】
/// - `cheap`: 访存密集型阈值（加/减/乘/除等），默认 262144
/// - `medium`: 中等代价阈值（sqrt/reciprocal 等），默认 49152
/// - `expensive`: 计算密集型阈值（sin/exp/log 等），默认 32768
///
/// 【使用示例】
/// ```python
/// import rsnumpy as np
/// # 降低阈值，让更小的数组也走并行
/// np.set_parallel_thresholds(expensive=1024, medium=2048, cheap=8192)
/// # 恢复默认值
/// np.set_parallel_thresholds(expensive=32768, medium=49152, cheap=262144)
/// ```
#[pyfunction]
#[pyo3(signature = (cheap=None, medium=None, expensive=None))]
pub fn set_parallel_thresholds(
    cheap: Option<usize>,
    medium: Option<usize>,
    expensive: Option<usize>,
) -> PyResult<()> {
    if let Some(v) = cheap {
        if v < 1 {
            return Err(PyValueError::new_err("cheap 阈值必须 >= 1"));
        }
        crate::PAR_THRESHOLD_CHEAP.store(v, Ordering::Relaxed);
    }
    if let Some(v) = medium {
        if v < 1 {
            return Err(PyValueError::new_err("medium 阈值必须 >= 1"));
        }
        crate::PAR_THRESHOLD_MEDIUM.store(v, Ordering::Relaxed);
    }
    if let Some(v) = expensive {
        if v < 1 {
            return Err(PyValueError::new_err("expensive 阈值必须 >= 1"));
        }
        crate::PAR_THRESHOLD.store(v, Ordering::Relaxed);
    }
    Ok(())
}

/// 并行上下文管理器（Python 接口）
///
/// 【使用示例】
/// ```python
/// import rsnumpy as np
/// with np.parallel_context(num_threads=4):
///     result = np.dot(large_matrix1, large_matrix2)
/// # 自动恢复原线程数
/// ```
#[pyclass(name = "parallel_context")]
pub struct ParallelContext {
    num_threads: usize,
    old_pool: Option<Arc<ThreadPool>>,
    old_num_threads: usize,
}

#[pymethods]
impl ParallelContext {
    #[new]
    #[pyo3(signature = (num_threads=0))]
    fn new(num_threads: usize) -> Self {
        let _ = ensure_pool(); // 确保已有池存在，避免 old_num_threads=0
        let old_num_threads = CURRENT_NUM_THREADS.load(Ordering::SeqCst);
        let old_pool = {
            GLOBAL_POOL
                .read()
                .expect("GLOBAL_POOL poisoned")
                .as_ref()
                .cloned()
        };
        let actual = if num_threads == 0 {
            default_num_threads()
        } else {
            num_threads.max(1).min(num_cpus_get())
        };
        let _new_pool = rebuild_pool(actual);
        ParallelContext {
            num_threads: actual,
            old_pool,
            old_num_threads,
        }
    }

    fn __enter__(&self) -> PyResult<usize> {
        Ok(self.num_threads)
    }

    fn __exit__(
        &mut self,
        _exc_type: &Bound<'_, PyAny>,
        _exc_val: &Bound<'_, PyAny>,
        _exc_tb: &Bound<'_, PyAny>,
    ) -> PyResult<bool> {
        // 恢复旧的线程池（直接写回，而不是用旧的线程数重建，避免额外分配）
        if let Some(old) = self.old_pool.take() {
            let mut guard = GLOBAL_POOL.write().expect("GLOBAL_POOL poisoned");
            *guard = Some(old);
            CURRENT_NUM_THREADS.store(self.old_num_threads, Ordering::SeqCst);
        }
        Ok(false)
    }
}

pub fn register_module(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(set_num_threads, m)?)?;
    m.add_function(wrap_pyfunction!(get_num_threads, m)?)?;
    m.add_function(wrap_pyfunction!(get_num_cpus, m)?)?;
    m.add_function(wrap_pyfunction!(get_parallel_thresholds, m)?)?;
    m.add_function(wrap_pyfunction!(set_parallel_thresholds, m)?)?;
    m.add_class::<ParallelContext>()?;
    Ok(())
}
