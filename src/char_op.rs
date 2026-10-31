//! 字符串批量算子（np.char 的 Rust 层实现）。
//!
//! 一次跨边界调用完成整列字符串运算，Python 层只保留薄包装。
//! bytes 数组由 Python 侧按 utf-8 解码/编码往返。

use crate::*;

fn strip_to(s: &str, chars: Option<&str>, left: bool, right: bool) -> String {
    match chars {
        Some(p) => {
            let mut t = s;
            if left {
                t = t.trim_start_matches(|c| p.contains(c));
            }
            if right {
                t = t.trim_end_matches(|c| p.contains(c));
            }
            t.to_string()
        }
        None => {
            let mut t = s;
            if left {
                t = t.trim_start();
            }
            if right {
                t = t.trim_end();
            }
            t.to_string()
        }
    }
}

/// numpy title 语义：非字母后的首个字母大写，其余小写
fn np_title(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut prev_alpha = false;
    for ch in s.chars() {
        if ch.is_alphabetic() {
            if prev_alpha {
                out.extend(ch.to_lowercase());
            } else {
                out.extend(ch.to_uppercase());
            }
            prev_alpha = true;
        } else {
            prev_alpha = false;
            out.push(ch);
        }
    }
    out
}

fn np_capitalize(s: &str) -> String {
    let mut chars = s.chars();
    match chars.next() {
        Some(c) => c.to_uppercase().collect::<String>() + &chars.as_str().to_lowercase(),
        None => String::new(),
    }
}

fn arg<'a>(a: Option<&'a String>, name: &str) -> PyResult<&'a str> {
    match a {
        Some(s) => Ok(s.as_str()),
        None => Err(PyValueError::new_err(format!("{} is required", name))),
    }
}

/// 字符串 → 字符串 的批量算子
#[pyfunction]
#[pyo3(signature = (op, vals, arg1=None, arg2=None, arg3=None))]
pub fn char_str_op(
    op: &str,
    vals: Vec<String>,
    arg1: Option<String>,
    arg2: Option<String>,
    arg3: Option<String>,
) -> PyResult<Vec<String>> {
    let v1 = || arg(arg1.as_ref(), "arg1");
    let v2 = || arg(arg2.as_ref(), "arg2");
    match op {
        "capitalize" => Ok(vals.iter().map(|v| np_capitalize(v)).collect()),
        "lower" => Ok(vals.iter().map(|v| v.to_lowercase()).collect()),
        "upper" => Ok(vals.iter().map(|v| v.to_uppercase()).collect()),
        "title" => Ok(vals.iter().map(|v| np_title(v)).collect()),
        "swapcase" => Ok(vals
            .iter()
            .map(|v| {
                v.chars()
                    .flat_map(|c| {
                        if c.is_uppercase() {
                            c.to_lowercase().collect::<Vec<_>>()
                        } else {
                            c.to_uppercase().collect::<Vec<_>>()
                        }
                    })
                    .collect()
            })
            .collect()),
        "strip" => {
            let c = arg1.as_deref();
            Ok(vals.iter().map(|v| strip_to(v, c, true, true)).collect())
        }
        "lstrip" => {
            let c = arg1.as_deref();
            Ok(vals.iter().map(|v| strip_to(v, c, true, false)).collect())
        }
        "rstrip" => {
            let c = arg1.as_deref();
            Ok(vals.iter().map(|v| strip_to(v, c, false, true)).collect())
        }
        "center" => {
            let w: usize = v1()?.parse().unwrap_or(0);
            let fill = arg2
                .as_deref()
                .and_then(|s| s.chars().next())
                .unwrap_or(' ');
            Ok(vals
                .iter()
                .map(|v| {
                    let n = v.chars().count();
                    if n >= w {
                        return v.clone();
                    }
                    let total = w - n;
                    let l = total / 2;
                    let r = total - l;
                    let mut out = String::with_capacity(w);
                    out.push_str(&fill.to_string().repeat(l));
                    out.push_str(v);
                    out.push_str(&fill.to_string().repeat(r));
                    out
                })
                .collect())
        }
        "ljust" => {
            let w: usize = v1()?.parse().unwrap_or(0);
            let fill = arg2
                .as_deref()
                .and_then(|s| s.chars().next())
                .unwrap_or(' ');
            Ok(vals
                .iter()
                .map(|v| {
                    let n = v.chars().count();
                    if n >= w {
                        return v.clone();
                    }
                    let mut out = String::with_capacity(w);
                    out.push_str(v);
                    out.push_str(&fill.to_string().repeat(w - n));
                    out
                })
                .collect())
        }
        "rjust" => {
            let w: usize = v1()?.parse().unwrap_or(0);
            let fill = arg2
                .as_deref()
                .and_then(|s| s.chars().next())
                .unwrap_or(' ');
            Ok(vals
                .iter()
                .map(|v| {
                    let n = v.chars().count();
                    if n >= w {
                        return v.clone();
                    }
                    let mut out = String::with_capacity(w);
                    out.push_str(&fill.to_string().repeat(w - n));
                    out.push_str(v);
                    out
                })
                .collect())
        }
        "zfill" => {
            let w: usize = v1()?.parse().unwrap_or(0);
            Ok(vals
                .iter()
                .map(|v| {
                    let n = v.chars().count();
                    if n >= w {
                        return v.clone();
                    }
                    let pad = "0".repeat(w - n);
                    if let Some(rest) = v.strip_prefix('-') {
                        format!("-{}{}", pad, rest)
                    } else {
                        format!("{}{}", pad, v)
                    }
                })
                .collect())
        }
        "replace" => {
            let old = v1()?;
            let new = v2()?;
            let count: usize = arg3
                .as_ref()
                .and_then(|s| s.parse().ok())
                .unwrap_or(usize::MAX);
            Ok(vals.iter().map(|v| v.replacen(old, new, count)).collect())
        }
        "expandtabs" => {
            let t: usize = arg1.as_ref().and_then(|s| s.parse().ok()).unwrap_or(8);
            Ok(vals
                .iter()
                .map(|v| v.replace('\t', &" ".repeat(t)))
                .collect())
        }
        "add" => {
            let other = arg(arg1.as_ref(), "arg1")?;
            // vals 与 other 等长序列的拼接由 Python 层传 join 前的成对序列；
            // 这里支持 other 为单串广播
            Ok(vals.iter().map(|v| format!("{}{}", v, other)).collect())
        }
        "mul" => {
            let n: usize = v1()?.parse().unwrap_or(0);
            Ok(vals.iter().map(|v| v.repeat(n)).collect())
        }
        "join" => {
            let sep = arg1.as_deref().unwrap_or("");
            Ok(vals.iter().map(|v| sep.to_string() + v).collect())
        }
        _ => Err(PyValueError::new_err(format!(
            "unsupported char op: {}",
            op
        ))),
    }
}

/// 字符串 → 整数 的批量算子（find/rindex/count/strlen）
#[pyfunction]
#[pyo3(signature = (op, vals, arg1=None, arg2=None, arg3=None))]
pub fn char_int_op(
    op: &str,
    vals: Vec<String>,
    arg1: Option<String>,
    arg2: Option<String>,
    arg3: Option<String>,
) -> PyResult<Vec<i64>> {
    match op {
        "find" | "index" | "rfind" | "rindex" => {
            let sub = arg(arg1.as_ref(), "arg1")?;
            let start: usize = arg2.as_ref().and_then(|s| s.parse().ok()).unwrap_or(0);
            let end: Option<usize> = arg3.as_ref().and_then(|s| s.parse().ok());
            let rev = op.starts_with('r');
            Ok(vals
                .iter()
                .map(|v| {
                    let e = end.unwrap_or(v.len());
                    if start > e || e > v.len() {
                        return -1;
                    }
                    let hay = &v[start..e];
                    let pos = if rev { hay.rfind(sub) } else { hay.find(sub) };
                    match pos {
                        Some(p) => (start + p) as i64,
                        None => -1,
                    }
                })
                .collect())
        }
        "count" => {
            let sub = arg(arg1.as_ref(), "arg1")?;
            Ok(vals.iter().map(|v| v.matches(sub).count() as i64).collect())
        }
        "strlen" => Ok(vals.iter().map(|v| v.chars().count() as i64).collect()),
        _ => Err(PyValueError::new_err(format!(
            "unsupported char int op: {}",
            op
        ))),
    }
}

/// 字符串 → 布尔 的批量算子（比较族与 is 判定族）
#[pyfunction]
#[pyo3(signature = (op, vals, arg1=None, other=None))]
pub fn char_bool_op(
    op: &str,
    vals: Vec<String>,
    arg1: Option<String>,
    other: Option<Vec<String>>,
) -> PyResult<Vec<bool>> {
    match op {
        "startswith" => {
            let p = arg(arg1.as_ref(), "arg1")?;
            Ok(vals.iter().map(|v| v.starts_with(p)).collect())
        }
        "endswith" => {
            let p = arg(arg1.as_ref(), "arg1")?;
            Ok(vals.iter().map(|v| v.ends_with(p)).collect())
        }
        "isalnum" => Ok(vals
            .iter()
            .map(|v| !v.is_empty() && v.chars().all(|c| c.is_alphanumeric()))
            .collect()),
        "isalpha" => Ok(vals
            .iter()
            .map(|v| !v.is_empty() && v.chars().all(|c| c.is_alphabetic()))
            .collect()),
        "isdigit" => Ok(vals
            .iter()
            .map(|v| !v.is_empty() && v.chars().all(|c| c.is_numeric()))
            .collect()),
        "isspace" => Ok(vals
            .iter()
            .map(|v| !v.is_empty() && v.chars().all(|c| c.is_whitespace()))
            .collect()),
        "islower" => Ok(vals
            .iter()
            .map(|v| {
                let has_c = v.chars().any(|c| c.is_alphabetic());
                has_c
                    && v.chars()
                        .filter(|c| c.is_alphabetic())
                        .all(|c| c.is_lowercase())
            })
            .collect()),
        "isupper" => Ok(vals
            .iter()
            .map(|v| {
                let has_c = v.chars().any(|c| c.is_alphabetic());
                has_c
                    && v.chars()
                        .filter(|c| c.is_alphabetic())
                        .all(|c| c.is_uppercase())
            })
            .collect()),
        "istitle" => Ok(vals
            .iter()
            .map(|v| v == &np_title(v) && v.chars().any(|c| c.is_alphabetic()))
            .collect()),
        "isdecimal" => Ok(vals
            .iter()
            .map(|v| !v.is_empty() && v.chars().all(|c| c.is_ascii_digit()))
            .collect()),
        "isnumeric" => Ok(vals
            .iter()
            .map(|v| !v.is_empty() && v.chars().all(|c| c.is_numeric()))
            .collect()),
        "equal" | "not_equal" | "greater" | "greater_equal" | "less" | "less_equal" => {
            let others =
                other.ok_or_else(|| PyValueError::new_err("comparison requires other sequence"))?;
            let lt = op == "less" || op == "less_equal";
            let eq = op.ends_with("equal");
            Ok(vals
                .iter()
                .zip(others.iter())
                .map(|(a, b)| {
                    let ord = a.cmp(b);
                    match op {
                        "equal" => a == b,
                        "not_equal" => a != b,
                        _ => {
                            if eq {
                                if lt {
                                    ord != std::cmp::Ordering::Greater
                                } else {
                                    ord != std::cmp::Ordering::Less
                                }
                            } else if lt {
                                ord == std::cmp::Ordering::Less
                            } else {
                                ord == std::cmp::Ordering::Greater
                            }
                        }
                    }
                })
                .collect())
        }
        _ => Err(PyValueError::new_err(format!(
            "unsupported char bool op: {}",
            op
        ))),
    }
}

/// 成对拼接（np.char.add 的广播语义在 Rust 层完成）
#[pyfunction]
pub fn char_add(a: Vec<String>, b: Vec<String>) -> PyResult<Vec<String>> {
    if a.len() == b.len() {
        Ok(a.iter()
            .zip(b.iter())
            .map(|(x, y)| format!("{}{}", x, y))
            .collect())
    } else if b.len() == 1 {
        Ok(a.iter().map(|x| format!("{}{}", x, b[0])).collect())
    } else if a.len() == 1 {
        Ok(b.iter().map(|y| format!("{}{}", a[0], y)).collect())
    } else {
        Err(PyValueError::new_err(
            "operands could not be broadcast together",
        ))
    }
}

/// 逐元素重复（np.char.multiply，i 可为序列）
#[pyfunction]
pub fn char_multiply(a: Vec<String>, i: Vec<i64>) -> PyResult<Vec<String>> {
    if a.len() == i.len() {
        Ok(a.iter()
            .zip(i.iter())
            .map(|(s, n)| s.repeat((*n).max(0) as usize))
            .collect())
    } else if i.len() == 1 {
        Ok(a.iter().map(|s| s.repeat((i[0]).max(0) as usize)).collect())
    } else {
        Err(PyValueError::new_err(
            "operands could not be broadcast together",
        ))
    }
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(char_str_op, m)?)?;
    m.add_function(wrap_pyfunction!(char_int_op, m)?)?;
    m.add_function(wrap_pyfunction!(char_bool_op, m)?)?;
    m.add_function(wrap_pyfunction!(char_add, m)?)?;
    m.add_function(wrap_pyfunction!(char_multiply, m)?)?;
    Ok(())
}
