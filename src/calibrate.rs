//! Activation-aware NVFP4 packing of one linear layer.
//!
//! Round-to-nearest NVFP4 of every linear in nomic-embed-text-v1.5 costs
//! nineteen points of neighbour recall (80.9% recall@10 against the f32 model
//! over our own prose). Two changes to the rounding, neither of which trains
//! anything, win eleven of them back (92.3%):
//!
//! * **AWQ channel scales.** Scale input channel `j` by `mean|x_j|^alpha`
//!   (geometric mean one over the tensor) before rounding and divide it back
//!   after, so the rounding grid follows the channels the inputs actually
//!   exercise. `alpha` is chosen per tensor from a grid by least output error
//!   `sum |(W - Wq) x|^2` over captured input rows.
//! * **GPTQ error feedback.** Round the columns left to right and push each
//!   column's rounding error onto the columns still to come through the
//!   inverse Hessian `(X^T X / R + damp)^-1` of the inputs, so the layer's
//!   output error is what gets minimised rather than the weight error. Block
//!   scales are fixed per row when a block of sixteen columns is reached, from
//!   the values the feedback has left there. Kept only where it beats nearest
//!   rounding on the same rows.
//!
//! The channel scale cannot live inside the codes (it is per input column, the
//! codes are per element under a per-block scale), so a packed linear is the
//! codes of `W * diag(s)` plus the vector `1 / s`, applied to the input. A
//! consumer that wants a dense f32 matrix multiplies column `j` by
//! `input_scale[j]`, which [`Calibrated::decode`] does and
//! `selection::fold_input_scales` does for a whole keymap.
//!
//! Numbers and framing: wiki 88b76d5f, Compass goal dcf9dbaf. Host code by
//! design: this is weight preprocessing done once per model, not model math.

use anyhow::{Result, ensure};

use crate::nvfp4::{BLOCK, Packed, block_unit, code_of, decode_code, global_scale, pack_nearest};

/// The alpha grid: 0.0 is plain rounding, the rest bend the grid toward the
/// busy channels by that power of their mean magnitude.
pub const AWQ_ALPHAS: [f32; 10] = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9];

/// What to try.
#[derive(Clone, Copy, Debug)]
pub struct Options<'a> {
    /// Channel-scale exponents to search; include 0.0 for the plain baseline.
    pub alphas: &'a [f32],
    /// Error-feedback rounding after the scale search.
    pub feedback: bool,
    /// Pack embedding tables too. They have no captured inputs (a lookup
    /// reads one row), so they get nearest rounding under the tensor scale;
    /// nomic-text's word table is 94 MB of f32, most of what is left once
    /// the linears are packed.
    pub embeddings: bool,
}

impl Default for Options<'static> {
    fn default() -> Self {
        Self {
            alphas: &AWQ_ALPHAS,
            feedback: true,
            embeddings: false,
        }
    }
}

/// One packed linear and how it was made.
#[derive(Clone, Debug)]
pub struct Calibrated {
    /// Codes of `W * diag(s)`.
    pub packed: Packed,
    /// `1 / s`, one per input column; all ones when no scale was applied.
    pub input_scale: Vec<f32>,
    /// The chosen channel-scale exponent (0.0: none).
    pub alpha: f32,
    /// Output error over the calibration rows relative to plain rounding.
    pub error_ratio: f64,
    /// Whether error feedback made the final rounding.
    pub feedback_used: bool,
}

impl Calibrated {
    /// Plain nearest rounding, no scales, no feedback.
    pub fn plain(packed: Packed) -> Self {
        let cols = packed.cols;
        Self {
            packed,
            input_scale: vec![1.0; cols],
            alpha: 0.0,
            error_ratio: 1.0,
            feedback_used: false,
        }
    }

    /// The dense f32 matrix a consumer of `W` should use: the decoded codes
    /// with the input scale folded into the columns.
    pub fn decode(&self) -> Vec<f32> {
        let mut w = self.packed.decode();
        let cols = self.packed.cols;
        for row in w.chunks_mut(cols) {
            for (v, s) in row.iter_mut().zip(&self.input_scale) {
                *v *= s;
            }
        }
        w
    }
}

/// Output error of a quantised weight on real inputs: the sum over rows of
/// `|(W - Wq) x|^2`, the quantity AWQ minimises.
pub fn output_error(w: &[f32], wq: &[f32], cols: usize, rows: &[Vec<f32>]) -> f64 {
    let diff: Vec<f32> = w.iter().zip(wq).map(|(a, b)| a - b).collect();
    let threads = std::thread::available_parallelism()
        .map_or(8, |n| n.get())
        .min(32)
        .max(1);
    let chunk = rows.len().div_ceil(threads).max(1);
    std::thread::scope(|scope| {
        let handles: Vec<_> = rows
            .chunks(chunk)
            .map(|part| {
                let diff = &diff;
                scope.spawn(move || {
                    let mut err = 0f64;
                    for x in part {
                        for row in diff.chunks(cols) {
                            let acc: f32 = row.iter().zip(x).map(|(d, xi)| d * xi).sum();
                            err += (acc as f64) * (acc as f64);
                        }
                    }
                    err
                })
            })
            .collect();
        handles
            .into_iter()
            .map(|h| h.join().expect("error worker"))
            .sum()
    })
}

/// Cholesky factor L (lower, row-major n x n) of a symmetric positive definite
/// matrix; panics on a non-positive pivot, which the damping prevents.
fn cholesky(a: &[f64], n: usize) -> Vec<f64> {
    let mut l = vec![0f64; n * n];
    for i in 0..n {
        for j in 0..=i {
            let mut sum = a[i * n + j];
            for k in 0..j {
                sum -= l[i * n + k] * l[j * n + k];
            }
            if i == j {
                assert!(sum > 0.0, "cholesky: non-positive pivot at {i}: {sum}");
                l[i * n + i] = sum.sqrt();
            } else {
                l[i * n + j] = sum / l[j * n + j];
            }
        }
    }
    l
}

/// Inverse of a symmetric positive definite matrix from its Cholesky factor,
/// one column of the identity per solve, columns spread over threads.
fn spd_inverse(l: &[f64], n: usize) -> Vec<f64> {
    let threads = std::thread::available_parallelism()
        .map_or(8, |v| v.get())
        .min(32)
        .max(1);
    let chunk = n.div_ceil(threads).max(1);
    let mut inv = vec![0f64; n * n];
    std::thread::scope(|scope| {
        let handles: Vec<_> = (0..n)
            .step_by(chunk)
            .map(|start| {
                let end = (start + chunk).min(n);
                scope.spawn(move || {
                    let mut cols = Vec::with_capacity((end - start) * n);
                    let mut y = vec![0f64; n];
                    for c in start..end {
                        for i in 0..n {
                            let mut sum = if i == c { 1.0 } else { 0.0 };
                            for k in 0..i {
                                sum -= l[i * n + k] * y[k];
                            }
                            y[i] = sum / l[i * n + i];
                        }
                        let mut x = vec![0f64; n];
                        for i in (0..n).rev() {
                            let mut sum = y[i];
                            for k in i + 1..n {
                                sum -= l[k * n + i] * x[k];
                            }
                            x[i] = sum / l[i * n + i];
                        }
                        cols.extend_from_slice(&x);
                    }
                    (start, cols)
                })
            })
            .collect();
        for h in handles {
            let (start, cols) = h.join().expect("inverse worker");
            for (c, col) in cols.chunks(n).enumerate() {
                for i in 0..n {
                    inv[i * n + start + c] = col[i];
                }
            }
        }
    });
    inv
}

/// Error-feedback (GPTQ) NVFP4 of `w` (`[rows, cols]`) against input rows `x`
/// (each of length `cols`), under one global scale from `w`'s own maximum.
pub fn pack_gptq(w: &[f32], cols: usize, x: &[Vec<f32>]) -> Result<Packed> {
    ensure!(
        cols > 0 && cols % BLOCK == 0,
        "cols {cols} is not a positive multiple of {BLOCK}"
    );
    ensure!(
        w.len() % cols == 0,
        "{} values do not fill rows of {cols}",
        w.len()
    );
    ensure!(!x.is_empty(), "error feedback needs input rows");
    ensure!(
        x.iter().all(|r| r.len() == cols),
        "an input row is not {cols} wide"
    );
    let rows = w.len() / cols;
    let n = cols;

    // H = X^T X / R, damped by one percent of its mean diagonal.
    let mut h = vec![0f64; n * n];
    for r in x {
        for i in 0..n {
            let xi = r[i] as f64;
            if xi == 0.0 {
                continue;
            }
            let row = &mut h[i * n..(i + 1) * n];
            for (hij, &xj) in row.iter_mut().zip(r.iter()) {
                *hij += xi * xj as f64;
            }
        }
    }
    let scale = 1.0 / x.len() as f64;
    for v in h.iter_mut() {
        *v *= scale;
    }
    let mean_diag = (0..n).map(|i| h[i * n + i]).sum::<f64>() / n as f64;
    let damp = 0.01 * mean_diag.max(1e-12);
    for i in 0..n {
        h[i * n + i] += damp;
    }
    let l = cholesky(&h, n);
    let hinv = spd_inverse(&l, n);
    // U = chol(Hinv)^T, upper; the feedback uses its rows.
    let lu = cholesky(&hinv, n);
    let mut u = vec![0f64; n * n];
    for i in 0..n {
        for j in 0..=i {
            u[j * n + i] = lu[i * n + j];
        }
    }

    let scale2 = global_scale(w);
    let threads = std::thread::available_parallelism()
        .map_or(8, |v| v.get())
        .min(32)
        .max(1);
    let chunk = rows.div_ceil(threads).max(1);
    let mut codes = vec![0u8; w.len() / 2];
    let mut scales = vec![0u8; w.len() / BLOCK];
    std::thread::scope(|scope| {
        let handles: Vec<_> = w
            .chunks(chunk * cols)
            .enumerate()
            .map(|(ci, part)| {
                let u = &u;
                scope.spawn(move || {
                    let mut work: Vec<f32> = part.to_vec();
                    let nrows = part.len() / cols;
                    let mut codes = vec![0u8; part.len() / 2];
                    let mut scales = vec![0u8; part.len() / BLOCK];
                    let mut units = vec![0f32; nrows];
                    for j in 0..cols {
                        if j % BLOCK == 0 {
                            for r in 0..nrows {
                                let block = &work[r * cols + j..r * cols + j + BLOCK];
                                let (byte, unit) = block_unit(block, scale2);
                                scales[(r * cols + j) / BLOCK] = byte;
                                units[r] = unit;
                            }
                        }
                        let ujj = u[j * n + j];
                        for r in 0..nrows {
                            let e = r * cols + j;
                            let v = work[e];
                            let code = code_of(v, units[r]);
                            codes[e / 2] |= code << (4 * (e % 2));
                            let qv = decode_code(code) * units[r];
                            let err = ((v - qv) as f64 / ujj) as f32;
                            if err != 0.0 {
                                let urow = &u[j * n + j + 1..(j + 1) * n];
                                let wrow = &mut work[e + 1..(r + 1) * cols];
                                for (wk, &ujk) in wrow.iter_mut().zip(urow) {
                                    *wk -= err * ujk as f32;
                                }
                            }
                        }
                    }
                    (ci, codes, scales)
                })
            })
            .collect();
        for h in handles {
            let (ci, c, s) = h.join().expect("gptq worker");
            let c0 = ci * chunk * cols / 2;
            codes[c0..c0 + c.len()].copy_from_slice(&c);
            let s0 = ci * chunk * cols / BLOCK;
            scales[s0..s0 + s.len()].copy_from_slice(&s);
        }
    });
    Ok(Packed {
        rows,
        cols,
        codes,
        scales,
        scale2,
    })
}

// ── voxtral-gptq lane: GPTQ from a precomputed H, blocked f64 algebra ──

fn par_rows<F: Fn(usize, usize) + Sync>(n: usize, f: F) {
    let threads = std::thread::available_parallelism()
        .map_or(8, |v| v.get())
        .min(32)
        .max(1);
    let chunk = n.div_ceil(threads).max(1);
    std::thread::scope(|scope| {
        for start in (0..n).step_by(chunk) {
            let f = &f;
            scope.spawn(move || f(start, (start + chunk).min(n)));
        }
    });
}

#[derive(Clone, Copy)]
struct SyncPtr(*mut f64);
unsafe impl Send for SyncPtr {}
unsafe impl Sync for SyncPtr {}

/// In-place lower Cholesky of row-major SPD `a` (n x n); the strict upper
/// triangle is left as garbage-free zeros on return.
fn cholesky_blocked(a: &mut [f64], n: usize) -> Result<()> {
    const NB: usize = 128;
    let p = SyncPtr(a.as_mut_ptr());
    let mut k0 = 0;
    while k0 < n {
        let kb = NB.min(n - k0);
        let k1 = k0 + kb;
        // diagonal block, unblocked
        for i in k0..k1 {
            for j in k0..=i {
                let mut sum = a[i * n + j];
                for k in k0..j {
                    sum -= a[i * n + k] * a[j * n + k];
                }
                if i == j {
                    ensure!(sum > 0.0, "cholesky: non-positive pivot at {i}: {sum}");
                    a[i * n + i] = sum.sqrt();
                } else {
                    a[i * n + j] = sum / a[j * n + j];
                }
            }
        }
        if k1 < n {
            // panel: rows k1..n, L21 = A21 L11^-T, row by row
            par_rows(n - k1, |s, e| {
                let p = p;
                for r in k1 + s..k1 + e {
                    for j in k0..k1 {
                        // SAFETY: row r of the panel is owned by this thread;
                        // the diagonal block is read-only here.
                        unsafe {
                            let mut sum = *p.0.add(r * n + j);
                            for k in k0..j {
                                sum -= *p.0.add(r * n + k) * *p.0.add(j * n + k);
                            }
                            *p.0.add(r * n + j) = sum / *p.0.add(j * n + j);
                        }
                    }
                }
            });
            // trailing: A22 -= L21 L21^T, split over row blocks of A22
            let m = n - k1;
            par_rows(m, |s, e| {
                let p = p;
                // SAFETY: each thread writes rows k1+s..k1+e of A22 (cols
                // k1..n), and reads the panel columns k0..k1, disjoint.
                unsafe {
                    matrixmultiply::dgemm(
                        e - s,
                        kb,
                        m,
                        -1.0,
                        p.0.add((k1 + s) * n + k0),
                        n as isize,
                        1,
                        p.0.add(k1 * n + k0),
                        1,
                        n as isize,
                        1.0,
                        p.0.add((k1 + s) * n + k1),
                        n as isize,
                        1,
                    );
                }
            });
        }
        k0 = k1;
    }
    for i in 0..n {
        for j in i + 1..n {
            a[i * n + j] = 0.0;
        }
    }
    Ok(())
}

/// Inverse of a lower-triangular row-major `l`, by column blocks solved with
/// forward substitution in parallel.
fn lower_inverse(l: &[f64], n: usize) -> Vec<f64> {
    const CB: usize = 64;
    let mut inv = vec![0f64; n * n];
    let p = SyncPtr(inv.as_mut_ptr());
    let blocks = n.div_ceil(CB);
    par_rows(blocks, |s, e| {
        let p = p;
        let mut y = vec![0f64; n * CB];
        for blk in s..e {
            let c0 = blk * CB;
            let cw = CB.min(n - c0);
            y.iter_mut().for_each(|v| *v = 0.0);
            // rows below c0 are zero in these columns
            for i in c0..n {
                let mut acc = [0f64; CB];
                if i - c0 < cw {
                    acc[i - c0] = 1.0;
                }
                let row = &l[i * n..i * n + i];
                for k in c0..i {
                    let lik = row[k];
                    if lik != 0.0 {
                        let yk = &y[k * CB..k * CB + cw];
                        for (a, &v) in acc[..cw].iter_mut().zip(yk) {
                            *a -= lik * v;
                        }
                    }
                }
                let d = l[i * n + i];
                for c in 0..cw {
                    y[i * CB + c] = acc[c] / d;
                }
            }
            for i in c0..n {
                for c in 0..cw {
                    // SAFETY: columns c0..c0+cw are owned by this block.
                    unsafe { *p.0.add(i * n + c0 + c) = y[i * CB + c] };
                }
            }
        }
    });
    inv
}

/// Captured input statistics for calibrated packing: per projection the
/// accumulated `H = X^T X` of its inputs, as raw little-endian f32
/// `{dir}/{stem}.h` files (`voxtral_capture`), summed over `dirs`, so several
/// capture runs (several machines, several clip sets) add up.
#[derive(Clone, Debug)]
pub struct Calibration {
    pub dirs: Vec<std::path::PathBuf>,
    /// Damping, as a fraction of `H`'s mean diagonal.
    pub damp: f64,
    pub scale: BlockScale,
}

impl Calibration {
    /// GPTQ's usual damping and the searched block scale.
    pub fn new(dirs: Vec<std::path::PathBuf>) -> Self {
        Self { dirs, damp: 0.01, scale: BlockScale::Search }
    }

    /// `H` for the projection stored as `stem`, `k x k` row major, summed over
    /// every directory that has it; None if none has.
    pub fn h(&self, stem: &str, k: usize) -> Result<Option<Vec<f64>>> {
        let mut sum: Option<Vec<f64>> = None;
        for dir in &self.dirs {
            let path = dir.join(format!("{stem}.h"));
            if !path.exists() {
                continue;
            }
            let bytes = std::fs::read(&path)?;
            ensure!(bytes.len() == k * k * 4, "{path:?}: {} bytes for k {k}", bytes.len());
            let acc = sum.get_or_insert_with(|| vec![0f64; k * k]);
            for (a, b) in acc.iter_mut().zip(bytes.chunks_exact(4)) {
                *a += f32::from_le_bytes(b.try_into().expect("four bytes")) as f64;
            }
        }
        Ok(sum)
    }
}

/// How GPTQ picks a block's E4M3 scale when it reaches the block.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum BlockScale {
    /// From the block's maximum, as round-to-nearest does: the largest
    /// element lands on E2M1's 6.
    #[default]
    Max,
    /// The least-error scale byte among the maximum's and its neighbours,
    /// four codes below (the largest element then lands near E2M1's 4, which
    /// is the "Four Over Six" choice, arXiv 2512.02010, and the steps between)
    /// and one above. Error is the block's squared rounding error, weighted
    /// per column as GPTQ weighs it.
    Search,
}

impl BlockScale {
    pub fn parse(value: &str) -> Result<Self> {
        match value {
            "max" => Ok(Self::Max),
            "search" => Ok(Self::Search),
            other => anyhow::bail!("block scale must be max or search, not {other:?}"),
        }
    }
}

/// The scale byte and unit of `block` under [`BlockScale::Search`]: every
/// candidate byte from four below the maximum's to one above, the one with
/// the least `sum weight_i (v_i - q_i)^2`, the maximum's on a tie.
pub fn block_unit_search(block: &[f32], weights: &[f32], scale2: f32) -> (u8, f32) {
    let (nominal, unit) = block_unit(block, scale2);
    if nominal == 0 {
        return (nominal, unit);
    }
    let error = |unit: f32| -> f32 {
        block
            .iter()
            .zip(weights)
            .map(|(&v, &wt)| {
                let d = v - decode_code(code_of(v, unit)) * unit;
                wt * d * d
            })
            .sum()
    };
    let mut best = (nominal, unit, error(unit));
    let lo = nominal.saturating_sub(4).max(1);
    let hi = (nominal + 1).min(0x7E);
    for byte in lo..=hi {
        if byte == nominal {
            continue;
        }
        let u = crate::nvfp4::e4m3_to_f32(byte) * scale2;
        let e = error(u);
        if e < best.2 {
            best = (byte, u, e);
        }
    }
    (best.0, best.1)
}

/// GPTQ NVFP4 of `w` (`[rows, cols]`) from an accumulated `H = X^T X`
/// (row-major cols x cols, any positive scale), damped by `damp` times its
/// mean diagonal, block scales chosen by `scale`. Returns the packing and its
/// H-weighted output error relative to nearest rounding,
/// `tr((W-Wq) H (W-Wq)^T)`.
pub fn pack_gptq_h(
    w: &[f32],
    cols: usize,
    mut h: Vec<f64>,
    damp: f64,
    scale: BlockScale,
) -> Result<(Packed, f64)> {
    let n = cols;
    ensure!(n > 0 && n % BLOCK == 0 && w.len() % n == 0, "bad shape");
    ensure!(h.len() == n * n, "H is not {n} x {n}");
    let rows = w.len() / n;
    let mean_diag = (0..n).map(|i| h[i * n + i]).sum::<f64>() / n as f64;
    let d = damp * mean_diag.max(1e-12);
    for i in 0..n {
        if h[i * n + i] <= 0.0 {
            h[i * n + i] = mean_diag.max(1e-12);
        }
        h[i * n + i] += d;
    }
    let h_orig = h.clone();
    // Hinv = U^T U, U upper: with J the reversal, chol(J H J) = M lower,
    // H = R R^T where R = J M J is upper, and U = R^-1 = J M^-1 J.
    let mut jhj = vec![0f64; n * n];
    for i in 0..n {
        for j in 0..n {
            jhj[i * n + j] = h[(n - 1 - i) * n + (n - 1 - j)];
        }
    }
    drop(h);
    cholesky_blocked(&mut jhj, n)?;
    let minv = lower_inverse(&jhj, n);
    drop(jhj);
    let mut u = vec![0f64; n * n];
    for i in 0..n {
        for j in i..n {
            u[i * n + j] = minv[(n - 1 - i) * n + (n - 1 - j)];
        }
    }
    drop(minv);

    let scale2 = global_scale(w);
    let mut codes = vec![0u8; w.len() / 2];
    let mut scales = vec![0u8; w.len() / BLOCK];
    let threads = std::thread::available_parallelism()
        .map_or(8, |v| v.get())
        .min(32)
        .max(1);
    let chunk = rows.div_ceil(threads).max(1);
    let u32: Vec<f32> = u.iter().map(|&v| v as f32).collect();
    drop(u);
    std::thread::scope(|scope| {
        let handles: Vec<_> = w
            .chunks(chunk * n)
            .enumerate()
            .map(|(ci, part)| {
                let u = &u32;
                scope.spawn(move || {
                    let mut work: Vec<f32> = part.to_vec();
                    let nrows = part.len() / n;
                    let mut codes = vec![0u8; part.len() / 2];
                    let mut scales = vec![0u8; part.len() / BLOCK];
                    let mut units = vec![0f32; nrows];
                    for j in 0..n {
                        if j % BLOCK == 0 {
                            // GPTQ's cost of a rounding error at column c is
                            // err^2 / U_cc^2.
                            let mut weights = [0f32; BLOCK];
                            for (c, wt) in weights.iter_mut().enumerate() {
                                let d = u[(j + c) * n + j + c];
                                *wt = 1.0 / (d * d);
                            }
                            for r in 0..nrows {
                                let block = &work[r * n + j..r * n + j + BLOCK];
                                let (byte, unit) = match scale {
                                    BlockScale::Max => block_unit(block, scale2),
                                    BlockScale::Search => block_unit_search(block, &weights, scale2),
                                };
                                scales[(r * n + j) / BLOCK] = byte;
                                units[r] = unit;
                            }
                        }
                        let ujj = u[j * n + j];
                        let urow = &u[j * n + j + 1..(j + 1) * n];
                        for r in 0..nrows {
                            let e = r * n + j;
                            let v = work[e];
                            let code = code_of(v, units[r]);
                            codes[e / 2] |= code << (4 * (e % 2));
                            let qv = decode_code(code) * units[r];
                            let err = (v - qv) / ujj;
                            if err != 0.0 {
                                let wrow = &mut work[e + 1..(r + 1) * n];
                                for (wk, &ujk) in wrow.iter_mut().zip(urow) {
                                    *wk -= err * ujk;
                                }
                            }
                        }
                    }
                    (ci, codes, scales)
                })
            })
            .collect();
        for hd in handles {
            let (ci, c, s) = hd.join().expect("gptq worker");
            let c0 = ci * chunk * n / 2;
            codes[c0..c0 + c.len()].copy_from_slice(&c);
            let s0 = ci * chunk * n / BLOCK;
            scales[s0..s0 + s.len()].copy_from_slice(&s);
        }
    });
    let packed = Packed { rows, cols: n, codes, scales, scale2 };
    let plain = pack_nearest(w, n)?;
    let err = |wq: &[f32]| -> f64 { h_weighted_error(w, wq, n, &h_orig) };
    let e_g = err(&packed.decode());
    let e_p = err(&plain.decode());
    Ok((packed, if e_p > 0.0 { e_g / e_p } else { 1.0 }))
}

/// `tr(D H D^T)` for `D = W - Wq`, row blocks in parallel through dgemm.
fn h_weighted_error(w: &[f32], wq: &[f32], n: usize, h: &[f64]) -> f64 {
    const RB: usize = 64;
    let rows = w.len() / n;
    let total = std::sync::Mutex::new(0f64);
    let blocks = rows.div_ceil(RB);
    par_rows(blocks, |s, e| {
        let mut acc = 0f64;
        let mut d = vec![0f64; RB * n];
        let mut dh = vec![0f64; RB * n];
        for blk in s..e {
            let r0 = blk * RB;
            let rb = RB.min(rows - r0);
            for i in 0..rb * n {
                d[i] = (w[r0 * n + i] - wq[r0 * n + i]) as f64;
            }
            // SAFETY: plain row-major buffers of the sizes passed.
            unsafe {
                matrixmultiply::dgemm(
                    rb, n, n, 1.0,
                    d.as_ptr(), n as isize, 1,
                    h.as_ptr(), n as isize, 1,
                    0.0,
                    dh.as_mut_ptr(), n as isize, 1,
                );
            }
            acc += d[..rb * n].iter().zip(&dh[..rb * n]).map(|(a, b)| a * b).sum::<f64>();
        }
        *total.lock().unwrap() += acc;
    });
    total.into_inner().unwrap()
}

/// `mean|x|^alpha`, geometric mean one, or all ones at alpha zero.
fn channel_scales(mean_abs: &[f32], alpha: f32) -> Vec<f32> {
    if alpha == 0.0 {
        return vec![1.0; mean_abs.len()];
    }
    let mut s: Vec<f32> = mean_abs.iter().map(|m| (m + 1e-8).powf(alpha)).collect();
    let log_mean = s.iter().map(|v| v.ln() as f64).sum::<f64>() / s.len() as f64;
    let norm = log_mean.exp() as f32;
    for v in &mut s {
        *v /= norm;
    }
    s
}

fn scale_columns(w: &[f32], cols: usize, s: &[f32]) -> Vec<f32> {
    let mut out = w.to_vec();
    for row in out.chunks_mut(cols) {
        for (v, si) in row.iter_mut().zip(s) {
            *v *= si;
        }
    }
    out
}

/// Calibrated NVFP4 of one linear `w` (`[rows, cols]`): input rows `rows`
/// (each `cols` wide) and per-channel `mean_abs` of the inputs drive the
/// channel-scale search and the error feedback.
pub fn pack_linear(
    w: &[f32],
    cols: usize,
    rows: &[Vec<f32>],
    mean_abs: &[f32],
    opts: &Options<'_>,
) -> Result<Calibrated> {
    ensure!(
        cols > 0 && cols % BLOCK == 0,
        "cols {cols} is not a positive multiple of {BLOCK}"
    );
    ensure!(
        w.len() % cols == 0,
        "{} values do not fill rows of {cols}",
        w.len()
    );
    ensure!(!rows.is_empty(), "calibration needs input rows");
    ensure!(
        rows.iter().all(|r| r.len() == cols),
        "an input row is not {cols} wide"
    );
    ensure!(
        mean_abs.len() == cols,
        "mean_abs has {} entries for {cols} columns",
        mean_abs.len()
    );

    let plain = pack_nearest(w, cols)?;
    let plain_err = output_error(w, &plain.decode(), cols, rows);
    let mut best = Calibrated::plain(plain);
    let mut best_err = plain_err;

    for &alpha in opts.alphas.iter().filter(|&&a| a != 0.0) {
        let s = channel_scales(mean_abs, alpha);
        let packed = pack_nearest(&scale_columns(w, cols, &s), cols)?;
        let cal = Calibrated {
            packed,
            input_scale: s.iter().map(|v| 1.0 / v).collect(),
            alpha,
            error_ratio: 1.0,
            feedback_used: false,
        };
        let err = output_error(w, &cal.decode(), cols, rows);
        if err < best_err {
            best = cal;
            best_err = err;
        }
    }

    if opts.feedback {
        let s = channel_scales(mean_abs, best.alpha);
        let xs: Vec<Vec<f32>> = rows
            .iter()
            .map(|r| r.iter().zip(&s).map(|(xi, si)| xi / si).collect())
            .collect();
        let packed = pack_gptq(&scale_columns(w, cols, &s), cols, &xs)?;
        let cal = Calibrated {
            packed,
            input_scale: s.iter().map(|v| 1.0 / v).collect(),
            alpha: best.alpha,
            error_ratio: 1.0,
            feedback_used: true,
        };
        let err = output_error(w, &cal.decode(), cols, rows);
        if err < best_err {
            best = cal;
            best_err = err;
        }
    }

    best.error_ratio = if plain_err > 0.0 {
        best_err / plain_err
    } else {
        1.0
    };
    Ok(best)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn matrix(rows: usize, cols: usize) -> Vec<f32> {
        (0..rows * cols)
            .map(|i| {
                let x = ((i * 7919) % 1000) as f32 / 1000.0 - 0.5;
                // Column j carries a magnitude that grows with j, so channel
                // scales have something to follow.
                x * (1.0 + (i % cols) as f32 / cols as f32)
            })
            .collect()
    }

    fn inputs(n: usize, cols: usize) -> Vec<Vec<f32>> {
        (0..n)
            .map(|r| {
                (0..cols)
                    .map(|j| {
                        let x = (((r * 131 + j * 17) % 997) as f32 / 997.0) - 0.5;
                        // The last quarter of the channels is busy.
                        if j >= 3 * cols / 4 { x * 8.0 } else { x }
                    })
                    .collect()
            })
            .collect()
    }

    fn mean_abs(rows: &[Vec<f32>], cols: usize) -> Vec<f32> {
        let mut m = vec![0f32; cols];
        for r in rows {
            for (mj, x) in m.iter_mut().zip(r) {
                *mj += x.abs();
            }
        }
        for v in &mut m {
            *v /= rows.len() as f32;
        }
        m
    }

    #[test]
    fn gptq_h_algebra_and_gain() {
        // n spans several Cholesky (128) and inverse (64) blocks.
        let n = 320;
        let x = inputs(2000, n);
        let mut h = vec![0f64; n * n];
        for r in &x {
            for i in 0..n {
                for j in 0..n {
                    h[i * n + j] += r[i] as f64 * r[j] as f64;
                }
            }
        }
        let mut a = h.clone();
        for i in 0..n { a[i * n + i] += 1.0; }
        let a0 = a.clone();
        cholesky_blocked(&mut a, n).unwrap();
        let mut worst = 0f64;
        for i in 0..n { for j in 0..n {
            let v: f64 = (0..n).map(|k| a[i * n + k] * a[j * n + k]).sum();
            worst = worst.max((v - a0[i * n + j]).abs());
        }}
        assert!(worst < 1e-6 * a0[0], "LL^T residual {worst}");
        let inv = lower_inverse(&a, n);
        for i in 0..n { for j in 0..n {
            let v: f64 = (0..n).map(|k| a[i * n + k] * inv[k * n + j]).sum();
            let want = if i == j { 1.0 } else { 0.0 };
            assert!((v - want).abs() < 1e-9, "L Linv at {i},{j}: {v}");
        }}
        let w = matrix(48, n);
        let (_, ratio) = pack_gptq_h(&w, n, h.clone(), 0.01, BlockScale::Max).unwrap();
        assert!(ratio < 1.0, "gptq ratio {ratio}");
        let (_, searched) = pack_gptq_h(&w, n, h, 0.01, BlockScale::Search).unwrap();
        assert!(searched < 1.0, "gptq+search ratio {searched}");
    }

    #[test]
    fn block_scale_search_never_loses_to_the_maximum() {
        let w = matrix(64, 256);
        let scale2 = global_scale(&w);
        let ones = [1f32; BLOCK];
        let mut better = 0;
        for block in w.chunks(BLOCK) {
            let err = |unit: f32| -> f32 {
                block.iter().map(|&v| (v - decode_code(code_of(v, unit)) * unit).powi(2)).sum()
            };
            let (_, max_unit) = block_unit(block, scale2);
            let (_, unit) = block_unit_search(block, &ones, scale2);
            assert!(err(unit) <= err(max_unit), "search lost: {} > {}", err(unit), err(max_unit));
            better += (err(unit) < err(max_unit)) as usize;
        }
        assert!(better > 0, "search never improved a block");
    }

    #[test]
    fn calibration_never_loses_to_plain_rounding() {
        let (rows, cols) = (8, 64);
        let w = matrix(rows, cols);
        let x = inputs(64, cols);
        let m = mean_abs(&x, cols);
        let cal = pack_linear(&w, cols, &x, &m, &Options::default()).expect("calibrates");
        assert!(cal.error_ratio <= 1.0, "ratio {}", cal.error_ratio);
        assert_eq!(cal.input_scale.len(), cols);
        assert_eq!(cal.decode().len(), rows * cols);
    }

    #[test]
    fn feedback_alone_beats_nearest_on_correlated_inputs() {
        let (rows, cols) = (4, 32);
        let w = matrix(rows, cols);
        let x = inputs(48, cols);
        let plain = pack_nearest(&w, cols).expect("packs");
        let fed = pack_gptq(&w, cols, &x).expect("feeds back");
        let e_plain = output_error(&w, &plain.decode(), cols, &x);
        let e_fed = output_error(&w, &fed.decode(), cols, &x);
        assert!(e_fed <= e_plain, "feedback {e_fed} vs plain {e_plain}");
        assert_eq!(
            fed.scale2, plain.scale2,
            "the global scale is the tensor's, not the feedback's"
        );
    }

    #[test]
    fn decode_folds_the_input_scale_into_the_columns() {
        let cols = 16;
        let w = vec![1.0f32; cols];
        let packed = pack_nearest(&w, cols).expect("packs");
        let cal = Calibrated {
            packed,
            input_scale: (0..cols).map(|j| j as f32).collect(),
            alpha: 0.5,
            error_ratio: 1.0,
            feedback_used: false,
        };
        let dec = cal.decode();
        for (j, v) in dec.iter().enumerate() {
            assert_eq!(*v, j as f32);
        }
    }
}

// ── a whole keymap, and a pile to keep it in ─────────────────────────────

use std::collections::HashMap;
use std::path::Path;

use ed25519_dalek::SigningKey;
use triblespace::macros::entity;
use triblespace::prelude::*;

/// A model as the f32 loaders speak it: name to (flat data, shape).
pub type Keymap = HashMap<String, (Vec<f32>, Vec<usize>)>;

/// Captured inputs of one linear: the rows and the per-channel mean `|x|`.
#[derive(Clone, Debug, Default)]
pub struct InputStats {
    pub rows: Vec<Vec<f32>>,
    pub mean_abs: Vec<f32>,
}

/// What [`pack_keymap`] did.
#[derive(Debug, Default)]
pub struct PackReport {
    /// The packed form of every tensor that was packed.
    pub packed: HashMap<String, Calibrated>,
    pub tensors: usize,
    pub elements: usize,
    /// Packed payload bytes, the pile's size for those tensors.
    pub packed_bytes: usize,
}

/// The nomic models' capture key for a weight name: the module prefix, with
/// the gated MLP's two input projections sharing their input under `.mlp.fc1`
/// (`encoder.layers.3.mlp.fc12.weight` captures as `encoder.layers.3.mlp.fc1`).
pub fn nomic_capture_key(tensor_name: &str) -> String {
    tensor_name
        .trim_end_matches(".weight")
        .replace(".mlp.fc11", ".mlp.fc1")
        .replace(".mlp.fc12", ".mlp.fc1")
}

/// Whether a keymap tensor is a weight this packer takes: rank two, named a
/// weight, not a norm, an embedding table only when `embeddings`, input width
/// a multiple of the block, and named by `only` when `only` is not empty.
pub fn packs(name: &str, shape: &[usize], only: &[String], embeddings: bool) -> bool {
    let lower = name.to_ascii_lowercase();
    shape.len() == 2
        && lower.contains("weight")
        && !(lower.contains("norm") || lower.contains("ln"))
        && (embeddings || !lower.contains("embed"))
        && shape[1] % BLOCK == 0
        && (only.is_empty() || only.iter().any(|p| lower.contains(&p.to_ascii_lowercase())))
}

/// Pack every eligible linear of a keymap in place. `stats_for(name)` gives a
/// tensor's captured inputs (none: plain nearest rounding). The keymap keeps
/// the DECODED packed values afterwards, so whatever a caller runs or scores
/// next is exactly what a pile written from the report would hold. `log` is
/// told one line per tensor as it lands.
pub fn pack_keymap<'s>(
    keymap: &mut Keymap,
    only: &[String],
    stats_for: &dyn Fn(&str) -> Option<&'s InputStats>,
    opts: &Options<'_>,
    log: &mut dyn FnMut(&str),
) -> Result<PackReport> {
    let mut report = PackReport::default();
    let mut names: Vec<String> = keymap.keys().cloned().collect();
    names.sort();
    for name in names {
        let (data, shape) = keymap.get_mut(&name).expect("listed key");
        if !packs(&name, shape, only, opts.embeddings) {
            continue;
        }
        let cols = shape[1];
        let cal = match stats_for(&name) {
            Some(st) if st.rows.len() >= 16 && st.mean_abs.len() == cols => {
                let cal = pack_linear(data, cols, &st.rows, &st.mean_abs, opts)?;
                log(&format!(
                    "{name}: alpha {:.1}, output error x{:.3} of plain ({} rows{})",
                    cal.alpha,
                    cal.error_ratio,
                    st.rows.len(),
                    if cal.feedback_used { ", feedback" } else { "" }
                ));
                cal
            }
            _ => {
                log(&format!("{name}: no captured inputs, plain rounding"));
                Calibrated::plain(pack_nearest(data, cols)?)
            }
        };
        data.copy_from_slice(&cal.decode());
        report.tensors += 1;
        report.elements += data.len();
        report.packed_bytes += cal.packed.payload().len() + cal.input_scale.len() * 4;
        report.packed.insert(name, cal);
    }
    Ok(report)
}

/// Write a packed model as a NEW model pile: every packed linear as a
/// `Tensor<NVFP4, 2>` leaf plus its `<name>.input_scale` F32 vector, every
/// other tensor as an F32 leaf, under one root labelled `quantization` and
/// named `source`, and, when given, the tokenizer JSON beside it so the pile
/// loads on its own. `parents` are the actual roots selected for the source
/// weights, not IDs inferred from `source`. Their facts need not have arrived
/// in this pile yet. All labels and parent links annotate the member-only root.
///
/// A new file unless `append`: a pile is append-only and a second root in it
/// is a choice the caller makes by name. With `append`, the root joins an
/// existing model pile's collection (which the key must be allowed to write),
/// so one pile carries the native root and the packed one and every reader,
/// old or new, can address the root it asks for. Several roots may share a
/// label; that is a choice for the reader, not a publication constraint. Any
/// explicitly supplied tokenizer is published alongside it. Unrelated
/// tokenizers in the collection do not suppress that publication.
pub fn write_packed_pile(
    out: &Path,
    key: &SigningKey,
    keymap: &Keymap,
    packed: &HashMap<String, Calibrated>,
    source: &str,
    quantization: &str,
    parents: &[Id],
    tokenizer_json: Option<&[u8]>,
    append: bool,
) -> Result<Id> {
    use crate::format::attrs;
    use crate::leaf::{Elem, put_leaf};

    if append {
        ensure!(
            out.exists(),
            "--append needs an existing pile, {} is not one",
            out.display()
        );
    } else {
        ensure!(
            !out.exists(),
            "refusing to write into an existing pile {}: name a new file, or --append",
            out.display()
        );
        std::fs::File::create(out).map_err(|e| anyhow::anyhow!("create {}: {e}", out.display()))?;
    }
    let mut pile = Pile::open(out).map_err(|e| anyhow::anyhow!("open {}: {e:?}", out.display()))?;
    pile.refresh()
        .map_err(|e| anyhow::anyhow!("refresh {}: {e:?}", out.display()))?;
    crate::model_collection::model_graph_collection_or_create(&mut pile, key)
        .map_err(|e| anyhow::anyhow!("create the model collection: {e}"))?;

    let mut graph = Fragment::empty();
    let mut members: Vec<Id> = Vec::new();
    let mut add_member = |pile: &mut Pile,
                          graph: &mut Fragment,
                          leaf: Fragment,
                          name: &str,
                          kind: &str|
     -> Result<()> {
        let leaf_id = leaf
            .root()
            .ok_or_else(|| anyhow::anyhow!("{name}: leaf has no root"))?;
        *graph += leaf;
        let name_h = pile
            .put::<blobencodings::UTF8String, _>(name.to_string())
            .map_err(|e| anyhow::anyhow!("{name}: store name: {e:?}"))?;
        let member = entity! { _ @ attrs::kind: kind, attrs::safetensor_path: name_h, attrs::weight: leaf_id };
        members.push(
            member
                .root()
                .ok_or_else(|| anyhow::anyhow!("{name}: member has no root"))?,
        );
        *graph += member;
        Ok(())
    };
    let mut names: Vec<&String> = keymap.keys().collect();
    names.sort();
    for name in names {
        let (values, shape) = &keymap[name];
        let dims: Vec<u64> = shape.iter().map(|&d| d as u64).collect();
        if let Some(cal) = packed.get(name) {
            ensure!(
                shape.len() == 2 && cal.packed.rows == shape[0] && cal.packed.cols == shape[1],
                "{name}: packed {}x{} for shape {shape:?}",
                cal.packed.rows,
                cal.packed.cols
            );
            let leaf = put_leaf(
                &mut pile,
                Elem::Nvfp4,
                &dims,
                anybytes::Bytes::from_source(cal.packed.payload()),
                name,
            )?;
            add_member(&mut pile, &mut graph, leaf, name, "matrix")?;
            let scale_name = format!("{name}.input_scale");
            let leaf = put_leaf(
                &mut pile,
                Elem::F32,
                &[cal.input_scale.len() as u64],
                anybytes::Bytes::from_source(cal.input_scale.clone()),
                &scale_name,
            )?;
            add_member(&mut pile, &mut graph, leaf, &scale_name, "vector")?;
        } else {
            let leaf = put_leaf(
                &mut pile,
                Elem::F32,
                &dims,
                anybytes::Bytes::from_source(values.clone()),
                name,
            )?;
            let kind = match shape.len() {
                1 => "vector",
                2 => "matrix",
                _ => "tensor",
            };
            add_member(&mut pile, &mut graph, leaf, name, kind)?;
        }
    }
    let name_h = pile
        .put::<blobencodings::UTF8String, _>(format!("{source} ({quantization})"))
        .map_err(|e| anyhow::anyhow!("store model name: {e:?}"))?;
    let source_h = pile
        .put::<blobencodings::UTF8String, _>(source.to_string())
        .map_err(|e| anyhow::anyhow!("store source: {e:?}"))?;
    let model = entity! { _ @ attrs::member*: members.iter() };
    let root = model
        .root()
        .ok_or_else(|| anyhow::anyhow!("model has no root"))?;
    graph += model;
    graph += entity! { ExclusiveId::force_ref(&root) @
        attrs::model_name: name_h,
        attrs::source: source_h,
        attrs::quantization: quantization,
        attrs::parent*: parents.iter().filter(|parent| **parent != root),
    };
    crate::model_collection::publish_model_fragment(&mut pile, key, graph)
        .map_err(|e| anyhow::anyhow!("publish the packed model: {e}"))?;

    if let Some(json) = tokenizer_json {
        let tok = crate::tokenizer::save_tokenizer_json(json, source, &mut pile)
            .map_err(|e| anyhow::anyhow!("build tokenizer graph: {e}"))?;
        crate::model_collection::publish_model_fragment(&mut pile, key, tok)
            .map_err(|e| anyhow::anyhow!("publish the tokenizer: {e}"))?;
    }
    pile.close()
        .map_err(|e| anyhow::anyhow!("close {}: {e:?}", out.display()))?;
    Ok(root)
}

#[cfg(test)]
mod pile_tests {
    use super::*;
    use crate::selection::ModelSelector;

    fn matrix(rows: usize, cols: usize) -> Vec<f32> {
        (0..rows * cols)
            .map(|i| ((i * 7919) % 1000) as f32 / 1000.0 - 0.5)
            .collect()
    }

    fn inputs(n: usize, cols: usize) -> Vec<Vec<f32>> {
        (0..n)
            .map(|r| {
                (0..cols)
                    .map(|j| {
                        let x = (((r * 131 + j * 17) % 997) as f32 / 997.0) - 0.5;
                        if j >= 3 * cols / 4 { x * 8.0 } else { x }
                    })
                    .collect()
            })
            .collect()
    }

    #[test]
    fn appending_a_model_preserves_its_explicit_tokenizer() {
        let dir = std::env::temp_dir().join(format!("mary-packed-tokenizers-{}", fucid().id));
        std::fs::create_dir(&dir).unwrap();
        let out = dir.join("models.pile");
        let key = SigningKey::from_bytes(&[0x4C; 32]);
        let keymap = HashMap::from([("norm.weight".to_owned(), (vec![1.0], vec![1]))]);
        let packed = HashMap::new();
        let first = r#"{"model":{"type":"WordPiece","vocab":{"[UNK]":0,"hello":1},"unk_token":"[UNK]"},"added_tokens":[]}"#;
        let second = first.replace("hello", "world");
        write_packed_pile(
            &out,
            &key,
            &keymap,
            &packed,
            "first",
            "native",
            &[],
            Some(first.as_bytes()),
            false,
        )
        .unwrap();
        write_packed_pile(
            &out,
            &key,
            &keymap,
            &packed,
            "second",
            "native",
            &[],
            Some(second.as_bytes()),
            true,
        )
        .unwrap();
        let snapshot = crate::model_collection::load_model_collection_local_latest(&out).unwrap();
        let first_root = crate::selection::select_tokenizer_root(
            snapshot.facts(),
            snapshot.store(),
            crate::selection::TokenizerSelector::Name("first"),
        )
        .unwrap();
        let second_root = crate::selection::select_tokenizer_root(
            snapshot.facts(),
            snapshot.store(),
            crate::selection::TokenizerSelector::Name("second"),
        )
        .unwrap();
        assert_ne!(first_root, second_root);
        drop(snapshot);
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn packed_root_labels_and_bases_do_not_change_member_identity() {
        let dir = std::env::temp_dir().join(format!(
            "mary-packed-root-{}-{}",
            std::process::id(),
            fucid().id,
        ));
        std::fs::create_dir(&dir).unwrap();
        let out = dir.join("models.pile");
        let key = SigningKey::from_bytes(&[0x4D; 32]);
        let base = fucid().id;
        let other_base = fucid().id;
        let mut keymap = HashMap::from([("norm.weight".to_owned(), (vec![1.0], vec![1]))]);
        let packed = HashMap::new();
        let first = write_packed_pile(
            &out,
            &key,
            &keymap,
            &packed,
            "fixture/model",
            "first",
            &[base],
            None,
            false,
        )
        .unwrap();
        let relabelled = write_packed_pile(
            &out,
            &key,
            &keymap,
            &packed,
            "fixture/alias",
            "second",
            &[other_base],
            None,
            true,
        )
        .unwrap();
        assert_eq!(
            first, relabelled,
            "annotations are not the root's identity core"
        );

        keymap.get_mut("norm.weight").unwrap().0[0] = 2.0;
        let second = write_packed_pile(
            &out,
            &key,
            &keymap,
            &packed,
            "fixture/model",
            "first",
            &[base],
            None,
            true,
        )
        .unwrap();
        assert_ne!(first, second, "a changed member produces a new root");
        let snapshot = crate::model_collection::load_model_collection_local_latest(&out).unwrap();
        let facts = snapshot.facts();
        use crate::format::attrs;
        for root in [first, second] {
            assert!(exists!(
                (),
                pattern!(facts, [{ root @ attrs::parent: base }])
            ));
        }
        assert!(exists!(
            (),
            pattern!(facts, [{ first @ attrs::parent: other_base }])
        ));
        assert!(!exists!(
            (),
            pattern!(facts, [{ second @ attrs::parent: first }])
        ));
        for (root, value) in [(first, 1.0), (second, 2.0)] {
            let read = crate::selection::load_keymap_from_graph(
                facts,
                snapshot.store(),
                ModelSelector::Root(root),
            )
            .unwrap();
            assert_eq!(read["norm.weight"].0, vec![value]);
        }
        drop(snapshot);
        std::fs::remove_dir_all(dir).unwrap();
    }

    /// The whole seam in one test: pack a tiny model, write it as a pile, read
    /// it back through the ordinary keymap loader, and get the decoded packed
    /// values with the input scale folded in and its vector gone.
    #[test]
    fn a_packed_pile_reads_back_through_the_keymap_loader() {
        let dir = std::env::temp_dir().join(format!(
            "mary-calibrate-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or(0)
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let out = dir.join("packed.pile");
        let key = SigningKey::from_bytes(&[0x4C; 32]);

        let (rows, cols) = (8, 32);
        let mut keymap: Keymap = HashMap::new();
        keymap.insert(
            "encoder.layers.0.mlp.fc2.weight".into(),
            (matrix(rows, cols), vec![rows, cols]),
        );
        keymap.insert(
            "encoder.layers.0.norm1.weight".into(),
            (vec![1.0; cols], vec![cols]),
        );
        keymap.insert(
            "embeddings.word_embeddings.weight".into(),
            (matrix(4, cols), vec![4, cols]),
        );
        let x = inputs(32, cols);
        let mut mean_abs = vec![0f32; cols];
        for r in &x {
            for (m, v) in mean_abs.iter_mut().zip(r) {
                *m += v.abs() / x.len() as f32;
            }
        }
        let stats = HashMap::from([(
            "encoder.layers.0.mlp.fc2".to_string(),
            InputStats { rows: x, mean_abs },
        )]);
        let stats_for = |name: &str| stats.get(&nomic_capture_key(name));

        let mut packed_keymap = keymap.clone();
        let report = pack_keymap(
            &mut packed_keymap,
            &[],
            &stats_for,
            &Options::default(),
            &mut |_| {},
        )
        .unwrap();
        assert_eq!(
            report.tensors, 1,
            "the linear packs; the norm and the embedding do not"
        );
        assert_ne!(
            packed_keymap["encoder.layers.0.mlp.fc2.weight"].0,
            keymap["encoder.layers.0.mlp.fc2.weight"].0
        );
        assert_eq!(
            packed_keymap["encoder.layers.0.norm1.weight"],
            keymap["encoder.layers.0.norm1.weight"]
        );

        let parent = fucid().id;
        let root = write_packed_pile(
            &out,
            &key,
            &packed_keymap,
            &report.packed,
            "test/model",
            "nvfp4-calibrated",
            &[parent],
            None,
            false,
        )
        .unwrap();
        assert!(
            write_packed_pile(
                &out,
                &key,
                &packed_keymap,
                &report.packed,
                "test/model",
                "nvfp4-calibrated",
                &[parent],
                None,
                false
            )
            .is_err(),
            "refuses an existing file"
        );

        let snapshot = crate::model_collection::load_model_collection_local_latest(&out).unwrap();
        assert!(exists!(
            (),
            pattern!(snapshot.facts(), [{
                root @ crate::format::attrs::parent: parent,
            }])
        ));
        let back = crate::selection::load_keymap_from_graph(
            snapshot.facts(),
            snapshot.store(),
            ModelSelector::Source {
                source: "test/model",
                quantization: "nvfp4-calibrated",
            },
        )
        .unwrap();
        assert_eq!(
            back.len(),
            3,
            "input_scale folded away, keys {:?}",
            back.keys().collect::<Vec<_>>()
        );
        for (name, (values, shape)) in &packed_keymap {
            let (b, s) = &back[name];
            assert_eq!(s, shape, "{name}");
            assert_eq!(
                b, values,
                "{name}: the loader decodes what the packer scored"
            );
        }
        assert_eq!(
            crate::selection::load_keymap_from_graph(
                snapshot.facts(),
                snapshot.store(),
                ModelSelector::Root(root),
            )
            .unwrap()
            .len(),
            3
        );
        let _ = std::fs::remove_dir_all(&dir);
    }
}
