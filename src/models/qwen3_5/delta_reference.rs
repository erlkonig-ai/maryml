//! Scalar correctness oracle only; this is not a model runtime or GPU kernel.
//!
//! Source: transformers 5.2.0, models/qwen3_5/modeling_qwen3_5.py:
//! l2norm lines 317–320, chunk rule 323–400, recurrent rule 403–442,
//! and grouped-head repeat_interleave at 587–589. Inspected installed source
//! SHA256: d1ae3856f53763591ec65054129af46e003a1715efcf7eef2b752a31e85526b8.
//! Upstream release commit: 7d9754a05193eb79b1d86aa744b622b8068008cd.
//!
//! Q/K are contiguous [batch, time, key_heads, key_dim]; V and output are
//! [batch, time, value_heads, value_dim]. Gates are [batch, time, value_heads].
//! State is contiguous [batch, value_heads, key_dim, value_dim], with value_dim
//! varying fastest. Adjacent value heads share one Q/K head (repeat_interleave),
//! rather than alternating Q/K heads. `log_decay` is the already-computed g,
//! and `beta` is the already-sigmoided update gate: this oracle starts AFTER
//! projection, causal convolution, and SiLU, and ends BEFORE gated RMSNorm.
//!
//! Everything here is f32, including retained recurrent state. HF normalizes
//! Q/K in their incoming dtype before promoting recurrence arithmetic to f32;
//! a BF16 gate must therefore pass pre-normalized/promoted Q/K with normalization
//! disabled and cast output back to BF16. This f32 oracle does not emulate BF16
//! normalization. Scalar, Torch-reduction, and chunked evaluation may associate
//! sums differently: test with tolerances, not bit equality. Resident GPU
//! recurrence, convolution state, masking, and full-model equivalence remain
//! separate implementation work.

#[derive(Clone, Copy, Debug)]
pub struct Shape {
    pub batch: usize,
    pub steps: usize,
    pub key_heads: usize,
    pub value_heads: usize,
    pub key_dim: usize,
    pub value_dim: usize,
}

pub struct DeltaInput<'a> {
    pub shape: Shape,
    pub query: &'a [f32],
    pub key: &'a [f32],
    pub value: &'a [f32],
    pub log_decay: &'a [f32],
    pub beta: &'a [f32],
    pub initial_state: Option<&'a [f32]>,
    pub normalize_qk: bool,
}

#[derive(Debug)]
pub struct DeltaOutput {
    pub output: Vec<f32>,
    pub final_state: Vec<f32>,
}

fn size(dims: &[usize]) -> Result<usize, String> {
    dims.iter().try_fold(1usize, |n, d| {
        n.checked_mul(*d).ok_or_else(|| "shape product overflow".to_owned())
    })
}

fn check(name: &str, values: &[f32], expected: usize) -> Result<(), String> {
    if values.len() != expected {
        return Err(format!("{name}: expected {expected} values, got {}", values.len()));
    }
    if values.iter().any(|v| !v.is_finite()) {
        return Err(format!("{name}: inputs must be finite"));
    }
    Ok(())
}

/// Evaluate the recurrent DeltaNet rule with zero or explicitly supplied state.
/// Zero time steps preserve the initial state and return an empty output.
pub fn recurrent(input: DeltaInput<'_>) -> Result<DeltaOutput, String> {
    let s = input.shape;
    if [s.batch, s.key_heads, s.value_heads, s.key_dim, s.value_dim].contains(&0)
        || s.value_heads % s.key_heads != 0
    {
        return Err("nonzero dimensions and value_heads divisible by key_heads required".into());
    }
    let qk_size = size(&[s.batch, s.steps, s.key_heads, s.key_dim])?;
    let value_size = size(&[s.batch, s.steps, s.value_heads, s.value_dim])?;
    let gate_size = size(&[s.batch, s.steps, s.value_heads])?;
    let state_size = size(&[s.batch, s.value_heads, s.key_dim, s.value_dim])?;
    check("query", input.query, qk_size)?;
    check("key", input.key, qk_size)?;
    check("value", input.value, value_size)?;
    check("log_decay", input.log_decay, gate_size)?;
    check("beta", input.beta, gate_size)?;
    let mut state = if let Some(initial) = input.initial_state {
        check("initial_state", initial, state_size)?;
        initial.to_vec()
    } else {
        vec![0.0; state_size]
    };
    let mut output = vec![0.0; value_size];
    let repeats = s.value_heads / s.key_heads;
    let scale = 1.0f32 / (s.key_dim as f32).sqrt();
    let mut query = vec![0.0f32; s.key_dim];
    let mut key = vec![0.0f32; s.key_dim];
    let mut delta = vec![0.0f32; s.value_dim];

    for batch in 0..s.batch {
        for step in 0..s.steps {
            for head in 0..s.value_heads {
                let key_head = head / repeats;
                let qk = ((batch * s.steps + step) * s.key_heads + key_head) * s.key_dim;
                query.copy_from_slice(&input.query[qk..qk + s.key_dim]);
                key.copy_from_slice(&input.key[qk..qk + s.key_dim]);
                if input.normalize_qk {
                    // HF uses rsqrt(sum(x*x) + eps), not max(norm, eps).
                    for vector in [&mut query, &mut key] {
                        let norm_squared: f32 = vector.iter().map(|x| x * x).sum();
                        let inverse_norm = 1.0f32 / (norm_squared + 1e-6f32).sqrt();
                        for x in vector {
                            *x *= inverse_norm;
                        }
                    }
                }
                for x in &mut query {
                    *x *= scale;
                }
                let gate = (batch * s.steps + step) * s.value_heads + head;
                let value_offset = gate * s.value_dim;
                let state_offset = (batch * s.value_heads + head) * s.key_dim * s.value_dim;
                let decay = input.log_decay[gate].exp();
                for x in &mut state[state_offset..state_offset + s.key_dim * s.value_dim] {
                    *x *= decay;
                }
                // Prediction observes the decayed state. Form every delta before
                // adding the outer product, then query the newly updated state.
                for value_dim in 0..s.value_dim {
                    let mut prediction = 0.0f32;
                    for key_dim in 0..s.key_dim {
                        prediction += state[state_offset + key_dim * s.value_dim + value_dim] * key[key_dim];
                    }
                    delta[value_dim] = (input.value[value_offset + value_dim] - prediction) * input.beta[gate];
                }
                for key_dim in 0..s.key_dim {
                    for value_dim in 0..s.value_dim {
                        let index = state_offset + key_dim * s.value_dim + value_dim;
                        state[index] += key[key_dim] * delta[value_dim];
                    }
                }
                for value_dim in 0..s.value_dim {
                    let mut result = 0.0f32;
                    for key_dim in 0..s.key_dim {
                        result += state[state_offset + key_dim * s.value_dim + value_dim] * query[key_dim];
                    }
                    output[value_offset + value_dim] = result;
                }
            }
        }
    }
    Ok(DeltaOutput { output, final_state: state })
}

// Standalone fixture runner, deliberately independent of Cargo/model wiring.
// stdin: B T Hk Hv K V normalize(0/1) initial(0/1), then flattened Q K V g beta
// and optional initial state. stdout: `output ...` and `state ...` float lines.
#[cfg(not(test))]
fn main() {
    if let Err(error) = run_stdin() {
        eprintln!("delta_reference: {error}");
        std::process::exit(1);
    }
}

#[cfg(not(test))]
fn run_stdin() -> Result<(), String> {
    use std::io::{Read, Write};
    let mut text = String::new();
    std::io::stdin().read_to_string(&mut text).map_err(|e| e.to_string())?;
    let mut words = text.split_whitespace();
    fn integer(words: &mut std::str::SplitWhitespace<'_>) -> Result<usize, String> {
        words.next().ok_or("missing shape/flag")?.parse().map_err(|_| "invalid shape/flag".into())
    }
    fn values(words: &mut std::str::SplitWhitespace<'_>, n: usize) -> Result<Vec<f32>, String> {
        (0..n).map(|_| words.next().ok_or("missing tensor value")?.parse().map_err(|_| "invalid float".into())).collect()
    }
    let shape = Shape {
        batch: integer(&mut words)?, steps: integer(&mut words)?,
        key_heads: integer(&mut words)?, value_heads: integer(&mut words)?,
        key_dim: integer(&mut words)?, value_dim: integer(&mut words)?,
    };
    let normalize = integer(&mut words)?;
    let initial = integer(&mut words)?;
    if normalize > 1 || initial > 1 {
        return Err("flags must be 0 or 1".into());
    }
    let qk = size(&[shape.batch, shape.steps, shape.key_heads, shape.key_dim])?;
    let v = size(&[shape.batch, shape.steps, shape.value_heads, shape.value_dim])?;
    let gates = size(&[shape.batch, shape.steps, shape.value_heads])?;
    let state_len = size(&[shape.batch, shape.value_heads, shape.key_dim, shape.value_dim])?;
    let query = values(&mut words, qk)?;
    let key = values(&mut words, qk)?;
    let value = values(&mut words, v)?;
    let log_decay = values(&mut words, gates)?;
    let beta = values(&mut words, gates)?;
    let initial_state = if initial == 1 { Some(values(&mut words, state_len)?) } else { None };
    if words.next().is_some() {
        return Err("unexpected trailing input".into());
    }
    let result = recurrent(DeltaInput {
        shape, query: &query, key: &key, value: &value, log_decay: &log_decay,
        beta: &beta, initial_state: initial_state.as_deref(), normalize_qk: normalize == 1,
    })?;
    let mut out = std::io::BufWriter::new(std::io::stdout().lock());
    for (name, vector) in [("output", result.output), ("state", result.final_state)] {
        write!(out, "{name}").map_err(|e| e.to_string())?;
        for x in vector {
            write!(out, " {x:.9e}").map_err(|e| e.to_string())?;
        }
        writeln!(out).map_err(|e| e.to_string())?;
    }
    out.flush().map_err(|e| e.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(actual: &[f32], expected: &[f32]) {
        assert_eq!(actual.len(), expected.len());
        for (index, (a, e)) in actual.iter().zip(expected).enumerate() {
            assert!((a - e).abs() <= 2e-6 + 2e-6 * e.abs(), "index {index}: {a} != {e}");
        }
    }

    fn one() -> Shape {
        Shape { batch: 1, steps: 1, key_heads: 1, value_heads: 1, key_dim: 1, value_dim: 1 }
    }

    #[test]
    fn nonzero_state_decays_before_prediction_and_update() {
        // S=2 -> decay to 1; prediction=3; delta=(5-3)/4=.5;
        // S'=1+3*.5=2.5; output=2*2.5=5.
        let result = recurrent(DeltaInput {
            shape: one(), query: &[2.0], key: &[3.0], value: &[5.0],
            log_decay: &[-std::f32::consts::LN_2], beta: &[0.25],
            initial_state: Some(&[2.0]), normalize_qk: false,
        }).unwrap();
        close(&result.output, &[5.0]);
        close(&result.final_state, &[2.5]);
    }

    #[test]
    fn grouped_heads_repeat_adjacent_not_cyclic() {
        let shape = Shape { key_heads: 2, value_heads: 4, ..one() };
        let result = recurrent(DeltaInput {
            shape, query: &[1.0, 10.0], key: &[1.0, 2.0], value: &[1.0, 2.0, 3.0, 4.0],
            log_decay: &[0.0; 4], beta: &[1.0; 4], initial_state: None, normalize_qk: false,
        }).unwrap();
        close(&result.output, &[1.0, 2.0, 60.0, 80.0]);
        close(&result.final_state, &[1.0, 2.0, 6.0, 8.0]);
    }

    #[test]
    fn l2_epsilon_is_inside_root_and_query_is_scaled() {
        let shape = Shape { key_dim: 2, ..one() };
        let result = recurrent(DeltaInput {
            shape, query: &[0.0, 0.001], key: &[0.0, 0.001], value: &[2.0],
            log_decay: &[0.0], beta: &[1.0], initial_state: None, normalize_qk: true,
        }).unwrap();
        close(&result.output, &[1.0 / 2.0f32.sqrt()]);
        close(&result.final_state, &[0.0, 2.0f32.sqrt()]);
    }

    #[test]
    fn zero_keys_and_zero_beta_leave_only_decay() {
        for (key, beta) in [([0.0, 0.0], 1.0), ([3.0, 4.0], 0.0)] {
            let result = recurrent(DeltaInput {
                shape: Shape { key_dim: 2, value_dim: 2, ..one() },
                query: &[0.0, 0.0], key: &key, value: &[10.0, -20.0],
                log_decay: &[-std::f32::consts::LN_2], beta: &[beta],
                initial_state: Some(&[2.0, 4.0, 6.0, 8.0]), normalize_qk: true,
            }).unwrap();
            close(&result.output, &[0.0, 0.0]);
            close(&result.final_state, &[1.0, 2.0, 3.0, 4.0]);
        }
    }

    #[test]
    fn split_sequence_preserves_batch_head_and_state_layout() {
        let shape = Shape { batch: 2, steps: 5, key_heads: 2, value_heads: 4, key_dim: 3, value_dim: 2 };
        let data = |n: usize, shift: usize| -> Vec<f32> {
            (0..n).map(|i| ((i + shift) % 13) as f32 / 13.0 - 0.5).collect()
        };
        let query = data(60, 0);
        let key = data(60, 3);
        let value = data(80, 7);
        let log_decay = vec![-0.2; 40];
        let beta = vec![0.3; 40];
        let initial = data(48, 4);
        let whole = recurrent(DeltaInput {
            shape, query: &query, key: &key, value: &value, log_decay: &log_decay,
            beta: &beta, initial_state: Some(&initial), normalize_qk: true,
        }).unwrap();
        let slice = |x: &[f32], width: usize, start: usize, stop: usize| -> Vec<f32> {
            (0..shape.batch).flat_map(|b| {
                x[(b * shape.steps + start) * width..(b * shape.steps + stop) * width].iter().copied()
            }).collect()
        };
        let mut state = initial;
        let mut joined = vec![0.0; whole.output.len()];
        for (start, stop) in [(0, 2), (2, 5)] {
            let part = recurrent(DeltaInput {
                shape: Shape { steps: stop - start, ..shape },
                query: &slice(&query, 6, start, stop), key: &slice(&key, 6, start, stop),
                value: &slice(&value, 8, start, stop), log_decay: &slice(&log_decay, 4, start, stop),
                beta: &slice(&beta, 4, start, stop), initial_state: Some(&state), normalize_qk: true,
            }).unwrap();
            for b in 0..shape.batch {
                joined[(b * shape.steps + start) * 8..(b * shape.steps + stop) * 8]
                    .copy_from_slice(&part.output[b * (stop - start) * 8..(b + 1) * (stop - start) * 8]);
            }
            state = part.final_state;
        }
        close(&joined, &whole.output);
        close(&state, &whole.final_state);
    }

    #[test]
    fn empty_sequence_preserves_state_and_invalid_groups_are_rejected() {
        let empty = recurrent(DeltaInput {
            shape: Shape { steps: 0, ..one() }, query: &[], key: &[], value: &[],
            log_decay: &[], beta: &[], initial_state: Some(&[3.0]), normalize_qk: true,
        }).unwrap();
        assert!(empty.output.is_empty());
        close(&empty.final_state, &[3.0]);
        assert!(recurrent(DeltaInput {
            shape: Shape { key_heads: 2, value_heads: 3, ..one() }, query: &[], key: &[],
            value: &[], log_decay: &[], beta: &[], initial_state: None, normalize_qk: true,
        }).is_err());
    }
}
