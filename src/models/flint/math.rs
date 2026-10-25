//! F32 values and operation boundaries, with f64 scalar dot/RMS reductions.
//! This deliberately independent scalar math is not GGML's SIMD reduction
//! order, BF16 accumulation, W4A4/W4A8 activation path, or a performance kernel.
//! Equations: pinned llama.cpp 1ddffe9 flint.cpp and llama-graph.cpp.

use anyhow::{Result, ensure};
use super::weights::Weights;

pub const H: usize = 2048;
const D: usize = 128;
const Q: usize = 32;
const KV: usize = 4;
const E: usize = 256;
const K: usize = 6;
const F: usize = 512;
const SH: usize = 1024;
const EPS: f32 = 1e-6;

pub enum Values { F32(Vec<f32>), I32(Vec<i32>) }
pub struct Stage { pub name: String, pub dims: [usize; 4], pub values: Values }
#[derive(Default)]
pub struct Trace { pub stages: Vec<Stage>, pub router_margins: Vec<f32> }
impl Trace {
    fn f(&mut self, name: &str, dims: [usize; 4], x: &[f32]) {
        assert_eq!(dims.iter().product::<usize>(), x.len());
        self.stages.push(Stage { name: name.into(), dims, values: Values::F32(x.to_vec()) });
    }
    fn i(&mut self, name: &str, dims: [usize; 4], x: &[i32]) {
        assert_eq!(dims.iter().product::<usize>(), x.len());
        self.stages.push(Stage { name: name.into(), dims, values: Values::I32(x.to_vec()) });
    }
}

fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(&x, &y)| (x as f64) * (y as f64)).sum::<f64>() as f32
}
fn matvec(w: &[f32], input: &[f32], width: usize) -> Vec<f32> {
    w.chunks_exact(width).map(|row| dot(row, input)).collect()
}
fn norm(x: &[f32], weight: &[f32]) -> Vec<f32> {
    x.chunks_exact(weight.len()).flat_map(|row| {
        let ss = row.iter().map(|&v| (v as f64) * (v as f64)).sum::<f64>();
        let inv = ((ss / row.len() as f64 + EPS as f64).sqrt().recip()) as f32;
        row.iter().zip(weight).map(move |(&v, &w)| (v * inv) * w)
    }).collect()
}
fn sigmoid(x: f32) -> f32 { 1.0 / (1.0 + (-x).exp()) }
fn swiglu(gate: &[f32], up: &[f32]) -> Vec<f32> {
    gate.iter().zip(up).map(|(&g, &u)| (g * sigmoid(g)) * u).collect()
}
fn add(a: &[f32], b: &[f32]) -> Vec<f32> { a.iter().zip(b).map(|(&a, &b)| a + b).collect() }

fn dense(w: &mut impl Weights, tr: &mut Trace, suffix: &str, label: &str,
    x: &[f32], ni: usize, no: usize, t: usize) -> Result<Vec<f32>> {
    let weight = w.get(&format!("{suffix}.weight"), &[ni, no], 40, None)?;
    let scale = w.get(&format!("{suffix}.scale"), &[1], 0, None)?[0];
    let raw: Vec<f32> = x.chunks_exact(ni).flat_map(|row| matvec(&weight, row, ni)).collect();
    tr.f(&format!("{label}_unscaled"), [no,t,1,1], &raw);
    let out: Vec<f32> = raw.iter().map(|&v| v * scale).collect();
    tr.f(label, [no,t,1,1], &out);
    Ok(out)
}

fn rope(x: &[f32], heads: usize, positions: &[i32]) -> Vec<f32> {
    let mut out = x.to_vec();
    for (t, &pos) in positions.iter().enumerate() {
        for h in 0..heads {
            for j in 0..D / 2 {
                let angle = pos as f32 * 10000.0f32.powf(-2.0 * j as f32 / D as f32);
                let (sin, cos) = angle.sin_cos();
                let i = (t * heads + h) * D + j;
                out[i] = x[i] * cos - x[i + D/2] * sin;
                out[i + D/2] = x[i] * sin + x[i + D/2] * cos;
            }
        }
    }
    out
}

fn route(prob: &[f32]) -> (Vec<usize>, Vec<f32>, f32, f32) {
    let mut ids: Vec<usize> = (0..prob.len()).collect();
    // Explicit deterministic tie policy; real-input margins are reported.
    ids.sort_by(|&a, &b| prob[b].total_cmp(&prob[a]).then(a.cmp(&b)));
    let margin = prob[ids[K-1]] - prob[ids[K]];
    let sum: f32 = ids[..K].iter().map(|&i| prob[i]).sum();
    let denom = sum.max(6.103515625e-5);
    let weights = ids[..K].iter().map(|&i| prob[i] / denom).collect();
    (ids[..K].to_vec(), weights, sum, margin)
}

fn down_scale(x: &[f32]) -> f32 {
    let amax = x.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
    (((amax / 1024.0).ln() * std::f32::consts::LOG2_E).ceil().clamp(0.0,127.0)
        * std::f32::consts::LN_2).exp()
}

/// Exactly one layer, 1..=3 tokens, no KV history. All positions must be
/// nonnegative and strictly increasing. Noncontiguous positions are supported.
pub fn layer2(w: &mut impl Weights, input: &[f32], positions: &[i32]) -> Result<Trace> {
    let t = positions.len();
    ensure!((1..=3).contains(&t) && input.len() == H*t, "expected 1..=3 residual rows of width 2048");
    ensure!(positions.iter().all(|&p| p >= 0) && positions.windows(2).all(|p| p[0] < p[1]), "invalid positions");
    ensure!(input.iter().all(|v| v.is_finite()), "non-finite residual input");
    let mut tr = Trace::default();
    tr.f("input", [H,t,1,1], input);
    tr.i("positions", [t,1,1,1], positions);
    let an = w.get("attn_norm.weight", &[H], 0, None)?;
    let n = norm(input, &an);
    tr.f("attn_norm", [H,t,1,1], &n);
    let q = dense(w,&mut tr,"attn_q","q_proj",&n,H,Q*D,t)?;
    let k = dense(w,&mut tr,"attn_k","k_proj",&n,H,KV*D,t)?;
    let v = dense(w,&mut tr,"attn_v","v_proj",&n,H,KV*D,t)?;
    let qn = norm(&q,&w.get("attn_q_norm.weight", &[D],0,None)?);
    let kn = norm(&k,&w.get("attn_k_norm.weight", &[D],0,None)?);
    tr.f("q_norm",[D,Q,t,1],&qn); tr.f("k_norm",[D,KV,t,1],&kn);
    let q = rope(&qn,Q,positions); let k = rope(&kn,KV,positions);
    tr.f("q_rope",[D,Q,t,1],&q); tr.f("k_rope",[D,KV,t,1],&k);
    let sinks = w.get("attn_sinks.weight", &[Q],0,None)?;
    let mut scores = vec![0.0;t*t*Q]; let mut probs = scores.clone();
    let mut mask = vec![0.0;t*t]; let mut heads = vec![0.0;D*t*Q];
    for query in 0..t { for key in 0..t {
        if positions[key] > positions[query] || positions[query] - positions[key] >= 513 { mask[query*t+key] = f32::NEG_INFINITY; }
    }}
    for h in 0..Q { for query in 0..t {
        let base = (h*t+query)*t;
        let mut logits = vec![0.0;t];
        for key in 0..t {
            scores[base+key] = dot(&q[(query*Q+h)*D..(query*Q+h+1)*D], &k[(key*KV+h/(Q/KV))*D..(key*KV+h/(Q/KV)+1)*D]);
            logits[key] = scores[base+key] * (D as f32).sqrt().recip() + mask[query*t+key];
        }
        let max = logits.iter().copied().fold(sinks[h],f32::max);
        let exps: Vec<f32> = logits.iter().map(|&v| (v-max).exp()).collect();
        let denom = exps.iter().map(|&v| v as f64).sum::<f64>() + ((sinks[h]-max).exp() as f64);
        for key in 0..t { probs[base+key] = (exps[key] as f64 / denom) as f32; }
        for d in 0..D {
            heads[(h*t+query)*D+d] = (0..t).map(|key| (probs[base+key] as f64) * (v[(key*KV+h/(Q/KV))*D+d] as f64)).sum::<f64>() as f32;
        }
    }}
    tr.f("attention_scores",[t,t,Q,1],&scores); tr.f("attention_mask",[t,t,1,1],&mask);
    tr.f("attention_probs",[t,t,Q,1],&probs); tr.f("attention_heads",[D,t,Q,1],&heads);
    let mut joined = vec![0.0;Q*D*t];
    for token in 0..t { for h in 0..Q { for d in 0..D { joined[(token*Q+h)*D+d] = heads[(h*t+token)*D+d]; }}}
    let ao = dense(w,&mut tr,"attn_output","attention_output",&joined,Q*D,H,t)?;
    let residual = add(input,&ao); tr.f("ffn_inp",[H,t,1,1],&residual);
    let n = norm(&residual,&w.get("ffn_norm.weight",&[H],0,None)?);
    tr.f("ffn_norm",[H,t,1,1],&n);
    let router = w.get("ffn_gate_inp.weight",&[H,E],30,None)?;
    let logits: Vec<f32> = n.chunks_exact(H).flat_map(|row|matvec(&router,row,H)).collect();
    let prob: Vec<f32> = logits.iter().map(|&v|sigmoid(v)).collect();
    tr.f("router_logits",[E,t,1,1],&logits); tr.f("router_probs",[E,t,1,1],&prob);
    let mut ids = Vec::new(); let mut rw = Vec::new(); let mut sums = Vec::new(); let mut selected = Vec::new();
    for row in prob.chunks_exact(E) {
        let (ii,ww,sum,margin) = route(row);
        selected.extend(ii.iter().map(|&i|row[i])); ids.extend(ii.iter().map(|&i|i as i32)); rw.extend(ww); sums.push(sum); tr.router_margins.push(margin);
    }
    tr.i("router_ids",[K,t,1,1],&ids); tr.f("router_selected_probs",[1,K,t,1],&selected);
    tr.f("router_sum",[1,t,1,1],&sums); tr.f("router_denominator",[1,t,1,1],&sums.iter().map(|&s|s.max(6.103515625e-5)).collect::<Vec<_>>());
    tr.f("router_weights",[1,K,t,1],&rw);
    let up = experts(w,&mut tr,"ffn_up_exps","expert_up",&n,H,F,t,&ids,false)?;
    let gate = experts(w,&mut tr,"ffn_gate_exps","expert_gate",&n,H,F,t,&ids,false)?;
    let sw = swiglu(&gate,&up); tr.f("expert_swiglu",[F,K,t,1],&sw);
    let scales: Vec<f32> = sw.chunks_exact(F).map(down_scale).collect();
    let di: Vec<f32> = sw.chunks_exact(F).zip(&scales).flat_map(|(row,&s)|row.iter().map(move |&x|x/s)).collect();
    tr.f("expert_down_scale",[1,K,t,1],&scales); tr.f("expert_down_input",[F,K,t,1],&di);
    let before = experts(w,&mut tr,"ffn_down_exps","expert_down_before_rescale",&di,F,H,t,&ids,true)?;
    let down: Vec<f32> = before.chunks_exact(H).zip(&scales).flat_map(|(row,&s)|row.iter().map(move |&x|x*s)).collect();
    tr.f("expert_down",[H,K,t,1],&down);
    let weighted: Vec<f32> = down.chunks_exact(H).zip(&rw).flat_map(|(row,&s)|row.iter().map(move |&x|x*s)).collect();
    tr.f("expert_weighted",[H,K,t,1],&weighted);
    let mut moe = vec![0.0;H*t];
    for token in 0..t { for d in 0..H { let mut sum = weighted[token*K*H+d]; for slot in 1..K { sum += weighted[(token*K+slot)*H+d]; } moe[token*H+d] = sum; }}
    tr.f("moe_out",[H,t,1,1],&moe);
    let su = dense(w,&mut tr,"ffn_up_shexp","shared_up",&n,H,SH,t)?;
    let sg = dense(w,&mut tr,"ffn_gate_shexp","shared_gate",&n,H,SH,t)?;
    let ss = swiglu(&sg,&su); tr.f("shared_swiglu",[SH,t,1,1],&ss);
    let sd = dense(w,&mut tr,"ffn_down_shexp","shared_down",&ss,SH,H,t)?;
    // Shared expert is ungated and outside routed-top-six normalization.
    let ffn = add(&moe,&sd); tr.f("ffn_out",[H,t,1,1],&ffn);
    tr.f("l_out",[H,t,1,1],&add(&ffn,&residual));
    Ok(tr)
}

fn experts(w: &mut impl Weights, tr: &mut Trace, suffix: &str, label: &str,
    x: &[f32], ni: usize, no: usize, t: usize, ids: &[i32], per_slot: bool) -> Result<Vec<f32>> {
    let all_scales = w.get(&format!("{suffix}.scale"), &[E],0,None)?;
    let mut raw = Vec::new(); let mut selected = Vec::new(); let mut out = Vec::new();
    for (slot,&id) in ids.iter().enumerate() {
        let weight = w.get(&format!("{suffix}.weight"), &[ni,no,E],40,Some(id as usize))?;
        let row = if per_slot { slot } else { slot/K };
        let r = matvec(&weight,&x[row*ni..(row+1)*ni],ni);
        let s = all_scales[id as usize]; selected.push(s);
        out.extend(r.iter().map(|&v|v*s)); raw.extend(r);
    }
    tr.f(&format!("{label}_unscaled"),[no,K,t,1],&raw);
    tr.f(&format!("{label}_selected_scale"),[1,K,t,1],&selected);
    tr.f(label,[no,K,t,1],&out);
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test] fn norm_rope_and_swiglu() {
        assert_eq!(norm(&[0.0;4],&[1.0;4]),[0.0;4]);
        let x: Vec<f32> = (0..D).map(|i|i as f32).collect();
        assert_eq!(rope(&x,1,&[0]),x);
        let r = rope(&x,1,&[1]);
        assert!((r[0] + x[64]*1.0f32.sin()).abs() < 1e-5);
        assert_eq!(swiglu(&[0.0],&[10.0]),[0.0]);
    }
    #[test] fn routing_clamp_and_ties() {
        let (ids,w,sum,margin) = route(&[0.0;E]);
        assert_eq!(ids,[0,1,2,3,4,5]); assert_eq!(w,[0.0;6]); assert_eq!(sum,0.0); assert_eq!(margin,0.0);
        let (_,w,_,_) = route(&[0.5;E]); assert!((w.iter().sum::<f32>()-1.0).abs() < 1e-6);
    }
    #[test] fn adaptive_down_scale_boundaries() {
        assert_eq!(down_scale(&[0.0]),1.0); assert_eq!(down_scale(&[1024.0]),1.0);
        assert!((down_scale(&[1025.0])-2.0).abs() < 1e-6);
        assert!((down_scale(&[-4096.0])-4.0).abs() < 1e-6);
    }
    struct NoReads;
    impl Weights for NoReads {
        fn get(&mut self,_:&str,_:&[usize],_:u32,_:Option<usize>)->Result<Vec<f32>> { panic!("invalid input must not read weights") }
        fn get_nvfp4(&mut self,_:&str,_:&[usize],_:Option<usize>)->Result<Vec<u8>> { panic!("invalid input must not read weights") }
    }
    #[test] fn rejects_invalid_inputs_before_weight_reads() {
        assert!(layer2(&mut NoReads,&[],&[]).is_err());
        assert!(layer2(&mut NoReads,&[0.0;H],&[-1]).is_err());
        assert!(layer2(&mut NoReads,&[0.0;H*2],&[1,1]).is_err());
        assert!(layer2(&mut NoReads,&[f32::NAN;H],&[0]).is_err());
    }
}
