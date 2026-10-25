//! F32 values and operation boundaries, with f64 scalar dot/RMS reductions.
//! This deliberately independent scalar math is not GGML's SIMD reduction
//! order, BF16 accumulation, W4A4/W4A8 activation path, or a performance kernel.
//! Equations: pinned llama.cpp 1ddffe9 flint.cpp and llama-graph.cpp.

use anyhow::{Result, ensure};
use super::weights::Weights;
use super::config::{BlockConfig, FfnKind};

pub const H: usize = 2048;
const D: usize = 128;
const Q: usize = 32;
const KV: usize = 4;
pub(super) const E: usize = 256;
const K: usize = 6;
pub(super) const F: usize = 512;
pub(super) const SH: usize = 1024;
const EPS: f32 = 1e-6;

pub enum Values { F32(Vec<f32>), I32(Vec<i32>) }
pub struct Stage { pub name: String, pub dims: [usize; 4], pub values: Values }
pub struct Trace { pub stages: Vec<Stage>, pub router_margins: Vec<f32>, pub(super) enabled: bool }
impl Default for Trace { fn default()->Self { Self::new(true) } }
impl Trace {
    pub(super) fn new(enabled:bool)->Self { Self {stages:Vec::new(),router_margins:Vec::new(),enabled} }
    fn f(&mut self, name: &str, dims: [usize; 4], x: &[f32]) {
        assert_eq!(dims.iter().product::<usize>(), x.len());
        if !self.enabled { return; }
        self.stages.push(Stage { name: name.into(), dims, values: Values::F32(x.to_vec()) });
    }
    fn i(&mut self, name: &str, dims: [usize; 4], x: &[i32]) {
        assert_eq!(dims.iter().product::<usize>(), x.len());
        if !self.enabled { return; }
        self.stages.push(Stage { name: name.into(), dims, values: Values::I32(x.to_vec()) });
    }
}
pub(super) struct BlockOutput { pub residual:Vec<f32>, pub trace:Trace }

fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(&x, &y)| (x as f64) * (y as f64)).sum::<f64>() as f32
}
pub(super) fn matvec(w: &[f32], input: &[f32], width: usize) -> Vec<f32> {
    w.chunks_exact(width).map(|row| dot(row, input)).collect()
}
pub(super) fn norm(x: &[f32], weight: &[f32]) -> Vec<f32> {
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

fn dense_ffn(w: &mut impl Weights, tr: &mut Trace, n: &[f32], hidden: usize,
    width: usize, t: usize) -> Result<Vec<f32>> {
    let up = dense(w,tr,"ffn_up","dense_up",n,hidden,width,t)?;
    let gate = dense(w,tr,"ffn_gate","dense_gate",n,hidden,width,t)?;
    // Pinned Flint loader initializes ordinary-FFN clamp to zero. Parallel
    // gating consumes the same normalized residual, with post-matmul scales.
    let sw = swiglu(&gate,&up);
    tr.f("dense_swiglu",[width,t,1,1],&sw);
    // Unlike routed expert down, there is no adaptive input normalization.
    tr.f("dense_down_input",[width,t,1,1],&sw);
    dense(w,tr,"ffn_down","dense_down",&sw,width,hidden,t)
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
    block(w, input, positions, BlockConfig::LAYER2)
}

pub fn block(w: &mut impl Weights, input: &[f32], positions: &[i32], config: BlockConfig) -> Result<Trace> {
    Ok(block_core(w,input,positions,config,None,true)?.trace)
}

pub(super) fn block_core(w:&mut impl Weights,input:&[f32],positions:&[i32],config:BlockConfig,
    cache:Option<&mut super::cache::CpuKv>,trace:bool)->Result<BlockOutput> {
    let t = positions.len();
    ensure!((1..=3).contains(&t) && input.len() == H*t, "expected 1..=3 residual rows of width 2048");
    ensure!(positions.iter().all(|&p| p >= 0) && positions.windows(2).all(|p| p[0] < p[1]), "invalid positions");
    ensure!(input.iter().all(|v| v.is_finite()), "non-finite residual input");
    ensure!(config.sliding_window.is_none_or(|w|w>0),"sliding window must be positive");
    let mut tr = Trace::new(trace);
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
    let q = if config.rope { rope(&qn,Q,positions) } else { qn };
    let k = if config.rope { rope(&kn,KV,positions) } else { kn };
    tr.f(if config.rope { "q_rope" } else { "q_no_rope" },[D,Q,t,1],&q);
    tr.f(if config.rope { "k_rope" } else { "k_no_rope" },[D,KV,t,1],&k);
    let (k,v,key_positions)=if let Some(c)=cache {
        c.append(&k,&v,positions)?; (c.k.as_slice(),c.v.as_slice(),c.positions.as_slice())
    } else { (k.as_slice(),v.as_slice(),positions) };
    let nt=key_positions.len();
    let geometry=super::cache::Geometry::new(t,nt)?;
    let sinks = w.get("attn_sinks.weight", &[Q],0,None)?;
    let mut scores = super::cache::zeros(geometry.scores)?;
    let mut probs = super::cache::zeros(geometry.scores)?;
    let mut mask = super::cache::zeros(geometry.mask)?; let mut heads = vec![0.0;D*t*Q];
    for query in 0..t { for key in 0..nt {
        if !super::cache::visible(positions[query],key_positions[key],config.sliding_window) { mask[query*nt+key] = f32::NEG_INFINITY; }
    }}
    for h in 0..Q { for query in 0..t {
        let base = (h*t+query)*nt;
        let mut logits = super::cache::zeros(nt)?;
        for key in 0..nt {
            scores[base+key] = dot(&q[(query*Q+h)*D..(query*Q+h+1)*D], &k[(key*KV+h/(Q/KV))*D..(key*KV+h/(Q/KV)+1)*D]);
            logits[key] = scores[base+key] * (D as f32).sqrt().recip() + mask[query*nt+key];
        }
        let max = logits.iter().copied().fold(sinks[h],f32::max);
        let exps: Vec<f32> = logits.iter().map(|&v| (v-max).exp()).collect();
        let denom = exps.iter().map(|&v| v as f64).sum::<f64>() + ((sinks[h]-max).exp() as f64);
        for key in 0..nt { probs[base+key] = (exps[key] as f64 / denom) as f32; }
        for d in 0..D {
            heads[(h*t+query)*D+d] = (0..nt).map(|key| (probs[base+key] as f64) * (v[(key*KV+h/(Q/KV))*D+d] as f64)).sum::<f64>() as f32;
        }
    }}
    tr.f("attention_scores",[nt,t,Q,1],&scores); tr.f("attention_mask",[nt,t,1,1],&mask);
    tr.f("attention_probs",[nt,t,Q,1],&probs); tr.f("attention_heads",[D,t,Q,1],&heads);
    let mut joined = vec![0.0;Q*D*t];
    for token in 0..t { for h in 0..Q { for d in 0..D { joined[(token*Q+h)*D+d] = heads[(h*t+token)*D+d]; }}}
    let ao = dense(w,&mut tr,"attn_output","attention_output",&joined,Q*D,H,t)?;
    let residual = add(input,&ao); tr.f("ffn_inp",[H,t,1,1],&residual);
    let n = norm(&residual,&w.get("ffn_norm.weight",&[H],0,None)?);
    tr.f("ffn_norm",[H,t,1,1],&n);
    let ffn = if let FfnKind::Dense { width } = config.ffn {
        dense_ffn(w,&mut tr,&n,H,width,t)?
    } else {
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
        add(&moe,&sd)
    };
    tr.f("ffn_out",[H,t,1,1],&ffn);
    let residual=add(&ffn,&residual);
    tr.f("l_out",[H,t,1,1],&residual);
    Ok(BlockOutput {residual,trace:tr})
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
    struct DenseOnly { reads: Vec<String>, down_scale: f32 }
    impl Weights for DenseOnly {
        fn get(&mut self, name: &str, dims: &[usize], ty: u32, expert: Option<usize>) -> Result<Vec<f32>> {
            assert!(expert.is_none(), "dense FFN must not load experts");
            self.reads.push(name.into());
            let (shape, kind, values) = match name {
                "ffn_up.weight" => (vec![2,3],40,vec![1.0,0.0,0.0,1.0,1.0,-1.0]),
                "ffn_gate.weight" => (vec![2,3],40,vec![2.0,-1.0,1.0,2.0,-1.0,-2.0]),
                "ffn_down.weight" => (vec![3,2],40,vec![1.0,0.0,-1.0,0.0,1.0,1.0]),
                "ffn_up.scale" => (vec![1],0,vec![-0.5]),
                "ffn_gate.scale" => (vec![1],0,vec![0.75]),
                "ffn_down.scale" => (vec![1],0,vec![self.down_scale]),
                _ => panic!("unexpected dense request {name}"),
            };
            assert_eq!(dims,shape); assert_eq!(ty,kind); Ok(values)
        }
        fn get_nvfp4(&mut self,_:&str,_:&[usize],_:Option<usize>)->Result<Vec<u8>> { panic!("CPU math uses decoded weights") }
    }
    #[test] fn dense_parallel_branch_scales_and_no_moe() -> Result<()> {
        for t in [1,3] {
            let x: Vec<f32> = (0..t).flat_map(|token|[2.0+token as f32,-4.0+token as f32]).collect();
            let mut w = DenseOnly {reads:Vec::new(),down_scale:-1.25};
            let mut tr = Trace::default();
            let out = dense_ffn(&mut w,&mut tr,&x,2,3,t)?;
            let mut expected = Vec::new();
            for row in x.chunks_exact(2) {
                let u = [-0.5*row[0],-0.5*row[1],-0.5*(row[0]-row[1])];
                let g = [0.75*(2.0*row[0]-row[1]),0.75*(row[0]+2.0*row[1]),0.75*(-row[0]-2.0*row[1])];
                let z: Vec<f32> = g.iter().zip(u).map(|(&g,u)|(g*(1.0/(1.0+(-g).exp())))*u).collect();
                expected.extend([(z[0]-z[2])*-1.25,(z[1]+z[2])*-1.25]);
            }
            assert_eq!(out,expected);
            let get = |name: &str| match &tr.stages.iter().find(|s|s.name==name).unwrap().values {
                Values::F32(v) => v.clone(), _ => unreachable!(),
            };
            assert_eq!(get("dense_down_input"),get("dense_swiglu"));
            assert_eq!(tr.stages.len(),8);
            assert_eq!(w.reads,["ffn_up.weight","ffn_up.scale","ffn_gate.weight","ffn_gate.scale","ffn_down.weight","ffn_down.scale"]);
            let mut quiet=Trace::new(false);
            assert_eq!(dense_ffn(&mut w,&mut quiet,&x,2,3,t)?,out);
            assert!(quiet.stages.is_empty());
            w.down_scale=0.0;
            assert!(dense_ffn(&mut w,&mut Trace::default(),&x,2,3,t)?.iter().all(|&v|v==0.0));
        }
        Ok(())
    }
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
