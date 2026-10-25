//! Synthetic projection-boundary checks. The fake reader deliberately refuses
//! decoded type-40 requests; CPU decoding is used only for the test oracle.

use super::*;
use crate::formats::gguf::nvfp4;

const DENSE: [&str; 10] = ["attn_q", "attn_k", "attn_v", "attn_output",
    "ffn_up_shexp", "ffn_gate_shexp", "ffn_down_shexp", "ffn_up", "ffn_gate", "ffn_down"];
const EXPERTS: [&str; 3] = ["ffn_up_exps", "ffn_gate_exps", "ffn_down_exps"];

#[test]
fn projection_modes_keep_k_only_and_decoded_paths() {
    for suffix in DENSE.into_iter().chain(EXPERTS) {
        assert!(!ProjectionMode::Decoded.packed(suffix));
        assert_eq!(ProjectionMode::PackedK.packed(suffix), suffix == "attn_k");
        assert!(ProjectionMode::PackedAll.packed(suffix));
    }
}

fn raw_matrix(ni: usize, no: usize, id: usize) -> Vec<u8> {
    let mut raw = Vec::new();
    for row in 0..no { for block in 0..ni / 64 {
        for group in 0..4 { raw.push(0x30 + ((id + row + block + group) % 3) as u8 * 8); }
        for group in 0..4 { for j in 0..8 {
            let low = ((id + row * 3 + block * 5 + group * 7 + j) % 16) as u8;
            let high = ((id * 3 + row * 7 + block + group * 5 + j + 9) % 16) as u8;
            raw.push(low | (high << 4));
        }}
    }}
    raw
}
fn scale(id: usize) -> f32 { match id { 0 => 0.0, 255 => -0.5, _ => 0.75 } }
fn expected(ni: usize, no: usize, id: usize, input: &[f32]) -> Vec<f32> {
    let w = nvfp4::decode(&raw_matrix(ni,no,id), ni*no).unwrap();
    w.chunks_exact(ni).map(|row|
        row.iter().zip(input).map(|(&w,&x)| w as f64*x as f64).sum::<f64>() as f32).collect()
}
fn trace_f32<'a>(trace: &'a Trace, name: &str) -> &'a [f32] {
    match &trace.stages.iter().find(|s| s.name == name).unwrap().values {
        Values::F32(v) => v, _ => panic!("expected F32 stage"),
    }
}
fn exact(actual: &[f32], expected: &[f32]) {
    assert_eq!(actual.len(), expected.len());
    for (&a,&e) in actual.iter().zip(expected) { assert!(a.is_finite()); assert_eq!(a,e); }
}

struct PackedOnly { ni: usize, no: usize, reads: Vec<(String, Option<usize>)> }
impl Weights for PackedOnly {
    fn get(&mut self, suffix: &str, dims: &[usize], ty: u32, expert: Option<usize>) -> Result<Vec<f32>> {
        assert_eq!(ty,0,"packed projections must never request decoded type40 weights");
        assert!(suffix.ends_with(".scale") && expert.is_none());
        match dims {
            [1] => Ok(vec![0.75]),
            [E] => Ok((0..E).map(scale).collect()),
            _ => panic!("invalid scale shape {dims:?}"),
        }
    }
    fn get_nvfp4(&mut self, suffix: &str, dims: &[usize], expert: Option<usize>) -> Result<Vec<u8>> {
        assert!(suffix.ends_with(".weight"));
        if let Some(id) = expert { assert!(id < E); assert_eq!(dims,&[self.ni,self.no,E]); }
        else { assert_eq!(dims,&[self.ni,self.no]); }
        self.reads.push((suffix.into(),expert));
        Ok(raw_matrix(self.ni,self.no,expert.unwrap_or(17)))
    }
}

#[test]
#[ignore = "requires an exclusive GB10 GPU reservation"]
fn cuda_packed_dense_and_selected_experts_t1_t3() -> Result<()> {
    let dev = Device::new();
    let (ni,no) = (128,9); // multiple blocks and a partial output cube
    for t in [1,3] {
        let input: Vec<f32> = (0..ni*t).map(|i| (i as i32 % 23-11) as f32/32.0).collect();
        let x = dev.upload(&input);
        for suffix in DENSE {
            let mut w = PackedOnly { ni,no,reads:Vec::new() };
            let mut trace = Trace::default();
            let out = dev.dense(&mut w,&mut trace,suffix,"test",&x,ni,no,t,ProjectionMode::PackedAll)?;
            let raw: Vec<f32> = input.chunks_exact(ni).flat_map(|x| expected(ni,no,17,x)).collect();
            let scaled: Vec<f32> = raw.iter().map(|v| v*0.75).collect();
            exact(trace_f32(&trace,"test_unscaled"),&raw);
            exact(&dev.read(&out)?,&scaled);
            assert_eq!(w.reads,vec![(format!("{suffix}.weight"),None)]);
        }
        let ids: Vec<i32> = (0..K*t).map(|slot| [255,0,17,255,1,0][(slot+slot/K)%K]).collect();
        for suffix in EXPERTS {
            let per_slot = suffix == "ffn_down_exps";
            let rows = if per_slot { K*t } else { t };
            let input: Vec<f32> = (0..ni*rows).map(|i|
                ((i/ni*7+i%13) as i32-6) as f32/32.0).collect();
            let mut w = PackedOnly { ni,no,reads:Vec::new() };
            let mut trace = Trace::default();
            let out = dev.experts(&mut w,&mut trace,suffix,"test",&dev.upload(&input),
                ni,no,t,&ids,per_slot,ProjectionMode::PackedAll)?;
            let raw: Vec<f32> = ids.iter().enumerate().flat_map(|(slot,&id)| {
                let row = if per_slot { slot } else { slot/K };
                expected(ni,no,id as usize,&input[row*ni..(row+1)*ni])
            }).collect();
            let selected: Vec<f32> = ids.iter().map(|&id| scale(id as usize)).collect();
            let scaled: Vec<f32> = raw.iter().enumerate().map(|(i,&v)| v*selected[i/no]).collect();
            exact(trace_f32(&trace,"test_unscaled"),&raw);
            exact(trace_f32(&trace,"test_selected_scale"),&selected);
            exact(&dev.read(&out)?,&scaled);
            assert_eq!(w.reads,ids.iter().map(|&id|(format!("{suffix}.weight"),Some(id as usize))).collect::<Vec<_>>());
        }
        println!("packed common projection T={t}: ten ordinary + three routed families; repeated/boundary IDs, token/slot offsets and post-scales exact");
    }
    Ok(())
}

#[test]
#[ignore = "requires an exclusive GB10 GPU reservation"]
fn cuda_packed_dense0_actual_shapes_t1_t3() -> Result<()> {
    let dev = Device::new();
    for (suffix,ni,no) in [("ffn_up",2048,6144),("ffn_gate",2048,6144),("ffn_down",6144,2048)] {
        for t in [1,3] {
            let input: Vec<f32> = (0..ni*t).map(|i|
                ((i%ni*7+i/ni*13)%23) as f32/32.0-11.0/32.0).collect();
            let mut w = PackedOnly {ni,no,reads:Vec::new()};
            let mut tr = Trace::default();
            let out = dev.dense(&mut w,&mut tr,suffix,"dense_test",&dev.upload(&input),ni,no,t,ProjectionMode::PackedAll)?;
            // Decode once for the independent dyadic scalar-dot check.
            let decoded = nvfp4::decode(&raw_matrix(ni,no,17),ni*no)?;
            let raw: Vec<f32> = input.chunks_exact(ni).flat_map(|x|
                decoded.chunks_exact(ni).map(move |row|
                    row.iter().zip(x).map(|(&w,&v)|w as f64*v as f64).sum::<f64>() as f32)).collect();
            exact(trace_f32(&tr,"dense_test_unscaled"),&raw);
            exact(&dev.read(&out)?,&raw.iter().map(|v|v*0.75).collect::<Vec<_>>());
            assert_eq!(w.reads,vec![(format!("{suffix}.weight"),None)]);
            println!("packed dense0 shape {suffix} [{ni},{no}] T={t}: asymmetric dyadic input and nonunit postscale exact");
        }
    }
    Ok(())
}

#[test]
#[ignore = "requires an exclusive GB10 GPU reservation"]
fn cuda_dense_parallel_swiglu_controls() -> Result<()> {
    let dev = Device::new();
    // Asymmetric signed raw projections, nonunit distinct postscales. Values
    // exceed the MoE adaptive threshold so a down-input normalization is visible.
    let up = dev.upload(&[4.0,-8.0,8192.0,-16384.0]);
    let gate = dev.upload(&[-4.0,2.0,4.0,8.0]);
    let u = dev.point(&up,&dev.upload(&[-0.5]),4,1);
    let g = dev.point(&gate,&dev.upload(&[0.75]),4,1);
    let sw = dev.point(&g,&u,4,3);
    let actual = dev.read(&sw)?;
    let swapped = dev.read(&dev.point(&u,&g,4,3))?;
    let wrong = dev.point(&gate,&u,4,3);
    let wrong = dev.read(&dev.point(&wrong,&dev.upload(&[0.75]),4,1))?;
    assert!(actual.iter().zip(swapped).any(|(&a,b)|a.to_bits()!=b.to_bits()));
    assert!(actual.iter().zip(wrong).any(|(&a,b)|a.to_bits()!=b.to_bits()));
    assert!(actual.iter().any(|v|v.abs()>1024.0));
    let zero = dev.point(&sw,&dev.upload(&[0.0]),4,1);
    assert!(dev.read(&zero)?.iter().all(|&v|v==0.0));
    let residual = dev.upload(&[1.0,-2.0,3.0,-4.0]);
    exact(&dev.read(&dev.point(&residual,&zero,4,0))?,&[1.0,-2.0,3.0,-4.0]);
    println!("dense controls: swapped gate/up and moving gate scale after SiLU differ; zero contribution preserves residual; activation exceeds1024");
    Ok(())
}
