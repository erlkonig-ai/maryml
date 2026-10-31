//! Diagnostic scalar reference, NOT a production CPU fallback.
//!
//! Implements the pinned Transformers torch_forward algebra: [z,x,B,C,dt],
//! biased depthwise causal conv then SiLU, softplus(dt+dt_bias) clamped below
//! by time_step_min, A=-exp(A_log), grouped B/C, D*x skip, gate BEFORE grouped
//! RMSNorm, ordinary gain. No input/output projections or outer residual.
//!
//! Arithmetic inside a call is f64, deliberately not a transcription of every
//! Torch BF16/F32 rounding/reduction. `chunk` emulates cache storage at call
//! boundaries; `decode` additionally emulates the pinned cached branch's
//! store-before-readout and BF16 C-readout cast. Thus this is an algebra oracle,
//! NOT full BF16 torch_forward parity. Chunk partition differences after casts
//! are expected. The fast path's dt_limit/softplus behavior remains a separate
//! comparison obligation. No padding masks or sequence packing are inferred.
use super::contract::{CacheDtype, Config, Parameters, Result};

#[derive(Clone, Debug, PartialEq)]
pub struct State {
    // Binding here covers geometry/batch/dtype only, NOT parameter/model/layer
    // identity. An owning native session must supply that immutable binding and
    // reset recurrent, convolution and attention histories coherently.
    config: Config,
    batch: usize,
    dtype: CacheDtype,
    pub attended: usize,
    /// Values are stored in f64 containers but rounded to the declared dtype.
    pub conv: Vec<f64>,
    pub recurrent: Vec<f64>,
}
impl State {
    /// Logical reset only; the production GPU implementation must also honor
    /// stream lifetime/fences. Both histories and position are cleared together.
    pub fn reset(&mut self) {
        self.conv.fill(0.0); self.recurrent.fill(0.0); self.attended=0;
    }
    pub fn cache_dtype(&self) -> CacheDtype { self.dtype }
}

pub struct Output { pub values: Vec<f64>, pub state: State }

fn finite(name: &str, values: &[f64]) -> Result<()> {
    if values.iter().all(|v|v.is_finite()) { Ok(()) }
    else { Err(format!("nonfinite {name}")) }
}
fn sized(name: &str, values: &[f64], size: usize) -> Result<()> {
    if values.len()!=size { return Err(format!("{name}: expected {size}, got {}",values.len())); }
    finite(name,values)
}
fn silu(x: f64) -> f64 {
    if x>=0.0 { x/(1.0+(-x).exp()) } else { let e=x.exp(); x*e/(1.0+e) }
}
fn softplus(x: f64) -> f64 { if x>20.0 { x } else { x.exp().ln_1p() } }
fn cast(value: f64, dtype: CacheDtype) -> f64 {
    let f=value as f32;
    match dtype {
        CacheDtype::F32=>f as f64,
        CacheDtype::Bf16=>{
            let bits=f.to_bits();
            let rounded=bits.wrapping_add(0x7fff+((bits>>16)&1))&0xffff0000;
            f32::from_bits(rounded) as f64
        }
    }
}

/// Immutable incoming state; validate all descriptors/finite inputs before
/// allocation. On failure no caller state changes. The budget covers this
/// reference's output, copied histories and one convolution row (f64), not
/// caller-owned input/parameters/state, allocator overhead, or a future GPU.
/// A continuation chunk is a sequence of mathematical recurrence steps; the
/// pinned public HF cached branch itself accepts only one decode token.
pub fn chunk(c: Config, p: &Parameters<'_>, projected: &[f64], batch: usize,
    tokens: usize, previous: Option<&State>, dtype: CacheDtype, max_bytes: usize) -> Result<Output>
{
    run(c,p,projected,batch,tokens,previous,dtype,max_bytes,false)
}

/// One cached torch_forward step with BF16 projected/C boundary: the updated
/// recurrence is stored (cast to cache dtype), then cast to BF16 before C dot.
/// This models those two explicit casts, NOT all BF16 operations/reductions.
/// A multi-token continuation in `chunk` is an algebra extension, not this HF
/// cached branch. A reset state must first be prefilled with `chunk`.
pub fn decode(c: Config, p: &Parameters<'_>, projected: &[f64], batch: usize,
    previous: &State, dtype: CacheDtype, max_bytes: usize) -> Result<Output>
{
    if previous.attended==0 {return Err("decode needs attended history".into());}
    run(c,p,projected,batch,1,Some(previous),dtype,max_bytes,true)
}

fn run(c: Config, p: &Parameters<'_>, projected: &[f64], batch: usize,
    tokens: usize, previous: Option<&State>, dtype: CacheDtype, max_bytes: usize,
    cached_decode: bool) -> Result<Output>
{
    let l=c.layout()?;
    let e=c.extents(batch,tokens)?;
    sized("projected input",projected,e.input)?;
    let weight_len=l.conv_channels.checked_mul(c.conv_kernel).ok_or("conv weight overflow")?;
    sized("conv weight",p.conv_weight,weight_len)?;
    if let Some(bias)=p.conv_bias { sized("conv bias",bias,l.conv_channels)?; }
    for (name,v) in [("A_log",p.a_log),("dt_bias",p.dt_bias),("D",p.d)] { sized(name,v,c.heads)?; }
    sized("norm gain",p.norm_gain,l.inner)?;
    if p.a_log.iter().any(|a|!a.exp().is_finite()) {return Err("A exponent overflow".into());}
    let attended=previous.map_or(0,|s|s.attended).checked_add(tokens).ok_or("position overflow")?;
    if let Some(s)=previous {
        if s.config!=c || s.batch!=batch || s.dtype!=dtype { return Err("state binding differs".into()); }
        sized("conv history",&s.conv,e.conv)?;
        sized("recurrent history",&s.recurrent,e.recurrent)?;
    }
    let allocated=e.output.checked_add(e.conv).and_then(|n|n.checked_add(e.recurrent))
        .and_then(|n|n.checked_add(l.conv_channels)).and_then(|n|n.checked_mul(8))
        .ok_or("reference allocation overflow")?;
    if allocated>max_bytes {return Err(format!("reference budget: {allocated}>{max_bytes}"));}
    let mut state=match previous {Some(s)=>s.clone(),None=>State {config:c,batch,dtype,attended:0,
        conv:vec![0.0;e.conv],recurrent:vec![0.0;e.recurrent]}};
    let mut values=vec![0.0;e.output];
    let mut conv=vec![0.0;l.conv_channels];
    for b in 0..batch {
        for t in 0..tokens {
            let input=&projected[(b*tokens+t)*l.projected..][..l.projected];
            for channel in 0..l.conv_channels {
                let hist=&mut state.conv[(b*l.conv_channels+channel)*c.conv_kernel..][..c.conv_kernel];
                hist.copy_within(1..,0);
                hist[c.conv_kernel-1]=input[l.inner+channel];
                let w=&p.conv_weight[channel*c.conv_kernel..][..c.conv_kernel];
                let sum=hist.iter().zip(w).fold(p.conv_bias.map_or(0.0,|v|v[channel]),|a,(x,w)|a+x*w);
                conv[channel]=silu(sum);
            }
            finite("convolution output",&conv)?;
            let out=&mut values[(b*tokens+t)*l.inner..][..l.inner];
            for h in 0..c.heads {
                let group=h/(c.heads/c.groups);
                let dt=softplus(input[l.offsets[4]+h]+p.dt_bias[h]).max(c.time_step_min);
                let decay=(-p.a_log[h].exp()*dt).exp();
                for d in 0..c.head_dim {
                    let col=h*c.head_dim+d;
                    let x=conv[col];
                    let s=&mut state.recurrent[(b*l.inner+col)*c.state_dim..][..c.state_dim];
                    let mut y=p.d[h]*x;
                    for n in 0..c.state_dim {
                        let bval=conv[l.inner+group*c.state_dim+n];
                        let cval=conv[l.inner+l.grouped_state+group*c.state_dim+n];
                        s[n]=decay*s[n]+dt*bval*x;
                        let readout=if cached_decode {
                            s[n]=cast(s[n],dtype);
                            cast(s[n],CacheDtype::Bf16)
                        } else {s[n]};
                        y+=readout*cval;
                    }
                    out[col]=y*silu(input[col]);
                }
            }
            for group in 0..c.groups {
                let base=group*l.norm_group;
                let norm=&mut out[base..base+l.norm_group];
                let variance=norm.iter().map(|x|x*x).sum::<f64>()/l.norm_group as f64;
                if !variance.is_finite() {return Err("nonfinite normalization variance".into());}
                let inv=1.0/(variance+c.norm_epsilon).sqrt();
                for (i,x) in norm.iter_mut().enumerate() { *x=*x*inv*p.norm_gain[base+i]; }
            }
        }
    }
    finite("output",&values)?; finite("updated recurrence",&state.recurrent)?;
    // Conv input is BF16 in the intended GPU boundary; explicitly emulate it
    // here as well. Recurrence storage can instead be F32 by explicit choice.
    for x in &mut state.conv { *x=cast(*x,CacheDtype::Bf16); }
    for x in &mut state.recurrent { *x=cast(*x,dtype); }
    finite("stored conv",&state.conv)?; finite("stored recurrence",&state.recurrent)?;
    state.attended=attended;
    Ok(Output {values,state})
}

#[cfg(test)]
mod tests {
    use super::*;
    fn config() -> Config { Config {heads:4,head_dim:2,groups:2,state_dim:2,
        conv_kernel:3,norm_epsilon:0.125,time_step_min:0.001} }
    fn near(a:&[f64],b:&[f64],tol:f64) {
        assert_eq!(a.len(),b.len());
        for (i,(&x,&y)) in a.iter().zip(b).enumerate() {
            assert!((x-y).abs()<=tol*(1.0+y.abs()),"index {i}: {x} != {y}");
        }
    }
    // Independent O(T^2) prefix-sum equation, NOT the recurrence implementation:
    // S_t = sum_{j<=t} exp(sum_{k=j+1..t} A*dt_k) * dt_j * B_j * x_j.
    // This fixture uses one-tap identity convolution, A=-1, dt=ln(2),
    // z and grouped B/C differ, with a nonzero per-head D skip. Expected
    // values are built directly below without the scan/conv/norm helpers.
    #[test]
    fn independent_closed_form_state_output_and_grouped_norm() {
        let c=Config {conv_kernel:1,..config()}; let l=c.layout().unwrap(); let t=4;
        let weights=vec![1.0;l.conv_channels]; let zeros=vec![0.0;c.heads];
        let gain:Vec<_>=(0..l.inner).map(|i|1.0+i as f64/8.0).collect();
        let skip=[0.0,0.25,-0.5,0.75];
        let p=Parameters {conv_weight:&weights,conv_bias:None,a_log:&zeros,dt_bias:&zeros,d:&skip,norm_gain:&gain};
        let mut input=vec![0.0;t*l.projected];
        for row in 0..t {
            let x=&mut input[row*l.projected..][..l.projected];
            for i in 0..l.inner { x[i]=1.0+i as f64/8.0; x[l.inner+i]=(row+i+1) as f64/8.0; }
            for g in 0..c.groups {
                x[l.offsets[2]+g*c.state_dim..][..c.state_dim].fill(1.0+g as f64);
                x[l.offsets[3]+g*c.state_dim..][..c.state_dim].fill(2.0+g as f64);
            }
        }
        let got=chunk(c,&p,&input,1,t,None,CacheDtype::F32,1<<20).unwrap();
        let mut expected=vec![0.0;t*l.inner]; let mut last=vec![0.0;l.inner*c.state_dim];
        for row in 0..t {
            for i in 0..l.inner {
                let mut s=0.0;
                let g=i/l.norm_group;
                let br=1.0+g as f64; let cr=2.0+g as f64;
                let b=br/(1.0+(-br).exp()); let cc=cr/(1.0+(-cr).exp());
                for j in 0..=row {
                    let raw=(j+i+1) as f64/8.0; let x=raw/(1.0+(-raw).exp());
                    s+=0.5f64.powi((row-j) as i32)*std::f64::consts::LN_2*b*x;
                }
                let zr=1.0+i as f64/8.0; let z=zr/(1.0+(-zr).exp());
                let xr=(row+i+1) as f64/8.0; let x=xr/(1.0+(-xr).exp());
                expected[row*l.inner+i]=(s*cc*2.0+skip[i/c.head_dim]*x)*z;
                if row+1==t {last[i*c.state_dim..(i+1)*c.state_dim].fill((s as f32) as f64);}
            }
            for g in 0..c.groups {
                let v=&mut expected[row*l.inner+g*l.norm_group..][..l.norm_group];
                let denom=(v.iter().map(|x|x*x).sum::<f64>()/v.len() as f64+0.125).sqrt();
                for (i,x) in v.iter_mut().enumerate() { *x=*x/denom*gain[g*l.norm_group+i]; }
            }
        }
        near(&got.values,&expected,1e-12); near(&got.state.recurrent,&last,0.0);
    }

    #[test]
    fn chunks_decode_reset_group_mapping_bias_and_rejection() {
        let c=config(); let l=c.layout().unwrap(); let tokens=5;
        let weights:Vec<_>=(0..l.conv_channels*c.conv_kernel).map(|i|(i%5) as f64/16.0-0.125).collect();
        let bias:Vec<_>=(0..l.conv_channels).map(|i|(i%3) as f64/8.0).collect();
        let a=vec![0.25;c.heads]; let dt=vec![-0.125;c.heads]; let d=vec![0.5;c.heads];
        let gain:Vec<_>=(0..l.inner).map(|i|0.5+i as f64/8.0).collect();
        let p=Parameters {conv_weight:&weights,conv_bias:Some(&bias),a_log:&a,dt_bias:&dt,d:&d,norm_gain:&gain};
        // Binary fractions: projected/conv history is exactly BF16, so the
        // partition comparison isolates the explicitly rounded recurrent state.
        let input:Vec<_>=(0..tokens*l.projected).map(|i|(i%17) as f64/8.0-1.0).collect();
        for dtype in [CacheDtype::F32,CacheDtype::Bf16] {
            let whole=chunk(c,&p,&input,1,tokens,None,dtype,1<<20).unwrap();
            for partition in [vec![1,1,1,1,1],vec![2,3],vec![3,1,1]] {
                let mut state=None; let mut values=Vec::new(); let mut start=0;
                for n in partition {
                    let out=chunk(c,&p,&input[start*l.projected..(start+n)*l.projected],1,n,state.as_ref(),dtype,1<<20).unwrap();
                    values.extend_from_slice(&out.values); state=Some(out.state); start+=n;
                }
                let mut state=state.unwrap(); assert_eq!(state.attended,tokens);
                let tolerance=if dtype==CacheDtype::F32 {1e-6} else {0.01};
                near(&values,&whole.values,tolerance);
                near(&state.recurrent,&whole.state.recurrent,tolerance);
                let before=state.clone();
                let mut bad=input[..l.projected].to_vec(); bad[0]=f64::NAN;
                assert!(chunk(c,&p,&bad,1,1,Some(&state),dtype,1<<20).is_err());
                assert!(chunk(c,&p,&input[..l.projected],1,1,Some(&state),dtype,0).is_err());
                assert_eq!(state,before);
                state.reset(); assert_eq!(state.attended,0); assert!(state.conv.iter().chain(&state.recurrent).all(|&x|x==0.0));
                let reset=chunk(c,&p,&input,1,tokens,Some(&state),dtype,1<<20).unwrap();
                near(&reset.values,&whole.values,0.0);
            }
        }
    }
    #[test]
    fn cache_cast_is_explicit_and_gate_precedes_norm() {
        assert_eq!(cast(1.00390625,CacheDtype::Bf16),1.0); // midpoint, even
        assert_eq!(cast(1.01171875,CacheDtype::Bf16),1.015625);
        assert_ne!(cast(1.00390625,CacheDtype::F32),cast(1.00390625,CacheDtype::Bf16));
        // Norm(y*SiLU(z)) differs from Norm(y)*SiLU(z), independently of code.
        let y=[1.0,2.0]; let z=[0.5,2.0];
        let gated=[y[0]*silu(z[0]),y[1]*silu(z[1])];
        let before=(gated[0]*gated[0]+gated[1]*gated[1])/2.0;
        let after=(y[0]*y[0]+y[1]*y[1])/2.0;
        assert!((gated[0]/(before+0.125).sqrt()-y[0]/(after+0.125).sqrt()*silu(z[0])).abs()>0.01);
    }

    #[test]
    fn batch_two_history_isolation_and_state_preflight() {
        let c=config(); let l=c.layout().unwrap(); let tokens=4;
        let weights:Vec<_>=(0..l.conv_channels*c.conv_kernel).map(|i|(i%7) as f64/8.0-0.25).collect();
        let bias=vec![0.125;l.conv_channels]; let a=vec![0.0;c.heads];
        let d=vec![0.25;c.heads]; let gain=vec![1.0;l.inner];
        let p=Parameters {conv_weight:&weights,conv_bias:Some(&bias),a_log:&a,dt_bias:&a,d:&d,norm_gain:&gain};
        let input:Vec<_>=(0..2*tokens*l.projected).map(|i|(i%29) as f64/8.0-1.5).collect();
        let both=chunk(c,&p,&input,2,tokens,None,CacheDtype::F32,1<<20).unwrap();
        let e=c.extents(1,tokens).unwrap();
        for b in 0..2 {
            let single=chunk(c,&p,&input[b*e.input..][..e.input],1,tokens,None,CacheDtype::F32,1<<20).unwrap();
            near(&both.values[b*e.output..][..e.output],&single.values,0.0);
            near(&both.state.conv[b*e.conv..][..e.conv],&single.state.conv,0.0);
            near(&both.state.recurrent[b*e.recurrent..][..e.recurrent],&single.state.recurrent,0.0);
            // Independent raw history indexing: last K projected samples,
            // oldest first, each batch isolated; no convolution helper used.
            for ch in 0..l.conv_channels { for k in 0..c.conv_kernel {
                assert_eq!(both.state.conv[(b*l.conv_channels+ch)*c.conv_kernel+k],
                    input[(b*tokens+tokens-c.conv_kernel+k)*l.projected+l.inner+ch]);
            }}
        }
        assert_ne!(&both.values[..e.output],&both.values[e.output..]);
        // Continuation input must be gathered by [batch,token], not flattened
        // across the old batch boundary. Compare cached decode to each batch.
        let next:Vec<_>=(0..2).flat_map(|b|input[b*e.input..b*e.input+l.projected].iter().copied()).collect();
        let step=decode(c,&p,&next,2,&both.state,CacheDtype::F32,1<<20).unwrap();
        for b in 0..2 {
            let single=chunk(c,&p,&input[b*e.input..][..e.input],1,tokens,None,CacheDtype::F32,1<<20).unwrap();
            let y=decode(c,&p,&next[b*l.projected..][..l.projected],1,&single.state,CacheDtype::F32,1<<20).unwrap();
            near(&step.values[b*l.inner..][..l.inner],&y.values,0.0);
        }
        let before=both.state.clone();
        assert!(decode(c,&p,&next,2,&before,CacheDtype::Bf16,1<<20).is_err());
        assert!(decode(c,&p,&next[..l.projected],1,&before,CacheDtype::F32,1<<20).is_err());
        let changed=Config {norm_epsilon:0.25,..c};
        assert!(decode(changed,&p,&next,2,&before,CacheDtype::F32,1<<20).is_err());
        let mut overflow=before.clone(); overflow.attended=usize::MAX;
        assert!(decode(c,&p,&next,2,&overflow,CacheDtype::F32,1<<20).is_err());
        assert_eq!(both.state,before);
        let mut reset=before.clone(); reset.reset();
        assert!(decode(c,&p,&next,2,&reset,CacheDtype::F32,1<<20).is_err());
    }

    #[test]
    fn independent_decode_cast_and_dt_floor() {
        let c=Config {heads:1,head_dim:1,groups:1,state_dim:1,conv_kernel:1,
            norm_epsilon:0.125,time_step_min:0.25};
        let p=Parameters {conv_weight:&[1.0;3],conv_bias:None,a_log:&[0.0],
            dt_bias:&[-100.0],d:&[0.25],norm_gain:&[1.0]};
        let input=[1.0,0.5,1.0,2.0,0.0];
        for dtype in [CacheDtype::F32,CacheDtype::Bf16] {
            let initial=chunk(c,&p,&input,1,1,None,dtype,1024).unwrap();
            let got=decode(c,&p,&input,1,&initial.state,dtype,1024).unwrap();
            let x=0.5/(1.0+(-0.5f64).exp());
            let b=1.0/(1.0+(-1.0f64).exp());
            let cc=2.0/(1.0+(-2.0f64).exp());
            let expected_stored=cast((-0.25f64).exp()*initial.state.recurrent[0]+0.25*b*x,dtype);
            assert_eq!(got.state.recurrent,[expected_stored]);
            let y=(cast(expected_stored,CacheDtype::Bf16)*cc+0.25*x)*b;
            near(&got.values,&[y/(y*y+0.125).sqrt()],1e-12);
            // Negative control: wider readout is measurably different here.
            if dtype==CacheDtype::F32 { assert_ne!(expected_stored,cast(expected_stored,CacheDtype::Bf16)); }
        }
    }
}
