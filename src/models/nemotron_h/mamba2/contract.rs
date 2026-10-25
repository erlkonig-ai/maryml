//! First Nemotron-H Mamba2 component contract; not wired into the model runtime.
//!
//! Authority: Transformers cc832f9055ba11c8c55f918ab4bda9472b910d48,
//! modeling_nemotron_h.py NemotronHMamba2Mixer and modeling_zamba2.py
//! Zamba2RMSNormGated. Projection/FP8, outer pre-norm, residual, and output
//! projection are deliberately outside this component.

pub type Result<T> = std::result::Result<T, String>;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Config {
    pub heads: usize,
    pub head_dim: usize,
    pub groups: usize,
    pub state_dim: usize,
    pub conv_kernel: usize,
    pub norm_epsilon: f64,
    /// The pinned torch_forward lower clamp, NOT the fast kernel's dt_limit.
    pub time_step_min: f64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Layout {
    pub inner: usize,
    pub grouped_state: usize,
    pub conv_channels: usize,
    pub projected: usize,
    pub norm_group: usize,
    /// [z, x, B, C, dt] starting offsets, plus total projected width.
    pub offsets: [usize; 6],
}

fn mul(a: usize, b: usize) -> Result<usize> {
    a.checked_mul(b).ok_or_else(|| "shape multiplication overflow".into())
}
fn add(a: usize, b: usize) -> Result<usize> {
    a.checked_add(b).ok_or_else(|| "shape addition overflow".into())
}

impl Config {
    pub fn layout(self) -> Result<Layout> {
        if self.heads == 0 || self.head_dim == 0 || self.groups == 0
            || self.state_dim == 0 || self.conv_kernel == 0
            || self.heads % self.groups != 0
            || !self.norm_epsilon.is_finite() || self.norm_epsilon <= 0.0
            || !self.time_step_min.is_finite() || self.time_step_min < 0.0
        {
            return Err("positive geometry, groups dividing heads, finite epsilon>0 and dt floor>=0 required".into());
        }
        let inner = mul(self.heads, self.head_dim)?;
        let grouped_state = mul(self.groups, self.state_dim)?;
        let conv_channels = add(inner, mul(2, grouped_state)?)?;
        let b = mul(2, inner)?;
        let c = add(b, grouped_state)?;
        let dt = add(c, grouped_state)?;
        let projected = add(dt, self.heads)?;
        Ok(Layout { inner, grouped_state, conv_channels, projected,
            norm_group: inner / self.groups, offsets: [0, inner, b, c, dt, projected] })
    }

    /// Geometry of a future contiguous GPU API: input [B,T,projected], output
    /// [B,T,inner], raw convolution history [B,conv_channels,K] oldest first,
    /// recurrent history [B,heads,head_dim,state_dim]. All products are checked.
    pub fn extents(self, batch: usize, tokens: usize) -> Result<Extents> {
        let l = self.layout()?;
        if batch == 0 || tokens == 0 { return Err("empty batch/chunk".into()); }
        let rows = mul(batch, tokens)?;
        Ok(Extents {
            input: mul(rows, l.projected)?, output: mul(rows, l.inner)?,
            conv: mul(mul(batch, l.conv_channels)?, self.conv_kernel)?,
            recurrent: mul(mul(batch, l.inner)?, self.state_dim)?,
        })
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Extents { pub input: usize, pub output: usize, pub conv: usize, pub recurrent: usize }

/// Explicit cache-cast boundary. In the pinned generic LinearAttentionLayer,
/// recurrent storage inherits conv_states.dtype; the config's float32 string
/// alone does not establish F32 storage. This is not an arithmetic precision
/// switch: the CPU reference uses f64 inside a call, whatever the cache dtype.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CacheDtype { Bf16, F32 }

/// Already decoded parameter values. Native tensor/FP8 ingress is not here.
/// All lengths follow Config; conv_bias is optional because the reference API
/// supports bias-free configurations, although the selected Super uses bias.
pub struct Parameters<'a> {
    pub conv_weight: &'a [f64], // [conv_channels,K], last tap is current sample
    pub conv_bias: Option<&'a [f64]>,
    pub a_log: &'a [f64],
    pub dt_bias: &'a [f64],
    pub d: &'a [f64],
    pub norm_gain: &'a [f64], // [inner], ordinary gain, not 1+weight
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn super_geometry_and_split_without_allocating_state() {
        let c = Config { heads:128, head_dim:64, groups:8, state_dim:128,
            conv_kernel:4, norm_epsilon:1e-5, time_step_min:0.001 };
        assert_eq!(c.layout().unwrap().offsets, [0,8192,16384,17408,18432,18560]);
        assert_eq!(c.layout().unwrap().norm_group, 1024);
        let e = c.extents(1,1).unwrap();
        assert_eq!((e.conv,e.recurrent), (40960,1048576));
        // Native BF16 conv / F32 recurrence budgets; NOT this f64 oracle's heap.
        assert_eq!((e.conv*2,e.recurrent*4), (81920,4194304));
    }
    #[test]
    fn rejects_empty_nonfinite_and_overflow_geometry() {
        let mut c = Config { heads:2,head_dim:2,groups:1,state_dim:2,
            conv_kernel:2,norm_epsilon:1e-5,time_step_min:0.001 };
        assert!(c.extents(0,1).is_err());
        assert!(c.extents(usize::MAX,2).is_err());
        c.head_dim=usize::MAX; assert!(c.layout().is_err());
        c.head_dim=2; c.groups=3; assert!(c.layout().is_err());
        c.groups=1; c.norm_epsilon=f64::NAN; assert!(c.layout().is_err());
    }
}
