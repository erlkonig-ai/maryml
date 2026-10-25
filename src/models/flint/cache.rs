//! Finite single-sequence KV storage. No eviction, batching, paging or persistence.
use anyhow::{Context, Result, ensure};
use super::forward::{Backend, BLOCKS};

pub const KV_WIDTH: usize = 4 * 128;
pub const MAX_QUERY: usize = 3;

/// Admission for retained KV/position storage and attention scratch only. Model
/// projections, tokenizer, diagnostic copies and CUDA runtime need extra headroom.
#[derive(Clone, Copy, Debug)]
pub struct Limits {
    pub capacity: usize,
    pub max_cache_bytes: usize,
    pub max_attention_bytes: usize,
}
impl Default for Limits {
    fn default() -> Self { Self { capacity:1024, max_cache_bytes:256<<20, max_attention_bytes:64<<20 } }
}
#[derive(Clone, Copy, Debug)]
pub struct Admission { pub cache_bytes: usize, pub attention_bytes: usize }
impl Limits {
    pub fn admit(self, model_context: usize) -> Result<Admission> {
        ensure!(self.capacity>0 && self.capacity<=model_context && self.capacity<=i32::MAX as usize,
            "capacity must be positive and bounded by model context/position range");
        let elements=self.capacity.checked_mul(KV_WIDTH).context("KV elements overflow")?;
        ensure!(elements<=u32::MAX as usize,"KV offsets exceed kernel range");
        let cache_bytes=elements.checked_mul(8).and_then(|n|self.capacity.checked_mul(4).and_then(|p|n.checked_add(p)))
            .and_then(|n|n.checked_mul(BLOCKS)).context("cache bytes overflow")?;
        let geometry=Geometry::new(MAX_QUERY,self.capacity)?;
        let attention_bytes=geometry.scores.checked_mul(2).and_then(|n|n.checked_add(geometry.mask))
            .and_then(|n|n.checked_add(MAX_QUERY*32*128*2))
            .and_then(|n|self.capacity.checked_mul(3).and_then(|p|p.checked_add(MAX_QUERY)).and_then(|p|n.checked_add(p)))
            .and_then(|n|n.checked_mul(4)).context("attention bytes overflow")?;
        ensure!(cache_bytes<=self.max_cache_bytes,"KV admission: {cache_bytes} > {} bytes",self.max_cache_bytes);
        ensure!(attention_bytes<=self.max_attention_bytes,"attention admission: {attention_bytes} > {} bytes",self.max_attention_bytes);
        Ok(Admission {cache_bytes,attention_bytes})
    }
}

pub(super) struct Geometry { pub mask:usize, pub scores:usize }
impl Geometry {
    pub fn new(q:usize,kv:usize)->Result<Self> {
        ensure!((1..=MAX_QUERY).contains(&q) && kv>0,"invalid Q/KV lengths");
        let mask=q.checked_mul(kv).context("mask size overflow")?;
        let scores=mask.checked_mul(32).context("attention size overflow")?;
        ensure!(scores<=u32::MAX as usize && kv.checked_mul(KV_WIDTH).is_some_and(|n|n<=u32::MAX as usize),
            "attention offsets exceed kernel range");
        scores.checked_mul(4).context("attention allocation overflow")?;
        Ok(Self {mask,scores})
    }
}

pub(super) fn zeros(len:usize)->Result<Vec<f32>> {
    len.checked_mul(4).context("F32 allocation overflow")?;
    let mut values=Vec::new(); values.try_reserve_exact(len).context("F32 allocation admission")?;
    values.resize(len,0.0); Ok(values)
}
pub(super) fn validate_append(old:&[i32],positions:&[i32],capacity:usize)->Result<usize> {
    ensure!((1..=MAX_QUERY).contains(&positions.len()),"query length must be 1..3");
    ensure!(positions.iter().all(|&p|p>=0) && positions.windows(2).all(|p|p[0]<p[1])
        && old.last().is_none_or(|&p|p<positions[0]),"KV positions must increase across appends");
    let end=old.len().checked_add(positions.len()).context("KV length overflow")?;
    ensure!(end<=capacity,"KV capacity exhausted"); Ok(end)
}
pub(super) fn visible(query:i32,key:i32,window:Option<i32>)->bool {
    key<=query && window.is_none_or(|w|query-key<w)
}

pub(super) struct CpuKv { pub k:Vec<f32>, pub v:Vec<f32>, pub positions:Vec<i32>, capacity:usize }
impl CpuKv {
    pub fn new(capacity:usize)->Result<Self> {
        let elements=capacity.checked_mul(KV_WIDTH).context("CPU KV size overflow")?;
        let mut k=Vec::new(); k.try_reserve_exact(elements)?;
        let mut v=Vec::new(); v.try_reserve_exact(elements)?;
        let mut positions=Vec::new(); positions.try_reserve_exact(capacity)?;
        Ok(Self {k,v,positions,capacity})
    }
    pub fn append(&mut self,k:&[f32],v:&[f32],positions:&[i32])->Result<()> {
        validate_append(&self.positions,positions,self.capacity)?;
        ensure!(k.len()==positions.len()*KV_WIDTH && v.len()==k.len(),"KV row shape differs");
        ensure!(k.iter().chain(v).all(|v|v.is_finite()),"nonfinite KV rows");
        self.k.extend_from_slice(k); self.v.extend_from_slice(v); self.positions.extend_from_slice(positions); Ok(())
    }
    fn reset(&mut self) { self.k.clear(); self.v.clear(); self.positions.clear(); }
}
pub(super) enum LayerCache {
    Cpu(CpuKv),
    #[cfg(feature="flint-cuda")]
    Cuda(super::gpu::KvCache),
}
impl LayerCache {
    pub fn len(&self)->usize { match self {
        Self::Cpu(c)=>c.positions.len(),
        #[cfg(feature="flint-cuda")] Self::Cuda(c)=>c.len(),
    }}
    pub fn reset(&mut self)->Result<()> { match self {
        Self::Cpu(c)=>c.reset(),
        #[cfg(feature="flint-cuda")] Self::Cuda(c)=>c.reset()?,
    } Ok(()) }
}
pub(super) fn allocate(backend:Backend,capacity:usize)->Result<Vec<LayerCache>> {
    (0..BLOCKS).map(|_| match backend {
        Backend::Cpu=>Ok(LayerCache::Cpu(CpuKv::new(capacity)?)),
        #[cfg(feature="flint-cuda")]
        Backend::Cuda|Backend::PackedAll=>Ok(LayerCache::Cuda(super::gpu::KvCache::new(capacity)?)),
        #[cfg(not(feature="flint-cuda"))]
        _=>anyhow::bail!("CUDA feature required"),
    }).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test] fn capacity_and_checked_admission() {
        assert!(Limits::default().admit(262144).is_ok());
        for capacity in [0,262145,usize::MAX] { assert!(Limits {capacity,..Limits::default()}.admit(262144).is_err()); }
        assert!(Limits {max_cache_bytes:1,..Limits::default()}.admit(262144).is_err());
        assert!(Limits {max_attention_bytes:1,..Limits::default()}.admit(262144).is_err());
        assert!(Geometry::new(3,usize::MAX).is_err());
        assert!(Limits {capacity:2048,max_cache_bytes:512<<20,..Limits::default()}.admit(262144).is_ok());
    }
    #[test] fn append_validation_and_reset() -> Result<()> {
        let mut c=CpuKv::new(2)?; let x=vec![0.0;KV_WIDTH];
        c.append(&x,&x,&[0])?;
        for pos in [-1,0] { assert!(c.append(&x,&x,&[pos]).is_err()); assert_eq!(c.positions,[0]); }
        c.append(&x,&x,&[1])?; assert!(c.append(&x,&x,&[2]).is_err());
        c.reset(); c.append(&x,&x,&[0])?; assert_eq!(c.positions,[0]); Ok(())
    }
    #[test] fn real_513_boundary_and_crossing_chunk() {
        let keys:Vec<i32>=(0..514).collect();
        for query in [511,512,513] {
            assert_eq!(keys.iter().filter(|&&k|visible(query,k,None)).count(),query as usize+1);
            assert_eq!(keys.iter().filter(|&&k|visible(query,k,Some(513))).count(),(query as usize+1).min(513));
        }
        assert!(visible(512,0,Some(513))); assert!(!visible(513,0,Some(513)));
        assert!(visible(513,1,Some(513))); assert!(!visible(512,513,None));
    }
}
