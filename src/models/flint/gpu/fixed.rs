//! Native-only immutable execution parameters. This is not an entity catalogue:
//! queries remain in PileWeights; only concrete GPU operands survive preparation.
use super::*;
use triblespace::core::blob::encodings::tensor::TensorView;
use super::super::{forward::{FixedAdmission,BLOCKS},pile::PileWeights};

struct PreparedProjection { view:TensorView, ni:usize, no:usize, scale:f32 }
impl PreparedProjection {
    fn read(w:&mut impl Weights,name:&str,ni:usize,no:usize,bytes:&mut usize)->Result<Self> {
        let (weight,scale)=w.packed_projection(name,&[ni,no],None)?;
        let PackedWeight::Native(view)=weight else {anyhow::bail!("resident fixed weights require native planes")};
        let scale=scale.context("resident fixed projection requires its paired postscale")?;
        ensure!(scale.is_finite(),"nonfinite resident fixed postscale");
        let (payload,_)=packed::native_geometry(&view,ni,no)?;
        add_bytes(bytes,payload)?;add_bytes(bytes,4)?;
        Ok(Self {view,ni,no,scale})
    }
    fn upload(self,dev:&Device)->Result<FixedProjection> {
        let storage=ProjectionStorage::Packed(dev.upload_native_nvfp4(&self.view,self.ni,self.no)?);
        Ok(FixedProjection {weight:Projection {ni:self.ni,no:self.no,storage},scale:dev.upload(&[self.scale])})
    }
}
fn add_bytes(total:&mut usize,bytes:usize)->Result<()> {
    *total=total.checked_add(bytes).context("resident fixed byte sum overflow")?;Ok(())
}
fn floats(w:&mut impl Weights,name:&str,dims:&[usize],ty:u32,bytes:&mut usize)->Result<Vec<f32>> {
    let expected=dims.iter().try_fold(1usize,|n,&d|n.checked_mul(d).context("fixed float shape overflow"))?;
    let values=w.get(name,dims,ty,None)?;
    ensure!(values.len()==expected && values.iter().all(|v|v.is_finite()),"resident fixed float geometry/value differs: {name}");
    add_bytes(bytes,values.len().checked_mul(4).context("fixed float bytes overflow")?)?;
    Ok(values)
}

struct PreparedFfn { up:PreparedProjection, gate:PreparedProjection, down:PreparedProjection }
impl PreparedFfn {
    fn read(w:&mut impl Weights,width:usize,shared:bool,bytes:&mut usize)->Result<Self> {
        let (up,gate,down)=if shared {("ffn_up_shexp","ffn_gate_shexp","ffn_down_shexp")}
            else {("ffn_up","ffn_gate","ffn_down")};
        Ok(Self {up:PreparedProjection::read(w,up,H,width,bytes)?,
            gate:PreparedProjection::read(w,gate,H,width,bytes)?,
            down:PreparedProjection::read(w,down,width,H,bytes)?})
    }
    fn upload(self,dev:&Device)->Result<FixedFfn> {
        Ok(FixedFfn {up:self.up.upload(dev)?,gate:self.gate.upload(dev)?,down:self.down.upload(dev)?})
    }
}
enum PreparedBranch { Dense(PreparedFfn), Moe {router:Vec<f32>,shared:PreparedFfn} }
struct PreparedBlock {
    config:BlockConfig, attn_norm:Vec<f32>, q_norm:Vec<f32>, k_norm:Vec<f32>, sinks:Vec<f32>, ffn_norm:Vec<f32>,
    q:PreparedProjection, k:PreparedProjection, v:PreparedProjection, o:PreparedProjection, ffn:PreparedBranch,
}
impl PreparedBlock {
    fn read(w:&mut impl Weights,config:BlockConfig,bytes:&mut usize)->Result<Self> {
        Ok(Self {
            config,attn_norm:floats(w,"attn_norm.weight",&[H],0,bytes)?,
            q_norm:floats(w,"attn_q_norm.weight",&[D],0,bytes)?,
            k_norm:floats(w,"attn_k_norm.weight",&[D],0,bytes)?,
            sinks:floats(w,"attn_sinks.weight",&[Q],0,bytes)?,
            ffn_norm:floats(w,"ffn_norm.weight",&[H],0,bytes)?,
            q:PreparedProjection::read(w,"attn_q",H,Q*D,bytes)?,
            k:PreparedProjection::read(w,"attn_k",H,KV*D,bytes)?,
            v:PreparedProjection::read(w,"attn_v",H,KV*D,bytes)?,
            o:PreparedProjection::read(w,"attn_output",Q*D,H,bytes)?,
            ffn:match config.ffn {
                FfnKind::Dense {width}=>PreparedBranch::Dense(PreparedFfn::read(w,width,false,bytes)?),
                FfnKind::Moe=>PreparedBranch::Moe {
                    // Preserve the existing BF16 storage -> F32 execution tier.
                    router:floats(w,"ffn_gate_inp.weight",&[H,E],30,bytes)?,
                    shared:PreparedFfn::read(w,SH,true,bytes)?,
                },
            },
        })
    }
    fn upload(self,dev:&Device)->Result<FixedBlock> {
        let block=FixedBlock {
            config:self.config,attn_norm:dev.upload(&self.attn_norm),q_norm:dev.upload(&self.q_norm),
            k_norm:dev.upload(&self.k_norm),sinks:dev.upload(&self.sinks),ffn_norm:dev.upload(&self.ffn_norm),
            q:self.q.upload(dev)?,k:self.k.upload(dev)?,v:self.v.upload(dev)?,o:self.o.upload(dev)?,
            ffn:match self.ffn {
                PreparedBranch::Dense(ffn)=>FixedBranch::Dense(ffn.upload(dev)?),
                PreparedBranch::Moe {router,shared}=>FixedBranch::Moe {router:dev.upload(&router),shared:shared.upload(dev)?},
            },
        };
        dev.sync()?;Ok(block)
    }
}

pub(super) struct FixedProjection { pub weight:Projection, pub scale:Buffer }
pub(super) struct FixedFfn { pub up:FixedProjection, pub gate:FixedProjection, pub down:FixedProjection }
pub(super) enum FixedBranch { Dense(FixedFfn), Moe {router:Buffer,shared:FixedFfn} }
pub(in crate::models::flint) struct FixedBlock {
    pub(super) config:BlockConfig, pub(super) attn_norm:Buffer, pub(super) q_norm:Buffer,
    pub(super) k_norm:Buffer, pub(super) sinks:Buffer, pub(super) ffn_norm:Buffer,
    pub(super) q:FixedProjection, pub(super) k:FixedProjection, pub(super) v:FixedProjection,
    pub(super) o:FixedProjection, pub(super) ffn:FixedBranch,
}

/// Ephemeral init-only views/F32 router vectors. No GPU allocation until the
/// complete actual selected geometry/byte sum passes its independent budget.
pub(in crate::models::flint) struct PreparedFixed { blocks:Vec<PreparedBlock>, admission:FixedAdmission }
impl PreparedFixed {
    pub fn new(w:&mut PileWeights,max_bytes:usize)->Result<Self> {
        let mut blocks=Vec::with_capacity(BLOCKS);let mut bytes=0;
        for index in 0..BLOCKS {
            w.select_block(index)?;
            let config=w.config;
            blocks.push(PreparedBlock::read(w,config,&mut bytes)?);
            // Early bounded refusal as well as the final whole-set admission.
            FixedAdmission::checked(blocks.len(),bytes,max_bytes)?;
        }
        w.select_block(0)?;
        let admission=FixedAdmission::checked(blocks.len(),bytes,max_bytes)?;
        Ok(Self {blocks,admission})
    }
    pub fn upload(self)->Result<ResidentFixed> {
        let dev=Device::new();
        let blocks=self.blocks.into_iter().map(|block|block.upload(&dev)).collect::<Result<Vec<_>>>()?;
        dev.sync()?;Ok(ResidentFixed {dev,blocks,admission:self.admission})
    }
}
pub(in crate::models::flint) struct ResidentFixed { dev:Device, blocks:Vec<FixedBlock>, admission:FixedAdmission }
impl ResidentFixed {
    pub fn admission(&self)->FixedAdmission {self.admission}
    pub fn block(&self,index:usize)->Result<&FixedBlock> {self.blocks.get(index).context("resident fixed block outside admitted set")}
    pub fn fence(&self)->Result<()> {self.dev.sync()}
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{leaf,formats::gguf::nvfp4::typed};
    #[derive(Default)]
    struct NativeFixture { fixed_reads:usize, routed_reads:usize, deny_fixed:bool, bad_scale:bool }
    impl Weights for NativeFixture {
        fn get_nvfp4(&mut self,_:&str,_:&[usize],_:Option<usize>)->Result<Vec<u8>> {
            anyhow::bail!("native fixed fixture must not request GGUF bytes")
        }
        fn get(&mut self,name:&str,dims:&[usize],ty:u32,expert:Option<usize>)->Result<Vec<f32>> {
            ensure!(!self.deny_fixed && expert.is_none(),"unexpected resident fixed reload");
            self.fixed_reads+=1;
            let n=dims.iter().product();
            match (name,ty) {
                ("ffn_gate_inp.weight",30)=>Ok((0..n).map(|i|((i/H)%17) as f32/65536.0).collect()),
                ("attn_sinks.weight",0)=>Ok((0..n).map(|i|i as f32/32.0).collect()),
                ("attn_norm.weight"|"attn_q_norm.weight"|"attn_k_norm.weight"|"ffn_norm.weight",0)=>
                    Ok((0..n).map(|i|0.5+(i%7) as f32/8.0).collect()),
                _=>anyhow::bail!("unexpected fixture float read: {name}"),
            }
        }
        fn paired_scales(&self)->bool {true}
        fn packed_projection(&mut self,name:&str,dims:&[usize],expert:Option<usize>)->Result<(PackedWeight,Option<f32>)> {
            if expert.is_some() {self.routed_reads+=1;} else {
                ensure!(!self.deny_fixed,"unexpected resident fixed projection reload");self.fixed_reads+=1;
            }
            let (ni,no)=(dims[0],dims[1]);
            let sign=if name.contains("gate") {0x99} else {0x22};
            let mut raw=Vec::with_capacity(ni*no/64*36);
            for _ in 0..ni*no/64 {
                raw.extend_from_slice(&[0x20;4]);raw.extend_from_slice(&[sign;32]);
            }
            let view=leaf::read_leaf(typed::to_tensor(&raw,[no as u64,ni as u64])?)?;
            Ok((PackedWeight::Native(view),Some(if self.bad_scale {f32::NAN}
                else if name.contains("gate") {-0.5} else {0.75})))
        }
    }
    #[test]
    fn fixed_byte_sum_refuses_overflow() {
        let mut bytes=usize::MAX-4;add_bytes(&mut bytes,4).unwrap();
        assert_eq!(bytes,usize::MAX);assert!(add_bytes(&mut bytes,1).is_err());
        assert_eq!(bytes,usize::MAX);
    }
    #[test]
    fn fixed_preparation_actual_bytes_and_typed_branches() -> Result<()> {
        let mut sizes=Vec::new();
        for ffn in [FfnKind::Dense {width:6144},FfnKind::Moe] {
            let mut w=NativeFixture::default();let mut bytes=0;
            let config=BlockConfig {ffn,..BlockConfig::LAYER2};
            let block=PreparedBlock::read(&mut w,config,&mut bytes)?;
            match block.ffn {
                PreparedBranch::Dense(_)=>{assert_eq!(w.fixed_reads,12);assert_eq!(bytes,31_868_088);},
                PreparedBranch::Moe {router,..}=>{
                    assert_eq!(w.fixed_reads,13);assert_eq!(router.len(),H*E);assert_eq!(bytes,16_270_520);
                },
            }
            assert_eq!(w.routed_reads,0); // No expert preparation, even transiently.
            sizes.push(bytes);
        }
        assert_eq!(2*sizes[0]+47*sizes[1],828_450_616);
        let mut w=NativeFixture {bad_scale:true,..Default::default()};
        assert!(PreparedProjection::read(&mut w,"attn_k",128,9,&mut 0).is_err());
        Ok(())
    }
    #[test]
    #[ignore = "requires an exclusive GB10 reservation; synthetic native fixed operands only"]
    fn cuda_resident_fixed_dense_moe_t1_t3_reset() -> Result<()> {
        for ffn in [FfnKind::Dense {width:64},FfnKind::Moe] {
            let config=BlockConfig {ffn,..BlockConfig::LAYER2};
            let mut w=NativeFixture::default();let mut bytes=0;
            let prepared=PreparedBlock::read(&mut w,config,&mut bytes)?;
            let admission=FixedAdmission::checked(1,bytes,1<<30)?;
            let resident=PreparedFixed {blocks:vec![prepared],admission}.upload()?;
            let mut cache=KvCache::new(3)?;
            let mut first:Option<Vec<f32>>=None;
            for (iteration,t) in [1,3,1].into_iter().enumerate() {
                let positions=&[0,512,513][..t];
                let input:Vec<f32>=(0..H*t).map(|i|((i%H*7+i/H*13)%23) as f32/32.0-11.0/32.0).collect();
                let mut streamed=NativeFixture::default();
                let expected=block_core(&mut streamed,&input,positions,config,ProjectionMode::PackedAll,None,true,None)?;
                let mut borrowed=NativeFixture {deny_fixed:true,..Default::default()};
                let trace=iteration<2;
                let actual=block_core(&mut borrowed,&input,positions,config,ProjectionMode::PackedAll,Some(&mut cache),trace,Some(resident.block(0)?))?;
                assert_eq!(borrowed.fixed_reads,0);assert_eq!(borrowed.routed_reads,streamed.routed_reads);
                assert_eq!(actual.residual,expected.residual);
                assert_eq!(actual.trace.stages.len(),if trace {expected.trace.stages.len()} else {0});
                for (a,b) in actual.trace.stages.iter().zip(&expected.trace.stages) {
                    assert_eq!(a.name,b.name);assert_eq!(a.dims,b.dims);
                    match (&a.values,&b.values) {
                        (Values::F32(a),Values::F32(b))=>assert_eq!(a,b),
                        (Values::I32(a),Values::I32(b))=>assert_eq!(a,b),
                        _=>panic!("stage type differs"),
                    }
                }
                assert!(actual.residual.iter().all(|v|v.is_finite()));
                if t==1 {if let Some(ref prior)=first {assert_eq!(&actual.residual,prior);} else {first=Some(actual.residual);}}
                resident.fence()?;cache.reset()?;
                assert_eq!(cache.len(),0);assert_eq!(resident.admission(),admission);
            }
            println!("resident fixed {ffn:?}: T1/T3/repeatedT1 value-exact streamed output; trace-on/off; reset retained operands; no fixed reload; routed calls unchanged");
        }
        Ok(())
    }
}
