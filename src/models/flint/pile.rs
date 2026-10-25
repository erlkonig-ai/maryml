//! Block0-only native model witness, using Mary's existing model collection,
//! member/index/kind/weight relations, jsonfacts and typed Tensor leaves.
//!
//! Root members are one metadata-config node and one indexed block. Block
//! members are named operations (attn_q, ffn_up, attn_norm, ...). Each operation
//! has a typed weight leaf; a projection additionally has a member of kind
//! post-matmul-scale with its OWN F32 weight leaf. Nothing folds that scale into
//! the NVFP4 global scale. Config is selected actual GGUF metadata as scalar
//! facts, not HF assumptions, a serialized opaque GGUF, or a second catalogue.
//!
//! Reopen retains a frozen ModelPileSnapshot. Each weight query joins those
//! relations at use; TensorView owns the mapped payload through upload. Packed
//! weights have no F32 fallback. Small F32/BF16 tensors alone materialize F32
//! for the existing diagnostic math tier. Actual dense block0 has no BF16
//! router; BF16 support is exercised by synthetic tests, not invented weights.

use std::path::Path;
use anyhow::{Context,Result,ensure};
use anybytes::Bytes;
use serde::Serialize;
use triblespace::core::blob::{Blob,encodings::tensor::{Tensor,TensorElement,TensorView,TENSOR_HEADER_LEN,elements::{BF16,F32,NVFP4}}};
use triblespace::core::metadata;
use triblespace::prelude::*;
use crate::{format::attrs,leaf,model_collection::{self,ModelPileSnapshot}};
use crate::formats::gguf::{index::{MetadataEntry,MAX_READ},nvfp4::typed};
use super::{config::{BlockConfig,FfnKind},weights::{GgufWeights,PackedWeight,ReadReceipt,ReceiptPolicy,Weights,row_selection}};

// Exact bounded inventory, not an index of stored model entities. GGML dims.
pub const PROJECTIONS:[(&str,usize,usize);7]=[
    ("attn_q",2048,4096),("attn_k",2048,512),("attn_v",2048,512),
    ("attn_output",4096,2048),("ffn_up",2048,6144),("ffn_gate",2048,6144),
    ("ffn_down",6144,2048),
];
pub const VECTORS:[(&str,usize);5]=[
    ("attn_norm",2048),("attn_q_norm",128),("attn_k_norm",128),
    ("attn_sinks",32),("ffn_norm",2048),
];
const BLOCK0:BlockConfig=BlockConfig {block:0,sliding_window:None,rope:false,ffn:FfnKind::Dense {width:6144}};

#[derive(Serialize)]
pub struct ImportedTensor {
    pub source:ReadReceipt, pub row_major_dims:Vec<u64>,
    pub tensor_bytes:usize, pub tensor_blake3:String, pub payload_blake3:String,
}

fn leaf_fragment<T:TensorElement,const R:usize>(blob:Blob<Tensor<T,R>>)->Fragment {
    let mut fragment=Fragment::empty();
    let handle=fragment.put(blob);
    fragment+=entity! { leaf::leaf::<T,R>():handle };
    fragment
}

/// Check the shape BEFORE invoking the core TensorView reader's dim product.
/// The caller's bounded expected shape, not an unchecked stored header, sizes
/// the payload. This reader handles rank1/2 only; selected experts are not yet
/// part of this block0 witness.
fn checked_view<T:TensorElement,const R:usize>(blob:Blob<Tensor<T,R>>,dims:&[usize])->Result<TensorView> {
    ensure!(R>0 && R<=2 && dims.len()==R,"typed witness rank differs");
    let (_,_,elements)=row_selection(dims,None)?;
    ensure!(elements<=MAX_READ as usize*2,"typed witness element bound exceeded");
    let payload=T::payload_len(elements);
    ensure!(payload<=MAX_READ as usize+4,"typed witness payload bound exceeded");
    ensure!(blob.bytes.len()==TENSOR_HEADER_LEN+payload,"typed tensor byte length differs");
    for (axis,&expected) in dims.iter().rev().enumerate() {
        let got=u64::from_le_bytes(blob.bytes[axis*8..axis*8+8].try_into()?);
        ensure!(got==u64::try_from(expected)?,"typed row-major axis{axis} differs");
    }
    leaf::read_leaf(blob)
}

fn small_floats(raw:&[u8],ty:u32)->Result<Vec<f32>> {
    let width=match ty {0=>4,30=>2,_=>anyhow::bail!("only F32/BF16 small weights may widen")};
    ensure!(raw.len()%width==0,"unaligned small tensor");
    let values:Vec<f32>=raw.chunks_exact(width).map(|c|if ty==0 {
        f32::from_le_bytes(c.try_into().unwrap())
    } else { f32::from_bits(u32::from(u16::from_le_bytes(c.try_into().unwrap()))<<16) }).collect();
    ensure!(values.iter().all(|v|v.is_finite()),"non-finite typed F32/BF16 weight");
    Ok(values)
}

fn selected_leaf(w:&mut GgufWeights,name:&str,dims:&[usize],ty:u32,
    receipts:&mut Vec<ImportedTensor>)->Result<Fragment> {
    let (raw,_,receipt)=w.read_weight(name,dims,ty,None)?;
    let source=receipt.context("import requires actual raw read receipts")?;
    let shape:Vec<u64>=dims.iter().rev().map(|&v|v as u64).collect();
    // All input reads retain IndexedFile's16MiB bound, shape/type/stat checks.
    let (fragment,bytes,digest,payload_digest)=if ty==40 {
        ensure!(dims.len()==2,"packed import requires a matrix");
        let blob=typed::to_tensor(&raw,[shape[0],shape[1]])?;
        let bytes=blob.bytes.len(); let digest=blake3::hash(&blob.bytes).to_hex().to_string();
        let payload_digest=blake3::hash(&blob.bytes[TENSOR_HEADER_LEN..]).to_hex().to_string();
        (leaf_fragment(blob),bytes,digest,payload_digest)
    } else {
        ensure!(ty==0 && dims.len()==1,"actual block0 import expects F32 vectors/scalars");
        small_floats(&raw,ty)?;
        let blob=leaf::leaf_blob::<F32,1>([shape[0]],Bytes::from_source(raw))?;
        let bytes=blob.bytes.len(); let digest=blake3::hash(&blob.bytes).to_hex().to_string();
        let payload_digest=blake3::hash(&blob.bytes[TENSOR_HEADER_LEN..]).to_hex().to_string();
        (leaf_fragment(blob),bytes,digest,payload_digest)
    };
    receipts.push(ImportedTensor {source,row_major_dims:shape,tensor_bytes:bytes,tensor_blake3:digest,payload_blake3:payload_digest});
    Ok(fragment)
}

/// Construct the complete bounded fragment BEFORE publishing anything. This
/// function performs selected GGUF reads, but no pile writes. The caller must
/// admit the file identity before calling it and publish to a fresh scratch pile.
pub fn import_block0(w:&mut GgufWeights)->Result<(Fragment,Vec<ImportedTensor>)> {
    ensure!(w.config==BLOCK0 && BlockConfig::from_metadata(&w.file.index.metadata,0)?==BLOCK0,
        "native witness is the metadata-bound full-attention/no-RoPE dense block0 only");
    ensure!(w.receipt_policy()==ReceiptPolicy::Collect,"import must collect source ranges/digests");
    let mut parameters=Fragment::empty(); let mut receipts=Vec::new();
    for (name,ni,no) in PROJECTIONS {
        let weight=selected_leaf(w,&format!("{name}.weight"),&[ni,no],40,&mut receipts)?;
        let scale=selected_leaf(w,&format!("{name}.scale"),&[1],0,&mut receipts)?;
        let scale=entity! { attrs::kind:"post-matmul-scale",attrs::weight*:scale };
        parameters+=entity! { attrs::kind:name,attrs::weight*:weight,attrs::member*:scale };
    }
    for (name,n) in VECTORS {
        let weight=selected_leaf(w,&format!("{name}.weight"),&[n],0,&mut receipts)?;
        parameters+=entity! { attrs::kind:name,attrs::weight*:weight };
    }
    let root=model_fragment(parameters,&w.file.index.metadata)?;
    w.file.check_identity()?;
    Ok((root,receipts))
}

fn model_fragment(parameters:Fragment,entries:&[MetadataEntry])->Result<Fragment> {
    let block=entity! { attrs::kind:"flint-block",attrs::index:0u64,attrs::member*:parameters };
    let mut config=Fragment::empty(); let mut facts=TribleSet::new();
    let config_root=crate::jsonfacts::save_json(&serde_json::to_value(entries)?,
        config.blobs_mut(),&mut facts).map_err(|e|anyhow::anyhow!("config facts: {e}"))?;
    *config.facts_mut()+=facts;
    config+=entity! { attrs::kind:"flint-config",crate::jsonfacts::attrs::document_root:config_root };
    let members=block+config;
    let root=entity! { attrs::member*:members };
    let id=root.root().context("native model root")?;
    let mut root=root;
    root+=entity! { ExclusiveId::force_ref(&id) @ attrs::source:"flint GGUF block0 witness",
        attrs::quantization:"native-nvfp4",metadata::name:"Flint typed block0" };
    Ok(root)
}

pub struct PileWeights {
    snapshot:ModelPileSnapshot,
    pub root:Id, pub block:Id, pub config_node:Id, pub config:BlockConfig,
}
impl PileWeights {
    pub fn open(path:&Path,root:Id)->Result<Self> {
        Self::from_snapshot(model_collection::load_model_collection_local_latest(path)?,root)
    }
    pub fn from_snapshot(snapshot:ModelPileSnapshot,root:Id)->Result<Self> {
        // Positive, typed membership joins. Unknown sibling modules/annotations
        // are irrelevant. Do not recompute root IDs or require a closed fact set.
        let mut selected=None;
        for (block,node) in find!((block:Id,node:Id),pattern!(snapshot.facts(),[
            {root @ attrs::member:?block,attrs::member:_?config},
            {?block @ attrs::kind:"flint-block",attrs::index:0u64},
            {_?config @ attrs::kind:"flint-config",crate::jsonfacts::attrs::document_root:?node}
        ])) {
            let Ok(value)=crate::jsonfacts::load_json(snapshot.facts(),snapshot.store(),node) else {continue};
            let Ok(entries)=serde_json::from_value::<Vec<MetadataEntry>>(value) else {continue};
            if BlockConfig::from_metadata(&entries,0).is_ok_and(|c|c==BLOCK0) {
                selected=Some((block,node)); break;
            }
        }
        let (block,config_node)=selected.context("root has no supported metadata-bound block0 binding")?;
        Ok(Self {snapshot,root,block,config_node,config:BLOCK0})
    }
    pub fn collection(&self)->String {
        use triblespace::core::inline::encodings::hash::{Hash,Blake3};
        Hash::<Blake3>::to_hex(&self.snapshot.support().collection().handle().transmute())
    }
    pub fn support_members(&self)->usize { self.snapshot.support().len() }

    /// Diagnostic only; runtime get/get_packed do not hash per-weight bytes.
    pub fn check_imported(&self,receipts:&[ImportedTensor])->Result<()> {
        for receipt in receipts {
            let name=receipt.source.name.strip_prefix("blk.0.").context("not a block0 import receipt")?;
            let dims:Vec<usize>=receipt.source.ggml_dims.iter().map(|&d|usize::try_from(d)).collect::<std::result::Result<_,_>>()?;
            let view=match receipt.source.ggml_type {
                40=>self.typed::<NVFP4,2>(name,&dims,None)?,
                0=>self.typed::<F32,1>(name,&dims,None)?,
                _=>anyhow::bail!("unexpected actual block0 imported dtype"),
            };
            ensure!(view.dims()==receipt.row_major_dims && view.payload().len()+TENSOR_HEADER_LEN==receipt.tensor_bytes,
                "reopened tensor geometry differs: {name}");
            ensure!(blake3::hash(view.payload()).to_hex().as_str()==receipt.payload_blake3,
                "reopened payload digest differs: {name}");
        }
        Ok(())
    }

    fn typed<T:TensorElement,const R:usize>(&self,suffix:&str,dims:&[usize],expert:Option<usize>)->Result<TensorView> {
        row_selection(dims,expert)?;
        ensure!(expert.is_none(),"typed block0 witness does not import expert banks/slices");
        let block=self.block;
        ensure!(suffix.len()<=32,"parameter kind exceeds ShortString bound");
        let h=if let Some(kind)=suffix.strip_suffix(".weight") {
            find!(h:Inline<inlineencodings::Handle<Tensor<T,R>>>,pattern!(self.snapshot.facts(),[
                {block @ attrs::member:_?op},
                {_?op @ attrs::kind:kind,attrs::weight:_?leaf},
                {_?leaf @ leaf::leaf::<T,R>():?h}
            ])).next()
        } else if let Some(kind)=suffix.strip_suffix(".scale") {
            find!(h:Inline<inlineencodings::Handle<Tensor<T,R>>>,pattern!(self.snapshot.facts(),[
                {block @ attrs::member:_?op},
                {_?op @ attrs::kind:kind,attrs::member:_?scale},
                {_?scale @ attrs::kind:"post-matmul-scale",attrs::weight:_?leaf},
                {_?leaf @ leaf::leaf::<T,R>():?h}
            ])).next()
        } else { anyhow::bail!("unsupported typed parameter suffix") };
        let h=h.with_context(||format!("no selected typed parameter {suffix}, rank{R}"))?;
        let blob:Blob<Tensor<T,R>>=self.snapshot.store().get(h)
            .map_err(|e|anyhow::anyhow!("read typed parameter {suffix}: {e}"))?;
        checked_view(blob,dims)
    }
}

impl Weights for PileWeights {
    fn get(&mut self,suffix:&str,dims:&[usize],ty:u32,expert:Option<usize>)->Result<Vec<f32>> {
        let view=match (ty,dims.len()) {
            (0,1)=>self.typed::<F32,1>(suffix,dims,expert)?,
            (0,2)=>self.typed::<F32,2>(suffix,dims,expert)?,
            (30,1)=>self.typed::<BF16,1>(suffix,dims,expert)?,
            (30,2)=>self.typed::<BF16,2>(suffix,dims,expert)?,
            _=>anyhow::bail!("typed block witness refuses packed-to-F32/unsupported dtype or rank"),
        };
        small_floats(view.payload(),ty)
    }
    fn get_nvfp4(&mut self,_:&str,_:&[usize],_:Option<usize>)->Result<Vec<u8>> {
        anyhow::bail!("native planes are not GGUF blocks; use get_packed")
    }
    fn get_packed(&mut self,suffix:&str,dims:&[usize],expert:Option<usize>)->Result<PackedWeight> {
        Ok(PackedWeight::Native(self.typed::<NVFP4,2>(suffix,dims,expert)?))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn scratch_pile_reopen_uses_typed_relations_and_opaque_root()->Result<()> {
        let raw:Vec<u8>=[0x8000u16,0x3f80,0x4000,0xc000].into_iter().flat_map(u16::to_le_bytes).collect();
        let weight=leaf_fragment(leaf::leaf_blob::<BF16,2>([2,2],Bytes::from_source(raw))?);
        let scale=leaf_fragment(leaf::leaf_blob::<F32,1>([1],Bytes::from_source(0.75f32.to_le_bytes().to_vec()))?);
        let scale=entity! {attrs::kind:"post-matmul-scale",attrs::weight*:scale};
        let mut params=entity! {attrs::kind:"probe",attrs::weight*:weight,attrs::member*:scale};
        let mut raw=[0x98;36]; raw[..4].copy_from_slice(&[0x38,0x7f,0x80,0xb8]);
        let weight=leaf_fragment(typed::to_tensor(&raw,[1,64])?);
        params+=entity! {attrs::kind:"packed",attrs::weight*:weight};
        let mut fragment=model_fragment(params,&super::super::config::tests::fixture())?;
        let original=fragment.root().unwrap();
        // Entity IDs are opaque. An independently named root with these
        // relations works, and unrelated annotations do not invalidate it.
        let members:Vec<Id>=find!(m:Id,pattern!(fragment.facts(),[{original @ attrs::member:?m}])).collect();
        let alias=fucid();
        fragment+=entity! {&alias @ attrs::member*:members,metadata::name:"unrelated alias annotation"};
        let future=entity! {attrs::kind:"future-block-annotation"};
        fragment+=entity! {&alias @ attrs::member*:future};
        let path=std::env::temp_dir().join(format!("mary-flint-typed-{}.pile",alias.id));
        std::fs::OpenOptions::new().write(true).create_new(true).open(&path)?;
        let mut pile=Pile::open(&path).map_err(|e|anyhow::anyhow!("{e}"))?;
        // Public synthetic-test key, never a production credential.
        let key=ed25519_dalek::SigningKey::from_bytes(&[42;32]);
        model_collection::publish_model_fragment(&mut pile,&key,fragment)?;
        pile.close().map_err(|e|anyhow::anyhow!("{e}"))?;
        let mut weights=PileWeights::open(&path,alias.id)?;
        assert_eq!(weights.config,BLOCK0);
        assert_eq!(weights.get("probe.weight",&[2,2],30,None)?.iter().map(|v|v.to_bits()).collect::<Vec<_>>(),
            [0x80000000,0x3f800000,0x40000000,0xc0000000]);
        assert_eq!(weights.get("probe.scale",&[1],0,None)?,[0.75]);
        assert!(weights.get("probe.weight",&[2,2],0,None).is_err());
        assert!(weights.get("probe.weight",&[4,1],30,None).is_err());
        assert!(weights.get("probe.weight",&[2,2,256],30,Some(0)).is_err());
        assert!(weights.get("packed.weight",&[64,1],40,None).is_err());
        assert!(weights.get_nvfp4("packed.weight",&[64,1],None).is_err());
        let PackedWeight::Native(view)=weights.get_packed("packed.weight",&[64,1],None)? else {panic!("not native")};
        assert_eq!(&view.payload()[..],typed::payload(&raw,64)?);
        drop(weights); // The payload view continues to own its mapped bytes.
        assert_eq!(&view.payload()[..],typed::payload(&raw,64)?);
        drop(view);
        std::fs::remove_file(path)?;
        Ok(())
    }

    #[test]
    fn typed_small_weights_keep_bf16_f32_and_refuse_bad_shapes()->Result<()> {
        let raw:Vec<u8>=[0x0000u16,0x8000,0x3f80,0xbf80].into_iter().flat_map(u16::to_le_bytes).collect();
        let blob=leaf::leaf_blob::<BF16,2>([2,2],Bytes::from_source(raw.clone()))?;
        let view=checked_view(blob.clone(),&[2,2])?;
        let actual=small_floats(view.payload(),30)?;
        assert_eq!(actual.iter().map(|v|v.to_bits()).collect::<Vec<_>>(),[0,0x80000000,0x3f800000,0xbf800000]);
        assert_eq!(&view.payload()[..],raw);
        assert!(checked_view(blob.clone(),&[4,1]).is_err());
        let mut malformed=blob.bytes.to_vec(); malformed[..8].copy_from_slice(&u64::MAX.to_le_bytes());
        assert!(checked_view::<BF16,2>(Blob::new(Bytes::from_source(malformed)),&[2,2]).is_err());
        assert!(small_floats(&[0],30).is_err());
        assert!(small_floats(&f32::NAN.to_le_bytes(),0).is_err());
        assert!(small_floats(&0x7f80u16.to_le_bytes(),30).is_err());
        assert!(small_floats(&[0;36],40).is_err());
        let raw:Vec<u8>=[-0.0f32,1.5].into_iter().flat_map(f32::to_le_bytes).collect();
        let view=checked_view(leaf::leaf_blob::<F32,1>([2],Bytes::from_source(raw))?,&[2])?;
        assert_eq!(small_floats(view.payload(),0)?[0].to_bits(),(-0.0f32).to_bits());
        Ok(())
    }
}
