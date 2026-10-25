//! Native Flint model, using Mary's existing model collection,
//! member/index/kind/weight relations, jsonfacts and typed Tensor leaves.
//!
//! Root members are metadata-config/tokenizer nodes, indexed blocks and globals. Block
//! members are named operations (attn_q, ffn_up, attn_norm, ...). Each operation
//! has a typed weight leaf; a projection additionally has a member of kind
//! post-matmul-scale with its OWN F32 weight leaf. Nothing folds that scale into
//! the NVFP4 global scale. Config is selected actual GGUF metadata as scalar
//! facts, not HF assumptions, a serialized opaque GGUF, or a second catalogue.
//!
//! Reopen retains a frozen ModelPileSnapshot. Each weight query joins those
//! relations at use; TensorView owns the mapped payload through upload. Packed
//! weights have no F32 fallback. Small F32/BF16 tensors alone materialize F32
//! for the existing diagnostic math tier. Full import preserves the actual
//! BF16 embedding/router, F32 norms/scales and packed NVFP4 projections.
//! Text fields use current native UTF8String handles: entity string arguments
//! retain their blobs; query string literals constrain those content handles.

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
pub mod import;

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
/// the payload. Indexed expert children are independent rank2 leaves.
fn checked_view<T:TensorElement,const R:usize>(blob:Blob<Tensor<T,R>>,dims:&[usize])->Result<TensorView> {
    ensure!(R>0 && R<=2 && dims.len()==R,"typed witness rank differs");
    let (_,_,elements)=row_selection(dims,None)?;
    // Full mapped embedding/head geometry is admitted separately from bounded
    // selected row materialization. The frozen snapshot owns these mapped bytes.
    ensure!(elements<=super::forward::VOCAB*super::math::H,"typed full geometry bound exceeded");
    let payload=T::payload_len(elements);
    ensure!(blob.bytes.len()==TENSOR_HEADER_LEN+payload,"typed tensor byte length differs");
    for (axis,&expected) in dims.iter().rev().enumerate() {
        let got=u64::from_le_bytes(blob.bytes[axis*8..axis*8+8].try_into()?);
        ensure!(got==u64::try_from(expected)?,"typed row-major axis{axis} differs");
    }
    leaf::read_leaf(blob)
}

fn row_range(view:&TensorView,first:usize,rows:usize)->Result<(usize,usize,usize)> {
    ensure!(view.dims().len()==2 && rows>0 && rows<=super::forward::HEAD_ROWS,"invalid bounded typed rows");
    let cols=usize::try_from(view.dims()[1])?;
    let total=usize::try_from(view.dims()[0])?;
    ensure!(first.checked_add(rows).is_some_and(|end|end<=total),"typed row range outside matrix");
    let start=first.checked_mul(cols).context("typed row offset overflow")?;
    let elements=rows.checked_mul(cols).context("typed row elements overflow")?;
    Ok((cols,start,elements))
}

/// Native planes are NOT interleaved rows. Copy the matching codes AND scale
/// ranges, retaining the original intrinsic global unchanged. Never dequantize.
pub(super) fn native_rows(view:&TensorView,first:usize,rows:usize)->Result<TensorView> {
    let (cols,start,elements)=row_range(view,first,rows)?;
    ensure!(cols%64==0,"native row alignment differs");
    let (codes,scales,global)=crate::nvfp4::split_payload(view.payload(),view.elems())?;
    ensure!(NVFP4::payload_len(elements)<=MAX_READ as usize+4,"native selected payload exceeds bound");
    let mut payload=Vec::with_capacity(NVFP4::payload_len(elements));
    payload.extend_from_slice(&codes[start/2..(start+elements)/2]);
    payload.extend_from_slice(&scales[start/16..(start+elements)/16]);
    payload.extend_from_slice(&global.to_le_bytes());
    leaf::read_leaf(leaf::leaf_blob::<NVFP4,2>([rows as u64,cols as u64],Bytes::from_source(payload))?)
}

fn bf16_rows(view:&TensorView,first:usize,rows:usize)->Result<Vec<f32>> {
    let (_,start,elements)=row_range(view,first,rows)?;
    ensure!(elements.checked_mul(2).is_some_and(|n|n<=MAX_READ as usize),"BF16 row bound exceeded");
    small_floats(&view.payload()[start*2..(start+elements)*2],30)
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
        Self::open_with_collection(path,root,None)
    }
    /// Select a descriptor independently from the model root. None retains
    /// sole-name discovery; explicit selection never falls back to another
    /// collection if its root is absent. All reads share this frozen observation.
    pub fn open_with_collection(path:&Path,root:Id,
        collection:Option<triblespace::core::collection::CollectionHandle>)->Result<Self> {
        // A read-only descriptor plus one frozen observation, not an append-
        // capable model helper. Another process still must not mutate its prefix.
        let store=crate::persist::read_model_pile_snapshot_read_only(path)?;
        let snapshot=match collection {
            Some(handle)=>{
                let collection=model_collection::open_model_collection(&store,handle)?;
                model_collection::snapshot_model_collection_for(&store,collection)?
            },
            None=>model_collection::snapshot_model_collection_in(&store)?,
        };
        Self::from_snapshot(snapshot,root)
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
    pub fn support_handles(&self)->Vec<String> {
        use triblespace::core::inline::encodings::hash::{Hash,Blake3};
        self.snapshot.support().members().map(|h|Hash::<Blake3>::to_hex(&h.transmute())).collect()
    }
    pub fn metadata(&self)->Result<Vec<MetadataEntry>> {
        Ok(serde_json::from_value(crate::jsonfacts::load_json(self.snapshot.facts(),
            self.snapshot.store(),self.config_node).map_err(|e|anyhow::anyhow!("{e}"))?)?)
    }
    pub fn select_block(&mut self,index:usize)->Result<()> {
        let config=BlockConfig::from_metadata(&self.metadata()?,index)?;
        let root=self.root;
        let block=find!(b:Id,pattern!(self.snapshot.facts(),[
            {root @ attrs::member:?b},{?b @ attrs::kind:"flint-block",attrs::index:index as u64}
        ])).next().context("model has no requested Flint block")?;
        self.block=block; self.config=config; Ok(())
    }
    #[cfg(feature="tokenizer")]
    pub fn tokenizer(&self)->Result<super::tokenizer::FlintTokenizer> {
        let root=self.root;
        for node in find!(node:Id,pattern!(self.snapshot.facts(),[
            {root @ attrs::member:_?metadata},
            {_?metadata @ attrs::kind:"flint-tokenizer-metadata",crate::jsonfacts::attrs::document_root:?node}
        ])) {
            let Ok(value)=crate::jsonfacts::load_json(self.snapshot.facts(),self.snapshot.store(),node) else {continue};
            let Ok(entries)=serde_json::from_value::<Vec<MetadataEntry>>(value) else {continue};
            if let Ok(codec)=super::tokenizer::FlintTokenizer::from_metadata(&entries) {return Ok(codec)}
        }
        anyhow::bail!("root has no supported Flint tokenizer metadata in this snapshot")
    }
    pub fn embedding(&self,id:usize)->Result<Vec<f32>> {
        let view=self.typed_at::<BF16,2>(self.root,"token_embd.weight",&[super::math::H,super::forward::VOCAB],None)?;
        bf16_rows(&view,id,1)
    }
    pub fn output_norm(&self)->Result<Vec<f32>> {
        small_floats(self.typed_at::<F32,1>(self.root,"output_norm.weight",&[super::math::H],None)?.payload(),0)
    }
    pub fn output_rows(&self,first:usize,rows:usize)->Result<(TensorView,f32)> {
        let (full,scale)=self.projection_at(self.root,"output",&[super::math::H,super::forward::VOCAB],None)?;
        Ok((native_rows(&full,first,rows)?,scale))
    }

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
        self.typed_at::<T,R>(self.block,suffix,dims,expert)
    }
    fn typed_at<T:TensorElement,const R:usize>(&self,block:Id,suffix:&str,dims:&[usize],expert:Option<usize>)->Result<TensorView> {
        row_selection(dims,expert)?;
        ensure!(expert.is_none(),"expert matrices require paired operation selection");
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

    fn projection_at(&self,parent:Id,kind:&str,dims:&[usize],expert:Option<usize>)->Result<(TensorView,f32)> {
        row_selection(dims,expert)?;
        ensure!(matches!(dims.len(),2|3),"invalid projection request");
        // Both arms project the SAME typed tuple. The expert child and scale
        // vector are joined through their operation, never independent lookups.
        let pair=if let Some(index)=expert {
            find!((w:Inline<inlineencodings::Handle<Tensor<NVFP4,2>>>,
                s:Inline<inlineencodings::Handle<Tensor<F32,1>>>),pattern!(self.snapshot.facts(),[
                {parent @ attrs::member:_?op},
                {_?op @ attrs::kind:kind,attrs::member:_?child,attrs::member:_?post},
                {_?child @ attrs::index:index as u64,attrs::weight:_?weight},
                {_?weight @ leaf::leaf::<NVFP4,2>():?w},
                {_?post @ attrs::kind:"post-matmul-scale",attrs::weight:_?scale},
                {_?scale @ leaf::leaf::<F32,1>():?s}
            ])).next()
        } else {
            find!((w:Inline<inlineencodings::Handle<Tensor<NVFP4,2>>>,
                s:Inline<inlineencodings::Handle<Tensor<F32,1>>>),pattern!(self.snapshot.facts(),[
                {parent @ attrs::member:_?op},
                {_?op @ attrs::kind:kind,attrs::weight:_?weight,attrs::member:_?post},
                {_?weight @ leaf::leaf::<NVFP4,2>():?w},
                {_?post @ attrs::kind:"post-matmul-scale",attrs::weight:_?scale},
                {_?scale @ leaf::leaf::<F32,1>():?s}
            ])).next()
        }.with_context(||format!("no paired native projection {kind}"))?;
        let weight:Blob<Tensor<NVFP4,2>>=self.snapshot.store().get(pair.0).map_err(|e|anyhow::anyhow!("{e}"))?;
        let scale:Blob<Tensor<F32,1>>=self.snapshot.store().get(pair.1).map_err(|e|anyhow::anyhow!("{e}"))?;
        let weight=checked_view(weight,&dims[..2])?;
        let scales=checked_view(scale,&[expert.map(|_|dims[2]).unwrap_or(1)])?;
        let scales=small_floats(scales.payload(),0)?;
        Ok((weight,scales[expert.unwrap_or(0)]))
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
        ensure!(view.payload().len()<=MAX_READ as usize,"small typed parameter exceeds bound");
        small_floats(view.payload(),ty)
    }
    fn get_nvfp4(&mut self,_:&str,_:&[usize],_:Option<usize>)->Result<Vec<u8>> {
        anyhow::bail!("native planes are not GGUF blocks; use get_packed")
    }
    fn get_packed(&mut self,suffix:&str,dims:&[usize],expert:Option<usize>)->Result<PackedWeight> {
        Ok(PackedWeight::Native(self.typed::<NVFP4,2>(suffix,dims,expert)?))
    }
    fn packed_projection(&mut self,suffix:&str,dims:&[usize],expert:Option<usize>)->Result<(PackedWeight,Option<f32>)> {
        let (view,scale)=self.projection_at(self.block,suffix,dims,expert)?;
        ensure!(view.payload().len()<=MAX_READ as usize+4,"selected native matrix exceeds bound");
        Ok((PackedWeight::Native(view),Some(scale)))
    }
    fn paired_scales(&self)->bool {true}
}

#[cfg(test)]
mod tests {
    use super::*;
    fn with_snapshot(fragment:Fragment,check:impl FnOnce(PileWeights)->Result<()>)->Result<()> {
        let root=fragment.root().context("fixture root")?;
        let path=std::env::temp_dir().join(format!("mary-flint-full-{}.pile",fucid().id));
        std::fs::OpenOptions::new().write(true).create_new(true).open(&path)?;
        let mut pile=Pile::open(&path).map_err(|e|anyhow::anyhow!("{e}"))?;
        let key=ed25519_dalek::SigningKey::from_bytes(&[42;32]);
        model_collection::publish_model_fragment(&mut pile,&key,fragment)?;
        pile.close().map_err(|e|anyhow::anyhow!("{e}"))?;
        let weights=PileWeights::open(&path,root)?;
        // Native text facts must carry resolvable UTF8 bytes, not merely hashes
        // that happen to satisfy a known string query.
        for attribute in [&*attrs::kind,&*attrs::source,&*attrs::quantization,&*metadata::name] {
            for handle in find!(h:Inline<inlineencodings::Handle<blobencodings::UTF8String>>,
                pattern!(weights.snapshot.facts(),[{_?entity @ attribute:?h}])) {
                let text:anybytes::View<str>=weights.snapshot.store().get(handle)
                    .map_err(|e|anyhow::anyhow!("native text attachment: {e}"))?;
                ensure!(!text.is_empty(),"empty fixture text");
            }
        }
        check(weights)?;
        std::fs::remove_file(path)?; Ok(())
    }
    fn packed_fixture(code:u8)->Result<Fragment> {
        let mut raw=[code;36];raw[..4].fill(0x38);
        Ok(leaf_fragment(typed::to_tensor(&raw,[1,64])?))
    }
    fn scale_fixture(values:&[f32])->Result<Fragment> {
        let bytes=values.iter().flat_map(|v|v.to_le_bytes()).collect::<Vec<_>>();
        let weight=leaf_fragment(leaf::leaf_blob::<F32,1>([values.len() as u64],Bytes::from_source(bytes))?);
        Ok(entity! {attrs::kind:"post-matmul-scale",attrs::weight*:weight})
    }

    #[test]
    fn explicit_collection_keeps_same_named_models_separate()->Result<()> {
        use triblespace::core::collection::{AdmissionPolicy,CollectionPolicy};
        let path=std::env::temp_dir().join(format!("mary-flint-collections-{}.pile",fucid().id));
        std::fs::OpenOptions::new().write(true).create_new(true).open(&path)?;
        let mut pile=Pile::open(&path).map_err(|e|anyhow::anyhow!("{e}"))?;
        // Ephemeral fixture coordinates only, not a retained model catalogue.
        let mut models=Vec::new();
        for (key_byte,value) in [(41,1.0f32),(42,2.0f32)] {
            let key=ed25519_dalek::SigningKey::from_bytes(&[key_byte;32]);
            let policy=CollectionPolicy::new(AdmissionPolicy::direct(key.verifying_key()),AdmissionPolicy::direct(key.verifying_key()));
            let collection=pile.collection(model_collection::mary_model_graph_name(),policy)?;
            let weight=leaf_fragment(leaf::leaf_blob::<F32,1>([1],Bytes::from_source(value.to_le_bytes().to_vec()))?);
            let parameters=entity! {attrs::kind:"probe",attrs::weight*:weight};
            let fragment=model_fragment(parameters,&super::super::config::tests::fixture())?;
            let root=fragment.root().context("fixture root")?;
            model_collection::publish_model_fragment_into(&mut pile,&key,collection,fragment)?;
            models.push((collection.handle(),root,value));
        }
        pile.close().map_err(|e|anyhow::anyhow!("{e}"))?;
        assert!(PileWeights::open(&path,models[0].1).is_err()); // Name is ambiguous.
        for &(handle,root,value) in &models {
            let mut weights=PileWeights::open_with_collection(&path,root,Some(handle))?;
            assert_eq!(model_collection::parse_collection_handle(&weights.collection())?,handle);
            assert_eq!(weights.get("probe.weight",&[1],0,None)?,[value]);
            assert_eq!(weights.support_members(),1);
        }
        assert!(PileWeights::open_with_collection(&path,models[1].1,Some(models[0].0)).is_err());
        assert!(PileWeights::open_with_collection(&path,models[0].1,Some(models[1].0)).is_err());
        std::fs::remove_file(path)?;
        Ok(())
    }

    #[test]
    fn native_rows_select_both_planes_global_and_bf16_bits()->Result<()> {
        let mut raw=Vec::new();
        for row in 0..4u8 {
            let mut block=[(row+1)*0x11;36];block[..4].fill(0x30+row);raw.extend(block);
        }
        let full=checked_view(typed::to_tensor(&raw,[4,64])?,&[64,4])?;
        let selected=native_rows(&full,1,2)?;
        assert_eq!(&selected.payload()[..],typed::payload(&raw[36..108],128)?);
        assert_eq!(selected.dims(),[2,64]);
        assert!(native_rows(&full,3,2).is_err());
        assert!(native_rows(&full,0,0).is_err());
        assert!(native_rows(&full,usize::MAX,1).is_err());
        assert!(native_rows(&full,0,8193).is_err());
        // Preserve, rather than silently set, an intrinsic global in row slicing.
        let mut payload=full.payload().to_vec();let end=payload.len();
        payload[end-4..].copy_from_slice(&0.75f32.to_le_bytes());
        let full=checked_view(leaf::leaf_blob::<NVFP4,2>([4,64],Bytes::from_source(payload))?,&[64,4])?;
        let selected=native_rows(&full,2,1)?;
        assert_eq!(crate::nvfp4::split_payload(selected.payload(),64)?.2,0.75);
        let raw=[0x3f80u16,0x4000,0x8000,0x4040,0xc000,0x0000].into_iter().flat_map(u16::to_le_bytes).collect::<Vec<_>>();
        let embedding=checked_view(leaf::leaf_blob::<BF16,2>([3,2],Bytes::from_source(raw))?,&[2,3])?;
        assert_eq!(bf16_rows(&embedding,1,1)?.iter().map(|v|v.to_bits()).collect::<Vec<_>>(),[0x80000000,0x40400000]);
        assert!(bf16_rows(&embedding,3,1).is_err());
        Ok(())
    }

    #[test]
    fn paired_alternatives_and_indexed_experts_never_cross_operations()->Result<()> {
        let mut params=Fragment::empty();
        for (code,scale) in [(0x11,2.0),(0x22,3.0)] {
            let weight=packed_fixture(code)?;let post=scale_fixture(&[scale])?;
            params+=entity! {attrs::kind:"native-projection-kind-longer-than-32-bytes",attrs::weight*:weight,attrs::member*:post};
            let mut children=Fragment::empty();
            for index in 0..2u64 {
                let weight=packed_fixture(code+index as u8*0x22)?;
                children+=entity! {attrs::index:index,attrs::weight*:weight};
            }
            let post=scale_fixture(&[scale,scale+4.0])?;
            let members=children+post;
            params+=entity! {attrs::kind:"experts",attrs::member*:members};
        }
        // Incomplete alternatives must not mix a stray weight and stray scale.
        let stray=packed_fixture(0x33)?;
        params+=entity! {attrs::kind:"incomplete",attrs::weight*:stray};
        let post=scale_fixture(&[9.0])?;
        params+=entity! {attrs::kind:"incomplete",attrs::member*:post};
        with_snapshot(model_fragment(params,&super::super::config::tests::fixture())?,|mut weights| {
            let (PackedWeight::Native(view),Some(scale))=weights.packed_projection("native-projection-kind-longer-than-32-bytes",&[64,1],None)?
                else {panic!("native pair missing")};
            let code=view.payload()[0];
            assert!((code==0x11 && scale==2.0)||(code==0x22 && scale==3.0));
            let (PackedWeight::Native(view),Some(scale))=weights.packed_projection("experts",&[64,1,2],Some(1))?
                else {panic!("native expert pair missing")};
            assert!((view.payload()[0]==0x33 && scale==6.0)||(view.payload()[0]==0x44 && scale==7.0));
            assert!(weights.packed_projection("experts",&[64,1,2],Some(2)).is_err());
            assert!(weights.packed_projection("incomplete",&[64,1],None).is_err());
            Ok(())
        })
    }

    #[test]
    fn opaque_root_block_selection_and_tokenizer_metadata_relations()->Result<()> {
        let mut fragment=model_fragment(Fragment::empty(),&super::super::config::tests::fixture())?;
        let original=fragment.root().unwrap();
        let members:Vec<Id>=find!(m:Id,pattern!(fragment.facts(),[{original @ attrs::member:?m}])).collect();
        let alias=fucid();
        let last=entity! {attrs::kind:"flint-block",attrs::index:48u64};
        fragment+=entity! {&alias @ attrs::member*:members,attrs::member*:last,metadata::name:"opaque native alias"};
        let tiny=vec![
            MetadataEntry {key:"tokenizer.ggml.add_bos_token".into(),gguf_type:7,value:crate::formats::gguf::index::MetadataValue::Bool(false)},
            MetadataEntry {key:"tokenizer.ggml.eog_token_ids".into(),gguf_type:9,value:crate::formats::gguf::index::MetadataValue::I32Array(vec![150001,150003,150005,150018])},
        ];
        let mut metadata_fragment=Fragment::empty();let mut facts=TribleSet::new();
        let node=crate::jsonfacts::save_json(&serde_json::to_value(&tiny)?,metadata_fragment.blobs_mut(),&mut facts)
            .map_err(|e|anyhow::anyhow!("{e}"))?;
        *metadata_fragment.facts_mut()+=facts;
        metadata_fragment+=entity! {attrs::kind:"flint-tokenizer-metadata",crate::jsonfacts::attrs::document_root:node};
        fragment+=entity! {&alias @ attrs::member*:metadata_fragment,metadata::name:"unknown annotation"};
        let (_,facts,metafacts,blobs)=fragment.into_parts();
        let fragment=Fragment::rooted_from_parts(alias.id,facts,metafacts,blobs);
        with_snapshot(fragment,|mut weights| {
            assert_eq!(weights.root,alias.id);
            weights.select_block(48)?;assert_eq!(weights.config.block,48);
            assert!(!weights.config.rope);assert!(weights.config.sliding_window.is_none());
            assert!(weights.select_block(49).is_err());
            let root=weights.root;
            let saved=find!(n:Id,pattern!(weights.snapshot.facts(),[
                {root @ attrs::member:_?m},{_?m @ attrs::kind:"flint-tokenizer-metadata",crate::jsonfacts::attrs::document_root:?n}
            ])).next().context("root-linked tokenizer missing")?;
            let restored:Vec<MetadataEntry>=serde_json::from_value(crate::jsonfacts::load_json(
                weights.snapshot.facts(),weights.snapshot.store(),saved).map_err(|e|anyhow::anyhow!("{e}"))?)?;
            assert_eq!(restored,tiny); // Boolean flags/types/order survive semantic storage.
            #[cfg(feature="tokenizer")]
            assert!(weights.tokenizer().is_err()); // Tiny graph is deliberately NOT a valid full vocabulary.
            let mut source=super::super::forward::Source::Native(weights);
            assert!(super::super::forward::run(&mut source,&[0],&[0],
                super::super::forward::Backend::Cpu,|_,_,_|Ok(())).is_err());
            Ok(())
        })
    }
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
