//! Explicit fresh-pile ingress. Put/drop one typed leaf at a time; the returned
//! Fragment retains facts/handles, never the model's payloads. Existing Tensor
//! encoding and core Pile content validation own the stored blob contract.
use super::*;
use crate::formats::gguf::index::TensorInfo;

#[derive(Default,Serialize)]
pub struct ImportSummary {
    pub tensor_leaves:u64, pub selected_source_bytes:u64,
    pub typed_tensor_bytes:u64, pub largest_payload_bytes:usize,
    pub largest_source_read_bytes:usize,
}

fn copy_packed_rows(payload:&mut [u8],elements:usize,cols:usize,row:usize,rows:usize,raw:&[u8])->Result<()> {
    let start=row.checked_mul(cols).context("chunk offset overflow")?;
    let n=rows.checked_mul(cols).context("chunk element overflow")?;
    ensure!(cols%64==0 && start.checked_add(n).is_some_and(|end|end<=elements)
        && payload.len()==NVFP4::payload_len(elements),"invalid native ingress chunk");
    let converted=typed::payload(raw,n)?;
    payload[start/2..(start+n)/2].copy_from_slice(&converted[..n/2]);
    payload[elements/2+start/16..elements/2+(start+n)/16]
        .copy_from_slice(&converted[n/2..n/2+n/16]);
    Ok(())
}

/// The largest permitted leaf is the known BF16 embedding (614,989,824B).
/// tensor_blob copies this payload into its256-byte-header blob, so the largest
/// two-buffer construction is1,229,979,904B, not a full-model anonymous arena.
fn store_selected<T:TensorElement,const R:usize>(w:&mut GgufWeights,pile:&mut Pile,
    name:&str,ggml_dims:&[usize],ty:u32,shape:[u64;R],first:usize,rows:usize,
    summary:&mut ImportSummary)->Result<Fragment> {
    let info:TensorInfo=w.file.index.tensor(name)?.clone();
    ensure!(info.ggml_type==ty && info.dims==ggml_dims.iter().map(|&n|n as u64).collect::<Vec<_>>(),
        "import tensor shape/type differs: {name}");
    ensure!(matches!(ty,0|30|40) && rows>0 && R>0 && R<=2
        && !ggml_dims.is_empty() && ggml_dims.iter().all(|&d|d>0)
        && shape.iter().all(|&d|d>0),"unsupported import geometry");
    let elements=shape.iter().try_fold(1usize,|n,&d|n.checked_mul(usize::try_from(d)?)
        .context("import element overflow"))?;
    ensure!(elements<=super::super::forward::VOCAB*super::super::math::H,"import leaf geometry bound");
    let cols=ggml_dims[0];
    ensure!(rows.checked_mul(cols)==Some(elements),"selected shape differs");
    let total_rows=info.elements()?.checked_div(cols as u64).context("empty row width")?;
    ensure!(first.checked_add(rows).is_some_and(|end|end as u64<=total_rows),"import row range outside source");
    let len=T::payload_len(elements);
    let mut payload=vec![0u8;len];
    let row_bytes=usize::try_from(info.row_bytes()?)?;
    let chunk_rows=(MAX_READ as usize/row_bytes).min(super::super::forward::HEAD_ROWS);
    ensure!(chunk_rows>0,"one source row exceeds read bound");
    for row in (0..rows).step_by(chunk_rows) {
        let count=(rows-row).min(chunk_rows);
        let (_,raw)=w.file.read_rows(name,(first+row) as u64,count as u64)?;
        summary.selected_source_bytes=summary.selected_source_bytes.checked_add(raw.len() as u64)
            .context("import source byte count overflow")?;
        summary.largest_source_read_bytes=summary.largest_source_read_bytes.max(raw.len());
        if ty==40 {
            copy_packed_rows(&mut payload,elements,cols,row,count,&raw)?;
        } else {
            small_floats(&raw,ty)?; // finite guard; original BF16/F32 bytes retained.
            let start=row*row_bytes;
            payload[start..start+raw.len()].copy_from_slice(&raw);
        }
    }
    if ty==40 {payload[len-4..].copy_from_slice(&1.0f32.to_le_bytes());}
    let blob=leaf::leaf_blob::<T,R>(shape,Bytes::from_source(payload))?;
    // Same native reader validates the constructed header/length before put.
    let expected:Vec<usize>=shape.iter().rev().map(|&d|d as usize).collect();
    checked_view(blob.clone(),&expected)?;
    summary.tensor_leaves+=1; summary.typed_tensor_bytes+=blob.bytes.len() as u64;
    summary.largest_payload_bytes=summary.largest_payload_bytes.max(len);
    let handle=pile.put(blob).map_err(|e|anyhow::anyhow!("store {name}: {e}"))?;
    Ok(entity! {leaf::leaf::<T,R>():handle})
}

fn vector(w:&mut GgufWeights,pile:&mut Pile,name:&str,n:usize,summary:&mut ImportSummary)->Result<Fragment> {
    store_selected::<F32,1>(w,pile,name,&[n],0,[n as u64],0,1,summary)
}

fn projection(w:&mut GgufWeights,pile:&mut Pile,prefix:&str,kind:&str,ni:usize,no:usize,
    experts:bool,summary:&mut ImportSummary)->Result<Fragment> {
    let name=format!("{prefix}{kind}.weight");
    let count=if experts {super::super::math::E} else {1};
    let scale=vector(w,pile,&format!("{prefix}{kind}.scale"),count,summary)?;
    let scale=entity! {attrs::kind:"post-matmul-scale",attrs::weight*:scale};
    if experts {
        let mut children=Fragment::empty();
        for index in 0..count {
            let weight=store_selected::<NVFP4,2>(w,pile,&name,&[ni,no,count],40,
                [no as u64,ni as u64],index.checked_mul(no).context("expert row overflow")?,no,summary)?;
            children+=entity! {attrs::index:index as u64,attrs::weight*:weight};
        }
        let members=children+scale;
        Ok(entity! {attrs::kind:kind,attrs::member*:members})
    } else {
        let weight=store_selected::<NVFP4,2>(w,pile,&name,&[ni,no],40,[no as u64,ni as u64],0,no,summary)?;
        Ok(entity! {attrs::kind:kind,attrs::weight*:weight,attrs::member*:scale})
    }
}

fn metadata(pile:&mut Pile,entries:&[MetadataEntry],kind:&str)->Result<Fragment> {
    let mut facts=TribleSet::new();
    let node=crate::jsonfacts::save_json(&serde_json::to_value(entries)?,pile,&mut facts)
        .map_err(|e|anyhow::anyhow!("store metadata facts: {e}"))?;
    let mut fragment=entity! {attrs::kind:kind,crate::jsonfacts::attrs::document_root:node};
    *fragment.facts_mut()+=facts;
    Ok(fragment)
}

/// Caller supplies a new, explicitly selected scratch pile and already-admitted
/// source descriptor. Publication happens once, after all leaves/config succeed.
/// Partial orphan blobs on error stay in that scratch pile for diagnosis.
#[cfg(feature="tokenizer")]
pub fn full(w:&mut GgufWeights,pile:&mut Pile,tokenizer:&[MetadataEntry])->Result<(Fragment,ImportSummary)> {
    use super::super::{math::{H,E,F,SH},forward::{VOCAB,BLOCKS}};
    ensure!(!w.file.index.tensors.iter().any(|i|i.name=="output.input_scale"
        || (i.name.starts_with("blk.") && i.name.ends_with(".bias"))),
        "unsupported input scale or block bias");
    for block in 0..BLOCKS {BlockConfig::from_metadata(&w.file.index.metadata,block)?;}
    super::super::tokenizer::FlintTokenizer::from_metadata(tokenizer)?;
    let mut summary=ImportSummary::default();
    let mut members=metadata(pile,&w.file.index.metadata,"flint-config")?
        +metadata(pile,tokenizer,"flint-tokenizer-metadata")?;
    let embedding=store_selected::<BF16,2>(w,pile,"token_embd.weight",&[H,VOCAB],30,
        [VOCAB as u64,H as u64],0,VOCAB,&mut summary)?;
    members+=entity! {attrs::kind:"token_embd",attrs::weight*:embedding};
    let norm=vector(w,pile,"output_norm.weight",H,&mut summary)?;
    members+=entity! {attrs::kind:"output_norm",attrs::weight*:norm};
    members+=projection(w,pile,"","output",H,VOCAB,false,&mut summary)?;
    for block in 0..BLOCKS {
        w.select_block(block)?;
        let prefix=format!("blk.{block}.");
        let mut parameters=Fragment::empty();
        for (name,ni,no) in PROJECTIONS.into_iter().take(4) {
            parameters+=projection(w,pile,&prefix,name,ni,no,false,&mut summary)?;
        }
        for (name,n) in VECTORS {
            let weight=vector(w,pile,&format!("{prefix}{name}.weight"),n,&mut summary)?;
            parameters+=entity! {attrs::kind:name,attrs::weight*:weight};
        }
        match w.config.ffn {
            FfnKind::Dense {width}=>{
                for (name,ni,no) in [("ffn_up",H,width),("ffn_gate",H,width),("ffn_down",width,H)] {
                    parameters+=projection(w,pile,&prefix,name,ni,no,false,&mut summary)?;
                }
            },
            FfnKind::Moe=>{
                let router=store_selected::<BF16,2>(w,pile,&format!("{prefix}ffn_gate_inp.weight"),
                    &[H,E],30,[E as u64,H as u64],0,E,&mut summary)?;
                parameters+=entity! {attrs::kind:"ffn_gate_inp",attrs::weight*:router};
                for (name,ni,no) in [("ffn_up_shexp",H,SH),("ffn_gate_shexp",H,SH),("ffn_down_shexp",SH,H)] {
                    parameters+=projection(w,pile,&prefix,name,ni,no,false,&mut summary)?;
                }
                for (name,ni,no) in [("ffn_up_exps",H,F),("ffn_gate_exps",H,F),("ffn_down_exps",F,H)] {
                    parameters+=projection(w,pile,&prefix,name,ni,no,true,&mut summary)?;
                }
            }
        }
        members+=entity! {attrs::kind:"flint-block",attrs::index:block as u64,attrs::member*:parameters};
    }
    w.file.check_identity()?;
    let mut root=entity! {attrs::member*:members};
    let id=root.root().context("full Flint model root")?;
    root+=entity! {ExclusiveId::force_ref(&id) @ attrs::source:"Flint GGUF typed import",
        attrs::quantization:"native-nvfp4",metadata::name:"Flint native"};
    Ok((root,summary))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn ingress_chunks_fill_native_planes_without_interleaving()->Result<()> {
        let mut raw=Vec::new();
        for row in 0..4u8 {
            let mut block=[(row+1)*0x11;36];block[..4].fill(0x30+row);raw.extend(block);
        }
        let mut payload=vec![0;NVFP4::payload_len(256)];
        for (first,rows) in [(0,1),(1,2),(3,1)] {
            copy_packed_rows(&mut payload,256,64,first,rows,&raw[first*36..(first+rows)*36])?;
        }
        let len=payload.len();payload[len-4..].copy_from_slice(&1.0f32.to_le_bytes());
        assert_eq!(payload,typed::payload(&raw,256)?);
        assert!(copy_packed_rows(&mut payload,256,64,4,1,&raw[..36]).is_err());
        assert!(copy_packed_rows(&mut payload,256,64,usize::MAX,1,&raw[..36]).is_err());
        let largest=super::super::super::forward::VOCAB*super::super::super::math::H*2;
        assert_eq!(largest,614_989_824);
        assert_eq!(largest*2+TENSOR_HEADER_LEN,1_229_979_904);
        Ok(())
    }
}
