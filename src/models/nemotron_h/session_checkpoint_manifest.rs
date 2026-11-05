//! A selected checkpoint as ordinary immutable facts and typed payload handles.
//!
//! This is a host codec, not cache discovery, publication, producer authentication
//! or permission to restore. The caller supplies an already-held exact model-facts
//! image handle. Neither encoding nor lookup rebuilds that image. Decoding visits
//! alternatives for ONE supplied root; it never derives a root ID to find it or
//! imposes attribute cardinality on the surrounding facts.
use super::{attention, KvChunk, LayerKind, Limits, PrefixCheckpoint, PrefixExecution,
    PrefixLayer, PrefixProvenance, Source, TextConfig};
use crate::{format::attrs as model, jsonfacts, leaf, tokenizer};
use std::{ops::{ControlFlow, Range}, sync::Arc};
use triblespace::{core::{id_hex, metadata, blob::encodings::tensor::{Tensor,
    elements::{BF16, F32}}}, prelude::*};
use blobencodings::{SimpleArchive, UTF8String};
use inlineencodings::{GenId, Handle, RangeU128, U256BE};

type Result<T> = std::result::Result<T, String>;
type Visitor<'a> = dyn FnMut(Result<PrefixCheckpoint>) -> ControlFlow<()> + 'a;

/// Gapless little-endian u32 token IDs. Length/4 is the count; empty is valid at
/// the encoding level. No native-endian cast, magic bytes or tokenizer semantics.
/// Identity minted with installed `trible genid`, 2026-11-05.
pub struct TokenIdsLe;
impl BlobEncoding for TokenIdsLe {}
impl MetaDescribe for TokenIdsLe {
    fn describe() -> Fragment {
        let id = id_hex!("CCC45261219E28B4CC2367E4DCA064D6");
        entity! { ExclusiveId::force_ref(&id) @
            metadata::name: "nemotron-token-ids-le-u32",
            metadata::description: "Gapless little-endian u32 token IDs; byte length must be divisible by four. Ordered and duplicate-preserving, independent of host byte order.",
            metadata::tag: metadata::KIND_BLOB_ENCODING,
        }
    }
}
impl TokenIdsLe {
    pub fn encode(tokens: &[u32]) -> Result<Blob<Self>> {
        let bytes = tokens.len().checked_mul(4).ok_or("token byte length overflow")?;
        let mut payload = Vec::new();
        payload.try_reserve_exact(bytes).map_err(|e| e.to_string())?;
        for token in tokens { payload.extend_from_slice(&token.to_le_bytes()); }
        Ok(Blob::new(anybytes::Bytes::from_source(payload)))
    }
    pub fn decode(blob: Blob<Self>, max_tokens: usize) -> Result<Vec<u32>> {
        if blob.bytes.len() % 4 != 0 { return Err("partial LE-u32 token".into()); }
        let count = blob.bytes.len() / 4;
        if count > max_tokens { return Err("manifest tokens exceed caller capacity".into()); }
        let mut tokens = Vec::new();
        tokens.try_reserve_exact(count).map_err(|e| e.to_string())?;
        for bytes in blob.bytes.chunks_exact(4) {
            tokens.push(u32::from_le_bytes(bytes.try_into().expect("four-byte chunk")));
        }
        Ok(tokens)
    }
}

/// Existing anchors retain their exact encoding/identity; no dependency on Drive
/// or Inkling runtime features. New anchors and tags were minted 2026-11-05;
/// the implementation receipt retains the actual command output.
pub mod schema {
    use super::*;
    pub use crate::format::attrs::{index, member, model_collection, model_root};
    pub use crate::tokenizer::attrs::{tokenizer, pre_tokenizer};
    attributes! {
        "65F7AF325535F0684556371F9956D242" as pub model_facts: Handle<SimpleArchive>;
        "3EFBF4582C0E801401E20154953DE408" as pub token_ids: Handle<TokenIdsLe>;
        "8BEF356CB8AE76E47B56EDC17D7A379E" as pub token_range: RangeU128;
        /// Opaque model-layer entity referenced by a checkpoint layer wrapper.
        "C7216375D355DC674EB07A87D50F4EEE" as pub source_layer: GenId;
        // Reused from Drive native_model_schema, minted 2026-10-26.
        "A29B9639EBC898B539F826F553C8D504" as pub selected_config: GenId;
        "CD278BBCC33E79BD623326384A683A41" as pub tokenizer_base: GenId;
        // Reused from Mary Inkling telemetry, minted 2026-08-28.
        "5D90F16E9D15FB0593A569ECBA0F60D6" as pub execution_profile: Handle<UTF8String>;
        "42DB94B3C5C091BC292D5B8FFB56E759" as pub position: U256BE;
    }
    pub const CHECKPOINT: Id = id_hex!("3A4919FF93E6254E09C8EDEAE655C36D");
    pub const ATTENTION: Id = id_hex!("D11F8D843F5EEB56419B0E435C4FA442");
    pub const MAMBA: Id = id_hex!("5511321A0C1835B70BECA399F88CC377");
    pub const STATELESS: Id = id_hex!("0CD355552A96AABEA2E066179FD78923");
    pub const KEYS: Id = id_hex!("1749747BD0C98321A5FBBBC7266C5C93");
    pub const VALUES: Id = id_hex!("DC94130DEAE89454EC14A1606F35D99B");
    pub const CONVOLUTION: Id = id_hex!("2274F905C046944E2FBEF56FCB50CF7D");
    pub const RECURRENT: Id = id_hex!("ED7C4A4EBF43039F64742C91055FEF77");
    pub const LOGITS: Id = id_hex!("1B2B26720BD76190553C7DC9BAE4CB26");
}

fn selected_config<B: BlobStoreGet>(store: &B, image: Inline<Handle<SimpleArchive>>,
    config: Id) -> Result<TextConfig>
{
    // Read the supplied immutable image, never republish/re-hash a ModelSnapshot.
    // Uses the same selected JSON-node interpretation as session_source::load.
    let facts: TribleSet = store.get(image).map_err(|e| format!("model image: {e}"))?;
    let value = jsonfacts::load_json(&facts, store, config).map_err(|e| e.to_string())?;
    serde_json::from_value(value.get("text").cloned().ok_or("native config has no text section")?)
        .map_err(|e| e.to_string())
}

/// Encode one structurally validated checkpoint. Only token/profile blobs are
/// written; existing tensor handles and the supplied model-image handle are
/// retained. The returned Fragment is NOT published or made durable here.
/// Read/write stores may be a frozen reader and its corresponding writer.
pub fn encode<R: BlobStoreGet, W: BlobStorePut>(read: &R, write: &mut W,
    checkpoint: &PrefixCheckpoint, model_facts: Inline<Handle<SimpleArchive>>,
    limits: Limits) -> Result<Fragment>
{
    let p = &checkpoint.provenance;
    let config = selected_config(read, model_facts, p.source.config)?;
    if config != p.config { return Err("checkpoint config differs from supplied model image".into()); }
    checkpoint.validate(read, &config, &p.layers, limits, &checkpoint.tokens, p.execution)?;
    let tokens = write.put(TokenIdsLe::encode(&checkpoint.tokens)?)
        .map_err(|e| format!("token put: {e}"))?;
    let PrefixExecution::NativeB1Bf16KvF32Mamba { attention } = p.execution;
    let profile = write.put::<UTF8String, _>(format!("native-b1-bf16-kv-f32-mamba/{}", attention.as_str()))
        .map_err(|e| format!("profile put: {e}"))?;
    let mut layers = Fragment::empty();
    for (index, layer) in checkpoint.layers.iter().enumerate() {
        let source = p.layers[index];
        let fragment = match layer {
            PrefixLayer::Attention(chunks) => {
                let mut children = Fragment::empty();
                for chunk in chunks {
                    let end = chunk.start.checked_add(chunk.rows).ok_or("KV range overflow")?;
                    let keys = entity! { metadata::tag: schema::KEYS,
                        leaf::leaf::<BF16, 3>(): chunk.keys };
                    let values = entity! { metadata::tag: schema::VALUES,
                        leaf::leaf::<BF16, 3>(): chunk.values };
                    children += entity! { schema::token_range: (chunk.start as u128, end as u128),
                        model::member*: keys + values };
                }
                entity! { metadata::tag: schema::ATTENTION, model::index: index as u64,
                    schema::source_layer: source, model::member*: children }
            }
            PrefixLayer::Mamba { position, conv, recurrent } => {
                let conv = entity! { metadata::tag: schema::CONVOLUTION,
                    leaf::leaf::<BF16, 3>(): *conv };
                let recurrent = entity! { metadata::tag: schema::RECURRENT,
                    leaf::leaf::<F32, 4>(): *recurrent };
                entity! { metadata::tag: schema::MAMBA, model::index: index as u64,
                    schema::source_layer: source, schema::position: *position as u64,
                    model::member*: conv + recurrent }
            }
            PrefixLayer::Stateless => entity! { metadata::tag: schema::STATELESS,
                model::index: index as u64, schema::source_layer: source },
        };
        layers += fragment;
    }
    let logits = entity! { metadata::tag: schema::LOGITS,
        leaf::leaf::<F32, 1>(): checkpoint.last_logits };
    Ok(entity! { metadata::tag: schema::CHECKPOINT,
        schema::model_facts: model_facts, schema::token_ids: tokens,
        schema::execution_profile: profile, schema::position: checkpoint.position as u64,
        model::model_collection: p.source.collection, model::model_root: p.source.root,
        schema::selected_config: p.source.config, tokenizer::attrs::tokenizer: p.source.tokenizer_binding,
        schema::tokenizer_base: p.source.tokenizer_base, tokenizer::attrs::pre_tokenizer: p.source.pre_tokenizer,
        model::member*: layers + logits,
    })
}

/// Visit complete typed interpretations of one explicitly selected root. Each
/// selected-value error is delivered to the visitor, not used to reject unrelated
/// facts or other alternatives. Unsupported typed rows/profile strings do not
/// match. `Break(())` stops immediately; enumeration is not a selection policy.
///
/// Only operation-local traversal state and one checkpoint are assembled. No
/// collection-wide catalogue, global uniqueness check, or derived-ID lookup.
/// Existing validate() checks geometry/payloads before a candidate is offered as
/// Ok. It does not authenticate the image, producer or arithmetic compatibility.
/// The required execution interpretation is explicit. A different profile is
/// refused after reading profile text, before token, model-image or tensor reads.
pub fn visit_selected<B: BlobStoreGet>(store: &B, facts: &TribleSet, root: Id,
    limits: Limits, execution: PrefixExecution,
    mut visitor: impl FnMut(Inline<Handle<SimpleArchive>>, Result<PrefixCheckpoint>)
        -> ControlFlow<()>) -> ControlFlow<()>
{
    for (image, tokens, profile, position, collection, model_root, config, binding, base, pre, logits) in find!(
        (image: Inline<Handle<SimpleArchive>>, tokens: Inline<Handle<TokenIdsLe>>,
         profile: Inline<Handle<UTF8String>>, position: u64, collection: Inline<Handle<SimpleArchive>>,
         model_root: Id, config: Id, binding: Id, base: Id, pre: Id,
         logits: Inline<Handle<Tensor<F32, 1>>>),
        pattern!(facts, [
            { (root) @ metadata::tag: schema::CHECKPOINT, schema::model_facts: ?image,
                schema::token_ids: ?tokens, schema::execution_profile: ?profile, schema::position: ?position,
                model::model_collection: ?collection, model::model_root: ?model_root,
                schema::selected_config: ?config, tokenizer::attrs::tokenizer: ?binding,
                schema::tokenizer_base: ?base, tokenizer::attrs::pre_tokenizer: ?pre,
                model::member: _?logits },
            { _?logits @ metadata::tag: schema::LOGITS, leaf::leaf::<F32, 1>(): ?logits }
        ]))
    {
        let selected = (|| -> Result<Option<PrefixCheckpoint>> {
            let profile: anybytes::View<str> = store.get(profile).map_err(|e| format!("profile get: {e}"))?;
            let Some(name) = profile.strip_prefix("native-b1-bf16-kv-f32-mamba/") else { return Ok(None) };
            let Ok(attention) = name.parse::<attention::Profile>() else { return Ok(None) };
            if (PrefixExecution::NativeB1Bf16KvF32Mamba { attention }) != execution {
                return Err("checkpoint execution profile differs".into());
            }
            let position = usize::try_from(position).map_err(|_| "manifest position does not fit host")?;
            if position > limits.capacity { return Err("manifest position exceeds caller capacity".into()); }
            let blob: Blob<TokenIdsLe> = store.get(tokens).map_err(|e| format!("token get: {e}"))?;
            let tokens = TokenIdsLe::decode(blob, limits.capacity)?;
            if tokens.len() != position { return Err("manifest token position differs".into()); }
            let config_value = selected_config(store, image, config)?;
            super::validate_config(&config_value, limits)?;
            super::resident_ownership(&config_value, limits)?;
            Ok(Some(PrefixCheckpoint { provenance: PrefixProvenance {
                source: Source { collection, root: model_root, config, tokenizer_binding: binding,
                    tokenizer_base: base, pre_tokenizer: pre }, config: config_value, layers: Vec::new(),
                execution: PrefixExecution::NativeB1Bf16KvF32Mamba { attention },
            }, position, tokens: Arc::from(tokens), last_logits: logits, layers: Vec::new() }))
        })();
        match selected {
            Ok(Some(mut checkpoint)) => visit_layers(store, facts, root, limits, &mut checkpoint,
                &mut |result| visitor(image, result))?,
            Ok(None) => {},
            Err(error) => visitor(image, Err(error))?,
        }
    }
    ControlFlow::Continue(())
}

fn visit_layers<B: BlobStoreGet>(store: &B, facts: &TribleSet, root: Id, limits: Limits,
    checkpoint: &mut PrefixCheckpoint, visitor: &mut Visitor<'_>) -> ControlFlow<()>
{
    let index = checkpoint.layers.len();
    if index == checkpoint.provenance.config.layers_block_type.len() {
        return visitor(checkpoint.validate(store, &checkpoint.provenance.config,
            &checkpoint.provenance.layers, limits, &checkpoint.tokens, checkpoint.provenance.execution)
            .map(|()| checkpoint.clone()));
    }
    let kind = checkpoint.provenance.config.layers_block_type[index];
    let tag = match kind { LayerKind::Attention => schema::ATTENTION,
        LayerKind::Mamba => schema::MAMBA, LayerKind::Moe => schema::STATELESS };
    let mut found = false;
    for (layer, source) in find!((layer: Id, source: Id), pattern!(facts, [
        { (root) @ model::member: ?layer },
        { ?layer @ metadata::tag: tag, model::index: (index as u64), schema::source_layer: ?source }
    ])) {
        found = true;
        checkpoint.provenance.layers.push(source);
        let position = checkpoint.position;
        let mut next = |value: Result<PrefixLayer>| {
            match value {
                Err(error) => visitor(Err(error)),
                Ok(value) => {
                    checkpoint.layers.push(value);
                    let flow = visit_layers(store, facts, root, limits, checkpoint, visitor);
                    checkpoint.layers.pop();
                    flow
                }
            }
        };
        match kind {
            LayerKind::Moe => next(Ok(PrefixLayer::Stateless))?,
            LayerKind::Mamba => {
                let mut matched = false;
                for (position, conv, recurrent) in find!((position: u64,
                    conv: Inline<Handle<Tensor<BF16, 3>>>, recurrent: Inline<Handle<Tensor<F32, 4>>>),
                    pattern!(facts, [
                        { (layer) @ schema::position: ?position, model::member: _?conv, model::member: _?recurrent },
                        { _?conv @ metadata::tag: schema::CONVOLUTION, leaf::leaf::<BF16, 3>(): ?conv },
                        { _?recurrent @ metadata::tag: schema::RECURRENT, leaf::leaf::<F32, 4>(): ?recurrent }
                    ])) {
                    matched = true;
                    next(usize::try_from(position).map(|position| PrefixLayer::Mamba { position, conv, recurrent })
                        .map_err(|_| "Mamba position does not fit host".into()))?;
                }
                if !matched { next(Err("missing typed Mamba boundary".into()))?; }
            }
            LayerKind::Attention => {
                visit_attention(facts, layer, position, &mut next)?;
            }
        }
        checkpoint.provenance.layers.pop();
    }
    if !found { visitor(Err(format!("missing typed checkpoint layer {index}")))?; }
    ControlFlow::Continue(())
}

fn visit_attention(facts: &TribleSet, layer: Id, position: usize,
    visitor: &mut dyn FnMut(Result<PrefixLayer>) -> ControlFlow<()>) -> ControlFlow<()>
{
    // Ephemeral alternatives from one selected layer, not retained cache state.
    let mut chunks = Vec::new();
    for (range, keys, values) in find!((range: Range<u128>,
        keys: Inline<Handle<Tensor<BF16, 3>>>, values: Inline<Handle<Tensor<BF16, 3>>>),
        pattern!(facts, [
            { (layer) @ model::member: _?chunk },
            { _?chunk @ schema::token_range: ?range, model::member: _?keys, model::member: _?values },
            { _?keys @ metadata::tag: schema::KEYS, leaf::leaf::<BF16, 3>(): ?keys },
            { _?values @ metadata::tag: schema::VALUES, leaf::leaf::<BF16, 3>(): ?values }
        ])) {
        if range.start >= range.end || range.end > position as u128 {
            visitor(Err("checkpoint KV empty/reversed/outside range".into()))?;
            continue;
        }
        chunks.push(KvChunk { start: range.start as usize, rows: (range.end - range.start) as usize,
            keys, values });
    }
    // Iterative path traversal: even chunk_rows=1 does not consume call stack.
    // Increasing absolute ends make this finite. Each complete path is validated
    // against the actual root position by PrefixCheckpoint::validate.
    let mut path = Vec::<KvChunk>::new();
    let mut frames = vec![(0usize, 0usize)]; // (next absolute start, next row to try)
    let mut offered = false;
    while let Some((start, cursor)) = frames.last_mut() {
        let choice = (*cursor..chunks.len()).find(|&i| chunks[i].start == *start);
        if let Some(i) = choice {
            *cursor = i + 1;
            let chunk = chunks[i].clone();
            let end = chunk.start + chunk.rows;
            path.push(chunk);
            if end == position {
                offered = true;
                visitor(Ok(PrefixLayer::Attention(path.clone())))?;
                path.pop();
            } else {
                frames.push((end, 0));
            }
        } else {
            frames.pop();
            if !frames.is_empty() { path.pop(); }
        }
    }
    if !offered { visitor(Err("missing complete typed KV path".into()))?; }
    ControlFlow::Continue(())
}

#[cfg(test)]
#[path = "session_checkpoint_manifest_tests.rs"]
mod tests;
