//! mary — a checkpoint's JSON sidecars as FACTS.
//!
//! The companion to [`crate::tokenizer`]. That module turns a `tokenizer.json`
//! into a graph so the tokenizer travels with the pile; this turns the rest of
//! the sidecars — `config.json`, `hf_quant_config.json`, `processor_config.json`,
//! the chat template — into one too, for the same reason and with the same
//! consequence: a pile stops needing the checkpoint directory beside it.
//!
//! # Why generic JSON rather than typed fields
//!
//! The obvious alternative is one minted attribute per config field:
//! `hidden_size`, `num_hidden_layers`, `sliding_window_size`, and so on. Fifty
//! ids for `config.json` alone, another set for `hf_quant_config.json`, and the
//! fields the Rust structs do not model — `architectures`, `eos_token_id`,
//! `model_type` — silently dropped, which is the failure this is supposed to
//! prevent. Worse, every field added upstream becomes a schema migration.
//!
//! Nine attributes model ALL of them exactly, and the result is still facts
//! rather than an opaque blob: "what does this checkpoint say `text_config.
//! hidden_size` is" is a path through `member` edges to a `json_int`, not a
//! substring search in a stored file. The unit of storage is the scalar,
//! because the scalar is the unit of the question.
//!
//! # Node identity carries the node's POSITION
//!
//! `entity! { _ @ … }` derives an id from the attributes, so two structurally
//! identical nodes collapse into one — which is right for a content-addressed
//! store and wrong if the position is attached afterwards. `[1, 1]` would
//! become a single entity carrying both `index: 0` and `index: 1`, and no
//! reader could untangle it.
//!
//! So the key or index is part of the entity at construction, never bolted on
//! after. Then `[1, 1]` is two nodes because their indices differ, `{"a": 1,
//! "b": 1}` is two nodes because their keys differ, and genuine structural
//! sharing (the same subtree under the same key in two documents) still
//! collapses, which is the property worth having.
//!
//! # The round trip is exact, and that is checkable
//!
//! `serde_json`'s object map is sorted, so re-serialising a `Value` is
//! canonical: `to_string(parse(original)) == to_string(load(save(parse(
//! original))))` is a byte comparison, and `inkling_meta_gate` makes it.
//! Numbers keep their integer/float distinction (`4096` does not come back as
//! `4096.0`) because the two are stored under different attributes rather than
//! both as `f64`.

use serde_json::Value;
use triblespace::core::metadata;
use triblespace::prelude::*;

type Err = Box<dyn std::error::Error>;

/// Node-type discriminants, via the canonical `metadata::tag`. Scalars need no
/// tag — they carry a value attribute, which is the discriminant — except
/// `null`, whose whole content is that it has none.
pub mod ty {
    use triblespace::macros::id_hex;
    use triblespace::prelude::Id;
    /// A JSON object. Minted 2026-08-13.
    pub const JSON_OBJECT: Id = id_hex!("0956A91F3198144BC1E1920467F9CE1A");
    /// A JSON array. Minted 2026-08-13.
    pub const JSON_ARRAY: Id = id_hex!("3DDDE71CF6B902D3CE15ABF3B3FE9A06");
    /// JSON `null`. Minted 2026-08-13.
    pub const JSON_NULL: Id = id_hex!("CBD6C0C8FD1DA5A4851429ABAE3CFA9F");
    /// A named document: a file name and the JSON root it parsed to. Minted
    /// 2026-08-13.
    pub const JSON_DOCUMENT: Id = id_hex!("686F919B71279386F31EB34A389E3241");
}

pub mod attrs {
    use triblespace::prelude::inlineencodings::{Boolean, F64, GenId, Handle, I256BE, U256BE};
    use triblespace::prelude::*;

    attributes! {
        // ── reused (same hex AND same encoding — do NOT re-mint) ──
        /// Homogeneous ordered membership: a container node → its children.
        /// From `format::attrs::member`, as `tokenizer::attrs::member`.
        "B4B6EC08A0CD70DE63A690168EE78F0F" as member: GenId;
        /// Position among siblings — an ARRAY element's index. From
        /// `format::attrs::index`.
        "33CE12B1B940B13E48D8E5B0ADFD2421" as index: U256BE;

        // ── minted 2026-08-13 ──
        /// An OBJECT member's key. A `UTF8String` handle rather than a
        /// `ShortString`: `logits_mup_width_multiplier` is 27 bytes and
        /// `num_nextn_predict_layers` 24, which fit, but the encoder answers a
        /// too-long value with `unwrap()` rather than an error, and a config
        /// key longer than 32 bytes is not a hypothetical.
        "051B238236962081B98F5D44BD9FA999" as json_key: Handle<blobencodings::UTF8String>;
        /// A JSON string's content. Content-addressed, so a value repeated
        /// across documents is stored once.
        "65D08DFB670E7E3729BE6BC2C0073CAD" as json_string: Handle<blobencodings::UTF8String>;
        /// A JSON number that is an INTEGER. Separate from `json_float` so the
        /// round trip is exact: stored as `f64`, `4096` comes back as `4096.0`
        /// and every re-serialised config differs from its source.
        "8657CA9A8F2864E21FA1793AF47AACA2" as json_int: I256BE;
        /// A JSON number that is NOT an integer (`1e-06`, `0.1`, `16.0`).
        "3B3EC066DE93041ACB2A26709CD11017" as json_float: F64;
        /// A JSON boolean.
        "AB52DBDC419DE0ECC7D56632D7B29B61" as json_bool: Boolean;
        /// A document node → its JSON root.
        "C3BE4C8C259D1A1C1BB2BF8C19DF41F3" as document_root: GenId;
    }
}

/// Where a node sits in its parent, folded into the node's identity.
#[derive(Clone, Copy)]
enum Pos<'a> {
    Root,
    Key(&'a str),
    Index(usize),
}

/// Ingest a `serde_json::Value` as facts, returning the root node's id.
pub fn save_json(
    v: &Value,
    blobs: &mut impl BlobStorePut,
    facts: &mut TribleSet,
) -> Result<Id, Err> {
    save_node(v, Pos::Root, blobs, facts)
}

fn save_node(
    v: &Value,
    pos: Pos,
    blobs: &mut impl BlobStorePut,
    facts: &mut TribleSet,
) -> Result<Id, Err> {
    let mut tags: Vec<Id> = Vec::new();
    let mut s_h = None;
    let mut i_v: Option<i64> = None;
    let mut f_v: Option<f64> = None;
    let mut b_v: Option<bool> = None;
    let mut members: Vec<Id> = Vec::new();

    match v {
        Value::Null => tags.push(ty::JSON_NULL),
        Value::Bool(b) => b_v = Some(*b),
        Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                i_v = Some(i);
            } else if let Some(f) = n.as_f64() {
                // Reached for non-integers AND for u64 above `i64::MAX`. The
                // second would lose precision silently, so it is refused rather
                // than rounded — a config with a 2^63 constant in it is a thing
                // to notice, not to approximate.
                if n.as_u64().is_some() {
                    return Err(format!("integer {n} does not fit in i64").into());
                }
                f_v = Some(f);
            } else {
                return Err(format!("number {n} is neither i64 nor f64").into());
            }
        }
        Value::String(s) => {
            s_h = Some(blobs.put::<blobencodings::UTF8String, _>(s.clone())?);
        }
        Value::Array(a) => {
            tags.push(ty::JSON_ARRAY);
            for (i, c) in a.iter().enumerate() {
                members.push(save_node(c, Pos::Index(i), blobs, facts)?);
            }
        }
        Value::Object(o) => {
            tags.push(ty::JSON_OBJECT);
            for (k, c) in o {
                members.push(save_node(c, Pos::Key(k), blobs, facts)?);
            }
        }
    }

    let key_h = match pos {
        Pos::Key(k) => Some(blobs.put::<blobencodings::UTF8String, _>(k.to_string())?),
        _ => None,
    };
    let idx: Option<u64> = match pos {
        Pos::Index(i) => Some(i as u64),
        _ => None,
    };

    let e = entity! { _ @
        metadata::tag*: tags.iter(),
        attrs::json_key?: key_h,
        attrs::index?: idx,
        attrs::json_string?: s_h,
        attrs::json_int?: i_v,
        attrs::json_float?: f_v,
        attrs::json_bool?: b_v,
        attrs::member*: members.iter(),
    };
    let id = e.root().expect("json node root");
    *facts += e.into_facts();
    Ok(id)
}

/// Ingest one NAMED document — a file name and the JSON it parsed to.
///
/// The name is what a reader asks by, so it is a fact on its own node rather
/// than something the caller has to remember about a bare root id.
pub fn save_document(
    name: &str,
    v: &Value,
    blobs: &mut impl BlobStorePut,
    facts: &mut TribleSet,
) -> Result<Id, Err> {
    let root = save_json(v, blobs, facts)?;
    let name_h = blobs.put::<blobencodings::UTF8String, _>(name.to_string())?;
    let doc = entity! { _ @
        metadata::tag: ty::JSON_DOCUMENT,
        metadata::name: name_h,
        attrs::document_root: root,
    };
    let id = doc.root().expect("document root");
    *facts += doc.into_facts();
    Ok(id)
}

/// Every document in a fact set, as `(name, root node)`.
pub fn documents(tribles: &TribleSet, blobs: &impl BlobStoreGet) -> Vec<(String, Id)> {
    let mut out = Vec::new();
    for (n, r) in find!(
        (n: Inline<inlineencodings::Handle<blobencodings::UTF8String>>, r: Id),
        pattern!(tribles, [
            { _?d @ metadata::tag: (ty::JSON_DOCUMENT),
                    metadata::name: ?n,
                    attrs::document_root: ?r },
        ])
    ) {
        if let Some(name) = read_string(blobs, n) {
            out.push((name, r));
        }
    }
    out.sort();
    out
}

/// The JSON of the document with this file name.
pub fn load_document(
    tribles: &TribleSet,
    blobs: &impl BlobStoreGet,
    name: &str,
) -> Result<Value, Err> {
    let root = documents(tribles, blobs)
        .into_iter()
        .find(|(n, _)| n == name)
        .map(|(_, r)| r)
        .ok_or_else(|| {
            let have: Vec<String> = documents(tribles, blobs)
                .into_iter()
                .map(|(n, _)| n)
                .collect();
            format!("no document named {name:?} in this pile; it has {have:?}")
        })?;
    load_json(tribles, blobs, root)
}

/// Read a `UTF8String` handle back to an owned `String`, as
/// `tokenizer::read_piece` does — a view over the store's bytes, materialised
/// once here because a `serde_json::Value` owns its strings.
fn read_string(
    blobs: &impl BlobStoreGet,
    h: Inline<inlineencodings::Handle<blobencodings::UTF8String>>,
) -> Option<String> {
    let v: anybytes::View<str> = blobs.get(h).ok()?;
    Some(v.to_string())
}

/// A node's optional `UTF8String`-handle field.
///
/// A macro rather than a function because `pattern!` takes an attribute PATH,
/// not an `Attribute` value — the same reason `tokenizer.rs` spells its field
/// readers as macros.
macro_rules! long_field {
    ($tribles:expr_2021, $blobs:expr_2021, $node:expr_2021, $attr:path) => {
        find!(
            (h: Inline<inlineencodings::Handle<blobencodings::UTF8String>>),
            pattern!($tribles, [{ ($node) @ $attr: ?h }])
        )
        .next()
        .and_then(|(h,)| read_string($blobs, h))
    };
}

/// A canonical serialization of `v`: compact, with every object's keys in
/// byte order, so equal values have equal text whatever the map's own order
/// and whatever features `serde_json` was built with (`preserve_order` would
/// keep insertion order, and [`load_json`] inserts in query order). Scalars
/// are `serde_json`'s own spelling; integers stay integers, as they are
/// stored. Use it to hash or compare a document read back from facts.
pub fn canonical_string(v: &Value) -> String {
    fn write(v: &Value, out: &mut String) {
        match v {
            Value::Array(items) => {
                out.push('[');
                for (i, item) in items.iter().enumerate() {
                    if i > 0 {
                        out.push(',');
                    }
                    write(item, out);
                }
                out.push(']');
            }
            Value::Object(map) => {
                let mut keys: Vec<&String> = map.keys().collect();
                keys.sort();
                out.push('{');
                for (i, key) in keys.into_iter().enumerate() {
                    if i > 0 {
                        out.push(',');
                    }
                    out.push_str(&Value::String(key.clone()).to_string());
                    out.push(':');
                    write(&map[key], out);
                }
                out.push('}');
            }
            scalar => out.push_str(&scalar.to_string()),
        }
    }
    let mut out = String::new();
    write(v, &mut out);
    out
}

/// Rebuild a `serde_json::Value` from a node.
pub fn load_json(tribles: &TribleSet, blobs: &impl BlobStoreGet, node: Id) -> Result<Value, Err> {
    let tags = crate::tokenizer::node_tags(tribles, node);
    if tags.contains(&ty::JSON_NULL) {
        return Ok(Value::Null);
    }
    if tags.contains(&ty::JSON_OBJECT) {
        let mut map = serde_json::Map::new();
        for kid in members(tribles, node) {
            let key = long_field!(tribles, blobs, kid, attrs::json_key)
                .ok_or("an object member carries no json_key")?;
            map.insert(key, load_json(tribles, blobs, kid)?);
        }
        return Ok(Value::Object(map));
    }
    if tags.contains(&ty::JSON_ARRAY) {
        let mut kids: Vec<(u64, Id)> = Vec::new();
        for kid in members(tribles, node) {
            let i = find!((i: u64), pattern!(tribles, [{ (kid) @ attrs::index: ?i }]))
                .next()
                .map(|(i,)| i)
                .ok_or("an array member carries no index")?;
            kids.push((i, kid));
        }
        kids.sort_by_key(|(i, _)| *i);
        // Contiguity is checked, not assumed: a dropped element would otherwise
        // come back as a shorter array with no complaint, which is the shape of
        // corruption this encoding could actually suffer.
        for (want, (got, _)) in kids.iter().enumerate() {
            if *got != want as u64 {
                return Err(
                    format!("array indices are not 0..n: saw {got} at position {want}").into(),
                );
            }
        }
        let mut out = Vec::with_capacity(kids.len());
        for (_, kid) in kids {
            out.push(load_json(tribles, blobs, kid)?);
        }
        return Ok(Value::Array(out));
    }
    if let Some(s) = long_field!(tribles, blobs, node, attrs::json_string) {
        return Ok(Value::String(s));
    }
    if let Some((i,)) =
        find!((i: i64), pattern!(tribles, [{ (node) @ attrs::json_int: ?i }])).next()
    {
        return Ok(Value::Number(i.into()));
    }
    if let Some((f,)) =
        find!((f: f64), pattern!(tribles, [{ (node) @ attrs::json_float: ?f }])).next()
    {
        return Ok(Value::Number(
            serde_json::Number::from_f64(f).ok_or("stored float is not finite")?,
        ));
    }
    if let Some((b,)) =
        find!((b: bool), pattern!(tribles, [{ (node) @ attrs::json_bool: ?b }])).next()
    {
        return Ok(Value::Bool(b));
    }
    Err(format!("node {node:?} carries no JSON value").into())
}

fn members(tribles: &TribleSet, node: Id) -> Vec<Id> {
    find!((m: Id), pattern!(tribles, [{ (node) @ attrs::member: ?m }]))
        .map(|(m,)| m)
        .collect()
}

#[cfg(test)]
mod tests {
    /// Writes the canonical text of `WEMM_CONFIG_JSON` read back from facts
    /// to `WEMM_CANONICAL_OUT`, for pinning its digest.
    #[test]
    #[ignore = "set WEMM_CONFIG_JSON and WEMM_CANONICAL_OUT"]
    fn canonical_text_of_a_config_read_back_from_facts() {
        let json = std::fs::read(std::env::var("WEMM_CONFIG_JSON").unwrap()).unwrap();
        let parsed: Value = serde_json::from_slice(&json).unwrap();
        let mut blobs = MemoryBlobStore::new();
        let mut facts = TribleSet::new();
        let root = save_json(&parsed, &mut blobs, &mut facts).unwrap();
        let reader = blobs.snapshot().unwrap();
        let loaded = load_json(&facts, &reader, root).unwrap();
        assert_eq!(loaded, parsed);
        assert_eq!(canonical_string(&loaded), canonical_string(&parsed));
        std::fs::write(
            std::env::var("WEMM_CANONICAL_OUT").unwrap(),
            canonical_string(&loaded),
        )
        .unwrap();
    }

    #[test]
    fn canonical_text_sorts_keys_and_keeps_integers() {
        let v: Value =
            serde_json::from_str(r#"{"b": [1, 2.5, null], "a": {"z": true, "y": "q\"x"}}"#)
                .unwrap();
        assert_eq!(
            canonical_string(&v),
            r#"{"a":{"y":"q\"x","z":true},"b":[1,2.5,null]}"#
        );
    }

    use super::*;

    fn roundtrip(src: &str) -> String {
        let v: Value = serde_json::from_str(src).expect("parse");
        let mut blobs = MemoryBlobStore::new();
        let mut facts = TribleSet::new();
        let root = save_json(&v, &mut blobs, &mut facts).expect("save");
        let reader = blobs.snapshot().expect("snapshot");
        let back = load_json(&facts, &reader, root).expect("load");
        serde_json::to_string(&back).expect("serialise")
    }

    /// Canonical form in, canonical form out. `serde_json`'s object map is
    /// sorted, so this is an exact comparison and not an approximate one.
    #[test]
    fn every_json_shape_survives() {
        for src in [
            r#"{"a":1,"b":"two","c":true,"d":null,"e":[1,2,3],"f":{"g":{}}}"#,
            r#"[]"#,
            r#"{}"#,
            r#"null"#,
            r#"[[],[[]],{}]"#,
            r#"{"neg":-42,"zero":0,"big":9007199254740991}"#,
        ] {
            let canonical =
                serde_json::to_string(&serde_json::from_str::<Value>(src).unwrap()).unwrap();
            assert_eq!(roundtrip(src), canonical, "round trip of {src}");
        }
    }

    /// THE reason integers and floats are different attributes. Stored as one
    /// `f64`, `4096` comes back `4096.0` and every re-serialised config differs
    /// from the file it came from.
    #[test]
    fn an_integer_does_not_come_back_as_a_float() {
        assert_eq!(
            roundtrip(r#"{"hidden_size":4096}"#),
            r#"{"hidden_size":4096}"#
        );
        assert_eq!(
            roundtrip(r#"{"eps":1e-06}"#),
            r#"{"eps":1e-6}"#.replace("1e-6", "1e-6")
        );
        // and a float that happens to be integral stays a float
        assert_eq!(roundtrip(r#"{"mup":16.0}"#), r#"{"mup":16.0}"#);
    }

    /// The identity trap this encoding is built around. Two structurally equal
    /// siblings must stay two nodes, or the array collapses.
    #[test]
    fn repeated_values_do_not_collapse_into_one_node() {
        assert_eq!(roundtrip("[1,1,1]"), "[1,1,1]");
        assert_eq!(roundtrip(r#"{"a":1,"b":1}"#), r#"{"a":1,"b":1}"#);
        assert_eq!(roundtrip(r#"[{"x":1},{"x":1}]"#), r#"[{"x":1},{"x":1}]"#);
        // Inkling's real one: 35 consecutive small integers, many repeated
        // across the two local_layer_ids lists.
        let ids: Vec<i64> = vec![0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 12, 13, 14];
        let src = serde_json::to_string(&serde_json::json!({ "a": ids, "b": ids })).unwrap();
        assert_eq!(roundtrip(&src), src);
    }

    /// A document is found by NAME, and asking for one that is not there says
    /// what is.
    #[test]
    fn documents_are_found_by_name() {
        let mut blobs = MemoryBlobStore::new();
        let mut facts = TribleSet::new();
        save_document(
            "config.json",
            &serde_json::json!({"hidden_size": 4096}),
            &mut blobs,
            &mut facts,
        )
        .expect("save");
        let reader = blobs.snapshot().expect("snapshot");
        assert_eq!(
            documents(&facts, &reader)
                .into_iter()
                .map(|(n, _)| n)
                .collect::<Vec<_>>(),
            vec!["config.json".to_string()]
        );
        let v = load_document(&facts, &reader, "config.json").expect("load");
        assert_eq!(v["hidden_size"], 4096);
        let err = load_document(&facts, &reader, "nope.json").expect_err("must refuse");
        assert!(format!("{err}").contains("config.json"), "{err}");
    }
}
