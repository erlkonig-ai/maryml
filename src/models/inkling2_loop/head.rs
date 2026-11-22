//! Inkling 2 Mini's output head on the CPU: the final block attention
//! residual, the final norm, the unembedding and the logit softcap, from the
//! block states and the last hidden state to the logits the release samples
//! from.
//!
//! Upstream (transformers 046bdb8c, the end of `Inkling2LoopTextModel.forward`,
//! then `Inkling2LoopForCausalLM.forward`), in this order:
//!
//! 0. `self.final_input`, a one-query `Inkling2LoopBlockAttentionResidual`
//!    over `model.llm.attn_res.output_query` (F32, renamed `final_input.query`
//!    on load): the completed block states with the last hidden state appended
//!    as one more (the partial block), all widened to f32 and put through an
//!    unweighted RMS norm (eps 1e-6, the class default, not the config's); the
//!    query scores each against the f32 query at scale `hidden^-0.5`, softmax
//!    in f32, and the output is the weighted sum of the *normed* states, f32;
//! 1. `self.norm`, an `Inkling2LoopRMSNorm` over `model.llm.norm.weight`: its
//!    input is that f32 vector, so `x * rsqrt(mean(x^2) + eps)` times the BF16
//!    gain stays f32, and `.to(inputs_embeds.dtype)` **rounds it to BF16**;
//! 2. `self.lm_head`, an `nn.Linear(hidden, unpadded_vocab_size)` without
//!    bias over `model.llm.unembed.weight`: BF16 by BF16, the output
//!    **rounded to BF16**, then `.float()`;
//! 3. `self.logit_softcap`: `tanh(x / cap) * cap` in f32, nothing rounded.
//!
//! So there are exactly two BF16 roundings, after the norm and after the
//! unembedding; the residual and the softcap are f32 end to end.
//!
//! The checkpoint stores the unembedding padded to `vocab_size` rows;
//! upstream's `Inkling2UnpadVocab` keeps only the first `unpadded_vocab_size`
//! on load. The head reads the padded tensor as stored and never computes
//! the pad rows, so it emits exactly upstream's `unpadded_vocab_size` logits.
//! The unembedding is its own tensor, not tied to `model.llm.embed.weight`
//! (upstream: "`embed` and `unembed` are separate tensors in the checkpoints,
//! never tied").
//!
//! Mary's own arithmetic where it computes the same thing:
//! [`crate::models::inkling::block::rms_norm`] for both norms (with a unit
//! gain for the unweighted one, `1.0 * y == y`). The one-query residual is
//! written here on slices for now; the device primitive is the sliding
//! layer's MLP-input residual, and the two are to meet when the branches
//! join. The unembedding stays the stored BF16 bytes (926 MB at the release's
//! shape) and is read row by row; only the gain and the query become f32.

use super::config::Inkling2LoopTextConfig;
use super::layout::Dtype;
use crate::models::inkling::block::rms_norm;

/// The checkpoint names the head reads.
pub const QUERY: &str = "model.llm.attn_res.output_query";
pub const NORM: &str = "model.llm.norm.weight";
pub const UNEMBED: &str = "model.llm.unembed.weight";

/// `Inkling2LoopUnweightedRMSNorm()`'s default eps, which the block residual
/// uses regardless of `rms_norm_eps`.
pub const RESIDUAL_EPS: f64 = 1.0e-6;

/// Round an f32 to the nearest BF16 (ties to even), kept as f32. This is
/// what torch's `.to(torch.bfloat16)` does to a finite value.
pub fn round_bf16(x: f32) -> f32 {
    half::bf16::from_f32(x).to_f32()
}

#[inline]
fn bf16_at(bytes: &[u8], i: usize) -> f32 {
    f32::from_bits((u16::from_le_bytes([bytes[2 * i], bytes[2 * i + 1]]) as u32) << 16)
}

/// One query attending over `n` states per token: `stacked` is
/// `[tokens, n, h]` row-major. Returns `[tokens, h]` in f32.
pub fn one_query_residual(query: &[f32], stacked: &[f32], tokens: usize, n: usize, h: usize) -> Vec<f32> {
    assert_eq!(query.len(), h);
    assert_eq!(stacked.len(), tokens * n * h);
    assert!(n > 0, "a block residual needs at least one state");
    let normed = rms_norm(stacked, &vec![1.0; h], RESIDUAL_EPS, tokens * n, h);
    let scale = (h as f32).powf(-0.5);
    let mut out = vec![0f32; tokens * h];
    let mut scores = vec![0f32; n];
    for t in 0..tokens {
        let rows = &normed[t * n * h..(t + 1) * n * h];
        for (j, s) in scores.iter_mut().enumerate() {
            let r = &rows[j * h..(j + 1) * h];
            *s = query.iter().zip(r).map(|(a, b)| a * b).sum::<f32>() * scale;
        }
        let max = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exp: Vec<f32> = scores.iter().map(|s| (s - max).exp()).collect();
        let total: f32 = exp.iter().sum();
        let o = &mut out[t * h..(t + 1) * h];
        for (j, e) in exp.iter().enumerate() {
            let w = e / total;
            for (oi, &ri) in o.iter_mut().zip(&rows[j * h..(j + 1) * h]) {
                *oi += w * ri;
            }
        }
    }
    out
}

/// The output head, bound to its three tensors. Borrows the unembedding bytes.
pub struct Head<'a> {
    hidden: usize,
    vocab: usize,
    eps: f64,
    cap: f32,
    query: Vec<f32>,
    gain: Vec<f32>,
    unembed: &'a [u8],
}

/// The head's taps for `tokens` rows: the residual's f32 output (the norm's
/// input) `[tokens, hidden]`, the BF16-rounded normed state, the BF16-rounded
/// raw logits `[tokens, vocab]`, and the softcapped f32 logits.
pub struct HeadOut {
    pub selected: Vec<f32>,
    pub normed: Vec<f32>,
    pub raw: Vec<f32>,
    pub logits: Vec<f32>,
}

/// A stored tensor as [`super::pile::MiniRoot::tensor`] answers it.
pub type Stored<'a> = (Dtype, Vec<usize>, &'a [u8]);

impl<'a> Head<'a> {
    /// Bind the head from its three stored tensors. Refuses any dtype or
    /// shape other than the release's.
    pub fn bind(
        cfg: &Inkling2LoopTextConfig,
        query: Stored<'_>,
        norm: Stored<'_>,
        unembed: Stored<'a>,
    ) -> Result<Self, String> {
        let h = cfg.hidden_size;
        let check = |name: &str, t: &Stored<'_>, dtype: Dtype, shape: &[usize]| {
            if t.0 != dtype || t.1 != shape {
                return Err(format!("{name}: stored {:?} {:?}, the head reads {:?} {:?}", t.0, t.1, dtype, shape));
            }
            let width = if dtype == Dtype::F32 { 4 } else { 2 };
            let want = shape.iter().product::<usize>() * width;
            if t.2.len() != want {
                return Err(format!("{name}: {} bytes, {:?} {:?} is {want}", t.2.len(), dtype, shape));
            }
            Ok(())
        };
        check(QUERY, &query, Dtype::F32, &[h])?;
        check(NORM, &norm, Dtype::Bf16, &[h])?;
        check(UNEMBED, &unembed, Dtype::Bf16, &[cfg.vocab_size, h])?;
        if cfg.unpadded_vocab_size > cfg.vocab_size {
            return Err(format!(
                "unpadded_vocab_size {} > vocab_size {}",
                cfg.unpadded_vocab_size, cfg.vocab_size
            ));
        }
        Ok(Head {
            hidden: h,
            vocab: cfg.unpadded_vocab_size,
            eps: cfg.rms_norm_eps,
            cap: cfg.final_logit_softcapping as f32,
            query: query.2.chunks_exact(4).map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect(),
            gain: (0..h).map(|i| bf16_at(norm.2, i)).collect(),
            unembed: unembed.2,
        })
    }

    /// Bind from a Mini model root in a pile.
    pub fn from_root(cfg: &Inkling2LoopTextConfig, root: &'a super::pile::MiniRoot) -> Result<Self, String> {
        Head::bind(cfg, root.tensor(QUERY)?, root.tensor(NORM)?, root.tensor(UNEMBED)?)
    }

    /// Logits emitted per token: `unpadded_vocab_size`.
    pub fn vocab(&self) -> usize {
        self.vocab
    }

    /// The whole head. `blocks` is `[tokens, n_blocks, hidden]` (the completed
    /// block states, upstream's `block_states`), `partial` is `[tokens,
    /// hidden]` (the last layer's output). In the release both are BF16
    /// values, so widening them is exact.
    pub fn forward(&self, blocks: &[f32], n_blocks: usize, partial: &[f32]) -> HeadOut {
        let h = self.hidden;
        assert!(partial.len() % h == 0, "partial is not [tokens, {h}]");
        let tokens = partial.len() / h;
        assert_eq!(blocks.len(), tokens * n_blocks * h, "blocks are not [{tokens}, {n_blocks}, {h}]");
        let n = n_blocks + 1;
        let mut stacked = Vec::with_capacity(tokens * n * h);
        for t in 0..tokens {
            stacked.extend_from_slice(&blocks[t * n_blocks * h..(t + 1) * n_blocks * h]);
            stacked.extend_from_slice(&partial[t * h..(t + 1) * h]);
        }
        let selected = one_query_residual(&self.query, &stacked, tokens, n, h);
        self.project(selected)
    }

    /// Norm, unembedding and softcap of the residual's f32 output.
    pub fn project(&self, selected: Vec<f32>) -> HeadOut {
        let h = self.hidden;
        assert!(selected.len() % h == 0, "input is not [tokens, {h}]");
        let tokens = selected.len() / h;
        // 1. weighted RMS norm in f32, then BF16.
        let normed: Vec<f32> = rms_norm(&selected, &self.gain, self.eps, tokens, h)
            .into_iter()
            .map(round_bf16)
            .collect();
        // 2. BF16 x BF16 with an f32 accumulator, output rounded to BF16.
        let mut raw = vec![0f32; tokens * self.vocab];
        let mut row = vec![0f32; h];
        for v in 0..self.vocab {
            let bytes = &self.unembed[v * h * 2..(v + 1) * h * 2];
            for (k, r) in row.iter_mut().enumerate() {
                *r = bf16_at(bytes, k);
            }
            for t in 0..tokens {
                let n = &normed[t * h..(t + 1) * h];
                let acc: f32 = n.iter().zip(&row).map(|(a, b)| a * b).sum();
                raw[t * self.vocab + v] = round_bf16(acc);
            }
        }
        // 3. softcap in f32.
        let cap = self.cap;
        let logits = raw.iter().map(|&l| (l / cap).tanh() * cap).collect();
        HeadOut { selected, normed, raw, logits }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Captured from upstream's own classes by `golden/capture_inkling2_head.py
    /// tiny` (`Inkling2LoopBlockAttentionResidual`, `Inkling2LoopRMSNorm`,
    /// `Inkling2UnpadVocab`, `nn.Linear`, `Inkling2LoopSoftcap`, in
    /// upstream's order, at BF16). Every value is stored as its exact bits.
    const FIXTURE: &str = include_str!("fixtures/head_tiny.json");

    struct Fixture {
        cfg: Inkling2LoopTextConfig,
        tokens: usize,
        n_blocks: usize,
        blocks: Vec<f32>,
        partial: Vec<f32>,
        query: Vec<u8>,
        norm: Vec<u8>,
        unembed: Vec<u8>,
        selected: Vec<f32>,
        normed: Vec<f32>,
        raw: Vec<f32>,
        logits: Vec<f32>,
    }

    fn ints(v: &serde_json::Value) -> Vec<u64> {
        v.as_array().unwrap().iter().map(|x| x.as_u64().unwrap()).collect()
    }
    fn bf16_bytes(v: &serde_json::Value) -> Vec<u8> {
        ints(v).into_iter().flat_map(|b| (b as u16).to_le_bytes()).collect()
    }
    fn bf16_vals(v: &serde_json::Value) -> Vec<f32> {
        ints(v).into_iter().map(|b| f32::from_bits((b as u32) << 16)).collect()
    }
    fn f32_bytes(v: &serde_json::Value) -> Vec<u8> {
        ints(v).into_iter().flat_map(|b| (b as u32).to_le_bytes()).collect()
    }
    fn f32_vals(v: &serde_json::Value) -> Vec<f32> {
        ints(v).into_iter().map(|b| f32::from_bits(b as u32)).collect()
    }

    fn fixture() -> Fixture {
        let j: serde_json::Value = serde_json::from_str(FIXTURE).unwrap();
        // The release's text config with the fixture's sizes; the head reads
        // only hidden_size, vocab_size, unpadded_vocab_size, rms_norm_eps and
        // final_logit_softcapping from it.
        let release: serde_json::Value =
            serde_json::from_str(include_str!("fixtures/config.json")).unwrap();
        let mut text = release["text_config"].clone();
        for k in ["hidden_size", "vocab_size", "unpadded_vocab_size", "final_logit_softcapping", "rms_norm_eps"] {
            text[k] = j["config"][k].clone();
        }
        let cfg: Inkling2LoopTextConfig = serde_json::from_value(text).unwrap();
        Fixture {
            tokens: j["tokens"].as_u64().unwrap() as usize,
            n_blocks: j["n_blocks"].as_u64().unwrap() as usize,
            blocks: bf16_vals(&j["block_states"]),
            partial: bf16_vals(&j["partial"]),
            query: f32_bytes(&j["output_query"]),
            norm: bf16_bytes(&j["norm_weight"]),
            unembed: bf16_bytes(&j["unembed_weight"]),
            selected: f32_vals(&j["selected"]),
            normed: bf16_vals(&j["normed"]),
            raw: bf16_vals(&j["raw_logits"]),
            logits: f32_vals(&j["logits"]),
            cfg,
        }
    }

    fn bind<'a>(f: &'a Fixture, unembed: &'a [u8]) -> Head<'a> {
        let (h, v) = (f.cfg.hidden_size, f.cfg.vocab_size);
        Head::bind(
            &f.cfg,
            (Dtype::F32, vec![h], &f.query),
            (Dtype::Bf16, vec![h], &f.norm),
            (Dtype::Bf16, vec![v, h], unembed),
        )
        .unwrap()
    }

    /// Distance in BF16 steps between two BF16 values of the same sign.
    fn bf16_steps(a: f32, b: f32) -> u32 {
        let (x, y) = (a.to_bits() >> 16, b.to_bits() >> 16);
        if (x ^ y) & 0x8000 != 0 {
            return u32::MAX;
        }
        x.abs_diff(y)
    }

    /// The tolerances, and why. The residual is f32 end to end but sums in a
    /// different order than torch (and `sqrt().recip()` against `rsqrt`), so
    /// its output agrees to a few f32 ulps: |d| <= 1e-5 (1 + |x|). That
    /// difference, and our dot product's order, can move a value that sits
    /// near a BF16 rounding boundary one BF16 step: so the BF16 taps within
    /// one step and at least 90% bit-equal. The softcapped logits within the
    /// image of one raw step (tanh' <= 1) plus f32 tanh error.
    fn compare(f: &Fixture, out: &HeadOut) -> Result<(usize, usize), String> {
        if out.selected.len() != f.selected.len() {
            return Err(format!("selected: {} values, reference has {}", out.selected.len(), f.selected.len()));
        }
        for (i, (&a, &b)) in out.selected.iter().zip(&f.selected).enumerate() {
            if (a - b).abs() > 1e-5 * (1.0 + b.abs()) {
                return Err(format!("selected[{i}]: ours {a}, reference {b}"));
            }
        }
        let mut equal = 0usize;
        let mut total = 0usize;
        for (name, ours, theirs) in [("normed", &out.normed, &f.normed), ("raw", &out.raw, &f.raw)] {
            if ours.len() != theirs.len() {
                return Err(format!("{name}: {} values, reference has {}", ours.len(), theirs.len()));
            }
            for (i, (&a, &b)) in ours.iter().zip(theirs.iter()).enumerate() {
                let s = if a == b { 0 } else { bf16_steps(a, b) };
                if s > 1 {
                    return Err(format!("{name}[{i}]: ours {a}, reference {b}, {s} BF16 steps"));
                }
                equal += (s == 0) as usize;
                total += 1;
            }
        }
        if equal * 10 < total * 9 {
            return Err(format!("only {equal} of {total} BF16 values bit-equal"));
        }
        if out.logits.len() != f.logits.len() {
            return Err(format!("logits: {} values, reference has {}", out.logits.len(), f.logits.len()));
        }
        for (i, (&a, &b)) in out.logits.iter().zip(f.logits.iter()).enumerate() {
            let step = f.raw[i].abs() / 128.0;
            if (a - b).abs() > step + 1e-5 * b.abs().max(1.0) {
                return Err(format!("logits[{i}]: ours {a}, reference {b}"));
            }
        }
        Ok((equal, total))
    }

    #[test]
    fn matches_upstream_on_the_tiny_fixture() {
        let f = fixture();
        let head = bind(&f, &f.unembed);
        assert_eq!(head.vocab(), f.cfg.unpadded_vocab_size);
        assert!(f.cfg.unpadded_vocab_size < f.cfg.vocab_size, "the fixture must exercise unpadding");
        assert!(f.tokens > 1 && f.n_blocks > 1);
        // The fixture is only useful if the softcap bends something.
        assert!(f.raw.iter().any(|r| r.abs() > f.cfg.final_logit_softcapping as f32));
        let out = head.forward(&f.blocks, f.n_blocks, &f.partial);
        let (equal, total) = compare(&f, &out).unwrap();
        let exact = out.logits.iter().zip(&f.logits).filter(|(a, b)| a == b).count();
        eprintln!("bf16 taps bit-equal {equal}/{total}, logits bit-equal {exact}/{}", f.logits.len());
    }

    #[test]
    fn a_broken_head_fails_the_comparison() {
        let f = fixture();
        let (h, nb) = (f.cfg.hidden_size, f.n_blocks);
        let head = bind(&f, &f.unembed);
        let good = head.forward(&f.blocks, nb, &f.partial);
        let fails = |what: &str, out: HeadOut| assert!(compare(&f, &out).is_err(), "{what} passed");

        // The partial block dropped: attend over the completed blocks only.
        fails("dropped partial", head.project(one_query_residual(&head.query, &f.blocks, f.tokens, nb, h)));
        // No residual: the last hidden state straight into the norm.
        fails("no residual", head.project(f.partial.clone()));
        // A zero query: uniform weights over the states.
        let flat = Head { query: vec![0.0; h], ..bind(&f, &f.unembed) };
        fails("zero query", flat.forward(&f.blocks, nb, &f.partial));
        // Norm gain dropped.
        let unit = Head { gain: vec![1.0; h], ..bind(&f, &f.unembed) };
        fails("unweighted norm", unit.forward(&f.blocks, nb, &f.partial));
        // Unembedding rows shifted by one.
        let mut shifted = f.unembed.clone();
        shifted.rotate_left(h * 2);
        fails("shifted rows", bind(&f, &shifted).forward(&f.blocks, nb, &f.partial));
        // No softcap.
        fails(
            "missing softcap",
            HeadOut { selected: good.selected.clone(), normed: good.normed.clone(), raw: good.raw.clone(), logits: good.raw.clone() },
        );
        // Tokens swapped in the output.
        let v = head.vocab();
        let mut logits = good.logits.clone();
        logits[..v].copy_from_slice(&good.logits[v..2 * v]);
        fails("swapped tokens", HeadOut { logits, ..good });
    }

    #[test]
    fn refuses_the_wrong_shapes() {
        let f = fixture();
        let (h, v, u) = (f.cfg.hidden_size, f.cfg.vocab_size, f.cfg.unpadded_vocab_size);
        let q = (Dtype::F32, vec![h], &f.query[..]);
        let n = (Dtype::Bf16, vec![h], &f.norm[..]);
        // Unpadded shape, as a converted HF checkpoint would store it.
        let short = (Dtype::Bf16, vec![u, h], &f.unembed[..u * h * 2]);
        assert!(Head::bind(&f.cfg, q.clone(), n.clone(), short).is_err());
        let full = (Dtype::Bf16, vec![v, h], &f.unembed[..]);
        assert!(Head::bind(&f.cfg, q.clone(), (Dtype::F32, vec![h], &f.norm[..]), full.clone()).is_err());
        assert!(Head::bind(&f.cfg, (Dtype::Bf16, vec![h], &f.query[..h * 2]), n, full).is_err());
    }
}
