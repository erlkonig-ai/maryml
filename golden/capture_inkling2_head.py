"""Run transformers' own Inkling 2 Mini output head and record it.

The head is the end of `Inkling2LoopTextModel.forward` and of
`Inkling2LoopForCausalLM.forward` at transformers 046bdb8c:

    hidden_states = self.final_input(block_states, hidden_states)[0]   # Inkling2LoopBlockAttentionResidual
    hidden_states = self.norm(hidden_states).to(inputs_embeds.dtype)   # Inkling2LoopRMSNorm
    logits = self.lm_head(hidden_states).float()                       # nn.Linear, no bias
    logits = self.logit_softcap(logits)                                # Inkling2LoopSoftcap

with the unembedding cut to `unpadded_vocab_size` rows by upstream's own
`Inkling2UnpadVocab` conversion op. This script instantiates those classes and
runs those three lines at bfloat16, the release's dtype; nothing here re-types
the arithmetic.

Two modes:

  tiny <transformers/src> <out.json>
      A deterministic fixture: hidden 16, padded vocab 24, unpadded 19, three
      tokens with three distinct block states and a distinct partial each, nonuniform values large enough that the softcap bends some
      logits. Weights and inputs are drawn from a seeded generator and rounded
      to bf16; every value is written as its exact bits (bf16 as u16, f32 as
      u32).

  real <transformers/src> <model dir> <out dir>
      The release's own `model.llm.attn_res.output_query`, `model.llm.norm.weight`
      and `model.llm.unembed.weight` read from its safetensors file, on two
      tokens of four block states and a partial each, built from embedding
      rows of the release and seeded gaussians. Writes raw little-endian arrays
      (block_states, partial, normed, raw as bf16; selected, logits as f32) and
      `reference.json` with shapes and the sha256 of the three weight tensors'
      bytes, so the Rust side can show it read the same bytes from the pile.

usage (offline container, no GPU):
  python3 -I capture_inkling2_head.py tiny <tf src> <out.json> [deps dir]
  python3 -I capture_inkling2_head.py real <tf src> <model dir> <out dir> [deps dir]
"""
import hashlib, json, os, sys, types

mode, tf_src = sys.argv[1:3]
rest = sys.argv[3:]
deps = rest[1:2] if mode == "tiny" else rest[2:3]
sys.path[:0] = [tf_src] + deps

import torch
import transformers

assert os.path.realpath(os.path.dirname(transformers.__file__)).startswith(os.path.realpath(tf_src)), transformers.__file__
from transformers.core_model_loading import Inkling2UnpadVocab
from transformers.models.inkling2_loop.modeling_inkling2_loop import (
    Inkling2LoopBlockAttentionResidual, Inkling2LoopRMSNorm, Inkling2LoopSoftcap)

torch.set_num_threads(4)
BF = torch.bfloat16


def head(query, norm_w, unembed_padded, blocks, partial, vocab, unpadded, eps, cap):
    """Upstream's classes, in upstream's order, at bf16. blocks [1, T, n, H], partial [1, T, H]."""
    hidden = norm_w.shape[0]
    text = types.SimpleNamespace(vocab_size=vocab, unpadded_vocab_size=unpadded,
                                 hidden_size=hidden, _attn_implementation="eager")
    cfg = types.SimpleNamespace(get_text_config=lambda: text)
    final_input = Inkling2LoopBlockAttentionResidual(text)
    lm_head = torch.nn.Linear(hidden, unpadded, bias=False, dtype=BF)
    model = types.SimpleNamespace(get_parameter=lambda name: lm_head.weight)
    cut = Inkling2UnpadVocab().convert(
        {"model.llm.unembed.weight": unembed_padded}, ["model.llm.unembed.weight"], ["lm_head.weight"],
        config=cfg, model=model, full_layer_name="lm_head.weight",
    )["lm_head.weight"]
    norm = Inkling2LoopRMSNorm(hidden, eps=eps).to(BF)
    softcap = Inkling2LoopSoftcap(cap)
    final_input.eval()
    # oneDNN's CPU bf16 GEMM (MKL-DNN 3.10.2 on aarch64, in this image) is
    # lossy: on the release's unembedding only 61% of its bf16 outputs equal
    # the exact dot of the same bf16 operands rounded once, off by up to 0.15.
    # With oneDNN off torch's own bf16 Linear accumulates in f32 (99.99% equal
    # to exact), which is what a bf16 linear means and what cuBLAS does. The
    # classes and their order are untouched; only the CPU backend is chosen.
    with torch.no_grad(), torch.backends.mkldnn.flags(enabled=False):
        final_input.query.copy_(query)
        norm.weight.copy_(norm_w)
        lm_head.weight.copy_(cut)
        # Inkling2LoopTextModel.forward, its last two lines
        selected = final_input(blocks, partial)[0]
        normed = norm(selected).to(partial.dtype)
        # Inkling2LoopForCausalLM.forward
        raw = lm_head(normed)
        logits = softcap(raw.float())
    assert selected.dtype == torch.float32 and normed.dtype == BF and raw.dtype == BF
    assert logits.dtype == torch.float32
    return selected[0], normed[0], raw[0], logits[0]


def bits16(t):
    return [b & 0xFFFF for b in t.contiguous().view(torch.int16).flatten().tolist()]


def bits32(t):
    return [b & 0xFFFFFFFF for b in t.contiguous().view(torch.int32).flatten().tolist()]


if mode == "tiny":
    (out_path,) = rest[:1]
    H, V, U, T, NB, EPS, CAP = 16, 24, 19, 3, 3, 1e-6, 30.0
    g = torch.Generator().manual_seed(20261122)
    # states of different scale, so both norms matter
    scales = torch.tensor([0.3, 2.0, 7.5, 40.0])
    blocks = (torch.randn(1, T, NB, H, generator=g) * scales[torch.randint(0, 4, (1, T, NB, 1), generator=g)]).to(BF)
    partial = (torch.randn(1, T, H, generator=g) * torch.tensor([[[0.5], [9.0], [3.0]]])).to(BF)
    query = torch.randn(H, generator=g) * 2.0
    norm_w = (0.5 + 1.5 * torch.rand(H, generator=g)).to(BF)
    unembed = (torch.randn(V, H, generator=g) * 4.0).to(BF)
    selected, normed, raw, logits = head(query, norm_w, unembed, blocks, partial, V, U, EPS, CAP)
    rec = {
        "source": "golden/capture_inkling2_head.py tiny, transformers 046bdb8cc2cee29d233964bb77fe6fd37cfe4aa3",
        "config": {"hidden_size": H, "vocab_size": V, "unpadded_vocab_size": U,
                   "rms_norm_eps": EPS, "final_logit_softcapping": CAP},
        "tokens": T,
        "n_blocks": NB,
        "block_states": bits16(blocks),
        "partial": bits16(partial),
        "output_query": bits32(query),
        "norm_weight": bits16(norm_w),
        "unembed_weight": bits16(unembed),
        "selected": bits32(selected),
        "normed": bits16(normed),
        "raw_logits": bits16(raw),
        "logits": bits32(logits),
    }
    with open(out_path, "w") as f:
        json.dump(rec, f, separators=(",", ":"))
        f.write("\n")
    print(f"tiny: max |raw| {raw.float().abs().max().item():.2f}, cap {CAP}", file=sys.stderr)
elif mode == "real":
    model_dir, out_dir = rest[:2]
    from safetensors import safe_open
    cfg = json.load(open(os.path.join(model_dir, "config.json")))["text_config"]
    st = os.path.join(model_dir, "model-00001-of-00001.safetensors")
    rows = [200005, 151083, 371, 13, 157021]
    with safe_open(st, "pt") as f:
        query = f.get_tensor("model.llm.attn_res.output_query")
        norm_w = f.get_tensor("model.llm.norm.weight")
        unembed = f.get_tensor("model.llm.unembed.weight")
        embed_rows = f.get_slice("model.llm.embed.weight")[rows, :]
    assert query.dtype == torch.float32 and norm_w.dtype == BF and unembed.dtype == BF
    digest = {k: hashlib.sha256(t.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
              for k, t in (("model.llm.attn_res.output_query", query), ("model.llm.norm.weight", norm_w),
                           ("model.llm.unembed.weight", unembed))}
    H = cfg["hidden_size"]
    g = torch.Generator().manual_seed(46)
    # token 0: four embedding rows as blocks, a gaussian as the partial;
    # token 1: four gaussians of different scale as blocks, an embedding row as the partial
    gauss = torch.randn(5, H, generator=g) * torch.tensor([[1.0], [0.5], [5.0], [20.0], [60.0]])
    blocks = torch.stack([embed_rows[:4].float(), gauss[1:5]]).unsqueeze(0).to(BF)  # [1, 2, 4, H]
    partial = torch.stack([gauss[0], embed_rows[4].float()]).unsqueeze(0).to(BF)    # [1, 2, H]
    selected, normed, raw, logits = head(query, norm_w, unembed, blocks, partial, cfg["vocab_size"],
                                         cfg["unpadded_vocab_size"], cfg["rms_norm_eps"],
                                         cfg["final_logit_softcapping"])
    with torch.no_grad():
        onednn_raw = torch.nn.functional.linear(normed, unembed[: cfg["unpadded_vocab_size"]])
        exact_raw = (normed.double() @ unembed[: cfg["unpadded_vocab_size"]].double().T).to(BF)
    backend = {
        "reference_raw_bit_equal_exact": (raw == exact_raw).float().mean().item(),
        "reference_raw_max_abs_from_exact": (raw.double() - exact_raw.double()).abs().max().item(),
        "onednn_raw_bit_equal_exact": (onednn_raw == exact_raw).float().mean().item(),
        "onednn_raw_max_abs_from_exact": (onednn_raw.double() - exact_raw.double()).abs().max().item(),
        "torch": torch.__version__,
    }
    print(f"real: backend {backend}", file=sys.stderr)
    os.makedirs(out_dir, exist_ok=True)
    arrays = {"block_states": blocks[0], "partial": partial[0], "selected": selected,
              "normed": normed, "raw": raw, "logits": logits}
    for k, t in arrays.items():
        open(os.path.join(out_dir, k + ".bin"), "wb").write(t.contiguous().view(torch.uint8).numpy().tobytes())
    json.dump({
        "source": "golden/capture_inkling2_head.py real, transformers 046bdb8cc2cee29d233964bb77fe6fd37cfe4aa3",
        "inputs": {"token 0": "blocks = embed rows 200005 151083 371 13, partial = gaussian x1 (seed 46)",
                   "token 1": "blocks = gaussians x0.5 x5 x20 x60 (seed 46), partial = embed row 157021"},
        "shapes": {k: list(t.shape) for k, t in arrays.items()},
        "dtypes": {k: str(t.dtype) for k, t in arrays.items()},
        "weight_sha256": digest,
        "cpu_backend": "torch.backends.mkldnn disabled for the head",
        "backend_check": backend,
        "argmax": logits.argmax(-1).tolist(),
    }, open(os.path.join(out_dir, "reference.json"), "w"), indent=1)
    print(f"real: argmax {logits.argmax(-1).tolist()}", file=sys.stderr)
else:
    sys.exit(f"unknown mode {mode}")
