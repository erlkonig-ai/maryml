"""Run transformers' own Inkling 2 Mini post-loop tail (decoder layers 18..23 and the output head) and record it.

The tail is the end of `Inkling2LoopTextModel.forward` at transformers 046bdb8c, its "Post loop layers" loop and
its last two lines, then `Inkling2LoopForCausalLM`'s unembedding and softcap:

    for decoder_layer in self.layers[loop_end:]:
        new_hidden_states = decoder_layer(hidden_states, block_states, attention_mask=..., relative_distance=...,
                                          conv_mask=..., cache=cache, position_ids=position_ids)
        if decoder_layer.is_block_boundary:
            block_states = torch.cat([block_states, hidden_states.unsqueeze(-2)], dim=-2)
        hidden_states = new_hidden_states
    hidden_states = self.final_input(block_states, hidden_states)[0]
    hidden_states = self.norm(hidden_states).to(inputs_embeds.dtype)
    logits = self.logit_softcap(self.lm_head(hidden_states).float())

Layers 18..22 are sliding (18 is a block boundary), 23 is full attention; 18 and 19 hold NVFP4 experts, 20..23
MXFP8. The tail's inputs are FIXED: the hidden state entering layer 18 [t, H] and the block states [t, 3, H]
(in the model: the normed embeddings, and the states entering 6 and 12 on the last pass). No cache: one prefill.

How the tail is built. A whole Mini at bf16 does not fit beside what else runs on the box (about 50 GB of
dequantised experts), so `real` builds only the six layers and the head: upstream's own classes
(`Inkling2LoopSlidingDecoderLayer`, `Inkling2LoopFullDecoderLayer`, `Inkling2LoopBlockAttentionResidual`,
`Inkling2LoopRMSNorm`, `nn.Linear`, `Inkling2LoopSoftcap`), the fp32 submodules of
`_keep_in_fp32_modules_strict` kept fp32, the checkpoint names mapped to parameter names by the table below, and
the expert weights produced by upstream's own conversion ops (`Inkling2LoopDequantizeExperts`, then
`Interleave(dim=1)` for w13; `Inkling2UnpadVocab` for the unembedding). `tiny` proves that construction: on a
synthetic checkpoint in the release's stored formats it loads the same six layers and head with
`from_pretrained` and asserts every parameter is the same dtype and the same bits as the manual build, and that
the whole model's own forward, with layer 18's input replaced by the fixed one (a forward pre-hook), produces
the same logits bit for bit as the manual tail; the masks the model handed layer 18 are compared with the ones
built here too.

Two runs, as in capture_inkling2_global5.py: float32 and bfloat16. The float32 run's tail is the reference
chain; its per-layer inputs are recorded, and the bfloat16 run is fed the same inputs twice over: as a chain
from the fixed input (`bf16.chain.*`) and layer by layer from the float32 run's own input to each layer
(`bf16.iso.*`), so their distance to float32 calibrates a tolerance for the chain and for each layer alone.

oneDNN (torch.backends.mkldnn) is disabled for both runs: in this image its CPU bf16 GEMM is lossy (on the
release's unembedding only 61% of outputs equal the exact dot of the same operands rounded once). The classes
and their order are untouched; only the CPU backend is chosen. Recorded in the metadata.

Recorded, float32 (indices int32), L in 18..23, run in f32 / bf16, mode in chain / iso (f32 has chain only):
  input.x / input.blocks / input.positions        the fixed tail input [t, H] / [t, 3, H] / [t]
  f32.in.L.x / f32.in.L.blocks                     the f32 chain's input to layer L
  <run>.<mode>.L.out / attn / attn_conv / mlp_input / experts / mlp_conv       [t, H]
  <run>.<mode>.L.router_logits / router_weights / router_indices               [t, E] / [t, k] / [t, k]
  <run>.<mode>.23.q_in / k_in / v_in               layer 23's three attention inputs [t, H]
  <run>.<mode>.head.selected / normed / raw / logits   the head's taps [t, H] / [t, H] / [t, V] / [t, V]
  tiny only: w.<checkpoint name>                   the synthetic checkpoint's tail tensors, as stored
  real only: (none of the weights; metadata records the sha256 of every tail tensor read)

usage (offline container, no GPU):
  python3 -I capture_inkling2_tail.py tiny <tf src> <scratch dir> <out.safetensors> [deps dir]
  python3 -I capture_inkling2_tail.py real <tf src> <model dir> <out.safetensors> [deps dir]
  python3 -I capture_inkling2_tail.py chained <tf src> <model dir> <out.safetensors> <deps dir> <core ref> <chain5 ref>

`chained` is upstream end to end on the chain5 capture's token ids, stage by stage: layers 0..5 (chain5 capture),
the core (core capture, from chain5's f32 layer-5 output rounded to bf16 once) and this tail from each run's own
core output. It adds `ids` [t], `f32.logits` and `bf16.logits` [t, V] for inkling2_session_check. The bf16 run
shares the layer-5 seam with the f32 run, so its distance to f32 understates a whole bf16 run's.
"""
import hashlib, json, os, sys, types

mode, tf_src, where, out_path = sys.argv[1:5]
sys.path[:0] = [tf_src] + sys.argv[5:6]

import torch
import transformers
from safetensors import safe_open
from safetensors.torch import save_file

assert os.path.realpath(os.path.dirname(transformers.__file__)).startswith(os.path.realpath(tf_src)), transformers.__file__
from transformers import masking_utils
from transformers.core_model_loading import Inkling2LoopDequantizeExperts, Inkling2UnpadVocab, Interleave
from transformers.models.inkling2_loop import modeling_inkling2_loop as M
from transformers.models.inkling2_loop.configuration_inkling2_loop import Inkling2LoopConfig

torch.backends.mkldnn.enabled = False
torch.set_num_threads(8)
BF, F32 = torch.bfloat16, torch.float32
FIRST, LAST = 18, 23
KEEP_FP32 = M.Inkling2LoopPreTrainedModel._keep_in_fp32_modules_strict


# --- the checkpoint-name table: stored name -> parameter of the layer module ------------------------------------

def layer_names(tc, L):
    """`model.llm.*` stored names of layer L's non-expert tensors -> parameter names inside the layer module."""
    p = f"model.llm.layers.{L}."
    sliding = tc.layer_types[L] == "hybrid_sliding"
    m = {
        p + "mlp_norm.weight": "post_attention_layernorm.weight",
        p + "attn.wq_du.weight": "self_attn.q_proj.linear.weight",
        p + "attn.wk_dv.weight": "self_attn.k_proj.linear.weight",
        p + "attn.wv_dv.weight": "self_attn.v_proj.linear.weight",
        p + "attn.wr_du.weight": "self_attn.relative_position_bias.linear.weight",
        p + "attn.q_l2_clip.scale": "self_attn.q_proj.norm.scale",
        p + "attn.rel_l2_clip.scale": "self_attn.relative_position_bias.norm.scale",
        p + "attn.rel_logits_proj.proj": "self_attn.relative_position_bias.proj",
        p + "attn.wo_ud.weight": "self_attn.o_proj.linear.weight",
        p + "attn.gate_attn.weight": "self_attn.o_proj.gate_proj.weight",
        p + "attn.attn_sink_proj.weight": "self_attn.sink_proj.weight",
        p + "attn_sconv.weight": "post_attention_convolution.conv1d.weight",
        p + "mlp_sconv.weight": "post_mlp_convolution.conv1d.weight",
        p + "mlp.gate.weight": "mlp.gate.weight",
        p + "mlp.gate.bias": "mlp.gate.e_score_correction_bias",
        # attn_queries.N is layer N + 1's attention input (upstream renames with offset=1)
        f"model.llm.attn_res.attn_queries.{L - 1}": "attention_input.query",
        f"model.llm.attn_res.mlp_queries.{L}": "mlp_input.query",
    }
    if sliding:
        m[p + "attn_norm.weight"] = "input_layernorm.weight"
        m[p + "attn.k_sconv.weight"] = "self_attn.k_proj.sconv.conv1d.weight"
        m[p + "attn.v_sconv.weight"] = "self_attn.v_proj.sconv.conv1d.weight"
    else:
        for s in "qkv":
            m[p + f"{s}_input_norm.weight"] = f"{s}_input_layernorm.weight"
        for s in "kv":
            m[p + f"attn.{s}_sconv.static_w"] = f"self_attn.{s}_proj.sconv.static_weight"
            m[p + f"attn.{s}_sconv.weight_proj"] = f"self_attn.{s}_proj.sconv.weight_proj.weight"
    return m


def expert_names(L, nvfp4):
    """Stored names of layer L's expert tensors, in the order of the conversion's source patterns (weight, scale,
    amax, denominator); MXFP8 has only the first two."""
    p = f"model.llm.layers.{L}.mlp.experts."
    tail = ("", "_scale", "_amax", "_outer_scale_denominator") if nvfp4 else ("", "_scale")
    return {mat: [p + mat + "_weight" + s for s in tail] for mat in ("w13", "w2")}


EXPERT_SUFFIXES = ("", "_scale", "_amax", "_outer_scale_denominator")


HEAD_NAMES = ("model.llm.attn_res.output_query", "model.llm.norm.weight", "model.llm.unembed.weight")


def tail_tensor_names(tc, get_dtype):
    names = []
    for L in range(FIRST, LAST + 1):
        names += list(layer_names(tc, L))
        nvfp4 = get_dtype(f"model.llm.layers.{L}.mlp.experts.w13_weight") == torch.uint8
        for v in expert_names(L, nvfp4).values():
            names += v
    return names + list(HEAD_NAMES)


# --- the manual build ----------------------------------------------------------------------------------------------

def keep_fp32(full_name):
    return any(k in full_name for k in KEEP_FP32)


def build_layer(tc, L, get, dtype):
    sliding = tc.layer_types[L] == "hybrid_sliding"
    cls = M.Inkling2LoopSlidingDecoderLayer if sliding else M.Inkling2LoopFullDecoderLayer
    layer = cls(tc, L).to(dtype)
    prefix = f"model.language_model.layers.{L}."
    for name, p in layer.named_parameters():
        if keep_fp32(prefix + name):
            p.data = p.data.to(F32)
    for name, b in layer.named_buffers():
        if keep_fp32(prefix + name):
            b.data = b.data.to(F32)
    params = dict(layer.named_parameters())
    params.update(dict(layer.named_buffers()))
    loaded = set()
    with torch.no_grad():
        for stored, pname in layer_names(tc, L).items():
            t = get(stored)
            assert params[pname].shape == t.shape, (stored, pname, params[pname].shape, t.shape)
            params[pname].copy_(t)
            loaded.add(pname)
        nvfp4 = get(f"model.llm.layers.{L}.mlp.experts.w13_weight").dtype == torch.uint8
        for mat, names in expert_names(L, nvfp4).items():
            target = "mlp.experts.gate_up_proj" if mat == "w13" else "mlp.experts.down_proj"
            fake = types.SimpleNamespace(get_parameter=lambda _n, p=params[target]: p)
            # the converter's four source patterns, with only the stored ones in its input (MXFP8: two)
            patterns = [names[0] + s for s in EXPERT_SUFFIXES[1:]]
            patterns.insert(0, names[0])
            # without a quantizer the loader casts every source to the parameter's dtype before converting: at
            # bf16 that rounds NVFP4's f32 amax and denominator (codes, E4M3 and UE8M0 scales are exact in bf16)
            dt = params[target].dtype
            out = Inkling2LoopDequantizeExperts().convert(
                {n: [get(n).to(dt)] for n in names}, patterns, [target], model=fake, full_layer_name=target)[target]
            if mat == "w13":
                out = Interleave(dim=1).convert({target: out}, [target], [target])[target]
            assert out.dtype == params[target].dtype and out.shape == params[target].shape, (target, out.dtype, out.shape)
            params[target].copy_(out)
            loaded.add(target)
    missing = sorted(set(params) - loaded)
    assert not missing, (L, missing)
    return layer.eval()


def build_head(tc, get, dtype):
    final_input = M.Inkling2LoopBlockAttentionResidual(tc)  # fp32 (kept)
    norm = M.Inkling2LoopRMSNorm(tc.hidden_size, eps=tc.rms_norm_eps).to(dtype)
    lm_head = torch.nn.Linear(tc.hidden_size, tc.unpadded_vocab_size, bias=False, dtype=dtype)
    model = types.SimpleNamespace(get_parameter=lambda _n: lm_head.weight)
    cfg = types.SimpleNamespace(get_text_config=lambda: tc)
    cut = Inkling2UnpadVocab().convert(
        {"model.llm.unembed.weight": get("model.llm.unembed.weight")}, ["model.llm.unembed.weight"],
        ["lm_head.weight"], config=cfg, model=model, full_layer_name="lm_head.weight")["lm_head.weight"]
    with torch.no_grad():
        final_input.query.copy_(get("model.llm.attn_res.output_query"))
        norm.weight.copy_(get("model.llm.norm.weight"))
        lm_head.weight.copy_(cut)
    return final_input.eval(), norm, lm_head, M.Inkling2LoopSoftcap(tc.final_logit_softcapping)


# --- running it --------------------------------------------------------------------------------------------------

def masks(tc, x, positions):
    """What `Inkling2LoopTextModel.forward` builds for a prefill with no cache and no padding."""
    emb = x.unsqueeze(0)
    kw = {"config": tc, "inputs_embeds": emb, "attention_mask": None, "cache": None,
          "position_ids": torch.arange(x.shape[0]).unsqueeze(0)}
    mask = {"full_attention": masking_utils.create_causal_mask(**kw),
            "sliding_attention": masking_utils.create_sliding_window_causal_mask(**kw),
            "linear_attention": masking_utils.create_recurrent_attention_mask(**kw)}
    dist = {"full_attention": M.get_relative_distance(emb, None, is_sliding=False),
            "sliding_attention": M.get_relative_distance(emb, None, is_sliding=True)}
    return mask, dist


def taps_of(layer, full):
    taps, hs = {}, []

    def hook(name, pick=lambda o: o):
        def f(_m, _i, o):
            taps[name] = pick(o)
        return f

    hs.append(layer.self_attn.register_forward_hook(hook("attn", lambda o: o[0])))
    hs.append(layer.post_attention_convolution.register_forward_hook(hook("attn_conv")))
    hs.append(layer.mlp_input.register_forward_hook(hook("mlp_input", lambda o: o[0])))
    hs.append(layer.mlp.gate.register_forward_hook(hook("router")))
    hs.append(layer.mlp.experts.register_forward_hook(hook("experts")))
    hs.append(layer.post_mlp_convolution.register_forward_hook(hook("mlp_conv")))
    if full:
        hs.append(layer.attention_input.register_forward_hook(hook("qkv_in")))
    return taps, hs


def record_layer(rec, key, taps, out, full):
    for n in ("attn", "attn_conv", "mlp_input", "mlp_conv"):
        rec[f"{key}.{n}"] = taps[n][0].float().contiguous()
    rec[f"{key}.experts"] = taps["experts"].float().contiguous()
    logits, weights, idx = taps["router"]
    rec[f"{key}.router_logits"] = logits.float().contiguous()
    rec[f"{key}.router_weights"] = weights.float().contiguous()
    rec[f"{key}.router_indices"] = idx.to(torch.int32).contiguous()
    rec[f"{key}.out"] = out[0].float().contiguous()
    if full:
        for n, v in zip(("q_in", "k_in", "v_in"), taps["qkv_in"]):
            rec[f"{key}.{n}"] = v[0].float().contiguous()


def run_layer(tc, layer, x, blocks, positions):
    """One post-loop visit, as the text model calls it. x [t, H], blocks [t, n, H]."""
    mask, dist = masks(tc, x, positions)
    kind = layer.attention_type
    with torch.no_grad():
        out = layer(x.unsqueeze(0), blocks.unsqueeze(0), attention_mask=mask[kind], relative_distance=dist[kind],
                    conv_mask=mask["linear_attention"], cache=None, position_ids=positions.unsqueeze(0))
        if layer.is_block_boundary:
            blocks = torch.cat([blocks, x.unsqueeze(-2)], dim=-2)
    return out, blocks


def run_head(head, blocks, x, dtype):
    final_input, norm, lm_head, softcap = head
    with torch.no_grad():
        selected = final_input(blocks.unsqueeze(0), x.unsqueeze(0))[0]
        normed = norm(selected).to(dtype)
        raw = lm_head(normed)
        logits = softcap(raw.float())
    return {"selected": selected[0], "normed": normed[0], "raw": raw[0], "logits": logits[0]}


def tail(tc, get, x0, blocks0, positions, rec, layers_cache=None, per_run=None):
    """Both runs. `layers_cache`, when given, keeps the built modules (tiny only, for the from_pretrained proof).
    `per_run`, when given, maps a run to its own (x0, blocks0) chain input (chained only)."""
    f32_inputs = {}
    for run, dtype in (("f32", F32), ("bf16", BF)):
        x, blocks = per_run[run] if per_run else (x0, blocks0)
        x, blocks = x.to(dtype), blocks.to(dtype)
        for L in range(FIRST, LAST + 1):
            layer = build_layer(tc, L, get, dtype)
            full = tc.layer_types[L] != "hybrid_sliding"
            if layers_cache is not None:
                layers_cache[(run, L)] = layer
            if run == "f32":
                f32_inputs[L] = (x, blocks)
                rec[f"f32.in.{L}.x"] = x.float().contiguous()
                rec[f"f32.in.{L}.blocks"] = blocks.float().contiguous()
            taps, hs = taps_of(layer, full)
            out, nb = run_layer(tc, layer, x, blocks, positions)
            record_layer(rec, f"{run}.chain.{L}", taps, out, full)
            if run == "bf16":
                ix, ib = f32_inputs[L]
                out_i, _ = run_layer(tc, layer, ix.to(BF), ib.to(BF), positions)
                record_layer(rec, f"{run}.iso.{L}", taps, out_i, full)
            for h in hs:
                h.remove()
            x, blocks = out[0], nb
            del layer
        head = build_head(tc, get, dtype)
        if layers_cache is not None:
            layers_cache[(run, "head")] = head
        for k, v in run_head(head, blocks, x, dtype).items():
            rec[f"{run}.chain.head.{k}"] = v.float().contiguous()
        if run == "f32":
            rec["f32.in.head.x"] = x.float().contiguous()
            rec["f32.in.head.blocks"] = blocks.float().contiguous()
        else:
            ix, ib = rec["f32.in.head.x"], rec["f32.in.head.blocks"]
            for k, v in run_head(head, ib.to(BF), ix.to(BF), dtype).items():
                rec[f"{run}.iso.head.{k}"] = v.float().contiguous()


# --- tiny: a synthetic checkpoint, and the proof that the manual build is from_pretrained's ------------------------

def tiny_config():
    here = os.path.dirname(os.path.abspath(__file__))
    cfg = json.load(open(os.path.join(here, "..", "src", "models", "inkling2_loop", "fixtures", "config.json")))
    t = cfg["text_config"]
    t.update(hidden_size=128, vocab_size=64, unpadded_vocab_size=56, num_attention_heads=8, num_key_value_heads=2,
             head_dim=16, swa_num_attention_heads=8, swa_num_key_value_heads=2, swa_head_dim=16, d_rel=4,
             rel_extent=4, sliding_window_size=4, n_routed_experts=8, num_experts_per_tok=2, intermediate_size=128,
             log_scaling_n_floor=2, model_max_length=4096)
    # intermediate 128, not 64: Mary's tuned BF16 lane (cubek) is wrong at k = 64 with more than one row
    # (2026-11-22, every row but the last), which no release shape has; the fixture must test the layers, not that
    t["loop_config"]["gate_rank"] = 4
    v = cfg["vision_config"]
    v.update(text_hidden_size=128, projection_hidden_dim=32, hidden_size=32, num_hidden_layers=1,
             num_attention_heads=2, posemb_grid_size=4)
    m = cfg["mtp_config"]
    m.update(hidden_size=128, num_attention_heads=8, num_key_value_heads=2, head_dim=16, intermediate_size=64,
             markov_rank=8)
    return cfg


def synthetic(tc, g):
    """The tail's tensors in the release's stored formats, from a seeded generator."""
    H, E, I, k = tc.hidden_size, tc.n_routed_experts, tc.moe_intermediate_size, tc.conv_kernel_size
    st = {}
    rnd = lambda *s, sc=1.0: torch.randn(*s, generator=g) * sc
    bf = lambda *s, sc=1.0: rnd(*s, sc=sc).to(BF)
    for L in range(FIRST, LAST + 1):
        sliding = tc.layer_types[L] == "hybrid_sliding"
        heads = tc.swa_num_attention_heads if sliding else tc.num_attention_heads
        kvh = tc.swa_num_key_value_heads if sliding else tc.num_key_value_heads
        hd = tc.swa_head_dim if sliding else tc.head_dim
        ext = tc.sliding_window_size if sliding else tc.rel_extent
        q, kv = heads * hd, kvh * hd
        p = f"model.llm.layers.{L}."
        st[p + "mlp_norm.weight"] = (0.5 + torch.rand(H, generator=g)).to(BF)
        if sliding:
            st[p + "attn_norm.weight"] = (0.5 + torch.rand(H, generator=g)).to(BF)
            st[p + "attn.k_sconv.weight"] = bf(kv, 1, k, sc=0.3)
            st[p + "attn.v_sconv.weight"] = bf(kv, 1, k, sc=0.3)
        else:
            for s in "qkv":
                st[p + f"{s}_input_norm.weight"] = (0.5 + torch.rand(H, generator=g)).to(BF)
            for s in "kv":
                st[p + f"attn.{s}_sconv.static_w"] = bf(k, kv, sc=0.2)
                st[p + f"attn.{s}_sconv.weight_proj"] = bf(kvh * k, H, sc=0.05)
        st[p + "attn.wq_du.weight"] = bf(q, H, sc=0.1)
        st[p + "attn.wk_dv.weight"] = bf(kv, H, sc=0.1)
        st[p + "attn.wv_dv.weight"] = bf(kv, H, sc=0.1)
        st[p + "attn.wr_du.weight"] = bf(heads * tc.d_rel, H, sc=0.1)
        # scales from 0.5 to 12, so some heads' vectors are longer than the clip thresholds
        st[p + "attn.q_l2_clip.scale"] = (0.5 + 11.5 * torch.rand(heads, hd, generator=g)).to(BF)
        st[p + "attn.rel_l2_clip.scale"] = (0.5 + 11.5 * torch.rand(heads, tc.d_rel, generator=g)).to(BF)
        st[p + "attn.rel_logits_proj.proj"] = bf(tc.d_rel, ext)
        st[p + "attn.wo_ud.weight"] = bf(H, q, sc=0.1)
        st[p + "attn.gate_attn.weight"] = bf(q, H, sc=0.1)
        st[p + "attn.attn_sink_proj.weight"] = bf(heads, H, sc=0.1)
        st[p + "attn_sconv.weight"] = bf(H, 1, k, sc=0.3)
        st[p + "mlp_sconv.weight"] = bf(H, 1, k, sc=0.3)
        st[p + "mlp.gate.weight"] = bf(E, H, sc=0.1)
        st[p + "mlp.gate.bias"] = rnd(E, sc=0.02)
        st[f"model.llm.attn_res.attn_queries.{L - 1}"] = rnd(H, sc=2.0) if sliding else rnd(3, H, sc=2.0)
        st[f"model.llm.attn_res.mlp_queries.{L}"] = rnd(H, sc=2.0)
        x = f"{p}mlp.experts."
        if L < 20:  # NVFP4, as the release
            for mat, (rows, cols) in (("w13", (2 * I, H)), ("w2", (H, I))):
                st[x + mat + "_weight"] = torch.randint(0, 256, (E, rows, cols // 2), generator=g, dtype=torch.uint8)
                st[x + mat + "_weight_scale"] = torch.exp2(torch.rand(E, rows, cols // 16, generator=g) * 3 - 2).to(torch.float8_e4m3fn)
                st[x + mat + "_weight_amax"] = 0.5 + torch.rand(E, generator=g)
                st[x + mat + "_weight_outer_scale_denominator"] = torch.tensor([40.0])
        else:  # MXFP8
            for mat, (rows, cols) in (("w13", (2 * I, H)), ("w2", (H, I))):
                st[x + mat + "_weight"] = rnd(E, rows, cols, sc=2.0).to(torch.float8_e4m3fn)
                st[x + mat + "_weight_scale"] = torch.randint(122, 126, (E, rows, cols // 32), generator=g, dtype=torch.uint8)
    st["model.llm.attn_res.output_query"] = rnd(H, sc=2.0)
    st["model.llm.norm.weight"] = (0.5 + torch.rand(H, generator=g)).to(BF)
    st["model.llm.unembed.weight"] = bf(tc.vocab_size, H, sc=4.0)
    return st


def prove_from_pretrained(scratch, cfg, st, x0, blocks0, positions, built, rec):
    """Every tail parameter of from_pretrained's model equals the manual build's, and so does its own forward."""
    save_file(st, os.path.join(scratch, "model.safetensors"))
    checked = 0
    for run, dtype in (("f32", F32), ("bf16", BF)):
        model, info = M.Inkling2LoopForConditionalGeneration.from_pretrained(
            scratch, dtype=dtype, attn_implementation="eager", output_loading_info=True)
        model.eval()
        assert not info["unexpected_keys"], info["unexpected_keys"]
        lm = model.model.language_model
        tail_missing = [k for k in info["missing_keys"]
                        if any(f"layers.{L}." in k for L in range(FIRST, LAST + 1))
                        or "final_input" in k or k.endswith("language_model.norm.weight") or "lm_head" in k]
        assert not tail_missing, tail_missing
        for L in range(FIRST, LAST + 1):
            ours = dict(built[(run, L)].named_parameters()) | dict(built[(run, L)].named_buffers())
            theirs = dict(lm.layers[L].named_parameters()) | dict(lm.layers[L].named_buffers())
            assert set(ours) == set(theirs), (L, set(ours) ^ set(theirs))
            for n, p in theirs.items():
                assert p.dtype == ours[n].dtype, (run, L, n, p.dtype, ours[n].dtype)
                assert torch.equal(p.view(torch.uint8) if p.dtype != torch.bool else p,
                                   ours[n].view(torch.uint8) if p.dtype != torch.bool else ours[n]), (run, L, n)
                checked += 1
        fi, norm, head, _ = built[(run, "head")]
        for a, b, n in ((lm.final_input.query, fi.query, "final_input.query"), (lm.norm.weight, norm.weight, "norm"),
                        (model.lm_head.weight, head.weight, "lm_head")):
            assert a.dtype == b.dtype and torch.equal(a.view(torch.uint8), b.view(torch.uint8)), (run, n)
            checked += 1
        # from_pretrained's own modules through the post-loop steps give the manual build's logits, bit for bit
        x, blocks = x0.to(dtype), blocks0.to(dtype)
        for L in range(FIRST, LAST + 1):
            out, blocks = run_layer(lm.config, lm.layers[L], x, blocks, positions)
            x = out[0]
        theirs = run_head((lm.final_input, lm.norm, model.lm_head, model.logit_softcap), blocks, x, dtype)["logits"]
        ours = rec[f"{run}.chain.head.logits"]
        assert torch.equal(theirs.float(), ours), (run, (theirs.float() - ours).abs().max())
        # and the masks the model itself hands its sliding (18) and full (23) layers are the ones built here
        seen = {}

        def pre(L):
            def f(_m, _args, kwargs):
                seen[L] = (kwargs["attention_mask"], kwargs["relative_distance"], kwargs["conv_mask"])
            return f

        hs = [lm.layers[L].register_forward_pre_hook(pre(L), with_kwargs=True) for L in (FIRST, LAST)]
        with torch.no_grad():
            model(input_ids=torch.zeros(1, x0.shape[0], dtype=torch.long), use_cache=False)
        for h in hs:
            h.remove()
        mask, dist = masks(lm.config, x0.to(dtype), positions)
        same = lambda a, b: (a is None and b is None) or (a is not None and b is not None and torch.equal(a, b))
        for L, kind in ((FIRST, "sliding_attention"), (LAST, "full_attention")):
            m, d, c = seen[L]
            assert same(m, mask[kind]) and same(d, dist[kind]) and same(c, mask["linear_attention"]), (run, L)
        print(f"tiny {run}: from_pretrained tail parameters and forward equal the manual build", file=sys.stderr)
        del model
    return checked


def main():
    meta = {"transformers": "046bdb8cc2cee29d233964bb77fe6fd37cfe4aa3", "torch": torch.__version__,
            "device": "cpu", "cpu_backend": "torch.backends.mkldnn disabled", "mode": mode,
            "attn_implementation": "eager", "cache": "none (one prefill)"}
    rec = {}
    if mode == "tiny":
        cfg = tiny_config()
        os.makedirs(where, exist_ok=True)
        json.dump(cfg, open(os.path.join(where, "config.json"), "w"))
        config = Inkling2LoopConfig.from_pretrained(where)
        tc = config.text_config
        tc._attn_implementation = "eager"
        g = torch.Generator().manual_seed(20261122)
        st = synthetic(tc, g)
        T = 7
        x0 = (torch.randn(T, tc.hidden_size, generator=g) * torch.tensor([0.5, 2, 6, 1, 20, 3, 9])[:, None]).to(BF)
        blocks0 = (torch.randn(T, 3, tc.hidden_size, generator=g) * torch.tensor([1.0, 4.0, 12.0])[None, :, None]).to(BF)
        positions = torch.arange(T)
        get = lambda n: st[n]
        built = {}
        tail(tc, get, x0, blocks0, positions, rec, layers_cache=built)
        meta["from_pretrained_checked_tensors"] = str(prove_from_pretrained(where, cfg, st, x0, blocks0, positions, built, rec))
        for n, t in st.items():
            rec["w." + n] = t.contiguous()
        meta["text_config"] = json.dumps(cfg["text_config"])
    elif mode == "real":
        config = Inkling2LoopConfig.from_pretrained(where)
        tc = config.text_config
        tc._attn_implementation = "eager"
        st_path = os.path.join(where, "model-00001-of-00001.safetensors")
        f = safe_open(st_path, "pt")
        digest = {}
        cache = {}

        def get(n):
            if n not in cache:
                t = f.get_tensor(n)
                digest[n] = hashlib.sha256(t.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
                cache[n] = t
            return cache[n]

        T, ORIGIN = 4, 9000
        g = torch.Generator().manual_seed(18)
        rows = [200005, 151083, 371, 13]
        # Inkling2LoopNormedEmbedding is `embed_norm(embedding(ids))`; its norm, on the rows it would look up
        embed_norm = M.Inkling2LoopRMSNorm(tc.hidden_size, eps=tc.rms_norm_eps).to(BF)
        with torch.no_grad():
            embed_norm.weight.copy_(f.get_tensor("model.llm.embed_norm.weight"))
            normed = embed_norm(f.get_slice("model.llm.embed.weight")[rows, :])
        H = tc.hidden_size
        b1 = torch.randn(T, H, generator=g) * 4.0
        b2 = torch.randn(T, H, generator=g) * 12.0
        blocks0 = torch.stack([normed.float(), b1, b2], dim=1).to(BF)
        x0 = (torch.randn(T, H, generator=g) * torch.tensor([0.5, 2.0, 8.0, 32.0])[:, None]).to(BF)
        positions = torch.arange(T) + ORIGIN
        # the cache holds every tail tensor read (about 5 GB), so the two runs read the file once
        tail(tc, get, x0, blocks0, positions, rec)
        missing = [n for n in tail_tensor_names(tc, lambda n: get(n).dtype) if n not in digest]
        assert not missing, missing
        meta["weight_sha256"] = json.dumps(digest, sort_keys=True)
        meta["inputs"] = ("blocks[:, 0] = embed_norm(embed rows 200005 151083 371 13), blocks[:, 1] / [:, 2] = "
                          "gaussians x4 / x12, x = gaussians x0.5 x2 x8 x32 (seed 18); positions 9000.. so the log "
                          "scale bends")
        meta["text_config"] = json.dumps(cfg_text(where))
    elif mode == "chained":
        # upstream end to end on real token ids, stage by stage: the core capture's own pass-3 output for the
        # whole sequence (which began from the chain5 capture's layer-5 output on the same ids) enters layer 18
        config = Inkling2LoopConfig.from_pretrained(where)
        tc = config.text_config
        tc._attn_implementation = "eager"
        f = safe_open(os.path.join(where, "model-00001-of-00001.safetensors"), "pt")
        digest = {}
        cache = {}

        def get(n):
            if n not in cache:
                t = f.get_tensor(n)
                digest[n] = hashlib.sha256(t.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
                cache[n] = t
            return cache[n]

        core_path, chain5_path = sys.argv[6], sys.argv[7]
        core = safe_open(core_path, "pt")
        c5 = safe_open(chain5_path, "pt")
        ids = torch.cat([c5.get_tensor("prompt"), c5.get_tensor("step_token")]).to(torch.int32)
        per_run = {run: (core.get_tensor(f"{run}.core.whole.hidden"), core.get_tensor(f"{run}.core.whole.blocks"))
                   for run in ("f32", "bf16")}
        x0, blocks0 = per_run["f32"]
        assert x0.shape[0] == ids.shape[0], (x0.shape, ids.shape)
        positions = torch.arange(ids.shape[0])
        tail(tc, get, x0, blocks0, positions, rec, per_run=per_run)
        rec["ids"] = ids
        rec["f32.logits"] = rec["f32.chain.head.logits"]
        rec["bf16.logits"] = rec["bf16.chain.head.logits"]
        sha = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()
        meta["weight_sha256"] = json.dumps(digest, sort_keys=True)
        meta["inputs"] = (f"ids = chain5 prompt + step_token; each run's layer-18 input and blocks = that run's "
                          f"core.whole.hidden / core.whole.blocks of the core capture (sha256 {sha(core_path)}), "
                          f"whose input was the chain5 capture's f32 layer-5 output (sha256 {sha(chain5_path)}), "
                          f"bf16-rounded once; positions 0..")
        meta["core_metadata_inputs"] = core.metadata().get("inputs", "")
        meta["text_config"] = json.dumps(cfg_text(where))
    else:
        sys.exit(f"unknown mode {mode}")
    rec["input.x"] = x0.float().contiguous()
    rec["input.blocks"] = blocks0.float().contiguous()
    rec["input.positions"] = positions.to(torch.int32)
    # several f32 inputs are views of one tensor (the blocks after 18 are not reallocated per layer)
    save_file({k: v.clone().contiguous() for k, v in rec.items()}, out_path, metadata=meta)
    for run in ("f32", "bf16"):
        lg = rec[f"{run}.chain.head.logits"]
        print(f"{mode} {run}: argmax {lg.argmax(-1).tolist()}", file=sys.stderr)


def cfg_text(model_dir):
    return json.load(open(os.path.join(model_dir, "config.json")))["text_config"]


main()
