"""Run transformers' own Inkling 2 Mini looped core (layers 6..17, three passes) on a fixed pre-loop state, and record
every visit, the pass boundaries and the core's output.

Upstream's code, as for layers 0, 5 and 6 (capture_inkling2_layer0/global5/loop6.py): the pinned transformers checkout
(046bdb8cc2cee29d233964bb77fe6fd37cfe4aa3) first on sys.path, eager attention, CPU, every layer built by
`from_pretrained` from the release's own tensors (key renames, NVFP4 dequantization, the w13 de-interleave and the
merge of the two cross-pass query tensors are upstream's). The full release config is kept (24 layers, core 6..18,
three passes), so cache slots and `cache_offset` are the real ones.

The core's input is FIXED, so the comparison measures the core alone: upstream's own float32 run of layers 0..5 on the
26-token prompt and its greedy step token (Dazov's chain5_reference_onednn_off.safetensors, oneDNN disabled):
  x       the pre-loop hidden state, f32.<phase>.layer5.out                 [t, H]
  emb     the one pre-loop block, the embedding, f32.<phase>.layer0.input   [t, H]
both rounded to bf16 once, and fed identically to both runs, so the bf16-vs-f32 spread is the core's own rounding.

Two drivers, both upstream's layer classes:
  1. per visit (both runs): a float32 core does not fit beside the other residents (12 layers of dequantized experts
     is ~49 GB), so each visit loads its one layer by `from_pretrained` (the other 23 decoder classes replaced at
     construction by an empty module, as in capture_inkling2_loop6.py), runs the prompt, the step token through that
     visit's cache slot, and all 27 rows with no cache, then frees it. The visits run in forward's order (pass-major),
     and the driver reproduces `Inkling2LoopTextModel.forward`'s loop lines: block states start at the pre-loop ones,
     a boundary layer appends its input, passes 2 and 3 restart from the pre-loop hidden and blocks and read
     `message_input(blocks[n_pre + 1:], previous pass output, loop_layer_idx)`, `cache_offset` by forward's formula.
     Masks and distances are built per visit from that visit's own slot (aliased into slots 0 and 5, which the mask
     functions read and which no core visit owns), so they are the sizes forward computes before any layer runs.
  2. forward itself (bf16 only, ~25 GB): `Inkling2LoopTextModel.forward` with the 12 real core layers, the pre- and
     post-loop layers stubbed (layers 0..4 return their input, layer 5 returns x, layers 18.. record what reaches
     them and return their input), called with inputs_embeds = emb and a cache passed in. Its 36 visit outputs and
     the hidden and block states that reach layer 18 are recorded as `fwd.*` and compared with driver 1's bf16 run
     inside this script; the metadata says whether they are bit-equal. This is what makes driver 1 upstream's loop
     and not a copy of it.

Recorded, float32 (indices int32), <run> in bf16 / f32, <p> pass 0..2, <l> layer 6..17, <phase> prefill/step/whole:
  x.<phase>, emb.<phase>                                                         inputs                [t, H]
  <run>.p<p>.l<l>.<phase>.out                                                    the visit's output    [t, H]
  <run>.p<p>.l<l>.<phase>.mlp_input / router_logits / router_indices                                  [t, H] / [t, 128] / [t, 8]
  <run>.p<p>.l<l>.<phase>.messages                     p > 0: the visit's four messages (Q K V MLP)   [t, 4, H]
  <run>.p<p>.l<l>.<phase>.gate_q / gate_k / gate_v / gate_mlp                      p > 0              [t, H]
  <run>.p<p>.l<l>.<phase>.k / .v / .conv0..conv3       prefill and step: the visit's cache slot      [kv, n, d] / [C, k]
  <run>.p<p>.<phase>.blocks                            block states at the end of pass p              [t, 3, H]
  <run>.core.<phase>.hidden / .blocks                  the core's output: pass 3's layer-17 output and blocks
  fwd.p<p>.l<l>.<phase>.out, fwd.core.<phase>.hidden / .blocks     driver 2 (bf16)

usage: python3 -I capture_inkling2_core.py <transformers/src> <model dir> <scratch dir> <chain5 reference>
                                           <out.safetensors> [deps dir]
"""
import copy, gc, hashlib, json, os, sys, time

tf_src, model_dir, scratch, c5_path, out_path = sys.argv[1:6]
sys.path[:0] = [tf_src] + sys.argv[6:7]

import torch
import torch.nn as nn
import transformers
from safetensors import safe_open
from safetensors.torch import save_file

assert os.path.realpath(os.path.dirname(transformers.__file__)).startswith(os.path.realpath(tf_src)), transformers.__file__
import transformers.models.inkling2_loop.modeling_inkling2_loop as M
from transformers.cache_utils import DynamicCache
from transformers.masking_utils import (create_causal_mask, create_recurrent_attention_mask,
                                        create_sliding_window_causal_mask)

torch.backends.mkldnn.enabled = False
T0 = time.time()


def log(*a):
    print(f"[{time.time() - T0:7.1f}s]", *a, file=sys.stderr, flush=True)


RealSL = M.Inkling2LoopSlidingLoopDecoderLayer
RealFL = M.Inkling2LoopFullLoopDecoderLayer
LIVE = set()  # the layers the next construction builds for real
STUB = {}  # what the stubs do in driver 2


class Absent(nn.Module):
    """Stands in for a decoder layer; holds no parameters. In driver 2 it is the pre-/post-loop stub."""

    def __init__(self, config, layer_idx):
        super().__init__()
        self.layer_idx = layer_idx
        self.is_block_boundary = layer_idx % config.n_layers_per_attn_res_block == 0
        self.attention_type = "sliding_attention" if config.layer_types[layer_idx] == "hybrid_sliding" else "full_attention"

    def forward(self, hidden_states, block_states, *args, **kwargs):
        if self.layer_idx == 5:
            return STUB["x"]
        if self.layer_idx == 18:
            STUB["reached"] = (hidden_states, block_states)
        return hidden_states


def factory(real):
    return lambda config, layer_idx: real(config, layer_idx) if layer_idx in LIVE else Absent(config, layer_idx)


M.Inkling2LoopSlidingDecoderLayer = Absent
M.Inkling2LoopFullDecoderLayer = Absent
M.Inkling2LoopSlidingLoopDecoderLayer = factory(RealSL)
M.Inkling2LoopFullLoopDecoderLayer = factory(RealFL)

CORE = list(range(6, 18))
os.makedirs(scratch, exist_ok=True)
cfg = json.load(open(os.path.join(model_dir, "config.json")))
cfg["vision_config"]["num_hidden_layers"] = 1
st_path = os.path.join(model_dir, "model-00001-of-00001.safetensors")
SHARED = ("model.llm.middle_cycle_cross_pass_queries", "model.llm.middle_cycle_cross_pass_kv_queries")


def layer_keys(f, L):
    # attn_queries.N is layer N + 1's attention input (upstream renames with offset=1)
    return [k for k in f.keys() if k.startswith(f"model.llm.layers.{L}.")
            or k in (f"model.llm.attn_res.attn_queries.{L - 1}", f"model.llm.attn_res.mlp_queries.{L}")]


# one directory per core layer (its tensors, the shared message queries, an index over both), and one indexing all
# twelve for driver 2; the per-layer files are written once
def write_once(path, keys, f):
    if not os.path.exists(path):
        save_file({k: f.get_tensor(k) for k in keys}, path + ".part")
        os.rename(path + ".part", path)


every = {}
with safe_open(st_path, "pt") as f:
    write_once(os.path.join(scratch, "queries.safetensors"), list(SHARED), f)
    for L in CORE:
        d = os.path.join(scratch, f"l{L}")
        os.makedirs(d, exist_ok=True)
        keys = layer_keys(f, L)
        write_once(os.path.join(d, "layer.safetensors"), keys, f)
        q = os.path.join(d, "queries.safetensors")
        if not os.path.lexists(q):
            os.symlink(os.path.join("..", "queries.safetensors"), q)
        json.dump(cfg, open(os.path.join(d, "config.json"), "w"))
        json.dump({"metadata": {}, "weight_map": {**{k: "layer.safetensors" for k in keys},
                                                   **{k: "queries.safetensors" for k in SHARED}}},
                  open(os.path.join(d, "model.safetensors.index.json"), "w"))
        link = os.path.join(scratch, "all", f"l{L}.safetensors")
        os.makedirs(os.path.dirname(link), exist_ok=True)
        if not os.path.lexists(link):
            os.symlink(os.path.join("..", f"l{L}", "layer.safetensors"), link)
        every.update({k: f"l{L}.safetensors" for k in keys})
alld = os.path.join(scratch, "all")
q = os.path.join(alld, "queries.safetensors")
if not os.path.lexists(q):
    os.symlink(os.path.join("..", "queries.safetensors"), q)
every.update({k: "queries.safetensors" for k in SHARED})
json.dump(cfg, open(os.path.join(alld, "config.json"), "w"))
json.dump({"metadata": {}, "weight_map": every}, open(os.path.join(alld, "model.safetensors.index.json"), "w"))
log("scratch prepared")

# fixed inputs
c5 = {}
with safe_open(c5_path, "pt") as f:
    meta5 = f.metadata()
    for ph in ("prefill", "step"):
        for k in (f"f32.{ph}.layer5.out", f"f32.{ph}.layer0.input"):
            c5[k] = f.get_tensor(k)
assert "enabled=False" in meta5.get("cpu_backend", ""), meta5
c5_sha = hashlib.sha256(open(c5_path, "rb").read()).hexdigest()


def bf(t):
    return t.to(torch.bfloat16).float()


x_all = bf(torch.cat([c5["f32.prefill.layer5.out"], c5["f32.step.layer5.out"]], 0))
emb_all = bf(torch.cat([c5["f32.prefill.layer0.input"], c5["f32.step.layer0.input"]], 0))
T1, H = x_all.shape
T = T1 - 1
PHASES = {"prefill": slice(0, T), "step": slice(T, T1), "whole": slice(0, T1)}
record = {}
for ph, sl in PHASES.items():
    record[f"x.{ph}"] = x_all[sl].clone()
    record[f"emb.{ph}"] = emb_all[sl].clone()


def load(dirname, dtype, live):
    LIVE.clear()
    LIVE.update(live)
    gc.collect()
    model, info = M.Inkling2LoopForConditionalGeneration.from_pretrained(
        os.path.join(scratch, dirname), dtype=dtype, attn_implementation="eager", output_loading_info=True)
    model.eval()
    lm = model.model.language_model
    ours = [k for k in info["missing_keys"]
            if any(f"layers.{L}." in k for L in live) or "message_input" in k]
    assert not ours, ours
    assert not info["unexpected_keys"], info["unexpected_keys"]
    for L in live:
        assert type(lm.layers[L]) in (RealSL, RealFL), (L, type(lm.layers[L]))
    return model, lm


def new_cache(tc):
    c = DynamicCache(config=tc)
    assert len(c.layers) == tc.num_hidden_layers
    return c


def with_loop_copies(c, tc):
    s, e = tc.loop_layer_index_start, tc.loop_layer_index_end
    c.layers += [copy.deepcopy(l) for l in c.layers[s:e] * (tc.loop_num_passes - 1)]
    return c


TAPS = {}


def hooks(layer):
    def hook(name, pick=lambda o: o):
        def f(_m, _i, o):
            TAPS[name] = pick(o)
        return f

    hs = [layer.register_forward_hook(hook("out")),
          layer.mlp_input.register_forward_hook(hook("mlp_input", lambda o: o[0])),
          layer.mlp.gate.register_forward_hook(hook("router"))]
    for g in ("q", "k", "v", "mlp"):
        hs.append(getattr(layer, f"{g}_message_gate").register_forward_hook(hook(f"gate_{g}")))
    return hs


def keep(tag, cache_layer, gated):
    record[f"{tag}.out"] = TAPS["out"][0].float().clone().contiguous()
    record[f"{tag}.mlp_input"] = TAPS["mlp_input"][0].float().clone().contiguous()
    logits, _w, idx = TAPS["router"]
    record[f"{tag}.router_logits"] = logits.float().clone().contiguous()
    record[f"{tag}.router_indices"] = idx.to(torch.int32).clone().contiguous()
    for g in ("q", "k", "v", "mlp"):
        assert (f"gate_{g}" in TAPS) == gated, (tag, g)
        if gated:
            record[f"{tag}.gate_{g}"] = TAPS[f"gate_{g}"][0].float().clone().contiguous()
    if cache_layer is not None:
        record[f"{tag}.k"] = cache_layer.keys[0].float().clone().contiguous()
        record[f"{tag}.v"] = cache_layer.values[0].float().clone().contiguous()
        for i in range(4):
            record[f"{tag}.conv{i}"] = cache_layer.conv_states[i][0].float().clone().contiguous()


# ---- driver 1: per visit, both runs
for run, dtype in (("bf16", torch.bfloat16), ("f32", torch.float32)):
    cache = None
    # per phase: the pre-loop state, then the running hidden and blocks of the pass in progress
    pre = {ph: (x_all[sl].unsqueeze(0).to(dtype), emb_all[sl].unsqueeze(0).unsqueeze(-2).to(dtype))
           for ph, sl in PHASES.items()}
    prev = {}
    for p in range(3):
        cur = {ph: [pre[ph][0], pre[ph][1]] for ph in PHASES}
        for L in CORE:
            model, lm = load(f"l{L}", dtype, [L])
            tc = lm.config
            s, e = tc.loop_layer_index_start, tc.loop_layer_index_end
            n_loop, n_pre = e - s, 1
            if cache is None:
                cache = with_loop_copies(new_cache(tc), tc)
            layer = lm.layers[L]
            hs = hooks(layer)
            slot = L if p == 0 else tc.num_hidden_layers + (p - 1) * n_loop + (L - s)
            offset = 0 if p == 0 else tc.num_hidden_layers + (p - 1) * n_loop - s
            # the mask functions read their sizes from the first sliding (0) and first full (5) slot; no core visit
            # owns those, so alias this visit's own slot there
            alias = 0 if layer.attention_type == "sliding_attention" else 5
            cache.layers[alias] = cache.layers[slot]
            with torch.no_grad():
                for ph, sl in PHASES.items():
                    c = None if ph == "whole" else cache
                    x, blocks = cur[ph]
                    start = sl.start
                    if c is not None:
                        assert c.layers[slot].get_seq_length() == start, (L, p, ph, c.layers[slot].get_seq_length())
                    pos = (torch.arange(x.shape[1]) + start).unsqueeze(0)
                    kw = {"config": tc, "inputs_embeds": x, "attention_mask": None, "cache": c, "position_ids": pos}
                    if layer.attention_type == "sliding_attention":
                        mask = create_sliding_window_causal_mask(**kw)
                    else:
                        mask = create_causal_mask(**kw)
                    conv = create_recurrent_attention_mask(**kw)
                    dist = M.get_relative_distance(x, c, is_sliding=layer.attention_type == "sliding_attention")
                    messages = None
                    tag = f"{run}.p{p}.l{L}.{ph}"
                    if p > 0:
                        pb, pp = prev[ph]
                        messages = lm.message_input(pb, pp, L - s)
                        record[f"{tag}.messages"] = messages[0].float().clone().contiguous()
                    TAPS.clear()
                    new = layer(x, blocks, messages, attention_mask=mask, relative_distance=dist, conv_mask=conv,
                                cache=c, cache_offset=offset, position_ids=pos)
                    keep(tag, None if c is None else c.layers[slot], p > 0)
                    if layer.is_block_boundary:
                        blocks = torch.cat([blocks, x.unsqueeze(-2)], dim=-2)
                    cur[ph] = [new, blocks]
            for h in hs:
                h.remove()
            del model, lm, layer
            log(f"{run} pass {p} layer {L} done")
        for ph in PHASES:
            hid, blocks = cur[ph]
            assert blocks.shape[-2] == n_pre + 2, blocks.shape
            record[f"{run}.p{p}.{ph}.blocks"] = blocks[0].float().clone().contiguous()
            # forward: previous_block_states = block_states[..., num_pre_loop_blocks + 1:, :]
            prev[ph] = (blocks[..., n_pre + 1:, :], hid)
    for ph in PHASES:
        hid, blocks = cur[ph]
        record[f"{run}.core.{ph}.hidden"] = hid[0].float().clone().contiguous()
        record[f"{run}.core.{ph}.blocks"] = blocks[0].float().clone().contiguous()

meta = {"transformers": "046bdb8cc2cee29d233964bb77fe6fd37cfe4aa3", "torch": torch.__version__, "device": "cpu",
        "attn_implementation": "eager", "layers": "6..17, three passes",
        "cpu_backend": f"torch.backends.mkldnn.enabled={torch.backends.mkldnn.enabled}, "
                       f"mkldnn available={torch.backends.mkldnn.is_available()}, threads={torch.get_num_threads()}",
        "inputs": f"x = f32.<phase>.layer5.out, emb = f32.<phase>.layer0.input of chain5 reference sha256 {c5_sha} "
                  f"({meta5.get('cpu_backend')}), bf16-rounded once; 26-token prompt + 1 step token",
        "cache_slots": "pass 0 = layer, pass p > 0 = 24 + (p - 1) * 12 + layer - 6",
        "conv_states": "0=k_proj.sconv 1=v_proj.sconv 2=post_attention_convolution 3=post_mlp_convolution"}
save_file(record, out_path, metadata={**meta, "forward_check": "not run"})
log(f"driver 1 saved: {len(record)} tensors")

# ---- driver 2: upstream's own forward, bf16, all twelve core layers live
model, lm = load("all", torch.bfloat16, CORE)
tc = lm.config
outs = {L: [] for L in CORE}
hs = [lm.layers[L].register_forward_hook(lambda m, i, o, L=L: outs[L].append(o[0].float().clone())) for L in CORE]
cache = new_cache(tc)
cache.layers[0] = cache.layers[6]  # see driver 1: the mask functions' slots, aliased to the core's first visits
cache.layers[5] = cache.layers[11]
fwd = {}
with torch.no_grad():
    for ph, sl in PHASES.items():
        for L in CORE:
            outs[L].clear()
        STUB.clear()
        STUB["x"] = x_all[sl].unsqueeze(0).to(torch.bfloat16)
        pos = (torch.arange(sl.stop - sl.start) + sl.start).unsqueeze(0)
        lm(inputs_embeds=emb_all[sl].unsqueeze(0).to(torch.bfloat16), position_ids=pos,
           cache=None if ph == "whole" else cache, use_cache=ph != "whole")
        for L in CORE:
            assert len(outs[L]) == 3, (L, len(outs[L]))
            for p in range(3):
                fwd[f"fwd.p{p}.l{L}.{ph}.out"] = outs[L][p].contiguous()
        hid, blocks = STUB["reached"]
        fwd[f"fwd.core.{ph}.hidden"] = hid[0].float().clone().contiguous()
        fwd[f"fwd.core.{ph}.blocks"] = blocks[0].float().clone().contiguous()
for h in hs:
    h.remove()
assert len(cache.layers) == 48, len(cache.layers)
worst = 0.0
unequal = []
for k, v in fwd.items():
    d = k.replace("fwd.", "bf16.", 1)
    if not torch.equal(v, record[d]):
        unequal.append(k)
        worst = max(worst, (v - record[d]).abs().max().item())
record.update(fwd)
check = ("all 36 visits x 3 phases and the core output bit-equal to driver 1's bf16 run" if not unequal else
         f"{len(unequal)} of {len(fwd)} differ from driver 1, max abs {worst:.3e}, first {unequal[:4]}")
log("forward check:", check)
save_file(record, out_path, metadata={**meta, "forward_check": check})
for k in sorted(record):
    print(k, tuple(record[k].shape), file=sys.stderr)
log("done")
