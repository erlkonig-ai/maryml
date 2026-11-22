"""Run transformers' own Inkling 2 Mini core layer 6 (a sliding loop layer), on its first pass and on a later pass
that reads cross-pass messages, and record both.

Upstream's code, as for layers 0 and 5 (capture_inkling2_layer0.py, capture_inkling2_global5.py): the pinned
transformers checkout (046bdb8cc2cee29d233964bb77fe6fd37cfe4aa3) first on sys.path, eager attention, CPU, the model
built by `from_pretrained` from the release's own tensors (so key renames, NVFP4 dequantization, the w13 de-interleave
and the merge of the two cross-pass query tensors are upstream's). The full release config is kept (24 layers, core
6..18, three passes), so cache slots are the real ones: layer 6's first pass is slot 6, its second pass slot 24.

Only layer 6 is built for real. The other 23 decoder layer classes are replaced, at construction only, by an empty
module, so their experts are never allocated; layer 6 is upstream's `Inkling2LoopSlidingLoopDecoderLayer` and its
forward is untouched. Layer 6 is called the way `Inkling2LoopTextModel.forward` calls a core layer: masks and relative
distances from upstream's functions, computed once per phase before any layer runs; a DynamicCache from the config
with the loop copies appended as forward appends them; `cache_offset` by forward's formula; messages from
`message_input(previous_blocks, previous_hidden, loop_layer_idx=0)`. Only layer 6 runs, so the cache's layer 0 (the
one the sliding mask and distance read their sizes from) is aliased to layer 6's own first-pass cache layer.

Inputs are fixed and identical for both runs (rounded to bf16 once, so the bf16-vs-f32 spread is layer 6's own
rounding):
  x        layer 6's input: upstream's own float32 layer-5 output on the 1040-token prompt and one step token
           (global5_reference.safetensors, f32.pos0.{prefill,step}.out), [T, H]
  blocks   layer 6's block states: the layer-0 input from the same file (blocks.pos0.*), [T, 1, H]
  prev.A / prev.B    two deliberately distinct, synthetic previous-pass states (seeded normal rows): the previous
           pass's block after the core's first (layer 12's input) [T, 1, H] and its output (layer 17's) [T, H].
           B = A plus independent noise. A drives the second-pass visit; B only re-runs it, to show the gated result
           follows the messages.
Positions 0..1039 are the prompt (past the 1024 window), position 1040 the step through the cache, and `whole` is
all 1041 rows as one prefill with no cache.

torch.backends.mkldnn (oneDNN) is disabled for both runs, and the metadata records the CPU backend.

Recorded, float32 (indices int32), run in {bf16, f32}, visit in {p0 (pass 1, no messages), p1A, p1B (pass 2)},
phase in {prefill, step, whole}:
  <run>.<visit>.<phase>.out / attn / attn_conv / mlp_input / experts / mlp_conv               [t, H]
  <run>.<visit>.<phase>.router_logits / router_weights / router_indices                      [t, 128] / [t, 8]
  <run>.p1X.<phase>.messages                                                                [t, 4, H] (Q K V MLP)
  <run>.p1X.<phase>.gate_q / gate_k / gate_v / gate_mlp       the four message gates' outputs [t, H]
  <run>.<visit>.<phase>.k / .v / .conv0..conv3     the visit's cache slot (not for whole)  [8, n, 128] / [C, 4]
  x.<phase>, blocks.<phase>, prevA_block.<phase>, prevA_partial.<phase>, prevB_*            inputs

usage: python3 -I capture_inkling2_loop6.py <transformers/src> <model dir> <scratch dir> <global5 reference>
                                            <out.safetensors> [deps dir]
"""
import copy, hashlib, json, os, sys

tf_src, model_dir, scratch, g5_path, out_path = sys.argv[1:6]
sys.path[:0] = [tf_src] + sys.argv[6:7]

import torch
import torch.nn as nn
import transformers
from safetensors import safe_open
from safetensors.torch import save_file

assert os.path.realpath(os.path.dirname(transformers.__file__)).startswith(os.path.realpath(tf_src)), transformers.__file__
import transformers.models.inkling2_loop.modeling_inkling2_loop as M
from transformers.cache_utils import DynamicCache
from transformers.masking_utils import create_recurrent_attention_mask, create_sliding_window_causal_mask

LAYER = 6
# oneDNN's CPU BF16 GEMM can widen the bf16-vs-f32 spread that calibrates the tolerance (found on the output head);
# both runs use torch's native kernels instead
torch.backends.mkldnn.enabled = False

RealLoopSliding = M.Inkling2LoopSlidingLoopDecoderLayer


class Absent(nn.Module):
    """Stands in for a decoder layer this capture never runs; holds no parameters."""

    def __init__(self, config, layer_idx):
        super().__init__()
        self.is_block_boundary = layer_idx % config.n_layers_per_attn_res_block == 0
        self.attention_type = "sliding_attention" if config.layer_types[layer_idx] == "hybrid_sliding" else "full_attention"


def loop_sliding(config, layer_idx):
    return RealLoopSliding(config, layer_idx) if layer_idx == LAYER else Absent(config, layer_idx)


# read by name in Inkling2LoopTextModel.__init__'s layer_classes
M.Inkling2LoopSlidingDecoderLayer = Absent
M.Inkling2LoopFullDecoderLayer = Absent
M.Inkling2LoopFullLoopDecoderLayer = Absent
M.Inkling2LoopSlidingLoopDecoderLayer = loop_sliding

os.makedirs(scratch, exist_ok=True)
cfg = json.load(open(os.path.join(model_dir, "config.json")))
cfg["vision_config"]["num_hidden_layers"] = 1
json.dump(cfg, open(os.path.join(scratch, "config.json"), "w"))

st_path = os.path.join(model_dir, "model-00001-of-00001.safetensors")
keep = {}
with safe_open(st_path, "pt") as f:
    for k in f.keys():
        if (k.startswith("model.llm.embed") or k.startswith(f"model.llm.layers.{LAYER}.")
                # attn_queries.N is layer N + 1's attention input (upstream renames with offset=1)
                or k in (f"model.llm.attn_res.attn_queries.{LAYER - 1}", f"model.llm.attn_res.mlp_queries.{LAYER}",
                         "model.llm.middle_cycle_cross_pass_queries", "model.llm.middle_cycle_cross_pass_kv_queries")):
            keep[k] = f.get_tensor(k)
print(f"layer-{LAYER} source tensors: {len(keep)}", file=sys.stderr)
save_file(keep, os.path.join(scratch, "model.safetensors"))
del keep

# fixed inputs
g5 = {}
with safe_open(g5_path, "pt") as f:
    for k in ("f32.pos0.prefill.out", "f32.pos0.step.out", "blocks.pos0.prefill", "blocks.pos0.step"):
        g5[k] = f.get_tensor(k)
g5_sha = hashlib.sha256(open(g5_path, "rb").read()).hexdigest()


def bf(t):
    return t.to(torch.bfloat16).float()


x_all = bf(torch.cat([g5["f32.pos0.prefill.out"], g5["f32.pos0.step.out"]], 0))  # [T+1, H]
blocks_all = bf(torch.cat([g5["blocks.pos0.prefill"], g5["blocks.pos0.step"]], 0))  # [T+1, 1, H]
T1, H = x_all.shape
T = T1 - 1
gen = torch.Generator().manual_seed(6)
rms = x_all.square().mean(-1, keepdim=True).sqrt()
prevA_block = bf(torch.randn(T1, 1, H, generator=gen) * rms.unsqueeze(-1))
prevA_partial = bf(torch.randn(T1, H, generator=gen) * 2.0 * rms)
prevB_block = bf(prevA_block + 0.5 * torch.randn(T1, 1, H, generator=gen) * rms.unsqueeze(-1))
prevB_partial = bf(prevA_partial + 0.5 * torch.randn(T1, H, generator=gen) * rms)
PHASES = {"prefill": slice(0, T), "step": slice(T, T1), "whole": slice(0, T1)}
PREV = {"A": (prevA_block, prevA_partial), "B": (prevB_block, prevB_partial)}

record = {}
for phase, sl in PHASES.items():
    record[f"x.{phase}"] = x_all[sl].clone()
    record[f"blocks.{phase}"] = blocks_all[sl].clone()
    for v, (pb, pp) in PREV.items():
        record[f"prev{v}_block.{phase}"] = pb[sl].clone()
        record[f"prev{v}_partial.{phase}"] = pp[sl].clone()

for run, dtype in (("bf16", torch.bfloat16), ("f32", torch.float32)):
    model, info = M.Inkling2LoopForConditionalGeneration.from_pretrained(
        scratch, dtype=dtype, attn_implementation="eager", output_loading_info=True
    )
    model.eval()
    lm = model.model.language_model
    layer = lm.layers[LAYER]
    assert type(layer) is RealLoopSliding, type(layer)
    ours = [k for k in info["missing_keys"]
            if f"layers.{LAYER}." in k or "message_input" in k or "embed_tokens" in k]
    assert not ours, ours
    assert not info["unexpected_keys"], info["unexpected_keys"]
    print(f"{run}: loaded; {len(info['missing_keys'])} missing keys, none in layer {LAYER}, message_input or the embedding",
          file=sys.stderr)
    tc = lm.config
    loop_start, loop_end = tc.loop_layer_index_start, tc.loop_layer_index_end
    num_loop = loop_end - loop_start

    taps = {}

    def hook(name, pick=lambda o: o):
        def f(_m, _i, o):
            taps[name] = pick(o)
        return f

    hs = [
        layer.register_forward_hook(hook("out")),
        layer.self_attn.register_forward_hook(hook("attn", lambda o: o[0])),
        layer.post_attention_convolution.register_forward_hook(hook("attn_conv")),
        layer.mlp_input.register_forward_hook(hook("mlp_input", lambda o: o[0])),
        layer.mlp.gate.register_forward_hook(hook("router")),
        layer.mlp.experts.register_forward_hook(hook("experts")),
        layer.post_mlp_convolution.register_forward_hook(hook("mlp_conv")),
        layer.q_message_gate.register_forward_hook(hook("gate_q")),
        layer.k_message_gate.register_forward_hook(hook("gate_k")),
        layer.v_message_gate.register_forward_hook(hook("gate_v")),
        layer.mlp_message_gate.register_forward_hook(hook("gate_mlp")),
    ]

    def new_cache():
        cache = DynamicCache(config=tc)
        assert len(cache.layers) == tc.num_hidden_layers
        cache.layers += [copy.deepcopy(l) for l in cache.layers[loop_start:loop_end] * (tc.loop_num_passes - 1)]
        # only layer 6 runs: the sliding mask and distance read their sizes from the first sliding cache layer
        cache.layers[0] = cache.layers[LAYER]
        return cache

    def visit(x, blocks, cache, pos, messages, pass_idx):
        """One layer-6 visit as Inkling2LoopTextModel.forward makes it, given masks built before any layer ran."""
        offset = 0 if pass_idx == 0 else tc.num_hidden_layers + (pass_idx - 1) * num_loop - loop_start
        return layer(
            x, blocks, messages,
            attention_mask=masks["sliding"], relative_distance=masks["dist"], conv_mask=masks["conv"],
            cache=cache, cache_offset=offset, position_ids=pos,
        )

    def keep(tag, cache_slot, cache, gated):
        names = ["out", "attn", "attn_conv", "mlp_input", "mlp_conv"] + (["gate_q", "gate_k", "gate_v", "gate_mlp"] if gated else [])
        for n in names:
            record[f"{tag}.{n}"] = taps[n][0].float().clone().contiguous()
        record[f"{tag}.experts"] = taps["experts"].float().clone().contiguous()
        logits, weights, idx = taps["router"]
        record[f"{tag}.router_logits"] = logits.float().clone().contiguous()
        record[f"{tag}.router_weights"] = weights.float().clone().contiguous()
        record[f"{tag}.router_indices"] = idx.to(torch.int32).clone().contiguous()
        if cache is not None:
            c = cache.layers[cache_slot]
            assert type(c).__name__ == "LinearAttentionAndSlidingWindowAttentionLayer", type(c)
            record[f"{tag}.k"] = c.keys[0].float().clone().contiguous()
            record[f"{tag}.v"] = c.values[0].float().clone().contiguous()
            for i in range(4):
                record[f"{tag}.conv{i}"] = c.conv_states[i][0].float().clone().contiguous()
        if not gated:
            assert not any(k in taps for k in ("gate_q", "gate_k", "gate_v", "gate_mlp")), "a first-pass visit gated"

    with torch.no_grad():
        for prev in ("A", "B"):
            # one cache per scenario, carried from prefill to step: pass 1's slot 6 and pass 2's slot 24
            cache = new_cache()
            for phase in ("prefill", "step", "whole"):
                sl = PHASES[phase]
                c = None if phase == "whole" else cache
                x = x_all[sl].unsqueeze(0).to(dtype)
                blocks = blocks_all[sl].unsqueeze(0).to(dtype)
                start = sl.start if phase != "whole" else 0
                pos = (torch.arange(x.shape[1]) + start).unsqueeze(0)
                kw = {"config": tc, "inputs_embeds": x, "attention_mask": None, "cache": c, "position_ids": pos}
                masks = {
                    "sliding": create_sliding_window_causal_mask(**kw),
                    "conv": create_recurrent_attention_mask(**kw),
                    "dist": M.get_relative_distance(x, c, is_sliding=True),
                }
                if c is not None:
                    assert c.get_seq_length() == start, (phase, c.get_seq_length(), start)
                # pass 1: no messages
                taps.clear()
                visit(x, blocks, c, pos, None, 0)
                if prev == "A":
                    keep(f"{run}.p0.{phase}", LAYER, c, gated=False)
                else:
                    # the first pass does not depend on which previous pass the later one reads
                    assert torch.equal(taps["out"][0].float(), record[f"{run}.p0.{phase}.out"]), "pass 1 moved"
                # pass 2: messages from the previous pass, loop_layer_idx 0
                pb, pp = PREV[prev]
                messages = lm.message_input(pb[sl].unsqueeze(0).to(dtype), pp[sl].unsqueeze(0).to(dtype), 0)
                assert messages.shape == (1, x.shape[1], 4, H), messages.shape
                taps.clear()
                visit(x, blocks, c, pos, messages, 1)
                tag = f"{run}.p1{prev}.{phase}"
                keep(tag, tc.num_hidden_layers + (LAYER - loop_start), c, gated=True)
                record[f"{tag}.messages"] = messages[0].float().clone().contiguous()
    for h in hs:
        h.remove()
    del model

meta = {"transformers": "046bdb8cc2cee29d233964bb77fe6fd37cfe4aa3", "torch": torch.__version__, "device": "cpu",
        "attn_implementation": "eager", "layer": str(LAYER),
        "cpu_backend": f"torch.backends.mkldnn.enabled={torch.backends.mkldnn.enabled}, "
                       f"mkldnn available={torch.backends.mkldnn.is_available()}, threads={torch.get_num_threads()}",
        "inputs": f"x and blocks from global5_reference.safetensors sha256 {g5_sha} (f32.pos0.*.out, blocks.pos0.*), "
                  "bf16-rounded; prev A/B synthetic, torch.Generator seed 6",
        "cache_slots": "p0 = slot 6, p1 = slot 24 (num_hidden_layers + (pass - 1) * 12 + layer - 6)",
        "conv_states": "0=k_proj.sconv 1=v_proj.sconv 2=post_attention_convolution 3=post_mlp_convolution"}
save_file(record, out_path, metadata=meta)
for k in sorted(record):
    print(k, tuple(record[k].shape), file=sys.stderr)
