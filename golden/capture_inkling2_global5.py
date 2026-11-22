"""Run transformers' own Inkling 2 Mini decoder layer 5 (the first full-attention layer) and record it.

Upstream's code end to end, as for layer 0 (capture_inkling2_layer0.py): the pinned transformers checkout
(046bdb8cc2cee29d233964bb77fe6fd37cfe4aa3) first on sys.path, the model built by `from_pretrained` from only the
tensors of layers 0..5 (plus the embedding and those layers' attention-residual queries), upstream's DynamicCache,
eager attention, CPU.

Layer 5's inputs are FIXED, so that the comparison measures layer 5 alone: the float32 run's real layer-5 input
(its hidden state and block states, from layers 0..4 on the prompt) is recorded as `input.*` / `blocks.*`, and the
bfloat16 run is fed exactly those (cast to bf16) by a forward pre-hook. Their distance is then layer 5's own
rounding, which calibrates the tolerance.

The prompt is 1040 tokens (the 26-token prompt of capture_inkling2_layer0.py, repeated), so the last rows reach back
past `rel_extent` (1024), where the relative bias is zero. Log scaling is exactly 1 below position 8191, so every
phase is run twice: `pos0` (positions from 0) and `far` (position_ids offset by ORIGIN; upstream takes them apart
from the cache, so masks and distances are unchanged and only the log scale moves).

Recorded, float32 (indices int32):
  input.<scen>.<phase> / blocks.<scen>.<phase>           layer 5's input [t, H] / block states [t, 1, H]
  <run>.<scen>.<phase>.q_in / k_in / v_in                attention_input's three outputs     [t, H]
  <run>.<scen>.<phase>.attn / attn_conv / mlp_input / experts / mlp_conv / out                [t, H]
  <run>.<scen>.<phase>.router_logits / router_weights / router_indices                       [t, 128] / [t, 8]
  <run>.<scen>.<phase>.k / .v                            layer 5's cached keys / values       [8, n, 128]
  <run>.<scen>.<phase>.conv0..conv3                      layer 5's conv states                [C, 4]
phase `prefill` (the prompt), `step` (one more token through the cache), `whole` (both as one prefill, no cache,
no states recorded).

usage: python3 -I capture_inkling2_global5.py <transformers/src> <model dir> <scratch dir> <out.safetensors> [deps dir]
"""
import json, os, sys

tf_src, model_dir, scratch, out_path = sys.argv[1:5]
sys.path[:0] = [tf_src] + sys.argv[5:6]

import torch
import transformers
from safetensors import safe_open
from safetensors.torch import save_file

assert os.path.realpath(os.path.dirname(transformers.__file__)).startswith(os.path.realpath(tf_src)), transformers.__file__
from transformers.models.inkling2_loop.modeling_inkling2_loop import Inkling2LoopForConditionalGeneration

BASE = [371, 157021, 161319, 220, 32329, 160581, 82, 150053, 149876, 161703, 153350, 13, 149954,
        152743, 11, 149876, 150131, 149890, 153939, 149876, 150908, 149879, 149878, 150299, 149884, 151015]
T = 1040
PROMPT = (BASE * (T // len(BASE) + 1))[:T]
STEP = 151083
ORIGIN = 100000
LAYER = 5
torch.manual_seed(0)

os.makedirs(scratch, exist_ok=True)
cfg = json.load(open(os.path.join(model_dir, "config.json")))
cfg["text_config"]["num_hidden_layers"] = LAYER + 1
cfg["text_config"]["local_layer_ids"] = list(range(LAYER))
cfg["vision_config"]["num_hidden_layers"] = 1
json.dump(cfg, open(os.path.join(scratch, "config.json"), "w"))
for f in ("generation_config.json",):
    src = os.path.join(model_dir, f)
    if os.path.exists(src):
        open(os.path.join(scratch, f), "w").write(open(src).read())

st_path = os.path.join(model_dir, "model-00001-of-00001.safetensors")
wanted_extra = {f"model.llm.attn_res.mlp_queries.{i}" for i in range(LAYER + 1)} | {
    f"model.llm.attn_res.attn_queries.{i}" for i in range(LAYER)}
keep = {}
with safe_open(st_path, "pt") as f:
    for k in f.keys():
        if k.startswith("model.llm.embed") or k in wanted_extra or any(
                k.startswith(f"model.llm.layers.{i}.") for i in range(LAYER + 1)):
            keep[k] = f.get_tensor(k)
print(f"source tensors: {len(keep)}", file=sys.stderr)
save_file(keep, os.path.join(scratch, "model.safetensors"))
del keep

record = {}
STATE_KEYS = ("q_in", "k_in", "v_in", "attn", "attn_conv", "mlp_input", "mlp_conv")


def run(dtype, tag):
    model, info = Inkling2LoopForConditionalGeneration.from_pretrained(
        scratch, dtype=dtype, attn_implementation="eager", output_loading_info=True
    )
    model.eval()
    lm = model.model.language_model
    layer = lm.layers[LAYER]
    assert type(layer).__name__ == "Inkling2LoopFullDecoderLayer", type(layer)
    missing = [k for k in info["missing_keys"] if k.startswith("model.language_model.layers.")
               or "embed_tokens" in k]
    assert not missing, missing
    assert not info["unexpected_keys"], info["unexpected_keys"]
    print(f"{tag}: loaded; {len(info['missing_keys'])} missing keys, none in layers 0..{LAYER}", file=sys.stderr)

    taps = {}
    feed = {}  # (hidden, blocks) to substitute for layer 5's input, or None to record it

    def hook(name, pick=lambda o: o):
        def f(_m, _i, o):
            taps[name] = pick(o)
        return f

    def pre(_m, args, kwargs):
        hs, bs = args[0], args[1]
        taps["input"], taps["blocks"] = hs, bs
        if feed.get("x") is not None:
            x, b = feed["x"]
            assert x.shape == hs.shape and b.shape == bs.shape, (x.shape, hs.shape, b.shape, bs.shape)
            return (x.to(hs.dtype), b.to(bs.dtype)) + tuple(args[2:]), kwargs
        return None

    def qkv(_m, _i, o):
        taps["q_in"], taps["k_in"], taps["v_in"] = o

    hs = [
        layer.register_forward_pre_hook(pre, with_kwargs=True),
        layer.register_forward_hook(hook("out")),
        layer.attention_input.register_forward_hook(qkv),
        layer.self_attn.register_forward_hook(hook("attn", lambda o: o[0])),
        layer.post_attention_convolution.register_forward_hook(hook("attn_conv")),
        layer.mlp_input.register_forward_hook(hook("mlp_input", lambda o: o[0])),
        layer.mlp.gate.register_forward_hook(hook("router")),
        layer.mlp.experts.register_forward_hook(hook("experts")),
        layer.post_mlp_convolution.register_forward_hook(hook("mlp_conv")),
    ]

    def keep_phase(scen, phase, cache):
        p = f"{tag}.{scen}.{phase}"
        for name in STATE_KEYS + ("out",):
            record[f"{p}.{name}"] = taps[name][0].float().clone().contiguous()
        record[f"{p}.experts"] = taps["experts"].float().clone().contiguous()
        logits, weights, idx = taps["router"]
        record[f"{p}.router_logits"] = logits.float().clone().contiguous()
        record[f"{p}.router_weights"] = weights.float().clone().contiguous()
        record[f"{p}.router_indices"] = idx.to(torch.int32).clone().contiguous()
        if cache is not None:
            c = cache.layers[LAYER]
            print(f"{p}: cache layer {type(c).__name__}", file=sys.stderr)
            record[f"{p}.k"] = c.keys[0].float().clone().contiguous()
            record[f"{p}.v"] = c.values[0].float().clone().contiguous()
            for i in range(4):
                record[f"{p}.conv{i}"] = c.conv_states[i][0].float().clone().contiguous()

    def phase_input(scen, phase):
        key = f"input.{scen}.{phase}"
        if tag == "f32":
            feed["x"] = None
            return lambda: (record.__setitem__(key, taps["input"][0].float().clone().contiguous()),
                            record.__setitem__(f"blocks.{scen}.{phase}", taps["blocks"][0].float().clone().contiguous()))
        feed["x"] = (record[key][None], record[f"blocks.{scen}.{phase}"][None])
        return lambda: None

    with torch.no_grad():
        for scen, off in (("pos0", 0), ("far", ORIGIN)):
            pos = lambda a, n: (torch.arange(n) + a + off)[None]
            done = phase_input(scen, "prefill")
            out = model(input_ids=torch.tensor([PROMPT]), position_ids=pos(0, T), use_cache=True)
            done()
            keep_phase(scen, "prefill", out.past_key_values)
            done = phase_input(scen, "step")
            out = model(input_ids=torch.tensor([[STEP]]), position_ids=pos(T, 1), cache=out.past_key_values,
                        use_cache=True)
            done()
            keep_phase(scen, "step", out.past_key_values)
            done = phase_input(scen, "whole")
            model(input_ids=torch.tensor([PROMPT + [STEP]]), position_ids=pos(0, T + 1), use_cache=False)
            done()
            keep_phase(scen, "whole", None)
    for h in hs:
        h.remove()
    del model


run(torch.float32, "f32")
# layers 0..4 do not read absolute positions: the far run's layer-5 input is the pos0 run's
for ph in ("prefill", "step", "whole"):
    assert torch.equal(record[f"input.pos0.{ph}"], record[f"input.far.{ph}"]), ph
    assert torch.equal(record[f"blocks.pos0.{ph}"], record[f"blocks.far.{ph}"]), ph
run(torch.bfloat16, "bf16")

record["prompt"] = torch.tensor(PROMPT, dtype=torch.int32)
record["step_token"] = torch.tensor([STEP], dtype=torch.int32)
record["origin"] = torch.tensor([ORIGIN], dtype=torch.int32)
meta = {"transformers": "046bdb8cc2cee29d233964bb77fe6fd37cfe4aa3", "torch": torch.__version__,
        "device": "cpu", "attn_implementation": "eager", "layer": str(LAYER),
        "inputs": "layer 5 input from the f32 run, fed to both runs",
        "conv_states": "0=k_proj.sconv 1=v_proj.sconv 2=post_attention_convolution 3=post_mlp_convolution"}
save_file(record, out_path, metadata=meta)
for k in sorted(record):
    print(k, tuple(record[k].shape), file=sys.stderr)
