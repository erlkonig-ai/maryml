"""Run transformers' own Inkling 2 Mini from the embedding through layer 5 and record every layer.

Upstream's code end to end, as for layer 0 (capture_inkling2_layer0.py) and layer 5 (capture_inkling2_global5.py):
the pinned transformers checkout (046bdb8cc2cee29d233964bb77fe6fd37cfe4aa3) first on sys.path, the model built by
`from_pretrained` from only the tensors of layers 0..5 (plus the embedding and those layers' attention-residual
queries), upstream's DynamicCache, eager attention, CPU.

Unlike the layer-5 capture, NOTHING is fed: both runs, float32 and bfloat16, compute every layer from the tokens, so
their distance at each layer is how far rounding alone has moved the chain by then. That calibrates a chained
comparison (Mary's layers 0..4 producing layer 5's input) at every depth.

`prepared` is a directory holding the 6-layer config.json and a model.safetensors of exactly those tensors (as
capture_inkling2_global5.py writes it). Every tensor in it is checked byte-equal to the release first.

The prompt is the 26 tokens of capture_inkling2_layer0.py; positions start at 0.

Recorded, float32 (indices int32), <run> in bf16 / f32, <phase> in prefill / step / whole:
  <run>.<phase>.layer<l>.input / .out                    layer l's input and output hidden    [t, H]
  <run>.<phase>.blocks                                   layer 5's block states               [t, 1, H]
  <run>.<phase>.q_in / k_in / v_in                       layer 5's attention_input outputs    [t, H]
  <run>.<phase>.attn / attn_conv / mlp_input / experts / mlp_conv                              [t, H]
  <run>.<phase>.router_logits / router_weights / router_indices    layer 5                   [t, 128] / [t, 8]
  <run>.<phase>.layer<l>.k / .v                          layer l's cached keys / values       [kv, n, d]
  <run>.<phase>.layer<l>.conv0..conv3                    layer l's conv states                [C, k]
phase `prefill` (the prompt), `step` (one more token through the cache), `whole` (both as one prefill, no cache,
no states recorded).

usage: python3 -I capture_inkling2_chain5.py <transformers/src> <release dir> <prepared dir> <out.safetensors> [deps]
"""
import json, os, sys

tf_src, model_dir, prepared, out_path = sys.argv[1:5]
sys.path[:0] = [tf_src] + sys.argv[5:6]

import torch
import transformers
from safetensors import safe_open
from safetensors.torch import save_file

assert os.path.realpath(os.path.dirname(transformers.__file__)).startswith(os.path.realpath(tf_src)), transformers.__file__
from transformers.models.inkling2_loop.modeling_inkling2_loop import Inkling2LoopForConditionalGeneration

PROMPT = [371, 157021, 161319, 220, 32329, 160581, 82, 150053, 149876, 161703, 153350, 13, 149954,
          152743, 11, 149876, 150131, 149890, 153939, 149876, 150908, 149879, 149878, 150299, 149884, 151015]
T = len(PROMPT)
STEP = 151083
LAYERS = 6
FULL = 5
torch.manual_seed(0)

cfg = json.load(open(os.path.join(prepared, "config.json")))["text_config"]
assert cfg["num_hidden_layers"] == LAYERS and cfg["local_layer_ids"] == list(range(FULL)), cfg

# the prepared tensors are the release's own bytes, and exactly layers 0..5's
wanted_extra = {f"model.llm.attn_res.mlp_queries.{i}" for i in range(LAYERS)} | {
    f"model.llm.attn_res.attn_queries.{i}" for i in range(FULL)}
with safe_open(os.path.join(model_dir, "model-00001-of-00001.safetensors"), "pt") as rel, \
        safe_open(os.path.join(prepared, "model.safetensors"), "pt") as prep:
    want = {k for k in rel.keys() if k.startswith("model.llm.embed") or k in wanted_extra
            or any(k.startswith(f"model.llm.layers.{i}.") for i in range(LAYERS))}
    assert set(prep.keys()) == want, sorted(set(prep.keys()) ^ want)[:8]
    for k in sorted(want):
        a, b = rel.get_tensor(k), prep.get_tensor(k)
        assert a.dtype == b.dtype and a.shape == b.shape and torch.equal(a.view(torch.uint8), b.view(torch.uint8)), k
print(f"prepared: {len(want)} tensors, every one byte-equal to the release", file=sys.stderr)

record = {}
L5_KEYS = ("q_in", "k_in", "v_in", "attn", "attn_conv", "mlp_input", "mlp_conv")


def run(dtype, tag):
    model, info = Inkling2LoopForConditionalGeneration.from_pretrained(
        prepared, dtype=dtype, attn_implementation="eager", output_loading_info=True
    )
    model.eval()
    lm = model.model.language_model
    assert type(lm.layers[FULL]).__name__ == "Inkling2LoopFullDecoderLayer", type(lm.layers[FULL])
    for i in range(FULL):
        assert type(lm.layers[i]).__name__ == "Inkling2LoopSlidingDecoderLayer", (i, type(lm.layers[i]))
    missing = [k for k in info["missing_keys"] if k.startswith("model.language_model.layers.")
               or "embed_tokens" in k]
    assert not missing, missing
    assert not info["unexpected_keys"], info["unexpected_keys"]
    print(f"{tag}: loaded; {len(info['missing_keys'])} missing keys, none in layers 0..{FULL}", file=sys.stderr)

    taps = {}

    def hook(name, pick=lambda o: o):
        def f(_m, _i, o):
            taps[name] = pick(o)
        return f

    def pre(i):
        def f(_m, args, kwargs):
            taps[f"layer{i}.input"] = args[0]
            if i == FULL:
                taps["blocks"] = args[1]
        return f

    def qkv(_m, _i, o):
        taps["q_in"], taps["k_in"], taps["v_in"] = o

    layer = lm.layers[FULL]
    hs = [
        layer.attention_input.register_forward_hook(qkv),
        layer.self_attn.register_forward_hook(hook("attn", lambda o: o[0])),
        layer.post_attention_convolution.register_forward_hook(hook("attn_conv")),
        layer.mlp_input.register_forward_hook(hook("mlp_input", lambda o: o[0])),
        layer.mlp.gate.register_forward_hook(hook("router")),
        layer.mlp.experts.register_forward_hook(hook("experts")),
        layer.post_mlp_convolution.register_forward_hook(hook("mlp_conv")),
    ]
    for i in range(LAYERS):
        hs.append(lm.layers[i].register_forward_pre_hook(pre(i), with_kwargs=True))
        hs.append(lm.layers[i].register_forward_hook(hook(f"layer{i}.out")))

    def keep_phase(phase, cache):
        p = f"{tag}.{phase}"
        for i in range(LAYERS):
            for n in ("input", "out"):
                record[f"{p}.layer{i}.{n}"] = taps[f"layer{i}.{n}"][0].float().clone().contiguous()
        record[f"{p}.blocks"] = taps["blocks"][0].float().clone().contiguous()
        for name in L5_KEYS:
            record[f"{p}.{name}"] = taps[name][0].float().clone().contiguous()
        record[f"{p}.experts"] = taps["experts"].float().clone().contiguous()
        logits, weights, idx = taps["router"]
        record[f"{p}.router_logits"] = logits.float().clone().contiguous()
        record[f"{p}.router_weights"] = weights.float().clone().contiguous()
        record[f"{p}.router_indices"] = idx.to(torch.int32).clone().contiguous()
        if cache is not None:
            for i in range(LAYERS):
                c = cache.layers[i]
                record[f"{p}.layer{i}.k"] = c.keys[0].float().clone().contiguous()
                record[f"{p}.layer{i}.v"] = c.values[0].float().clone().contiguous()
                for j in range(4):
                    record[f"{p}.layer{i}.conv{j}"] = c.conv_states[j][0].float().clone().contiguous()

    with torch.no_grad():
        pos = lambda a, n: (torch.arange(n) + a)[None]
        out = model(input_ids=torch.tensor([PROMPT]), position_ids=pos(0, T), use_cache=True)
        keep_phase("prefill", out.past_key_values)
        out = model(input_ids=torch.tensor([[STEP]]), position_ids=pos(T, 1), cache=out.past_key_values,
                    use_cache=True)
        keep_phase("step", out.past_key_values)
        model(input_ids=torch.tensor([PROMPT + [STEP]]), position_ids=pos(0, T + 1), use_cache=False)
        keep_phase("whole", None)
    for h in hs:
        h.remove()
    del model


run(torch.float32, "f32")
run(torch.bfloat16, "bf16")
for run_ in ("f32", "bf16"):
    a, b = record[f"{run_}.step.layer{FULL}.out"][-1], record[f"{run_}.whole.layer{FULL}.out"][-1]
    print(f"{run_}: step vs whole last row, layer {FULL} out: max |d| {(a - b).abs().max().item():.3e}", file=sys.stderr)

record["prompt"] = torch.tensor(PROMPT, dtype=torch.int32)
record["step_token"] = torch.tensor([STEP], dtype=torch.int32)
meta = {"transformers": "046bdb8cc2cee29d233964bb77fe6fd37cfe4aa3", "torch": torch.__version__,
        "device": "cpu", "attn_implementation": "eager", "layers": "0..5",
        "inputs": "tokens only; nothing fed, both runs chain from the embedding",
        "conv_states": "0=k_proj.sconv 1=v_proj.sconv 2=post_attention_convolution 3=post_mlp_convolution"}
save_file(record, out_path, metadata=meta)
for k in sorted(record):
    print(k, tuple(record[k].shape), file=sys.stderr)
