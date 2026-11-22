"""Run transformers' own Inkling 2 Mini decoder layer 0 on a real prompt and record it.

Upstream's code end to end: the pinned transformers checkout
(046bdb8cc2cee29d233964bb77fe6fd37cfe4aa3) is put first on sys.path, the model
is built by `from_pretrained` (so its key renames, the NVFP4 dequantization and
the w13 de-interleave are upstream's), and layer 0 runs inside upstream's text
model with upstream's DynamicCache. Nothing here re-types the arithmetic.

Only the tensors layer 0 needs are read from the release: `model.llm.embed*`,
`model.llm.layers.0.*` and `model.llm.attn_res.mlp_queries.0` (layer 0's MLP-input query).
They are written to a scratch directory (a tmpfs in the container) beside a config with one decoder layer and a one-layer vision
tower, so the load is small. Every other parameter is left at its init value and
is never on the path to what is recorded; the loader's own missing-key report is
checked so that none of layer 0's parameters is among them.

Recorded, as float32 (indices as int32), for dtype bfloat16 (the shipped
precision) and float32 (the same weights with no bf16 rounding, which is what
calibrates a tolerance):
  <run>.input                    layer 0 input, the normed embeddings, [T, H]
  <run>.<phase>.out              layer 0 output                         [t, H]
  <run>.<phase>.attn             self_attn output (before attn sconv)  [t, H]
  <run>.<phase>.attn_conv        post_attention_convolution output     [t, H]
  <run>.<phase>.mlp_input        mlp_input (block attention residual)  [t, H]
  <run>.<phase>.router_logits / router_weights / router_indices        [t, 128] / [t, 8]
  <run>.<phase>.experts          experts output (before mlp sconv)     [t, H]
  <run>.<phase>.mlp_conv         post_mlp_convolution output           [t, H]
  <run>.<phase>.k / .v           layer 0's cached keys / values        [8, n, 128]
  <run>.<phase>.conv0..conv3     layer 0's conv states (k, v, attn, mlp) [C, 4]
with phase `prefill` (the prompt) and `step` (one more token, through the cache),
and `<run>.whole.*`: prompt and step token as one prefill with no cache.

usage (inside a container with torch and safetensors):
  python3 -I capture_inkling2_layer0.py <transformers/src> <model dir> <scratch dir> <out.safetensors> [deps dir]
"""
import json, os, sys

tf_src, model_dir, scratch, out_path = sys.argv[1:5]
# optional: a directory holding the dependency versions the pinned commit requires
sys.path[:0] = [tf_src] + sys.argv[5:6]

import torch
import transformers
from safetensors import safe_open
from safetensors.torch import save_file

assert os.path.realpath(os.path.dirname(transformers.__file__)).startswith(os.path.realpath(tf_src)), transformers.__file__
from transformers.models.inkling2_loop.modeling_inkling2_loop import Inkling2LoopForConditionalGeneration

PROMPT = [371, 157021, 161319, 220, 32329, 160581, 82, 150053, 149876, 161703, 153350, 13, 149954,
          152743, 11, 149876, 150131, 149890, 153939, 149876, 150908, 149879, 149878, 150299, 149884, 151015]
STEP = 151083  # the release's own greedy continuation of PROMPT (" len"), parity-greedy.json raw_text

os.makedirs(scratch, exist_ok=True)
cfg = json.load(open(os.path.join(model_dir, "config.json")))
cfg["text_config"]["num_hidden_layers"] = 1
cfg["text_config"]["local_layer_ids"] = [0]
cfg["vision_config"]["num_hidden_layers"] = 1
json.dump(cfg, open(os.path.join(scratch, "config.json"), "w"))
for f in ("generation_config.json",):
    src = os.path.join(model_dir, f)
    if os.path.exists(src):
        open(os.path.join(scratch, f), "w").write(open(src).read())

st_path = os.path.join(model_dir, "model-00001-of-00001.safetensors")
keep = {}
with safe_open(st_path, "pt") as f:
    for k in f.keys():
        # layer 0's attention-residual query is stored as attn_res.mlp_queries.0, outside layers.0
        if k.startswith("model.llm.embed") or k.startswith("model.llm.layers.0.") or k in (
            "model.llm.attn_res.mlp_queries.0",
        ):
            keep[k] = f.get_tensor(k)
print(f"layer-0 source tensors: {len(keep)}", file=sys.stderr)
save_file(keep, os.path.join(scratch, "model.safetensors"))
del keep

record = {}
for run, dtype in (("bf16", torch.bfloat16), ("f32", torch.float32)):
    model, info = Inkling2LoopForConditionalGeneration.from_pretrained(
        scratch, dtype=dtype, attn_implementation="eager", output_loading_info=True
    )
    model.eval()
    lm = model.model.language_model
    layer = lm.layers[0]
    l0_missing = [k for k in info["missing_keys"]
                  if k.startswith("model.language_model.layers.0.") or "embed_tokens" in k]
    assert not l0_missing, l0_missing
    l0_unexpected = [k for k in info["unexpected_keys"]]
    assert not l0_unexpected, l0_unexpected
    print(f"{run}: loaded; {len(info['missing_keys'])} missing keys, none in layer 0 or the embedding",
          file=sys.stderr)

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
        layer.mlp.gate.register_forward_hook(hook("router", lambda o: o)),
        layer.mlp.experts.register_forward_hook(hook("experts")),
        layer.post_mlp_convolution.register_forward_hook(hook("mlp_conv")),
        lm.embed_tokens.register_forward_hook(hook("input")),
    ]

    def keep_phase(phase, cache):
        for name in ("out", "attn", "attn_conv", "mlp_input", "mlp_conv"):
            record[f"{run}.{phase}.{name}"] = taps[name][0].float().contiguous()
        # the experts see the flattened [batch * tokens, H] rows, batch 1
        record[f"{run}.{phase}.experts"] = taps["experts"].float().contiguous()
        logits, weights, idx = taps["router"]
        record[f"{run}.{phase}.router_logits"] = logits.float().contiguous()
        record[f"{run}.{phase}.router_weights"] = weights.float().contiguous()
        record[f"{run}.{phase}.router_indices"] = idx.to(torch.int32).contiguous()
        c = cache.layers[0]
        record[f"{run}.{phase}.k"] = c.keys[0].float().contiguous()
        record[f"{run}.{phase}.v"] = c.values[0].float().contiguous()
        assert type(c).__name__ == "LinearAttentionAndSlidingWindowAttentionLayer", type(c)
        for i in range(4):
            record[f"{run}.{phase}.conv{i}"] = c.conv_states[i][0].float().contiguous()

    with torch.no_grad():
        ids = torch.tensor([PROMPT])
        out = model(input_ids=ids, use_cache=True)
        cache = out.past_key_values
        record[f"{run}.input"] = taps["input"][0].float().contiguous()
        keep_phase("prefill", cache)
        # the cache's tensors are updated in place by the next call: clone what was kept
        for k in [k for k in record if k.startswith(f"{run}.prefill.")]:
            record[k] = record[k].clone()
        out = model(input_ids=torch.tensor([[STEP]]), cache=cache, use_cache=True)
        keep_phase("step", out.past_key_values)
        record[f"{run}.step_input"] = taps["input"][0].float().contiguous()
        # the same 27 tokens as one prefill with no cache: its last row is what the
        # step through the cache should have computed
        model(input_ids=torch.tensor([PROMPT + [STEP]]), use_cache=False)
        for name in ("out", "attn", "attn_conv", "mlp_input", "mlp_conv"):
            record[f"{run}.whole.{name}"] = taps[name][0].float().contiguous()
    for h in hs:
        h.remove()
    del model

record["prompt"] = torch.tensor(PROMPT, dtype=torch.int32)
record["step_token"] = torch.tensor([STEP], dtype=torch.int32)
meta = {"transformers": "046bdb8cc2cee29d233964bb77fe6fd37cfe4aa3", "torch": torch.__version__,
        "device": "cpu", "attn_implementation": "eager",
        "conv_states": "0=k_proj.sconv 1=v_proj.sconv 2=post_attention_convolution 3=post_mlp_convolution"}
save_file(record, out_path, metadata=meta)
for k in sorted(record):
    print(k, tuple(record[k].shape), file=sys.stderr)
