"""Run transformers' own Inkling 2 Mini cross-pass message classes on fixed tensors and record them.

Upstream's code, not a re-typing: the pinned transformers checkout
(046bdb8cc2cee29d233964bb77fe6fd37cfe4aa3) is put first on sys.path and the three
pieces under test are its own classes:

  Inkling2LoopMergeMessageQueries      core_model_loading.py: the release's two
                                       [2 * core, H] query tensors -> [core, 4, H]
  Inkling2LoopMessageAttentionResidual modeling_inkling2_loop.py: the four messages
                                       (Q, K, V, MLP) one core layer reads from the
                                       previous pass
  Inkling2LoopMessageGate              modeling_inkling2_loop.py:
                                       (1 - tanh(forget)) * h + tanh(gain) * message

The config is upstream's Inkling2LoopTextConfig with the release's loop geometry
(core 6..18, so 12 core layers) and rms_norm_eps, and a small hidden size and gate
rank so the fixture stays small. attn_implementation is "eager", the only one the
model supports.

Every weight value is bf16-representable, as in the release (controllers.* are BF16
there, the cross-pass queries F32). Two runs:
  bf16  as the release loads: down_proj stays BF16 (it is not in
        _keep_in_fp32_modules_strict), so the gate's normed input is rounded to bf16
        and its rank-r product is a bf16 GEMM; forget/gain and the queries are f32.
  f32   the same values with down_proj in f32, which isolates the bf16 rounding.

Inputs are nonuniform (per-row scales over three decades, offsets), T = 5 tokens,
batch 1. The message read is recorded for 0, 1 and 2 complete previous-pass block
states (the release has exactly 1: the state appended at layer 12) plus the partial
block state (the previous pass's layer-17 output), at three core layer indices.

Recorded: every tensor as its float32 bit pattern (u32), row-major, with its shape.
Taps of the gate's three projections (down output, forget and gain pre-tanh) come
from forward hooks on upstream's own nn.Linear modules.

usage (inside the offline container, from a directory that is not this one):
  python3 -I capture_inkling2_messages.py <transformers/src> <out.json> [deps dir]
"""
import hashlib, json, sys

tf_src, out_path = sys.argv[1:3]
sys.path[:0] = [tf_src] + sys.argv[3:4]

import os
import torch
import transformers

assert os.path.realpath(os.path.dirname(transformers.__file__)).startswith(os.path.realpath(tf_src)), transformers.__file__
from transformers.core_model_loading import Inkling2LoopMergeMessageQueries
from transformers.models.inkling2_loop.configuration_inkling2_loop import Inkling2LoopTextConfig
from transformers.models.inkling2_loop.modeling_inkling2_loop import (
    Inkling2LoopMessageAttentionResidual,
    Inkling2LoopMessageGate,
)

torch.manual_seed(0)
H, RANK, T = 32, 6, 5
cfg = Inkling2LoopTextConfig(hidden_size=H, loop_gate_rank=RANK)
cfg._attn_implementation = "eager"
CORE = cfg.loop_layer_index_end - cfg.loop_layer_index_start
assert (cfg.loop_layer_index_start, cfg.loop_layer_index_end, cfg.rms_norm_eps) == (6, 18, 1e-6), cfg

g = torch.Generator().manual_seed(20261122)


def bf16_values(*shape, scale=1.0):
    return (torch.randn(*shape, generator=g) * scale).to(torch.bfloat16).float()


def nonuniform(*shape):
    """randn rows with per-row scales from 1e-2 to 1e1 and a per-row offset."""
    x = torch.randn(*shape, generator=g)
    rows = x.reshape(-1, shape[-1])
    scales = torch.logspace(-2, 1, rows.shape[0])[torch.randperm(rows.shape[0], generator=g)]
    offsets = torch.randn(rows.shape[0], 1, generator=g) * 0.5
    return (rows * scales[:, None] + offsets * scales[:, None]).reshape(*shape).contiguous()


def bits(t):
    return [v & 0xFFFFFFFF for v in t.detach().float().contiguous().view(torch.int32).flatten().tolist()]


def rec(t):
    return {"shape": list(t.shape), "bits": bits(t)}


record = {}

# The release's layout: queries = [Q, MLP] per core layer, kv_queries = [K, V] per core layer.
src_queries = nonuniform(2 * CORE, H)
src_kv_queries = nonuniform(2 * CORE, H)
merged = Inkling2LoopMergeMessageQueries().convert(
    {"q": src_queries, "kv": src_kv_queries}, ["q", "kv"], ["message_input.query"]
)["message_input.query"]
assert tuple(merged.shape) == (CORE, 4, H)
record["queries"] = rec(src_queries)
record["kv_queries"] = rec(src_kv_queries)
record["merged"] = rec(merged)

attn = Inkling2LoopMessageAttentionResidual(cfg)
with torch.no_grad():
    attn.query.copy_(merged)

messages = {}
partial = nonuniform(1, T, H)
record["partial"] = rec(partial[0])
for n in (0, 1, 2):
    blocks = nonuniform(1, T, n, H) if n else torch.zeros(1, T, 0, H)
    record[f"blocks{n}"] = rec(blocks[0])
    for li in (0, 7, CORE - 1):
        with torch.no_grad():
            m = attn(blocks, partial, li)
        assert tuple(m.shape) == (1, T, 4, H) and m.dtype == torch.float32, (m.shape, m.dtype)
        record[f"messages.n{n}.l{li}"] = rec(m[0])
        messages[(n, li)] = m

STREAMS = ("q", "k", "v", "mlp")
gates = {}
for s in STREAMS:
    gate = Inkling2LoopMessageGate(cfg)
    with torch.no_grad():
        gate.down_proj.weight.copy_(bf16_values(RANK, 2 * H, scale=(2 * H) ** -0.5 * 3))
        gate.forget_gate_proj.weight.copy_(bf16_values(H, RANK, scale=0.6))
        gate.message_gate_proj.weight.copy_(bf16_values(H, RANK, scale=0.6))
    gates[s] = gate
    record[f"gate.{s}.down"] = rec(gate.down_proj.weight)
    record[f"gate.{s}.forget"] = rec(gate.forget_gate_proj.weight)
    record[f"gate.{s}.gain"] = rec(gate.message_gate_proj.weight)

# The release's geometry: one complete previous-pass block. Gate inputs are the
# four streams' own hidden states (global layers give Q, K, V separate inputs).
msg = messages[(1, 7)][0]
hidden = {s: nonuniform(T, H) for s in STREAMS}
for s in STREAMS:
    record[f"hidden.{s}"] = rec(hidden[s])

for run, down_dtype in (("bf16", torch.bfloat16), ("f32", torch.float32)):
    for i, s in enumerate(STREAMS):
        gate = gates[s]
        gate.down_proj.to(down_dtype)
        assert gate.forget_gate_proj.weight.dtype == torch.float32
        taps = {}
        hooks = [
            getattr(gate, name).register_forward_hook(lambda _m, _i, o, name=name: taps.__setitem__(name, o))
            for name in ("down_proj", "forget_gate_proj", "message_gate_proj")
        ]
        h = hidden[s]
        with torch.no_grad():
            out = gate(h[None], msg[None, :, i, :])
        for hk in hooks:
            hk.remove()
        assert taps["down_proj"].dtype == down_dtype
        record[f"{run}.{s}.down_out"] = rec(taps["down_proj"][0])
        record[f"{run}.{s}.forget_pre"] = rec(taps["forget_gate_proj"][0])
        record[f"{run}.{s}.gain_pre"] = rec(taps["message_gate_proj"][0])
        record[f"{run}.{s}.out"] = rec(out[0])

meta = {
    "transformers": "046bdb8cc2cee29d233964bb77fe6fd37cfe4aa3",
    "torch": torch.__version__,
    "hidden_size": H,
    "loop_gate_rank": RANK,
    "core_layers": CORE,
    "tokens": T,
    "rms_norm_eps": cfg.rms_norm_eps,
    "message_layers": [0, 7, CORE - 1],
    "gate_message": "messages.n1.l7",
    "streams": list(STREAMS),
}
text = json.dumps({"meta": meta, "tensors": record}, separators=(",", ":"), sort_keys=True)
open(out_path, "w").write(text + "\n")
print(f"wrote {out_path}: {len(text) + 1} bytes, sha256 {hashlib.sha256((text + chr(10)).encode()).hexdigest()}",
      file=sys.stderr)
