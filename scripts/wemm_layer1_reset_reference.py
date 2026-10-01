#!/usr/bin/env python3
"""One actual-HF L1 reset-input diagnostic; no model formula on the CPU.

CHECKPOINT CONFIG RETAINED_FULL_HF_JSON NATIVE_SOURCE_ROOT NEW_REPORT_JSON
The retained L0 BF16 bits are transported to CUDA unchanged. Only layer1 runs.
This never changes the retained cumulative comparison or its failing bounds.
"""
import hashlib
import json
from pathlib import Path
import sys
import types

import torch
import transformers
from safetensors import safe_open
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5Config

HF_SHA = "d1ae3856f53763591ec65054129af46e003a1715efcf7eef2b752a31e85526b8"
CONFIG_SHA = "34abd67be4bab3d749ba7b3ad2daa5fc0a09ab12064a64f053fa244fa26c6004"
BASELINE_SHA = "036b467c8c9b80c7263a6a18737e1f58c01c3932368bbb7bec39432468a6190b"
CHECKPOINT_SHA = "b6d5dff9e632973991f1d0cbfcfd26c42ffd66fbb8ebd6f852aece08e9794fa4"
NAMES = ("input_norm", "mixer_out", "residual1", "post_norm", "mlp_gate",
         "mlp_up", "mlp_product", "mlp_down", "output")


def sha(b):
    return hashlib.sha256(b).hexdigest()


def bits(t):
    assert t.is_cuda and t.dtype == torch.bfloat16
    return t.detach().contiguous().view(torch.uint16).flatten().cpu().tolist()


def raw(values):
    return b"".join(x.to_bytes(2, "little") for x in values)


def main():
    assert len(sys.argv) == 6, __doc__
    checkpoint, config_path, baseline_path, native, output = map(Path, sys.argv[1:])
    assert not output.exists() and not output.is_symlink()
    assert transformers.__version__ == "5.2.0" and torch.cuda.is_bf16_supported()
    assert baseline_path.stat().st_size < 64 * 1024 * 1024
    baseline_bytes = baseline_path.read_bytes()
    assert sha(baseline_bytes) == BASELINE_SHA
    base = json.loads(baseline_bytes)
    assert base["checkpoint_sha256"] == CHECKPOINT_SHA
    config_bytes = config_path.read_bytes()
    assert sha(config_bytes) == CONFIG_SHA
    with checkpoint.open("rb") as stream:
        assert hashlib.file_digest(stream, "sha256").hexdigest() == CHECKPOINT_SHA
    assert checkpoint.stat().st_size == 18815757458
    hf_path = Path(transformers.__file__).resolve().parent / "models/qwen3_5/modeling_qwen3_5.py"
    source = hf_path.read_bytes()
    assert sha(source) == HF_SHA
    hf = types.ModuleType("transformers.models.qwen3_5.modeling_qwen3_5")
    hf.__file__ = str(hf_path)
    hf.__package__ = "transformers.models.qwen3_5"
    sys.modules[hf.__name__] = hf
    exec(compile(source, str(hf_path), "exec"), hf.__dict__)
    paths = (native / "scripts/wemm_layer1_reset_sources.txt").read_text().splitlines()
    assert len(paths) == len(set(paths))
    sources = {p: sha((native / p).read_bytes()) for p in paths}
    config = Qwen3_5Config(**json.loads(config_bytes)).text_config
    assert config.layer_types[1] == "linear_attention" and config.hidden_size == 4096
    assert base["layers"][0]["layer"] == 0 and base["layers"][1]["layer"] == 1
    input_bits = base["layers"][0]["hidden"]
    expected = base["layers"][1]["hidden"]
    assert len(base["ids"]) == 9 and len(input_bits) == len(expected) == 9 * 4096
    torch.manual_seed(1001426077)
    torch.cuda.manual_seed_all(1001426077)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    hf.FusedRMSNormGated = None
    stages, weights = {}, []

    def capture(name, t):
        assert name not in stages
        stages[name] = bits(t)

    with torch.device("cuda"), torch.no_grad():
        previous = torch.get_default_dtype()
        torch.set_default_dtype(torch.bfloat16)
        try:
            layer = hf.Qwen3_5DecoderLayer(config, layer_idx=1).eval()
        finally:
            torch.set_default_dtype(previous)
        core = layer.linear_attn
        core.causal_conv1d_fn = None
        core.causal_conv1d_update = hf.torch_causal_conv1d_update
        core.chunk_gated_delta_rule = hf.torch_chunk_gated_delta_rule
        core.recurrent_gated_delta_rule = hf.torch_recurrent_gated_delta_rule
        with safe_open(checkpoint, framework="pt", device=0) as store:
            for name, parameter in layer.named_parameters():
                name = "model.language_model.layers.1." + name
                tensor = store.get_tensor(name)
                assert tensor.is_cuda and tensor.dtype == parameter.dtype == torch.bfloat16
                assert tensor.shape == parameter.shape
                parameter.copy_(tensor)
                payload_hash = sha(parameter.contiguous().view(torch.uint8).flatten().cpu().numpy().tobytes())
                row = dict(name=name, shape=list(parameter.shape), sha256=payload_hash)
                assert row in base["weights"]
                weights.append(row)
                del tensor
        assert len(weights) == 14
        hooks = []
        for module, name in ((layer.input_layernorm, "input_norm"), (core, "mixer_out"),
                             (layer.post_attention_layernorm, "post_norm"),
                             (layer.mlp.gate_proj, "mlp_gate"), (layer.mlp.up_proj, "mlp_up"),
                             (layer.mlp.down_proj, "mlp_down")):
            hooks.append(module.register_forward_hook(lambda _m, _i, o, name=name: capture(name, o)))
        hooks.append(layer.post_attention_layernorm.register_forward_pre_hook(
            lambda _m, i: capture("residual1", i[0])))
        hooks.append(layer.mlp.down_proj.register_forward_pre_hook(
            lambda _m, i: capture("mlp_product", i[0])))
        # Integer byte transport of a retained GPU witness, not CPU model math.
        x = torch.tensor(input_bits, dtype=torch.uint16, device="cuda").view(torch.bfloat16).reshape(1, 9, 4096)
        assert bits(x) == input_bits
        cache = hf.Qwen3_5DynamicCache(config=config)
        y = layer(x, position_embeddings=None, attention_mask=None, past_key_values=cache,
                  cache_position=torch.arange(9, dtype=torch.int64, device="cuda"))
        capture("output", y)
        for hook in hooks:
            hook.remove()
        assert set(stages) == set(NAMES)
        for name, values in stages.items():
            assert len(values) == 9 * (12288 if name in ("mlp_gate", "mlp_up", "mlp_product") else 4096)
        reproduced = stages["output"] == expected
        result = dict(oracle="actual-HF-WeMM-layer1-reset-CUDA-v1", hf_sha256=HF_SHA,
            config_sha256=CONFIG_SHA, checkpoint_sha256=CHECKPOINT_SHA,
            baseline_sha256=BASELINE_SHA, native_sources=sources, ids=base["ids"],
            input_bits=input_bits, input_sha256=sha(raw(input_bits)),
            hf_reproduces_retained_layer1=reproduced, weights=weights,
            stages=[dict(name=n, bits=stages[n], sha256=sha(raw(stages[n]))) for n in NAMES],
            device=torch.cuda.get_device_name(0), torch=torch.__version__,
            transformers=transformers.__version__,
            scope="one reset-input layer; original cumulative full-decoder FAIL remains unchanged")
    assert sources == {p: sha((native / p).read_bytes()) for p in paths}
    assert sha(hf_path.read_bytes()) == HF_SHA and sha(baseline_path.read_bytes()) == BASELINE_SHA
    with output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, allow_nan=False, sort_keys=True)
    print(f"L1 RESET HF CONTROL: matches retained HF layer1 = {reproduced}")
    assert reproduced, "standalone HF control differs; report preserved, do not silently substitute its output"


if __name__ == "__main__":
    main()
