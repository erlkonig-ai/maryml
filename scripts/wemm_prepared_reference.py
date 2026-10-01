#!/usr/bin/env python3
"""Full actual HF WeMM decoder oracle for PREPARED single-sequence IDs.

CHECKPOINT CONFIG WRAPPER TOKENIZER_JSON PREPARED_IDS_JSON MARY_ROOT NEW_REPORT
Prepared IDs already include <embedding>; no tokenizer/template is invented.
The actual multimodal model runs the no-image leg here, not a second embedder.
All initialization and model arithmetic are CUDA BF16; CPU work is metadata,
byte hashing/transport and result serialization only. This source is unrun.
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
WRAPPER_SHA = "ac255e1fad459cc3e68891d6c3327f4486922aed02fb3c5c13fb53277ba8e94f"


def sha(data):
    return hashlib.sha256(data).hexdigest()


def bits(t):
    assert t.is_cuda and t.dtype == torch.bfloat16
    return t.detach().contiguous().view(torch.uint16).flatten().cpu().tolist()


def load_exact(path, name, expected, package=None):
    source = path.read_bytes()
    assert sha(source) == expected
    module = types.ModuleType(name)
    module.__file__ = str(path)
    module.__package__ = package
    sys.modules[name] = module
    exec(compile(source, str(path), "exec"), module.__dict__)
    return module


def main():
    assert len(sys.argv) == 8, __doc__
    checkpoint, config_path, wrapper_path, tokenizer_path, ids_path, native, output = map(Path, sys.argv[1:])
    assert not output.exists() and not output.is_symlink()
    assert transformers.__version__ == "5.2.0" and sys.byteorder == "little"
    assert torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    cfg_bytes = config_path.read_bytes()
    assert sha(cfg_bytes) == CONFIG_SHA
    ids_bytes = ids_path.read_bytes()
    ids = json.loads(ids_bytes)
    assert isinstance(ids, list) and 1 <= len(ids) <= 256
    assert all(type(i) is int and 0 <= i < 248078 for i in ids)
    assert ids[-1] == 248077 and not set(ids).intersection((248053, 248054, 248056, 248057))
    tokenizer_bytes = tokenizer_path.read_bytes()
    tokenizer = json.loads(tokenizer_bytes)
    assert any(t["id"] == 248077 and t["content"] == "<embedding>" and t["special"]
               for t in tokenizer["added_tokens"])
    # Gate includes exactly these source bytes at compilation. This binds the
    # model path and gate, not a claim about every transitive dependency binary.
    source_paths = (native / "scripts/wemm_prepared_sources.txt").read_text().splitlines()
    assert source_paths and len(source_paths) == len(set(source_paths))
    sources = {p: sha((native / p).read_bytes()) for p in source_paths}
    hf_path = Path(transformers.__file__).resolve().parent / "models/qwen3_5/modeling_qwen3_5.py"
    hf = load_exact(hf_path, "transformers.models.qwen3_5.modeling_qwen3_5", HF_SHA,
                    "transformers.models.qwen3_5")
    wrapper = load_exact(wrapper_path, "exact_wemm_prepared_wrapper", WRAPPER_SHA)
    config = Qwen3_5Config(**json.loads(cfg_bytes))
    config._attn_implementation = "eager"
    config.text_config._attn_implementation = "eager"
    torch.manual_seed(1001426077)
    torch.cuda.manual_seed_all(1001426077)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    hf.FusedRMSNormGated = None
    weights, layers = [], []
    gathered, final_hidden = [], []
    with torch.device("cuda"), torch.no_grad():
        # A later .to(bfloat16) would transiently initialize ~37.6GB of F32
        # weights. Construct in BF16 on CUDA from the first allocation instead.
        previous_dtype = torch.get_default_dtype()
        torch.set_default_dtype(torch.bfloat16)
        try:
            model = hf.Qwen3_5Model(config).eval()
        finally:
            torch.set_default_dtype(previous_dtype)
        for layer in model.language_model.layers:
            if layer.layer_type == "linear_attention":
                core = layer.linear_attn
                core.causal_conv1d_fn = None
                core.causal_conv1d_update = hf.torch_causal_conv1d_update
                core.chunk_gated_delta_rule = hf.torch_chunk_gated_delta_rule
                core.recurrent_gated_delta_rule = hf.torch_recurrent_gated_delta_rule
        with safe_open(checkpoint, framework="pt", device=0) as source:
            for name, parameter in model.named_parameters():
                tensor = source.get_tensor("model." + name)
                assert tensor.is_cuda and tensor.dtype == parameter.dtype == torch.bfloat16
                assert tensor.shape == parameter.shape
                parameter.copy_(tensor)
                if name.startswith("language_model."):
                    raw = parameter.contiguous().view(torch.uint8).flatten().cpu().numpy().tobytes()
                    weights.append(dict(name="model." + name, shape=list(parameter.shape), sha256=sha(raw)))
                    del raw
                del tensor
        assert len(weights) == 426
        hooks = [model.language_model.embed_tokens.register_forward_hook(
            lambda _m, _i, o: gathered.append(bits(o))),
            model.language_model.norm.register_forward_hook(lambda _m, _i, o: final_hidden.append(bits(o)))]
        for index, layer in enumerate(model.language_model.layers):
            hooks.append(layer.register_forward_hook(
                lambda _m, _i, o, index=index: layers.append(dict(layer=index, hidden=bits(o)))))
        input_ids = torch.tensor([ids], dtype=torch.int64, device="cuda")
        # Fresh cache, ordinary sequential positions, no padded/packed sequence.
        # The actual wrapper resets rope_deltas before calling the actual model.
        embedding = wrapper.WeMMEmbedding.embedding(types.SimpleNamespace(model=model),
            input_ids=input_ids, attention_mask=None, use_cache=True)
        assert len(layers) == 32 and len(gathered) == len(final_hidden) == 1
        assert list(embedding.shape) == [1, 4096] and torch.isfinite(embedding).all()
        for hook in hooks:
            hook.remove()
        result = dict(oracle="actual-HF-WeMM-prepared-decoder-CUDA-v1", hf_sha256=HF_SHA,
            config_sha256=CONFIG_SHA, wrapper_sha256=WRAPPER_SHA,
            generator_sha256=sha(Path(__file__).read_bytes()), native_sources=sources,
            tokenizer_sha256=sha(tokenizer_bytes), prepared_ids_sha256=sha(ids_bytes),
            transformers=transformers.__version__, torch=torch.__version__,
            device=torch.cuda.get_device_name(0), ids=ids, weights=weights,
            gathered=gathered[0], layers=layers, final_hidden=final_hidden[0], embedding=bits(embedding))
    # Exact checkpoint identity, streamed byte hashing; not CPU model arithmetic.
    with checkpoint.open("rb") as stream:
        result["checkpoint_sha256"] = hashlib.file_digest(stream, "sha256").hexdigest()
    result["checkpoint_bytes"] = checkpoint.stat().st_size
    assert result["checkpoint_bytes"] == 18815757458
    assert sha(hf_path.read_bytes()) == HF_SHA and sha(wrapper_path.read_bytes()) == WRAPPER_SHA
    assert sources == {p: sha((native / p).read_bytes()) for p in source_paths}
    with output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, allow_nan=False, sort_keys=True)
    print("REFERENCE COMPLETE: 32-layer prepared decoder; not vision/batched/full-model admission")


if __name__ == "__main__":
    main()
