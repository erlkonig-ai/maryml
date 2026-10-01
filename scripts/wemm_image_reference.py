#!/usr/bin/env python3
"""Actual pinned HF WeMM prepared single-image/text CUDA oracle.

CHECKPOINT CONFIG WRAPPER TOKENIZER_JSON MARY_ROOT NEW_REPORT
One GPU-generated normalized still image, already-sized256x256, duplicated
temporal frame and exact merge-major packing ->256 BF16 patches. This is NOT
raw-image decoding/resizing, a CPU tensor prototype or retrieval admission.
The actual wrapper owns vision, scatter, MRoPE, decoder and normalization.
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
from wemm_prepared_reference import HF_SHA, CONFIG_SHA, WRAPPER_SHA, sha, bits, load_exact


def prepared_ids():
    # Same caption IDs as the existing9-ID prepared diagnostic; no invented
    # chat-template or silent tokenizer/embedding-token postprocessing.
    return [32, 11012, 13245, 1752, 28428, 264, 14367, 13, 248053] + [248056] * 64 + [248054, 248077]


def pixels_cuda():
    # All numeric input production is CUDA. Exact multiples of1/128 fit BF16.
    flat = torch.arange(3 * 256 * 256, device="cuda", dtype=torch.int64)
    rgb = (((flat * 17 + 3) % 251).to(torch.float32) - 125.0) / 128.0
    rgb = rgb.to(torch.bfloat16).reshape(3, 256, 256)
    frames = rgb.unsqueeze(0).expand(2, -1, -1, -1).contiguous()
    # Pinned Qwen2VL image_processing_qwen2_vl.py:306-324 packing, expressed
    # solely on CUDA. The actual processor's CPU numpy conversion is not used.
    return frames.reshape(1, 2, 3, 8, 2, 16, 8, 2, 16).permute(
        0, 3, 6, 4, 7, 2, 1, 5, 8).reshape(256, 1536).contiguous()


def main():
    assert len(sys.argv) == 7, __doc__
    checkpoint, config_path, wrapper_path, tokenizer_path, native, output = map(Path, sys.argv[1:])
    assert not output.exists() and not output.is_symlink()
    assert transformers.__version__ == "5.2.0" and sys.byteorder == "little"
    assert torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    config_bytes = config_path.read_bytes()
    assert sha(config_bytes) == CONFIG_SHA
    tokenizer_bytes = tokenizer_path.read_bytes()
    tokenizer = json.loads(tokenizer_bytes)
    for token_id, content in [(248053, "<|vision_start|>"), (248054, "<|vision_end|>"),
                              (248056, "<|image_pad|>"), (248077, "<embedding>")]:
        assert any(t["id"] == token_id and t["content"] == content and t["special"]
                   for t in tokenizer["added_tokens"])
    paths = (native / "scripts/wemm_image_sources.txt").read_text().splitlines()
    assert paths and len(paths) == len(set(paths))
    sources = {p: sha((native / p).read_bytes()) for p in paths}
    hf_path = Path(transformers.__file__).resolve().parent / "models/qwen3_5/modeling_qwen3_5.py"
    hf = load_exact(hf_path, "transformers.models.qwen3_5.modeling_qwen3_5", HF_SHA,
                    "transformers.models.qwen3_5")
    wrapper = load_exact(wrapper_path, "exact_wemm_image_wrapper", WRAPPER_SHA)
    config = Qwen3_5Config(**json.loads(config_bytes))
    config._attn_implementation = "eager"
    config.text_config._attn_implementation = "eager"
    config.vision_config._attn_implementation = "eager"
    torch.manual_seed(1001426077)
    torch.cuda.manual_seed_all(1001426077)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    hf.FusedRMSNormGated = None
    weights, layers, merged, scattered, positions, final_hidden = [], [], [], [], [], []
    with torch.device("cuda"), torch.no_grad():
        # Construct directly in BF16, not transient37.6GB CPU/F32 parameters.
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
        # Do NOT cast/rewrite rotary buffers to match the native recipe. Record
        # what the pinned actual constructor really produced, even on failure.
        vision_rotary_dtype = str(model.visual.rotary_pos_emb.inv_freq.dtype)
        text_rotary_dtype = str(model.language_model.rotary_emb.inv_freq.dtype)
        with safe_open(checkpoint, framework="pt", device=0) as source:
            for name, parameter in model.named_parameters():
                tensor = source.get_tensor("model." + name)
                assert tensor.is_cuda and tensor.dtype == parameter.dtype == torch.bfloat16
                assert tensor.shape == parameter.shape
                parameter.copy_(tensor)
                raw = parameter.contiguous().view(torch.uint8).flatten().cpu().numpy().tobytes()
                weights.append(dict(name="model." + name, shape=list(parameter.shape), sha256=sha(raw)))
                del raw, tensor
        assert len(weights) == 759

        def decoder_input(_module, _args, kwargs):
            scattered.append(bits(kwargs["inputs_embeds"]))
            table = kwargs["position_ids"]
            assert tuple(table.shape) == (3, 1, 75)
            positions.append(table[:, 0, :].T.cpu().tolist())

        hooks = [model.visual.merger.register_forward_hook(lambda _m, _i, o: merged.append(bits(o))),
                 model.language_model.register_forward_pre_hook(decoder_input, with_kwargs=True),
                 model.language_model.norm.register_forward_hook(lambda _m, _i, o: final_hidden.append(bits(o)))]
        for index, layer in enumerate(model.language_model.layers):
            hooks.append(layer.register_forward_hook(
                lambda _m, _i, o, index=index: layers.append(dict(layer=index, hidden=bits(o)))))
        ids = prepared_ids()
        pixels = pixels_cuda()
        pixel_bits = bits(pixels)
        pixel_bytes = pixels.contiguous().view(torch.uint8).flatten().cpu().numpy().tobytes()
        image_grid = torch.tensor([[1, 16, 16]], device="cuda", dtype=torch.int64)
        input_ids = torch.tensor([ids], device="cuda", dtype=torch.int64)
        mask = torch.ones_like(input_ids)
        # Actual processor mask semantics: explicit all-ones mask prevents
        # false packed-sequence inference from repeated image temporal IDs.
        embedding = wrapper.WeMMEmbedding.embedding(types.SimpleNamespace(model=model),
            input_ids=input_ids, attention_mask=mask, pixel_values=pixels,
            image_grid_thw=image_grid, use_cache=False)
        assert len(merged) == len(scattered) == len(positions) == len(final_hidden) == 1
        assert len(layers) == 32 and list(embedding.shape) == [1, 4096]
        assert torch.isfinite(embedding).all()
        rope_delta = int(model.rope_deltas[0, 0].item())
        for hook in hooks:
            hook.remove()
        result = dict(oracle="actual-HF-WeMM-prepared-image-CUDA-v1", hf_sha256=HF_SHA,
            config_sha256=CONFIG_SHA, wrapper_sha256=WRAPPER_SHA, config_json=config_bytes.decode(),
            generator_sha256=sha(Path(__file__).read_bytes()), native_sources=sources,
            tokenizer_sha256=sha(tokenizer_bytes), transformers=transformers.__version__, torch=torch.__version__,
            device=torch.cuda.get_device_name(0), ids=ids, grid=[1, 16, 16], weights=weights,
            pixels=pixel_bits, pixels_sha256=sha(pixel_bytes), positions=positions[0], rope_delta=rope_delta,
            vision_rotary_dtype=vision_rotary_dtype, text_rotary_dtype=text_rotary_dtype,
            mask_policy="explicit all-ones B1 unpadded; ordinary causal token order",
            merged=merged[0], scattered=scattered[0], layers=layers,
            final_hidden=final_hidden[0], embedding=bits(embedding))
    with checkpoint.open("rb") as stream:
        result["checkpoint_sha256"] = hashlib.file_digest(stream, "sha256").hexdigest()
    result["checkpoint_bytes"] = checkpoint.stat().st_size
    assert result["checkpoint_bytes"] == 18815757458
    assert sha(hf_path.read_bytes()) == HF_SHA and sha(wrapper_path.read_bytes()) == WRAPPER_SHA
    assert sources == {p: sha((native / p).read_bytes()) for p in paths}
    with output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, allow_nan=False, sort_keys=True)
    print("REFERENCE COMPLETE: one prepared image/text case; no raw-image/full-model admission")


if __name__ == "__main__":
    main()
