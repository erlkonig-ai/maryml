#!/usr/bin/env python3
"""BF16 local baselines on the same assets as wemm_reference.py.

Ovis is deliberately labelled provisional: the publisher names Swift's
qwen3_5_emb in training args but supplies no executable inference template.
This run uses its shipped HF template with no assistant generation prompt,
and the card's last-nonpadding pooling rule, not a claimed score reproduction.
Nomic uses pinned base + adapter and the trained-era explicit query prefix.
Its raw-text document path is a provisional extension, not a published
text-retrieval contract: documents pool the final lexical/newline token while
queries/images pool EOS. Modality bands also reflect these template choices;
they do not isolate a causal modality offset.
"""

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import re
import time

from wemm_reference import digest, read_probe, write_report


def nomic_adapter_key(source):
    """The pinned adapter predates HF's model -> language_model rename."""
    match = re.fullmatch(
        r"base_model\.model\.model\.(layers\.\d+\.(?:mlp|self_attn)\."
        r"(?:down_proj|gate_proj|up_proj|k_proj|q_proj|v_proj|o_proj)\.lora_[AB])\.weight", source)
    if match is None:
        raise ValueError(f"unrecognized pinned Nomic adapter key: {source}")
    return f"language_model.{match[1]}.default.weight"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", choices=("ovis", "nomic"))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--base", type=Path)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-pixels", type=int, default=602112)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("refusing to overwrite an earlier reference")
    if args.model == "nomic" and not args.base:
        parser.error("Nomic requires its separately pinned local base")
    if args.model == "nomic" and args.max_pixels != 602112:
        parser.error("Nomic control uses its native 602112 pixel budget; override is unsupported")
    manifest = read_probe(args.manifest, args.assets)
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
    import torch
    from PIL import Image
    from transformers import AutoModel, AutoProcessor
    from transformers.utils.logging import disable_progress_bar
    disable_progress_bar()
    torch.set_num_threads(8)
    torch.manual_seed(0)
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("BF16 CUDA is required")
    started = time.monotonic()
    adapter_evidence = None
    if args.model == "ovis":
        processor = AutoProcessor.from_pretrained(args.checkpoint, local_files_only=True, use_fast=False)
        processor.tokenizer.padding_side = "right"
        model, loading = AutoModel.from_pretrained(
            args.checkpoint, local_files_only=True, dtype=torch.bfloat16,
            attn_implementation="sdpa", output_loading_info=True)
        identity = {"repo": "ATH-MaaS/Ovis-VL-Embedding-9B",
                    "revision": "724f4a25d5ede0744eda3a11d2a1ec806a8f5cb0",
                    "contract": "PROVISIONAL: shipped HF template, add_generation_prompt=False; last nonpadding, FP32 L2",
                    "query_instruction": "Retrieve the relevant document or image for this query: "}
    else:
        from colpali_engine.models import BiQwen2_5, BiQwen2_5_Processor
        from peft import PeftConfig
        from safetensors.torch import load_file
        processor = BiQwen2_5_Processor.from_pretrained(args.checkpoint, local_files_only=True, use_fast=False)
        processor.tokenizer.padding_side = "left"
        model, loading = BiQwen2_5.from_pretrained(
            args.base, local_files_only=True, dtype=torch.bfloat16,
            attn_implementation="sdpa", output_loading_info=True,
            # Old checkpoint text weights use model.*, current BiQwen uses
            # language_model.*. ColPali's default reaches a removed TF API.
            key_mapping={r"^model\.(?!language_model\.|visual\.)": "language_model."})
        # Load the base explicitly first: the adapter's base revision is null.
        expected = load_file(args.checkpoint / "adapter_model.safetensors")
        model.add_adapter(PeftConfig.from_pretrained(args.checkpoint, local_files_only=True), "default")
        renamed = {nomic_adapter_key(key): value for key, value in expected.items()}
        actual = {key: value for key, value in model.named_parameters() if ".lora_" in key}
        if len(renamed) != len(expected) or renamed.keys() != actual.keys():
            raise ValueError("adapter parameter names do not match the pinned checkpoint")
        # Avoid a second, implicit HF/PEFT namespace conversion. PyTorch only
        # loads these explicitly named adapter parameters; the base is fixed.
        adapter_load = model.load_state_dict(renamed, strict=False)
        if adapter_load.unexpected_keys or any(".lora_" in k for k in adapter_load.missing_keys):
            raise ValueError(f"incomplete adapter: {adapter_load}")
        tensor_evidence = {}
        for key, source in renamed.items():
            loaded = actual[key].detach().cpu()
            if not torch.equal(loaded, source.to(dtype=loaded.dtype)):
                raise ValueError(f"adapter tensor not exactly loaded: {key}")
            tensor_evidence[key] = {
                "shape": list(loaded.shape), "source_dtype": str(source.dtype),
                "loaded_dtype": str(loaded.dtype),
                "loaded_f32_sha256": hashlib.sha256(loaded.float().contiguous().numpy().tobytes()).hexdigest()}
        adapter_evidence = {"source_tensor_count": len(expected), "loaded_tensor_count": len(actual),
                            "named_values_exact_after_dtype_cast": True,
                            "tensors": tensor_evidence, "source_names": sorted(expected)}
        identity = {"repo": "nomic-ai/nomic-embed-multimodal-7b",
                    "revision": "1291f1b6ca07061b0329df9d5713c09b294be576",
                    "base_repo": "Qwen/Qwen2.5-VL-7B-Instruct",
                    "base_revision": "cc594898137f460bfe9f0759e9844b3ce807cfb5",
                    "colpali_revision": "97487f8871ff4d5d2284411fe61bdcd2cfe99894",
                    "query_contract_source": "https://github.com/illuin-tech/colpali/blob/cae32d372ce8ce40363f50283d81760d1e301424/colpali_engine/models/qwen2_5/biqwen2_5/processing_biqwen2_5.py",
                    "contract": "explicit historical Query: / endoftext; native process_images and raw process_texts; ColPali last-token pooling"}
    # A removed unused language-model head is allowed, not a missing backbone.
    unexpected = [key for key in loading.get("unexpected_keys", []) if key != "lm_head.weight"]
    if loading.get("missing_keys") or loading.get("mismatched_keys") or loading.get("error_msgs") or unexpected:
        raise ValueError(f"incomplete backbone: {loading}")
    model = model.cuda().eval()
    torch.cuda.synchronize()
    loaded_s = time.monotonic() - started

    def encode(item):
        if args.model == "ovis":
            from qwen_vl_utils import process_vision_info
            if item["modality"] == "image":
                with Image.open(item["resolved_path"]) as image:
                    image = image.copy()
                content = [{"type": "image", "image": image, "max_pixels": args.max_pixels}]
            else:
                prefix = identity["query_instruction"] if item["role"] == "query" else ""
                content = [{"type": "text", "text": prefix + item["text"]}]
            text = processor.apply_chat_template(
                [{"role": "user", "content": content}], tokenize=False, add_generation_prompt=False)
            images, _ = process_vision_info([{"role": "user", "content": content}], image_patch_size=16)
            inputs = processor(text=text, images=images, return_tensors="pt")
        elif item["modality"] == "image":
            with Image.open(item["resolved_path"]) as image:
                inputs = processor.process_images([image.convert("RGB")])
            text = processor.visual_prompt_prefix
        else:
            text = "Query: " + item["text"] + "<|endoftext|>" if item["role"] == "query" else item["text"]
            inputs = processor.process_texts([text])
        ids = inputs["input_ids"][0].tolist()
        if not ids or len(ids) > 4096:
            raise ValueError("empty or oversized input; no silent truncation")
        tensors = {key: {"shape": list(t.shape), "dtype": str(t.dtype),
                         "sha256": hashlib.sha256(t.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()}
                   for key, t in inputs.items()}
        inputs = inputs.to("cuda")
        # Each input is independent. Do not let prior image positions leak.
        model.rope_deltas = None
        torch.cuda.synchronize()
        start = time.monotonic()
        with torch.inference_mode():
            if args.model == "ovis":
                last = int(inputs["attention_mask"][0].sum().item()) - 1
                embedding = model(**inputs, use_cache=False).last_hidden_state[0, last].float()
            else:
                embedding = model(**inputs)[0].float()
        torch.cuda.synchronize()
        elapsed = time.monotonic() - start
        norm = embedding.norm().item()
        dimension = 4096 if args.model == "ovis" else 3584
        if embedding.shape != (dimension,) or not torch.isfinite(embedding).all() or norm < 1e-10:
            raise ValueError("invalid baseline embedding")
        print(json.dumps({"id": item["id"], "tokens": len(ids), "seconds": elapsed}), flush=True)
        return {**item, "embedding": (embedding / norm).cpu().tolist(), "output_norm_before_fp32_l2": norm,
                "seconds": elapsed, "input_ids": ids, "inputs": tensors, "rendered_prompt": text}

    encoded = [encode(item) for item in manifest["items"]]
    first = next(item for item in manifest["items"] if item["modality"] == "text")
    encode(next(item for item in manifest["items"] if item["modality"] == "image"))
    repeated = encode(first)
    original = next(item for item in encoded if item["id"] == first["id"])
    delta = max(abs(a - b) for a, b in zip(repeated["embedding"], original["embedding"]))
    if delta > 1e-5:
        raise ValueError(f"post-image repeat changed: {delta}")
    versions = {name: importlib.metadata.version(name) for name in
                ("torch", "torchvision", "transformers", "qwen-vl-utils", "accelerate", "tokenizers", "safetensors")}
    if args.model == "nomic":
        versions.update({name: importlib.metadata.version(name) for name in ("peft", "colpali-engine")})
    roots = [args.checkpoint] + ([args.base] if args.base else [])
    report = {"model": {**identity, "dtype": "BF16", "versions": versions,
                        "device": torch.cuda.get_device_name(), "cuda": torch.version.cuda,
                        "capability": list(torch.cuda.get_device_capability()),
                        "processor_class": type(processor.image_processor).__name__,
                        "max_pixels": args.max_pixels,
                        "load_seconds": loaded_s, "repeat_max_abs": delta,
                        "loading": loading, "adapter_evidence": adapter_evidence,
                        "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
                        "files_sha256": {str(p): digest(p) for root in roots for p in sorted(root.iterdir())
                                         if p.is_file() and p.suffix in (".json", ".jinja", ".safetensors")}},
              "manifest_sha256": digest(args.manifest), "probe_sources": manifest.get("sources", {}),
              "limitations": manifest["limitations"], "items": encoded}
    write_report(report, args.output)


if __name__ == "__main__":
    main()
