#!/usr/bin/env python3
"""Offline, pinned WeMM BF16 reference for Mary's native implementation.

This is an import/parity tool, not Mary's runtime. It accepts only local
checkpoint files and local probe assets. Download the pinned checkpoint first;
the only remote model code executed is the hash-checked embedding wrapper.
Use a separate environment: torch 2.13.0+cu130, torchvision 0.28.0+cu130,
transformers 5.2.0, qwen-vl-utils 0.0.14, accelerate 1.14.0.
Run under the host's gb10-lock.sh reservation, not beside another GPU job.
"""

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import time


REPO = "tencent/WeMM-Embedding-9B"
REVISION = "00c52839de57a6d4fd5b78cf5522ccf0ac8ea482"
WRAPPER_SHA256 = "ac255e1fad459cc3e68891d6c3327f4486922aed02fb3c5c13fb53277ba8e94f"


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_report(report, path):
    def diagnostic(value):
        if isinstance(value, set):
            return sorted(value)
        raise TypeError(f"unserializable diagnostic: {type(value).__name__}")

    # Resolve diagnostics before creating a file: an API's set-valued loader
    # metadata must not leave a partial artifact that resembles a result.
    rendered = json.dumps(report, indent=2, allow_nan=False, default=diagnostic)
    with path.open("x") as output:
        output.write(rendered + "\n")


def read_probe(manifest_path, asset_root):
    manifest = json.loads(manifest_path.read_text())
    groups, ids = {}, set()
    for item in manifest["items"]:
        if item["id"] in ids:
            raise ValueError("duplicate probe id")
        ids.add(item["id"])
        if item["split"] not in ("calibration", "heldout"):
            raise ValueError("unknown split")
        if groups.setdefault(item["group"], item["split"]) != item["split"]:
            raise ValueError("a topic/asset family spans both splits")
        if item["modality"] not in ("text", "image"):
            raise ValueError("unknown modality")
        if item["role"] not in ("query", "document"):
            raise ValueError("unknown role")
        if "path" in item:
            path = (asset_root / item["path"]).resolve()
            if not path.is_relative_to(asset_root.resolve()):
                raise ValueError("probe path escapes asset root")
            if digest(path) != item["sha256"]:
                raise ValueError(f"probe asset changed: {item['id']}")
            item["resolved_path"] = str(path)
        if item["modality"] == "text":
            text = (Path(item["resolved_path"]).read_text()
                    if "path" in item else item["text"])
            if not text.strip():
                raise ValueError("empty probe text")
            item["text"] = text
        elif "path" not in item:
            raise ValueError("image probe needs a local path")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-pixels", type=int, default=602112)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("refusing to overwrite an earlier reference")
    if args.max_pixels < 65536 or args.max_tokens < 1 or args.threads < 1:
        parser.error("invalid resource bound")
    checkpoint = args.checkpoint.resolve()
    if digest(checkpoint / "modeling_wemm_embedding.py") != WRAPPER_SHA256:
        raise ValueError("unreviewed model wrapper")
    config = json.loads((checkpoint / "config.json").read_text())
    if config.get("auto_map") != {
        "AutoModel": "modeling_wemm_embedding.WeMMEmbedding",
        "AutoModelForCausalLM": "modeling_wemm_embedding.WeMMEmbedding",
    }:
        raise ValueError("unreviewed custom-code mapping")
    manifest = read_probe(args.manifest, args.assets)
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
    import torch
    from PIL import Image
    from qwen_vl_utils import process_vision_info
    from transformers import AutoModel, AutoProcessor
    from transformers.utils.logging import disable_progress_bar

    disable_progress_bar()

    torch.set_num_threads(args.threads)
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("this reference requires a BF16 CUDA device")
    torch.manual_seed(0)
    processor = AutoProcessor.from_pretrained(checkpoint, local_files_only=True, use_fast=False)
    processor.tokenizer.padding_side = "right"
    started = time.monotonic()
    model, loading = AutoModel.from_pretrained(
        checkpoint, trust_remote_code=True, local_files_only=True,
        dtype=torch.bfloat16, attn_implementation="sdpa", output_loading_info=True,
    )
    if any(loading.get(key) for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")):
        raise ValueError(f"incomplete checkpoint: {loading}")
    model = model.cuda().eval()
    torch.cuda.synchronize()
    loaded_s = time.monotonic() - started
    if not hasattr(model, "embedding"):
        raise RuntimeError("loaded model does not implement the pinned wrapper")

    def encode(item):
        content = []
        if item["modality"] == "image":
            with Image.open(item["resolved_path"]) as image:
                image = image.copy()
            content.append({"type": "image", "image": image,
                            "max_pixels": args.max_pixels})
        else:
            content.append({"type": "text", "text": item["text"]})
        messages = [{"role": "user", "content": content}]
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False)
        images, videos = process_vision_info(messages, image_patch_size=16)
        if videos is not None:
            raise ValueError("this probe does not test video")
        inputs = processor(text=text, images=images, return_tensors="pt")
        ids = inputs["input_ids"][0].tolist()
        # The checkpoint tokenizer postprocessor appends the embedding token.
        if ids[-1] != 248077:
            raise ValueError("pooling would not select the embedding token")
        if len(ids) > args.max_tokens:
            raise ValueError(f"{item['id']}: {len(ids)} tokens exceeds bound; never silently truncate")
        tensors = {}
        for key, tensor in inputs.items():
            raw = tensor.contiguous().view(torch.uint8).numpy().tobytes()
            tensors[key] = {"shape": list(tensor.shape), "dtype": str(tensor.dtype),
                            "sha256": hashlib.sha256(raw).hexdigest()}
        inputs = inputs.to("cuda")
        torch.cuda.synchronize()
        start = time.monotonic()
        with torch.inference_mode():
            embedding = model.embedding(**inputs, use_cache=False)[0].float()
        torch.cuda.synchronize()
        elapsed = time.monotonic() - start
        norm = embedding.norm().item()
        if embedding.shape != (4096,) or not torch.isfinite(embedding).all() or norm < 1e-10:
            raise ValueError("invalid reference embedding")
        # Retain the wrapper's BF16 normalization error and normalize in F32
        # for actual cosines. No projection or learned calibration happens here.
        embedding = (embedding / norm).cpu().tolist()
        print(json.dumps({"id": item["id"], "tokens": len(ids), "seconds": elapsed}), flush=True)
        return {**item, "embedding": embedding, "wrapper_output_norm": norm,
                "seconds": elapsed, "input_ids": ids, "inputs": tensors,
                "rendered_prompt": text}

    encoded = [encode(item) for item in manifest["items"]]
    # Text after vision must not inherit the previous image's rope_deltas.
    first_text = next(item for item in manifest["items"] if item["modality"] == "text")
    encode(next(item for item in manifest["items"] if item["modality"] == "image"))
    repeated = encode(first_text)
    first = next(item for item in encoded if item["id"] == first_text["id"])
    repeat_delta = max(abs(a - b) for a, b in zip(first["embedding"], repeated["embedding"]))
    if repeat_delta > 1e-5:
        raise ValueError(f"repeat after mixed inputs changed the vector: {repeat_delta}")
    versions = {name: importlib.metadata.version(name) for name in
                ("torch", "torchvision", "transformers", "qwen-vl-utils", "accelerate", "tokenizers", "safetensors")}
    report = {
        "model": {"repo": REPO, "revision": REVISION, "dtype": "BF16",
                  "backend": "CUDA / sdpa",
                  "processor_class": type(processor.image_processor).__name__,
                  "processor_use_fast": False,
                  "cuda": torch.version.cuda,
                  "capability": list(torch.cuda.get_device_capability()),
                  "loading": loading,
                  "linear_attention_implementation": type(model.model.language_model.layers[0].linear_attn).__name__,
                  "delta_kernel": model.model.language_model.layers[0].linear_attn.chunk_gated_delta_rule.__name__,
                  "device": torch.cuda.get_device_name(), "versions": versions,
                  "files_sha256": {p.name: digest(p) for p in sorted(checkpoint.iterdir())
                                   if p.is_file() and p.suffix in (".json", ".jinja", ".py", ".safetensors")},
                  "max_pixels": args.max_pixels, "max_tokens": args.max_tokens,
                  "load_seconds": loaded_s,
                  "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
                  "repeat_max_abs": repeat_delta},
        "manifest_sha256": digest(args.manifest),
        "probe_sources": manifest.get("sources", {}),
        "limitations": manifest.get("limitations", []),
        "items": encoded,
    }
    write_report(report, args.output)


if __name__ == "__main__":
    main()
