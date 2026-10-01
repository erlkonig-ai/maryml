#!/usr/bin/env python3
"""Bounded behavior/repro/performance probe; HF coordinate error is diagnostic.

prepare CHECKPOINT_DIR ASSET_DIR SOURCE_ROOT NEW_DIR
reference CHECKPOINT_DIR SOURCE_ROOT FIXTURE_JSON NEW_REPORT
score FIXTURE_JSON HF_REPORT NATIVE_REPORT NATIVE_NEW_PROCESS_REPORT NEW_REPORT

CPU: image codec, tokenizer/integer metadata, byte transport, labeled ranking.
CUDA: resize/normalize/patch packing, model inference and cosine arithmetic.
No model math or image-preprocessing fallback on CPU; no truncation/threshold fit.
"""
import hashlib
import json
from pathlib import Path
import re
import sys
import time
import types

import torch
import torch.nn.functional as F
import transformers
from PIL import Image
from safetensors import safe_open
from transformers import AutoProcessor
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5Config
from wemm_prepared_reference import HF_SHA, CONFIG_SHA, WRAPPER_SHA, sha, bits, load_exact

CHECKPOINT_SHA = "b6d5dff9e632973991f1d0cbfcfd26c42ffd66fbb8ebd6f852aece08e9794fa4"
TOKEN_FILES = {
    "chat_template.jinja": "273d8e0e683b885071fb17e08d71e5f2a5ddfb5309756181681de4f5a1822d80",
    "tokenizer.json": "40e444c744512f423da4c8443c47c21e22ff76056ba4e9796a81c04c13a9daf0",
    "tokenizer_config.json": "bef247a6b4d35e878d074738ae63532fb230d31d2fe362c326b6a729a5c7a7bb",
    "processor_config.json": "d89ef49ce9cd37fbf510158e13c1ef063d9286411c1ec9049932dbe0487143b1",
}


def save(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, sort_keys=True, allow_nan=False)


def read(path):
    path = Path(path)
    assert path.stat().st_size < 16 * 1024 * 1024
    raw = path.read_bytes()
    return json.loads(raw), sha(raw)


def sources(root):
    paths = (root / "scripts/wemm_behavior_sources.txt").read_text().splitlines()
    assert paths and len(paths) == len(set(paths))
    return {p: sha((root / p).read_bytes()) for p in paths}


def setup():
    assert transformers.__version__ == "5.2.0" and sys.byteorder == "little"
    assert torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    torch.manual_seed(1001426077)
    torch.cuda.manual_seed_all(1001426077)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def image_cuda(path, crop):
    # CPU work is codec-only. RGBA values become a CUDA tensor before alpha
    # compositing, resize, normalization or patch packing. No PIL resize.
    with Image.open(path) as encoded:
        rgba = encoded.convert("RGBA")
        width, height = original = rgba.size
        assert 0 < width <= 8192 and 0 < height <= 8192
        raw = bytearray(rgba.tobytes())
    x = torch.frombuffer(raw, dtype=torch.uint8).to("cuda").reshape(height, width, 4).float()
    if crop is not None:
        assert len(crop) == 4 and all(type(v) is int for v in crop)
        x0, y0, x1, y1 = crop
        assert 0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height
        x = x[y0:y1, x0:x1]
        width, height = x1 - x0, y1 - y0
    # Hash the exact cropped RGBA bytes, not an inferred PDF content identity.
    crop_raw = x.to(torch.uint8).contiguous().cpu().numpy().tobytes()
    alpha = x[..., 3:] / 255.0
    rgb = x[..., :3] * alpha + 255.0 * (1.0 - alpha)
    new_h = max(1, round(height * min(256 / height, 256 / width)))
    new_w = max(1, round(width * min(256 / height, 256 / width)))
    small = F.interpolate(rgb.permute(2, 0, 1)[None], size=(new_h, new_w),
                          mode="bicubic", align_corners=False, antialias=True)[0].clamp(0, 255)
    canvas = torch.full((3, 256, 256), 255.0, device="cuda", dtype=torch.float32)
    top, left = (256 - new_h) // 2, (256 - new_w) // 2
    canvas[:, top:top + new_h, left:left + new_w] = small
    # CUDA does rounding and layout; CPU only encodes the resulting RGB bytes.
    preview = canvas.round().to(torch.uint8).permute(1, 2, 0).contiguous().cpu().numpy().tobytes()
    normalized = ((canvas / 255.0 - 0.5) / 0.5).to(torch.bfloat16)
    frames = normalized[None].expand(2, -1, -1, -1).contiguous()
    pixels = frames.reshape(1, 2, 3, 8, 2, 16, 8, 2, 16).permute(
        0, 3, 6, 4, 7, 2, 1, 5, 8).reshape(256, 1536).contiguous()
    payload = pixels.view(torch.uint8).flatten().cpu().numpy().tobytes()
    return payload, preview, dict(original_wh=list(original), crop_xyxy=crop,
                         crop_wh=[width, height], crop_rgba_sha256=sha(crop_raw),
                         resized_wh=[new_w, new_h], padding_top_left=[top, left], grid=[1, 16, 16])


def prepare(checkpoint, assets, native, out):
    setup()
    assert not out.exists() and not out.is_symlink()
    out.mkdir()
    fixture, fixture_sha = read(native / "scripts/wemm_behavior_fixture.json")
    original_sources = sources(native)
    for name, wanted in TOKEN_FILES.items():
        assert sha((checkpoint / name).read_bytes()) == wanted
    processor = AutoProcessor.from_pretrained(checkpoint, local_files_only=True, use_fast=False)
    rows = []
    for item in fixture["items"]:
        row = dict(item)
        if "path" in item:
            path = assets / item["path"]
            raw = path.read_bytes()
            assert sha(raw) == item["sha256"]
            row["source_path"] = str(path.resolve())
        if item["modality"] == "image":
            if "source_document" in item:
                document = fixture["sources"][item["source_document"]]
                assert sha((assets / item["source_document"]).read_bytes()) == document["sha256"]
            torch.cuda.synchronize()
            start = time.monotonic()
            payload, preview, geometry = image_cuda(path, item.get("crop_xyxy"))
            prepare_ms = (time.monotonic() - start) * 1000
            pixel_path = out / (item["id"] + ".bf16")
            with pixel_path.open("xb") as stream:
                stream.write(payload)
            preview_path = out / (item["id"] + "-prepared.png")
            Image.frombytes("RGB", (256, 256), preview).save(preview_path)
            row.update(pixels_path=str(pixel_path.resolve()), pixels_sha256=sha(payload),
                       geometry=geometry, tensor_shape=[256, 1536], tensor_dtype="BF16",
                       prepare_ms=prepare_ms, prepared_preview_path=str(preview_path.resolve()),
                       prepared_preview_sha256=sha(preview_path.read_bytes()))
            content = [{"type": "image"}]
        else:
            text = item.get("text", raw.decode("utf-8") if "path" in item else "")
            if "excerpt" in item:
                text = " ".join(text.split())
                a, b = item["excerpt"]["start"], item["excerpt"]["end"]
                assert text.count(a) == text.count(b) == 1
                start, end = text.index(a), text.index(b) + len(b)
                assert start < end
                row["normalized_source_character_range"] = [start, end]
                text = text[start:end]
            row["text"] = text
            row["text_sha256"] = sha(text.encode())
            content = [{"type": "text", "text": text}]
        rendered = processor.apply_chat_template([{"role": "user", "content": content}],
                                                  tokenize=False, add_generation_prompt=False)
        if item["modality"] == "image":
            assert rendered.count("<|image_pad|>") == 1
            # Exact processor metadata expansion for grid1x16x16, merge2.
            rendered = rendered.replace("<|image_pad|>", "<|image_pad|>" * 64)
        ids = processor.tokenizer(rendered, truncation=False, padding=False)["input_ids"]
        assert 1 <= len(ids) <= 256, f"{item['id']}: {len(ids)} tokens; no truncation"
        assert ids[-1] == 248077 and ids.count(248077) == 1
        assert ids.count(248056) == (64 if item["modality"] == "image" else 0)
        row.update(ids=ids, token_count=len(ids), rendered_text=rendered, rendered_sha256=sha(rendered.encode()))
        rows.append(row)
    assert len(rows) == 11 and sources(native) == original_sources
    result = dict(fixture, schema="wemm-prepared-behavior-v1", items=rows,
                  fixture_definition_sha256=fixture_sha, tokenizer_files=TOKEN_FILES,
                  native_sources=original_sources, preprocessing="CUDA F32 bicubic antialias aspect-fit256+whitepad, /255, mean/std0.5, BF16, duplicate temporal, merge-major; not processor resize parity")
    save(out / "prepared.json", result)


def reference(checkpoint, native, fixture_path, output):
    setup()
    assert not output.exists()
    fixture, fixture_sha = read(fixture_path)
    own_sources = sources(native)
    assert fixture["native_sources"] == own_sources
    config_bytes = (checkpoint / "config.json").read_bytes()
    assert sha(config_bytes) == CONFIG_SHA
    checkpoint_path = checkpoint / "model.safetensors"
    with checkpoint_path.open("rb") as stream:
        assert hashlib.file_digest(stream, "sha256").hexdigest() == CHECKPOINT_SHA
    hf_path = Path(transformers.__file__).resolve().parent / "models/qwen3_5/modeling_qwen3_5.py"
    hf = load_exact(hf_path, "transformers.models.qwen3_5.modeling_qwen3_5", HF_SHA,
                    "transformers.models.qwen3_5")
    wrapper = load_exact(checkpoint / "modeling_wemm_embedding.py", "wemm_behavior_wrapper", WRAPPER_SHA)
    config = Qwen3_5Config(**json.loads(config_bytes))
    for c in [config, config.text_config, config.vision_config]:
        c._attn_implementation = "eager"
    hf.FusedRMSNormGated = None
    start = time.monotonic()
    with torch.device("cuda"), torch.no_grad():
        previous = torch.get_default_dtype()
        torch.set_default_dtype(torch.bfloat16)
        try:
            model = hf.Qwen3_5Model(config).eval()
        finally:
            torch.set_default_dtype(previous)
        for layer in model.language_model.layers:
            if layer.layer_type == "linear_attention":
                core = layer.linear_attn
                core.causal_conv1d_fn = None
                core.causal_conv1d_update = hf.torch_causal_conv1d_update
                core.chunk_gated_delta_rule = hf.torch_chunk_gated_delta_rule
                core.recurrent_gated_delta_rule = hf.torch_recurrent_gated_delta_rule
        weights = []
        with safe_open(checkpoint_path, framework="pt", device=0) as source:
            for name, parameter in model.named_parameters():
                tensor = source.get_tensor("model." + name)
                assert tensor.is_cuda and tensor.dtype == parameter.dtype == torch.bfloat16
                assert tensor.shape == parameter.shape
                parameter.copy_(tensor)
                raw = parameter.contiguous().view(torch.uint8).flatten().cpu().numpy().tobytes()
                weights.append(dict(name="model." + name, shape=list(parameter.shape), sha256=sha(raw)))
                del raw, tensor
        assert len(weights) == 759
        load_ms = (time.monotonic() - start) * 1000
        embeddings = []
        for reverse in [False, True]:
            rows = list(reversed(fixture["items"])) if reverse else fixture["items"]
            for item in rows:
                start = time.monotonic()
                ids = torch.tensor([item["ids"]], device="cuda", dtype=torch.int64)
                extra = {}
                if item["modality"] == "image":
                    raw = bytearray(Path(item["pixels_path"]).read_bytes())
                    assert sha(raw) == item["pixels_sha256"] and len(raw) == 256 * 1536 * 2
                    extra = dict(pixel_values=torch.frombuffer(raw, dtype=torch.uint16).to("cuda").view(torch.bfloat16).reshape(256, 1536),
                                 image_grid_thw=torch.tensor([[1, 16, 16]], device="cuda", dtype=torch.int64))
                torch.cuda.synchronize()
                prepare_ms = (time.monotonic() - start) * 1000
                start = time.monotonic()
                y = wrapper.WeMMEmbedding.embedding(types.SimpleNamespace(model=model), input_ids=ids,
                    attention_mask=torch.ones_like(ids), use_cache=False, **extra)
                torch.cuda.synchronize()
                forward_ms = (time.monotonic() - start) * 1000
                start = time.monotonic()
                assert y.shape == (1, 4096) and torch.isfinite(y).all()
                raw = y.contiguous().view(torch.uint8).flatten().cpu().numpy().tobytes()
                encoded = bits(y)
                readback_ms = (time.monotonic() - start) * 1000
                embeddings.append(dict(id=item["id"], **{"pass":"reverse" if reverse else "forward"},
                    bits=encoded, sha256=sha(raw), elapsed_ms=forward_ms + readback_ms,
                    prepare_ms=prepare_ms, forward_ms=forward_ms, readback_ms=readback_ms))
        result = dict(schema="wemm-behavior-embeddings-v1", engine="HF-CUDA-BF16", fixture_sha256=fixture_sha,
            checkpoint_sha256=CHECKPOINT_SHA, native_sources=own_sources, weights=weights, load_ms=load_ms,
            embeddings=embeddings, vision_rotary_dtype=str(model.visual.rotary_pos_emb.inv_freq.dtype),
            text_rotary_dtype=str(model.language_model.rotary_emb.inv_freq.dtype))
    assert sources(native) == own_sources
    save(output, result)


def retrieval_tasks(fixture):
    rows = {i["id"]: i for i in fixture["items"]}
    assert len(rows) == len(fixture["items"])
    docs = [i["id"] for i in fixture["items"] if i["role"] == "document"]
    text_queries = [i["id"] for i in fixture["items"] if i["role"] == "query"]
    tasks = []
    for query in text_queries + fixture["reciprocal_queries"]:
        assert query in rows
        all_candidates = [d for d in docs if d != query]
        modes = [("all-documents", all_candidates)]
        opposite = "text" if rows[query]["modality"] == "image" else "image"
        modes.append((f"{rows[query]['modality']}-to-{opposite}",
                      [d for d in all_candidates if rows[d]["modality"] == opposite]))
        for mode, candidates in modes:
            relevant = [d for d in candidates if rows[d]["group"] == rows[query]["group"]]
            irrelevant = [d for d in candidates if d not in relevant]
            # Bloom has no labeled image positive in this small fixture. Keep
            # this unscorable row visible, never silently call it a success.
            tasks.append(dict(query=query, mode=mode, candidates=candidates,
                              relevant=relevant, irrelevant=irrelevant))
    return tasks


def validate_report_provenance(reports, paths, digests):
    expected = {"hf":"HF-CUDA-BF16", "native":"native-CUDA-BF16", "native_new_process":"native-CUDA-BF16"}
    for name, engine in expected.items():
        assert reports[name]["engine"] == engine, f"{name}: wrong engine"
        assert reports[name]["schema"] == "wemm-behavior-embeddings-v1", f"{name}: wrong schema"
        assert re.fullmatch("[0-9a-f]{64}", digests[name]), f"{name}: invalid report digest"
    assert paths["native"].resolve() != paths["native_new_process"].resolve(), "same native report path is not another process"
    assert digests["native"] != digests["native_new_process"], "same native report bytes are not another process"
    for name in ["native", "native_new_process"]:
        assert reports[name]["hf_report_sha256"] == digests["hf"], f"{name}: different HF report binding"
        process = reports[name].get("process")
        assert isinstance(process, dict), f"{name}: missing process evidence"
        assert type(process.get("pid")) is int and process["pid"] > 0, f"{name}: invalid PID"
        assert isinstance(process.get("run_nonce"), str) and re.fullmatch("[0-9a-f]{64}", process["run_nonce"]), f"{name}: invalid nonce"
        stamp = process.get("started_unix_ns")
        assert isinstance(stamp, str) and stamp.isascii() and stamp.isdigit() and int(stamp) > 0, f"{name}: invalid start time"
    first, second = reports["native"]["process"], reports["native_new_process"]["process"]
    for key in ["pid", "run_nonce", "started_unix_ns"]:
        assert first[key] != second[key], f"distinct native process evidence required: {key}"


def score(fixture_path, hf_path, native_path, second_path, output):
    setup()
    fixture, fixture_sha = read(fixture_path)
    paths = {"hf":hf_path, "native":native_path, "native_new_process":second_path}
    loaded = {name:read(path) for name,path in paths.items()}
    reports = {name:value for name,(value,_) in loaded.items()}
    digests = {name:digest for name,(_,digest) in loaded.items()}
    validate_report_provenance(reports, paths, digests)
    ids = [i["id"] for i in fixture["items"]]
    assert len(ids) == len(set(ids)) == 11
    rows = {i["id"]: i for i in fixture["items"]}
    tasks = retrieval_tasks(fixture)
    matrices, gpu_matrices, identity = {}, {}, {}
    for name, report in reports.items():
        assert report["fixture_sha256"] == fixture_sha and report["checkpoint_sha256"] == CHECKPOINT_SHA
        assert report["native_sources"] == fixture["native_sources"]
        vectors = {}
        for e in report["embeddings"]:
            key = (e["pass"], e["id"])
            assert key not in vectors and e["id"] in rows and len(e["bits"]) == 4096
            raw = b"".join(b.to_bytes(2, "little") for b in e["bits"])
            assert sha(raw) == e["sha256"]
            vectors[key] = e
        assert set(vectors) == {(p, i) for p in ["forward", "reverse"] for i in ids}
        identity[name] = {i: vectors["forward", i]["bits"] == vectors["reverse", i]["bits"] for i in ids}
        x = torch.tensor([vectors["forward", i]["bits"] for i in ids], device="cuda", dtype=torch.uint16).view(torch.bfloat16).float()
        assert torch.isfinite(x).all() and (x.norm(dim=-1) > 0).all()
        x = F.normalize(x, dim=-1)
        gpu_matrices[name] = x @ x.T
        matrices[name] = gpu_matrices[name].cpu().tolist()
    metrics = {}
    for name, matrix in matrices.items():
        results = []
        for task in tasks:
            query, candidates = task["query"], task["candidates"]
            ranked = sorted(candidates, key=lambda d: (-matrix[ids.index(query)][ids.index(d)], d))
            relevant = set(task["relevant"])
            hardest = next((d for d in ranked if d not in relevant), None)
            gpu_row = gpu_matrices[name][ids.index(query)]
            margins = {positive: (gpu_row[ids.index(positive)] - gpu_row[ids.index(hardest)]).item()
                       for positive in sorted(relevant)} if hardest is not None else {}
            results.append(dict(query=query, mode=task["mode"], ranked=ranked, relevant=sorted(relevant),
                scorable=bool(relevant), unscorable_reason=None if relevant else "no fixed labeled positive in this modality-restricted pool",
                recall_at_1=len(set(ranked[:1]) & relevant)/len(relevant) if relevant else None,
                recall_at_3=len(set(ranked[:3]) & relevant)/len(relevant) if relevant else None,
                top1_relevant=ranked[0] in relevant if relevant and ranked else None,
                hardest_irrelevant=hardest, positive_margins_against_hardest_irrelevant=margins,
                scores={d:matrix[ids.index(query)][ids.index(d)] for d in ranked}))
        metrics[name] = results
    cross_process = {i: next(e["bits"] for e in reports["native"]["embeddings"] if e["id"]==i and e["pass"]=="forward") ==
                        next(e["bits"] for e in reports["native_new_process"]["embeddings"] if e["id"]==i and e["pass"]=="forward") for i in ids}
    save(output, dict(scope="eleven fixed local inputs; no universal threshold or broad quality claim", fixture_sha256=fixture_sha,
        items=fixture["items"], rankings=metrics, cosine_matrices=matrices, matrix_order=ids,
        within_process_byte_identity=identity, native_new_process_byte_identity=cross_process,
        no_coordinate_admission_gate=True, threshold_fitted=False,
        native_hf_top3_overlap={n["query"]+":"+n["mode"]:len(set(n["ranked"][:3]) & set(h["ranked"][:3]))/min(3,len(n["ranked"]))
            for n,h in zip(metrics["native"],metrics["hf"])},
        report_sha256=digests,
        native_process_evidence={name:reports[name]["process"] for name in ["native","native_new_process"]},
        process_evidence_caveat="report metadata plus separately retained launcher exits; not authenticated attestation, vectorized-batch or cross-host evidence",
        performance={name:{"one_time_load_or_bind_ms":r.get("load_ms",r.get("bind_ms")),
            "items":[{k:e[k] for k in ("id","pass","elapsed_ms","prepare_ms","forward_ms","readback_ms","forward_dispatch_ms","readback_wait_ms") if k in e}
                     for e in r["embeddings"]]} for name,r in reports.items()},
        timing_caveat="native forward_dispatch_ms is host dispatch duration and readback_wait_ms may include GPU execution; compare summed elapsed_ms, not dispatch alone"))


def self_test():
    # Metadata-only controls: no setup(), CUDA context, tokenization or models.
    fixture, _ = read(Path(__file__).with_name("wemm_behavior_fixture.json"))
    tasks = retrieval_tasks(fixture)
    assert len(tasks) == 10
    assert all(t["query"] not in t["candidates"] for t in tasks)
    specific = {(t["query"], t["mode"]):t for t in tasks}
    assert specific["attention-query", "text-to-image"]["relevant"] == ["attention-page"]
    assert specific["ferris-query", "text-to-image"]["relevant"] == ["ferris-image"]
    assert specific["attention-page", "image-to-text"]["relevant"] == ["attention-extract"]
    assert specific["ferris-image", "image-to-text"]["relevant"] == ["ferris-short"]
    assert specific["bloom-query", "text-to-image"]["relevant"] == []
    schema = "wemm-behavior-embeddings-v1"
    good = {"hf":dict(schema=schema,engine="HF-CUDA-BF16"),
            "native":dict(schema=schema,engine="native-CUDA-BF16",hf_report_sha256="a"*64,
                          process=dict(pid=11,run_nonce="1"*64,started_unix_ns="100")),
            "native_new_process":dict(schema=schema,engine="native-CUDA-BF16",hf_report_sha256="a"*64,
                          process=dict(pid=12,run_nonce="2"*64,started_unix_ns="200"))}
    paths = {name:Path("/fixture") / name for name in good}
    digests = dict(hf="a"*64,native="b"*64,native_new_process="c"*64)
    validate_report_provenance(good, paths, digests)
    cases = ["both_native_are_hf", "second_native_is_hf", "same_path", "same_digest",
             "same_pid", "same_nonce", "same_start", "missing_process", "wrong_hf_binding"]
    for case in cases:
        report = json.loads(json.dumps(good))
        p, d = dict(paths), dict(digests)
        if case == "both_native_are_hf":
            report["native"]["engine"] = report["native_new_process"]["engine"] = "HF-CUDA-BF16"
        elif case == "second_native_is_hf": report["native_new_process"]["engine"] = "HF-CUDA-BF16"
        elif case == "same_path": p["native_new_process"] = p["native"]
        elif case == "same_digest": d["native_new_process"] = d["native"]
        elif case == "same_pid": report["native_new_process"]["process"]["pid"] = 11
        elif case == "same_nonce": report["native_new_process"]["process"]["run_nonce"] = "1"*64
        elif case == "same_start": report["native_new_process"]["process"]["started_unix_ns"] = "100"
        elif case == "missing_process": del report["native_new_process"]["process"]
        elif case == "wrong_hf_binding": report["native"]["hf_report_sha256"] = "d"*64
        try:
            validate_report_provenance(report, p, d)
        except AssertionError:
            continue
        raise AssertionError(f"provenance negative control passed: {case}")
    print("BEHAVIOR METADATA: 17 checks pass; no GPU execution")


if __name__ == "__main__":
    command, *args = sys.argv[1:]
    args = list(map(Path, args))
    if command == "prepare" and len(args) == 4:
        prepare(*args)
    elif command == "reference" and len(args) == 4:
        reference(*args)
    elif command == "score" and len(args) == 5:
        score(*args)
    elif command == "self-test" and not args:
        self_test()
    else:
        raise SystemExit(__doc__)
