"""Inkling 2 Mini saved-image fixture: published processor and vision tower, CPU only.

Run inside vllm/vllm-openai:v0.26.0 (torch 2.11.0, torchvision 0.26.0, PIL 12.3.0)
with transformers 046bdb8 ("Add Inkling2 Mini") on PYTHONPATH. See run.sh.

Usage: make_fixture.py MODEL_DIR OUT_DIR FIXTURE_DIR
"""

import hashlib
import json
import math
import sys

import numpy as np
import torch
from PIL import Image
from safetensors import safe_open

from transformers.models.inkling2_loop.configuration_inkling2_loop import Inkling2LoopConfig
from transformers.models.inkling2.image_processing_inkling2 import Inkling2ImageProcessor
from transformers.models.inkling2.modeling_inkling2 import Inkling2VisionModel
from transformers.vision_utils import get_vision_interpolation_indices_and_weights, get_vision_position_ids

model_dir, out_dir, fix_dir = sys.argv[1:4]
torch.set_num_threads(4)


def sha(b):
    return hashlib.sha256(b).hexdigest()


def f32_digest(t):
    a = np.ascontiguousarray(t.detach().to(torch.float32).cpu().numpy()).astype("<f4")
    return sha(a.tobytes()), a


def file_sha(p):
    with open(p, "rb") as f:
        return sha(f.read())


# 1. Two deterministic synthetic images, saved as PNG and read back.
def rgb_image():
    w, h = 100, 60
    a = np.zeros((h, w, 3), np.uint8)
    for y in range(h):
        for x in range(w):
            a[y, x] = ((3 * x + 2 * y) % 256, (x * y) % 256, (255 - 4 * x + 5 * y) % 256)
    return Image.fromarray(a, "RGB")


def rgba_image():
    w, h = 52, 44
    a = np.zeros((h, w, 4), np.uint8)
    for y in range(h):
        for x in range(w):
            a[y, x] = ((5 * x) % 256, (7 * y) % 256, 128, (5 * x + 3 * y) % 256)
    return Image.fromarray(a, "RGBA")


images = []
for name, img, formula in [
    ("rgb_100x60", rgb_image(), "RGB, (x,y): R=(3x+2y)%256, G=(x*y)%256, B=(255-4x+5y)%256"),
    ("rgba_52x44", rgba_image(), "RGBA, (x,y): R=(5x)%256, G=(7y)%256, B=128, A=(5x+3y)%256"),
]:
    path = f"{fix_dir}/inkling2_mini_{name}.png"
    img.save(path, format="PNG", compress_level=9, optimize=False)
    back = Image.open(path)
    back.load()
    raw = np.asarray(back)
    images.append(
        {
            "name": name,
            "file": path.rsplit("/", 1)[1],
            "formula": formula,
            "width": back.width,
            "height": back.height,
            "mode": back.mode,
            "png_sha256": file_sha(path),
            "raw_sha256": sha(raw.tobytes()),
            "pil": back,
            "raw": raw,
        }
    )

# 2. The published processor, configured from the checkpoint's processor_config.json.
with open(f"{model_dir}/processor_config.json") as f:
    pcfg = json.load(f)
ip_kwargs = {k: v for k, v in pcfg["image_processor"].items() if k != "image_processor_type"}
proc = Inkling2ImageProcessor(**ip_kwargs)
batch = proc([im["pil"] for im in images], return_tensors="pt")
pv, thw = batch["pixel_values"], batch["image_grid_thw"]
pv_sha, pv_np = f32_digest(pv)
pv_np.tofile(f"{out_dir}/pixel_values.f32")

PAD = (-1.0 / 255.0 - 0.5) / 0.5


def ref_value(im, py, px, c):
    """Expected processor value at image pixel (py, px), channel c, from the raw bytes."""
    if py >= im["height"] or px >= im["width"]:
        return PAD
    if im["mode"] == "RGBA":
        r = im["raw"][py, px].astype(np.int64)
        bg = 204 if ((py // 8) + (px // 8)) % 2 == 0 else 153
        v = (int(r[c]) * int(r[3]) + bg * (255 - int(r[3])) + 127) // 255
    else:
        v = int(im["raw"][py, px, c])
    return (v / 255.0 - 0.5) / 0.5


selected, token_base, mismatches = [], 0, 0
probes = [(0, 0, 0), (0, 0, 2), (39, 39, 1), (17, 23, 0), (5, 25, 2), (25, 5, 1), (39, 0, 0)]
for im, g in zip(images, thw.tolist()):
    _, rows, cols = g
    for r in range(rows):
        for c in range(cols):
            tok = token_base + r * cols + c
            for (y, x, ch) in probes:
                py, px = r * 40 + y, c * 40 + x
                got = float(pv[tok, y, x, ch])
                want = ref_value(im, py, px, ch)
                if abs(got - want) > 1e-6:
                    mismatches += 1
                selected.append(
                    {"image": im["name"], "token": tok, "y": y, "x": x, "c": ch, "pixel_y": py,
                     "pixel_x": px, "padded": py >= im["height"] or px >= im["width"], "value": got}
                )
    token_base += rows * cols
assert mismatches == 0, f"{mismatches} probe values disagree with the raw-byte formula"

# 3. Weight-free vision-tower input contract: 2x2 native 20px patches per token, positions.
vcfg = Inkling2LoopConfig.from_pretrained(model_dir).vision_config
assert type(vcfg).__name__ == "Inkling2VisionConfig", type(vcfg)
merge, patch = vcfg.spatial_merge_size, vcfg.patch_size
native = pv.reshape(-1, merge, patch, merge, patch, 3).transpose(2, 3).reshape(-1, 3 * patch * patch)
native_sha, native_np = f32_digest(native)
patch_grid_thw = thw * torch.tensor([1, merge, merge])
ii, iw = get_vision_interpolation_indices_and_weights(
    patch_grid_thw, num_grid_per_side=vcfg.posemb_grid_size, mode="bilinear", align_corners=False,
    spatial_merge_size=merge,
)
pos_ids = get_vision_position_ids(patch_grid_thw, merge)
native_rows = [0, 1, 2, 3, 4, 23, 24, native.shape[0] - 1]
native_cols = [0, 1, 2, 3, 600, 1199]

# 4. CPU vision-tower forward from the checkpoint's BF16 visual tensors.
renames = [
    ("model.visual.backbone.embeddings.position_embedding", "embeddings.position_embedding.weight"),
    ("model.visual.backbone.embeddings.", "embeddings."),
    ("model.visual.backbone.ln_pre.", "pre_layernorm."),
    ("model.visual.backbone.layers.", "layers."),
    ("model.visual.proj_in.", "merger.linear_1."),
    ("model.visual.proj_out.", "merger.linear_2."),
]
layer_renames = [(".ln_1.", ".layernorm_before."), (".ln_2.", ".layernorm_after."), (".mlp.c_fc.", ".mlp.fc1."),
                 (".mlp.c_proj.", ".mlp.fc2."), (".ls_1.gamma", ".lambda_1"), (".ls_2.gamma", ".lambda_2")]
state, src_keys = {}, 0
with safe_open(f"{model_dir}/model-00001-of-00001.safetensors", framework="pt") as f:
    for k in f.keys():
        if not k.startswith("model.visual."):
            continue
        src_keys += 1
        n = k
        for a, b in renames:
            if n.startswith(a):
                n = b + n[len(a):]
                break
        for a, b in layer_renames:
            n = n.replace(a, b)
        state[n] = f.get_tensor(k)
assert all(t.dtype == torch.bfloat16 for t in state.values())

outputs = {}
for dtype_name, dtype in [("float32", torch.float32), ("bfloat16", torch.bfloat16)]:
    vcfg._attn_implementation = "eager"
    model = Inkling2VisionModel(vcfg).eval()
    missing, unexpected = model.load_state_dict({k: v.to(dtype) for k, v in state.items()}, strict=False)
    model = model.to(dtype)
    assert not missing and not unexpected, (missing, unexpected)
    with torch.no_grad():
        out = model(pv, thw)
    po = out.pooler_output
    po_sha, po_np = f32_digest(po)
    po_np.tofile(f"{out_dir}/pooler_output_{dtype_name}.f32")
    rows_sel = [0, 1, 5, 6, po.shape[0] - 1]
    cols_sel = [0, 1, 2, 1151, 2303]
    outputs[dtype_name] = {
        "pooler_output_shape": list(po.shape),
        "pooler_output_f32_sha256": po_sha,
        "last_hidden_state_shape": list(out.last_hidden_state.shape),
        "selected": [{"row": r, "col": c, "value": float(po[r, c])} for r in rows_sel for c in cols_sel],
        "row_l2": [float(x) for x in po.float().norm(dim=-1)],
        "sum": float(po.float().double().sum()),
        "sum_sq": float((po.float().double() ** 2).sum()),
    }
    del model

receipt = {
    "what": "Inkling 2 Mini saved-image fixture: published processor and vision tower on CPU",
    "checkpoint": {"source": "thinkingmachines/Inkling-2-Mini", "revision": "b85c3cfa615fc413cc9af3379f491b963f3b586a",
                   "dir": model_dir, "processor_config_sha256": file_sha(f"{model_dir}/processor_config.json"),
                   "config_sha256": file_sha(f"{model_dir}/config.json"),
                   "visual_tensors": src_keys},
    "reference": {"transformers_commit": "046bdb8cc2cee29d233964bb77fe6fd37cfe4aa3",
                  "image_processing_inkling2_sha256": file_sha(sys.modules[Inkling2ImageProcessor.__module__].__file__),
                  "modeling_inkling2_sha256": file_sha(sys.modules[Inkling2VisionModel.__module__].__file__),
                  "torch": torch.__version__, "numpy": np.__version__,
                  "pil": Image.__version__ if hasattr(Image, "__version__") else None,
                  "processor_kwargs": ip_kwargs},
    "images": [{k: v for k, v in im.items() if k not in ("pil", "raw")} for im in images],
    "processor": {
        "pixel_values_shape": list(pv.shape),
        "pixel_values_dtype": str(pv.dtype),
        "pixel_values_layout": "[token][y 0..40][x 0..40][rgb], tokens raster-ordered per image, images concatenated",
        "pixel_values_f32_sha256": pv_sha,
        "image_grid_thw": thw.tolist(),
        "image_tokens": [int(math.prod(g)) for g in thw.tolist()],
        "pad_value": PAD,
        "probe_mismatches_vs_raw_formula": mismatches,
        "selected": selected,
        "sum": float(pv.double().sum()),
    },
    "vision_input": {
        "native_patch_size": patch,
        "spatial_merge_size": merge,
        "native_patches": int(native.shape[0]),
        "patch_embedding_input_shape": list(native.shape),
        "patch_embedding_input_layout": "token-major, 2x2 native patches in (my, mx) order, each [y 0..20][x 0..20][rgb]",
        "patch_embedding_input_f32_sha256": native_sha,
        "selected": [{"row": r, "col": c, "value": float(native[r, c])} for r in native_rows for c in native_cols],
        "patch_grid_thw": patch_grid_thw.tolist(),
        "interp_indices_first8": ii[:8].tolist(),
        "interp_weights_first8": iw[:8].tolist(),
        "interp_indices_sha256": sha(np.ascontiguousarray(ii.numpy()).astype("<i8").tobytes()),
        "interp_weights_f32_sha256": f32_digest(iw)[0],
        "position_ids_first8": pos_ids[:8].tolist(),
        "position_ids_shape": list(pos_ids.shape),
        "sequence": "per image: one class token at position (0,0), then its patches at 1-indexed (row, col)",
    },
    "vision_output": outputs,
}
with open(f"{fix_dir}/inkling2_mini_vision_reference.json", "w") as f:
    json.dump(receipt, f, indent=1)
print(json.dumps({"pixel_values": receipt["processor"]["pixel_values_shape"], "thw": thw.tolist(),
                  "native": list(native.shape), "pooler": outputs["float32"]["pooler_output_shape"],
                  "f32_vs_bf16_max_abs": float(np.abs(np.fromfile(f"{out_dir}/pooler_output_float32.f32", "<f4")
                                                       - np.fromfile(f"{out_dir}/pooler_output_bfloat16.f32", "<f4")).max())}))
