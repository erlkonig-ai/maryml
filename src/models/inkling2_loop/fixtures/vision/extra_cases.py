"""Extra Inkling 2 Mini processor cases (processor only, no weights): exact-multiple grids and
PIL conversion paths the two saved fixtures do not exercise. Same container and transformers
046bdb8 as mini-vision-fixture-20261123/scripts/make_fixture.py. Usage: extra_cases.py MODEL_DIR OUT_DIR"""
import hashlib, json, sys
import numpy as np
from PIL import Image
from transformers.models.inkling2.image_processing_inkling2 import Inkling2ImageProcessor

model_dir, out = sys.argv[1:3]
pcfg = json.load(open(f"{model_dir}/processor_config.json"))
proc = Inkling2ImageProcessor(**{k: v for k, v in pcfg["image_processor"].items() if k != "image_processor_type"})

def grid(w, h, ch, f):
    a = np.zeros((h, w, ch), np.uint8)
    for y in range(h):
        for x in range(w):
            a[y, x] = f(x, y)
    return a

cases = []
cases.append(("rgb_80x40", Image.fromarray(grid(80, 40, 3, lambda x, y: ((7*x+y) % 256, (3*y+x*x) % 256, (11*x+13*y) % 256)), "RGB")))
cases.append(("rgba_40x80", Image.fromarray(grid(40, 80, 4, lambda x, y: ((9*x) % 256, (5*y+x) % 256, (x*y) % 256, (17*x+29*y) % 256)), "RGBA")))
cases.append(("la_33x47", Image.fromarray(grid(33, 47, 2, lambda x, y: ((6*x+4*y) % 256, (19*x+7*y) % 256)), "LA")))
p = Image.fromarray(grid(30, 20, 1, lambda x, y: ((x // 3 + y // 4) % 6,))[:, :, 0], "P")
p.putpalette([0, 0, 0, 255, 0, 0, 0, 255, 0, 0, 0, 255, 200, 100, 50, 10, 20, 30] + [0] * (768 - 18))
p.info["transparency"] = bytes([255, 0, 128, 255, 40, 255])
cases.append(("p_trns_30x20", p))
cases.append(("l_45x41", Image.fromarray(grid(45, 41, 1, lambda x, y: ((5*x+3*y) % 256,))[:, :, 0], "L")))
receipt = []
for name, img in cases:
    path = f"{out}/{name}.png"
    img.save(path, format="PNG", compress_level=9, optimize=False)
    back = Image.open(path); back.load()
    b = proc([back], return_tensors="pt")
    pv = np.ascontiguousarray(b["pixel_values"].numpy()).astype("<f4")
    pv.tofile(f"{out}/{name}.pixel_values.f32")
    receipt.append({"name": name, "file": f"{name}.png", "mode": back.mode, "width": back.width, "height": back.height,
                    "has_transparency_data": bool(back.has_transparency_data),
                    "png_sha256": hashlib.sha256(open(path, "rb").read()).hexdigest(),
                    "image_grid_thw": b["image_grid_thw"].tolist()[0], "pixel_values_shape": list(pv.shape),
                    "pixel_values_f32_sha256": hashlib.sha256(pv.tobytes()).hexdigest()})
json.dump(receipt, open(f"{out}/extra_cases.json", "w"), indent=1)
print(json.dumps(receipt, indent=1))
