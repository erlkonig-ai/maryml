"""Run transformers' own Inkling2LoopDequantizeExperts on a deterministic case.

The class source is taken verbatim from the pinned core_model_loading.py
(transformers 046bdb8cc2cee29d233964bb77fe6fd37cfe4aa3) and executed with a
stub base class, so the arithmetic under test is upstream's, not a re-typing.
Output dtype is float32 (a stand-in model whose parameter is float32), which
is the value before upstream's cast to the model dtype.

usage: python -I golden/capture_inkling2_dequant.py <core_model_loading.py> \
       src/models/inkling2_loop/fixtures/dequant_reference.json
"""
import ast, json, sys, hashlib
import torch

src_path, out_path = sys.argv[1], sys.argv[2]
src = open(src_path).read()
tree = ast.parse(src)
cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Inkling2LoopDequantizeExperts")
cls_src = ast.get_source_segment(src, cls)
ns = {"torch": torch, "ConversionOps": object}
exec(cls_src, ns)
Op = ns["Inkling2LoopDequantizeExperts"]

class F32Model:
    def get_parameter(self, name):
        return torch.empty(0, dtype=torch.float32)

def lcg(seed):
    x = seed
    while True:
        x = (x * 6364136223846793005 + 1442695040888963407) % (1 << 64)
        yield x >> 56

def bytes_cover(n, seed):
    """n bytes: every value 0..255 once (when n >= 256), then pseudo-random."""
    g = lcg(seed)
    base = list(range(256))
    # deterministic shuffle
    for i in range(255, 0, -1):
        j = next(g) % (i + 1)
        base[i], base[j] = base[j], base[i]
    out = (base * (n // 256 + 1))[:n] if n >= 256 else [next(g) for _ in range(n)]
    return out

def run(sources):
    op = Op()
    out = op.convert(sources, ["w", "s", "a", "d"], ["target"], model=F32Model(), full_layer_name="x")
    return out["target"]

def bits(t):
    return [int(v) for v in t.contiguous().view(torch.int32).flatten().tolist()]

# NVFP4: 3 experts, 4 rows, 64 logical columns (32 code bytes, 4 scales per row).
E, R, C = 3, 4, 64
codes = bytes_cover(E * R * C // 2, 1)
# E4M3 block scales: positive finite codes only (0x01..0x7E), incl. subnormals.
g = lcg(7)
scales = [1 + next(g) % 0x7E for _ in range(E * R * C // 16)]
scales[0], scales[1], scales[2] = 0x01, 0x07, 0x7E  # min subnormal, max subnormal, max finite
amax = [2.75, 0.03125, 1337.5]
den = [2688.0]
nv = run({
    "w": torch.tensor(codes, dtype=torch.uint8).view(E, R, C // 2),
    "s": torch.tensor(scales, dtype=torch.uint8).view(E, R, C // 16).view(torch.float8_e4m3fn),
    "a": torch.tensor(amax, dtype=torch.float32),
    "d": torch.tensor(den, dtype=torch.float32),
})
assert nv.dtype == torch.float32 and tuple(nv.shape) == (E, R, C)

# MXFP8: 2 experts, 4 rows, 64 columns (2 scales per row). Every E4M3 byte
# appears, NaNs included; scales span 0..254 with the extremes present.
E2, R2, C2 = 2, 4, 64
vals = bytes_cover(E2 * R2 * C2, 3)
g = lcg(11)
mscales = [100 + next(g) % 55 for _ in range(E2 * R2 * C2 // 32)]
mscales[0], mscales[1], mscales[2], mscales[3] = 0, 1, 127, 254
mx = run({
    "w": torch.tensor(vals, dtype=torch.uint8).view(E2, R2, C2).view(torch.float8_e4m3fn),
    "s": torch.tensor(mscales, dtype=torch.uint8).view(E2, R2, C2 // 32),
})
assert mx.dtype == torch.float32 and tuple(mx.shape) == (E2, R2, C2)

doc = {
    "source": "transformers 046bdb8cc2cee29d233964bb77fe6fd37cfe4aa3 core_model_loading.py Inkling2LoopDequantizeExperts",
    "source_class_sha256": hashlib.sha256(cls_src.encode()).hexdigest(),
    "torch": torch.__version__,
    "nvfp4": {"experts": E, "rows": R, "cols": C, "codes": codes, "scales": scales,
              "amax_bits": bits(torch.tensor(amax, dtype=torch.float32)),
              "denominator_bits": bits(torch.tensor(den, dtype=torch.float32)),
              "out_bits": bits(nv)},
    "mxfp8": {"experts": E2, "rows": R2, "cols": C2, "values": vals, "scales": mscales,
              "out_bits": bits(mx)},
}
json.dump(doc, open(out_path, "w"), separators=(",", ":"))
print("wrote", out_path, "nvfp4", tuple(nv.shape), "mxfp8", tuple(mx.shape),
      "nan", int(torch.isnan(mx).sum()), "inf", int(torch.isinf(mx).sum()))
