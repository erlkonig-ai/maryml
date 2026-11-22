"""Read-only: a few KB of expert scale/sidecar bytes from the real checkpoint.

usage: python3 -I golden/probe_inkling2_scales.py <dir>/model-00001-of-00001.safetensors
"""
import json, struct, sys
path = sys.argv[1]
f = open(path, "rb")
n = struct.unpack("<Q", f.read(8))[0]
h = json.loads(f.read(n)); base = 8 + n
def raw(name, start=0, length=None):
    t = h[name]; a, b = t["data_offsets"]
    f.seek(base + a + start); return f.read((b - a - start) if length is None else length)
for L in (20, 23):
    for m in ("w13", "w2"):
        name = f"model.llm.layers.{L}.mlp.experts.{m}_weight_scale"
        E, R, C = h[name]["shape"]
        # experts 0 and 127 entirely
        rep_ok = rep_bad = 0; ff = 0; lo, hi = 255, 0
        for e in (0, E - 1):
            blob = raw(name, e * R * C, R * C)
            rows = [blob[r * C:(r + 1) * C] for r in range(R)]
            for g in range(0, R, 32):
                if all(rows[g + i] == rows[g] for i in range(32)): rep_ok += 1
                else: rep_bad += 1
            ff += blob.count(0xFF); lo = min(lo, min(blob)); hi = max(hi, max(blob))
        print(f"L{L} {m} scale [{E},{R},{C}] 32-row groups identical {rep_ok}/{rep_ok+rep_bad} (experts 0,{E-1}); 0xFF count {ff}; byte range {lo}..{hi}")
for L in (0, 19):
    for m in ("w13", "w2"):
        a = struct.unpack("<128f", raw(f"model.llm.layers.{L}.mlp.experts.{m}_weight_amax"))
        d = struct.unpack("<f", raw(f"model.llm.layers.{L}.mlp.experts.{m}_weight_outer_scale_denominator"))[0]
        print(f"L{L} {m} amax min {min(a):.4g} max {max(a):.4g}; denominator {d}")
