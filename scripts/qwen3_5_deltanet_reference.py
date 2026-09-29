#!/usr/bin/env python3
"""Small CUDA-only fixtures from the pinned Transformers Qwen3.5 operators.

Run with the reference venv, then pass the JSON to qwen3_5_deltanet_gate.
All recurrence/normalization/gate arithmetic runs on CUDA. CPU transfers here
only serialize the fixture at the test boundary; there is no CPU oracle.
"""

import argparse
import hashlib
import inspect
import json
from pathlib import Path

import torch
import transformers
from transformers.models.qwen3_5 import modeling_qwen3_5 as reference

REFERENCE_SHA256 = "d1ae3856f53763591ec65054129af46e003a1715efcf7eef2b752a31e85526b8"


def bf16_bits(tensor):
    return tensor.contiguous().view(torch.int16).cpu().reshape(-1).to(torch.int32).bitwise_and(65535).tolist()


def floats(tensor):
    return tensor.contiguous().float().cpu().reshape(-1).tolist()


def make_case(name, dims, seed, initial):
    batch, tokens, key_heads, value_heads, key_dim, value_dim = dims
    generator = torch.Generator(device="cuda").manual_seed(seed)

    def rand(shape, scale=1.0):
        return (torch.randn(shape, generator=generator, device="cuda") * scale).to(torch.bfloat16)

    q = rand((batch, tokens, key_heads, key_dim))
    k = rand(q.shape)
    v = rand((batch, tokens, value_heads, value_dim))
    a = rand((batch, tokens, value_heads), 0.6)
    b = rand(a.shape)
    a_log = rand((value_heads,), 0.5)
    dt_bias = rand((value_heads,), 0.3)
    # Exercise the epsilon-dominated norm and stable positive softplus branch.
    q[:, 0, 0] = 0
    k[:, 0, 0] = 0
    a[:, -1, -1] = 24
    # A tiny softplus can be amplified by exp(A_log): log(1+exp(-20))
    # incorrectly rounds to zero in F32, while log1p preserves the decay.
    a_log[0] = 20
    dt_bias[0] = 0
    a[:, :, 0] = -20
    dt_bias[1] = 0
    a[:, 2, 1] = 19.875
    a[:, 3, 1] = 20
    a[:, 4, 1] = 20.125
    b[:, 2, 1] = -80
    b[:, 3, 1] = 80
    state = torch.randn((batch, value_heads, key_dim, value_dim), generator=generator, device="cuda") * 0.1 if initial else None
    beta = b.sigmoid()
    g = -a_log.float().exp() * torch.nn.functional.softplus(a.float() + dt_bias)
    repeats = value_heads // key_heads
    qr = q.repeat_interleave(repeats, dim=2)
    kr = k.repeat_interleave(repeats, dim=2)
    kwargs = dict(initial_state=state, output_final_state=True, use_qk_l2norm_in_kernel=True)
    out, final = reference.torch_recurrent_gated_delta_rule(qr, kr, v, g, beta, **kwargs)
    chunk_out, chunk_final = reference.torch_chunk_gated_delta_rule(qr, kr, v, g, beta, chunk_size=4, **kwargs)
    return dict(
        name=name, dims=dims, seed=seed,
        query=bf16_bits(q), key=bf16_bits(k), value=bf16_bits(v),
        a=bf16_bits(a), b=bf16_bits(b), a_log=bf16_bits(a_log), dt_bias=bf16_bits(dt_bias),
        initial_state=floats(state) if state is not None else None,
        recurrent_output=bf16_bits(out), recurrent_state=floats(final),
        chunk_output=bf16_bits(chunk_out), chunk_state=floats(chunk_final),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA reference required; CPU fallback is forbidden")
    if transformers.__version__ != "5.2.0":
        raise RuntimeError(f"expected Transformers 5.2.0, got {transformers.__version__}")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    source = Path(inspect.getsourcefile(reference))
    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    if source_hash != REFERENCE_SHA256:
        raise RuntimeError(f"pinned Qwen3.5 source hash mismatch: {source_hash}")
    result = dict(
        transformers=transformers.__version__, torch=torch.__version__,
        device=torch.cuda.get_device_name(), reference_path=str(source),
        reference_sha256=source_hash,
        cases=[
            make_case("small_zero", [3, 9, 2, 4, 8, 7], 171, False),
            make_case("small_initial", [3, 9, 2, 4, 8, 7], 172, True),
            make_case("qwen_width", [2, 7, 16, 32, 128, 128], 173, True),
        ],
    )
    args.output.write_text(json.dumps(result, allow_nan=False) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "cases"}))
    print(f"wrote {len(result['cases'])} CUDA cases to {args.output}")


if __name__ == "__main__":
    main()
