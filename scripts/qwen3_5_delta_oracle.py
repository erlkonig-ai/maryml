#!/usr/bin/env python3
"""CPU-only tiny Qwen3.5 DeltaNet fixtures; no model loading or GPU execution.

Extracts just l2norm and the two Torch fallback functions from the SHA-pinned
installed Transformers 5.2.0 source. The module body (including optional FLA
imports) is never executed. Optional --rust-binary compares the standalone
delta_reference.rs runner; this script does not build it.

Example gate, after the owning agent releases its resource reservation:
  ../venv-wemm/bin/python scripts/qwen3_5_delta_oracle.py \
      --rust-binary /explicit/disposable/path/delta-reference --dtype float32
Repeat with --dtype bfloat16 for the dtype-boundary check. This is a scalar
recurrence oracle, not a resident GPU implementation or a full-model gate.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


SOURCE_SHA256 = "d1ae3856f53763591ec65054129af46e003a1715efcf7eef2b752a31e85526b8"
SOURCE_COMMIT = "7d9754a05193eb79b1d86aa744b622b8068008cd"
SOURCE_URL = (
    "https://github.com/huggingface/transformers/blob/"
    + SOURCE_COMMIT
    + "/src/transformers/models/qwen3_5/modeling_qwen3_5.py"
)
FUNCTIONS = {
    "l2norm": (317, 320),
    "torch_chunk_gated_delta_rule": (323, 400),
    "torch_recurrent_gated_delta_rule": (403, 442),
}


def load_reference(path, torch):
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != SOURCE_SHA256:
        raise ValueError(f"source identity changed: expected {SOURCE_SHA256}, got {digest}")
    tree = ast.parse(raw, filename=str(path))
    selected = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in FUNCTIONS]
    if {node.name for node in selected} != set(FUNCTIONS):
        raise ValueError("missing pinned reference function")
    lines = {node.name: [node.lineno, node.end_lineno] for node in selected}
    if any(tuple(lines[name]) != expected for name, expected in FUNCTIONS.items()):
        raise ValueError(f"unexpected reference line ranges: {lines}")
    namespace = {"torch": torch, "F": torch.nn.functional}
    # Execute only the inspected pure tensor function definitions, not imports
    # or the parent model. Pin validation occurs before this restricted extract.
    module = ast.Module(body=selected, type_ignores=[])
    exec(compile(module, str(path), "exec"), namespace)
    return namespace, lines


def fixture(name, shape, values, normalize=True):
    return {"name": name, "shape": shape, "normalize_qk": normalize, **values}


def make_fixtures(torch, dtype):
    def tensor(values, shape, target_dtype=dtype):
        return torch.tensor(values, dtype=target_dtype, device="cpu").reshape(shape)

    yield fixture("decay_before_prediction", [1, 1, 1, 1, 1, 1], {
        "query": tensor([2], [1, 1, 1, 1]),
        "key": tensor([3], [1, 1, 1, 1]),
        "value": tensor([5], [1, 1, 1, 1]),
        "log_decay": tensor([-0.6931471805599453], [1, 1, 1], torch.float32),
        "beta": tensor([0.25], [1, 1, 1]),
        "initial_state": tensor([2], [1, 1, 1, 1], torch.float32),
    }, normalize=False)
    yield fixture("adjacent_grouped_heads", [1, 1, 2, 4, 1, 1], {
        "query": tensor([1, 10], [1, 1, 2, 1]),
        "key": tensor([1, 2], [1, 1, 2, 1]),
        "value": tensor([1, 2, 3, 4], [1, 1, 4, 1]),
        "log_decay": torch.zeros(1, 1, 4, dtype=torch.float32, device="cpu"),
        "beta": torch.ones(1, 1, 4, dtype=dtype, device="cpu"),
        "initial_state": None,
    }, normalize=False)
    yield fixture("epsilon_inside_root", [1, 1, 1, 1, 2, 1], {
        "query": tensor([0, 0.001], [1, 1, 1, 2]),
        "key": tensor([0, 0.001], [1, 1, 1, 2]),
        "value": tensor([2], [1, 1, 1, 1]),
        "log_decay": torch.zeros(1, 1, 1, dtype=torch.float32, device="cpu"),
        "beta": torch.ones(1, 1, 1, dtype=dtype, device="cpu"),
        "initial_state": None,
    })
    generator = torch.Generator(device="cpu").manual_seed(3509)
    for name, shape, initial in [
        ("grouped_nonzero_state", [2, 9, 2, 6, 4, 3], True),
        ("crosses_chunk_64", [1, 67, 1, 2, 3, 2], False),
    ]:
        batch, steps, key_heads, value_heads, key_dim, value_dim = shape
        def random(dims):
            return torch.randn(*dims, generator=generator, device="cpu", dtype=torch.float32)
        gates = (batch, steps, value_heads)
        yield fixture(name, shape, {
            "query": random((batch, steps, key_heads, key_dim)).to(dtype),
            "key": random((batch, steps, key_heads, key_dim)).to(dtype),
            "value": random((batch, steps, value_heads, value_dim)).to(dtype),
            "log_decay": -random(gates).abs() * 0.25,
            "beta": random(gates).sigmoid().to(dtype),
            "initial_state": random((batch, value_heads, key_dim, value_dim)) * 0.1 if initial else None,
        })


def flatten(tensor):
    return tensor.detach().float().reshape(-1).tolist()


def run_rust(binary, case, reference, torch, dtype):
    batch, steps, key_heads, value_heads, key_dim, value_dim = case["shape"]
    normalize = case["normalize_qk"]
    query, key = case["query"], case["key"]
    if dtype == torch.bfloat16 and normalize:
        # HF l2norm precedes its cast to f32. Preserve the real BF16 intermediate
        # rounding; do not pretend Rust's f32 normalization is a BF16 reference.
        query = reference["l2norm"](query)
        key = reference["l2norm"](key)
        normalize = False
    initial = case["initial_state"]
    words = [*case["shape"], int(normalize), int(initial is not None)]
    for value in [query, key, case["value"], case["log_decay"], case["beta"], initial]:
        if value is not None:
            words.extend(flatten(value))
    completed = subprocess.run(
        [str(binary)], input=" ".join(map(str, words)), text=True,
        capture_output=True, check=True, timeout=30,
    )
    lines = completed.stdout.splitlines()
    if len(lines) != 2 or not lines[0].startswith("output ") or not lines[1].startswith("state "):
        raise ValueError(f"unexpected Rust output: {completed.stdout[:200]!r}")
    output, state = [
        torch.tensor([float(word) for word in line.split()[1:]], dtype=torch.float32, device="cpu")
        for line in lines
    ]
    return (
        output.reshape(batch, steps, value_heads, value_dim).to(dtype),
        state.reshape(batch, value_heads, key_dim, value_dim),
    )


def compare(name, actual, expected, atol, rtol, torch):
    if actual.shape != expected.shape or actual.device.type != "cpu" or expected.device.type != "cpu":
        raise ValueError(f"{name}: invalid comparison shape or device")
    actual, expected = actual.float(), expected.float()
    difference = (actual - expected).abs()
    return {
        "name": name, "atol": atol, "rtol": rtol,
        "max_abs": difference.max().item(),
        "max_relative_with_floor_1e-8": (difference / expected.abs().clamp_min(1e-8)).max().item(),
        "passed": bool(torch.allclose(actual, expected, atol=atol, rtol=rtol)),
    }


def evaluate(case, reference, binary, torch, dtype):
    batch, steps, key_heads, value_heads, key_dim, value_dim = case["shape"]
    repeats = value_heads // key_heads
    query = case["query"].repeat_interleave(repeats, dim=2)
    key = case["key"].repeat_interleave(repeats, dim=2)
    common = {
        "query": query, "key": key, "value": case["value"],
        "g": case["log_decay"], "beta": case["beta"],
        "initial_state": case["initial_state"], "output_final_state": True,
        "use_qk_l2norm_in_kernel": case["normalize_qk"],
    }
    recurrent = reference["torch_recurrent_gated_delta_rule"](**common)
    chunked = reference["torch_chunk_gated_delta_rule"](**common, chunk_size=64)
    # These are proposed oracle gate tolerances, not full-model acceptance bounds.
    # BF16 output cast can straddle a representable value even when f32 states
    # agree; state is still tested tightly as f32 in both modes.
    output_atol, output_rtol = (1e-5, 1e-4) if dtype == torch.float32 else (1 / 128, 1 / 64)
    checks = [
        compare("chunk_vs_recurrent_output", chunked[0], recurrent[0], output_atol, output_rtol, torch),
        compare("chunk_vs_recurrent_state", chunked[1], recurrent[1], 5e-5, 1e-4, torch),
    ]
    if steps > 1:
        cut = steps // 2
        state = case["initial_state"]
        outputs = []
        for start, stop in [(0, cut), (cut, steps)]:
            part = {
                name: value[:, start:stop].contiguous()
                for name, value in common.items() if name in {"query", "key", "value", "g", "beta"}
            }
            out, state = reference["torch_recurrent_gated_delta_rule"](
                **part, initial_state=state, output_final_state=True,
                use_qk_l2norm_in_kernel=case["normalize_qk"],
            )
            outputs.append(out)
        checks.extend([
            compare("split_vs_whole_output", torch.cat(outputs, dim=1), recurrent[0], output_atol, output_rtol, torch),
            compare("split_vs_whole_state", state, recurrent[1], 5e-5, 1e-4, torch),
        ])
    rust = None
    if binary:
        rust = run_rust(binary, case, reference, torch, dtype)
        checks.extend([
            compare("rust_vs_recurrent_output", rust[0], recurrent[0], output_atol, output_rtol, torch),
            compare("rust_vs_recurrent_state", rust[1], recurrent[1], 5e-5, 1e-4, torch),
        ])
    record = {
        "name": case["name"], "shape_B_T_Hk_Hv_K_V": case["shape"],
        "normalize_qk": case["normalize_qk"],
        "inputs": {
            name: None if case[name] is None else flatten(case[name])
            for name in ("query", "key", "value", "log_decay", "beta", "initial_state")
        },
        "torch_recurrent_output": flatten(recurrent[0]), "torch_recurrent_state": flatten(recurrent[1]),
        "torch_chunk_output": flatten(chunked[0]), "torch_chunk_state": flatten(chunked[1]),
        "rust_output": None if rust is None else flatten(rust[0]),
        "rust_state": None if rust is None else flatten(rust[1]),
        "checks": checks,
    }
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    default_source = Path(__file__).resolve().parents[2] / (
        "venv-wemm/lib/python3.12/site-packages/transformers/models/qwen3_5/modeling_qwen3_5.py"
    )
    parser.add_argument("--source", type=Path, default=default_source)
    parser.add_argument("--rust-binary", type=Path)
    parser.add_argument("--dtype", choices=("float32", "bfloat16"), default="float32")
    parser.add_argument("--output", type=Path, help="JSON fixture/report path; otherwise stdout")
    args = parser.parse_args()
    # Hide GPUs before torch import. All factory calls also request device=cpu.
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    dtype = getattr(torch, args.dtype)
    reference, lines = load_reference(args.source, torch)
    binary = args.rust_binary.resolve() if args.rust_binary else None
    with torch.inference_mode():
        cases = [evaluate(case, reference, binary, torch, dtype) for case in make_fixtures(torch, dtype)]
    passed = all(check["passed"] for case in cases for check in case["checks"])
    report = {
        "purpose": "scalar recurrence oracle only; no resident GPU runtime coverage",
        "passed": passed, "device": "cpu", "dtype": args.dtype, "torch_version": torch.__version__,
        "source_path": str(args.source.resolve()), "source_sha256": SOURCE_SHA256,
        "source_url": SOURCE_URL, "reference_function_lines": lines,
        "grouped_head_lines": [587, 589], "state_layout": "B,Hv,K,V contiguous; V fastest; FP32",
        "rust_binary": None if binary is None else str(binary),
        "rust_binary_sha256": None if binary is None else hashlib.sha256(binary.read_bytes()).hexdigest(),
        "bf16_rule": "normalize Q/K in BF16 before promoting state math; cast output to BF16",
        "cases": cases,
    }
    rendered = json.dumps(report, indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.write_text(rendered)
    else:
        sys.stdout.write(rendered)
    print(f"{len(cases)} CPU fixtures: {'PASS' if passed else 'FAIL'}; Rust compared={binary is not None}", file=sys.stderr)
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
