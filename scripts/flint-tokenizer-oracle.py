#!/usr/bin/env python3
"""Generate independent fixtures, only after an operator grants the CPU slot.

Usage: python3 scripts/flint-tokenizer-oracle.py MODEL NATIVE_ORACLE NEW_OUTPUT_DIR
No build, backend, model context or inference request is launched by this script.
The native oracle receives only a hash-checked, zero-tensor metadata snapshot.
"""
import hashlib
import json
import os
from pathlib import Path
import resource
import subprocess
import sys

COMMIT = "1ddffe9f70a63e2d260d4bfad9ad200009007f62"
PREFIX_BYTES = 5436996
PREFIX_SHA256 = "e5e5e957b8e7f1a02e3155bcb3899ec50ef534c477a3f8c6e1c18163b2653b88"
MODEL_BYTES = 22863084160

def bounded_child():
    resource.setrlimit(resource.RLIMIT_AS, (1024**3, 1024**3))
    resource.setrlimit(resource.RLIMIT_CPU, (90, 90))
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))

def identity(st):
    return {"dev": st.st_dev, "ino": st.st_ino, "bytes": st.st_size, "mtime_ns": st.st_mtime_ns}

def main():
    if len(sys.argv) != 4:
        raise SystemExit(__doc__)
    model, oracle, output = map(Path, sys.argv[1:])
    cases_path = Path(__file__).resolve().parent.parent / "tests/fixtures/flint-tokenizer-cases.json"
    proposed = json.loads(cases_path.read_text())
    assert proposed["reference_commit"] == COMMIT
    output.mkdir(exist_ok=False)
    fd = os.open(model, os.O_RDONLY | os.O_NOFOLLOW)
    with os.fdopen(fd, "rb", buffering=0) as source:
        before = os.fstat(source.fileno())
        assert before.st_size == MODEL_BYTES, "wrong source size"
        prefix = source.read(PREFIX_BYTES)
        after = os.fstat(source.fileno())
    assert identity(before) == identity(after), "source changed during bounded read"
    assert len(prefix) == PREFIX_BYTES and hashlib.sha256(prefix).hexdigest() == PREFIX_SHA256, "wrong source metadata hash"
    # Fixed header-word checks, not a second GGUF metadata parser.
    assert prefix[:8] == b"GGUF\x03\0\0\0"
    assert int.from_bytes(prefix[8:16], "little") == 1264
    assert int.from_bytes(prefix[16:24], "little") == 43
    snapshot = bytearray(prefix)
    snapshot[8:16] = bytes(8)  # metadata identical; intentionally no tensor table.
    snapshot += bytes((-len(snapshot)) % 32)
    assert len(snapshot) == 5437024
    snapshot_path = output / "metadata-only.gguf"
    snapshot_path.write_bytes(snapshot)
    requests, cases = [], []
    for case in proposed["raw_text_cases"]:
        for add in [False, True]:
            for parse in [False, True]:
                text = case["text"]
                index = len(cases)
                requests.append(f"E {index} {int(add)} {int(parse)} {text.encode().hex() or '-'}")
                cases.append({"name": f"{case['name']}/add={add}/parse={parse}", "text": text, "add_special": add, "parse_special": parse})
    for index, case in enumerate(proposed["decode_cases"]):
        ids = case["ids"]
        requests.append(f"D {len(cases)} {len(ids)} " + " ".join(map(str, ids)))
        cases.append({"name": f"decode/{index}"})
    # EOG and byte-boundary coverage are independent of any generated prose.
    ids = proposed["stop_cases"]["artifact_eog_ids"] + proposed["stop_cases"]["non_stop_markers"]
    requests.append(f"D {len(cases)} {len(ids)} " + " ".join(map(str, ids)))
    cases.append({"name": "stop-policy"})
    request_bytes = ("\n".join(requests) + "\n").encode()
    (output / "requests.txt").write_bytes(request_bytes)
    completed = subprocess.run([str(oracle.resolve()), str(snapshot_path.resolve())], input=request_bytes,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=120, preexec_fn=bounded_child, check=False)
    (output / "native.stderr").write_bytes(completed.stderr)
    (output / "native.jsonl").write_bytes(completed.stdout)
    assert completed.returncode == 0, f"native oracle failed: {completed.returncode}; see native.stderr"
    assert len(completed.stdout) <= 16 * 1024 * 1024
    rows = [json.loads(row) for row in completed.stdout.splitlines()]
    assert len(rows) == len(cases)
    for index, (case, row) in enumerate(zip(cases, rows)):
        assert row.pop("case") == index
        case["result"] = row
    # Retained independent server observations, not newly invented expectations.
    for source_case in proposed["raw_text_cases"]:
        if "retained_observed_ids_when_both_flags_true" in source_case:
            matches = [case for case in cases if case.get("text") == source_case["text"] and case.get("add_special") and case.get("parse_special")]
            assert len(matches) == 1
            assert matches[0]["result"]["ids"] == source_case["retained_observed_ids_when_both_flags_true"]
    fixture = {
        "reference_commit": COMMIT, "model_metadata_sha256": PREFIX_SHA256,
        "model_metadata_bytes": PREFIX_BYTES, "source_observation": identity(before),
        "oracle_binary_sha256": hashlib.sha256(oracle.read_bytes()).hexdigest(),
        "request_sha256": hashlib.sha256(request_bytes).hexdigest(),
        "snapshot": {"bytes": len(snapshot), "sha256": hashlib.sha256(snapshot).hexdigest(),
            "derivation": "source bytes [0,5436996); tensor_count bytes [8,16) zeroed; 28 zero alignment bytes appended; all 43 metadata entries unchanged"},
        "scope": "native vocab only, no backend/context init or tensor payload; chat templates not executed",
        "cases": cases,
    }
    (output / "oracle.json").write_text(json.dumps(fixture, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"oracle": str(output / "oracle.json"), "cases": len(cases), "snapshot": fixture["snapshot"]}))

if __name__ == "__main__":
    main()
