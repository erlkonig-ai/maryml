"""Check shared literal fixtures against the pinned HF template, CPU/offline.

Usage: python tests/nemotron_chat_reference.py /path/to/chat_template.jinja
Requires already-installed Jinja2; installs nothing. This pinned template uses
standard Jinja syntax and JSON serialization, not Transformers-only extensions.
This checks the published interface, not training examples or model behavior.
Rust's ordinary_id_checks independently compares complete prompt IDs with the
original tokenizer bytes; literal/orphan departures are explicitly classified.
"""
import hashlib
import json
from pathlib import Path
import sys
from jinja2.sandbox import ImmutableSandboxedEnvironment

template = Path(sys.argv[1]).read_bytes()
assert hashlib.sha256(template).hexdigest() == "1369e90b19976f99ce2070eb21ade8fcb4af640f008e14e82af5c5e89a9cc816"
environment = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True)
environment.filters["tojson"] = lambda value: json.dumps(value, ensure_ascii=False)
render = environment.from_string(template.decode()).render
cases = json.loads(Path(__file__).with_name("fixtures").joinpath("nemotron_chat_reference.json").read_text())
for case in cases:
    assert case["parity"] in {"ordinary", "literal-departure", "orphan-departure"}
    messages = case["messages"]
    for message in messages:
        if "command" in message:
            message["tool_calls"] = [{"function": {"name": "exec", "arguments": {"command": message.pop("command")}}}]
    tools = [{"type": "function", "function": {"name": "exec", "description": "Propose a shell command.",
              "parameters": {"type": "object", "properties": {"command": {"type": "string",
                             "description": "Shell command text."}}, "required": ["command"]}}}] if case["tools"] else []
    actual = render(messages=messages, enable_thinking=case["thinking"], medium_effort=case["medium"],
                    add_generation_prompt=case["generation"], truncate_history_thinking=False, tools=tools)
    assert actual == case["expected"], (case["name"], actual, case["expected"])
    print("PASS", case["name"])
print(f"{len(cases)} pinned-template fixtures passed")

# New typed cases share expected publisher fragments with the Rust gate. The
# reference uses the retained template, not Mary's rendering implementation.
registered = json.loads(Path(__file__).with_name("fixtures").joinpath("nemotron_registered_chat_reference.json").read_text())
for case in registered:
    actual = render(messages=[{"role":"system", "content":"policy"},
        {"role":"assistant", "content":"", "tool_calls":[{"function":{
            "name":case["tool"]["name"], "arguments":case["arguments"]}}]}],
        tools=[{"type":"function", "function":case["tool"]}],
        enable_thinking=False, medium_effort=False, add_generation_prompt=True,
        truncate_history_thinking=False)
    assert case["expected_schema"] in actual, (case["name"], actual)
    assert "<tool_call>" + case["expected_body"] + "</tool_call>" in actual, (case["name"], actual)
    print("PASS", case["name"])
print(f"{len(registered)} registered-tool publisher fragments passed")
