"""Check shared literal fixtures against the pinned HF template, CPU/offline.

Usage: python tests/nemotron_chat_reference.py /path/to/chat_template.jinja
Requires the already-installed Transformers Jinja environment; installs nothing.
This is reference text evidence, not the content-safe production tokenizer.
"""
import hashlib
import json
from pathlib import Path
import sys
from transformers.utils.chat_template_utils import _compile_jinja_template

template = Path(sys.argv[1]).read_bytes()
assert hashlib.sha256(template).hexdigest() == "1369e90b19976f99ce2070eb21ade8fcb4af640f008e14e82af5c5e89a9cc816"
render = _compile_jinja_template(template.decode()).render
cases = json.loads(Path(__file__).with_name("fixtures").joinpath("nemotron_chat_reference.json").read_text())
for case in cases:
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
