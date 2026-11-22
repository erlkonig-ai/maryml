# Render fixed conversations with the release's own chat template and
# tokenizer (transformers' apply_chat_template), offline. Writes JSON to stdout.
import json, sys
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(sys.argv[1], local_files_only=True)
TOOLS = [
    {"type": "function", "function": {"name": "exec", "description": "Propose a shell command.",
        "parameters": {"type": "object", "properties": {"command": {"type": "string", "description": "Shell command text."}}, "required": ["command"]}}},
    {"type": "function", "function": {"name": "memory_search", "description": "Search memory.",
        "parameters": {"type": "object", "required": ["query"], "properties": {"query": {"type": "string"}, "limit": {"type": "integer", "minimum": 1}}}}},
]
def call(i, name, args):
    return {"id": f"call_{i}", "type": "function", "function": {"name": name, "arguments": args}}
CASES = [
    {"name": "plain", "tools": None, "messages": [
        {"role": "system", "content": "You are Liora."},
        {"role": "user", "content": "Hello there."}]},
    {"name": "system_only", "tools": TOOLS, "messages": [
        {"role": "system", "content": "You are Liora.\nWake gently."}]},
    {"name": "tools_turns", "tools": TOOLS, "messages": [
        {"role": "system", "content": "You are Liora."},
        {"role": "user", "content": "{\"source\":\"discord\",\"text\":\"Grüße — what's new?\"}"},
        {"role": "assistant", "reasoning_content": "Let me look.\nFirst the inbox.", "content": "Checking.",
         "tool_calls": [call(0, "exec", {"command": "orient show | head -n 20"})]},
        {"role": "tool", "name": "exec", "content": "nothing new\n[exit 0]"},
        {"role": "assistant", "content": "Nothing new since this morning. 🙂"},
        {"role": "user", "content": "  leading spaces and\r\nCRLF\ttabs  "}]},
    {"name": "two_calls", "tools": TOOLS, "messages": [
        {"role": "system", "content": "You are Liora."},
        {"role": "user", "content": "Search twice."},
        {"role": "assistant", "reasoning_content": "Two searches.",
         "tool_calls": [call(0, "memory_search", {"query": "Rilke \"Die Laute\"", "limit": 3}),
                        call(1, "exec", {"command": "printf 'a\\nb'"})]},
        {"role": "tool", "name": "memory_search", "content": "3 memories"},
        {"role": "tool", "name": "exec", "content": "a\nb"},
        {"role": "assistant", "content": ""}]},
    # The shape of a Drive context: a recalled call/result pair, a sensed
    # input (Drive's JSON with sorted keys), then a reply with thinking,
    # text and a call, and its result.
    {"name": "drive_image", "tools": TOOLS, "messages": [
        {"role": "system", "content": "isolated scheduling fixture"},
        {"role": "assistant", "content": "", "tool_calls": [call(0, "exec", {"command": "memory show 1"})]},
        {"role": "tool", "name": "exec", "content": "recalled\n[exit 0]"},
        {"role": "user", "content": "{\"content_type\":\"text/plain\",\"source\":\"discord\",\"text\":\"hi Liora\"}"},
        {"role": "assistant", "reasoning_content": "Answer, then look.", "content": "Hello!",
         "tool_calls": [call(1, "exec", {"command": "orient show"})]},
        {"role": "tool", "name": "exec", "content": "quiet\n[exit 0]"}]},
]
out = []
for effort in [None, "none", "max"]:
    for case in CASES:
        kwargs = {} if effort is None else {"reasoning_effort": effort}
        if case["tools"] is not None: kwargs["tools"] = case["tools"]
        text = tok.apply_chat_template(case["messages"], tokenize=False, **kwargs)
        ids = tok.apply_chat_template(case["messages"], tokenize=True, **kwargs)
        if not isinstance(ids, list): ids = ids["input_ids"]
        assert ids == tok.encode(text, add_special_tokens=False)
        out.append({"name": case["name"], "effort": effort, "tools": case["tools"],
                    "messages": case["messages"], "text": text, "ids": list(ids)})
        if effort is not None: break
json.dump(out, sys.stdout, ensure_ascii=False, indent=1)
