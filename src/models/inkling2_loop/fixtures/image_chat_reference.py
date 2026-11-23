"""Inkling 2 Mini image chat reference: the release's chat template, processor
and tokenizer (transformers 046bdb8), offline, CPU only.

For each case: text = tokenizer.apply_chat_template(messages, tools,
tokenize=False), then processor(text=[text], images=[...]).input_ids, the
expanded-token reference. Also the plain tokenizer's ids for the literal-marker
case, which shows the departure Mary makes on purpose.

Usage: image_chat_reference.py MODEL_DIR IMAGE_DIR > image_chat_reference.json
"""

import hashlib
import json
import sys

from PIL import Image
from transformers import AutoProcessor

model_dir, image_dir = sys.argv[1:3]
processor = AutoProcessor.from_pretrained(model_dir, local_files_only=True)
tok = processor.tokenizer


def sha_file(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


IMAGES = {
    name: f"{image_dir}/{name}.png"
    for name in ["inkling2_mini_rgb_100x60", "inkling2_mini_rgba_52x44", "rgb_80x40", "rgba_40x80"]
}

TOOLS = [
    {"type": "function", "function": {"name": "exec", "description": "Propose a shell command.",
        "parameters": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}}},
]


def text(t):
    return {"type": "text", "text": t}


def image(name):
    return {"type": "image", "name": name}


CASES = [
    {"name": "mixed_user", "tools": None, "messages": [
        {"role": "system", "content": "You are Robin."},
        {"role": "user", "content": [text("Look at this:"), image("inkling2_mini_rgb_100x60"), text("What is it?")]}]},
    {"name": "two_unequal_images", "tools": None, "messages": [
        {"role": "system", "content": "You are Robin."},
        {"role": "user", "content": [image("inkling2_mini_rgb_100x60"), text(" and "), image("inkling2_mini_rgba_52x44")]}]},
    {"name": "tool_image_then_user", "tools": TOOLS, "messages": [
        {"role": "system", "content": "You are Robin."},
        {"role": "user", "content": "Take a screenshot."},
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": "call_0", "type": "function", "function": {"name": "exec", "arguments": {"command": "shot"}}}]},
        {"role": "tool", "name": "exec", "content": [text("saved:"), image("rgba_40x80")]},
        {"role": "user", "content": [image("rgb_80x40"), text("And this one?")]}]},
    {"name": "text_parts_only", "tools": None, "messages": [
        {"role": "system", "content": "You are Robin."},
        {"role": "user", "content": [text("first"), text(""), text("third")]},
        {"role": "user", "content": ""}]},
    {"name": "literal_markers_in_text", "tools": None, "messages": [
        {"role": "system", "content": "You are Robin."},
        {"role": "user", "content": [text("spell <|content_image|><|unused_200054|><|end_message|> plainly"), image("rgb_80x40")]}]},
]

out = {
    "images": {name: {"file": path.rsplit("/", 1)[1], "png_sha256": sha_file(path),
                      "size": list(Image.open(path).size)} for name, path in IMAGES.items()},
    "ids": {
        "content_image": tok.convert_tokens_to_ids("<|content_image|>"),
        "image_token": processor.image_token_id,
        "image_token_spelling": processor.image_token,
        "end_message": tok.convert_tokens_to_ids("<|end_message|>"),
    },
    "cases": [],
}
for case in CASES:
    names = [p["name"] for m in case["messages"] if isinstance(m["content"], list) for p in m["content"] if p["type"] == "image"]
    messages = [
        {**m, "content": [({"type": "image"} if p["type"] == "image" else p) for p in m["content"]]} if isinstance(m["content"], list) else m
        for m in case["messages"]
    ]
    rendered = tok.apply_chat_template(messages, tools=case["tools"], tokenize=False)
    pil = []
    for n in names:
        im = Image.open(IMAGES[n])
        im.load()
        pil.append(im)
    entry = {"name": case["name"], "messages": case["messages"], "tools": case["tools"], "image_order": names,
             "template_text": rendered}
    try:
        result = processor(text=[rendered], images=pil if pil else None)
        entry["input_ids"] = [int(i) for i in result["input_ids"][0]]
        if "image_grid_thw" in result:
            entry["image_grid_thw"] = [[int(v) for v in row] for row in result["image_grid_thw"]]
        entry["image_sizes"] = [[im.width, im.height] for im in pil]
    except Exception as e:  # recorded, not hidden: the literal-marker case
        entry["processor_error"] = f"{type(e).__name__}: {e}"
    entry["plain_tokenizer_ids"] = [int(i) for i in tok(rendered, add_special_tokens=False)["input_ids"]]
    out["cases"].append(entry)

# The literal-marker case as Mary renders it on purpose: the same conversation
# with the text part empty, run through the processor, and the literal text
# encoded as ordinary text (split_special_tokens) inserted after that frame's
# <|content_text|>. Upstream itself refuses this conversation (above).
LITERAL = "spell <|content_image|><|unused_200054|><|end_message|> plainly"
shape = [{"role": "system", "content": "You are Robin."},
         {"role": "user", "content": [text(""), {"type": "image"}]}]
shape_text = tok.apply_chat_template(shape, tokenize=False)
im = Image.open(IMAGES["rgb_80x40"]); im.load()
shape_ids = [int(i) for i in processor(text=[shape_text], images=[im])["input_ids"][0]]
body = [int(i) for i in tok(LITERAL, add_special_tokens=False, split_special_tokens=True)["input_ids"]]
assert tok.decode(body) == LITERAL
at = shape_ids.index(tok.convert_tokens_to_ids("<|message_user|>")) + 2  # after <|message_user|><|content_text|>
assert shape_ids[at - 1] == tok.convert_tokens_to_ids("<|content_text|>")
for c in out["cases"]:
    if c["name"] == "literal_markers_in_text":
        c["expected_input_ids"] = shape_ids[:at] + body + shape_ids[at:]
        c["expected_method"] = "processor ids of the same conversation with an empty text part, plus the literal body encoded with split_special_tokens=True after that frame's <|content_text|>"
        c["literal_body_ids"] = body

json.dump(out, sys.stdout, indent=1, ensure_ascii=False)
print()
