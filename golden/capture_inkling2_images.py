"""Upstream Inkling 2 Mini end to end with two images and text: processor, chat template, vision tower,
masked_scatter, decoder, logits; float32 and bfloat16, each run whole.

Derived from capture_inkling2_whole.py (lanes/mini-independent-reference-20261122, liora-tivoh-divod): the same
layer streaming of the 24 decoder layers. What is new: the published Inkling2Processor (the release's
processor_config.json, tokenizer and chat template) renders one user message holding the two fixture PNGs and
text, and transformers' own `Inkling2LoopForConditionalGeneration.forward(input_ids, pixel_values,
image_grid_thw, use_cache=True)` runs it, so the vision tower (all its layers, loaded by from_pretrained with the
shell), get_placeholder_mask and masked_scatter are upstream's. A forward pre-hook on the language model records
the decoder's `inputs_embeds` after the scatter.

Recorded: ids [t] int32, image_grid_thw [n,3] int32, pixel_values [tokens, 40, 40, 3] f32; per run (f32, bf16):
<run>.inputs_embeds [t, H] (the merged decoder input), <run>.embeds [t, H] (embed_tokens, before the scatter),
<run>.image_embeds [k, H] (the tower's pooler output, concatenated), <run>.logits [t, V], <run>.final [t, H].

usage (offline container): python3 -I capture_inkling2_images.py <tf src> <model dir> <png,png> <out.safetensors> [deps]
"""
import gc, hashlib, json, os, resource, struct, sys, time

tf_src, model_dir, pngs_arg, out_path = sys.argv[1:5]
sys.path[:0] = [tf_src] + sys.argv[5:6]

import torch
import torch.nn as nn
import transformers
from safetensors.torch import save_file

assert os.path.realpath(os.path.dirname(transformers.__file__)).startswith(os.path.realpath(tf_src)), transformers.__file__
import transformers.models.inkling2_loop.modeling_inkling2_loop as M
from transformers.models.inkling2_loop.configuration_inkling2_loop import Inkling2LoopConfig

torch.backends.mkldnn.enabled = False
torch.set_num_threads(int(os.environ.get("THREADS", "12")))
T0 = time.time()


def log(*a):
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1048576
    print(f"[{time.time() - T0:7.1f}s maxrss {rss:5.1f} GiB]", *a, file=sys.stderr, flush=True)


REAL = {"SL": M.Inkling2LoopSlidingDecoderLayer, "FL": M.Inkling2LoopFullDecoderLayer,
        "SLL": M.Inkling2LoopSlidingLoopDecoderLayer, "FLL": M.Inkling2LoopFullLoopDecoderLayer}
LIVE = set()  # layers the next construction builds for real (visit loads); empty for the shell
STATE = {}


class Streamed(nn.Module):
    """A decoder layer of the running model: no parameters; loads the real layer for each call."""

    def __init__(self, config, layer_idx):
        super().__init__()
        self.layer_idx = layer_idx
        self.is_block_boundary = layer_idx % config.n_layers_per_attn_res_block == 0
        self.attention_type = "sliding_attention" if config.layer_types[layer_idx] == "hybrid_sliding" else "full_attention"

    def forward(self, *args, **kwargs):
        L = self.layer_idx
        real = load_layer(L)
        assert real.is_block_boundary == self.is_block_boundary and real.attention_type == self.attention_type, L
        with torch.no_grad():
            out = real(*args, **kwargs)
        n = STATE["visit"]
        STATE["visit"] += 1
        STATE["rec"][f"{STATE['run']}.visit.{n:02d}.l{L}.out"] = out[0].float().clone().contiguous()
        STATE["order"].append(L)
        del real
        gc.collect()
        log(f"{STATE['run']} visit {n} layer {L} done")
        return out


def factory(real):
    return lambda config, layer_idx: real(config, layer_idx) if layer_idx in LIVE else Streamed(config, layer_idx)


M.Inkling2LoopSlidingDecoderLayer = factory(REAL["SL"])
M.Inkling2LoopFullDecoderLayer = factory(REAL["FL"])
M.Inkling2LoopSlidingLoopDecoderLayer = factory(REAL["SLL"])
M.Inkling2LoopFullLoopDecoderLayer = factory(REAL["FLL"])

config = Inkling2LoopConfig.from_pretrained(model_dir)
config.text_config._attn_implementation = "eager"
tc = config.text_config


def from_pretrained(dtype, live):
    LIVE.clear()
    LIVE.update(live)
    model, info = M.Inkling2LoopForConditionalGeneration.from_pretrained(
        model_dir, config=config, dtype=dtype, attn_implementation="eager", output_loading_info=True)
    LIVE.clear()
    model.eval()
    lm = model.model.language_model
    # nothing of the text model we run may be missing: the live layers, or for the shell everything not a layer
    if live:
        missing = [k for k in info["missing_keys"] if any(f"layers.{L}." in k for L in live)]
    else:
        missing = list(info["missing_keys"])  # the shell includes the whole vision tower here
    assert not missing, missing
    return model, lm, info


def load_layer(L):
    model, lm, _ = from_pretrained(STATE["dtype"], {L})
    real = lm.layers[L]
    assert type(real) in REAL.values(), (L, type(real))
    for p in real.parameters():
        assert p.device.type == "cpu" and not p.is_meta, L
    del model, lm
    return real


def file_identity(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header = f.read(n)
    return {"bytes": os.path.getsize(path), "header_len": n, "header_sha256": hashlib.sha256(header).hexdigest()}


def prompt():
    from PIL import Image
    from transformers import AutoProcessor
    proc = AutoProcessor.from_pretrained(model_dir)
    images = []
    for p in pngs_arg.split(","):
        im = Image.open(p)
        im.load()
        images.append(im)
    messages = [{"role": "user", "content": [
        {"type": "image", "image": images[0]},
        {"type": "text", "text": "Here is the first picture."},
        {"type": "image", "image": images[1]},
        {"type": "text", "text": "Which of the two is more colourful?"}]}]
    text = proc.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    batch = proc(text=[text], images=images, return_tensors="pt")
    return proc, text, batch


def main():
    proc, text, batch = prompt()
    ids = batch["input_ids"][0].to(torch.long)
    pv, thw = batch["pixel_values"], batch["image_grid_thw"]
    assert pv.dtype == torch.float32, pv.dtype
    image_id = config.image_token_id
    assert proc.image_token_id == image_id, (proc.image_token_id, image_id)
    k = int((ids == image_id).sum())
    assert k == int(thw.prod(-1).sum()), (k, thw)
    log(f"prompt: {ids.shape[0]} ids, {k} image placeholders (id {image_id}), grids {thw.tolist()}")
    rec = {"ids": ids.to(torch.int32), "image_grid_thw": thw.to(torch.int32), "pixel_values": pv.contiguous()}
    meta = {"transformers": "046bdb8cc2cee29d233964bb77fe6fd37cfe4aa3", "torch": torch.__version__, "device": "cpu",
            "cpu_backend": f"torch.backends.mkldnn.enabled={torch.backends.mkldnn.enabled}",
            "attn_implementation": "eager", "threads": str(torch.get_num_threads()),
            "forward": "Inkling2LoopForConditionalGeneration.forward(input_ids, pixel_values, image_grid_thw, "
                       "use_cache=True), decoder layers streamed, vision tower whole",
            "prompt": text, "pngs": json.dumps([os.path.basename(p) for p in pngs_arg.split(",")]),
            "png_sha256": json.dumps([hashlib.sha256(open(p, "rb").read()).hexdigest() for p in pngs_arg.split(",")]),
            "image_token_id": str(image_id),
            "model_file": json.dumps(file_identity(os.path.join(model_dir, "model-00001-of-00001.safetensors"))),
            "loop": json.dumps({"start": tc.loop_layer_index_start, "end": tc.loop_layer_index_end,
                                "passes": tc.loop_num_passes})}
    assert not torch.backends.mkldnn.enabled
    for run, dtype in (("f32", torch.float32), ("bf16", torch.bfloat16)):
        STATE.update(run=run, dtype=dtype, visit=0, rec=rec, order=[])
        model, lm, info = from_pretrained(dtype, set())
        log(f"{run} shell loaded: {len(info['unexpected_keys'])} unexpected (other layers), "
            f"{len(info['missing_keys'])} missing (vision only)")
        assert all(isinstance(l, Streamed) for l in lm.layers)
        emb = {}
        h = lm.embed_tokens.register_forward_hook(lambda _m, _i, o: emb.update(x=o.detach().clone()))  # returns None
        vt = model.model.vision_tower
        nvis = sum(p.numel() for p in vt.parameters())
        assert nvis > 0 and all(not p.is_meta for p in vt.parameters()), "vision tower not loaded"
        log(f"{run} vision tower: {nvis} parameters, {vt.config.num_hidden_layers} layers")
        merged = {}
        h3 = lm.register_forward_pre_hook(
            lambda _m, _a, kw: merged.update(x=kw["inputs_embeds"].detach().clone()), with_kwargs=True)
        img = {}
        h4 = vt.register_forward_hook(lambda _m, _i, o: img.update(x=o.pooler_output.detach().clone()))
        fin = {}
        h2 = lm.register_forward_hook(lambda _m, _i, o: fin.update(x=o.last_hidden_state.detach().clone()))  # returns None
        with torch.no_grad():
            out = model(input_ids=ids.unsqueeze(0), pixel_values=pv.to(dtype), image_grid_thw=thw, use_cache=True)
        h.remove()
        h2.remove()
        h3.remove()
        h4.remove()
        logits = out.logits[0]
        assert logits.dtype == torch.float32 and logits.shape == (ids.shape[0], tc.unpadded_vocab_size), logits.shape
        rec[f"{run}.logits"] = logits.contiguous()
        rec[f"{run}.embeds"] = emb["x"][0].float().contiguous()
        rec[f"{run}.inputs_embeds"] = merged["x"][0].float().contiguous()
        rec[f"{run}.image_embeds"] = img["x"].float().contiguous()
        mask = ids == image_id
        assert torch.equal(merged["x"][0][~mask], emb["x"][0][~mask]), "text rows changed by the scatter"
        assert torch.equal(merged["x"][0][mask], img["x"].to(merged["x"].dtype)), "image rows are not the tower's"
        rec[f"{run}.final"] = fin["x"][0].float().contiguous()
        meta[f"{run}.visit_order"] = json.dumps(STATE["order"])
        meta[f"{run}.cache_layers"] = str(len(out.cache.layers)) if out.cache is not None else "none"
        log(f"{run}: argmax {logits.argmax(-1).tolist()}")
        del model, lm, out
        gc.collect()
    save_file({k: v.clone().contiguous() for k, v in rec.items()}, out_path, metadata=meta)
    log(f"saved {len(rec)} tensors to {out_path}")


main()
