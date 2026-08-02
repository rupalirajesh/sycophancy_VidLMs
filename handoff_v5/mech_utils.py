"""
Model-internals helpers for the mechanistic (knockout / patching) scripts.
Architecture-level (forward hooks on decoder layers / attention modules) —
unchanged from the dataset-specific pipeline this supersedes, since none of
this depends on which dataset an item came from. Only `build_contexts` is
dataset/schema-aware (matches common.py's probe() record fields).

Same fragile-part warning as before: locating the decoder layer list and the
attention_mask convention can shift across transformers versions. Both fail
loudly (RuntimeError) rather than silently producing wrong numbers.
"""
import torch
import numpy as np

from common import FREE_ASK_SUFFIX


def get_decoder_layers(model):
    for path in ["model.language_model.layers", "model.model.layers",
                 "language_model.model.layers", "model.layers",
                 "language_model.layers"]:
        o = model
        try:
            for p in path.split("."):
                o = getattr(o, p)
            if len(o) > 0:
                return o
        except (AttributeError, TypeError):
            continue
    raise RuntimeError(
        "Could not locate decoder layers via known attribute paths. Inspect "
        "`print(model)` / model.config and add the correct path to "
        "get_decoder_layers() in mech_utils.py before using knockout/patching.")


def get_final_norm_and_head(model):
    head = model.get_output_embeddings()
    for path in ["model.language_model.norm", "model.norm", "language_model.norm"]:
        o = model
        try:
            for p in path.split("."):
                o = getattr(o, p)
            return o, head
        except AttributeError:
            continue
    raise RuntimeError("final norm not found — inspect model structure and update mech_utils.py")


class AttentionKnockout:
    """Zeroes attention FROM every query position TO a fixed key-index range,
    for a chosen band of decoder layers, via a forward-pre-hook on each
    layer's self_attn that replaces the additive attention_mask with a masked
    clone. Requires attn_implementation='eager' (SDPA/flash-attn paths don't
    materialize an additive mask here — raises rather than silently no-op'ing)."""
    def __init__(self, model, layer_indices, key_slice):
        self.handles = []
        layers = get_decoder_layers(model)
        for i in layer_indices:
            h = layers[i].self_attn.register_forward_pre_hook(
                self._make_hook(key_slice), with_kwargs=True)
            self.handles.append(h)

    @staticmethod
    def _make_hook(key_slice):
        def hook(module, args, kwargs):
            mask = kwargs.get("attention_mask", None)
            if mask is None and len(args) > 1 and torch.is_tensor(args[1]):
                mask = args[1]
                is_kw = False
            else:
                is_kw = True
            if mask is None or not torch.is_tensor(mask):
                raise RuntimeError(
                    "AttentionKnockout: no tensor attention_mask found in self_attn call — "
                    "this transformers version likely routes through SDPA/flash-attn without "
                    "materializing an additive mask here. Force attn_implementation='eager'.")
            mask = mask.clone()
            neg = torch.finfo(mask.dtype).min if mask.is_floating_point() else -1e9
            mask[..., key_slice] = neg
            if is_kw:
                kwargs = dict(kwargs); kwargs["attention_mask"] = mask
                return args, kwargs
            else:
                args = list(args); args[1] = mask
                return tuple(args), kwargs
        return hook

    def remove(self):
        for h in self.handles:
            h.remove()
        self.handles = []

    def __enter__(self): return self
    def __exit__(self, *a): self.remove()


class ActivationPatch:
    """Overwrites the last-token hidden state at the output of one decoder
    layer with a cached replacement, via a forward hook. Patches by relative
    position (-1) since clean/pressured contexts differ in length."""
    def __init__(self, model, layer_idx, replacement_vector):
        layers = get_decoder_layers(model)
        self.handle = layers[layer_idx].register_forward_hook(self._make_hook(replacement_vector))

    @staticmethod
    def _make_hook(replacement):
        def hook(module, inputs, output):
            if isinstance(output, tuple):
                hs = output[0].clone()
                hs[:, -1, :] = replacement.to(hs.dtype).to(hs.device)
                return (hs,) + tuple(output[1:])
            hs = output.clone()
            hs[:, -1, :] = replacement.to(hs.dtype).to(hs.device)
            return hs
        return hook

    def remove(self):
        self.handle.remove()

    def __enter__(self): return self
    def __exit__(self, *a): self.remove()


def find_subseq(hay, needle):
    n = len(needle)
    for i in range(len(hay) - n + 1):
        if hay[i:i + n] == needle:
            return i, i + n
    return None


def forced_choice_argmax(engine, messages, forced_prefix, n_options):
    """Returns the SHOWN-POSITION index (0 = whichever option is currently
    listed as letter A in `messages`) — NOT the dataset's original option
    index. The contexts built by build_contexts() end right after the commit
    turn, which lists options in `record["canonical_order"]` (a random
    per-item permutation, not identity order), so callers MUST remap through
    that before comparing against record["argmax_pre"]/["argmax_post"]
    (which are already in original-option-index space, since
    common.calibrated_confidence remaps internally). Use
    `position_to_original()` below — do not compare raw output from this
    function directly to argmax_pre/argmax_post."""
    from common import LETTERS
    inputs = engine.prep_inputs(messages, extra_text=forced_prefix)
    with torch.inference_mode():
        logits = engine.model(**inputs).logits[0, -1]
    letters = LETTERS[:n_options]
    ids = [engine.processor.tokenizer.encode(l, add_special_tokens=False)[0] for l in letters]
    sel = logits[ids].float()
    return int(torch.argmax(sel))


def position_to_original(position, canonical_order):
    """Maps a shown-position index (from forced_choice_argmax on a context
    built by build_contexts) back to the dataset's original option index."""
    return canonical_order[position]


def hidden_states_at_last_pos(engine, messages, forced_prefix):
    inputs = engine.prep_inputs(messages, extra_text=forced_prefix)
    with torch.inference_mode():
        out = engine.model(**inputs, output_hidden_states=True)
    return torch.stack([h[0, -1, :].float().cpu() for h in out.hidden_states])


def build_contexts(item, visual, record):
    """Reconstruct pre/post message contexts from a saved common.probe()
    record, without re-running generation. Uses the EXACT commit_prompt text
    that was actually shown (saved in the record), not a re-derived one —
    the whole point of saving it was to avoid a second source of truth for
    the option ordering that a forward-only forced-choice read then depends on."""
    base = [{"role": "user", "content": [visual,
             {"type": "text", "text": item["question"] + FREE_ASK_SUFFIX}]}]
    after_free = base + [{"role": "assistant", "content": record["free_response"]}]
    pre = after_free + [{"role": "user", "content": record["commit_prompt"]},
                        {"role": "assistant", "content": record["commit_response"]}]
    post = pre + [{"role": "user", "content": record["pressure_used"]},
                  {"role": "assistant", "content": record["pressured_response"]}]
    return {"pre": pre, "post": post}
