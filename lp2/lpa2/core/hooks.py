from __future__ import annotations
from typing import Sequence, Tuple, Dict, Any, Optional, Set
import types
import torch
from diffusers.models.attention_processor import AttnProcessor, Attention, XFormersAttnProcessor

# ---- Token index selection utilities ----

def _as_int_timestep(t):
    import torch
    if isinstance(t, torch.Tensor):
        if t.numel() == 1:
            return int(t.item())
        return int(t.flatten()[0].item())
    if isinstance(t, (list, tuple)) and len(t) > 0:
        return int(t[0])
    try:
        return int(t)
    except Exception:
        return 1000



def _token_indices_for_words(tokenizer, prompt: str, words: Sequence[str]) -> Set[int]:
    """
    Very simple mapping: we tokenize prompt and collect indices whose decoded piece
    overlaps any target word (case-insensitive). Works for CLIP tokenizer.
    """
    ids = tokenizer(prompt, return_tensors="pt", padding=False, truncation=True)["input_ids"][0]
    tokens = [tokenizer.decode([int(i)]).strip() for i in ids]
    target = {w.lower() for w in words if w}
    idxs: Set[int] = set()
    for i, tk in enumerate(tokens):
        clean = tk.lower().strip()
        for w in target:
            if w and w in clean:
                idxs.add(i)
                break
    return idxs

# ---- Attention processor wrapper ----

class MaskingAttnProcessor(AttnProcessor):
    """
    Wraps the default attention processor and masks attention
    to either style or content token indices depending on layer routing.
    """
    def __init__(self, base: AttnProcessor, layer_name: str, state: Dict[str, Any]):
        super().__init__()
        self.base = base
        self.layer_name = layer_name
        self.state = state  # contains: style_layers, content_layers, t_window, idx_style, idx_content, get_timestep()

    def __call__(self, attn: Attention, hidden_states, encoder_hidden_states=None, attention_mask=None, temb=None):
        # Decide if we should mask given current layer and timestep
        t = self.state["get_timestep"]()
        t0, t1 = self.state["t_window"]
        if not (t0 <= int(t) <= t1):
            return self.base(attn, hidden_states, encoder_hidden_states, attention_mask, temb)

        layer = self.layer_name
        style_layers = self.state["style_layers"]
        content_layers = self.state["content_layers"]
        idx_style: Set[int] = self.state["idx_style"]
        idx_content: Set[int] = self.state["idx_content"]

        if (layer not in style_layers) and (layer not in content_layers):
            return self.base(attn, hidden_states, encoder_hidden_states, attention_mask, temb)

        # Run base to get attention probabilities then apply masking-by-keys
        # To stay backend-agnostic, we intercept encoder_hidden_states before score calc.
        # Strategy: zero out entries of encoder_hidden_states for the forbidden token set by multiplying by 0 mask.
        if encoder_hidden_states is None:
            return self.base(attn, hidden_states, encoder_hidden_states, attention_mask, temb)

        # Build mask vector over token dimension [B, 1, 1, S]
        bsz, seqlen, _ = encoder_hidden_states.shape
        keep = torch.ones(seqlen, device=encoder_hidden_states.device, dtype=encoder_hidden_states.dtype)

        if layer in style_layers and len(idx_content) > 0:
            # at style layers, suppress content tokens
            keep[list(idx_content)] = 0.0
        if layer in content_layers and len(idx_style) > 0:
            # at content layers, suppress style tokens
            keep[list(idx_style)] = 0.0

        # Expand to keys: multiply encoder_hidden_states so attention can't attend to suppressed tokens.
        encoder_hidden_states = encoder_hidden_states * keep.unsqueeze(-1)

        return self.base(attn, hidden_states, encoder_hidden_states, attention_mask, temb)

def _wrap_all_cross_attention(pipe, layer_prefixes: Sequence[str], state: Dict[str, Any], saved_out):
    """
    Replace processors for targeted cross-attn modules (names containing '.attn2.')
    in a single set_attn_processor call. Save originals to saved_out.
    """
    unet = pipe.unet
    procs = dict(unet.attn_processors)  # full mapping name -> processor
    changed = 0

    def match_prefix(name: str) -> Optional[str]:
        for p in layer_prefixes:
            if name.startswith(p):
                return p
        return None

    for name, base in procs.items():
        if ".attn2." not in name:
            continue
        tag = match_prefix(name)
        if tag is None:
            continue
        wrapper = MaskingAttnProcessor(base, tag, state)
        procs[name] = wrapper
        saved_out.append((name, base))
        changed += 1

    if changed > 0:
        unet.set_attn_processor(procs)



def _restore_processors(pipe, saved):
    if not saved:
        return
    procs = dict(pipe.unet.attn_processors)
    for name, base in saved:
        procs[name] = base
    pipe.unet.set_attn_processor(procs)


# ---- Public API ----
def attach_lpa_hooks(
    pipe,
    style_layers: Sequence[str],
    content_layers: Sequence[str],
    timestep_window: Tuple[int, int],
    parser: str = "spacy",
):
    """
    Attach masking processors on selected cross-attn layers and capture
    UNet timesteps via a forward pre-hook (no monkey-patching).
    """
    pipe._lpa_saved_processors = []  # type: ignore[attr-defined]
    pipe._lpa_state: Dict[str, Any] = {}  # type: ignore[attr-defined]

    # tokenizer and parsing
    tokenizer = pipe.tokenizer if hasattr(pipe, "tokenizer") else pipe.tokenizer_2
    from lpa2.core.parsing import split_style_content

    def prepare_indices(prompt: str):
        content_words, style_words = split_style_content(prompt, backend=parser)
        idx_style = _token_indices_for_words(tokenizer, prompt, style_words)
        idx_content = _token_indices_for_words(tokenizer, prompt, content_words)
        return idx_style, idx_content

    # init shared state
    pipe._lpa_state["idx_style"] = set()
    pipe._lpa_state["idx_content"] = set()
    pipe._lpa_state["style_layers"] = list(style_layers)
    pipe._lpa_state["content_layers"] = list(content_layers)
    pipe._lpa_state["t_window"] = tuple(timestep_window)

    # forward pre-hook to capture timestep each UNet call
    unet = pipe.unet
    def _pre_hook(module, args, kwargs=None):
        t = None
        if kwargs and "timestep" in kwargs:
            t = kwargs["timestep"]
        elif isinstance(args, tuple) and len(args) >= 2:
            t = args[1]
        pipe._lpa_state["_t"] = _as_int_timestep(t)
        return None

    try:
        handle = unet.register_forward_pre_hook(_pre_hook, with_kwargs=True)  # torch>=2.0
    except TypeError:
        handle = unet.register_forward_pre_hook(_pre_hook)  # fallback

    pipe._lpa_pre_hook_handle = handle  # type: ignore[attr-defined]
    pipe._lpa_state["get_timestep"] = lambda: pipe._lpa_state.get("_t", 1000)

    # wrap processors on targeted cross-attn modules
    all_layers = list(set(style_layers) | set(content_layers))
    _wrap_all_cross_attention(pipe, all_layers, pipe._lpa_state, pipe._lpa_saved_processors)

    # callback: update token indices per prompt
    def _lpa_on_prompt(prompt: str):
        idx_s, idx_c = prepare_indices(prompt)
        pipe._lpa_state["idx_style"] = idx_s
        pipe._lpa_state["idx_content"] = idx_c
    pipe._lpa_on_prompt = _lpa_on_prompt  # type: ignore[attr-defined]

    return pipe

def detach_lpa_hooks(pipe):
    if hasattr(pipe, "_lpa_saved_processors"):
        _restore_processors(pipe, pipe._lpa_saved_processors)  # type: ignore[arg-type]
        pipe._lpa_saved_processors = []  # type: ignore[attr-defined]
    if hasattr(pipe, "_lpa_pre_hook_handle") and pipe._lpa_pre_hook_handle is not None:
        try:
            pipe._lpa_pre_hook_handle.remove()  # type: ignore[attr-defined]
        except Exception:
            pass
        pipe._lpa_pre_hook_handle = None  # type: ignore[attr-defined]
    if hasattr(pipe, "_lpa_state"):
        pipe._lpa_state = {}  # type: ignore[attr-defined]
