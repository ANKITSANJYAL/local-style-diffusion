from __future__ import annotations
import argparse, os, csv, json, io, tempfile
from typing import List, Dict, Any, Tuple
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm
from lpa2.core.models import load_pipe
from lpa2.core.routing import PRESETS, layers_from_preset
from lpa2.core.utils import set_seed, make_run_dir, save_json, short_hash
from lpa2.core.profiling import profiler
from lpa2.core.hooks import attach_lpa_hooks, detach_lpa_hooks

# ----------------------------
# Prompt loading (schema-agnostic)
# ----------------------------
def _load_prompt_items(path: str) -> List[Any]:
    """
    Supports:
      - JSON list of dicts (keys: text/prompt/neg/etc.)
      - JSON list of strings
      - JSON { "prompts": [...] }
      - JSONL (one JSON object per line)
    """
    with open(path, "r", encoding="utf-8") as f:
        raw = f.read()

    # Try JSONL first
    items: List[Any] = []
    is_jsonl = False
    for line in io.StringIO(raw.strip()):
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
            items.append(obj)
            is_jsonl = True
        except json.JSONDecodeError:
            is_jsonl = False
            break

    if not is_jsonl:
        data = json.loads(raw)
        if isinstance(data, dict) and "prompts" in data:
            items = data["prompts"]
        else:
            items = data
    return items

def _normalize_item(
    item: Any, default_neg: str
) -> Tuple[str, str, Dict[str, Any]]:
    """
    Returns (prompt, negative_prompt, meta).
    Accepts dict or string. For dicts, prefers 'text' then 'prompt'.
    Passes through known meta like 'task' and 'category'.
    """
    if isinstance(item, str):
        return item, default_neg, {}

    if isinstance(item, dict):
        prompt = (
            item.get("text")
            or item.get("prompt")
            or item.get("caption")
            or item.get("input")
            or ""
        )
        neg = item.get("neg", default_neg)
        meta: Dict[str, Any] = {}
        for k in ("task", "category"):
            if k in item:
                meta[k] = item[k]
        return prompt, neg, meta

    return "", default_neg, {}

# ----------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", default="lpa2/configs/default.yaml")
    args = ap.parse_args()

    cfg = OmegaConf.load(args.cfg)
    set_seed(int(cfg.seed))

    run_dir = make_run_dir(cfg.data.output_root)
    os.makedirs(os.path.join(run_dir, "images"), exist_ok=True)
    # Save resolved config alongside the run (yaml-as-json is fine for quick inspection)
    save_json(os.path.join(run_dir, "config.yaml"), OmegaConf.to_container(cfg, resolve=True))  # type: ignore

    # Model
    models_cfg = OmegaConf.load("lpa2/configs/models.yaml")
    repo_id = models_cfg.models[cfg.model.name].repo_id  # type: ignore
    pipe = load_pipe(repo_id, cfg.model.scheduler, cfg.device, cfg.dtype)

    # Routing preset
    preset = PRESETS[cfg.schedule.preset]
    layers = layers_from_preset(preset)

    # Attach LPA (if vanilla, lists are empty → no-op)
    attach_lpa_hooks(
        pipe,
        style_layers=layers["style_layers"],
        content_layers=layers["content_layers"],
        timestep_window=tuple(cfg.schedule.timestep_window),
        parser=str(cfg.schedule.parser),
    )

    # Prompts
    items = _load_prompt_items(cfg.data.prompts_file)
    default_neg = getattr(cfg.data, "negative_prompt", "") or ""
    rows: List[Dict[str, Any]] = []

    num_per_prompt = int(cfg.sampling.num_images_per_prompt if "sampling" in cfg and "num_images_per_prompt" in cfg.sampling else cfg.num_images_per_prompt)
    guidance_scale = float(cfg.guidance_scale)
    steps = int(cfg.steps)

    with profiler():
        for i, raw in enumerate(tqdm(items, desc="Generating")):
            prompt, neg, meta = _normalize_item(raw, default_neg)
            if not prompt:
                # skip malformed or empty record
                continue

            # Update token indices for current prompt (for LPA processors)
            if hasattr(pipe, "_lpa_on_prompt"):
                pipe._lpa_on_prompt(prompt)  # type: ignore[attr-defined]

            for k in range(num_per_prompt):
                out = pipe(
                    prompt=prompt,
                    negative_prompt=neg,
                    guidance_scale=guidance_scale,
                    num_inference_steps=steps,
                )
                img: Image.Image = out.images[0]
                base = f"{i:03d}_{short_hash(prompt)}_{k}"
                fn = f"{base}.png"
                img.save(os.path.join(run_dir, "images", fn))
                row = {
                    "idx": i,
                    "file": fn,
                    "prompt": prompt,
                    "negative": neg,
                    "model": cfg.model.name,
                    "scheduler": cfg.model.scheduler,
                    "preset": cfg.schedule.preset,
                }
                # carry meta fields (e.g., task/category) into manifest
                row.update(meta)
                rows.append(row)

    # Manifest (atomic write)
    manifest_path = os.path.join(run_dir, "generation.csv")
    if rows:
        fieldnames = list(rows[0].keys())
        # Ensure stable header across rows
        for r in rows:
            for k in r.keys():
                if k not in fieldnames:
                    fieldnames.append(k)
        with tempfile.NamedTemporaryFile("w", delete=False, newline="", dir=run_dir, prefix="generation_", suffix=".tmp") as tmpf:
            w = csv.DictWriter(tmpf, fieldnames=fieldnames)
            w.writeheader()
            for r in rows:
                w.writerow({k: r.get(k, "") for k in fieldnames})
            tmp_name = tmpf.name
        os.replace(tmp_name, manifest_path)

    print(f"[OK] Saved to {run_dir}")

    # Detach hooks to restore model state
    detach_lpa_hooks(pipe)

if __name__ == "__main__":
    main()
