from __future__ import annotations
import argparse, os, csv, json, re
from typing import List, Dict, Tuple
from PIL import Image
import numpy as np
import torch
import clip

def style_phrase(prompt: str) -> str:
    m = re.findall(r"in ([A-Za-z0-9,\- ]+?) (?:style|aesthetic)", prompt, flags=re.IGNORECASE)
    return (m[0] + " style") if m else prompt

def read_manifest(run_dir: str) -> List[Dict[str,str]]:
    return list(csv.DictReader(open(os.path.join(run_dir, "generation.csv"))))

def group_by_prompt(rows: List[Dict[str,str]]):
    g = {}
    for r in rows:
        g.setdefault(r["prompt"], []).append(r)
    return g

def clip_score_images(imgs: List[Image.Image], text: str, device: str="cpu") -> float:
    model, preprocess = clip.load("ViT-B/32", device=device, jit=False)
    toks = clip.tokenize([text], truncate=True).to(device)
    with torch.no_grad():
        t = model.encode_text(toks); t = t / t.norm(dim=-1, keepdim=True)
        sims = []
        for im in imgs:
            x = preprocess(im).unsqueeze(0).to(device)
            if next(model.parameters()).dtype != torch.float32:
                model.float()
            v = model.encode_image(x); v = v / v.norm(dim=-1, keepdim=True)
            sims.append((v @ t.T).squeeze(0).item())
    return float(np.mean(sims))

def load_imgs(run_dir: str, files: List[str]) -> List[Image.Image]:
    return [Image.open(os.path.join(run_dir, "images", f)).convert("RGB") for f in files]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="Run folder to pick examples from")
    ap.add_argument("--out", required=True, help="Output dir to copy images/grids")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--k", type=int, default=12, help="How many top/bottom prompts to select")
    ap.add_argument("--pick_index", type=int, default=0, help="Which sample index per prompt to copy")
    args = ap.parse_args()

    rows = read_manifest(args.run)
    g = group_by_prompt(rows)
    scores = []
    for p, items in g.items():
        files = [r["file"] for r in sorted(items, key=lambda r: r["file"])]
        imgs = load_imgs(args.run, files)
        sclip = clip_score_images(imgs, style_phrase(p), args.device)
        scores.append((p, sclip, files))

    scores.sort(key=lambda x: x[1])
    bottom = scores[:args.k]
    top = scores[-args.k:]

    os.makedirs(args.out, exist_ok=True)
    def dump(name, subset):
        d = os.path.join(args.out, name); os.makedirs(d, exist_ok=True)
        man = csv.DictWriter(open(os.path.join(d, "manifest.csv"), "w", newline=""),
                             fieldnames=["prompt","style_clip","file"])
        man.writeheader()
        for p, sc, files in subset:
            pick = files[min(max(args.pick_index, 0), len(files)-1)]
            src = os.path.join(args.run, "images", pick)
            dst = os.path.join(d, pick)
            Image.open(src).save(dst)
            man.writerow({"prompt": p, "style_clip": sc, "file": pick})
        print(f"[OK] wrote {name} -> {d}")

    dump("topk", top)
    dump("bottomk", bottom)

if __name__ == "__main__":
    main()
