from __future__ import annotations
import argparse, os, csv, glob, json
from typing import List, Tuple
from PIL import Image
import numpy as np
import torch
import torchvision.transforms as T

# CLIP (OpenAI)
import clip
# LPIPS
import lpips

def _load_manifest(run_dir: str):
    rows = list(csv.DictReader(open(os.path.join(run_dir, "generation.csv"))))
    return rows

def _group_by_prompt(rows):
    g = {}
    for r in rows:
        g.setdefault(r["prompt"], []).append(r)
    return g

def _clip_scores(images: List[Image.Image], text: str, device: str) -> np.ndarray:
    model, preprocess = clip.load("ViT-B/32", device=device, jit=False)
    text_tok = clip.tokenize([text], truncate=True).to(device)
    with torch.no_grad():
        tfeat = model.encode_text(text_tok)
        tfeat = tfeat / tfeat.norm(dim=-1, keepdim=True)
        scores = []
        for im in images:
            x = preprocess(im).unsqueeze(0).to(device)
            if x.dtype == torch.float16:
                x = x.float()
            if next(model.parameters()).dtype == torch.float16:
                model.float()
            if next(model.parameters()).dtype != torch.float32:
                model.float()
            if device.startswith("cuda"):
                model.cuda()
            vfeat = model.encode_image(x)
            vfeat = vfeat / vfeat.norm(dim=-1, keepdim=True)
            sim = (vfeat @ tfeat.T).squeeze(0).item()
            scores.append(sim)
    return np.array(scores)

def _style_phrase(prompt: str) -> str:
    import re
    m = re.findall(r"in ([A-Za-z0-9,\- ]+?) (?:style|aesthetic)", prompt, flags=re.IGNORECASE)
    return (m[0] + " style") if m else prompt

def _lpips_diversity(images: List[Image.Image], device: str) -> float:
    loss_fn = lpips.LPIPS(net='vgg').to(device)
    tfm = T.Compose([T.ToTensor()])
    xs = [tfm(img).unsqueeze(0).to(device) for img in images]
    if len(xs) < 2: return 0.0
    d = []
    with torch.no_grad():
        for i in range(len(xs)):
            for j in range(i+1, len(xs)):
                d.append(loss_fn(xs[i], xs[j]).item())
    return float(np.mean(d))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    rows = _load_manifest(args.run_dir)
    groups = _group_by_prompt(rows)

    clip_prompt, clip_style, divs = [], [], []
    for prompt, items in groups.items():
        imgs = [Image.open(os.path.join(args.run_dir, "images", r["file"])).convert("RGB") for r in items]
        clip_prompt.extend(_clip_scores(imgs, prompt, args.device))
        clip_style.extend(_clip_scores(imgs, _style_phrase(prompt), args.device))
        divs.append(_lpips_diversity(imgs, args.device))

    out = {
        "clip_prompt_mean": float(np.mean(clip_prompt)),
        "clip_prompt_std": float(np.std(clip_prompt)),
        "clip_style_mean": float(np.mean(clip_style)),
        "clip_style_std": float(np.std(clip_style)),
        "div_lpips_mean": float(np.mean(divs)),
    }
    os.makedirs(os.path.join(args.run_dir, "metrics"), exist_ok=True)
    with open(os.path.join(args.run_dir, "metrics", "summary.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(out)

if __name__ == "__main__":
    main()
