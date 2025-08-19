from __future__ import annotations
import argparse, os, csv, math, textwrap
from typing import List, Dict, Tuple
from PIL import Image, ImageDraw, ImageFont

def _read_manifest(run_dir: str) -> List[Dict[str, str]]:
    man_path = os.path.join(run_dir, "generation.csv")
    if not os.path.exists(man_path):
        raise FileNotFoundError(f"Missing manifest: {man_path}")
    rows = list(csv.DictReader(open(man_path)))
    if not rows:
        raise ValueError(f"No rows in manifest: {man_path}")
    return rows

def _group_by_prompt(rows: List[Dict[str, str]]) -> Dict[str, List[Dict[str, str]]]:
    g: Dict[str, List[Dict[str, str]]] = {}
    for r in rows:
        g.setdefault(r["prompt"], []).append(r)
    # Keep a consistent order per prompt (by file name)
    for k in g:
        g[k] = sorted(g[k], key=lambda r: r["file"])
    return g

def _pick_by_index(items: List[Dict[str, str]], k: int) -> Dict[str, str]:
    if not items:
        raise ValueError("Empty item list for a prompt.")
    k = max(0, min(k, len(items) - 1))
    return items[k]

def _safe_load(path: str) -> Image.Image:
    im = Image.open(path).convert("RGB")
    return im

def _fit_same_height(imA: Image.Image, imB: Image.Image, target_h: int | None = None) -> Tuple[Image.Image, Image.Image]:
    # Resize both to same height, preserve aspect. If target_h None, use min of both heights (clamped to 1024).
    ha, hb = imA.height, imB.height
    if target_h is None:
        target_h = min(ha, hb, 1024)
    wa = int(round(imA.width * (target_h / ha)))
    wb = int(round(imB.width * (target_h / hb)))
    return imA.resize((wa, target_h), Image.BICUBIC), imB.resize((wb, target_h), Image.BICUBIC)

def _draw_header(draw: ImageDraw.ImageDraw, W: int, text: str, pad: int = 10):
    # Wrap long prompts
    wrapped = textwrap.wrap(text, width=110)
    y = pad
    for line in wrapped:
        draw.text((pad, y), line, fill=(0, 0, 0))
        y += 18
    return y + pad  # new y after header

def _panel(imA: Image.Image, imB: Image.Image, labelA: str, labelB: str, prompt: str, header: bool = True) -> Image.Image:
    # Resize to same height then assemble with white header strip
    imA, imB = _fit_same_height(imA, imB, None)
    label_h = 30
    header_h = 46 if header else 0
    pad = 8
    W = imA.width + imB.width + pad * 3
    H = max(imA.height, imB.height) + label_h + header_h + pad * 2
    out = Image.new("RGB", (W, H), (255, 255, 255))
    d = ImageDraw.Draw(out)
    y = pad
    if header:
        y = _draw_header(d, W, f"Prompt: {prompt}", pad=pad)

    # Method labels row
    d.text((pad, y + 6), labelA, fill=(0, 0, 0))
    d.text((pad + imA.width + pad, y + 6), labelB, fill=(0, 0, 0))
    y += label_h

    # Paste images
    out.paste(imA, (pad, y))
    out.paste(imB, (pad + imA.width + pad, y))
    return out

def _ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)

def make_per_prompt_grids(runA: str, runB: str, out_dir: str, labelA: str, labelB: str, kA: int, kB: int, limit: int | None) -> str:
    rowsA = _read_manifest(runA)
    rowsB = _read_manifest(runB)
    gA = _group_by_prompt(rowsA)
    gB = _group_by_prompt(rowsB)

    prompts = sorted(set(gA.keys()) & set(gB.keys()))
    if not prompts:
        raise ValueError("No overlapping prompts between runs; cannot build grids.")

    if limit is not None:
        prompts = prompts[:limit]

    _ensure_dir(out_dir)
    out_manifest = os.path.join(out_dir, "grids_manifest.csv")
    mw = csv.DictWriter(open(out_manifest, "w", newline=""), fieldnames=["idx", "prompt", "grid_file", "fileA", "fileB"])
    mw.writeheader()

    for idx, p in enumerate(prompts):
        rA = _pick_by_index(gA[p], kA)
        rB = _pick_by_index(gB[p], kB)
        imA = _safe_load(os.path.join(runA, "images", rA["file"]))
        imB = _safe_load(os.path.join(runB, "images", rB["file"]))
        grid = _panel(imA, imB, labelA, labelB, p, header=True)
        fn = f"{idx:03d}.png"
        grid.save(os.path.join(out_dir, fn))
        mw.writerow({"idx": idx, "prompt": p, "grid_file": fn, "fileA": rA["file"], "fileB": rB["file"]})

    return out_manifest

def make_contact_sheet(grids_dir: str, out_path: str, cols: int = 2, thumb_w: int = 640, pad: int = 12):
    # Collect all grid images and tile them into pages (single page here)
    files = [f for f in sorted(os.listdir(grids_dir)) if f.lower().endswith(".png")]
    if not files:
        raise ValueError("No per-prompt grids found to assemble.")
    ims = [Image.open(os.path.join(grids_dir, f)).convert("RGB") for f in files]

    # Resize each grid to uniform width (height keeps aspect)
    def resize_w(im: Image.Image, w: int) -> Image.Image:
        h = int(round(im.height * (w / im.width)))
        return im.resize((w, h), Image.BICUBIC)

    ims = [resize_w(im, thumb_w) for im in ims]
    rows = math.ceil(len(ims) / cols)
    cell_w = thumb_w
    cell_h = max(im.height for im in ims)
    W = cols * cell_w + (cols + 1) * pad
    H = rows * cell_h + (rows + 1) * pad
    sheet = Image.new("RGB", (W, H), (255, 255, 255))

    for i, im in enumerate(ims):
        r = i // cols
        c = i % cols
        x = pad + c * (cell_w + pad)
        y = pad + r * (cell_h + pad)
        sheet.paste(im, (x, y))

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    sheet.save(out_path)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runA", required=True)
    ap.add_argument("--runB", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--labelA", default="LPA")
    ap.add_argument("--labelB", default="Baseline")
    ap.add_argument("--kA", type=int, default=0, help="Which image index to pick for runA per prompt (0-based).")
    ap.add_argument("--kB", type=int, default=0, help="Which image index to pick for runB per prompt (0-based).")
    ap.add_argument("--limit", type=int, default=None, help="Limit number of prompts for quick preview.")
    ap.add_argument("--contact", action="store_true", help="Also build a contact-sheet PNG.")
    ap.add_argument("--cols", type=int, default=2, help="Columns in contact sheet.")
    ap.add_argument("--thumb_w", type=int, default=640, help="Thumb width for contact sheet.")
    args = ap.parse_args()

    grids_dir = os.path.join(args.out, "grids")
    _ensure_dir(grids_dir)

    manifest = make_per_prompt_grids(
        runA=args.runA,
        runB=args.runB,
        out_dir=grids_dir,
        labelA=args.labelA,
        labelB=args.labelB,
        kA=args.kA,
        kB=args.kB,
        limit=args.limit,
    )
    print(f"[OK] Per-prompt grids -> {grids_dir}")
    print(f"[OK] Manifest         -> {manifest}")

    if args.contact:
        sheet_path = os.path.join(args.out, "contact_sheet.png")
        make_contact_sheet(grids_dir, sheet_path, cols=args.cols, thumb_w=args.thumb_w)
        print(f"[OK] Contact sheet    -> {sheet_path}")

if __name__ == "__main__":
    main()
