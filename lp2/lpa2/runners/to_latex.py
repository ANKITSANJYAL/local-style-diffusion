from __future__ import annotations
import argparse, csv, os

def fmt(x, n=3): 
    try: return f"{float(x):.3f}"
    except: return str(x)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    rows = list(csv.DictReader(open(args.csv)))
    # Sort for nice display (SDXL first, then SD1.5; LPA before vanilla)
    def key(r): return (r["model"], 0 if "lpa" in r["preset"] else 1)
    rows.sort(key=key)

    tex = []
    tex.append(r"""\begin{table}[t]
\centering
\small
\begin{tabular}{l l c c c}
\toprule
Backbone & Method & CLIP$\uparrow$ & Style-CLIP$\uparrow$ & LPIPS$\uparrow$ \\
\midrule""")
    for r in rows:
        model = r["model"].upper()
        method = "LPA" if "lpa" in r["preset"] else "Vanilla"
        clipm = fmt(r["clip_prompt_mean"])
        stylem = fmt(r["clip_style_mean"])
        divm = fmt(r["div_lpips_mean"])
        tex.append(f"{model} & {method} & {clipm} & {stylem} & {divm} \\\\")
    tex.append(r"""\bottomrule
\end{tabular}
\caption{Text alignment (CLIP), style alignment (Style-CLIP), and diversity (LPIPS). Higher is better.}
\label{tab:main}
\end{table}""")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f: f.write("\n".join(tex))
    print(f"[OK] wrote {args.out}")

if __name__ == "__main__":
    main()
