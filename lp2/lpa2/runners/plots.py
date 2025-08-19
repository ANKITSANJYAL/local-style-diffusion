from __future__ import annotations
import argparse, json, os, glob
import matplotlib.pyplot as plt  # simple defaults; no styles/colors set

def plot_runtime(profiles: list[str], out: str):
    labels, times, mems = [], [], []
    for p in profiles:
        try:
            with open(p) as f: d = json.load(f)
            labels.append(os.path.basename(os.path.dirname(p)))
            times.append(d.get("time_sec", 0))
            mems.append(d.get("peak_mem_mb", 0))
        except Exception:
            pass
    if not labels: 
        print("No profiles found."); return
    plt.figure()
    plt.bar(range(len(labels)), times)
    plt.xticks(range(len(labels)), labels, rotation=90)
    plt.ylabel("Time (s)")
    plt.title("Runtime per Run")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    plt.tight_layout(); plt.savefig(out, dpi=200); plt.close()
    print(f"[OK] wrote {out}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--profiles", nargs="*")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    if args.profiles:
        if len(args.profiles) == 1 and "*" in args.profiles[0]:
            args.profiles = glob.glob(args.profiles[0])
        plot_runtime(args.profiles, args.out)

if __name__ == "__main__":
    main()
