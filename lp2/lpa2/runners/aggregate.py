from __future__ import annotations
import argparse, os, json, csv
from typing import List, Dict

def load_summary(run_dir: str) -> Dict:
    p = os.path.join(run_dir, "metrics", "summary.json")
    with open(p) as f: return json.load(f)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    rows = []
    for rd in args.runs:
        cfgp = os.path.join(rd, "config.yaml")
        cfg = {}
        try:
            import yaml
            with open(cfgp) as f: cfg = yaml.safe_load(f)
        except Exception:
            pass
        summ = load_summary(rd)
        rows.append({
            "run_dir": rd,
            "model": cfg.get("model", {}).get("name", ""),
            "preset": cfg.get("schedule", {}).get("preset", ""),
            **summ
        })

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"[OK] wrote {args.out}")

if __name__ == "__main__":
    main()
