from __future__ import annotations
import argparse, os, json, subprocess, sys, time, glob, csv
from typing import List, Tuple
from omegaconf import OmegaConf

def parse_windows(raw: str) -> List[Tuple[int, int]]:
    out: List[Tuple[int, int]] = []
    for chunk in raw.split(","):
        a, b = chunk.split("-")
        out.append((int(a), int(b)))
    return out

def run_module(mod: str, args: List[str]) -> None:
    cmd = [sys.executable, "-m", mod] + args
    subprocess.run(cmd, check=True)

def newest_run(prev: List[str]) -> str:
    after = sorted(glob.glob("artifacts/runs/*"))
    # pick the first path that wasn't in prev; fallback to newest by mtime
    new = [p for p in after if p not in prev]
    if new:
        # some filesystems don’t guarantee sorted order; pick the newest among new
        new.sort(key=lambda p: os.path.getmtime(p), reverse=True)
        return new[0]
    # fallback
    after.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return after[0]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", default="lpa2/configs/default.yaml")
    ap.add_argument("--presets", required=True, help="comma-separated")
    ap.add_argument("--windows", required=True, help="e.g., 200-700,300-650")
    ap.add_argument("--parsers", default="spacy,pos,naive")
    ap.add_argument(
        "--n_prompts",
        type=int,
        default=20,
        help="If 0 or negative, use ALL prompts from the source file. "
             "If >0 and smaller than file size, a sliced TEMP file is used."
    )
    ap.add_argument(
        "--prompts_file",
        default=None,
        help="Optional: path to prompts JSON to use. If omitted, uses cfg.data.prompts_file."
    )
    args = ap.parse_args()

    cfg = OmegaConf.load(args.cfg)
    presets = [p for p in args.presets.split(",") if p]
    windows = parse_windows(args.windows)
    parsers = [p for p in args.parsers.split(",") if p]

    # Choose the source prompts file (CLI wins over cfg)
    use_prompts_path = args.prompts_file or cfg.data.prompts_file

    # Load prompts once and (optionally) slice for speed.
    with open(use_prompts_path) as f:
        all_prompts = json.load(f)

    # Only slice if explicitly requested and smaller than file
    if isinstance(all_prompts, list) and args.n_prompts > 0 and len(all_prompts) > args.n_prompts:
        tmp_path = os.path.join("artifacts", "tmp", "prompts_slice.json")
        os.makedirs(os.path.dirname(tmp_path), exist_ok=True)
        with open(tmp_path, "w") as f:
            json.dump(all_prompts[: args.n_prompts], f, indent=2)
        use_prompts_path = tmp_path  # point runs to the sliced list

    out_rows = []
    for preset in presets:
        for (a, b) in windows:
            for parser in parsers:
                # build a temp cfg for this cell
                cfg.schedule.preset = preset
                cfg.schedule.timestep_window = [a, b]
                cfg.schedule.parser = parser
                cfg.data.prompts_file = use_prompts_path
                tmp_cfg_path = "lpa2/configs/_tmp_run.yaml"
                os.makedirs(os.path.dirname(tmp_cfg_path), exist_ok=True)
                OmegaConf.save(cfg, tmp_cfg_path)

                before = sorted(glob.glob("artifacts/runs/*"))

                # generate
                run_module("lpa2.runners.generate", ["--cfg", tmp_cfg_path])

                run_dir = newest_run(before)

                # evaluate (device kept explicit to avoid MPS/CUDA assumptions)
                run_module("lpa2.runners.evaluate", ["--run_dir", run_dir, "--device", "cpu"])

                # collect summary
                with open(os.path.join(run_dir, "metrics", "summary.json")) as f:
                    summ = json.load(f)
                out_rows.append({
                    "run_dir": run_dir,
                    "preset": preset,
                    "window": f"{a}-{b}",
                    "parser": parser,
                    **summ,
                })
                print(f"[OK] {preset} {a}-{b} {parser} -> {run_dir}")

    if out_rows:
        ab_path = os.path.join("artifacts", "runs", "ablation_summary.csv")
        os.makedirs(os.path.dirname(ab_path), exist_ok=True)
        with open(ab_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
            w.writeheader()
            w.writerows(out_rows)
        print(f"[OK] ablation table -> {ab_path}")

if __name__ == "__main__":
    main()
