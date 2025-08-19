#!/usr/bin/env python3
from __future__ import annotations
import argparse, os, sys, subprocess, glob, time, csv, json
from typing import Dict, Any, List, Tuple, Optional
from omegaconf import OmegaConf


def has_manifest(run_dir: str) -> bool:
    manifest = os.path.join(run_dir, "generation.csv")
    try:
        return os.path.isfile(manifest) and os.path.getsize(manifest) > 0
    except OSError:
        return False
    
def log(msg: str) -> None:
    print(msg, flush=True)

def newest_run(prev: List[str]) -> str:
    after = sorted(glob.glob("artifacts/runs/*"))
    new = [p for p in after if p not in prev]
    if new:
        new.sort(key=lambda p: os.path.getmtime(p), reverse=True)
        return new[0]
    after.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return after[0]

def parse_window(raw: str) -> Tuple[int,int]:
    a,b = raw.split("-")
    return int(a), int(b)

def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)

def collect_metrics(run_dir: str) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    summ_path = os.path.join(run_dir, "metrics", "summary.json")
    if os.path.exists(summ_path):
        try:
            with open(summ_path) as f: out.update(json.load(f))
        except Exception: pass
    for p in [os.path.join(run_dir, "profile.json"),
              os.path.join(run_dir, "metrics", "profile.json")]:
        if os.path.exists(p):
            try:
                with open(p) as f: prof = json.load(f)
                if isinstance(prof, dict):
                    out["total_seconds"] = float(prof.get("total_seconds", out.get("total_seconds", 0.0)))
                    out["seconds_per_image"] = float(prof.get("seconds_per_image", out.get("seconds_per_image", 0.0)))
                    out["num_images"] = int(prof.get("num_images", out.get("num_images", 0)))
                break
            except Exception:
                pass
    return out

def eval_run(run_dir: str, device: str) -> None:
    if not has_manifest(run_dir):
        log(f"[SKIP] No generation.csv in {run_dir}; skipping evaluation.")
        return
    cmd = [sys.executable, "-m", "lpa2.runners.evaluate", "--run_dir", run_dir, "--device", device]
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError:
        if device.lower() != "cpu":
            log(f"[WARN] Eval failed on {device}; falling back to CPU for {run_dir}")
            subprocess.run([sys.executable, "-m", "lpa2.runners.evaluate",
                            "--run_dir", run_dir, "--device", "cpu"], check=True)
        else:
            raise


def backfill_evaluate_all(device: str) -> int:
    def already_evaluated(d: str) -> bool:
        # treat any of these as 'done'; adjust if your evaluate.py writes a different file
        for fname in ("metrics/summary.json", "evaluation.csv", "metrics.csv", "scores.csv"):
            if os.path.isfile(os.path.join(d, fname)):
                return True
        return False

    run_dirs = [d for d in glob.glob("artifacts/runs/*") if os.path.isdir(d)]
    if not run_dirs:
        log("[OK] No backfill needed (no runs found).")
        return 0

    todo = []
    for d in run_dirs:
        if not has_manifest(d):
            log(f"[SKIP] Backfill: {d} has no generation.csv")
            continue
        if already_evaluated(d):
            log(f"[SKIP] Backfill: {d} already evaluated")
            continue
        todo.append(d)

    if not todo:
        log("[OK] No backfill needed.")
        return 0

    log(f"[RUN] Backfill evaluate {len(todo)} run(s)...")
    for d in sorted(todo, key=os.path.getmtime):
        log(f" -> {d}")
        eval_run(d, device)
    log("[OK] Backfill complete.")
    return len(todo)

def save_tmp_cfg(base_cfg_path: str,
                 model: str,
                 preset: str,
                 window: Tuple[int,int],
                 parser: str,
                 prompts_file: Optional[str],
                 images_per_prompt: int,
                 seed: int,
                 n_prompts: int,
                 stub: str) -> str:
    cfg = OmegaConf.load(base_cfg_path)
    # Be defensive with keys; set only if present
    # Model name
    if "model" in cfg and "name" in cfg.model:
        cfg.model.name = model
    elif "model" in cfg:
        cfg.model["name"] = model
    else:
        cfg["model"] = {"name": model}

    # Schedule fields
    if "schedule" not in cfg: cfg["schedule"] = {}
    cfg.schedule["preset"] = preset
    cfg.schedule["timestep_window"] = [int(window[0]), int(window[1])]
    cfg.schedule["parser"] = parser

    # Data
    if "data" not in cfg: cfg["data"] = {}
    if prompts_file is not None:
        cfg.data["prompts_file"] = prompts_file
    if n_prompts and n_prompts > 0:
        cfg.data["n_prompts"] = int(n_prompts)

    # Sampling
    if "sampling" not in cfg: cfg["sampling"] = {}
    cfg.sampling["num_images_per_prompt"] = int(images_per_prompt)
    cfg.sampling["seed"] = int(seed)

    out_dir = "artifacts/tmp_bench_cfgs"
    ensure_dir(out_dir)
    tmp_path = os.path.join(out_dir, f"{stub}.yaml")
    OmegaConf.save(cfg, tmp_path)
    return tmp_path

def append_csv_row(out_csv: str, row: Dict[str, Any]) -> None:
    ensure_dir(os.path.dirname(out_csv))
    write_header = not os.path.exists(out_csv)
    # keep column order stable
    keys = list(row.keys())
    if os.path.exists(out_csv) and write_header is False:
        # reuse existing header order if possible
        with open(out_csv, "r", newline="") as f:
            try:
                rdr = csv.reader(f)
                header = next(rdr)
                if header: keys = header
            except Exception:
                pass
    with open(out_csv, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        if write_header: w.writeheader()
        w.writerow({k: row.get(k, "") for k in keys})

def main():
    ap = argparse.ArgumentParser(description="Robust benchmark runner (resume-friendly).")
    ap.add_argument("--cfg", required=True)
    ap.add_argument("--models", default="sdxl", help="Comma-separated: e.g., sdxl,sd15")
    ap.add_argument("--benchmark", choices=["all","custom","t2i","geneval"], default="all")
    ap.add_argument("--prompts_custom", default=None)
    ap.add_argument("--prompts_t2i", default=None)
    ap.add_argument("--prompts_geneval", default=None)
    ap.add_argument("--out", default="artifacts/runs/benchmark_summary.csv")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--images_per_prompt", type=int, default=1)
    ap.add_argument("--window", default="300-650")
    ap.add_argument("--parser", default="pos")
    ap.add_argument("--lpa_preset", default="lpa_late_only")
    ap.add_argument("--vanilla_preset", default="vanilla")
    ap.add_argument("--n_prompts", type=int, default=0)
    ap.add_argument("--eval_device", default="cpu", help="cpu|mps|cuda")
    ap.add_argument("--only_eval", action="store_true", help="Only backfill evaluate existing runs and exit")
    args = ap.parse_args()

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    a,b = parse_window(args.window)

    # Resolve which prompt sets to run
    bench_sets: List[Tuple[str, Optional[str]]] = []
    if args.benchmark in ("all","custom"):
        if not args.prompts_custom:
            ap.error("--prompts_custom required for custom")
        bench_sets.append(("custom", args.prompts_custom))
    if args.benchmark in ("all","t2i"):
        if not args.prompts_t2i:
            ap.error("--prompts_t2i required for t2i")
        bench_sets.append(("t2i", args.prompts_t2i))
    if args.benchmark in ("all","geneval"):
        if not args.prompts_geneval:
            ap.error("--prompts_geneval required for geneval")
        bench_sets.append(("geneval", args.prompts_geneval))

    # 0) Backfill evaluate anything already generated
    backfill_evaluate_all(args.eval_device)
    if args.only_eval:
        return

    # 1) Generate missing combos (we don't try to dedupe old runs; we just proceed)
    for model in models:
        for bench_name, prompt_path in bench_sets:
            for preset in [args.lpa_preset, args.vanilla_preset]:
                for seed in seeds:
                    stub = f"{bench_name}-{model}-{preset}-seed{seed}"
                    tmp_cfg = save_tmp_cfg(
                        base_cfg_path=args.cfg,
                        model=model,
                        preset=preset,
                        window=(a,b),
                        parser=args.parser,
                        prompts_file=prompt_path,
                        images_per_prompt=args.images_per_prompt,
                        seed=seed,
                        n_prompts=args.n_prompts,
                        stub=stub,
                    )
                    # Generate
                    before = sorted(glob.glob("artifacts/runs/*"))
                    log(f"[GEN] {bench_name} | {model} | {preset} | seed={seed}")
                    subprocess.run([sys.executable, "-m", "lpa2.runners.generate", "--cfg", tmp_cfg], check=True)
                    run_dir = newest_run(before)
                    log(f"[OK] Saved to {run_dir}")

                    # Evaluate with robust fallback
                    log(f"[EVAL] {run_dir} on {args.eval_device}")
                    eval_run(run_dir, args.eval_device)

                    # Collect and append a row
                    summ = collect_metrics(run_dir)
                    row = {
                        "run_dir": run_dir,
                        "benchmark": bench_name,
                        "model": model,
                        "preset": preset,
                        "window": f"{a}-{b}",
                        "parser": args.parser,
                        "seed": seed,
                        "eval_device": args.eval_device,
                    }
                    row.update(summ)
                    append_csv_row(args.out, row)
                    log(f"[CSV] appended -> {args.out}")

    log("[DONE] All planned runs generated + evaluated.")

if __name__ == "__main__":
    main()
