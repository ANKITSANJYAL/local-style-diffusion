
#!/usr/bin/env python3
from __future__ import annotations
import argparse, os, json, subprocess, sys, random
from pathlib import Path

T2I_REPO = "https://github.com/Karine-Huang/T2I-CompBench.git"
GENEVAL_REPO = "https://github.com/djghosh13/geneval.git"

def sh(cmd: list[str], cwd: str | None = None) -> None:
    subprocess.run(cmd, cwd=cwd, check=True)

def ensure_repo(url: str, dest: Path) -> Path:
    if dest.exists():
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    sh(["git", "clone", "--depth", "1", url, str(dest)])
    return dest

def load_t2i_prompts(repo_dir: Path) -> dict[str, list[str]]:
    """
    Reads the official *_val.txt lists under examples/dataset/:
      color_val.txt, shape_val.txt, texture_val.txt,
      spatial_val.txt, non_spatial_val.txt, complex_val.txt
    Returns dict: {category: [prompts...]}
    """
    ds_dir = repo_dir / "examples" / "dataset"
    mapping = {
        "color_binding": "color_val.txt",
        "shape_binding": "shape_val.txt",
        "texture_binding": "texture_val.txt",
        "spatial_relationship": "spatial_val.txt",
        "non_spatial_relationship": "non_spatial_val.txt",
        "complex_composition": "complex_val.txt",
    }
    out: dict[str, list[str]] = {}
    for cat, fname in mapping.items():
        fpath = ds_dir / fname
        if not fpath.exists():
            raise FileNotFoundError(f"Missing {fpath}. Repo layout may have changed.")
        prompts = [ln.strip() for ln in fpath.read_text(encoding="utf-8").splitlines() if ln.strip()]
        out[cat] = prompts
    return out

def sample_t2i_subset(t2i: dict[str, list[str]], per_cat: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    subset = []
    for cat, prompts in t2i.items():
        picks = prompts if len(prompts) <= per_cat else rng.sample(prompts, per_cat)
        subset.extend({"prompt": p, "category": cat} for p in picks)
    return subset

def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)

def load_geneval_prompts(repo_dir: Path) -> list[dict]:
    """
    GenEval keeps a prompts/evaluation_metadata.jsonl with task tags.
    We'll parse that and export a list of objects with {'prompt', 'task'}.
    """
    meta = repo_dir / "prompts" / "evaluation_metadata.jsonl"
    if not meta.exists():
        raise FileNotFoundError(f"Missing {meta}. Repo layout may have changed.")
    rows = []
    with meta.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            prompt = rec.get("prompt") or rec.get("caption") or ""
            task = rec.get("task") or ""
            if prompt:
                rows.append({"prompt": prompt, "task": task})
    return rows

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="bench")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--per_cat", type=int, default=20, help="T2I-CompBench prompts per sub-category")
    ap.add_argument("--repos_dir", default=".bench_repos", help="Where to clone repos")
    args = ap.parse_args()

    repos_dir = Path(args.repos_dir)
    outdir = Path(args.outdir)

    # T2I-CompBench subset
    t2i_dir = ensure_repo(T2I_REPO, repos_dir / "T2I-CompBench")
    t2i_all = load_t2i_prompts(t2i_dir)
    t2i_subset = sample_t2i_subset(t2i_all, args.per_cat, args.seed)
    write_json(outdir / "t2i_compbench_subset_120.json", t2i_subset)

    # GenEval prompts
    geneval_dir = ensure_repo(GENEVAL_REPO, repos_dir / "geneval")
    geneval_prompts = load_geneval_prompts(geneval_dir)
    write_json(outdir / "geneval_prompts.json", geneval_prompts)

    print("[OK] Wrote:")
    print(" -", outdir / "t2i_compbench_subset_120.json")
    print(" -", outdir / "geneval_prompts.json")

if __name__ == "__main__":
    main()
