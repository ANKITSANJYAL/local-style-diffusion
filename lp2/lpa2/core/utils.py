from __future__ import annotations
import os, json, random, hashlib
from datetime import datetime
from typing import Any, Dict, Tuple
import numpy as np
import torch

def set_seed(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def make_run_dir(root: str) -> str:
    run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    out = os.path.join(root, run_id)
    for sub in ["images", "metrics", "logs"]:
        os.makedirs(os.path.join(out, sub), exist_ok=True)
    return out

def short_hash(s: str) -> str:
    return hashlib.sha1(s.encode()).hexdigest()[:8]

def save_json(path: str, obj: Dict[str, Any]) -> None:
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)

def save_text(path: str, text: str) -> None:
    with open(path, "w") as f:
        f.write(text)

def resolved_config(cfg: Any) -> Dict[str, Any]:
    # OmegaConf to plain dict if needed
    try:
        from omegaconf import OmegaConf
        if isinstance(cfg, OmegaConf.__class__):
            return OmegaConf.to_container(cfg, resolve=True)  # type: ignore[arg-type]
    except Exception:
        pass
    return cfg  # type: ignore[return-value]
