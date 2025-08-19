from __future__ import annotations
from typing import Dict
import torch
from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler, EulerAncestralDiscreteScheduler

SCHED = {
    "dpm": DPMSolverMultistepScheduler,
    "euler_a": EulerAncestralDiscreteScheduler,
}

def load_pipe(repo_id: str, scheduler: str, device: str = "cuda:0", dtype: str = "fp16") -> DiffusionPipeline:
    use_fp16 = dtype == "fp16" and (torch.cuda.is_available() or torch.backends.mps.is_available())
    torch_dtype = torch.float16 if use_fp16 else torch.float32
    pipe = DiffusionPipeline.from_pretrained(repo_id, torch_dtype=torch_dtype, safety_checker=None)
    pipe.scheduler = SCHED[scheduler].from_config(pipe.scheduler.config)
    pipe.to(device)
    pipe.set_progress_bar_config(disable=True)
    return pipe
