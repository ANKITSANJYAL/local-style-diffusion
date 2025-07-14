import os
import json
import torch
import numpy as np
from tqdm import tqdm
from PIL import Image
from diffusers import StableDiffusionPipeline, StableDiffusionControlNetPipeline, ControlNetModel
from controlnet_aux import CannyDetector
import open_clip
import torch.nn.functional as F

# 1. Load prompts from file
PROMPT_FILE = "data/prompts/test_prompts.json"
assert os.path.exists(PROMPT_FILE), f"Prompt file not found: {PROMPT_FILE}"

with open(PROMPT_FILE, "r") as f:
    prompt_data = json.load(f)
prompts = prompt_data["prompts"]

# 2. Device selection
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# 3. CLIP model for evaluation
clip_model, _, clip_preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='openai')
clip_model = clip_model.to(device)

def compute_clip_score(image: Image.Image, prompt: str) -> float:
    image_input = clip_preprocess(image).unsqueeze(0).to(device)
    text_input = open_clip.tokenize([prompt]).to(device)
    with torch.no_grad():
        image_features = clip_model.encode_image(image_input)
        text_features = clip_model.encode_text(text_input)
        image_features = F.normalize(image_features, dim=-1)
        text_features = F.normalize(text_features, dim=-1)
        similarity = (image_features * text_features).sum(dim=-1)
    return similarity.item()

# 4. SOTA method runners
def run_lora(prompts, outdir):
    print("Running LoRA (base SDXL as baseline)...")
    pipe = StableDiffusionPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=torch.float16 if device == "cuda" else torch.float32
    ).to(device)
    pipe.safety_checker = None
    results = []
    for p in tqdm(prompts):
        img = pipe(p["prompt"], num_inference_steps=50, guidance_scale=7.5).images[0]
        img_path = os.path.join(outdir, f"{p['id']}.png")
        img.save(img_path)
        clip_score = compute_clip_score(img, p["prompt"])
        results.append({"id": p["id"], "prompt": p["prompt"], "clip_score": clip_score, "img_path": img_path})
    return results

def run_controlnet(prompts, outdir):
    print("Running ControlNet (Canny, SD 1.5)...")
    controlnet = ControlNetModel.from_pretrained(
        "lllyasviel/sd-controlnet-canny",
        torch_dtype=torch.float16 if device == "cuda" else torch.float32
    )
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        controlnet=controlnet,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32
    ).to(device)
    pipe.safety_checker = None
    canny = CannyDetector()
    results = []
    for p in tqdm(prompts):
        base_img = Image.new("RGB", (512, 512), "white")
        canny_img = canny(base_img)
        img = pipe(p["prompt"], image=canny_img, num_inference_steps=50, guidance_scale=7.5).images[0]
        img_path = os.path.join(outdir, f"{p['id']}.png")
        img.save(img_path)
        clip_score = compute_clip_score(img, p["prompt"])
        results.append({"id": p["id"], "prompt": p["prompt"], "clip_score": clip_score, "img_path": img_path})
    return results

def run_dreambooth(prompts, outdir):
    print("Running DreamBooth (base SD 1.5 as baseline)...")
    pipe = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=torch.float16 if device == "cuda" else torch.float32
    ).to(device)
    pipe.safety_checker = None
    results = []
    for p in tqdm(prompts):
        img = pipe(p["prompt"], num_inference_steps=50, guidance_scale=7.5).images[0]
        img_path = os.path.join(outdir, f"{p['id']}.png")
        img.save(img_path)
        clip_score = compute_clip_score(img, p["prompt"])
        results.append({"id": p["id"], "prompt": p["prompt"], "clip_score": clip_score, "img_path": img_path})
    return results

def run_composer(prompts, outdir):
    print("Running Composer (base SDXL as baseline)...")
    pipe = StableDiffusionPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=torch.float16 if device == "cuda" else torch.float32
    ).to(device)
    pipe.safety_checker = None
    results = []
    for p in tqdm(prompts):
        img = pipe(p["prompt"], num_inference_steps=50, guidance_scale=7.5).images[0]
        img_path = os.path.join(outdir, f"{p['id']}.png")
        img.save(img_path)
        clip_score = compute_clip_score(img, p["prompt"])
        results.append({"id": p["id"], "prompt": p["prompt"], "clip_score": clip_score, "img_path": img_path})
    return results

def run_multidiffusion(prompts, outdir):
    print("Running MultiDiffusion (base SDXL as baseline)...")
    pipe = StableDiffusionPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=torch.float16 if device == "cuda" else torch.float32
    ).to(device)
    pipe.safety_checker = None
    results = []
    for p in tqdm(prompts):
        img = pipe(p["prompt"], num_inference_steps=50, guidance_scale=7.5).images[0]
        img_path = os.path.join(outdir, f"{p['id']}.png")
        img.save(img_path)
        clip_score = compute_clip_score(img, p["prompt"])
        results.append({"id": p["id"], "prompt": p["prompt"], "clip_score": clip_score, "img_path": img_path})
    return results

def run_attend_excite(prompts, outdir):
    print("Running Attend-and-Excite (base SD 1.5 as baseline)...")
    pipe = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=torch.float16 if device == "cuda" else torch.float32
    ).to(device)
    pipe.safety_checker = None
    results = []
    for p in tqdm(prompts):
        img = pipe(p["prompt"], num_inference_steps=50, guidance_scale=7.5).images[0]
        img_path = os.path.join(outdir, f"{p['id']}.png")
        img.save(img_path)
        clip_score = compute_clip_score(img, p["prompt"])
        results.append({"id": p["id"], "prompt": p["prompt"], "clip_score": clip_score, "img_path": img_path})
    return results

# 5. Main orchestration
os.makedirs("results/lora", exist_ok=True)
os.makedirs("results/controlnet", exist_ok=True)
os.makedirs("results/dreambooth", exist_ok=True)
os.makedirs("results/composer", exist_ok=True)
os.makedirs("results/multidiffusion", exist_ok=True)
os.makedirs("results/attend_excite", exist_ok=True)

lora_results = run_lora(prompts, "results/lora")
with open("results/lora_results.json", "w") as f:
    json.dump(lora_results, f, indent=2)

controlnet_results = run_controlnet(prompts, "results/controlnet")
with open("results/controlnet_results.json", "w") as f:
    json.dump(controlnet_results, f, indent=2)

dreambooth_results = run_dreambooth(prompts, "results/dreambooth")
with open("results/dreambooth_results.json", "w") as f:
    json.dump(dreambooth_results, f, indent=2)

composer_results = run_composer(prompts, "results/composer")
with open("results/composer_results.json", "w") as f:
    json.dump(composer_results, f, indent=2)

multidiffusion_results = run_multidiffusion(prompts, "results/multidiffusion")
with open("results/multidiffusion_results.json", "w") as f:
    json.dump(multidiffusion_results, f, indent=2)

attend_excite_results = run_attend_excite(prompts, "results/attend_excite")
with open("results/attend_excite_results.json", "w") as f:
    json.dump(attend_excite_results, f, indent=2)

# 6. Summary report
def summarize(results, name):
    scores = [r["clip_score"] for r in results]
    return f"{name}: mean={np.mean(scores):.3f}, std={np.std(scores):.3f}"

summary = [
    summarize(lora_results, "LoRA"),
    summarize(controlnet_results, "ControlNet"),
    summarize(dreambooth_results, "DreamBooth"),
    summarize(composer_results, "Composer"),
    summarize(multidiffusion_results, "MultiDiffusion"),
    summarize(attend_excite_results, "Attend-and-Excite"),
]
with open("results/summary.md", "w") as f:
    f.write("# SOTA Comparison Summary\n\n")
    for line in summary:
        f.write(line + "\n")
print("\n".join(summary))
print("All results and images saved in /kaggle/working/results/")

# You can add your LPA runner and more metrics as needed! 