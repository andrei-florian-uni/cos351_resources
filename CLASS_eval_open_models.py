# %% [markdown]
# **Evaluate Models**
# 
# Loads a single model from `MODEL_DIR` and the HF datasets from `DATA_DIR`, constructs interleaved 
# image+metadata prompts per the MCD task specification, and runs greedy inference over the full eval set. 
# Edit `ACTIVE_MODEL` and `SHOT` at the top of the config block to select the run; all other logic is 
# identical across models. Results are written to `EVAL_DIR` as `{model}-{shot}-shot.csv` with columns 
# `file_name`, `label`, `score`. Unparseable outputs are logged separately rather than silently dropped.
#
# InternVL2.5 uses a separate loading and inference path (AutoModel + model.chat()) due to its
# custom architecture; LLaVA-OV and Llama 3.2 Vision use AutoModelForVision2Seq + apply_chat_template.
# InternVL2.5 requires dynamic_preprocess tiling with the official find_closest_aspect_ratio logic,
# and num_patches_list must be passed to model.chat() so it correctly expands <image> tokens per image.

# %%
import os
import io
import re
import json
import base64
from pathlib import Path

import torch
import pandas as pd
from PIL import Image as PILImage
from torchvision import transforms
from tqdm import tqdm
from datasets import load_from_disk
from transformers import (
    AutoProcessor,
    AutoTokenizer,
    AutoModel,
    AutoModelForVision2Seq,
    BitsAndBytesConfig,
)

# %%
# ── Config ────────────────────────────────────────────────────────────────────
ACTIVE_MODEL = "llama32_vision_11b"   # internvl2_5_8b | llava_onevision_7b | llama32_vision_11b
SHOT         = 8                   # 0 | 4 | 8

# %%
MODEL_DIR = Path("/scratch/gpfs/FELLBAUM/af3158/cos351/open_weight_n_shot/models")
DATA_DIR  = Path("/scratch/gpfs/FELLBAUM/af3158/cos351/open_weight_n_shot/hf_dataset")
EVAL_DIR  = Path("/scratch/gpfs/FELLBAUM/af3158/cos351/open_weight_n_shot/evals")
HF_HOME   = Path("/scratch/gpfs/FELLBAUM/af3158/.cache/huggingface")

os.environ["HF_HOME"]               = str(HF_HOME)
os.environ["TRANSFORMERS_CACHE"]    = str(HF_HOME / "transformers")
os.environ["HF_DATASETS_CACHE"]     = str(HF_HOME / "datasets")
os.environ["TRANSFORMERS_OFFLINE"]  = "1"
os.environ["HF_DATASETS_OFFLINE"]   = "1"

EVAL_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = MODEL_DIR / ACTIVE_MODEL
assert MODEL_PATH.exists(), f"Model not found at {MODEL_PATH} — run 01_download_models.py first"

TASK_DESCRIPTION = (
    "Respond with a single integer from 0 to 100. Do not write anything else.\n\n"
    "A Machine Gun Conversion Device (MCD) is a part that modifies a semi-automatic "
    "firearm to fire automatically. You will be shown a 3D model render accompanied by "
    "structured metadata. When present, items labelled \"role\": \"example\" are annotated "
    "reference cases with a known \"label\" (1 = MCD, 0 = not MCD). The item labelled "
    "\"role\": \"eval\" is the item to classify.\n\n"
    "Estimate the probability that the eval item is an MCD. "
    "0 = almost certainly not an MCD. 100 = almost certainly an MCD. 50 = maximum uncertainty.\n"
    "Integer:"
)

# ── Data ──────────────────────────────────────────────────────────────────────
shot_ds = load_from_disk(str(DATA_DIR / f"{SHOT}shot")) if SHOT > 0 else []
eval_ds = load_from_disk(str(DATA_DIR / "eval"))

print(f"Model          : {ACTIVE_MODEL}")
print(f"Shot regime    : {SHOT}-shot  ({len(shot_ds)} exemplars)")
print(f"Eval items     : {len(eval_ds)}")

# ── Model ─────────────────────────────────────────────────────────────────────
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

COMPUTE_DTYPE = {
    "llama32_vision_11b": torch.bfloat16,
    "llava_onevision_7b": torch.float16,
    "internvl2_5_8b":     torch.bfloat16,
}

quant_cfg = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=COMPUTE_DTYPE[ACTIVE_MODEL],
    bnb_4bit_use_double_quant=True,
)

if ACTIVE_MODEL == "internvl2_5_8b":
    tokenizer = AutoTokenizer.from_pretrained(
        str(MODEL_PATH),
        trust_remote_code=True,
        local_files_only=True,
        use_fast=False,
    )
    model = AutoModel.from_pretrained(
        str(MODEL_PATH),
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        local_files_only=True,
        low_cpu_mem_usage=True,
    ).eval()
else:
    tokenizer = None
    processor = AutoProcessor.from_pretrained(
        str(MODEL_PATH),
        trust_remote_code=True,
        local_files_only=True,
    )
    model = AutoModelForVision2Seq.from_pretrained(
        str(MODEL_PATH),
        quantization_config=quant_cfg,
        device_map="auto",
        trust_remote_code=True,
        local_files_only=True,
    ).eval()

model.eval()
print(f"✓ Model loaded on {DEVICE}")

# ── InternVL image preprocessing (official implementation) ────────────────────
INTERNVL_MEAN     = (0.485, 0.456, 0.406)
INTERNVL_STD      = (0.229, 0.224, 0.225)
INTERNVL_IMG_SIZE = 448

internvl_transform = transforms.Compose([
    transforms.Lambda(lambda img: img.convert("RGB")),
    transforms.Resize((INTERNVL_IMG_SIZE, INTERNVL_IMG_SIZE),
                      interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize(mean=INTERNVL_MEAN, std=INTERNVL_STD),
])

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    """Official InternVL implementation — considers image area, not just ratio diff."""
    best_ratio_diff = float("inf")
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio

def dynamic_preprocess(image, min_num=1, max_num=6, image_size=448, use_thumbnail=True):
    """
    Official InternVL implementation. Returns a list of PIL tile images.
    len(tiles) is the value to use in num_patches_list for this image.
    """
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    target_ratios = set(
        (i, j)
        for n in range(min_num, max_num + 1)
        for i in range(1, n + 1)
        for j in range(1, n + 1)
        if min_num <= i * j <= max_num
    )
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size
    )

    target_width  = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks        = target_aspect_ratio[0] * target_aspect_ratio[1]

    resized = image.resize((target_width, target_height))
    tiles   = []
    for i in range(blocks):
        col = i % target_aspect_ratio[0]
        row = i // target_aspect_ratio[0]
        box = (col * image_size, row * image_size,
               (col + 1) * image_size, (row + 1) * image_size)
        tiles.append(resized.crop(box))

    if use_thumbnail and len(tiles) != 1:
        tiles.append(image.resize((image_size, image_size), PILImage.BICUBIC))

    return tiles


def process_image_internvl(pil_img):
    """Returns (pixel_values tensor, n_tiles int)."""
    tiles  = dynamic_preprocess(pil_img.convert("RGB"))
    tensor = torch.stack([internvl_transform(t) for t in tiles])
    return tensor, len(tiles)

# ── Prompt builders ───────────────────────────────────────────────────────────
def build_prompt(shot_ds, eval_item):
    """Standard interleaved blocks format for LLaVA-OV and Llama 3.2 Vision."""
    blocks = [{"type": "text", "text": TASK_DESCRIPTION}]
    images = []

    for ex in shot_ds:
        images.append(ex["image"].convert("RGB"))
        meta = json.dumps({
            "role":        "example",
            "label":       int(ex["label"]),
            "description": ex["description"],
            "comments":    ex["key_comments"],
        })
        blocks.append({"type": "image"})
        blocks.append({"type": "text", "text": meta})

    images.append(eval_item["image"].convert("RGB"))
    eval_meta = json.dumps({
        "role":        "eval",
        "description": eval_item["description"],
        "comments":    eval_item["key_comments"],
    })
    blocks.append({"type": "image"})
    blocks.append({"type": "text", "text": eval_meta})

    return blocks, images


def build_prompt_internvl(shot_ds, eval_item):
    transform = transforms.Compose([
        transforms.Lambda(lambda img: img.convert("RGB")),
        transforms.Resize((448, 448), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])

    tiles            = []
    num_patches_list = []
    text_parts       = [TASK_DESCRIPTION]
    img_idx          = 1

    for ex in shot_ds:
        tiles.append(transform(ex["image"].convert("RGB")))
        num_patches_list.append(1)
        meta = json.dumps({
            "role":        "example",
            "label":       int(ex["label"]),
            "description": ex["description"],
            "comments":    ex["key_comments"],
        })
        text_parts.append(f"Image-{img_idx}: <image>\n{meta}")
        img_idx += 1

    tiles.append(transform(eval_item["image"].convert("RGB")))
    num_patches_list.append(1)
    eval_meta = json.dumps({
        "role":        "eval",
        "description": eval_item["description"],
        "comments":    eval_item["key_comments"],
    })
    text_parts.append(f"Image-{img_idx}: <image>\n{eval_meta}")

    question     = "\n".join(text_parts)
    pixel_values = torch.stack(tiles).to(torch.bfloat16).to(DEVICE)

    return question, pixel_values, num_patches_list

# ── Inference ─────────────────────────────────────────────────────────────────
def parse_score(raw):
    raw = raw.strip()
    matches = re.findall(r"\b(100|[1-9][0-9]?|0)\b", raw)
    return int(matches[0]) if matches else None


def run_inference_internvl(question, pixel_values, num_patches_list):
    gen_config = dict(max_new_tokens=256, do_sample=False)
    return model.chat(
        tokenizer,
        pixel_values,
        question,
        gen_config,
        num_patches_list=num_patches_list,
        history=None,
        return_history=False,
    )


def run_inference(blocks, images):
    messages = [{"role": "user", "content": blocks}]
    prompt   = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs   = processor(text=prompt, images=images, return_tensors="pt").to(DEVICE)

    if "pixel_values" in inputs:
        inputs["pixel_values"] = inputs["pixel_values"].to(COMPUTE_DTYPE[ACTIVE_MODEL])

    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=False,
            temperature=None,
            top_p=None,
        )

    input_len  = inputs["input_ids"].shape[1]
    new_tokens = output_ids[0][input_len:]
    return processor.decode(new_tokens, skip_special_tokens=True).strip()

# ── Eval loop ─────────────────────────────────────────────────────────────────
records     = []
unparseable = []

for i, eval_item in enumerate(tqdm(eval_ds, desc=f"{ACTIVE_MODEL} {SHOT}-shot")):
    try:
        if ACTIVE_MODEL == "internvl2_5_8b":
            question, pixel_values, num_patches_list = build_prompt_internvl(shot_ds, eval_item)
            raw = run_inference_internvl(question, pixel_values, num_patches_list)
        else:
            blocks, images = build_prompt(shot_ds, eval_item)
            raw = run_inference(blocks, images)
        score = parse_score(raw)
    except Exception as e:
        raw   = f"ERROR: {e}"
        score = None
        print(f"\n  ✗  index {i} ({eval_item['file_name']}): {e}")

    if score is None:
        unparseable.append({"index": i, "file_name": eval_item["file_name"], "raw": raw})

    records.append({
        "file_name": eval_item["file_name"],
        "label":     int(eval_item["label"]),
        "score":     score,
    })

# ── Output ────────────────────────────────────────────────────────────────────
results_df = pd.DataFrame(records)[["file_name", "label", "score"]]

out_csv = EVAL_DIR / f"{ACTIVE_MODEL}-{SHOT}-shot.csv"
results_df.to_csv(out_csv, index=False)
print(f"\n✓ Results → {out_csv}")
print(f"  {len(records)} items | {len(unparseable)} unparseable")

if unparseable:
    log_path = EVAL_DIR / f"{ACTIVE_MODEL}_{SHOT}shot_unparseable.json"
    with open(log_path, "w") as f:
        json.dump(unparseable, f, indent=2)
    print(f"  Unparseable log → {log_path}")