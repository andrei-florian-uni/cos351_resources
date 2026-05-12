# Given a directory of 16 PNG renders for a single part, plus the 3 
# reference images, this module sends them all to Claude in one API 
# call along with the classification prompt, parses the SUMMARY block 
# from the response, and returns a dict with the structured result.

import random
import anthropic
import base64
import os
import re
import time
from pathlib import Path

MODEL_NAME = "claude-sonnet-4-6"
MAX_TOKENS = 2048

# Retry behavior for API failures.
MAX_RETRIES = 3
INITIAL_BACKOFF_SECONDS = 10

# Path to the three reference images (mcd diagram + disconnector + selector).
# Override these if your reference files live somewhere else.
REFERENCE_IMAGES = [
    "references/reference_1.png",    # GOOD- part upright, arm extends latterally, side profile visible
    "references/reference_2.png",    # GOOD - part upright, arm extends latterally, side profile visible
    "references/reference_3.png",    # BAD - part flat, arm extends vertically, top profile visible
    "references/reference_4.png",    # BAD - part sideways, arm extends laterally away from camera, side profile visible
]

# The classification prompt used during manual testing.
TASK_DESCRIPTION = """
Respond with only the render label in the exact format specified. Do not write anything else until the SUMMARY block.

The first two images show the GOOD reference orientation: the part extends laterally at a 3/4 view, is upright, and no major features point toward or away from the camera.
The third and fourth images show BAD orientations to actively avoid: the third shows the arm pointing straight down with the part viewed from above, and the fourth shows the part tilted with the arm pointing away from the camera. Reject any render resembling these.
The remaining 16 images are renders of a different part labeled xNNN_yNNN_zNNN. Find the single render whose orientation most closely matches the reference images.
Before the SUMMARY block, you MUST first reason through your evaluation step by step: list each image, state whether it passes or fails the orientation criteria and why.
Then identify your top candidates and compare them. After completing your reasoning, end with the SUMMARY block and no text after it.

SUMMARY:
BEST_RENDER: xNNN_yNNN_zNNN
CONFIDENCE: NN
REASONING_TIER: clearly_superior_OR_top_with_close_alternatives_OR_multiple_close_OR_none_informative
"""

# Helper methods.

# Read an image file and return a dict in the format the API expects.
def encode_image_for_api(image_path):
    with open(image_path, "rb") as f:
        image_data = base64.standard_b64encode(f.read()).decode("utf-8")
    return {
        "type": "image",
        "source": {
            "type": "base64",
            "media_type": "image/png",
            "data": image_data,
        },
    }
def orientation_label_from_path(image_path):
    stem = Path(image_path).stem
    match = re.search(r"(x\d{3}_y\d{3}_z\d{3})", stem)
    if match:
        return match.group(0)
    raise ValueError(f"Could find orientation label from {image_path}")
    
# Extract the BEST_RENDER, CONFIDENCE, and REASONING_TIER values from
# the model's SUMMARY block at the end of its response.
def parse_summary_block(response_text):
    response_text = response_text.strip()
    best_match = re.search(r"BEST_RENDER:\s*(x\d{3}_y\d{3}_z\d{3})", response_text)
    conf_match = re.search(r"CONFIDENCE:\s*(\d+)", response_text)
    tier_match = re.search(
        r"REASONING_TIER:\s*\[?(clearly_superior|top_with_close_alternatives|multiple_close|none_informative)\]?",
        response_text,
        re.IGNORECASE
    )
    if not (best_match and conf_match and tier_match):
        print(f"\n[DEBUG] Full response:\n{response_text}\n")
        raise ValueError(
            "Could not parse SUMMARY block from response. Got:\n"
            f"{response_text[-500:]}"
        )
    return {
        "best_render": best_match.group(1),
        "confidence": int(conf_match.group(1)),
        "reasoning_tier": tier_match.group(1),
    }


def classify_part(render_paths, reference_paths=None, client=None):
    if reference_paths is None:
        reference_paths = REFERENCE_IMAGES
    if client is None:
        client = anthropic.Anthropic()

    render_paths = list(render_paths)
    random.shuffle(render_paths)

    content = []
    for ref in reference_paths:
        content.append(encode_image_for_api(ref))
    for render in render_paths:
        label = orientation_label_from_path(render)
        content.append({"type": "text", "text": f"Render: {label}"})
        content.append(encode_image_for_api(render))
    content.append({"type": "text", "text": TASK_DESCRIPTION})

    last_error = None
    for attempt in range(MAX_RETRIES):
        try:
            response = client.messages.create(
                model=MODEL_NAME,
                max_tokens=MAX_TOKENS,
                messages=[{"role": "user", "content": content}],
            )
            response_text = response.content[0].text
            parsed = parse_summary_block(response_text)
            parsed["raw_response"] = response_text
            return parsed
        except (anthropic.APIError, anthropic.APIConnectionError) as e:
            last_error = e
            backoff = INITIAL_BACKOFF_SECONDS * (2 ** attempt)
            print(f"    [WARN] API call failed (attempt {attempt + 1}): {e}")
            if attempt < MAX_RETRIES - 1:
                print(f"    Retrying in {backoff} seconds...")
                time.sleep(backoff)

    raise RuntimeError(f"All {MAX_RETRIES} API attempts failed. Last error: {last_error}")
# Usage:
# python classify_api.py <render_folder>
# Where <render_folder> contains 16 PNG renders for one part.

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python classify_api.py <render_folder>")
        sys.exit(1)

    folder = Path(sys.argv[1])
    renders = sorted(folder.glob("*.png"))
    if len(renders) != 16:
        print(f"Expected 16 PNGs in {folder}, found {len(renders)}.")
        sys.exit(1)

    print(f"Classifying {folder.name} via Claude API...")
    result = classify_part([str(p) for p in renders])
    print(f"\n--- RAW RESPONSE ---\n{result['raw_response']}")
    print(f"\nBEST_RENDER:    {result['best_render']}")
    print(f"CONFIDENCE:     {result['confidence']}")
    print(f"REASONING_TIER: {result['reasoning_tier']}")
