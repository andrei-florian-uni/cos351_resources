import csv
import os
import shutil
import subprocess
import sys
from pathlib import Path
from classify_api import classify_part


# Paths to the existing standalone scripts. The script invokes them
# as subprocesses rather than reimplementing their logic.
ORIENT_SCRIPT = "orient_stl.py"
RENDER_SCRIPT = "render_stl.py"

# How orient_stl.py is invoked — Blender-mediated on Windows.
BLENDER_PATH = r"C:\Program Files\Blender Foundation\Blender 5.1\blender.exe"

# Output CSV columns. Each row records one part's classification result.
CSV_COLUMNS = [
    "file_name",
    "best_render",
    "confidence",
    "reasoning_tier",
    "status",   # "ok" or "error" — tracks per-part success across runs.
]

# Run orient_stl.py to get 16 oriented STLs per source file.
def stage_orient(input_dir, oriented_dir):
    print(f"\n Orienting STLs from {input_dir} into {oriented_dir}")
    cmd = [BLENDER_PATH, "--background", "--python", ORIENT_SCRIPT,
           "--", str(input_dir), str(oriented_dir)]
    subprocess.run(cmd, check=True)

# Run render_stl.py to convert each STL to a PNG.
def stage_render(oriented_dir, rendered_dir):
    print(f"\n[Stage 2/4] Rendering STLs from {oriented_dir} into {rendered_dir}")
    cmd = [sys.executable, RENDER_SCRIPT, str(oriented_dir), str(rendered_dir)]
    subprocess.run(cmd, check=True)

# Copy each render to an anonymized folder, stripping the
# part-identifying prefix from each filename so only the rotation is left.
def stage_anonymize(rendered_dir, anonymized_dir):
    print(f"\n[Stage 3/4] Anonymizing renders into {anonymized_dir}")
    rendered_dir = Path(rendered_dir)
    anonymized_dir = Path(anonymized_dir)
    anonymized_dir.mkdir(exist_ok=True, parents=True)

    for part_folder in rendered_dir.iterdir():
        if not part_folder.is_dir():
            continue
        out_folder = anonymized_dir / part_folder.name
        out_folder.mkdir(exist_ok=True)

        for png in part_folder.glob("*.png"):
            # Find the rotation part (xNNN_yNNN_zNNN.png) and use that alone.
            stem = png.stem
            # Find the last token starting with "x" followed by digits.
            parts = stem.split("_")
            rotation_start = next(
                (i for i, p in enumerate(parts)
                 if p.startswith("x") and len(p) == 4 and p[1:].isdigit()),
                None,
            )
            new_name = "_".join(parts[rotation_start:]) + ".png"
            shutil.copy2(png, out_folder / new_name)

# For each part folder, send the 16 anonymized renders to
# Claude and append a row to the results CSV.
def stage_classify(anonymized_dir, csv_path):
    print(f"\n[Stage 4/4] Classifying parts via Claude API")
    anonymized_dir = Path(anonymized_dir)
    csv_path = Path(csv_path)

    # Open the CSV in append mode so progress is preserved on interruption.
    write_header = not csv_path.exists()
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        if write_header:
            writer.writeheader()

        for part_folder in sorted(anonymized_dir.iterdir()):
            if not part_folder.is_dir():
                continue

            renders = sorted(part_folder.glob("*.png"))
            if len(renders) != 16:
                print(f"  [SKIP] {part_folder.name}: expected 16 PNGs, "
                      f"found {len(renders)}")
                writer.writerow({
                    "file_name": part_folder.name,
                    "best_render": "", "confidence": "", "reasoning_tier": "",
                    "status": f"error: {len(renders)} renders",
                })
                continue

            try:
                print(f"  Classifying {part_folder.name}...")
                result = classify_part([str(p) for p in renders])
                writer.writerow({
                    "file_name": part_folder.name,
                    "best_render": result["best_render"],
                    "confidence": result["confidence"],
                    "reasoning_tier": result["reasoning_tier"],
                    "status": "ok",
                })
                f.flush()  # Ensure each row hits disk before the next API call.
                print(f"    -> {result['best_render']} "
                      f"(conf {result['confidence']}, "
                      f"{result['reasoning_tier']})")
            except Exception as e:
                print(f"    [ERROR] {part_folder.name}: {e}")
                writer.writerow({
                    "file_name": part_folder.name,
                    "best_render": "", "confidence": "", "reasoning_tier": "",
                    "status": f"error: {e}",
                })
                f.flush()


# Usage: python pipeline.py <input_dir> <work_dir>
# All four stages run in sequence on every STL in <input_dir>.

if len(sys.argv) != 3:
    print("Usage: python pipeline.py <input_dir> <work_dir>")
    sys.exit(1)

input_dir = Path(sys.argv[1])
work_dir = Path(sys.argv[2])
work_dir.mkdir(exist_ok=True)

oriented_dir = work_dir / "oriented"
rendered_dir = work_dir / "rendered"
anonymized_dir = work_dir / "anonymized"
csv_path = work_dir / "results.csv"

stage_orient(input_dir, oriented_dir)
stage_render(oriented_dir, rendered_dir)
stage_anonymize(rendered_dir, anonymized_dir)
stage_classify(anonymized_dir, csv_path)

print(f"\n=========================================")
print(f"Pipeline complete. Results saved to {csv_path}")
