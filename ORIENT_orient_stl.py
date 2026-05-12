import bpy
import sys
import math         
import os

# The 15 orientations to render per STL, as XYZ triples in degrees. 
# Blender applies these as rotation_euler with rotation_mode='XYZ',
# which means R = Rz @ Ry @ Rx (X first, then Y, then Z).
#
# Y is held at 0 throughout these orientations. Intital renders 
# typically start as vertically-pointing or horizontally-pointing 
# thus X & Z axis are rotated.

ORIENTATIONS = [
    (  0, 0,  90),
    (  0, 0, 180),
    (  0, 0, 270),
    ( 90, 0,   0),
    (90, 0, 90),
    ( 90, 0, 180),
    (90, 0, 270),
    (180, 0,   0),
    (180, 0,  90),
    (180, 0, 180),
    (180, 0, 270),
    (270, 0,   0),
    (270, 0, 90),
    (270, 0, 180),
    (270, 0, 270)
]

# Save copy of original STL in its respective oriented folder for
# classification purposes.
SAVE_ORIGINAL_COPY = True

# Helper methods.

def clear_scene():
    # Reset Blender to factory defaults with an empty scene.
    bpy.ops.wm.read_factory_settings(use_empty=True)

def import_stl(filepath):
    # Import an STL file and return the resulting Blender object.
    bpy.ops.wm.stl_import(filepath=filepath)
    return bpy.context.active_object

def center_object(obj):
    # Move the object so its resides on the origin (0, 0, 0).
    bpy.ops.object.origin_set(type='ORIGIN_GEOMETRY', center='BOUNDS')
    obj.location = (0, 0, 0)

def normalize_scale(obj):
    # Retieve STL dimensions, and choose max value to scale dimensions
    # so that the last largest side is 1.
    dims = obj.dimensions
    max_dim = max(dims.x, dims.y, dims.z)
    if max_dim > 0:
        scale_factor = 1.0 / max_dim
        # Set scale and apply to the STL file.
        obj.scale = (scale_factor, scale_factor, scale_factor)
        bpy.ops.object.transform_apply(scale=True)

def apply_orientation(obj, euler_deg):
    # Derive the degree values and convert to radians for Blender API.
    x_deg = euler_deg[0]
    y_deg = euler_deg[1]
    z_deg = euler_deg[2]
    rad_x = math.radians(x_deg)
    rad_y = math.radians(y_deg)
    rad_z = math.radians(z_deg)
    # Set rotating convention/order and rotation values and apply to STL.
    obj.rotation_mode = 'XYZ'
    obj.rotation_euler = (rad_x, rad_y, rad_z)
    bpy.ops.object.transform_apply(rotation=True)

def export_stl(obj, filepath):
    # Unselect any objects, select the designated STL to export.
    bpy.ops.object.select_all(action='DESELECT')
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    # Use Blender export function.
    bpy.ops.wm.stl_export(filepath=filepath, 
                          export_selected_objects=True,
                          apply_modifiers=False)

def process_stl(input_path, output_dir):
    # Obtain the filename without STL extension. Create a subfolder with
    # the current STL as a leaf on the output directory.
    base = os.path.splitext(os.path.basename(input_path))[0]
    stl_folder = os.path.join(output_dir, base)
    os.makedirs(stl_folder, exist_ok=True)

    print(f"\n=== Processing: {input_path}")
    print(f"    Output folder: {stl_folder}")
    print(f"    Orientations: {len(ORIENTATIONS)}")

    # Save the centered, normalized original STL.
    if SAVE_ORIGINAL_COPY:
        clear_scene()
        obj = import_stl(input_path)
        center_object(obj)
        normalize_scale(obj)
        orig_path = os.path.join(stl_folder, f"{base}_x000_y000_z000.stl")
        export_stl(obj, orig_path)
        print(f"    [OK] Saved original (centered, normalized): "
              f"{orig_path}")

    # Render one STL per orientation, keeping track of failed renders.
    success, failure = 0, 0
    # Iterate through all rotation combos, naming oriented STL with the 
    # rotation values.
    for euler in ORIENTATIONS:
        x, y, z = euler
        out_name = f"{base}_x{x:03d}_y{y:03d}_z{z:03d}.stl"
        out_path = os.path.join(stl_folder, out_name)
        # Use try and except to prevent script from crashing if 
        # complications occur. 
        try:
            clear_scene()
            obj = import_stl(input_path)
            center_object(obj)
            normalize_scale(obj)
            apply_orientation(obj, euler)
            export_stl(obj, out_path)
            print(f"    [OK] {euler} -> {out_name}")
            success += 1
        except Exception as e:
            print(f"    [ERROR] orientation {euler}: {e}")
            failure += 1

    return success, failure

def batch_process(input_dir, output_dir):
    # Process every .stl in input directory.
    stls = os.listdir(input_dir)
    if not stls:
        print("No STL files found in input directory.")
        return

    print(f"Found {len(stls)} STL file(s) in {input_dir}")
    total_success, total_failure = 0, 0
    for file in stls:
        input_path = os.path.join(input_dir, file)
        # Use try and except to prevent script from crashing if 
        # complications occur.
        try:
            s, f = process_stl(input_path, output_dir)
        except Exception as e:
            print(f"\n=== [FATAL] Skipping {input_path}: {e}")
            s, f = 0, 1
        total_success += s
        total_failure += f

    print(f"\n=========================================")
    print(f"Done. {total_success} renders succeeded, "
          f"{total_failure} failed across {len(stls)} STL file(s).")


# Command line execution (Done on Windows):
#"C:\Program Files\Blender Foundation\Blender 5.1\blender.exe" 
# --background --python orient_stl.py -- <input_dir> <output_dir>
# Blender arguments are inputed in the command line (--background and
# -- python are processed by Blender when running the script, with 
# background telling Blender to run without opening its GUI window, and 
# python telling Blender what script to execute). Ensure that files are 
# not named with spaces to avoid command line parsing issues.
argv = sys.argv
try:
    # Blender arguments are separated with '--', thus the next argument
    # is the input directory.
    index = argv.index("--") + 1
    input_dir = argv[index]
    output_dir = argv[index + 1]
except (ValueError, IndexError):
    print("Usage: blender --background --python orient_stl.py "
            "-- <input_dir> <output_dir>")
    sys.exit(1)

batch_process(input_dir, output_dir)
