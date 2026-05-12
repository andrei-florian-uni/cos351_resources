import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import trimesh
import numpy as np
import os
import sys

# Output image size in pixels. Value set to 750 to keep 
# features visible. Feed output images into LLM manually because:
# 1) API token limitations.
# 2) Small scale implementation prior to building a full pipeline 
# with automated LLM API calls.
IMAGE_SIZE = 750
# Fixed camera orientation. 
# 30 degree tilt from above.
CAMERA_ELEVATION = 30
# 45 degree rotation around vertical axis.
CAMERA_AZIMUTH = 45
# Lighting that helps the LLM read 3D structure and features.
# Set light vector to a diagonal direction.
LIGHT_DIRECTION = np.array([0.5, 0.5, 1.0])
# Base brightness on faces facing away from the light.
AMBIENT = 0.4
# Extra brightness on faces facing toward the light.
DIFFUSE = 0.6
# Surface base color (Light gray).
BASE_COLOR = np.array([0.75, 0.75, 0.78])

# Helper methods.

def compute_face_colors(mesh):
    # Compute RGBA colors per face based on diffuse shading.

    # Normalize light vector for dot product.
    light = LIGHT_DIRECTION / np.linalg.norm(LIGHT_DIRECTION)
    # Dot product of each face normal with normal light direction. 
    # Limit values (clip) to [0, 1] so faces pointing away from 
    # the light don't contribute negative shading.(a·b=|a||b|cos(theta))
    shading = np.clip(mesh.face_normals @ light, 0.0, 1.0)
    # Combine ambient and diffuse lighting for face brightness.
    intensity = AMBIENT + DIFFUSE * shading
    # Multiply reshaped intensity (n,1) with base color (3,), 
    # one color per face.
    face_colors = intensity.reshape(-1, 1) * BASE_COLOR
    # Create 2D (n,1) alpha (opacity) array of all ones to appended 
    # to the face colors array for RGBA format. 
    # All ones signify opaque faces.
    alpha = np.ones((len(face_colors), 1))
    # Append alpha column array to face colors array.
    return np.concatenate([face_colors, alpha], axis=1)

def render_stl(stl_path, png_path):
    # Load the STL file as a triangle mesh.
    mesh = trimesh.load(stl_path)
    # Obtain 3D coordinates for all triangles for rendering. 
    triangles = mesh.triangles
    face_colors = compute_face_colors(mesh)

    # Pixel count = dpi * inches.
    # figure size in inches = pixel count / dpi.
    dpi = 100
    figsize = IMAGE_SIZE / dpi
    # Plot the STL triangles with face colors.
    fig = plt.figure(figsize=(figsize, figsize), dpi=dpi)
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    # Initialize the collection of 3D triangles with their
    # corresponding face colors to add to the 3D plot.
    poly3d = Poly3DCollection(triangles, edgecolor='none')
    poly3d.set_facecolor(face_colors)
    ax.add_collection3d(poly3d)

    # Set the axes limits to center the mesh so that it fits 
    # within the view.
    # Extract the bounding box of the mesh and find the center.
    bounds = mesh.bounds
    center = bounds.mean(axis=0)
    # Calculate the longest dimension of the mesh, half it and add 
    # padding to prevent cropping.
    size = (bounds[1] - bounds[0]).max() * 0.55
    # Set the axes limits.
    ax.set_xlim(center[0]-size, center[0]+size)
    ax.set_ylim(center[1]-size, center[1]+size)
    ax.set_zlim(center[2]-size, center[2]+size)
    ax.set_box_aspect([1, 1, 1])
    # Set the camera view angles.
    ax.view_init(elev=CAMERA_ELEVATION, azim=CAMERA_AZIMUTH)
    ax.set_axis_off()
    fig.patch.set_facecolor('white')

    plt.savefig(png_path, bbox_inches='tight', dpi=dpi, facecolor='white')
    plt.close(fig)

def render_folder(stl_folder, png_folder):
    # Make png output folder to store rendered images.
    os.makedirs(png_folder, exist_ok=True)
    stls = os.listdir(stl_folder)
    # Iterate through all files in the STL folder, keeping track of 
    # failures and successes.
    success, failure = 0, 0
    for file in stls:
        # Use STL filename but change extension to .png for output.
        png_name = os.path.splitext(file)[0] + '.png'
        stl_path = os.path.join(stl_folder, file)
        png_path = os.path.join(png_folder, png_name)
        try:
            render_stl(stl_path, png_path)
            print(f"    [OK] {file} -> {png_name}")
            success += 1
        except Exception as e:
            print(f"    [ERROR] {file}: {e}")
            failure += 1
    return success, failure

def batch_process(input_dir, output_dir):
    # Extract leaf folders in input directory.
    subfolders = os.listdir(input_dir)

    print(f"Found {len(subfolders)} STL folder(s) in {input_dir}")
    total_success, total_failure = 0, 0
    for sub in subfolders:
        # Initialize input and output paths for each STL folder.
        in_sub = os.path.join(input_dir, sub)
        out_sub = os.path.join(output_dir, sub)
        print(f"\n=== Rendering: {sub}")
        s, f = render_folder(in_sub, out_sub)
        total_success += s
        total_failure += f

    print(f"\n=========================================")
    print(f"Done. {total_success} renders succeeded, "
          f"{total_failure} failed across {len(subfolders)} part folder(s).")


# Command line execution:
# python render_stl.py <input_dir> <output_dir>
# Check that the correct number of command-line arguments are inputed.
if len(sys.argv) != 3:
    print("Usage: python render_stls.py <input_dir> <output_dir>")
    sys.exit(1)
# Extract input and output directories from command-line arguments and
# execute batch processing.
input_dir = sys.argv[1]
output_dir = sys.argv[2]
batch_process(input_dir, output_dir)
