import coacd
import trimesh
import numpy as np

# Load and pre-scale mesh (mug.stl is in mm -> convert to meters)
mesh = trimesh.load('mug.stl')
mesh.apply_scale(0.001)

# Convert to CoACD format
coacd_mesh = coacd.Mesh(mesh.vertices, mesh.faces)

# Run decomposition with fine threshold
parts = coacd.run_coacd(coacd_mesh, threshold=0.02)

# Save each convex hull as a separate STL (already in meters)
for i, (verts, faces) in enumerate(parts):
    hull = trimesh.Trimesh(vertices=verts, faces=faces)
    hull.export(f'mug_hull_{i}.stl')

print(f'Decomposed into {len(parts)} convex parts')
# Print bounding box info for each part
for i, (verts, faces) in enumerate(parts):
    hull = trimesh.Trimesh(vertices=verts, faces=faces)
    print(f'  Part {i}: {len(verts)} verts, {len(faces)} faces, '
          f'bounds min={hull.bounds[0].round(5)}, max={hull.bounds[1].round(5)}')