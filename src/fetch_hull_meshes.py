"""
Build a solid translucent "brain shell" directly from our own real neuron
positions, using a density-based isosurface (marching cubes over a smoothed
3D occupancy grid) rather than a convex hull.

This matters scientifically: a convex hull can only represent a *convex*
shape -- it can't capture the concavities of a real fly brain (the cleft
between optic lobes and central brain) or the curve of the nerve cord, and
produces flat, faceted, sometimes degenerate wedges when the point sample is
sparse. A density isosurface follows the actual point cloud locally, so it
naturally curves and can represent concavity -- much closer to a real
anatomical silhouette, still built entirely from our own real, aligned data
(no external template mesh, no alignment risk).

Splits into brain vs. nerve cord using real region labels. Neurons with an
unresolved region ("NO_CONS") are excluded entirely -- they can be located
anywhere in the body and must not be defaulted into either bucket.

Output: data/connectome/hull_meshes.json
"""
import gzip
import json
from pathlib import Path

import numpy as np
import pandas as pd
import trimesh
from scipy.ndimage import gaussian_filter
from skimage import measure

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "connectome"

VNC_MARKERS = ("T1_", "T2_", "T3_", "ABDNM", "TCT", "AMNP", "FLA")


def is_vnc(region: str) -> bool:
    if not isinstance(region, str):
        return False
    return any(region.startswith(m) or m in region for m in VNC_MARKERS)


def convex_hull_surface(points: np.ndarray, subdivisions: int = 3) -> dict:
    """The earlier faceted/smoothed convex-hull look, kept as an explicit
    stylistic option -- not scientifically representative (can't show
    concavity, see module docstring), but visually preferred by some."""
    from scipy.spatial import ConvexHull

    hull = ConvexHull(points)
    used = np.unique(hull.simplices)
    remap = {old: new for new, old in enumerate(used)}
    verts = points[used]
    faces = np.array([[remap[i] for i in tri] for tri in hull.simplices])

    mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=True)
    for _ in range(subdivisions):
        mesh = mesh.subdivide()
    trimesh.smoothing.filter_laplacian(mesh, iterations=8, lamb=0.4, volume_constraint=False)

    return {"vertices": np.round(mesh.vertices, 1).tolist(), "faces": mesh.faces.tolist()}


def density_surface(points: np.ndarray, voxel_size: float, sigma: float = 1.6, enclose_fraction: float = 0.85) -> dict:
    pad = voxel_size * 3
    mins = points.min(axis=0) - pad
    maxs = points.max(axis=0) + pad
    dims = np.ceil((maxs - mins) / voxel_size).astype(int)

    idx = np.clip(((points - mins) / voxel_size).astype(int), 0, dims - 1)
    grid = np.zeros(dims, dtype=np.float32)
    np.add.at(grid, (idx[:, 0], idx[:, 1], idx[:, 2]), 1.0)
    grid = gaussian_filter(grid, sigma=sigma)

    # pick the density level so ~enclose_fraction of the real points end up inside the surface
    density_at_points = grid[idx[:, 0], idx[:, 1], idx[:, 2]]
    level = np.percentile(density_at_points, (1 - enclose_fraction) * 100)
    level = max(level, grid.max() * 0.02)

    verts, faces, _, _ = measure.marching_cubes(grid, level=level)
    verts_world = verts * voxel_size + mins

    # keep only the largest connected piece -- density noise can spawn tiny stray specks
    mesh = trimesh.Trimesh(vertices=verts_world, faces=faces, process=False)
    pieces = mesh.split(only_watertight=False)
    mesh = max(pieces, key=lambda m: len(m.vertices))

    return {"vertices": np.round(mesh.vertices, 1).tolist(), "faces": mesh.faces.tolist()}


def main():
    context = pd.read_csv(DATA_DIR / "context_positions.csv")
    with gzip.open(DATA_DIR / "neurons.csv.gz", "rt") as f:
        neurons = pd.read_csv(f, usecols=["Root ID", "Top in/out region"], dtype={"Root ID": "int64"})
    neurons = neurons.rename(columns={"Root ID": "root_id", "Top in/out region": "region"})

    merged = context.merge(neurons, on="root_id", how="left")
    merged["region_primary"] = merged["region"].fillna("").apply(lambda r: r.split(".")[0])
    UNRESOLVED = {"", "NO_CONS"}
    merged = merged[~merged["region_primary"].isin(UNRESOLVED)]
    merged["is_vnc"] = merged["region_primary"].apply(is_vnc)

    brain_pts = merged.loc[~merged["is_vnc"], ["x_nm", "y_nm", "z_nm"]].to_numpy(dtype=np.float64)
    vnc_pts = merged.loc[merged["is_vnc"], ["x_nm", "y_nm", "z_nm"]].to_numpy(dtype=np.float64)
    print(f"brain points: {len(brain_pts)}, vnc points: {len(vnc_pts)}")

    brain_smooth = density_surface(brain_pts, voxel_size=13000, sigma=1.6)
    vnc_smooth = density_surface(vnc_pts, voxel_size=13000, sigma=1.6)
    brain_convex = convex_hull_surface(brain_pts)
    vnc_convex = convex_hull_surface(vnc_pts)

    result = {
        "smooth": {"brain": brain_smooth, "vnc": vnc_smooth},
        "convex": {"brain": brain_convex, "vnc": vnc_convex},
    }
    out_path = DATA_DIR / "hull_meshes.json"
    with open(out_path, "w") as f:
        json.dump(result, f)
    print(f"Saved: {out_path}")
    print(f"smooth brain: {len(brain_smooth['vertices'])} verts, {len(brain_smooth['faces'])} faces")
    print(f"smooth vnc: {len(vnc_smooth['vertices'])} verts, {len(vnc_smooth['faces'])} faces")
    print(f"convex brain: {len(brain_convex['vertices'])} verts, {len(brain_convex['faces'])} faces")
    print(f"convex vnc: {len(vnc_convex['vertices'])} verts, {len(vnc_convex['faces'])} faces")


if __name__ == "__main__":
    main()
