"""Batch 3D garment rendering: mesh → 2D render for conditioning.

Uses train_pairs.csv to render ONLY needed (person, garment) pairs
instead of all combinations.
"""

import argparse
import csv
from pathlib import Path
import numpy as np
import cv2
import torch
from tqdm import tqdm

from src.modules.mesh_renderer import MeshRenderer
from src.modules.garment_draper import GarmentDraper, load_garment_mesh
from src.modules.smplx_estimator import SMPLXEstimator
from src.modules.conditioning3d import azim_from_global_orient


def render_garments(garments_dir: str, smplx_params_dir: str, output_dir: str,
                    pairs_csv: str = "data/raw/train_pairs.csv",
                    normal_maps_dir: str = None, depth_maps_dir: str = None,
                    resolution: int = 512, device: str = "cuda"):
    """
    Render 3D garment meshes draped onto SMPL-X bodies.

    Only processes pairs defined in pairs_csv instead of all combinations.
    """
    garments = Path(garments_dir)
    params_dir = Path(smplx_params_dir)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if normal_maps_dir:
        Path(normal_maps_dir).mkdir(parents=True, exist_ok=True)
    if depth_maps_dir:
        Path(depth_maps_dir).mkdir(parents=True, exist_ok=True)

    # Load pairs from CSV
    pairs = []
    csv_path = Path(pairs_csv)
    if csv_path.exists():
        with open(csv_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                pairs.append((row["person_id"], row["garment_id"]))
        print(f"Loaded {len(pairs)} pairs from {pairs_csv}")
    else:
        print(f"⚠️ {pairs_csv} not found! Cannot proceed.")
        return

    # Build garment mesh lookup: garment_id → path
    mesh_extensions = {".obj", ".glb", ".ply", ".off"}
    garment_lookup = {}
    for ext in mesh_extensions:
        for mesh_path in garments.rglob(f"*{ext}"):
            # Try garment_id from parent folder name (e.g., 00047_Top/mesh.obj → 00047_Top)
            garment_id = mesh_path.parent.name
            garment_lookup[garment_id] = mesh_path
            # Also register by filename stem (e.g., mesh.obj won't help but Top.obj → Top)
            if mesh_path.stem != "mesh":
                garment_lookup[mesh_path.stem] = mesh_path

    print(f"Found {len(garment_lookup)} garment meshes")

    # Setup renderer and draper
    renderer = MeshRenderer(image_size=resolution, device=device)
    renderer.setup()

    draper = GarmentDraper().to(device)
    draper.eval()

    smplx_est = SMPLXEstimator(device=device)

    # Cache for loaded garment meshes
    garment_cache = {}
    skipped = 0
    rendered = 0

    for person_id, garment_id in tqdm(pairs, desc="Rendering pairs"):
        output_name = f"{person_id}_{garment_id}"

        # Skip if already rendered
        if (out_dir / f"{output_name}.png").exists():
            continue

        # Load SMPL-X params
        param_path = params_dir / f"{person_id}.npz"
        if not param_path.exists():
            skipped += 1
            continue

        # Find garment mesh
        if garment_id not in garment_lookup:
            skipped += 1
            continue

        # Load garment (with caching)
        if garment_id not in garment_cache:
            try:
                garment_data = load_garment_mesh(str(garment_lookup[garment_id]))
                garment_cache[garment_id] = garment_data
            except Exception as e:
                skipped += 1
                continue
        garment_data = garment_cache[garment_id]

        try:
            garment_verts = torch.tensor(
                garment_data["vertices"], dtype=torch.float32
            ).unsqueeze(0).to(device)
            garment_faces = torch.tensor(
                garment_data["faces"], dtype=torch.long
            ).to(device)

            # Load body params and create mesh
            params = SMPLXEstimator.load_params(str(param_path))
            body_mesh = smplx_est.get_body_mesh(params)
            body_verts = torch.tensor(
                body_mesh["vertices"], dtype=torch.float32
            ).unsqueeze(0).to(device)
            body_faces = torch.tensor(
                body_mesh["faces"], dtype=torch.long
            ).to(device)

            # Camera azimuth from the person's global orientation so the rendered
            # garment matches the person's facing direction (front=0, back=180).
            # MUST mirror src.modules.conditioning3d.build_conditioning_3d so train
            # and inference conditioning come from the same distribution.
            cam = {"dist": 2.7, "elev": 0,
                   "azim": azim_from_global_orient(params["global_orient"])}

            # Drape garment onto body
            with torch.no_grad():
                drape_result = draper(
                    garment_verts, garment_faces,
                    body_verts, body_faces,
                )

            draped_verts = drape_result["draped_verts"]

            # Textured render: bake the garment's real per-vertex colors instead of
            # a flat gray, so GarmentNet/IP-Adapter see the true appearance.
            vc = garment_data.get("vertex_colors")
            if vc is not None and vc.shape[0] == draped_verts.shape[1]:
                tex_colors = torch.tensor(vc, dtype=torch.float32, device=device).unsqueeze(0)
            else:
                tex_colors = torch.ones_like(draped_verts) * 0.7

            # RGB render
            render = renderer.render(draped_verts, garment_faces, tex_colors, cam)
            render_rgb = render[:, :3]
            render_np = (render_rgb[0].permute(1, 2, 0).cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
            cv2.imwrite(str(out_dir / f"{output_name}.png"), render_np[:, :, ::-1])

            # Normal map (same camera azimuth as RGB)
            if normal_maps_dir:
                normal_render = renderer.render_normal_map(draped_verts, garment_faces, cam)
                normal_np = (normal_render[0, :3].permute(1, 2, 0).cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
                cv2.imwrite(str(Path(normal_maps_dir) / f"{output_name}.png"), normal_np[:, :, ::-1])

            # Depth map (same camera azimuth as RGB)
            if depth_maps_dir:
                depth_render = renderer.render_depth_map(draped_verts, garment_faces, cam)
                depth_np = (depth_render[0, 0].cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
                cv2.imwrite(str(Path(depth_maps_dir) / f"{output_name}.png"), depth_np)

            rendered += 1

        except Exception as e:
            if skipped < 3:
                import traceback
                print(f"\n[render fail] {output_name}: {e}")
                traceback.print_exc()
            skipped += 1
            continue

    print(f"\n✅ Rendering complete: {rendered} rendered, {skipped} skipped")
    print(f"Output: {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="3D garment rendering")
    parser.add_argument("--garments_dir", default="data/garments_3d")
    parser.add_argument("--smplx_params_dir", default="data/processed/smplx_params")
    parser.add_argument("--output_dir", default="data/processed/renders_3d")
    parser.add_argument("--pairs_csv", default="data/raw/train_pairs.csv")
    parser.add_argument("--normal_maps_dir", default="data/processed/normal_maps")
    parser.add_argument("--depth_maps_dir", default="data/processed/depth_maps")
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    render_garments(args.garments_dir, args.smplx_params_dir, args.output_dir,
                    args.pairs_csv, args.normal_maps_dir, args.depth_maps_dir,
                    args.resolution, args.device)
