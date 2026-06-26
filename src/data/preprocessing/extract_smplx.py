"""Batch SMPL-X body parameter extraction from person images."""

import argparse
from pathlib import Path
import numpy as np
import cv2
from tqdm import tqdm

from src.modules.smplx_estimator import SMPLXEstimator, RealSMPLXEstimator


def extract_smplx(image_dir: str, output_dir: str, model_dir: str,
                   device: str = "cuda", save_mesh: bool = True,
                   mesh_dir: str = None, regressor: str = "hmr2"):
    """Extract SMPL-X parameters from every person image in a directory.

    `regressor='hmr2'/'pymaf'/'expose'` uses the real estimator
    (RealSMPLXEstimator) so global_orient reflects the person's true facing
    direction — required for the conditioning to match the person, and identical
    to the backend the inference notebook uses (hmr2). `regressor='simple'` keeps
    the legacy untrained regressor (NOT recommended; pose is meaningless).
    """
    img_dir = Path(image_dir)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if save_mesh and mesh_dir:
        Path(mesh_dir).mkdir(parents=True, exist_ok=True)

    if regressor == "simple":
        estimator = SMPLXEstimator(model_path=model_dir, regressor="simple", device=device)
    else:
        estimator = RealSMPLXEstimator(model_path=model_dir, regressor=regressor, device=device)
    estimator.load_model()

    extensions = {".jpg", ".jpeg", ".png", ".webp"}
    images = sorted([f for f in img_dir.iterdir() if f.suffix.lower() in extensions])

    print(f"Extracting SMPL-X parameters from {len(images)} images...")
    for img_path in tqdm(images):
        image = cv2.imread(str(img_path))
        if image is None:
            continue

        result = estimator.estimate(image)
        stem = img_path.stem

        # Save parameters
        np.savez(
            out_dir / f"{stem}.npz",
            betas=result["betas"],
            body_pose=result["body_pose"],
            global_orient=result["global_orient"],
            transl=result["transl"],
        )

        # Save mesh as OBJ (optional)
        if save_mesh and mesh_dir and "vertices" in result:
            _save_obj(
                Path(mesh_dir) / f"{stem}.obj",
                result["vertices"], result["faces"],
            )

    print(f"SMPL-X parameters extracted to: {out_dir}")


def _save_obj(path: Path, vertices: np.ndarray, faces: np.ndarray):
    """Write a simple OBJ file."""
    with open(path, "w") as f:
        for v in vertices:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        for face in faces:
            f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SMPL-X body estimation")
    parser.add_argument("--image_dir", default="data/raw/images")
    parser.add_argument("--output_dir", default="data/processed/smplx_params")
    parser.add_argument("--model_dir", default="checkpoints/pretrained")
    parser.add_argument("--mesh_dir", default="data/processed/smplx_meshes")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--save_mesh", action="store_true", default=True)
    parser.add_argument("--regressor", default="hmr2",
                        choices=["hmr2", "pymaf", "expose", "simple"],
                        help="real estimator (hmr2/pymaf/expose) vs legacy untrained (simple)")
    args = parser.parse_args()
    extract_smplx(args.image_dir, args.output_dir, args.model_dir,
                   args.device, args.save_mesh, args.mesh_dir, args.regressor)
