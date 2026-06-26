"""Batch preprocessing: python scripts/preprocess_dataset.py --config configs/data/preprocessing.yaml"""

import argparse
from omegaconf import OmegaConf

from src.data.preprocessing.extract_pose import extract_poses
from src.data.preprocessing.extract_segment import extract_segments
from src.data.preprocessing.build_agnostic import build_agnostic
from src.data.preprocessing.extract_smplx import extract_smplx
from src.data.preprocessing.render_garment import render_garments


def main():
    parser = argparse.ArgumentParser(description="Preprocess dataset")
    parser.add_argument("--config", default="configs/data/preprocessing.yaml")
    parser.add_argument("--steps", nargs="+",
                        default=["pose", "segment", "agnostic", "smplx", "render3d"])
    parser.add_argument("--pairs_csv", default="data/raw/train_pairs.csv",
                        help="(person_id, garment_id) pairs for 3D garment rendering")
    args = parser.parse_args()

    config = OmegaConf.to_container(OmegaConf.load(args.config), resolve=True)
    image_dir = config.get("data_root", "data/raw/images")
    device = config.get("device", "cuda")

    if "pose" in args.steps and config.get("pose", {}).get("enabled", True):
        print("=== Extracting poses ===")
        extract_poses(image_dir, config["pose"]["output_dir"],
                      config["pose"].get("model", "dwpose"), device)

    if "segment" in args.steps and config.get("segmentation", {}).get("enabled", True):
        print("=== Extracting segmentation ===")
        extract_segments(image_dir, config["segmentation"]["output_dir"],
                         config["segmentation"].get("model", "atr"), device)

    if "agnostic" in args.steps and config.get("agnostic", {}).get("enabled", True):
        print("=== Building agnostic representations ===")
        build_agnostic(
            image_dir,
            config.get("pose", {}).get("output_dir", "data/processed/poses"),
            config.get("segmentation", {}).get("output_dir", "data/processed/segments"),
            config["agnostic"]["output_dir"],
            config["agnostic"].get("mask_type", "upper"),
        )

    if "smplx" in args.steps and config.get("smplx", {}).get("enabled", True):
        print("=== Extracting SMPL-X (real pose) ===")
        sx = config["smplx"]
        # regressor must be a real backend (hmr2/pymaf/expose) so global_orient is
        # meaningful — this is what makes the garment follow the person's facing.
        extract_smplx(
            image_dir, sx["output_dir"], sx.get("model_dir", "checkpoints/pretrained"),
            device=device, save_mesh=sx.get("save_mesh", True),
            mesh_dir=sx.get("mesh_dir"), regressor=sx.get("regressor", "hmr2"),
        )

    if "render3d" in args.steps and config.get("garment_render", {}).get("enabled", True):
        print("=== Rendering 3D garments (textured, pose-aligned) ===")
        gr = config["garment_render"]
        render_garments(
            gr["garments_dir"], config["smplx"]["output_dir"], gr["output_dir"],
            pairs_csv=args.pairs_csv,
            normal_maps_dir=gr.get("normal_maps_dir"),
            depth_maps_dir=gr.get("depth_maps_dir"),
            resolution=gr.get("render_resolution", 512), device=device,
        )

    print("Preprocessing complete!")


if __name__ == "__main__":
    main()
