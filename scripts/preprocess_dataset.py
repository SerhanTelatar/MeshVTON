"""Batch preprocessing: python scripts/preprocess_dataset.py --config configs/data/preprocessing.yaml"""

import argparse
from omegaconf import OmegaConf

from src.data.preprocessing.extract_pose import extract_poses
from src.data.preprocessing.extract_segment import extract_segments
from src.data.preprocessing.build_agnostic import build_agnostic


def main():
    parser = argparse.ArgumentParser(description="Preprocess dataset")
    parser.add_argument("--config", default="configs/data/preprocessing.yaml")
    parser.add_argument("--steps", nargs="+", default=["pose", "segment", "agnostic"])
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

    print("Preprocessing complete!")


if __name__ == "__main__":
    main()
