"""Inference entry point: python scripts/inference.py --person img.jpg --garment garment.obj"""

import argparse
from pathlib import Path
from omegaconf import OmegaConf
import torch

from src.models.tryon_pipeline import TryOnPipeline
from src.inference.image_tryon import ImageTryOn
from src.inference.video_tryon import VideoTryOn


def main():
    parser = argparse.ArgumentParser(description="DiffFit-3D Inference")
    parser.add_argument("--config", default="configs/inference.yaml")
    parser.add_argument("--person", required=True, help="Person image path")
    parser.add_argument("--garment", required=True,
                        help="Garment path — .obj (3D mesh) or .jpg/.png (2D image)")
    parser.add_argument("--output", default="results/output.png")
    parser.add_argument("--mode", default="image", choices=["image", "video"])
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--guidance", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--view_angle", type=float, default=0.0,
                        help="Camera azimuth for 3D rendering (0=front, 90=side, 180=back)")
    parser.add_argument("--local", action="store_true",
                        help="Use placeholder models (no HuggingFace download)")
    args = parser.parse_args()

    config = OmegaConf.to_container(OmegaConf.load(args.config), resolve=True)

    # Override from CLI
    if args.steps:
        config.setdefault("sampling", {})["num_inference_steps"] = args.steps
    if args.guidance:
        config.setdefault("sampling", {})["guidance_scale"] = args.guidance
    if args.seed is not None:
        config.setdefault("sampling", {})["seed"] = args.seed

    # Build pipeline
    model_cfg = config.get("model", {})

    if args.local or not model_cfg.get("pretrained_idm_vton"):
        pipeline = TryOnPipeline.from_config(model_cfg)
    else:
        pipeline = TryOnPipeline.from_pretrained(
            pretrained_model_id=model_cfg["pretrained_idm_vton"],
            controlnet_3d_channels=model_cfg.get("controlnet_3d_channels", 9),
        )

    # Load fine-tuned ControlNet3D weights if available
    controlnet_path = model_cfg.get("controlnet_3d_checkpoint")
    if controlnet_path and Path(controlnet_path).exists():
        state = torch.load(controlnet_path, map_location="cpu", weights_only=True)
        pipeline.controlnet_3d.load_state_dict(state, strict=False)
        print(f"Loaded ControlNet3D weights: {controlnet_path}")

    # Check if garment is 3D mesh
    garment_path = Path(args.garment)
    is_3d = garment_path.suffix.lower() in {".obj", ".glb", ".ply", ".stl"}

    if args.mode == "image":
        tryon = ImageTryOn(pipeline, config)
        if is_3d:
            result = tryon.run_with_3d_garment(
                args.person, args.garment, args.output,
                view_angle=args.view_angle,
            )
        else:
            result = tryon.run(args.person, args.garment, args.output)
        print(f"Result saved to: {args.output}")
    else:
        tryon = ImageTryOn(pipeline, config)
        video_tryon = VideoTryOn(tryon, config.get("video", {}))
        video_tryon.run(args.person, args.garment, args.output)
        print(f"Video saved to: {args.output}")


if __name__ == "__main__":
    main()
