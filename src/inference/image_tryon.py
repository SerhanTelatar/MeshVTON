"""
Single-Frame Virtual Try-On Inference.

Complete pipeline for generating a try-on result from person + garment images.
"""

from typing import Optional
from pathlib import Path
import torch
import numpy as np
from PIL import Image

from src.models.tryon_pipeline import TryOnPipeline
from src.modules.pose_estimator import PoseEstimator
from src.modules.segmentation import HumanSegmentation
from src.modules.agnostic_mask import AgnosticMaskGenerator
from src.modules.densepose import DensePoseExtractor
from src.inference.postprocess import PostProcessor
from src.utils.image_utils import load_image, save_image, pil_to_tensor, tensor_to_pil


class ImageTryOn:
    """
    End-to-end single-frame virtual try-on inference.

    Args:
        pipeline: Trained TryOnPipeline model.
        config: Inference configuration dict.
    """

    def __init__(self, pipeline: TryOnPipeline, config: Optional[dict] = None):
        self.config = config or {}
        self.pipeline = pipeline
        self.device = torch.device(self.config.get("device", "cuda"))
        self.pipeline.to(self.device).eval()

        # Resolution
        self.resolution = self.config.get("image", {}).get("resolution", 512)

        # Preprocessing modules
        self.pose_estimator = PoseEstimator(device=str(self.device))
        self.segmentation = HumanSegmentation(device=str(self.device))
        self.agnostic_gen = AgnosticMaskGenerator()
        self.densepose = DensePoseExtractor(device=str(self.device))
        self.postprocessor = PostProcessor(config=self.config.get("postprocess", {}))

        # Sampling settings
        sampling = self.config.get("sampling", {})
        self.num_steps = sampling.get("num_inference_steps", 50)
        self.guidance_scale = sampling.get("guidance_scale", 7.5)
        self.seed = sampling.get("seed", None)

    @torch.no_grad()
    def run(self, person_image: str | np.ndarray | Image.Image,
            garment_image: str | np.ndarray | Image.Image,
            output_path: Optional[str] = None) -> Image.Image:
        """
        Run virtual try-on on a single image.

        Args:
            person_image: Person image (path, numpy, or PIL).
            garment_image: Garment image (path, numpy, or PIL).
            output_path: Optional output save path.

        Returns:
            PIL Image of the try-on result.
        """
        # Load and preprocess
        person_np = self._load_as_numpy(person_image)
        garment_np = self._load_as_numpy(garment_image)

        # Extract conditioning
        pose_result = self.pose_estimator.estimate(person_np)
        seg_result = self.segmentation.segment(person_np)
        agnostic_result = self.agnostic_gen.generate(
            person_np, seg_result["segmentation"], pose_result.get("keypoints"),
        )
        dp_result = self.densepose.extract(person_np)

        # Convert to tensors
        person_tensor = pil_to_tensor(
            Image.fromarray(person_np[:, :, ::-1]), self.resolution
        ).unsqueeze(0).to(self.device)

        garment_tensor = pil_to_tensor(
            Image.fromarray(garment_np[:, :, ::-1]), self.resolution
        ).unsqueeze(0).to(self.device)

        agnostic_tensor = pil_to_tensor(
            Image.fromarray(agnostic_result["agnostic_image"][:, :, ::-1]), self.resolution
        ).unsqueeze(0).to(self.device)

        pose_tensor = torch.from_numpy(
            pose_result["pose_image"]
        ).float().permute(2, 0, 1).unsqueeze(0).to(self.device) / 255.0

        # Resize pose tensor to match
        pose_tensor = torch.nn.functional.interpolate(
            pose_tensor, size=(self.resolution, self.resolution), mode="bilinear", align_corners=False
        )

        densepose_tensor = self.densepose.iuv_to_tensor(
            dp_result["iuv"]
        ).unsqueeze(0).to(self.device)
        densepose_tensor = torch.nn.functional.interpolate(
            densepose_tensor, size=(self.resolution, self.resolution), mode="bilinear", align_corners=False
        )
        # Use only first channel for conditioning
        densepose_cond = densepose_tensor[:, :3]

        # Generator for reproducibility
        generator = None
        if self.seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(self.seed)

        # Generate
        result_tensor = self.pipeline.generate(
            garment_image=garment_tensor,
            agnostic_mask=agnostic_tensor,
            pose_map=pose_tensor,
            densepose_map=densepose_cond,
            num_inference_steps=self.num_steps,
            guidance_scale=self.guidance_scale,
            generator=generator,
        )

        # Post-process
        result_pil = tensor_to_pil(result_tensor[0])
        result_pil = self.postprocessor.process(result_pil, person_np)

        # Save
        if output_path:
            save_image(result_pil, output_path)

        return result_pil

    def _load_as_numpy(self, image) -> np.ndarray:
        """Convert various image formats to BGR numpy array."""
        if isinstance(image, str):
            import cv2
            return cv2.imread(image)
        elif isinstance(image, Image.Image):
            arr = np.array(image)
            if arr.shape[-1] == 3:
                return arr[:, :, ::-1]  # RGB to BGR
            return arr
        elif isinstance(image, np.ndarray):
            return image
        raise ValueError(f"Unsupported image type: {type(image)}")
