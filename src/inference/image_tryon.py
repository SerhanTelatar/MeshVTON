"""
Single-Frame Virtual Try-On Inference.

Complete pipeline supporting both:
  - 2D garment images (traditional try-on)
  - 3D garment meshes (.obj) with SMPL-X body estimation, garment draping,
    and PyTorch3D rendering for geometry-aware try-on
"""

from typing import Optional
from pathlib import Path
import torch
import numpy as np
from PIL import Image

from src.models.tryon_pipeline import TryOnPipeline
from src.models.controlnet_3d import ControlNet3D
from src.modules.pose_estimator import PoseEstimator
from src.modules.segmentation import HumanSegmentation
from src.modules.agnostic_mask import AgnosticMaskGenerator
from src.inference.postprocess import PostProcessor
from src.utils.image_utils import load_image, save_image, pil_to_tensor, tensor_to_pil


class ImageTryOn:
    """
    End-to-end single-frame virtual try-on inference.

    Supports two modes:
      1. Traditional 2D: person photo + garment photo
      2. 3D-aware: person photo + garment mesh (.obj) → SMPL-X → drape → render → ControlNet3D

    Args:
        pipeline: TryOnPipeline model (IDM-VTON + ControlNet3D).
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
        self.postprocessor = PostProcessor(config=self.config.get("postprocess", {}))

        # Sampling settings
        sampling = self.config.get("sampling", {})
        self.num_steps = sampling.get("num_inference_steps", 50)
        self.guidance_scale = sampling.get("guidance_scale", 7.5)
        self.seed = sampling.get("seed", None)

        # 3D pipeline modules (lazy loaded)
        self._smplx_estimator = None
        self._garment_draper = None
        self._mesh_renderer = None

    def _init_3d_pipeline(self):
        """Lazy-initialize 3D pipeline modules."""
        if self._smplx_estimator is not None:
            return

        from src.modules.smplx_estimator import RealSMPLXEstimator
        from src.modules.garment_draper import GarmentDraper
        from src.modules.mesh_renderer import MeshRenderer

        # Real SMPL-X estimator so global_orient reflects the person's true facing
        # direction (front/back/side). Falls back loudly (raises) if unavailable —
        # we never silently use the untrained regressor (that produced wrong poses).
        self._smplx_estimator = RealSMPLXEstimator(device=str(self.device))
        self._garment_draper = GarmentDraper().to(self.device)
        self._mesh_renderer = MeshRenderer(
            image_size=self.resolution, device=str(self.device)
        )
        self._mesh_renderer.setup()
        print("3D pipeline initialized: RealSMPL-X + GarmentDraper + MeshRenderer")

    @torch.no_grad()
    def run(self, person_image, garment_image,
            output_path: Optional[str] = None) -> Image.Image:
        """
        Run virtual try-on with a 2D garment image (traditional mode).

        Args:
            person_image: Person image (path, numpy, or PIL).
            garment_image: Garment image (path, numpy, or PIL).
            output_path: Optional output save path.

        Returns:
            PIL Image of the try-on result.
        """
        person_np = self._load_as_numpy(person_image)
        garment_np = self._load_as_numpy(garment_image)

        # Extract conditioning
        pose_result = self.pose_estimator.estimate(person_np)
        seg_result = self.segmentation.segment(person_np)
        agnostic_result = self.agnostic_gen.generate(
            person_np, seg_result["segmentation"], pose_result.get("keypoints"),
        )

        person_tensor = pil_to_tensor(
            Image.fromarray(person_np[:, :, ::-1]), self.resolution
        ).unsqueeze(0).to(self.device)

        garment_tensor = pil_to_tensor(
            Image.fromarray(garment_np[:, :, ::-1]), self.resolution
        ).unsqueeze(0).to(self.device)

        agnostic_tensor = pil_to_tensor(
            Image.fromarray(agnostic_result["agnostic_image"][:, :, ::-1]), self.resolution
        ).unsqueeze(0).to(self.device)

        generator = None
        if self.seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(self.seed)

        result_tensor = self.pipeline.generate(
            garment_image=garment_tensor,
            agnostic_mask=agnostic_tensor,
            conditioning_3d=None,
            num_inference_steps=self.num_steps,
            guidance_scale=self.guidance_scale,
            generator=generator,
        )

        result_pil = tensor_to_pil(result_tensor[0])
        result_pil = self.postprocessor.process(result_pil, person_np)

        if output_path:
            save_image(result_pil, output_path)

        return result_pil

    @torch.no_grad()
    def run_with_3d_garment(
        self,
        person_image,
        garment_mesh_path: str,
        output_path: Optional[str] = None,
        view_angle: Optional[float] = None,
    ) -> Image.Image:
        """
        Run virtual try-on with a 3D garment mesh (NOVEL 3D-aware mode).

        The 3D conditioning is built through the shared
        `src.modules.conditioning3d.build_conditioning_3d` so it is *identical*
        to what training produced (real SMPL-X pose, camera azimuth from
        global_orient, textured render, 9-channel rgb+normal+depth). This is the
        single source of truth — do not render conditioning inline here.

        NOTE: the validated runtime backbone is the real IDM-VTON pipeline used by
        notebooks/meshvton_inference.ipynb (via src.idm_vton). This method still
        calls the legacy `self.pipeline.generate`; the conditioning is now correct
        but the legacy backbone remains abandoned — prefer the notebook path.

        Args:
            person_image: Person image (path, numpy, or PIL).
            garment_mesh_path: Path to 3D garment mesh (.obj/.glb/.ply).
            output_path: Optional output save path.
            view_angle: Camera azimuth override (deg). If None, derived from pose.

        Returns:
            PIL Image of the 3D-aware try-on result.
        """
        self._init_3d_pipeline()

        from src.modules.conditioning3d import build_conditioning_3d

        person_np = self._load_as_numpy(person_image)

        # Build conditioning + textured garment render through the shared builder.
        cond = build_conditioning_3d(
            person_image=Image.fromarray(person_np[:, :, ::-1]),
            mesh_path=garment_mesh_path,
            estimator=self._smplx_estimator,
            renderer=self._mesh_renderer,
            draper=self._garment_draper,
            height=self.resolution,
            width=self.resolution,
            view_angle=view_angle,
            device=str(self.device),
            dtype=torch.float32,
        )
        conditioning_3d = cond["conditioning_3d"]

        # Textured garment render as the 2D garment input for GarmentNet.
        garment_front = pil_to_tensor(cond["render_rgb"], self.resolution).unsqueeze(0).to(self.device)

        # Standard 2D preprocessing
        pose_result = self.pose_estimator.estimate(person_np)
        seg_result = self.segmentation.segment(person_np)
        agnostic_result = self.agnostic_gen.generate(
            person_np, seg_result["segmentation"], pose_result.get("keypoints"),
        )

        agnostic_tensor = pil_to_tensor(
            Image.fromarray(agnostic_result["agnostic_image"][:, :, ::-1]),
            self.resolution,
        ).unsqueeze(0).to(self.device)

        generator = None
        if self.seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(self.seed)

        result_tensor = self.pipeline.generate(
            garment_image=garment_front,
            agnostic_mask=agnostic_tensor,
            conditioning_3d=conditioning_3d,
            num_inference_steps=self.num_steps,
            guidance_scale=self.guidance_scale,
            generator=generator,
        )

        result_pil = tensor_to_pil(result_tensor[0])
        result_pil = self.postprocessor.process(result_pil, person_np)

        if output_path:
            save_image(result_pil, output_path)

        return result_pil

    def _load_as_numpy(self, image) -> np.ndarray:
        """Convert various image formats to BGR numpy array."""
        if isinstance(image, (str, Path)):
            import cv2
            return cv2.imread(str(image))
        elif isinstance(image, Image.Image):
            arr = np.array(image)
            if arr.shape[-1] == 3:
                return arr[:, :, ::-1]  # RGB to BGR
            return arr
        elif isinstance(image, np.ndarray):
            return image
        raise ValueError(f"Unsupported image type: {type(image)}")
