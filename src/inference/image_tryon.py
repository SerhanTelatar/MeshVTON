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
from src.modules.densepose import DensePoseExtractor
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
        self.densepose = DensePoseExtractor(device=str(self.device))
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

        from src.modules.smplx_estimator import SMPLXEstimator
        from src.modules.garment_draper import GarmentDraper
        from src.modules.mesh_renderer import MeshRenderer

        self._smplx_estimator = SMPLXEstimator(device=str(self.device))
        self._garment_draper = GarmentDraper().to(self.device)
        self._mesh_renderer = MeshRenderer(
            image_size=self.resolution, device=str(self.device)
        )
        print("3D pipeline initialized: SMPL-X + GarmentDraper + MeshRenderer")

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
        pose_tensor = torch.nn.functional.interpolate(
            pose_tensor, size=(self.resolution, self.resolution),
            mode="bilinear", align_corners=False,
        )

        densepose_tensor = self.densepose.iuv_to_tensor(
            dp_result["iuv"]
        ).unsqueeze(0).to(self.device)
        densepose_tensor = torch.nn.functional.interpolate(
            densepose_tensor, size=(self.resolution, self.resolution),
            mode="bilinear", align_corners=False,
        )
        densepose_cond = densepose_tensor[:, :3]

        # Generator
        generator = None
        if self.seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(self.seed)

        # Generate (no 3D conditioning in traditional mode)
        result_tensor = self.pipeline.generate(
            garment_image=garment_tensor,
            agnostic_mask=agnostic_tensor,
            pose_map=pose_tensor,
            densepose_map=densepose_cond,
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
        view_angle: float = 0.0,
    ) -> Image.Image:
        """
        Run virtual try-on with a 3D garment mesh (NOVEL 3D-aware mode).

        Pipeline:
          1. Estimate SMPL-X body from person image
          2. Load 3D garment mesh (.obj)
          3. Drape garment onto SMPL-X body
          4. Render draped garment via PyTorch3D → RGB + normal + depth
          5. Generate try-on via ControlNet3D conditioning

        Args:
            person_image: Person image (path, numpy, or PIL).
            garment_mesh_path: Path to 3D garment mesh (.obj/.glb/.ply).
            output_path: Optional output save path.
            view_angle: Camera azimuth angle (0=front, 90=side, 180=back).

        Returns:
            PIL Image of the 3D-aware try-on result.
        """
        self._init_3d_pipeline()

        person_np = self._load_as_numpy(person_image)

        # 1. Estimate SMPL-X body
        smplx_params = self._smplx_estimator.estimate(person_np)

        # 2. Load 3D garment mesh
        from src.modules.garment_draper import load_garment_mesh
        garment_mesh = load_garment_mesh(garment_mesh_path)

        # 3. Drape garment onto body
        garment_verts = torch.from_numpy(garment_mesh["vertices"]).float().unsqueeze(0).to(self.device)
        garment_faces = torch.from_numpy(garment_mesh["faces"]).long().to(self.device)
        body_verts = torch.from_numpy(smplx_params["vertices"]).float().unsqueeze(0).to(self.device)
        body_faces = torch.from_numpy(smplx_params["faces"]).long().to(self.device)

        draped = self._garment_draper(
            garment_verts, garment_faces, body_verts, body_faces,
        )

        # 4. Render from specified view angle
        camera_params = {"dist": 2.7, "elev": 0, "azim": view_angle}

        render_rgb = self._mesh_renderer.render(
            draped["draped_verts"], garment_faces,
            camera_params=camera_params,
        )[:, :3]  # Take RGB only

        normal_map = self._mesh_renderer.render_normal_map(
            draped["draped_verts"], garment_faces,
        )[:, :3]

        depth_map = self._mesh_renderer.render_depth_map(
            draped["draped_verts"], garment_faces,
        )[:, :3]

        # Resize to model resolution
        render_rgb = torch.nn.functional.interpolate(
            render_rgb, size=(self.resolution, self.resolution),
            mode="bilinear", align_corners=False,
        )
        normal_map = torch.nn.functional.interpolate(
            normal_map, size=(self.resolution, self.resolution),
            mode="bilinear", align_corners=False,
        )
        depth_map = torch.nn.functional.interpolate(
            depth_map, size=(self.resolution, self.resolution),
            mode="bilinear", align_corners=False,
        )

        # 5. Build 3D conditioning tensor
        conditioning_3d = ControlNet3D.prepare_conditioning(
            render_rgb, normal_map, depth_map,
        )

        # Standard 2D preprocessing
        pose_result = self.pose_estimator.estimate(person_np)
        seg_result = self.segmentation.segment(person_np)
        agnostic_result = self.agnostic_gen.generate(
            person_np, seg_result["segmentation"], pose_result.get("keypoints"),
        )
        dp_result = self.densepose.extract(person_np)

        # Use rendered garment as 2D garment input for GarmentNet
        # (render from front view for the garment encoder)
        garment_front = self._mesh_renderer.render(
            draped["draped_verts"], garment_faces,
            camera_params={"dist": 2.7, "elev": 0, "azim": 0},
        )[:, :3]
        garment_front = torch.nn.functional.interpolate(
            garment_front, size=(self.resolution, self.resolution),
            mode="bilinear", align_corners=False,
        )

        agnostic_tensor = pil_to_tensor(
            Image.fromarray(agnostic_result["agnostic_image"][:, :, ::-1]),
            self.resolution,
        ).unsqueeze(0).to(self.device)

        pose_tensor = torch.from_numpy(
            pose_result["pose_image"]
        ).float().permute(2, 0, 1).unsqueeze(0).to(self.device) / 255.0
        pose_tensor = torch.nn.functional.interpolate(
            pose_tensor, size=(self.resolution, self.resolution),
            mode="bilinear", align_corners=False,
        )

        densepose_tensor = self.densepose.iuv_to_tensor(
            dp_result["iuv"]
        ).unsqueeze(0).to(self.device)
        densepose_tensor = torch.nn.functional.interpolate(
            densepose_tensor, size=(self.resolution, self.resolution),
            mode="bilinear", align_corners=False,
        )

        # Generator
        generator = None
        if self.seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(self.seed)

        # Generate with 3D conditioning
        result_tensor = self.pipeline.generate(
            garment_image=garment_front,
            agnostic_mask=agnostic_tensor,
            pose_map=pose_tensor,
            densepose_map=densepose_tensor[:, :3],
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
