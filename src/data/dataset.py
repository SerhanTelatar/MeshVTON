"""
TryOn Dataset — PyTorch Dataset for person-garment pairs.

Loads person images with preprocessing outputs (pose for augmentation,
agnostic mask) alongside garment images and optional 3D conditioning.
"""

import csv
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from PIL import Image

from src.data.transforms import TryOnTransforms
from src.utils.image_utils import pil_to_tensor


def _load_norm(path: Path, height: int, width: int, fill=(128, 128, 128)) -> torch.Tensor:
    """Load an RGB image, resize to (height,width), normalize to [-1,1] -> (3,H,W)."""
    if path.exists():
        img = Image.open(path).convert("RGB")
    else:
        img = Image.new("RGB", (width, height), fill)
    img = img.resize((width, height), Image.LANCZOS)
    t = torch.from_numpy(np.array(img, dtype=np.float32) / 255.0).permute(2, 0, 1)
    return t * 2.0 - 1.0


class MeshVTONDataset(Dataset):
    """
    MeshVTON training dataset — IDM-VTON input contract + 3D conditioning.

    Yields per sample (all images at (height, width)):
        person          (3,H,W) [-1,1]   target person wearing the garment
        masked_image    (3,H,W) [-1,1]   clothing-agnostic person (= agnostic)
        mask            (1,H,W) [0,1]     inpaint region (1 = clothing area)
        pose_img        (3,H,W) [-1,1]   densepose of the person
        cloth           (3,H,W) [-1,1]   3D garment rendered on the person's pose
        conditioning_3d (9,H,W) [-1,1]   render(3)+normal(3)+depth(3) for ControlNet3D
        caption         str

    Expected layout:
        {data_root}/raw/images/{person_id}.jpg
        {data_root}/processed/agnostic/{person_id}.jpg
        {data_root}/processed/densepose/{person_id}.jpg|png
        {data_root}/processed/renders_3d/{person_id}_{garment_id}.png
        {data_root}/processed/normal_maps/{person_id}_{garment_id}.png
        {data_root}/processed/depth_maps/{person_id}_{garment_id}.png
    """

    def __init__(self, pairs_file: str, data_root: str = "data",
                 height: int = 512, width: int = 384,
                 caption: str = "a photo of a person wearing clothes",
                 mask_threshold: float = 0.08):
        self.data_root = Path(data_root)
        self.height = height
        self.width = width
        self.caption = caption
        self.mask_threshold = mask_threshold
        self.image_dir = self.data_root / "raw" / "images"
        self.agnostic_dir = self.data_root / "processed" / "agnostic"
        self.densepose_dir = self.data_root / "processed" / "densepose"
        self.renders_3d_dir = self.data_root / "processed" / "renders_3d"
        self.normal_maps_dir = self.data_root / "processed" / "normal_maps"
        self.depth_maps_dir = self.data_root / "processed" / "depth_maps"

        # Keep only pairs that actually have a 3D render (render_garment.py skips
        # pairs without SMPL-X params or a matching mesh) — otherwise we'd train
        # on gray-fill conditioning, which is useless.
        all_pairs = TryOnDataset._load_pairs(self, pairs_file)
        self.pairs = [
            (p, g) for (p, g) in all_pairs
            if (self.renders_3d_dir / f"{p}_{g}.png").exists()
        ]
        print(f"MeshVTONDataset: {len(self.pairs)}/{len(all_pairs)} pairs have 3D renders")

    def __len__(self) -> int:
        return len(self.pairs)

    def _densepose_path(self, person_id: str) -> Path:
        for ext in (".jpg", ".png", ".jpeg"):
            p = self.densepose_dir / f"{person_id}{ext}"
            if p.exists():
                return p
        return self.densepose_dir / f"{person_id}.jpg"

    def __getitem__(self, idx: int) -> dict:
        person_id, garment_id = self.pairs[idx]
        H, W = self.height, self.width
        render_name = f"{person_id}_{garment_id}"

        person = _load_norm(self.image_dir / f"{person_id}.jpg", H, W)
        masked_image = _load_norm(self.agnostic_dir / f"{person_id}.jpg", H, W)
        pose_img = _load_norm(self._densepose_path(person_id), H, W, fill=(0, 0, 0))

        cloth = _load_norm(self.renders_3d_dir / f"{render_name}.png", H, W)
        normal = _load_norm(self.normal_maps_dir / f"{render_name}.png", H, W)
        depth = _load_norm(self.depth_maps_dir / f"{render_name}.png", H, W)
        conditioning_3d = torch.cat([cloth, normal, depth], dim=0)  # (9,H,W)

        # Inpaint mask = where the agnostic differs from the person (clothing area).
        diff = (person - masked_image).abs().mean(dim=0, keepdim=True)  # (1,H,W) in [0,2]
        mask = (diff > self.mask_threshold).float()

        return {
            "person": person,
            "masked_image": masked_image,
            "mask": mask,
            "pose_img": pose_img,
            "cloth": cloth,
            "conditioning_3d": conditioning_3d,
            "caption": self.caption,
            "person_id": person_id,
            "garment_id": garment_id,
        }


class TryOnDataset(Dataset):
    """
    Dataset for virtual try-on training.

    Expects preprocessed data in the following structure:
        data/raw/images/{image_id}.jpg
        data/processed/poses/{image_id}.npy          (for flip augmentation)
        data/processed/agnostic/{image_id}.jpg
        data/processed/renders_3d/{p}_{g}.png        (3D conditioning)
        data/processed/normal_maps/{p}_{g}.png
        data/processed/depth_maps/{p}_{g}.png

    Args:
        pairs_file: CSV file with columns (person_id, garment_id).
        data_root: Root data directory.
        resolution: Target image resolution.
        transforms: Optional augmentation transforms.
        use_3d: Include 3D conditioning tensor (9-ch: RGB+normal+depth).
    """

    def __init__(self, pairs_file: str, data_root: str = "data",
                 resolution: int = 512, transforms: Optional[TryOnTransforms] = None,
                 use_3d: bool = True):
        self.data_root = Path(data_root)
        self.resolution = resolution
        self.transforms = transforms
        self.use_3d = use_3d

        # Load pairs
        self.pairs = self._load_pairs(pairs_file)

        # 2D Directories
        self.image_dir = self.data_root / "raw" / "images"
        self.pose_dir = self.data_root / "processed" / "poses"
        self.agnostic_dir = self.data_root / "processed" / "agnostic"

        # 3D Directories
        self.garments_3d_dir = self.data_root / "garments_3d"
        self.renders_3d_dir = self.data_root / "processed" / "renders_3d"
        self.normal_maps_dir = self.data_root / "processed" / "normal_maps"
        self.depth_maps_dir = self.data_root / "processed" / "depth_maps"

    def _load_pairs(self, pairs_file: str) -> list[tuple[str, str]]:
        pairs = []
        path = Path(pairs_file)
        if not path.exists():
            return pairs
        with open(path, "r") as f:
            reader = csv.reader(f)
            next(reader, None)  # Skip header
            for row in reader:
                if len(row) >= 2:
                    pairs.append((row[0].strip(), row[1].strip()))
        return pairs

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        person_id, garment_id = self.pairs[idx]

        # Load person image
        person_img = self._load_image(self.image_dir / f"{person_id}.jpg")

        # Load garment image (2D flat photo as fallback)
        garment_img = self._load_image(self.image_dir / f"{garment_id}.jpg")

        # Load 2D preprocessed data
        pose_map = self._load_pose(person_id)
        agnostic_img = self._load_agnostic(person_id)

        # Apply transforms
        if self.transforms:
            person_img, garment_img, pose_map, agnostic_img = self.transforms(
                person_img, garment_img, pose_map, agnostic_img
            )

        sample = {
            "person_image": pil_to_tensor(person_img, self.resolution),
            "garment_image": pil_to_tensor(garment_img, self.resolution),
            "agnostic_image": pil_to_tensor(agnostic_img, self.resolution),
            "person_id": person_id,
            "garment_id": garment_id,
        }

        # 3D conditioning: RGB render + normal map + depth map → (9, H, W)
        if self.use_3d:
            render_name = f"{person_id}_{garment_id}"
            render_3d = pil_to_tensor(
                self._load_image(self.renders_3d_dir / f"{render_name}.png"),
                self.resolution,
            )
            normal_map = pil_to_tensor(
                self._load_image(self.normal_maps_dir / f"{render_name}.png"),
                self.resolution,
            )
            depth_map = self._load_depth(
                self.depth_maps_dir / f"{render_name}.png"
            ).expand(3, -1, -1)
            sample["conditioning_3d"] = torch.cat([render_3d, normal_map, depth_map], dim=0)

        return sample

    def _load_image(self, path: Path) -> Image.Image:
        if path.exists():
            return Image.open(path).convert("RGB")
        return Image.new("RGB", (self.resolution, self.resolution), (128, 128, 128))

    def _load_pose(self, person_id: str) -> np.ndarray:
        path = self.pose_dir / f"{person_id}.npy"
        if path.exists():
            return np.load(path)
        return np.zeros((18, 3), dtype=np.float32)

    def _load_agnostic(self, person_id: str) -> Image.Image:
        path = self.agnostic_dir / f"{person_id}.jpg"
        if path.exists():
            return Image.open(path).convert("RGB")
        return Image.new("RGB", (self.resolution, self.resolution), (128, 128, 128))

    def _load_depth(self, path: Path) -> torch.Tensor:
        """Load a depth map (grayscale PNG → tensor)."""
        if path.exists():
            depth = np.array(Image.open(path).convert("L"), dtype=np.float32) / 255.0
        else:
            depth = np.zeros((self.resolution, self.resolution), dtype=np.float32)
        tensor = torch.from_numpy(depth).unsqueeze(0)  # (1, H, W)
        if tensor.shape[-1] != self.resolution:
            tensor = torch.nn.functional.interpolate(
                tensor.unsqueeze(0), size=(self.resolution, self.resolution),
                mode="bilinear", align_corners=False,
            ).squeeze(0)
        return tensor

