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
from torch.utils.data import Dataset
from PIL import Image

from src.data.transforms import TryOnTransforms
from src.utils.image_utils import pil_to_tensor


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

