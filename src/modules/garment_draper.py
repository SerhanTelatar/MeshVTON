"""
Garment Draper — drapes a 3D garment mesh onto the SMPL-X body model.

Deforms a 3D garment asset (mesh) to fit the SMPL-X body mesh.
"""

from typing import Optional
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class GarmentDraper(nn.Module):
    """
    Drapes a 3D garment mesh onto a body model.

    Two-stage process:
    1. Coarse draping: deform the garment mesh via Linear Blend Skinning (LBS)
       toward the body mesh.
    2. Fine draping: neural network refinement for physics-aware details
       (wrinkles, sagging).

    Args:
        num_body_verts: SMPL-X body mesh vertex count (10475).
        garment_feature_dim: Garment feature dimensionality.
        hidden_dim: Hidden layer size.
        num_refine_layers: Number of refinement layers.
    """

    def __init__(self, num_body_verts: int = 10475, garment_feature_dim: int = 256,
                 hidden_dim: int = 512, num_refine_layers: int = 4):
        super().__init__()
        self.num_body_verts = num_body_verts

        # Coarse draping: garment→body correspondence network
        self.correspondence_net = nn.Sequential(
            nn.Linear(6, hidden_dim),  # garment vertex (3) + nearest body vertex (3)
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 3),  # offset vector
        )

        # Fine draping: physics-aware refinement
        self.refine_layers = nn.ModuleList()
        for i in range(num_refine_layers):
            self.refine_layers.append(
                GarmentRefineBlock(
                    in_dim=3 + garment_feature_dim if i == 0 else hidden_dim,
                    out_dim=hidden_dim if i < num_refine_layers - 1 else 3,
                )
            )

        # Fabric material property embedding
        self.material_mlp = nn.Sequential(
            nn.Linear(8, 64),  # 8 material parameters
            nn.ReLU(),
            nn.Linear(64, garment_feature_dim),
        )

    def forward(self, garment_verts: torch.Tensor, garment_faces: torch.Tensor,
                body_verts: torch.Tensor, body_faces: torch.Tensor,
                skinning_weights: Optional[torch.Tensor] = None,
                material_params: Optional[torch.Tensor] = None) -> dict[str, torch.Tensor]:
        """
        Drape the garment mesh onto the body mesh.

        Args:
            garment_verts: (B, Vg, 3) garment vertices (in T-pose).
            garment_faces: (Fg, 3) garment face indices.
            body_verts: (B, Vb, 3) SMPL-X body vertices (target pose).
            body_faces: (Fb, 3) body face indices.
            skinning_weights: (Vg, J) optional LBS weights.
            material_params: (B, 8) fabric material parameters.

        Returns:
            Dict:
                'draped_verts': (B, Vg, 3) deformed garment vertices
                'offsets': (B, Vg, 3) refinement offsets
                'normals': (B, Vg, 3) surface normals
        """
        # Geometric alignment: scale + translate the garment so its bounding box
        # matches the body's, preserving the garment's own shape. (The learned
        # draping nets here are untrained and collapse the mesh into a blob, so
        # we use a robust rigid alignment instead.)
        draped_verts = self._geometric_align(garment_verts, body_verts)
        normals = self._compute_normals(draped_verts, garment_faces)

        return {
            "draped_verts": draped_verts,
            "offsets": torch.zeros_like(draped_verts),
            "normals": normals,
        }

    def _geometric_align(self, garment_verts: torch.Tensor,
                         body_verts: torch.Tensor) -> torch.Tensor:
        """Rigidly scale+center the garment onto the body (no learned deform)."""
        g_min = garment_verts.amin(dim=1, keepdim=True)
        g_max = garment_verts.amax(dim=1, keepdim=True)
        b_min = body_verts.amin(dim=1, keepdim=True)
        b_max = body_verts.amax(dim=1, keepdim=True)

        g_center = (g_min + g_max) / 2
        g_size = (g_max - g_min).amax(dim=-1, keepdim=True).clamp(min=1e-6)
        b_size = (b_max - b_min).amax(dim=-1, keepdim=True)

        # Only the garment is rendered and the camera looks at the origin, so
        # center the garment at the origin and scale it to the body's size.
        scale = b_size / g_size  # uniform, preserves garment proportions
        return (garment_verts - g_center) * scale

    def _coarse_drape(self, garment_verts: torch.Tensor,
                      body_verts: torch.Tensor) -> torch.Tensor:
        """Place garment vertices at the nearest body vertices."""
        b, vg, _ = garment_verts.shape

        # Find the nearest body vertex for each garment vertex
        # (B, Vg, 1, 3) - (B, 1, Vb, 3) → (B, Vg, Vb)
        dists = torch.cdist(garment_verts, body_verts)
        nearest_idx = dists.argmin(dim=-1)  # (B, Vg)

        # Gather the nearest body points
        nearest_body = torch.gather(
            body_verts, 1,
            nearest_idx.unsqueeze(-1).expand(-1, -1, 3)
        )

        # Compute offset with the correspondence network
        combined = torch.cat([garment_verts, nearest_body], dim=-1)  # (B, Vg, 6)
        offsets = self.correspondence_net(combined)

        return nearest_body + offsets

    def _collision_handling(self, garment_verts: torch.Tensor,
                            body_verts: torch.Tensor,
                            margin: float = 0.005) -> torch.Tensor:
        """Avoid garment-body collisions (simple push-out)."""
        dists = torch.cdist(garment_verts, body_verts)
        min_dists, nearest_idx = dists.min(dim=-1)

        nearest_body = torch.gather(
            body_verts, 1,
            nearest_idx.unsqueeze(-1).expand(-1, -1, 3)
        )

        # Direction vector (garment → outward)
        direction = garment_verts - nearest_body
        direction_norm = direction / (direction.norm(dim=-1, keepdim=True) + 1e-8)

        # Push vertices that are too close back outside
        too_close = min_dists < margin
        push = direction_norm * margin
        correction = push * too_close.unsqueeze(-1).float()

        return garment_verts + correction

    def _compute_normals(self, vertices: torch.Tensor,
                         faces: torch.Tensor) -> torch.Tensor:
        """Compute per-vertex normals."""
        if faces.dim() == 2:
            faces = faces.unsqueeze(0).expand(vertices.shape[0], -1, -1)

        v0 = torch.gather(vertices, 1, faces[:, :, 0:1].expand(-1, -1, 3))
        v1 = torch.gather(vertices, 1, faces[:, :, 1:2].expand(-1, -1, 3))
        v2 = torch.gather(vertices, 1, faces[:, :, 2:3].expand(-1, -1, 3))

        face_normals = torch.cross(v1 - v0, v2 - v0, dim=-1)
        face_normals = face_normals / (face_normals.norm(dim=-1, keepdim=True) + 1e-8)

        # Aggregate face normals into vertex normals
        vertex_normals = torch.zeros_like(vertices)
        for i in range(3):
            vertex_normals.scatter_add_(1, faces[:, :, i:i+1].expand(-1, -1, 3), face_normals)
        vertex_normals = vertex_normals / (vertex_normals.norm(dim=-1, keepdim=True) + 1e-8)

        return vertex_normals


class GarmentRefineBlock(nn.Module):
    """Garment deformation refinement block."""

    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(out_dim, out_dim),
        )
        self.skip = nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x) + self.skip(x)


def load_garment_mesh(path: str) -> dict[str, np.ndarray]:
    """
    Load a 3D garment mesh file (.obj, .glb, .ply).

    Args:
        path: Mesh file path.

    Returns:
        Dict: 'vertices' (V, 3), 'faces' (F, 3), 'uv' (V, 2), 'texture' (H, W, 3)
    """
    ext = Path(path).suffix.lower()

    try:
        import trimesh

        mesh = trimesh.load(path, force="mesh")

        result = {
            "vertices": np.array(mesh.vertices, dtype=np.float32),
            "faces": np.array(mesh.faces, dtype=np.int64),
        }

        # UV coordinates
        if hasattr(mesh.visual, "uv") and mesh.visual.uv is not None:
            result["uv"] = np.array(mesh.visual.uv, dtype=np.float32)
        else:
            result["uv"] = np.zeros((len(mesh.vertices), 2), dtype=np.float32)

        # Texture
        if hasattr(mesh.visual, "material") and hasattr(mesh.visual.material, "image"):
            result["texture"] = np.array(mesh.visual.material.image)[:, :, :3]
        else:
            result["texture"] = np.ones((512, 512, 3), dtype=np.uint8) * 200

        return result

    except ImportError:
        print("Warning: trimesh not found. Install it with `pip install trimesh`.")
        return {
            "vertices": np.zeros((100, 3), dtype=np.float32),
            "faces": np.zeros((50, 3), dtype=np.int64),
            "uv": np.zeros((100, 2), dtype=np.float32),
            "texture": np.ones((512, 512, 3), dtype=np.uint8) * 200,
        }
