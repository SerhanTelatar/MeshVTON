"""
Differentiable Mesh Renderer — PyTorch3D-based 3D→2D rendering.

Renders the garment mesh draped on the SMPL-X body into a 2D image
in a differentiable way (gradients flow back into the mesh).
"""

from typing import Optional
import numpy as np
import torch
import torch.nn as nn


class MeshRenderer:
    """
    PyTorch3D-based differentiable mesh renderer.

    Converts (3D mesh + texture) into a 2D rendered image. Supports
    backpropagation during training.

    Args:
        image_size: Output render resolution.
        device: Compute device.
        shader_type: 'phong', 'flat', or 'silhouette'.
        faces_per_pixel: Number of faces rasterized per pixel.
        bg_color: Background color (R, G, B) in [0, 1].
    """

    def __init__(self, image_size: int = 512, device: str = "cuda",
                 shader_type: str = "phong", faces_per_pixel: int = 1,
                 bg_color: tuple[float, ...] = (1.0, 1.0, 1.0)):
        self.image_size = image_size
        self.device = device
        self.shader_type = shader_type
        self.faces_per_pixel = faces_per_pixel
        self.bg_color = bg_color
        self.renderer = None

    def setup(self):
        """Set up the PyTorch3D renderer components."""
        try:
            from pytorch3d.renderer import (
                MeshRenderer as PT3DMeshRenderer,
                MeshRasterizer,
                RasterizationSettings,
                SoftPhongShader,
                SoftSilhouetteShader,
                FoVPerspectiveCameras,
                PointLights,
                TexturesVertex,
                TexturesUV,
                look_at_view_transform,
            )

            # Camera setup
            R, T = look_at_view_transform(dist=2.7, elev=0, azim=0)
            self.cameras = FoVPerspectiveCameras(device=self.device, R=R, T=T)

            # Lighting
            self.lights = PointLights(
                device=self.device,
                location=[[0.0, 1.0, 2.0]],
                ambient_color=((0.5, 0.5, 0.5),),
                diffuse_color=((0.7, 0.7, 0.7),),
                specular_color=((0.3, 0.3, 0.3),),
            )

            # Rasterization settings
            raster_settings = RasterizationSettings(
                image_size=self.image_size,
                blur_radius=0.0,
                faces_per_pixel=self.faces_per_pixel,
            )

            # Shader selection
            if self.shader_type == "silhouette":
                shader = SoftSilhouetteShader()
            else:
                shader = SoftPhongShader(
                    device=self.device,
                    cameras=self.cameras,
                    lights=self.lights,
                )

            self.renderer = PT3DMeshRenderer(
                rasterizer=MeshRasterizer(
                    cameras=self.cameras,
                    raster_settings=raster_settings,
                ),
                shader=shader,
            )
            print(f"PyTorch3D renderer initialized: {self.shader_type}")

        except ImportError:
            print("Warning: PyTorch3D not found. Falling back to placeholder renderer.")
            self.renderer = None

    def render(self, vertices: torch.Tensor, faces: torch.Tensor,
               textures: Optional[torch.Tensor] = None,
               camera_params: Optional[dict] = None) -> torch.Tensor:
        """
        Render a 3D mesh into a 2D image.

        Args:
            vertices: (B, V, 3) vertex positions.
            faces: (F, 3) triangle face indices.
            textures: (B, V, 3) vertex colors or (B, H, W, 3) UV texture.
            camera_params: Optional camera parameter override.

        Returns:
            (B, 4, H, W) RGBA render result.
        """
        if self.renderer is None:
            self.setup()

        if self.renderer is not None:
            return self._render_pytorch3d(vertices, faces, textures, camera_params)
        else:
            return self._render_placeholder(vertices.shape[0])

    def _render_pytorch3d(self, vertices, faces, textures, camera_params):
        """Render using PyTorch3D."""
        from pytorch3d.structures import Meshes
        from pytorch3d.renderer import TexturesVertex

        b = vertices.shape[0]

        # Prepare texture
        if textures is None:
            textures = torch.ones_like(vertices) * 0.7  # default gray
        tex = TexturesVertex(verts_features=textures)

        # Build mesh
        faces_batch = faces.unsqueeze(0).expand(b, -1, -1) if faces.dim() == 2 else faces
        meshes = Meshes(verts=vertices, faces=faces_batch, textures=tex)

        # Update camera
        if camera_params is not None:
            from pytorch3d.renderer import look_at_view_transform, FoVPerspectiveCameras
            R, T = look_at_view_transform(
                dist=camera_params.get("dist", 2.7),
                elev=camera_params.get("elev", 0),
                azim=camera_params.get("azim", 0),
            )
            cameras = FoVPerspectiveCameras(device=self.device, R=R, T=T)
            self.renderer.rasterizer.cameras = cameras
            if hasattr(self.renderer.shader, "cameras"):
                self.renderer.shader.cameras = cameras

        # Render
        images = self.renderer(meshes)  # (B, H, W, 4) RGBA
        images = images.permute(0, 3, 1, 2)  # (B, 4, H, W)

        return images

    def render_uv(self, vertices: torch.Tensor, faces: torch.Tensor,
                  verts_uvs: torch.Tensor, faces_uvs: torch.Tensor,
                  texture_map: torch.Tensor,
                  camera_params: Optional[dict] = None,
                  flat: bool = True) -> torch.Tensor:
        """Render a mesh with a real UV texture map (preserves the garment pattern).

        Unlike `render` (TexturesVertex, which bakes color per vertex and loses
        fine pattern), this uses TexturesUV so the garment's actual print/texture
        shows up in the render that feeds GarmentNet / IP-Adapter.

        Args:
            vertices: (B, V, 3) draped vertex positions.
            faces: (F, 3) triangle vertex indices.
            verts_uvs: (1, V, 2) per-vertex UV coords (trimesh visual.uv).
            faces_uvs: (1, F, 3) UV index per face (== faces when uv is per-vertex).
            texture_map: (1, H, W, 3) texture image in [0, 1].
            camera_params: optional camera override (must match RGB/normal/depth azim).
        Returns:
            (B, 4, H, W) RGBA render.
        """
        if self.renderer is None:
            self.setup()
        if self.renderer is None:
            return self._render_placeholder(vertices.shape[0])

        from pytorch3d.structures import Meshes
        from pytorch3d.renderer import TexturesUV

        b = vertices.shape[0]
        tex = TexturesUV(
            maps=texture_map.to(self.device),
            faces_uvs=faces_uvs.to(self.device).long(),
            verts_uvs=verts_uvs.to(self.device),
            padding_mode="border",
            align_corners=True,
        )
        faces_batch = faces.unsqueeze(0).expand(b, -1, -1) if faces.dim() == 2 else faces
        meshes = Meshes(verts=vertices, faces=faces_batch, textures=tex)

        if camera_params is not None:
            from pytorch3d.renderer import look_at_view_transform, FoVPerspectiveCameras
            R, T = look_at_view_transform(
                dist=camera_params.get("dist", 2.7),
                elev=camera_params.get("elev", 0),
                azim=camera_params.get("azim", 0),
            )
            cameras = FoVPerspectiveCameras(device=self.device, R=R, T=T)
            self.renderer.rasterizer.cameras = cameras
            if hasattr(self.renderer.shader, "cameras"):
                self.renderer.shader.cameras = cameras

        # Flat/unlit: show the garment's true color without shading (like a product photo) →
        # in-distribution for the frozen GarmentNet/IP-Adapter. AmbientLights = ambient
        # term only (no diffuse/specular) → flat texture.
        orig_lights = getattr(self.renderer.shader, "lights", None)
        if flat:
            try:
                from pytorch3d.renderer import AmbientLights
                self.renderer.shader.lights = AmbientLights(device=self.device)
            except Exception:
                flat = False
        try:
            images = self.renderer(meshes)  # (B, H, W, 4)
        finally:
            if flat and orig_lights is not None:
                self.renderer.shader.lights = orig_lights
        return images.permute(0, 3, 1, 2)

    def _render_placeholder(self, batch_size: int) -> torch.Tensor:
        """Placeholder render when PyTorch3D is unavailable."""
        return torch.ones(batch_size, 4, self.image_size, self.image_size,
                          device=self.device) * 0.5

    def render_normal_map(self, vertices: torch.Tensor, faces: torch.Tensor,
                          camera_params: Optional[dict] = None) -> torch.Tensor:
        """Render a normal map (for geometry conditioning).

        `camera_params` (with `azim`) must match the RGB render so the 9-channel
        conditioning stays geometrically consistent across views.
        """
        b = vertices.shape[0]
        faces_in = faces  # keep original (F,3) for self.render below
        if faces.dim() == 2:
            faces = faces.unsqueeze(0).expand(b, -1, -1)

        # Compute face normals
        v0 = torch.gather(vertices, 1, faces[:, :, 0:1].expand(-1, -1, 3))
        v1 = torch.gather(vertices, 1, faces[:, :, 1:2].expand(-1, -1, 3))
        v2 = torch.gather(vertices, 1, faces[:, :, 2:3].expand(-1, -1, 3))

        normals = torch.cross(v1 - v0, v2 - v0, dim=-1)
        normals = normals / (normals.norm(dim=-1, keepdim=True) + 1e-8)

        # Average face normals into vertex normals
        vertex_normals = torch.zeros_like(vertices)
        for i in range(3):
            vertex_normals.scatter_add_(
                1,
                faces[:, :, i:i+1].expand(-1, -1, 3),
                normals,
            )
        vertex_normals = vertex_normals / (vertex_normals.norm(dim=-1, keepdim=True) + 1e-8)

        # Render normals as colors
        normal_colors = (vertex_normals + 1) / 2  # [-1,1] → [0,1]
        return self.render(vertices, faces_in, normal_colors, camera_params)

    def render_depth_map(self, vertices: torch.Tensor, faces: torch.Tensor,
                         camera_params: Optional[dict] = None) -> torch.Tensor:
        """Render a depth map.

        Depth is measured along the camera's view axis (camera-space z) so it
        stays correct when `camera_params` rotates the camera (e.g. azim=180 for
        a back-facing person). At azim=0 this matches the original world-z depth.
        """
        depth_z = self._camera_space_depth(vertices, camera_params)  # (B, V, 1)
        d_min = depth_z.min(dim=1, keepdim=True)[0]
        d_max = depth_z.max(dim=1, keepdim=True)[0]
        depth_normalized = (depth_z - d_min) / (d_max - d_min + 1e-8)
        depth_colors = depth_normalized.expand(-1, -1, 3)
        return self.render(vertices, faces, depth_colors, camera_params)

    def _camera_space_depth(self, vertices: torch.Tensor,
                            camera_params: Optional[dict] = None) -> torch.Tensor:
        """Per-vertex depth along the camera view axis → (B, V, 1).

        Falls back to world-z (the original behavior) if PyTorch3D is missing.
        """
        try:
            from pytorch3d.renderer import look_at_view_transform, FoVPerspectiveCameras
            cp = camera_params or {}
            R, T = look_at_view_transform(
                dist=cp.get("dist", 2.7),
                elev=cp.get("elev", 0),
                azim=cp.get("azim", 0),
            )
            cameras = FoVPerspectiveCameras(device=vertices.device, R=R, T=T)
            view_pts = cameras.get_world_to_view_transform().transform_points(vertices)
            return view_pts[..., 2:3]  # camera-space z = depth toward camera
        except Exception:
            return vertices[:, :, 2:3]

    def render_multiview(self, vertices: torch.Tensor, faces: torch.Tensor,
                         textures: Optional[torch.Tensor] = None,
                         num_views: int = 4) -> list[torch.Tensor]:
        """Render the mesh from multiple camera azimuths."""
        azimuths = torch.linspace(0, 360, num_views + 1)[:-1]
        renders = []
        for azim in azimuths:
            cam_params = {"dist": 2.7, "elev": 0, "azim": float(azim)}
            img = self.render(vertices, faces, textures, cam_params)
            renders.append(img)
        return renders
