import torch
import torch.nn as nn
import torch.nn.functional as F

class TPSGridGen(nn.Module):
    def __init__(self, height, width, num_control_points):
        super(TPSGridGen, self).__init__()
        self.height = height
        self.width = width
        self.num_control_points = num_control_points

        self.grid_size = int(num_control_points ** 0.5)

        target_control_points = self._create_control_points()
        self.register_buffer("target_control_points", target_control_points)

        target_coordinate = self._create_coordinate_grid()
        self.register_buffer("target_coordinate", target_coordinate)

    def _create_control_points(self):
        axis = torch.linspace(-1, 1, 5)
        grid_y, grid_x = torch.meshgrid(axis, axis, indexing="ij")
        control_points = torch.stack([grid_x.flatten(), grid_y.flatten()], dim=1)
        return control_points

    def _create_coordinate_grid(self):
        y = torch.linspace(-1, 1, self.height)
        x = torch.linspace(-1, 1, self.width)
        grid_y, grid_x = torch.meshgrid(y, x, indexing="ij")
        coordinates = torch.stack([grid_x.flatten(), grid_y.flatten()], dim=1)
        return coordinates
        
    def _compute_radial_basis(self, p1, p2):
        diff = p1.unsqueeze(-2) - p2.unsqueeze(-3)
        diff_sq = (diff ** 2).sum(dim=-1)
        diff_sq = torch.clamp(diff_sq, min=1e-6)
        return diff_sq * torch.log(diff_sq) * 0.5

    def _precompute_tps_matrices(self):


    def forward(self, source_control_points):
         