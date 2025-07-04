import numpy as np
from typing import Optional

from .bl import BoundaryLayer

class Body:
    def __init__(self, XB: np.ndarray, YB: np.ndarray):
        self.XB = XB
        self.YB = YB
        self.N = int(0.5 * (len(XB) - 1))
        self.boundary_layer = None
        self.U_e = None
        self.stagnation_index = None
        self.transition_index_upper = None
        self.transition_index_lower = None

    def get_stagnation_points(self):
        assert self.U_e is not None, "Edge velocity is not set"
        self.stagnation_index = int(np.argmin(np.abs(self.U_e)))
        return self.XB[self.stagnation_index], self.YB[self.stagnation_index], self.U_e[self.stagnation_index]

    def update_edge_velocity(self, U_e: np.ndarray):
        self.U_e = U_e
        if self.boundary_layer is not None:
            self.boundary_layer.update_edge_velocity(U_e=U_e)
        else:
            self.boundary_layer = BoundaryLayer(U_e=U_e, AoA_rad=0, N=self.N, methods="thwaites")

    def get_body_points(self) -> tuple[np.ndarray, np.ndarray]:
        return self.XB, self.YB

    def add_boundary_layer(self, boundary_layer: BoundaryLayer):
        self.boundary_layer = boundary_layer

    def update_boundary_layer(self, theta: np.ndarray, delta_star: np.ndarray, H: np.ndarray, transition_index_upper: Optional[int] = None, transition_index_lower: Optional[int] = None):
        if transition_index_upper is None:
            transition_index_upper = len(theta) - 1
        if transition_index_lower is None:
            transition_index_lower = transition_index_upper
        self.transition_index_upper = transition_index_upper
        self.transition_index_lower = transition_index_lower
        self.boundary_layer.update_boundary_layer(theta=theta, delta_star=delta_star, H=H, transition_index_upper=transition_index_upper, transition_index_lower=transition_index_lower)

    def displace_geometry_by_dstar(self):
        # Midpoints (control points)
        Xc = 0.5 * (self.XB[:-1] + self.XB[1:])
        Yc = 0.5 * (self.YB[:-1] + self.YB[1:])

        # Panel direction and normals
        dX = self.XB[1:] - self.XB[:-1]
        dY = self.YB[1:] - self.YB[:-1]
        lengths = np.sqrt(dX**2 + dY**2)
        tx = dX / lengths
        ty = dY / lengths
        nx = -ty
        ny = tx
        
        delta_bl = self.boundary_layer.delta_star if self.boundary_layer is not None else np.zeros_like(Xc)
        
        # Project delta_star onto the panel normals
        delta_bl_x = delta_bl * nx
        delta_bl_y = delta_bl * ny
        
        # # Pad delta_star to match the number of panels for displacement
        # delta_star_full = np.zeros_like(Xc)
        # if self.boundary_layer is not None and self.boundary_layer.delta_star is not None:
        #     n_partial = len(self.boundary_layer.delta_star)
        #     delta_star_full[:n_partial] = self.boundary_layer.delta_star
            
        # displacement_sign = np.ones_like(delta_star_full)
        # displacement_sign[:self.stagnation_index] = -1  # Flip for lower surface (reverse indexing)

        # Xc_disp = Xc + delta_star_full * nx
        # Yc_disp = Yc + delta_star_full * ny
        
        Xc_disp = Xc + delta_bl_x
        Yc_disp = Yc + delta_bl_y

        # Rebuild new boundary nodes
        XB_new = np.zeros_like(self.XB)
        YB_new = np.zeros_like(self.YB)

        for i in range(1, len(self.XB) - 1):
            XB_new[i] = 0.5 * (Xc_disp[i - 1] + Xc_disp[i])
            YB_new[i] = 0.5 * (Yc_disp[i - 1] + Yc_disp[i])
            
        XB_new[0] = 2 * Xc_disp[0] - XB_new[1]
        YB_new[0] = 2 * Yc_disp[0] - YB_new[1]
        XB_new[-1] = 2 * Xc_disp[-1] - XB_new[-2]
        YB_new[-1] = 2 * Yc_disp[-1] - YB_new[-2]

        return XB_new, YB_new
    
    def update_geometry(self, XB_new: np.ndarray, YB_new: np.ndarray):
        self.XB = XB_new
        self.YB = YB_new