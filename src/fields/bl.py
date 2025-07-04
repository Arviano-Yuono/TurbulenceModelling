import numpy as np


class BoundaryLayer:
    def __init__(self, U_e: np.ndarray, AoA_rad: float, N: int, methods: str = "thwaites"):
        self.U_e = U_e
        self.AoA_rad = AoA_rad
        self.methods = methods
        self.theta = None
        self.delta_star = None
        self.H = None
        self.transition_index_upper = None
        self.transition_index_lower = None

    def update_edge_velocity(self, U_e: float):
        self.U_e = U_e

    def update_boundary_layer(self, theta: np.ndarray, delta_star: np.ndarray, H: np.ndarray, transition_index_upper: int, transition_index_lower: int):
        self.theta = theta
        self.delta_star = delta_star
        self.H = H
        self.transition_index_upper = transition_index_upper
        self.transition_index_lower = transition_index_lower

    def get_boundary_layer_thickness(self):
        pass