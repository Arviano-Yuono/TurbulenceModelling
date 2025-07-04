import numpy as np
from typing import Optional

class Freestream:
    def __init__(self, 
                 U_inf: float, 
                 AoA_rad: float,
                 density: float,
                 viscosity: float,
                 nu: float,
                 Re: Optional[float] = None):
        """
        Freestream class.

        Args:
            U_inf: Freestream velocity.
            AoA_rad: Angle of attack in radians.
            density: Density.
            viscosity: Viscosity.
            nu: Kinematic viscosity.
            Re: Reynolds number (default: None).
        """
        self.U_inf = U_inf
        self.AoA_rad = AoA_rad
        self.density = density
        self.viscosity = viscosity
        self.nu = nu
        self.Re = Re

    def get_freestream_velocity(self) -> np.ndarray:
        return self.U_inf * np.array([np.cos(self.AoA_rad), np.sin(self.AoA_rad)])