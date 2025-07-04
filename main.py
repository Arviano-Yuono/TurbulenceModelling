import numpy as np
from src.fields.body import Body
from src.fields.freestream import Freestream
from src.solver.inviscid import VortexPanelMethod
from src.solver.bl_march import BoundaryLayerSolver
from src.utils.naca_generator import generate_naca4


class Simulation:
    def __init__(
        self,
        naca_series: str,
        N_panels: int,
        AoA_deg: float,
        Re: float,
        body: Body,
        freestream: Freestream,
        inviscid_solver: VortexPanelMethod,
        bl_solver: BoundaryLayerSolver,
        max_outer_iter: int = 50,
        convergence_tol: float = 1e-4,
    ):
        self.naca_series = naca_series
        self.N_panels = N_panels
        self.AoA_deg = AoA_deg
        self.Re = Re
        self.body = body
        self.freestream = freestream
        self.inviscid_solver = inviscid_solver
        self.bl_solver = bl_solver
        self.max_outer_iter = max_outer_iter
        self.convergence_tol = convergence_tol

    def solve(self):
        # Step 3: Outer iteration loop
        for outer_iter in range(self.max_outer_iter):

        # Step 3a: Solve inviscid flow
        Xc, Ue_ratio, Cp = self.inviscid_solver.solve(self.body, self.freestream)
        Ue = self.freestream.U_inf * Ue_ratio

        # Step 3b: Solve BL over chord
        theta, delta_star, Cf, transition_index = self.bl_solver.march(self.body, self.freestream)

        # Step 3c: Displace geometry using delta*
        XB_new, YB_new = self.body.displace_geometry_by_dstar(delta_star)

        # Step 3d: Recompute inviscid flow with displaced boundary
        Xc_new, Ue_ratio_new, _ = self.inviscid_solver.solve(self.body, self.freestream)
        Ue_new = self.freestream.U_inf * Ue_ratio_new

        # Step 3e: Check for convergence
        rel_err = np.linalg.norm(Ue_new - Ue) / np.linalg.norm(Ue)
        if rel_err < self.convergence_tol:
            print(f"Converged in {outer_iter+1} iterations.")
            break

        # Step 3f: Update XB, YB for next iteration
        XB, YB = XB_new.copy(), YB_new.copy()

    # Return all relevant results
    return {
        "X": Xc_new,
        "Cp": Cp,
        "Ue": Ue_new,
        "theta": theta,
        "delta_star": delta_star,
        "Cf": Cf,
        "transition_index": transition_index,
        "geometry": (XB, YB)
    }


# if __name__ == "__main__":
    # solve()
