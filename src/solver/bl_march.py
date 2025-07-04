import numpy as np
from src.fields.body import Body
from src.fields.freestream import Freestream

class BoundaryLayerSolver:
    def __init__(self, 
                 laminar_methods: str = "thwaites",
                 transition_method: str = "michel",
                 turbulent_methods: str = "drela"
                 ):
        """
        Boundary layer flow solver over a body by marching method.

        Args:
            laminar_methods: Methods for laminar boundary layer (default: thwaites).
            transition_method: Method for transition from laminar to turbulent (default: michel).
            turbulent_methods: Methods for turbulent boundary layer (default: drela)..
        """
        self.laminar_methods = laminar_methods
        self.transition_method = transition_method
        self.turbulent_methods = turbulent_methods

    def solve(self, body: Body, freestream: Freestream):
        """
        Solve boundary layer flow over a body by marching method.

        Args:
            x: Chordwise position.
            Ue: Edge velocity.
            nu: Kinematic viscosity.
            rho: Density.
            Re: Reynolds number.

        Returns:
            theta: Displacement thickness.
            delta_star: Displacement thickness.
            Cf: Skin friction coefficient.
            transition_index: Index of the transition point
        """
        # theta, delta_star, Cf, transition_index = solve_bl_march(
        #     x=Xc,
        #     Ue=Ue,
        #     nu=nu,
        #     rho=rho,
        #     Re=Re,
        #     method_transition="michel"
        # )


    # def compute_momentum_thickness(self, body: Body, freestream: Freestream):
    #     """
    #     Compute θ, δ*, H along the laminar boundary layer using Thwaites' method.

    #     Args:
    #         body: Body object
    #         freestream: Freestream object

    #     Returns:
    #         theta : momentum thickness array
    #         delta_star : displacement thickness array
    #         H : shape factor array
    #         transition_index : index at which transition occurs
    #     """
    #     assert body.stagnation_index is not None, "Stagnation index is not set"
    #     N = body.N
    #     U_e = body.U_e
    #     stagnation_index = body.stagnation_index
    #     lower_bl_index = stagnation_index
    #     upper_bl_index = stagnation_index + 1
    #     nu = freestream.nu
    #     theta2 = np.zeros(N)
    #     theta = np.zeros(N)
    #     H = np.zeros(N)
    #     delta_star = np.zeros(N)
    #     Re_theta = np.zeros(N)

    #     u5 = U_e**5
    #     integral = np.zeros(N)
    #     # print(f"U_e before loop: {U_e}")
    #     for i in range(1, N):
    #         integral[i] = integral[i-1] + 0.5 * (u5[i] + u5[i-1]) * (body.XB[i] - body.XB[i-1])
    #         # print(f"integral: {integral[i]}")
    #         theta2[i] = 0.45 * integral[i] / U_e[i]**6
    #         theta[i] = np.sqrt(theta2[i])

    def solve_laminar_boundary_layer(self, body: Body, freestream: Freestream):
        """
        Compute θ, δ*, H, and transition index for both upper and lower surfaces using Thwaites’ method.

        Returns:
            theta          : full momentum thickness array
            delta_star     : full displacement thickness array
            H              : full shape factor array
            transition_upper_index : index of transition on upper surface
            transition_lower_index : index of transition on lower surface
        """
        def S_laminar(lam):
            if lam >= 0 and lam <= 0.1:
                return 0.22 + 1.57 * lam - 1.8 * lam**2
            elif lam >= -0.1 and lam < 0:
                return 0.22 + 1.402 * lam + (0.018 * lam) / (0.107 + lam)
            else:
                # fallback to White's general formula
                return (lam + 0.09)**0.62

        def H_laminar(lam):
            if lam >= 0 and lam <= 0.1:
                return 2.61 - 3.75 * lam + 5.24 * lam**2
            elif lam >= -0.1 and lam < 0:
                return (0.0731 / (0.14 + lam)) + 2.088
            else:
                # fallback to White's general formula
                z = 0.25 - lam
                return 2.0 + 4.14*z - 83.5*z**2 + 854*z**3 - 3337*z**4 + 4576*z**5
            
        assert body.stagnation_index is not None
        N = 2 * body.N
        mid = body.stagnation_index
        U_e = body.U_e
        X = body.XB
        nu = freestream.nu

        # Allocate arrays
        theta = np.zeros(N)
        delta_star = np.zeros(N)
        H = np.zeros(N)
        Re_theta = np.zeros(N)
        dudx_log = np.zeros(N)

        transition_upper_index = N - 1
        transition_lower_index = 0

        # --- Lower Surface (Backward Marching) ---
        integral = 0.0
        for i in range(mid - 1, 0, -1):
            dx = X[i] - X[i + 1]
            integral += 0.5 * (U_e[i]**5 + U_e[i+1]**5) * abs(dx)
            theta2 = 0.45 * nu * np.abs(integral) / U_e[i]**6

            theta[i] = np.sqrt(theta2)
            dUedx = (U_e[i] - U_e[i + 1]) / abs(dx)
            dudx_log[i] = dUedx
            lam = theta2 * dUedx / nu
            S = S_laminar(lam)
            H[i] = H_laminar(lam)
            delta_star[i] = H[i] * theta[i]
            # print(f"dx: {dx}, theta[{i}]: {theta[i]}, dUedx[{i}]: {dUedx}, lam[{i}]: {lam}, S[{i}]: {S}, z[{i}]: {z}, theta2[{i}]: {theta2}, integral[{i}]: {integral}, delta_star[{i}]: {delta_star[i]}, H[{i}]: {H[i]}")


            Re_theta[i] = U_e[i] * theta[i] / nu
            michel_crit = Re_theta[i] * (H[i] + 3.3) / (H[i] + 1)
            if michel_crit > 1.174e6:
                transition_lower_index = i
                break

        # --- Upper Surface (Forward Marching) ---
        integral = 0.0
        for i in range(mid + 1, N ):
            dx = X[i] - X[i - 1]
            integral += 0.5 * (U_e[i]**5 + U_e[i-1]**5) * dx # ini trapezoid

            theta2 = 0.45 * nu * np.abs(integral) / U_e[i]**6
            theta[i] = np.sqrt(theta2)

            dUedx = (U_e[i] - U_e[i - 1]) / dx
            dudx_log[i] = dUedx
            lam = theta2 * dUedx / nu
            S = S_laminar(lam)
            H[i] = H_laminar(lam)
            delta_star[i] = H[i] * theta[i]
            if i == body.stagnation_index:
                delta_star[i] = 0
            # print(f"dx: {dx}, theta[{i}]: {theta[i]}, dUedx[{i}]: {dUedx}, lam[{i}]: {lam}, S[{i}]: {S}, z[{i}]: {z}, theta2[{i}]: {theta2}, integral[{i}]: {integral}, delta_star[{i}]: {delta_star[i]}, H[{i}]: {H[i]}")

            Re_theta[i] = U_e[i] * theta[i] / nu
            michel_crit = Re_theta[i] * (H[i] + 3.3) / (H[i] + 1)
            if michel_crit > 1.174e6:
                transition_upper_index = i
                break

        return theta, delta_star, H, transition_upper_index, transition_lower_index, dudx_log

