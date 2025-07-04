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
            turbulent_methods: Methods for turbulent boundary layer (default: drela).
        """
        self.laminar_methods = laminar_methods
        self.transition_method = transition_method
        self.turbulent_methods = turbulent_methods

    def solve(self, body: Body, freestream: Freestream):
        """
        Solve boundary layer flow over a body by marching method.
        """
        if self.laminar_methods == "thwaites":
            return self.solve_laminar_boundary_layer(body, freestream)
        else:
            raise NotImplementedError("Only Thwaites method is implemented for laminar flow.")

    def solve_laminar_boundary_layer(self, body: Body, freestream: Freestream):
        """
        Compute θ, δ*, H, and transition index for both upper and lower surfaces using Thwaites’ method.
        """
        def head_method_rhs(s, thetaH, Ue, dUeds, nu):
            theta, H = thetaH
            if H <= 1.6:
                H1 = 3.3 + 0.8234 * (H - 1.1)**(-1.287)
                dH1dH = -1.287 * 0.8234 * (H - 1.1)**(-2.287)
            else:
                H1 = 3.3 + 1.5501 * (H - 0.6778)**(-3.064)
                dH1dH = -3.064 * 1.5501 * (H - 0.6778)**(-4.064)
            Re_theta = Ue * theta / nu
            Cf = 0.246 * Re_theta**(-0.268) * 10**(-0.678 * H)
            dthetads = Cf / 2 - theta / Ue * (2 + H) * dUeds
            F1 = 0.0306 * (H1 - 3)**(-0.6169)
            dHds = (F1 - H1 * (theta * dUeds / Ue + dthetads)) / (theta * dH1dH)
            return np.array([dthetads, dHds])

        def rk4_step(func, s, y, h, *args):
            k1 = func(s, y, *args)
            k2 = func(s + h / 2, y + h / 2 * k1, *args)
            k3 = func(s + h / 2, y + h / 2 * k2, *args)
            k4 = func(s + h, y + h * k3, *args)
            return y + (h / 6) * (k1 + 2 * k2 + 2 * k3 + k4)

        assert body.stagnation_index is not None
        assert body.U_e is not None
        assert body.XB is not None
        assert body.YB is not None

        N = 2 * body.N
        mid = body.stagnation_index
        U_e = body.U_e
        X, Y = body.XB, body.YB
        nu = freestream.nu

        s = np.zeros(N)
        panel_lengths = np.sqrt((X[1:] - X[:-1])**2 + (Y[1:] - Y[:-1])**2)
        
        for i in range(mid - 1, -1, -1):
            s[i] = s[i+1] - panel_lengths[i]
        for i in range(mid + 1, N):
            s[i] = s[i-1] + panel_lengths[i-1]

        theta = np.zeros(N)
        delta_star = np.zeros(N)
        H = np.zeros(N)
        duds = np.zeros(N)
        transition_upper_index = N - 1
        transition_lower_index = 0

        # --- Lower Surface (Laminar) ---
        integral = 0.0
        for i in range(mid - 1, -1, -1):
            ds = s[i+1] - s[i]
            integral += 0.5 * (U_e[i]**5 + U_e[i+1]**5) * ds
            theta2 = 0.45 * nu * integral / U_e[i]**6 if U_e[i] > 1e-6 else 0
            theta[i] = np.sqrt(max(0, theta2))
            
            dUeds = (U_e[i+1] - U_e[i]) / ds if ds > 0 else 0
            duds[i] = dUeds
            lam = theta2 * dUeds / nu
            
            z = 0.25 - lam
            H[i] = 2.0 + 4.14*z - 83.5*z**2 + 854*z**3 - 3337*z**4 + 4576*z**5
            delta_star[i] = H[i] * theta[i]

            Re_s = U_e[i] * abs(s[i]) / nu
            Re_theta = U_e[i] * theta[i] / nu
            
            if (Re_s > 0 and Re_theta > 2.8 * Re_s**0.4) or lam < -0.09:
                transition_lower_index = i
                break
        
        # --- Upper Surface (Laminar) ---
        integral = 0.0
        for i in range(mid + 1, N):
            ds = s[i] - s[i-1]
            integral += 0.5 * (U_e[i]**5 + U_e[i-1]**5) * ds
            theta2 = 0.45 * nu * integral / U_e[i]**6 if U_e[i] > 1e-6 else 0
            theta[i] = np.sqrt(max(0, theta2))
            
            dUeds = (U_e[i] - U_e[i-1]) / ds if ds > 0 else 0
            duds[i] = dUeds
            lam = theta2 * dUeds / nu
            
            z = 0.25 - lam
            H[i] = 2.0 + 4.14*z - 83.5*z**2 + 854*z**3 - 3337*z**4 + 4576*z**5
            delta_star[i] = H[i] * theta[i]

            Re_s = U_e[i] * s[i] / nu
            Re_theta = U_e[i] * theta[i] / nu
            
            if (Re_s > 0 and Re_theta > 2.8 * Re_s**0.4) or lam < -0.09:
                transition_upper_index = i
                break
        
        # --- Turbulent Boundary Layer (Lower) ---
        for i in range(transition_lower_index - 1, -1, -1):
            ds = s[i+1] - s[i]
            Ue = U_e[i]
            dUeds = (U_e[i+1] - U_e[i]) / ds
            y0 = np.array([theta[i + 1], H[i + 1]])
            thetaH_new = rk4_step(head_method_rhs, s[i + 1], y0, -ds, Ue, dUeds, nu)
            theta[i], H[i] = thetaH_new
            delta_star[i] = H[i] * theta[i]

        # --- Turbulent Boundary Layer (Upper) ---
        for i in range(transition_upper_index + 1, N):
            ds = s[i] - s[i-1]
            Ue = U_e[i]
            dUeds = (U_e[i] - U_e[i-1]) / ds
            y0 = np.array([theta[i-1], H[i-1]])
            thetaH_new = rk4_step(head_method_rhs, s[i-1], y0, ds, Ue, dUeds, nu)
            theta[i], H[i] = thetaH_new
            delta_star[i] = H[i] * theta[i]

        return theta, delta_star, H, transition_upper_index, transition_lower_index, duds