import numpy as np
from scipy.integrate import solve_ivp
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
        Compute θ, δ*, H, and transition index for both upper and lower surfaces using Thwaites' method.
        """
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

        self.theta = np.zeros(N)
        self.delta_star = np.zeros(N)
        self.H = np.zeros(N)
        self.duds = np.gradient(U_e, s)
        self.transition_upper_index = N - 1
        self.transition_lower_index = 0
        self.Re_theta = np.zeros(N)

        # --- Lower Surface (Laminar) ---
        integral = 0.0
        for i in range(mid - 1, -1, -1):
            ds = s[i+1] - s[i]
            integral += 0.5 * (U_e[i]**5 + U_e[i+1]**5) * ds
            theta2 = 0.45 * nu * integral / U_e[i]**6 if U_e[i] > 1e-6 else 0
            self.theta[i] = np.sqrt(max(0, theta2))
            
            dUeds = self.duds[i]
            lam = theta2 * dUeds / nu
            
            z = 0.25 - lam
            self.H[i] = 2.0 + 4.14*z - 83.5*z**2 + 854*z**3 - 3337*z**4 + 4576*z**5
            self.delta_star[i] = self.H[i] * self.theta[i]

            Re_s = U_e[i] * abs(s[i]) / nu
            self.Re_theta[i] = U_e[i] * self.theta[i] / nu
            
            if (Re_s > 0 and self.Re_theta[i] > 2.8 * Re_s**0.4) or lam < -0.09:
                if self.transition_lower_index == 0:
                    self.transition_lower_index = i
                
        # --- Upper Surface (Laminar) ---
        integral = 0.0
        for i in range(mid + 1, N):
            ds = s[i] - s[i-1]
            integral += 0.5 * (U_e[i]**5 + U_e[i-1]**5) * ds
            theta2 = 0.45 * nu * integral / U_e[i]**6 if U_e[i] > 1e-6 else 0
            self.theta[i] = np.sqrt(max(0, theta2))
            
            dUeds = self.duds[i]
            lam = theta2 * dUeds / nu
            
            z = 0.25 - lam
            self.H[i] = 2.0 + 4.14*z - 83.5*z**2 + 854*z**3 - 3337*z**4 + 4576*z**5
            self.delta_star[i] = self.H[i] * self.theta[i]

            Re_s = U_e[i] * s[i] / nu
            self.Re_theta[i] = U_e[i] * self.theta[i] / nu
            
            if (Re_s > 0 and self.Re_theta[i] > 2.8 * Re_s**0.4) or lam < -0.09:
                if self.transition_upper_index == N - 1:
                    self.transition_upper_index = i
        
        self.solve_green(X, U_e, nu, s)
        
        return self.theta, self.delta_star, self.H, self.transition_upper_index, self.transition_lower_index, self.duds
    
    def initialize_green(self, i):
        Re_theta_i = self.Re_theta[i]
        if Re_theta_i <= 0:
            return 0,0,0,0,0

        Cf_0 = 0.01013 / (np.log10(Re_theta_i) - 1.02) - 0.00075
        H_0 = 1 / (1 - 6.55 * np.sqrt(Cf_0 / 2))
        H_i = self.H[i]
        dHdH1 = - (H_i - 1)**2 / (1.72 + 0.02 * (H_i - 1)**3)
        H1 = 3.15 + 1.72 / (H_i - 1) - 0.01 * (H_i - 1)**2
        delta = self.theta[i] * (H1 + H_i)

        return Cf_0, H_0, dHdH1, H1, delta
    
    def green_rhs(self, x, y, Ue, dUedx, nu):
        theta, H, F = y

        # Step 1: Skin friction Cf_0 and H0
        Re_theta = Ue * theta / nu
        if Re_theta <= 0: return [0,0,0]
        
        Cf_0 = 0.01013 / (np.log10(Re_theta) - 1.02) - 0.00075
        if Cf_0 <= 0: return [0,0,0]

        H0 = 1 / (1 - 6.55 * np.sqrt(Cf_0 / 2))
        Cf = Cf_0 * ((H / H0 - 0.4) ** -0.5) * 0.9

        # Step 2: H1, dH/dH1
        H1 = 3.15 + 1.72 / (H - 1) - 0.01 * (H - 1)**2
        dHdH1 = - (H - 1)**2 / (1.72 + 0.02 * (H - 1)**3)

        # Step 3: dθ/dx
        dthetadx = Cf / 2 - (H + 2) * theta / Ue * dUedx

        # Step 4: dH/dx
        dHdx = (1 / theta) * dHdH1 * (F - H1) * ((theta / Ue) * (-dUedx) + dthetadx)

        # Step 5: δ = θ(H + H1)
        delta = theta * (H + H1)

        # Step 6: Equilibrium terms
        theta_over_Ue_dUedx_EQ = (1.25 / H) * ((Cf / 2) - ((H - 1) / (6.432 * H)))**2
        FEQ = H1 * ((Cf / 2 - (H + 1)) * theta_over_Ue_dUedx_EQ)

        # Step 7: dF/dx (RHS of eq)
        F_safe = max(F, -0.0089)
        term_in_sqrt_1 = 0.32 * Cf_0 + 0.024 * FEQ + 1.2 * FEQ**2
        term_in_sqrt_2 = 0.32 * Cf_0 + 0.024 * F_safe + 1.2 * F_safe**2
        if term_in_sqrt_1 < 0 or term_in_sqrt_2 < 0: return [0,0,0]

        dFdx_mult = (2.8 / (theta * (H1 + H))) * (np.sqrt(term_in_sqrt_1) - np.sqrt(term_in_sqrt_2))
        dFdx = (F**2 + 0.02*F + 0.2667*Cf_0) / (F + 0.01) * dFdx_mult + (1 / theta) * (theta_over_Ue_dUedx_EQ - (theta / Ue * dUedx))
        
        # F is not allowed to fall below -0.009
        if F + dFdx < -0.009:
            dFdx = -0.009 - F

        return [dthetadx, dHdx, dFdx]

    def solve_green(self, X, Ue, nu, s):
        N = len(X)
        dUeds = np.gradient(Ue, s)
        
        F_arr = np.zeros(N)
        F_arr[self.transition_upper_index] = 0.0
        F_arr[self.transition_lower_index] = 0.0

        # March on upper surface
        for i in range(self.transition_upper_index, N - 2):
            y0 = [self.theta[i], self.H[i], F_arr[i]]
            def rhs(x, y):
                return self.green_rhs(x, y, Ue[i], dUeds[i], nu)
            
            sol = solve_ivp(rhs, [s[i], s[i+1]], y0, method='RK45')
            if sol.status == 0:
                self.theta[i+1], self.H[i+1], F_arr[i+1] = sol.y[:, -1]
                self.delta_star[i+1] = self.H[i+1] * self.theta[i+1]
            else:
                self.transition_upper_index = i
                break

        # March on lower surface
        for i in range(self.transition_lower_index, 0, -1):
            y0 = [self.theta[i], self.H[i], F_arr[i]]
            def rhs(x, y):
                return self.green_rhs(x, y, Ue[i], dUeds[i], nu)
            
            sol = solve_ivp(rhs, [s[i], s[i-1]], y0, method='RK45')
            if sol.status == 0:
                self.theta[i-1], self.H[i-1], F_arr[i-1] = sol.y[:, -1]
                self.delta_star[i-1] = self.H[i-1] * self.theta[i-1]
            else:
                self.transition_lower_index = i
                break