import numpy as np
from scipy.integrate import solve_ivp
from src.fields.body import Body
from src.fields.freestream import Freestream

class BoundaryLayerSolver:
    def __init__(self,
                 laminar_methods: str = "thwaites",
                 transition_method: str = "michel",
                 turbulent_methods: str = "green"  # or "head"
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
        Compute θ, δ*, H, and transition index for both upper and lower surfaces using Thwaites’ method.        """
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
            dHds = (F1 - H1 * (theta * dUeds / Ue + dthetads)) / (theta * dH1dH)  # Avoid division by zero
            return np.array([dthetads, dHds])

        def green_method_rhs(s, thetaHF, Ue, dUeds, nu):
            theta, H, F = thetaHF
            Re_theta = Ue * theta / nu
            Cf0 = 0.01013 / (np.log10(Re_theta) - 1.02) - 0.00075
            H0 = 1 / (1 - 6.55 * np.sqrt(Cf0 / 2))
            Cf = Cf0 * (0.9 / (H / H0 - 0.4) - 0.5)
            
            dHdH1 = -(H - 1)**2 / (1.72 + 0.02 * (H - 1)**3)
            H1 = 3.15 + 1.72 / (H - 1) - 0.01 * (H - 1)**2
            
            delta = theta * (H1 + H)
            
            thetapUtdUedxEQ = 1.25 / H * (Cf / 2 - ((H - 1) / (6.432 * H))**2)
            FEQ = H1 * (Cf / 2 - (H+1) * thetapUtdUedxEQ)
            
            if F is None:
                F = FEQ
            
            dthetads = Cf / 2 - theta / Ue * (2 + H) * dUeds
            dHds = 1 / theta * dHdH1 * (F - H1 * (theta * dUeds / Ue + dthetads))
            dFds = (F**2 + 0.02 * F + 0.2667 * Cf0) / (F + 0.01) * (
                2.8 / (theta * (H1 + H)) * ((0.32 * Cf0 + 0.024 * FEQ + 1.2 * FEQ**2)**(0.5) - (0.32 * Cf0 + 0.024 * F + 1.2 * F**2)**(0.5))
                + 1 / (theta + 1e-6) * thetapUtdUedxEQ - 1 / (Ue + 1e-6) * dUeds
            )
            
            return np.array([dthetads, dHds, dFds])
            
            
            
        assert body.stagnation_index is not None
        assert body.U_e is not None
        assert body.XB is not None
        assert body.YB is not None

        N = 2 * body.N
        mid = body.stagnation_index
        U_e = body.U_e
        
        XB, YB = body.XB, body.YB
        nu = freestream.nu
        mu = freestream.viscosity

        # Change coordinates to arc length along the body
        s = np.zeros(N)
        panel_lengths = np.sqrt((XB[1:] - XB[:-1])**2 + (YB[1:] - YB[:-1])**2)

        s[body.N - 1] = -panel_lengths[body.N - 1] / 2
        for i in range(body.N - 1, -1, -1):
            s[i] = s[i+1] - 0.5 * (panel_lengths[i] + panel_lengths[i+1])

        s[body.N] = panel_lengths[body.N] / 2
        for i in range(body.N + 1, N):
            s[i] = s[i-1] + 0.5 * (panel_lengths[i-1] + panel_lengths[i])

        epsilon = 1e-6

        if U_e[mid] > epsilon:
            if U_e[mid] * U_e[mid+1] < 0:
                stagnation_s = s[mid] + (-U_e[mid] / (U_e[mid+1] - U_e[mid])) * (s[mid+1] - s[mid])
            elif U_e[mid] * U_e[mid-1] < 0:
                stagnation_s = s[mid-1] + (-U_e[mid-1] / (U_e[mid] - U_e[mid-1])) * (s[mid] - s[mid-1])
        else:
            stagnation_s = s[mid]
        
        # U_e[mid] = 0
        # stagnation_s = s[mid]

        s = s - stagnation_s  # Shift s to have stagnation point at 0
        
        # Separate the surface based on location of the stagnation point
        upper_s = s[s >= 0]
        lower_s = np.abs(s[s < 0][::-1])  # Reverse the lower surface coordinates
        
        # Separate the upper and lower surfaces' velocities
        upper_U_e = U_e[s >= 0]
        lower_U_e = np.abs(U_e[s < 0][::-1])  # Reverse the lower surface velocities

        # Get the number of points in each surface
        upper_N = len(upper_s)
        lower_N = len(lower_s)
        
        # Initialize arrays for the upper and lower surfaces
        upper_theta = np.zeros(upper_N)
        upper_lambda = np.zeros(upper_N)
        
        lower_theta = np.zeros(lower_N)
        lower_lambda = np.zeros(lower_N)
        
        # --- Lower Surface (Laminar) ---
        # Compute momentum thickness (theta) and corresponding parameter (lambda)
        lower_integral = lower_U_e[0]**5 / 2 * (lower_s[0]) 
        
        lower_theta_squared = 0.45 * nu / lower_U_e[0]**6 * lower_integral
        lower_theta[0] = np.sqrt(lower_theta_squared)
        
        lower_lambda[0] = lower_theta_squared / nu * (lower_U_e[0] - 0) / lower_s[0]
        
        for i in range(1, lower_N-1):
            lower_integral += 0.5 * (lower_U_e[i]**5 + lower_U_e[i-1]**5) * (lower_s[i] - lower_s[i-1])
            
            lower_theta_squared = 0.45 * nu / lower_U_e[i]**6 * lower_integral
            lower_theta[i] = np.sqrt(lower_theta_squared)
            
            lower_lambda[i] = lower_theta_squared / nu * 0.5 * ((lower_U_e[i] - lower_U_e[i-1]) / (lower_s[i] - lower_s[i-1]) + (lower_U_e[i+1] - lower_U_e[i]) / (lower_s[i+1] - lower_s[i]))
            
        lower_integral += 0.5 * (lower_U_e[-1]**5 + lower_U_e[-2]**5) * (lower_s[-1] - lower_s[-2])
        
        lower_theta_squared = 0.45 * nu / lower_U_e[-1]**6 * lower_integral
        lower_theta[-1] = np.sqrt(lower_theta_squared)
        
        lower_lambda[-1] = lower_theta_squared / nu * (lower_U_e[-1] - lower_U_e[-2]) / (lower_s[-1] - lower_s[-2])
        
        # Compute z, S(lambda), and H(lambda)
        lower_z = 0.25 - lower_lambda
        lower_S = (lower_lambda + 0.09)**0.62
        lower_H = 2.0 + 4.14 * lower_z - 83.5 * lower_z**2 + 854 * lower_z**3 - 3337 * lower_z**4 + 4576 * lower_z**5
        
        lower_delta_star = lower_H * lower_theta
        lower_tau_w = mu * lower_U_e / lower_theta * lower_S
        lower_cf = lower_tau_w / (0.5 * freestream.density * freestream.U_inf**2)
        
        
        # --- Upper Surface (Laminar) ---
        # Compute momentum thickness (theta) and corresponding parameter (lambda)
        upper_integral = upper_U_e[0]**5 / 2 * (upper_s[0])
        
        upper_theta_squared = 0.45 * nu / upper_U_e[0]**6 * upper_integral
        upper_theta[0] = np.sqrt(upper_theta_squared)
        
        upper_lambda[0] = upper_theta_squared / nu * (upper_U_e[0] - 0) / upper_s[0]
        
        for i in range(1, upper_N-1):
            upper_integral += 0.5 * (upper_U_e[i]**5 + upper_U_e[i-1]**5) * (upper_s[i] - upper_s[i-1])
            
            upper_theta_squared = 0.45 * nu / upper_U_e[i]**6 * upper_integral
            upper_theta[i] = np.sqrt(upper_theta_squared)
            
            upper_lambda[i] = upper_theta_squared / nu * 0.5 * ((upper_U_e[i] - upper_U_e[i-1]) / (upper_s[i] - upper_s[i-1]) + (upper_U_e[i+1] - upper_U_e[i]) / (upper_s[i+1] - upper_s[i]))
            
        upper_integral += 0.5 * (upper_U_e[-1]**5 + upper_U_e[-2]**5) * (upper_s[-1] - upper_s[-2])
        
        upper_theta_squared = 0.45 * nu / upper_U_e[-1]**6 * upper_integral
        upper_theta[-1] = np.sqrt(upper_theta_squared)
        
        upper_lambda[-1] = upper_theta_squared / nu * (upper_U_e[-1] - upper_U_e[-2]) / (upper_s[-1] - upper_s[-2])
        
        # Compute z, S(lambda), and H(lambda)
        upper_z = 0.25 - upper_lambda
        upper_S = (upper_lambda + 0.09)**0.62
        upper_H = 2.0 + 4.14 * upper_z - 83.5 * upper_z**2 + 854 * upper_z**3 - 3337 * upper_z**4 + 4576 * upper_z**5
        
        upper_delta_star = upper_H * upper_theta
        upper_tau_w = mu * upper_U_e / upper_theta * upper_S
        upper_cf = upper_tau_w / (0.5 * freestream.density * freestream.U_inf**2)
        
        # Check for transition points with Michel's criterion
        lower_re_s = lower_U_e * lower_s / nu
        lower_re_theta = lower_U_e * lower_theta / nu
        
        upper_re_s = upper_U_e * upper_s / nu
        upper_re_theta = upper_U_e * upper_theta / nu
        
        lower_transition_index = lower_N - 1
        upper_transition_index = upper_N
        
        for i in range(lower_N):
            if (lower_re_s[i] > 0 and lower_re_theta[i] >= 1.174 * (1 + 22400 / lower_re_s[i]) * lower_re_s[i]**0.46) :
                lower_transition_index = i
                break
            
        for i in range(upper_N):
            if (upper_re_s[i] > 0 and upper_re_theta[i] >= 1.174 * (1 + 22400 / upper_re_s[i]) * upper_re_s[i]**0.46) :
                upper_transition_index = i
                break
        
        
        lower_laminar_separation_index = lower_N - 1
        for i in range(lower_N):
            if (lower_lambda[i] < -0.09):
                lower_laminar_separation_index = i
                break
            
        upper_laminar_separation_index = upper_N - 1
        for i in range(upper_N):
            if (upper_lambda[i] < -0.09):
                upper_laminar_separation_index = i
                break
    
        # Starts the turbulent boundary layer calculations
        
        # --- Turbulent Boundary Layer (Lower Surface) --
        # Copy the arrays for turbulent calculations
        lower_theta_turbulent = np.copy(lower_theta)
        lower_H_turbulent = np.copy(lower_H)
        lower_delta_star_turbulent = np.copy(lower_delta_star)
        lower_tau_w_turbulent = np.copy(lower_tau_w)
        lower_cf_turbulent = np.copy(lower_cf)
        
        # At the transition point (lower_transition_index), impose initial conditions based on AerocalPak4
        lower_theta_turbulent[lower_transition_index] = lower_theta[lower_transition_index] # No change in theta
        lower_H_turbulent[lower_transition_index] = lower_H[lower_transition_index] - 1.2
        
        if self.turbulent_methods == "head":
            for i in range(lower_transition_index + 1, lower_N):
                y0 = np.array([lower_theta_turbulent[i-1], lower_H_turbulent[i-1]])
                h = lower_s[i] - lower_s[i-1]
                
                K_1 = head_method_rhs(lower_s[i-1], y0, lower_U_e[i-1], (lower_U_e[i] - lower_U_e[i-1]) / (lower_s[i] - lower_s[i-1]), nu)
                K_2 = head_method_rhs(lower_s[i-1] + h / 2,
                                    y0 + h / 2 * K_1,
                                    lower_U_e[i-1] + h / 2 * (lower_U_e[i] - lower_U_e[i-1]) / (lower_s[i] - lower_s[i-1]),
                                    (lower_U_e[i] - lower_U_e[i-1]) / (lower_s[i] - lower_s[i-1]),
                                    nu)
                K_3 = head_method_rhs(lower_s[i-1] + h / 2,
                                    y0 + h / 2 * K_2,
                                    lower_U_e[i-1] + h / 2 * (lower_U_e[i] - lower_U_e[i-1]) / (lower_s[i] - lower_s[i-1]),
                                    (lower_U_e[i] - lower_U_e[i-1]) / (lower_s[i] - lower_s[i-1]),
                                    nu)
                K_4 = head_method_rhs(lower_s[i],
                                    y0 + h * K_3,
                                    lower_U_e[i],
                                    (lower_U_e[i] - lower_U_e[i-1]) / (lower_s[i] - lower_s[i-1]),
                                    nu)
                
                lower_theta_turbulent[i] = lower_theta_turbulent[i-1] + (h / 6) * (K_1[0] + 2 * K_2[0] + 2 * K_3[0] + K_4[0])
                lower_H_turbulent[i] = lower_H_turbulent[i-1] + (h / 6) * (K_1[1] + 2 * K_2[1] + 2 * K_3[1] + K_4[1])
                lower_delta_star_turbulent[i] = lower_H_turbulent[i] * lower_theta_turbulent[i]
                
                cf = 0.246 * (lower_re_theta[i]**(-0.268)) * 10**(-0.678 * lower_H_turbulent[i])
                lower_cf_turbulent[i] = cf
                lower_tau_w_turbulent[i] = cf * 0.5 * freestream.density * lower_U_e[i]**2
        elif self.turbulent_methods == "green":
            lower_F_turbulent = np.zeros(lower_N)
            lower_delta = np.copy(lower_delta_star)
                        
            for i in range(lower_transition_index + 1, lower_N):
                if i == lower_transition_index + 1:
                    # Initialize F at the transition point
                    lower_F_turbulent[i-1] = None
                
                y0 = np.array([lower_theta_turbulent[i-1], lower_H_turbulent[i-1], lower_F_turbulent[i-1]])
                h = lower_s[i] - lower_s[i-1]
                
                K_1 = green_method_rhs(lower_s[i-1], y0, lower_U_e[i-1], (lower_U_e[i] - lower_U_e[i-1]) / (lower_s[i] - lower_s[i-1]), nu)
                K_2 = green_method_rhs(lower_s[i-1] + h / 2,
                                    y0 + h / 2 * K_1,
                                    lower_U_e[i-1] + h / 2 * (lower_U_e[i] - lower_U_e[i-1]) / (lower_s[i] - lower_s[i-1]),
                                    (lower_U_e[i] - lower_U_e[i-1]) / (lower_s[i] - lower_s[i-1]),
                                    nu)
                K_3 = green_method_rhs(lower_s[i-1] + h / 2,
                                    y0 + h / 2 * K_2,
                                    lower_U_e[i-1] + h / 2 * (lower_U_e[i] - lower_U_e[i-1]) / (lower_s[i] - lower_s[i-1]),
                                    (lower_U_e[i] - lower_U_e[i-1]) / (lower_s[i] - lower_s[i-1]),
                                    nu)
                K_4 = green_method_rhs(lower_s[i],
                                    y0 + h * K_3,
                                    lower_U_e[i],
                                    (lower_U_e[i] - lower_U_e[i-1]) / (lower_s[i] - lower_s[i-1]),
                                    nu)
                
                lower_theta_turbulent[i] = lower_theta_turbulent[i-1] + (h / 6) * (K_1[0] + 2 * K_2[0] + 2 * K_3[0] + K_4[0])
                lower_H_turbulent[i] = lower_H_turbulent[i-1] + (h / 6) * (K_1[1] + 2 * K_2[1] + 2 * K_3[1] + K_4[1])
                lower_F_turbulent[i] = lower_F_turbulent[i-1] + (h / 6) * (K_1[2] + 2 * K_2[2] + 2 * K_3[2] + K_4[2])
                
                lower_F_turbulent[i] = np.max([lower_F_turbulent[i], -0.009])  # Ensure F is not negative
                         
                Re_theta = lower_U_e[i] * lower_theta_turbulent[i] / nu
                cf0 = 0.01013 / (np.log10(Re_theta) - 1.02) - 0.00075
                H0 = 1 / (1 - 6.55 * np.sqrt(cf0 / 2))
                cf = cf0 * (0.9 / (lower_H_turbulent[i] / H0 - 0.4) - 0.5)
                
                H1 = 3.15 + 1.72 / (lower_H_turbulent[i] - 1) - 0.01 * (lower_H_turbulent[i] - 1)**2
                lower_delta[i] = lower_theta_turbulent[i] * (H1 + lower_H_turbulent[i])
                
                lower_cf_turbulent[i] = cf
                lower_tau_w_turbulent[i] = cf * 0.5 * freestream.density * lower_U_e[i]**2
            
        # --- Turbulent Boundary Layer (Upper Surface) ---
        # Copy the arrays for turbulent calculations
        upper_theta_turbulent = np.copy(upper_theta)
        upper_H_turbulent = np.copy(upper_H)
        upper_delta_star_turbulent = np.copy(upper_delta_star)
        upper_tau_w_turbulent = np.copy(upper_tau_w)
        upper_cf_turbulent = np.copy(upper_cf)

        # At the transition point (upper_transition_index), impose initial conditions based on AerocalPak4
        upper_theta_turbulent[upper_transition_index] = upper_theta[upper_transition_index] # No change in theta
        upper_H_turbulent[upper_transition_index] = upper_H[upper_transition_index] - 1.2

        if self.turbulent_methods == "head":
            for i in range(upper_transition_index + 1, upper_N):
                y0 = np.array([upper_theta_turbulent[i-1], upper_H_turbulent[i-1]])
                h = upper_s[i] - upper_s[i-1]

                K_1 = head_method_rhs(upper_s[i-1], y0, upper_U_e[i-1], (upper_U_e[i] - upper_U_e[i-1]) / (upper_s[i] - upper_s[i-1]), nu)
                K_2 = head_method_rhs(upper_s[i-1] + h / 2,
                                    y0 + h / 2 * K_1,
                                    upper_U_e[i-1] + h / 2 * (upper_U_e[i] - upper_U_e[i-1]) / (upper_s[i] - upper_s[i-1]),
                                    (upper_U_e[i] - upper_U_e[i-1]) / (upper_s[i] - upper_s[i-1]),
                                    nu)
                K_3 = head_method_rhs(upper_s[i-1] + h / 2,
                                    y0 + h / 2 * K_2,
                                    upper_U_e[i-1] + h / 2 * (upper_U_e[i] - upper_U_e[i-1]) / (upper_s[i] - upper_s[i-1]),
                                    (upper_U_e[i] - upper_U_e[i-1]) / (upper_s[i] - upper_s[i-1]),
                                    nu)
                K_4 = head_method_rhs(upper_s[i],
                                    y0 + h * K_3,
                                    upper_U_e[i],
                                    (upper_U_e[i] - upper_U_e[i-1]) / (upper_s[i] - upper_s[i-1]),
                                    nu)

                upper_theta_turbulent[i] = upper_theta_turbulent[i-1] + (h / 6) * (K_1[0] + 2 * K_2[0] + 2 * K_3[0] + K_4[0])
                upper_H_turbulent[i] = upper_H_turbulent[i-1] + (h / 6) * (K_1[1] + 2 * K_2[1] + 2 * K_3[1] + K_4[1])
                upper_delta_star_turbulent[i] = upper_H_turbulent[i] * upper_theta_turbulent[i]

                cf = 0.246 * (upper_re_theta[i]**(-0.268)) * 10**(-0.678 * upper_H_turbulent[i])
                upper_tau_w_turbulent[i] = cf * 0.5 * freestream.density * upper_U_e[i]**2
                upper_cf_turbulent[i] = cf
        elif self.turbulent_methods == "green":
            upper_F_turbulent = np.zeros(upper_N)
            upper_delta = np.copy(upper_delta_star)

            for i in range(upper_transition_index + 1, upper_N):
                if i == upper_transition_index + 1:
                    # Initialize F at the transition point
                    upper_F_turbulent[i-1] = None

                y0 = np.array([upper_theta_turbulent[i-1], upper_H_turbulent[i-1], upper_F_turbulent[i-1]])
                h = upper_s[i] - upper_s[i-1]

                K_1 = green_method_rhs(upper_s[i-1], y0, upper_U_e[i-1], (upper_U_e[i] - upper_U_e[i-1]) / (upper_s[i] - upper_s[i-1]), nu)
                K_2 = green_method_rhs(upper_s[i-1] + h / 2,
                                    y0 + h / 2 * K_1,
                                    upper_U_e[i-1] + h / 2 * (upper_U_e[i] - upper_U_e[i-1]) / (upper_s[i] - upper_s[i-1]),
                                    (upper_U_e[i] - upper_U_e[i-1]) / (upper_s[i] - upper_s[i-1]),
                                    nu)
                K_3 = green_method_rhs(upper_s[i-1] + h / 2,
                                    y0 + h / 2 * K_2,
                                    upper_U_e[i-1] + h / 2 * (upper_U_e[i] - upper_U_e[i-1]) / (upper_s[i] - upper_s[i-1]),
                                    (upper_U_e[i] - upper_U_e[i-1]) / (upper_s[i] - upper_s[i-1]),
                                    nu)
                K_4 = green_method_rhs(upper_s[i],
                                    y0 + h * K_3,
                                    upper_U_e[i],
                                    (upper_U_e[i] - upper_U_e[i-1]) / (upper_s[i] - upper_s[i-1]),
                                    nu)

                upper_theta_turbulent[i] = upper_theta_turbulent[i-1] + (h / 6) * (K_1[0] + 2 * K_2[0] + 2 * K_3[0] + K_4[0])
                upper_H_turbulent[i] = upper_H_turbulent[i-1] + (h / 6) * (K_1[1] + 2 * K_2[1] + 2 * K_3[1] + K_4[1])
                upper_F_turbulent[i] = upper_F_turbulent[i-1] + (h / 6) * (K_1[2] + 2 * K_2[2] + 2 * K_3[2] + K_4[2])
                
                upper_F_turbulent[i] = np.max([upper_F_turbulent[i], -0.009])  # Ensure F is not negative
                
                Re_theta = upper_U_e[i] * upper_theta_turbulent[i] / nu
                cf0 = 0.01013 / (np.log10(Re_theta) - 1.02) - 0.00075
                H0 = 1 / (1 - 6.55 * np.sqrt(cf0 / 2))
                
                cf = cf0 * (0.9 / (upper_H_turbulent[i] / H0 - 0.4) - 0.5)
                
                H1 = 3.15 + 1.72 / (upper_H_turbulent[i] - 1) - 0.01 * (upper_H_turbulent[i] - 1)**2
                upper_delta[i] = upper_theta_turbulent[i] * (H1 + upper_H_turbulent[i])
                
                upper_tau_w_turbulent[i] = cf * 0.5 * freestream.density * upper_U_e[i]**2
                upper_cf_turbulent[i] = cf
        
        # Combine the results for both surfaces
        full_theta = np.zeros(N)
        full_theta[:lower_N] = lower_theta_turbulent[::-1]  # Reverse for lower surface
        full_theta[lower_N:] = upper_theta_turbulent
        
        full_delta_star = np.zeros(N)
        full_delta_star[:lower_N] = lower_delta_star_turbulent[::-1]
        full_delta_star[lower_N:] = upper_delta_star_turbulent
        
        full_H = np.zeros(N)
        full_H[:lower_N] = lower_H_turbulent[::-1]
        full_H[lower_N:] = upper_H_turbulent
        
        full_tau_w = np.zeros(N)
        full_tau_w[:lower_N] = lower_tau_w_turbulent[::-1]
        full_tau_w[lower_N:] = upper_tau_w_turbulent
        
        # Transition indices
        full_transition_index_lower = lower_N - lower_transition_index
        full_transition_index_upper = upper_transition_index + lower_N
        
        full_laminar_separation_index_lower = lower_N - lower_laminar_separation_index
        full_laminar_separation_index_upper = upper_laminar_separation_index + lower_N
        
        full_cf = np.zeros(N)
        full_cf[:lower_N] = lower_cf_turbulent[::-1]
        full_cf[lower_N:] = upper_cf_turbulent

        # theta = np.zeros(N)
        # delta_star = np.zeros(N)        
        # H = np.zeros(N)
        
        # duds = np.zeros(N)
        
        # transition_upper_index = N - 1
        # transition_lower_index = 0

        # # --- Lower Surface (Laminar) ---
        # integral = 0.0
        # for i in range(mid - 1, -1, -1):
        #     ds = S[i+1] - S[i]
        #     integral += 0.5 * (U_e[i]**5 + U_e[i+1]**5) * ds
        #     theta2 = 0.45 * nu * integral / U_e[i]**6 if U_e[i] > 1e-6 else 0
        #     theta[i] = np.sqrt(max(0, theta2))
            
        #     dUeds = (U_e[i+1] - U_e[i]) / ds if ds > 0 else 0
        #     duds[i] = dUeds
        #     lam = theta2 * dUeds / nu
            
        #     z = 0.25 - lam
        #     H[i] = 2.0 + 4.14*z - 83.5*z**2 + 854*z**3 - 3337*z**4 + 4576*z**5
        #     delta_star[i] = H[i] * theta[i]

        #     Re_s = U_e[i] * abs(s[i]) / nu
        #     Re_theta = U_e[i] * theta[i] / nu
            
        #     if (Re_s > 0 and Re_theta > 2.8 * Re_s**0.4) or lam < -0.09:
        #         transition_lower_index = i
        #         break
        
        # # --- Upper Surface (Laminar) ---
        # integral = 0.0
        # for i in range(mid + 1, N):
        #     ds = S[i] - S[i-1]
        #     integral += 0.5 * (U_e[i]**5 + U_e[i-1]**5) * ds
        #     theta2 = 0.45 * nu * integral / U_e[i]**6 if U_e[i] > 1e-6 else 0
        #     theta[i] = np.sqrt(max(0, theta2))
            
        #     dUeds = (U_e[i] - U_e[i-1]) / ds if ds > 0 else 0
        #     duds[i] = dUeds
        #     lam = theta2 * dUeds / nu
            
        #     z = 0.25 - lam
        #     H[i] = 2.0 + 4.14*z - 83.5*z**2 + 854*z**3 - 3337*z**4 + 4576*z**5
        #     delta_star[i] = H[i] * theta[i]

        #     Re_s = U_e[i] * S[i] / nu
        #     Re_theta = U_e[i] * theta[i] / nu
            
        #     if (Re_s > 0 and Re_theta > 2.8 * Re_s**0.4) or lam < -0.09:
        #         transition_upper_index = i
        #         break
        
        # # --- Turbulent Boundary Layer (Lower) ---
        # for i in range(transition_lower_index - 1, -1, -1):
        #     ds = S[i+1] - S[i]
        #     Ue = U_e[i]
        #     dUeds = (U_e[i+1] - U_e[i]) / ds
        #     y0 = np.array([theta[i + 1], H[i + 1]])
        #     thetaH_new = rk4_step(head_method_rhs, S[i + 1], y0, -ds, Ue, dUeds, nu)
        #     theta[i], H[i] = thetaH_new
        #     delta_star[i] = H[i] * theta[i]

        # # --- Turbulent Boundary Layer (Upper) ---
        # for i in range(transition_upper_index + 1, N):
        #     ds = S[i] - S[i-1]
        #     Ue = U_e[i]
        #     dUeds = (U_e[i] - U_e[i-1]) / ds
        #     y0 = np.array([theta[i-1], H[i-1]])
        #     thetaH_new = rk4_step(head_method_rhs, S[i-1], y0, ds, Ue, dUeds, nu)
        #     theta[i], H[i] = thetaH_new
        #     delta_star[i] = H[i] * theta[i]

        return full_theta, full_delta_star, full_H, full_cf, full_transition_index_lower, full_transition_index_upper, full_laminar_separation_index_lower, full_laminar_separation_index_upper