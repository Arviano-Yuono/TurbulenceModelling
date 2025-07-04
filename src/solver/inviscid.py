import numpy as np

from src.fields import Body, Freestream

class VortexPanelMethod:
    @staticmethod
    def solve(body: Body, freestream: Freestream) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Solve the vortex panel method for an inviscid flow and update the edge velocity of the body.

        Args:
            body (Body): Body object.
            freestream (Freestream): Freestream object.

        Returns:
            tuple[np.ndarray, np.ndarray, np.ndarray]: x-coordinates, dimensionless velocity, and pressure coefficient.
        """
        XB, YB = body.get_body_points()
        AoA_rad = freestream.AoA_rad

        N = body.N
        M = 2 * N
        X, Y = 0.5 * (XB[:-1] + XB[1:]), 0.5 * (YB[:-1] + YB[1:])
        S = np.sqrt((XB[1:] - XB[:-1])**2 + (YB[1:] - YB[:-1])**2)
        Phi = np.arctan2((YB[1:] - YB[:-1]), (XB[1:] - XB[:-1]))
        RHS = np.sin(Phi - AoA_rad)

        CN1, CN2, CT1, CT2 = np.zeros((M, M)), np.zeros((M, M)), np.zeros((M, M)), np.zeros((M, M))

        for i in range(M):
            for j in range(M):
                if i == j:
                    CN1[i, j] = -1
                    CN2[i, j] = 1
                    CT1[i, j] = CT2[i, j] = 0.5 * np.pi
                else:
                    A = -(X[i] - XB[j]) * np.cos(Phi[j]) - (Y[i] - YB[j]) * np.sin(Phi[j])
                    B = (X[i] - XB[j])**2 + (Y[i] - YB[j])**2
                    C = np.sin(Phi[i] - Phi[j])
                    D = np.cos(Phi[i] - Phi[j])
                    E = (X[i] - XB[j]) * np.sin(Phi[j]) - (Y[i] - YB[j]) * np.cos(Phi[j])
                    F = np.log(1 + S[j] * (S[j] + 2 * A) / B)
                    G = np.arctan2(E * S[j], B + A * S[j])

                    P = (X[i] - XB[j]) * np.sin(Phi[i] - 2 * Phi[j]) + (Y[i] - YB[j]) * np.cos(Phi[i] - 2 * Phi[j])
                    Q = (X[i] - XB[j]) * np.cos(Phi[i] - 2 * Phi[j]) - (Y[i] - YB[j]) * np.sin(Phi[i] - 2 * Phi[j])

                    CN2[i, j] = D + 0.5 * Q * F / S[j] - (A * C + D * E) * G / S[j]
                    CN1[i, j] = 0.5 * D * F + C * G - CN2[i, j]
                    CT2[i, j] = C + 0.5 * P * F / S[j] + (A * D - C * E) * G / S[j]
                    CT1[i, j] = 0.5 * C * F - D * G - CT2[i, j]

        AN = np.zeros((M + 1, M + 1))
        for i in range(M):
            AN[i, 0], AN[i, -1] = CN1[i, 0], CN2[i, -1]
            for j in range(1, M):
                AN[i, j] = CN1[i, j] + CN2[i, j - 1]
        AN[-1, 0], AN[-1, -1] = 1, 1

        RHS = np.append(RHS, 0)  # Kutta condition
        Gamma = np.linalg.solve(AN, RHS)
        AT = np.zeros((M, M + 1))
        for i in range(M):
            AT[i, 0], AT[i, -1] = CT1[i, 0], CT2[i, -1]
            for j in range(1, M):
                AT[i, j] = CT1[i, j] + CT2[i, j - 1]
        VpVinf = np.cos(Phi - AoA_rad) + AT @ Gamma
        Cp = 1 - VpVinf**2

        return X, VpVinf, Cp

def vortex_panel_method(XB: np.ndarray, YB: np.ndarray, AoA_rad: float, N: int = 200) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Solve the vortex panel method for an inviscid flow.

    Args:
        N (int): Number of panels.
        XB (np.ndarray): x-coordinates of the airfoil.
        YB (np.ndarray): y-coordinates of the airfoil.
        AoA_rad (float): Angle of attack in radians.

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray]: x-coordinates, dimensionless velocity, and pressure coefficient.
    """
    M = 2 * N
    X, Y = 0.5 * (XB[:-1] + XB[1:]), 0.5 * (YB[:-1] + YB[1:])
    S = np.sqrt((XB[1:] - XB[:-1])**2 + (YB[1:] - YB[:-1])**2)
    Phi = np.arctan2((YB[1:] - YB[:-1]), (XB[1:] - XB[:-1]))
    RHS = np.sin(Phi - AoA_rad)

    CN1, CN2, CT1, CT2 = np.zeros((M, M)), np.zeros((M, M)), np.zeros((M, M)), np.zeros((M, M))

    for i in range(M):
        for j in range(M):
            if i == j:
                CN1[i, j] = -1
                CN2[i, j] = 1
                CT1[i, j] = CT2[i, j] = 0.5 * np.pi
            else:
                A = -(X[i] - XB[j]) * np.cos(Phi[j]) - (Y[i] - YB[j]) * np.sin(Phi[j])
                B = (X[i] - XB[j])**2 + (Y[i] - YB[j])**2
                C = np.sin(Phi[i] - Phi[j])
                D = np.cos(Phi[i] - Phi[j])
                E = (X[i] - XB[j]) * np.sin(Phi[j]) - (Y[i] - YB[j]) * np.cos(Phi[j])
                F = np.log(1 + S[j] * (S[j] + 2 * A) / B)
                G = np.arctan2(E * S[j], B + A * S[j])

                P = (X[i] - XB[j]) * np.sin(Phi[i] - 2 * Phi[j]) + (Y[i] - YB[j]) * np.cos(Phi[i] - 2 * Phi[j])
                Q = (X[i] - XB[j]) * np.cos(Phi[i] - 2 * Phi[j]) - (Y[i] - YB[j]) * np.sin(Phi[i] - 2 * Phi[j])

                CN2[i, j] = D + 0.5 * Q * F / S[j] - (A * C + D * E) * G / S[j]
                CN1[i, j] = 0.5 * D * F + C * G - CN2[i, j]
                CT2[i, j] = C + 0.5 * P * F / S[j] + (A * D - C * E) * G / S[j]
                CT1[i, j] = 0.5 * C * F - D * G - CT2[i, j]

    AN = np.zeros((M + 1, M + 1))
    for i in range(M):
        AN[i, 0], AN[i, -1] = CN1[i, 0], CN2[i, -1]
        for j in range(1, M):
            AN[i, j] = CN1[i, j] + CN2[i, j - 1]
    AN[-1, 0], AN[-1, -1] = 1, 1

    RHS = np.append(RHS, 0)  # Kutta condition
    Gamma = np.linalg.solve(AN, RHS)
    AT = np.zeros((M, M + 1))
    for i in range(M):
        AT[i, 0], AT[i, -1] = CT1[i, 0], CT2[i, -1]
        for j in range(1, M):
            AT[i, j] = CT1[i, j] + CT2[i, j - 1]
    VpVinf = np.cos(Phi - AoA_rad) + AT @ Gamma
    Cp = 1 - VpVinf**2

    return X, VpVinf, Cp
