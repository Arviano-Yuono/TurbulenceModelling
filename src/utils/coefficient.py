import numpy as np

def compute_cl_from_cp(X: np.ndarray, Cp: np.ndarray) -> np.float64:
    """
    Compute the lift coefficient from the pressure coefficient.

    Args:
        X (np.ndarray): x-coordinates.
        Cp (np.ndarray): pressure coefficient.

    Returns:
        float: lift coefficient.
    """
    M = len(X)
    Cpl = Cp[:M // 2][::-1]
    Cpu = Cp[M // 2:]
    dCp = Cpl - Cpu
    dX = X[M // 2:]
    Cl = np.float64(np.trapezoid(dCp, dX))
    return Cl
