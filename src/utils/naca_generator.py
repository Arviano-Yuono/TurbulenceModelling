import numpy as np


def generate_naca4(series: str, N: int, cosine_spacing: bool = True, closed_te: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate NACA 4-digit airfoil coordinates. Generates from trailing edge and clockwise.

    Args:
        series (str): NACA 4-digit series.
        N (int): Number of points.
        cosine_spacing (bool): Whether to use cosine spacing.
        closed_te (bool): Whether to close the trailing edge.

    Returns:
        tuple[np.ndarray, np.ndarray]: x-coordinates and y-coordinates of the airfoil.
    """
    m = int(series[0]) / 100
    p = int(series[1]) / 10
    t = int(series[2:]) / 100

    if cosine_spacing:
        beta = np.linspace(0, np.pi, N + 1)
        x = (1 - np.cos(beta)) / 2
    else:
        x = np.linspace(0, 1, N + 1)

    yt = 5 * t * (
        0.2969 * np.sqrt(x)
        - 0.1260 * x
        - 0.3516 * x**2
        + 0.2843 * x**3
        - (0.1036 if closed_te else 0.1015) * x**4
    )

    yc = np.where(x < p,
                  m / p**2 * (2 * p * x - x**2),
                  m / (1 - p)**2 * ((1 - 2 * p) + 2 * p * x - x**2))

    dyc_dx = np.where(x < p,
                      2 * m / p**2 * (p - x),
                      2 * m / (1 - p)**2 * (p - x))

    theta = np.arctan(dyc_dx)

    xu = x - yt * np.sin(theta)
    yu = yc + yt * np.cos(theta)
    xl = x + yt * np.sin(theta)
    yl = yc - yt * np.cos(theta)

    x_coords = np.concatenate([xl[::-1], xu[1:]])
    y_coords = np.concatenate([yl[::-1], yu[1:]])

    return x_coords, y_coords
