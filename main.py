from __future__ import annotations

import json
import os

import numpy as np
from tqdm import tqdm


os.makedirs("results", exist_ok=True)
MU0 = 4.0 * np.pi * 1.0e-7
ETA = 2.0e-8


def get_boundary_flux(
    W: int, H: int, dx: float, dy: float, flux_x_bound: np.ndarray, flux_y_bound: np.ndarray
) -> np.ndarray:
    """Compute boundary flux source vector (Neumann boundary term)."""
    boundary_flux = np.zeros(W * H, dtype=float)
    boundary_flux[::H] -= flux_x_bound[0] / dx**2
    boundary_flux[H - 1 :: H] -= flux_x_bound[1] / dx**2
    boundary_flux[:W] -= flux_y_bound[0] / dy**2
    boundary_flux[-W:] -= flux_y_bound[1] / dy**2
    return boundary_flux


def get_surface_potential(W: int, H: int, dx: float, dy: float, plate_height: float) -> np.ndarray:
    """
    Compute the dense surface potential matrix. It returns a (W*H, W*H) dense array.
    """
    xs = np.tile(np.arange(W), H) * dx
    ys = np.tile(np.arange(H)[:, np.newaxis], (1, W)).ravel() * dy
    coef = MU0 * plate_height * dx * dy / 4.0 / np.pi
    surface_potential = coef / np.sqrt((xs - xs[:, None])**2 + (ys - ys[:, None])**2 + (plate_height / 2.0)**2) ** 3
    return surface_potential


def build_delta_operator(W: int, H: int, dx: float, dy: float) -> np.ndarray:
    """
    Build Laplacian (finite difference) operator on W*H nodes.

    Args:
        W: Number of nodes in x direction.
        H: Number of nodes in y direction.
        dx: Grid spacing in x direction.
        dy: Grid spacing in y direction.
    
    Returns:
        delta: (W*H, W*H) Laplacian operator matrix.
    """
    delta = -2.0 * (1 / dx**2 + 1 / dy**2) * np.eye(W*H)
    inds = np.arange(W*H)
    delta[inds[1:], inds[1:] - 1] = 1.0 / dx**2
    delta[inds[:-1], inds[:-1] + 1] = 1.0 / dx**2
    delta[inds[W::W], inds[W::W] - 1] = 0.0
    delta[inds[W - 1:-1:W], inds[W - 1:-1:W] + 1] = 0.0
    delta[inds[:-W], inds[:-W] + W] = 1.0 / dy**2
    delta[inds[W:], inds[W:] - W] = 1.0 / dy**2
    return delta


def build_gradient_operators(W: int, H: int, dx: float, dy: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Build discrete gradient operators on W*H nodes.

    Args:
        W: Number of nodes in x direction.
        H: Number of nodes in y direction.
        dx: Grid spacing in x direction.
        dy: Grid spacing in y direction.
    
    Returns:
        (nabla_x, nabla_y): (W*H, W*H) gradient operator matrices.
    """
    nabla_x = np.zeros((W * H, W * H), dtype=float)
    nabla_y = np.zeros((W * H, W * H), dtype=float)
    inds = np.arange(W*H)
    nabla_x[inds[1:], inds[1:] - 1] = -0.5 / dx
    nabla_x[inds[:-1], inds[:-1] + 1] = 0.5 / dx
    nabla_x[inds[W::W], inds[W::W] - 1] = 0.0
    nabla_x[inds[W - 1:-1:W], inds[W - 1:-1:W] + 1] = 0.0
    nabla_y[inds[:-W], inds[:-W] + W] = 0.5 / dy
    nabla_y[inds[W:], inds[W:] - W] = -0.5 / dy
    return nabla_x, nabla_y


def main(
    potential_step: np.ndarray,
    potential_mass: np.ndarray,
    temp_step: np.ndarray,
    temp_boundary_source: np.ndarray,
    nabla_x: np.ndarray,
    nabla_y: np.ndarray,
    n_steps: int,
    freq: float,
    dt: float,
    coef2: float,
    element: str,
) -> None:
    omega = 2 * np.pi * freq
    potential = np.zeros(nabla_x.shape[0], dtype=float)
    temps = np.zeros((n_steps, nabla_x.shape[0]), dtype=float)

    for t in tqdm(range(n_steps)):
        potential = potential_step @ (potential_mass @ potential - omega * np.sin(omega * t * dt))
        pot_dy = nabla_y @ potential
        pot_dx = -nabla_x @ potential
        joule_head_squared = pot_dx @ pot_dx + pot_dy @ pot_dy
        temps[t] = temp_step @ (temps[max(0, t - 1)] + temp_boundary_source + coef2 * joule_head_squared)

    with open(f"results/temperature_{element}.json", "w") as f:
        json.dump(temps.tolist(), f)


if __name__ == "__main__":
    element = ["Au", "Hg", "Cu", "Fe"][0]
    freq = 1.0
    n_steps = 20
    W = 5
    H = 5
    X = 0.1
    Y = 0.1
    plate_height = 0.01
    dt = 0.1
    thermal_conductivity = {"Au": 316.0, "Hg": 8.6, "Cu": 399, "Fe": 77}[element]
    thermal_capacity = {
        "Au": 129.0 * 19320.0, "Hg": 139.0 * 13546.0, "Cu": 385.0 * 8960.0, "Fe": 444.0 * 7874.0
    }[element]
    coef1 = thermal_conductivity * dt / thermal_capacity
    coef2 = ETA * dt / thermal_capacity

    flux_x_bound = np.full((2, W), 0)
    flux_y_bound = np.full((2, H), 0)
    dx = float(X / (W + 1))
    dy = float(Y / (H + 1))

    surface_potential = get_surface_potential(W, H, dx, dy, plate_height)
    boundary_flux = get_boundary_flux(W, H, dx, dy, flux_x_bound, flux_y_bound)
    delta = build_delta_operator(W, H, dx, dy)
    I = np.identity(W * H)
    nabla_x, nabla_y = build_gradient_operators(W, H, dx, dy)
    potential_step = np.linalg.inv((MU0 * I - surface_potential) / dt - ETA * delta)
    potential_mass = (MU0 * I - surface_potential) / dt
    temp_step = np.linalg.inv(I - coef1 * delta)
    temp_boundary_source = -coef1 * boundary_flux

    main(potential_step, potential_mass, temp_step, temp_boundary_source, nabla_x, nabla_y, n_steps, freq, dt, coef2, element)

