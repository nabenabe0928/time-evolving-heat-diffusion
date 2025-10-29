from __future__ import annotations

import json
import os

import numpy as np
from tqdm import tqdm


os.makedirs("results", exist_ok=True)
MU0 = 4.0 * np.pi * 1.0e-7
ETA = 2.0e-8
THERMAL_CONDUCTIVITY_TABLE = {"Au": 316.0, "Hg": 8.6, "Cu": 399, "Fe": 77}
THERMAL_CAPACITY_TABLE = {"Au": 129.0 * 19320.0, "Hg": 139.0 * 13546.0, "Cu": 385.0 * 8960.0, "Fe": 444.0 * 7874.0}


class HeatDiffusionSimulator:
    def __init__(
        self,
        *,
        element: str,
        nx: int = 50,
        ny: int = 50,
        X: float = 0.1,
        Y: float = 0.1,
        Z: float = 0.01,
        dt: float = 1e-2,
        flux_x_bound: float = 0.0,
        flux_y_bound: float = 0.0,
    ) -> None:
        thermal_conductivity = THERMAL_CONDUCTIVITY_TABLE[element]
        thermal_capacity = THERMAL_CAPACITY_TABLE[element]
        self._element = element
        self._nx = nx
        self._ny = ny
        self._dx = float(X / (nx + 1))
        self._dy = float(Y / (ny + 1))
        self._dt = dt
        self._nabla_x, self._nabla_y = self._build_gradient_operators()
        self._joule_heat_coef = ETA * dt / thermal_capacity

        surface_potential = self._get_surface_potential(Z)
        thermal_diffusivity_factor = thermal_conductivity * dt / thermal_capacity
        delta = self._build_delta_operator()
        self._potential_step, self._potential_mass = self._get_potential_matrices(Z)
        self._temp_step = np.linalg.inv(np.eye(self._nx * self._ny, dtype=float) - thermal_diffusivity_factor * delta)
        self._temp_boundary_source = self._get_temp_boundary_source(
            thermal_diffusivity_factor, flux_x_bound, flux_y_bound
        )

    def _get_temp_boundary_source(
        self, thermal_diffusivity_factor: float, flux_x_bound: float, flux_y_bound: float
    ) -> np.ndarray:
        """Compute the temperature flux source from boundaries (Neumann boundary term)."""
        boundary_flux = np.zeros(self._nx * self._ny, dtype=float)
        boundary_flux[:: self._ny] -= flux_x_bound / self._dx**2
        boundary_flux[self._ny - 1 :: self._ny] -= flux_x_bound / self._dx**2
        boundary_flux[: self._nx] -= flux_y_bound / self._dy**2
        boundary_flux[-self._nx :] -= flux_y_bound / self._dy**2
        return -thermal_diffusivity_factor * boundary_flux

    def _get_surface_potential(self, Z: float) -> np.ndarray:
        """
        Compute the dense surface potential matrix. It returns a (nx*ny, nx*ny) dense array.
        """
        dx, dy, nx, ny = self._dx, self._dy, self._nx, self._ny
        xs = np.tile(np.arange(nx), ny) * dx
        ys = np.tile(np.arange(ny)[:, None], (1, nx)).ravel() * dy
        coef = MU0 * Z * dx * dy / 4.0 / np.pi
        surface_potential = coef / np.sqrt((xs - xs[:, None]) ** 2 + (ys - ys[:, None]) ** 2 + (Z / 2.0) ** 2) ** 3
        return surface_potential

    def _get_potential_matrices(self, Z: float) -> tuple[np.ndarray, np.ndarray]:
        surface_potential = self._get_surface_potential(Z)
        delta = self._build_delta_operator()
        mu0_I = MU0 * np.eye(self._nx * self._ny, dtype=float)
        potential_mass = (mu0_I - surface_potential) / self._dt
        potential_step = np.linalg.inv(potential_mass - ETA * delta)
        return potential_step, potential_mass

    def _build_delta_operator(self) -> np.ndarray:
        """
        Build Laplacian (finite difference) operator on nx*ny nodes.

        Returns:
            delta: (nx*ny, nx*ny) Laplacian operator matrix.
        """
        dx, dy, nx, ny = self._dx, self._dy, self._nx, self._ny
        delta = -2.0 * (1 / dx**2 + 1 / dy**2) * np.eye(nx * ny)
        inds = np.arange(nx * ny)
        delta[inds[1:], inds[1:] - 1] = 1.0 / dx**2
        delta[inds[:-1], inds[:-1] + 1] = 1.0 / dx**2
        delta[inds[nx::nx], inds[nx::nx] - 1] = 0.0
        delta[inds[nx - 1 : -1 : nx], inds[nx - 1 : -1 : nx] + 1] = 0.0
        delta[inds[:-nx], inds[:-nx] + nx] = 1.0 / dy**2
        delta[inds[nx:], inds[nx:] - nx] = 1.0 / dy**2
        return delta

    def _build_gradient_operators(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Build discrete gradient operators on nx*ny nodes.

        Returns:
            (nabla_x, nabla_y): (nx*ny, nx*ny) gradient operator matrices.
        """
        dx, dy, nx, ny = self._dx, self._dy, self._nx, self._ny
        nabla_x = np.zeros((nx * ny, nx * ny), dtype=float)
        nabla_y = np.zeros((nx * ny, nx * ny), dtype=float)
        inds = np.arange(nx * ny)
        nabla_x[inds[1:], inds[1:] - 1] = -0.5 / dx
        nabla_x[inds[:-1], inds[:-1] + 1] = 0.5 / dx
        nabla_x[inds[nx::nx], inds[nx::nx] - 1] = 0.0
        nabla_x[inds[nx - 1 : -1 : nx], inds[nx - 1 : -1 : nx] + 1] = 0.0
        nabla_y[inds[:-nx], inds[:-nx] + nx] = 0.5 / dy
        nabla_y[inds[nx:], inds[nx:] - nx] = -0.5 / dy
        return nabla_x, nabla_y

    def start(self, freq: float = 1.0, n_steps: int = 500) -> None:
        omega = 2 * np.pi * freq
        potential = np.zeros(self._nx * self._ny, dtype=float)
        temps = np.zeros((n_steps, self._nx * self._ny), dtype=float)

        for t in tqdm(range(n_steps)):
            potential = self._potential_step @ (self._potential_mass @ potential - omega * np.sin(omega * t * self._dt))
            pot_dy = self._nabla_y @ potential
            pot_dx = -self._nabla_x @ potential
            joule_heat_squared = pot_dx * pot_dx + pot_dy * pot_dy
            prev = max(0, t - 1)
            temps[t] = self._temp_step @ (
                temps[prev] + self._temp_boundary_source + self._joule_heat_coef * joule_heat_squared
            )

        with open(f"results/temperature_{self._element}.json", "w") as f:
            json.dump(temps.tolist(), f)


if __name__ == "__main__":
    HeatDiffusionSimulator(element="Au").start()
