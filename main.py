from __future__ import annotations

import json
import os

import numpy as np
from tqdm import tqdm


os.makedirs("results", exist_ok=True)
MU0 = 4.0 * np.pi * 1.0e-7
ETA = 2.0e-8


class HeatDiffusionSimulator:
    def __init__(
        self,
        *,
        element: str,
        freq: float = 1.0,
        n_steps: int = 20,
        W: int = 5,
        H: int = 5,
        X: float = 0.1,
        Y: float = 0.1,
        Z: float = 0.01,
        dt=0.1,
        flux_x_bound: float = 0.0,
        flux_y_bound: float = 0.0,
    ) -> None:
        thermal_conductivity = {"Au": 316.0, "Hg": 8.6, "Cu": 399, "Fe": 77}[element]
        thermal_capacity = {
            "Au": 129.0 * 19320.0,
            "Hg": 139.0 * 13546.0,
            "Cu": 385.0 * 8960.0,
            "Fe": 444.0 * 7874.0,
        }[element]
        self._element = element
        self._freq = freq
        self._n_steps = n_steps
        self._W = W
        self._H = H
        self._dx = float(X / (W + 1))
        self._dy = float(Y / (H + 1))
        self._Z = Z
        self._dt = dt
        coef1 = thermal_conductivity * dt / thermal_capacity
        self._coef2 = ETA * dt / thermal_capacity

        surface_potential = self._get_surface_potential()
        boundary_flux = self._get_boundary_flux(flux_x_bound, flux_y_bound)
        delta = self._build_delta_operator()
        self._nabla_x, self._nabla_y = self._build_gradient_operators()
        I = np.identity(W * H)
        self._potential_step = np.linalg.inv((MU0 * I - surface_potential) / dt - ETA * delta)
        self._potential_mass = (MU0 * I - surface_potential) / dt
        self._temp_step = np.linalg.inv(I - coef1 * delta)
        self._temp_boundary_source = -coef1 * boundary_flux

    def _get_boundary_flux(self, flux_x_bound: float, flux_y_bound: float) -> np.ndarray:
        """Compute boundary flux source vector (Neumann boundary term)."""
        boundary_flux = np.zeros(self._W * self._H, dtype=float)
        boundary_flux[:: self._H] -= flux_x_bound / self._dx**2
        boundary_flux[self._H - 1 :: self._H] -= flux_x_bound / self._dx**2
        boundary_flux[: self._W] -= flux_y_bound / self._dy**2
        boundary_flux[-self._W :] -= flux_y_bound / self._dy**2
        return boundary_flux

    def _get_surface_potential(self) -> np.ndarray:
        """
        Compute the dense surface potential matrix. It returns a (W*H, W*H) dense array.
        """
        dx, dy, Z, W, H = self._dx, self._dy, self._Z, self._W, self._H
        xs = np.tile(np.arange(W), H) * dx
        ys = np.tile(np.arange(H)[:, None], (1, W)).ravel() * dy
        coef = MU0 * Z * dx * dy / 4.0 / np.pi
        surface_potential = (
            coef / np.sqrt((xs - xs[:, None]) ** 2 + (ys - ys[:, None]) ** 2 + (Z / 2.0) ** 2) ** 3
        )
        return surface_potential

    def _build_delta_operator(self) -> np.ndarray:
        """
        Build Laplacian (finite difference) operator on W*H nodes.

        Returns:
            delta: (W*H, W*H) Laplacian operator matrix.
        """
        dx, dy, W, H = self._dx, self._dy, self._W, self._H
        delta = -2.0 * (1 / dx**2 + 1 / dy**2) * np.eye(W * H)
        inds = np.arange(W * H)
        delta[inds[1:], inds[1:] - 1] = 1.0 / dx**2
        delta[inds[:-1], inds[:-1] + 1] = 1.0 / dx**2
        delta[inds[W::W], inds[W::W] - 1] = 0.0
        delta[inds[W - 1 : -1 : W], inds[W - 1 : -1 : W] + 1] = 0.0
        delta[inds[:-W], inds[:-W] + W] = 1.0 / dy**2
        delta[inds[W:], inds[W:] - W] = 1.0 / dy**2
        return delta

    def _build_gradient_operators(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Build discrete gradient operators on W*H nodes.

        Returns:
            (nabla_x, nabla_y): (W*H, W*H) gradient operator matrices.
        """
        dx, dy, W, H = self._dx, self._dy, self._W, self._H
        nabla_x = np.zeros((W * H, W * H), dtype=float)
        nabla_y = np.zeros((W * H, W * H), dtype=float)
        inds = np.arange(W * H)
        nabla_x[inds[1:], inds[1:] - 1] = -0.5 / dx
        nabla_x[inds[:-1], inds[:-1] + 1] = 0.5 / dx
        nabla_x[inds[W::W], inds[W::W] - 1] = 0.0
        nabla_x[inds[W - 1 : -1 : W], inds[W - 1 : -1 : W] + 1] = 0.0
        nabla_y[inds[:-W], inds[:-W] + W] = 0.5 / dy
        nabla_y[inds[W:], inds[W:] - W] = -0.5 / dy
        return nabla_x, nabla_y

    def main(self) -> None:
        omega = 2 * np.pi * self._freq
        potential = np.zeros(self._W * self._H, dtype=float)
        temps = np.zeros((self._n_steps, self._W * self._H), dtype=float)

        for t in tqdm(range(self._n_steps)):
            potential = self._potential_step @ (
                self._potential_mass @ potential - omega * np.sin(omega * t * self._dt)
            )
            pot_dy = self._nabla_y @ potential
            pot_dx = -self._nabla_x @ potential
            joule_head_squared = pot_dx @ pot_dx + pot_dy @ pot_dy
            prev = max(0, t - 1)
            temps[t] = self._temp_step @ (
                temps[prev] + self._temp_boundary_source + self._coef2 * joule_head_squared
            )

        with open(f"results/temperature_{self._element}.json", "w") as f:
            json.dump(temps.tolist(), f)


if __name__ == "__main__":
    HeatDiffusionSimulator(element="Au").main()
