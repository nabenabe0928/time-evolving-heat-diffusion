import os

import numpy as np
import csv
import matplotlib.pyplot as plt
import matplotlib.animation as animation


os.makedirs("results", exist_ok=True)


def get_psi(W, H, dx, dy, fx_bound, fy_bound):
    psi = np.zeros(W * H, dtype=float)
    psi[::H] -= fx_bound[0] / dx**2
    psi[H - 1 :: H] -= fx_bound[1] / dx**2
    psi[:W] -= fy_bound[0] / dy**2
    psi[-W:] -= fy_bound[1] / dy**2
    return psi


def get_surface_potential(W, H, dx, dy, d, mu0):
    sur_pot = np.zeros((W * H, W * H))
    coef = mu0 * d * dx * dy / 4.0 / np.pi

    for i1 in range(W):
        for j1 in range(H):
            alpha1 = j1 * W + i1
            r1 = np.array([i1 * dx, j1 * dy, 0])

            for i2 in range(W):
                for j2 in range(H):
                    alpha2 = j2 * W + i2
                    r2 = np.array([i2 * dx, j2 * dy, d / 2.0])

                    sur_pot[alpha1][alpha2] = coef / (np.linalg.norm(r1 - r2) ** 3)

    return sur_pot


def get_delta(W, H, dx, dy):
    delta = -2.0 * (1 / dx**2 + 1 / dy**2) * np.eye(W*H)
    inds = np.arange(W*H)
    delta[inds[1:], inds[1:] - 1] = 1.0 / dx**2
    delta[inds[:-1], inds[:-1] + 1] = 1.0 / dx**2
    delta[inds[W::W], inds[W::W] - 1] = 0.0
    delta[inds[W - 1:-1:W], inds[W - 1:-1:W] + 1] = 0.0
    delta[inds[:-W], inds[:-W] + W] = 1.0 / dy**2
    delta[inds[W:], inds[W:] - W] = 1.0 / dy**2
    return delta


def get_nablax(W, H, dx):
    nablax = np.zeros((W * H, W * H))
    inds = np.arange(W*H)
    nablax[inds[1:], inds[1:] - 1] = -0.5 / dx
    nablax[inds[:-1], inds[:-1] + 1] = 0.5 / dx
    nablax[inds[W::W], inds[W::W] - 1] = 0.0
    nablax[inds[W - 1:-1:W], inds[W - 1:-1:W] + 1] = 0.0
    return nablax


def get_nablay(W, H, dy):
    nablay = np.zeros((W * H, W * H))
    inds = np.arange(W*H)
    nablay[inds[:-W], inds[:-W] + W] = 0.5 / dy
    nablay[inds[W:], inds[W:] - W] = -0.5 / dy
    return nablay


def main(matB1, matB2, matT1, vecT2, nablax, nablay, itr, f, W, H, dt, coef2, element):
    coef3 = 2 * np.pi * f
    cur_pot = np.zeros(W * H)
    T = np.zeros(W * H)
    Ts = []

    for t in range(itr):
        B = coef3 * np.sin(coef3 * t * dt)
        cur_pot = matB1 @ (matB2 @ cur_pot - np.full(W * H, B))
        J2 = np.linalg.norm(nablay @ cur_pot) ** 2 + np.linalg.norm(-nablax @ cur_pot) ** 2
        T = matT1 @ (T + vecT2 + coef2 * J2)
        Ts.append(T[:])
        print(t)

    write_csv(Ts, "thermal_dist_{}".format(element))


def write_csv(M, name):
    with open("results/" + name + ".csv", "w", newline="") as f:
        writer = csv.writer(f, delimiter=",", quotechar='"')

        for array in M:
            writer.writerow(array)


if __name__ == "__main__":
    element = ["Au", "Hg", "Cu", "Fe"][0]
    f = 1.0
    itr = 20
    W = 5
    H = 5
    X = 0.1
    Y = 0.1
    d = 0.01
    eta = 2.0 * 1.0e-8
    mu0 = 4.0 * np.pi * 1.0e-7
    dt = 0.1
    lmd = {"Au": 316.0, "Hg": 8.6, "Cu": 399, "Fe": 77}[element]
    thrm_cap = {
        "Au": 129.0 * 19320.0,
        "Hg": 139.0 * 13546.0,
        "Cu": 385.0 * 8960.0,
        "Fe": 444.0 * 7874.0,
    }[element]
    coef1 = lmd * dt / thrm_cap
    coef2 = eta * dt / thrm_cap
    x, y = np.meshgrid(np.linspace(0, X, W + 2), np.linspace(0, Y, H + 2))

    fx_bound = np.full((2, W), 0)
    fy_bound = np.full((2, H), 0)
    dx = float(X / (W + 1))
    dy = float(Y / (H + 1))

    sur_pot = get_surface_potential(W, H, dx, dy, d, mu0)
    psi = get_psi(W, H, dx, dy, fx_bound, fy_bound)
    delta = get_delta(W, H, dx, dy)
    I = np.identity(W * H)
    nablax = get_nablax(W, H, dx)
    nablay = get_nablay(W, H, dy)

    matB1 = np.linalg.inv((mu0 * I - sur_pot) / dt - eta * delta)
    matB2 = (mu0 * I - sur_pot) / dt
    matT1 = np.linalg.inv(I - coef1 * delta)
    vecT2 = -coef1 * psi

    main(matB1, matB2, matT1, vecT2, nablax, nablay, itr, f, W, H, dt, coef2, element)
