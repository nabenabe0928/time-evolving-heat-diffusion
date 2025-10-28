import numpy as np
from numba import jit, f8, i8, b1, void, u1
import csv
import matplotlib.pyplot as plt
import matplotlib.animation as animation


@jit(f8[:, :](i8, i8, f8, f8, f8[:, :], f8[:, :]))
def get_psi(W, H, dx, dy, fx_bound, fy_bound):
    psi = np.zeros(W * H)
    dx_inv2 = 1.0 / (dx**2)
    dy_inv2 = 1.0 / (dy**2)
    c = -2.0 * dx_inv2 - 2.0 * dy_inv2

    for x in range(W):
        for y in range(H):
            alpha = y * W + x
            psi[alpha] = 0

            if x == 0:
                psi[alpha] -= dx_inv2 * fx_bound[0][y]
            elif x == W - 1:
                psi[alpha] -= dx_inv2 * fx_bound[1][y]
            if y == 0:
                psi[alpha] -= dy_inv2 * fy_bound[0][x]
            elif y == H - 1:
                psi[alpha] -= dy_inv2 * fy_bound[1][x]

    return psi


@jit(f8[:, :](i8, i8, f8, f8, f8, f8))
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


@jit(f8[:, :](i8, i8, f8, f8))
def get_delta(W, H, dx, dy):
    delta = np.zeros((W * H, W * H))
    dx_inv2 = 1.0 / (dx**2)
    dy_inv2 = 1.0 / (dy**2)
    c = -2.0 * dx_inv2 - 2.0 * dy_inv2

    for i in range(W):
        for j in range(H):
            alpha = j * W + i
            delta[alpha][alpha] = c

            if 0 < i < W - 1:
                delta[alpha][alpha - 1] = dx_inv2
                delta[alpha][alpha + 1] = dx_inv2
            elif i == 0:
                delta[alpha][alpha + 1] = dx_inv2
            elif i == W - 1:
                delta[alpha][alpha - 1] = dx_inv2
            if 0 < j < H - 1:
                delta[alpha][alpha - W] = dy_inv2
                delta[alpha][alpha + W] = dy_inv2
            elif j == 0:
                delta[alpha][alpha + W] = dy_inv2
            elif j == H - 1:
                delta[alpha][alpha - W] = dy_inv2

    return delta


@jit(f8[:, :](i8, i8, f8))
def get_nablax(W, H, dx):
    dx_inv = 1.0 / dx / 2.0
    nablax = np.zeros((W * H, W * H))

    for i in range(W):
        for j in range(H):
            alpha = j * W + i

            if 0 < i < W - 1:
                nablax[alpha][alpha - 1] = -dx_inv
                nablax[alpha][alpha + 1] = dx_inv
            elif i == 0:
                nablax[alpha][alpha + 1] = dx_inv
            elif i == W - 1:
                nablax[alpha][alpha - 1] = -dx_inv

    return nablax


@jit(f8[:, :](i8, i8, f8))
def get_nablay(W, H, dy):
    dy_inv = 1.0 / dy / 2.0
    nablay = np.zeros((W * H, W * H))

    for i in range(W):
        for j in range(H):
            alpha = j * W + i

            if 0 < j < H - 1:
                nablay[alpha][alpha - W] = -dy_inv
                nablay[alpha][alpha + W] = dy_inv
            elif j == 0:
                nablay[alpha][alpha + W] = dy_inv
            elif j == H - 1:
                nablay[alpha][alpha - W] = -dy_inv

    return nablay


@jit(void(f8[:, :], f8[:, :], f8[:, :], f8[:], f8[:, :], f8[:, :], i8, f8, i8, i8, f8, f8, u1))
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
    with open("data/" + name + ".csv", "w", newline="") as f:
        writer = csv.writer(f, delimiter=",", quotechar='"')

        for array in M:
            writer.writerow(array)


if __name__ == "__main__":
    element = ["Au", "Hg", "Cu", "Fe"][0]
    f = 1.0
    itr = 50
    W = 50
    H = 50
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
