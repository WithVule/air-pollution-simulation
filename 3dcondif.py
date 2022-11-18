def con_dif(L, d, dt, Vx0, Vy0, Vz0, gas, T, R):
    import numpy as np
    import matplotlib.pyplot as plt
    import math as m

    fig = plt.figure()

    dx = L / d
    dy = L / d
    dz = L / d

    # konstante
    g = 9.81
    Msv = 0.0289
    T0 = 288.16
    R0 = 8.314
    A = 1.859 * 10 ** (-3)
    colint = 1
    dcolv = 710 * 10 ** (-2)

    gasovi = {
        'CO2': [660 * 10 ** (-2), 44],
        "CO": [752 * 10 ** (-2), 28],
        "NH3": [520 * 10 ** (-2), 17],
        "NO": [634 * 10 ** (-2), 30],
        "SO2": [720 * 10 ** (-2), 64]
    }

    tgas = gasovi[gas]
    dcol = tgas[0]
    M = tgas[1]

    dcol12 = (dcol + dcolv) / 2

    plt.ion()

    t = np.arange(0, 50, dt)
    C = np.zeros((d,) * 3)
    dCdt = np.zeros((d,) * 3)
    Vx = np.zeros((d,) * 3)
    Vy = np.zeros((d,) * 3)
    Vz = np.zeros((d,) * 3)
    Vxp = np.zeros((d,) * 3)
    Vyp = np.zeros((d,) * 3)
    Vzp = np.zeros((d,) * 3)
    alfa = np.zeros((d,) * 3)

    x = np.linspace(dx / 2, L - dx / 2, d)
    y = np.linspace(dy / 2, L - dy / 2, d)
    z = np.linspace(dz / 2, L - dz / 2, d)
    X, Y, Z = np.meshgrid(x, y, z)

    colors = np.random.standard_normal(len(x))

    C[:, :, :] = 0
    dCdt[:, :, :] = 0
    Vx[:, :, :] = Vy[:, :, :] = Vz[:, :, :] = 0.0001

    C[8, 8, 8] = R
    Vx[1:5, 1:5, 1:5] = Vx0
    Vy[1:5, 1:5, 1:5] = Vy0
    Vz[1:5, 1:5, 1:5] = Vz0

    for _ in range(1, len(t)):
        plt.clf()
        Vxp = Vx.copy()
        Vyp = Vy.copy()
        Vzp = Vz.copy()
        # print(Vx, Vy, Vz)
        # C[5, 5, 5] = C0 + (m-1)*R
        for i in range(d - 1):
            for j in range(d - 1):
                for k in range(d - 1):

                    p = m.exp((-g * Msv * k) / T0 * R0)

                    D = ((A * T ** (3 / 2)) / (p * dcol12 ** 2 * colint)) * m.sqrt(1 / M + 1 / (Msv * 1000))
                    print(D)

                    Vx[i, j, k] = Vxp[i, j, k] - dt * (((Vxp[i, j, k] / dx) * (Vxp[i, j, k] - Vxp[i - 1, j, k]) + (
                                Vyp[i, j, k] / dy) * (Vxp[i, j, k] - Vxp[i, j - 1, k]) + Vzp[i, j, k] / dx) * (
                                                                   Vxp[i, j, k] - Vxp[i - 1, j, k]))
                    Vy[i, j, k] = Vyp[i, j, k] - dt * (((Vxp[i, j, k] / dx) * (Vyp[i, j, k] - Vyp[i - 1, j, k]) + (
                                Vyp[i, j, k] / dy) * (Vyp[i, j, k] - Vyp[i, j - 1, k]) + Vzp[i, j, k] / dx) * (
                                                                   Vyp[i, j, k] - Vyp[i - 1, j, k]))
                    Vz[i, j, k] = Vzp[i, j, k] - dt * (((Vzp[i, j, k] / dx) * (Vzp[i, j, k] - Vzp[i - 1, j, k]) + (
                                Vyp[i, j, k] / dy) * (Vzp[i, j, k] - Vzp[i, j - 1, k]) + Vzp[i, j, k] / dx) * (
                                                                   Vzp[i, j, k] - Vzp[i - 1, j, k]))

                    dCdt[i, j, k] = D * ((C[i + 1, j, k] - 2 * C[i, j, k] + C[i - 1, j, k]) / dx ** 2 + (
                                C[i, j + 1, k] - 2 * C[i, j, k] + C[i, j - 1, k]) / dy ** 2 + (
                                                     C[i, j, k + 1] - 2 * C[i, j, k] + C[i, j, k - 1]) / dz ** 2) - (
                                                Vx[i, j, k] * ((C[i, j, k] - C[i - 1, j, k]) / dx) + Vy[i, j, k] * (
                                                    (C[i, j, k] - C[i, j - 1, k]) / dy) + Vz[i, j, k] * (
                                                            (C[i, j, k] - C[i, j, k - 1]) / dz))

                    C = C + dCdt * dt

                    alfa[i, j, k] = 0.5 if C[i, j, k] > 0.0001 else 0
        # print(Vx)

        ax = plt.axes(projection="3d")
        plot = ax.scatter3D(X, Y, Z, c=C, alpha=0.5, marker='s', cmap=plt.hot())
        fig.colorbar(plot)
        plt.show()
        plt.pause(0.1)


con_dif(100, 10, 0.1, -0.005, -0.005, -0.005, 'CO2', 293, 2)
