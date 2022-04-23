import numpy as np
import matplotlib.pyplot as plt

L = 0.1
n = 10
T0 = 0
T1 = 60
T2 = 30
dx = L/n
Ka = 0.0001
dt = 0.1

plt.ion()

x = np.linspace(dx/2, L - dx/2, n)

T = np.ones(n)*T0
dTdt = np.empty(n)

t = np.arange(0, 50, dt)

for j in range(1, len(t)):
    plt.clf()
    T[-1] = T2
    T[0]  = T1 
    for i in range(1, n-1):
        dTdt[i] = Ka*(-(T[i]-T[i-1])/dx**2+(T[i+1]-T[i])/dx**2)
    T = T + dTdt*dt
    plt.figure(1)
    plt.plot(x, T)
    plt.axis([0, L, 0, 100])
    plt.show()
    plt.pause(0.01)