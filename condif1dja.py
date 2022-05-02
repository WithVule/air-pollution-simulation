import numpy as np
import matplotlib.pyplot as plt

n = 10
dx = 1
dt = 1
nu = 1
tmax = 50
xmax = 50

plt.ion()

v = np.zeros(n)
u = np.zeros(n)
t = np.arange(0, tmax, dt)
x = np.linspace(dx/2, xmax - dx/2, n)

for j in range(0, tmax):
    plt.clf()
    for i in range(1, n-1):
        v[i+1] =  v[i] - dt*(v[i]*(v[i] - v[i-1])/dx +
                             -dt*nu*(v[i-1] - 2*v[i] + v[i+1])/dx**2)
        u[i+1] =  u[i] - dt*(u[i]*(u[i] - u[i-1])/dx +
                             -v[i+1]*(u[i-1] - 2*u[i] + u[i+1])/dx**2)
    plt.figure(1)
    plt.plot(x, u)
    plt.axis([0, xmax, 0, 10])
    plt.show()
    plt.pause(0.01)
    
    
