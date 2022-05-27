import numpy as np
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt

n = 10
tmax = 50
xmax = 50
dt = tmax/n
dx = xmax/n
nu = 0.1


plt.ion()

v = np.zeros((n,n))
u = np.zeros((n,n))
t = np.linspace(0, tmax, n)
x = np.linspace(0, xmax, n)

u[0,:] = u[n-1,:] = u[:,:] = 1
v[0,:] = v[n-1,:] = v[:,:] = 1

u[int((n-1)/4):int((n-1)/2),0] = 2
v[int((n-1)/4):int((n-1)/2),0] = 2

for j in range(0, n-1):
    plt.clf()
    
    for i in range(1, n-1):
        v[i,j+1] =  v[i,j] - dt*(v[i,j]*(v[i,j] - v[i-1,j])/dx +
                             -dt*nu*(v[i-1,j] - 2*v[i,j] + v[i+1,j])/dx**2)
        u[i,j+1] =  u[i,j] - dt*(u[i,j]*(u[i,j] - u[i-1,j])/dx +
                             -dt*nu*(u[i-1,j] - 2*u[i,j] + u[i+1,j])/dx**2)
    
for i in range(0,n):
    x[i] = i*dx
    t[i] = i*dt
    
fig = plt.figure()
ax = plt.axes(projection='3d')
ax.contour3D(x, t, u, 50, cmap='binary')
ax.set_xlabel('x')
ax.set_ylabel('t')
ax.set_zlabel('u')
