import numpy as np
import matplotlib.pyplot as plt


u = 0.1
C0 = 1
D = 0.001
L = 5
m = 100
N = 200
dx = L/m
x = np.ones(m)
C = np.ones(m)
t = 0
dt = 0.1

a = u*dt/(2*dx)
b = D*dt/(dx*dx)
C[0:m+1] = 0
C[0] = C0
x[0] = 0

for n in range(1, m-1):
    t = t + dt
   
for i in range(1, m-1):
    C[i] = (a+b)*C[i-1]+(1-2*b)*C[i]+(-a+b)*C[i+1]
    x[i] = i*dx
   
    plt.plot(x, C)
    plt.show()
