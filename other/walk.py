import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import math

nsteps = 5000

position = np.zeros((nsteps, 2))
current = np.zeros((nsteps, 2))
x = np.zeros((nsteps, 1))
y = np.zeros((nsteps, 1))
t = np.zeros((nsteps, 1))
Amp = np.zeros((nsteps, 1))
R = 30
count = 0
w = 12
h = 12
region = 4
A = 1  # A
Vr = -65  # mV
tau = 10  # ms


# Random walk and counting
for i in range(0, nsteps - 1):
    theta = np.random.uniform(0, 2 * np.pi)
    dx = np.math.cos(theta)
    dy = np.math.sin(theta)
    position[i + 1][0] = position[i][0] + dx
    position[i + 1][1] = position[i][1] + dy
    if position[i + 1][0] > w or position[i + 1][0] < -w:
        position[i + 1][0] = position[i][0] - dx
        if position[i + 1][1] > h or position[i + 1][1] < -h:
            position[i + 1][1] = position[i][1] - dy
        else:
            position[i + 1][1] = position[i][1] + dy
    else:
        position[i + 1][0] = position[i][0] + dx
        if position[i + 1][1] > h or position[i + 1][1] < -h:
            position[i + 1][1] = position[i][1] - dy
        else:
            position[i + 1][1] = position[i][1] + dy

    if abs(position[i][0]) >= region and abs(position[i][1]) >= region:
        if abs(position[i + 1][0]) < region and abs(position[i + 1][1]) < region:
            count += 1
    # Integrate and fire model
    if abs(position[i][0]) < region and abs(position[i][1]) < region:
        current[i][0] = i
        current[i][1] = A
    else:
        current[i][0] = i
        current[i][1] = 0

    x[i] = position[i][0]
    y[i] = position[i][1]
    t[i] = current[i][0]
    Amp[i] = current[i][1]

t = np.linspace(0, nsteps, 5000)


# def Vmodel(V, t):
#     dVdt = -(V - Vr) + R * Amp[t] / tau
#     return dVdt
#
#
# V0 = Vr
# V = odeint(Vmodel, V0, t)
#
# plt.plot(t, I(t))
# plt.show()

# plt.plot(x,y)
# plt.xlabel('X-axis')
# plt.ylabel('Y-axis')
# plt.title("2D random walk")
# plt.show()

plt.plot(t,Amp)
plt.xlabel('time')
plt.ylabel('Amp')
plt.title('t-A')
plt.show()

print(count)

print(current)
