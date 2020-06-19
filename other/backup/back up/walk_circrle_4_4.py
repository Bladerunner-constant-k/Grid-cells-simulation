import cvxpy as cvx
from pylab import *
import dccp
from cffi.backend_ctypes import xrange
import numpy as np

# ==============================================================================#
# setting
nsteps = 5000

position = np.zeros((nsteps, 2))
R = 30
w = 12
h = 12
A = 0.005  # A
Vr = -65  # mV
tau = 10  # ms
A_random = np.random.random_sample(nsteps)

current_1 = np.zeros((nsteps, 2))
current_2 = np.zeros((nsteps, 2))
current_3 = np.zeros((nsteps, 2))
current_4 = np.zeros((nsteps, 2))
current_5 = np.zeros((nsteps, 2))
current_6 = np.zeros((nsteps, 2))
current_7 = np.zeros((nsteps, 2))
current_8 = np.zeros((nsteps, 2))
current_9 = np.zeros((nsteps, 2))
current_10 = np.zeros((nsteps, 2))
current_11 = np.zeros((nsteps, 2))
current_12 = np.zeros((nsteps, 2))
current_13 = np.zeros((nsteps, 2))
current_14 = np.zeros((nsteps, 2))
current_15 = np.zeros((nsteps, 2))
current_16 = np.zeros((nsteps, 2))

x_1 = np.zeros((nsteps, 1))
x_2 = np.zeros((nsteps, 1))
x_3 = np.zeros((nsteps, 1))
x_4 = np.zeros((nsteps, 1))
x_5 = np.zeros((nsteps, 1))
x_6 = np.zeros((nsteps, 1))
x_7 = np.zeros((nsteps, 1))
x_8 = np.zeros((nsteps, 1))
x_9 = np.zeros((nsteps, 1))
x_10 = np.zeros((nsteps, 1))
x_11 = np.zeros((nsteps, 1))
x_12 = np.zeros((nsteps, 1))
x_13 = np.zeros((nsteps, 1))
x_14 = np.zeros((nsteps, 1))
x_15 = np.zeros((nsteps, 1))
x_16 = np.zeros((nsteps, 1))

y_1 = np.zeros((nsteps, 1))
y_2 = np.zeros((nsteps, 1))
y_3 = np.zeros((nsteps, 1))
y_4 = np.zeros((nsteps, 1))
y_5 = np.zeros((nsteps, 1))
y_6 = np.zeros((nsteps, 1))
y_7 = np.zeros((nsteps, 1))
y_8 = np.zeros((nsteps, 1))
y_9 = np.zeros((nsteps, 1))
y_10 = np.zeros((nsteps, 1))
y_11 = np.zeros((nsteps, 1))
y_12 = np.zeros((nsteps, 1))
y_13 = np.zeros((nsteps, 1))
y_14 = np.zeros((nsteps, 1))
y_15 = np.zeros((nsteps, 1))
y_16 = np.zeros((nsteps, 1))

t_1 = np.zeros((nsteps, 1))
t_2 = np.zeros((nsteps, 1))
t_3 = np.zeros((nsteps, 1))
t_4 = np.zeros((nsteps, 1))
t_5 = np.zeros((nsteps, 1))
t_6 = np.zeros((nsteps, 1))
t_7 = np.zeros((nsteps, 1))
t_8 = np.zeros((nsteps, 1))
t_9 = np.zeros((nsteps, 1))
t_10 = np.zeros((nsteps, 1))
t_11 = np.zeros((nsteps, 1))
t_12 = np.zeros((nsteps, 1))
t_13 = np.zeros((nsteps, 1))
t_14 = np.zeros((nsteps, 1))
t_15 = np.zeros((nsteps, 1))
t_16 = np.zeros((nsteps, 1))

Amp_1 = np.zeros((nsteps, 1))
Amp_2 = np.zeros((nsteps, 1))
Amp_3 = np.zeros((nsteps, 1))
Amp_4 = np.zeros((nsteps, 1))
Amp_5 = np.zeros((nsteps, 1))
Amp_6 = np.zeros((nsteps, 1))
Amp_7 = np.zeros((nsteps, 1))
Amp_8 = np.zeros((nsteps, 1))
Amp_9 = np.zeros((nsteps, 1))
Amp_10 = np.zeros((nsteps, 1))
Amp_11 = np.zeros((nsteps, 1))
Amp_12 = np.zeros((nsteps, 1))
Amp_13 = np.zeros((nsteps, 1))
Amp_14 = np.zeros((nsteps, 1))
Amp_15 = np.zeros((nsteps, 1))
Amp_16 = np.zeros((nsteps, 1))

# ==============================================================================#

# Random walk and counting
# ==============================================================================#

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

# find the maximum area

if np.amax(position[:, 0]) > np.amax(position[:, 1]):
    length = np.amax(position[:, 0])
else:
    length = np.amax(position[:, 1])

# find firing point

np.random.seed(0)
n = 16
radius = 2 * length / np.sqrt(n * np.pi)
r = [radius for i in range(n)]

c = cvx.Variable((n, 2))
constr = []

for i in range(n - 1):
    for j in range(i + 1, n):
        constr.append(cvx.norm(cvx.vec(c[i, :] - c[j, :]), 2) >= r[i] + r[j])
prob = cvx.Problem(cvx.Minimize(cvx.max(cvx.max(cvx.abs(c), axis=1) + r)), constr)
prob.solve(method='dccp', solver='ECOS', ep=1e-2, max_slack=1e-2)

l = cvx.max(cvx.max(cvx.abs(c), axis=1) + r).value * 2
pi = np.pi
ratio = pi * cvx.sum(cvx.square(r)).value / cvx.square(l).value

plt.figure(figsize=(5, 5))
circ = np.linspace(0, 2 * pi)
x_border = [-l / 2, l / 2, l / 2, -l / 2, -l / 2]
y_border = [-l / 2, -l / 2, l / 2, l / 2, -l / 2]
for i in xrange(n):
    plt.plot(c[i, 0].value + r[i] * np.cos(circ), c[i, 1].value + r[i] * np.sin(circ), 'b')

tag_1 = "Length =" + str(2*length)
tag_2 = "Radius =" + str(radius)
tag = tag_1 + "\n" + tag_2

plt.title(str(tag))
plt.plot(x_border, y_border, 'g')
plt.axes().set_aspect('equal')
plt.xlim([-l / 2, l / 2])
plt.ylim([-l / 2, l / 2])
plt.show()

# arrange the point we have

point_x = np.zeros((n, 1))
point_y = np.zeros((n, 1))

for i in range(n):
    point_x[i] = c.value[i][0]
    point_y[i] = c.value[i][1]

point = np.column_stack((point_x, point_y))
point = sorted(point, key=lambda k: [k[1], k[0]], reverse=True)

# Integrate and fire model

z_1 = 0
z_2 = 0
z_3 = 0
z_4 = 0
z_5 = 0
z_6 = 0
z_7 = 0
z_8 = 0
z_9 = 0
z_10 = 0
z_11 = 0
z_12 = 0
z_13 = 0
z_14 = 0
z_15 = 0
z_16 = 0

firing_position_x_1 = np.zeros((nsteps, 1))
firing_position_y_1 = np.zeros((nsteps, 1))
firing_position_x_2 = np.zeros((nsteps, 1))
firing_position_y_2 = np.zeros((nsteps, 1))
firing_position_x_3 = np.zeros((nsteps, 1))
firing_position_y_3 = np.zeros((nsteps, 1))
firing_position_x_4 = np.zeros((nsteps, 1))
firing_position_y_4 = np.zeros((nsteps, 1))
firing_position_x_5 = np.zeros((nsteps, 1))
firing_position_y_5 = np.zeros((nsteps, 1))
firing_position_x_6 = np.zeros((nsteps, 1))
firing_position_y_6 = np.zeros((nsteps, 1))
firing_position_x_7 = np.zeros((nsteps, 1))
firing_position_y_7 = np.zeros((nsteps, 1))
firing_position_x_8 = np.zeros((nsteps, 1))
firing_position_y_8 = np.zeros((nsteps, 1))
firing_position_x_9 = np.zeros((nsteps, 1))
firing_position_y_9 = np.zeros((nsteps, 1))
firing_position_x_10 = np.zeros((nsteps, 1))
firing_position_y_10 = np.zeros((nsteps, 1))
firing_position_x_11 = np.zeros((nsteps, 1))
firing_position_y_11 = np.zeros((nsteps, 1))
firing_position_x_12 = np.zeros((nsteps, 1))
firing_position_y_12 = np.zeros((nsteps, 1))
firing_position_x_13 = np.zeros((nsteps, 1))
firing_position_y_13 = np.zeros((nsteps, 1))
firing_position_x_14 = np.zeros((nsteps, 1))
firing_position_y_14 = np.zeros((nsteps, 1))
firing_position_x_15 = np.zeros((nsteps, 1))
firing_position_y_15 = np.zeros((nsteps, 1))
firing_position_x_16 = np.zeros((nsteps, 1))
firing_position_y_16 = np.zeros((nsteps, 1))

###  point(1)

for i in range(0, nsteps - 1):

    if np.all(np.sqrt((position[i][0] - point_x[0]) ** 2 + (position[i][1] - point_y[0]) ** 2) < radius):
        if A_random[i] > 0.3:

            current_1[i][0] = i
            current_1[i][1] = A * A_random[i]
            firing_position_x_1[i] = position[i][0]
            firing_position_y_1[i] = position[i][1]
            z_1 += 1
        else:
            current_1[i][0] = i
            current_1[i][1] = 0

    else:
        current_1[i][0] = i
        current_1[i][1] = 0

    x_1[i] = position[i][0]
    y_1[i] = position[i][1]
    t_1[i] = current_1[i][0]
    Amp_1[i] = current_1[i][1]

###  point(2)
for i in range(0, nsteps - 1):

    if np.all(np.sqrt((position[i][0] - point_x[1]) ** 2 + (position[i][1] - point_y[1]) ** 2) < radius):
        if A_random[i] > 0.3:

            current_2[i][0] = i
            current_2[i][1] = A * A_random[i]
            firing_position_x_2[i] = position[i][0]
            firing_position_y_2[i] = position[i][1]
            z_2 += 1
        else:
            current_2[i][0] = i
            current_2[i][1] = 0

    else:
        current_2[i][0] = i
        current_2[i][1] = 0

    x_2[i] = position[i][0]
    y_2[i] = position[i][1]
    t_2[i] = current_2[i][0]
    Amp_2[i] = current_2[i][1]

###  point(3)
for i in range(0, nsteps - 1):

    if np.all(np.sqrt((position[i][0] - point_x[2]) ** 2 + (position[i][1] - point_y[2]) ** 2) < radius):
        if A_random[i] > 0.3:

            current_3[i][0] = i
            current_3[i][1] = A * A_random[i]
            firing_position_x_3[i] = position[i][0]
            firing_position_y_3[i] = position[i][1]
            z_3 += 1
        else:
            current_3[i][0] = i
            current_3[i][1] = 0

    else:
        current_3[i][0] = i
        current_3[i][1] = 0

    x_3[i] = position[i][0]
    y_3[i] = position[i][1]
    t_3[i] = current_3[i][0]
    Amp_3[i] = current_3[i][1]

###  point(4)
for i in range(0, nsteps - 1):

    if np.all(np.sqrt((position[i][0] - point_x[3]) ** 2 + (position[i][1] - point_y[3]) ** 2) < radius):
        if A_random[i] > 0.3:

            current_4[i][0] = i
            current_4[i][1] = A * A_random[i]
            firing_position_x_4[i] = position[i][0]
            firing_position_y_4[i] = position[i][1]
            z_4 += 1
        else:
            current_4[i][0] = i
            current_4[i][1] = 0

    else:
        current_4[i][0] = i
        current_4[i][1] = 0

    x_4[i] = position[i][0]
    y_4[i] = position[i][1]
    t_4[i] = current_4[i][0]
    Amp_4[i] = current_4[i][1]

###  point(5)
for i in range(0, nsteps - 1):

    if np.all(np.sqrt((position[i][0] - point_x[4]) ** 2 + (position[i][1] - point_y[4]) ** 2) < radius):
        if A_random[i] > 0.3:

            current_5[i][0] = i
            current_5[i][1] = A * A_random[i]
            firing_position_x_5[i] = position[i][0]
            firing_position_y_5[i] = position[i][1]
            z_5 += 1
        else:
            current_5[i][0] = i
            current_5[i][1] = 0

    else:
        current_5[i][0] = i
        current_5[i][1] = 0

    x_5[i] = position[i][0]
    y_5[i] = position[i][1]
    t_5[i] = current_5[i][0]
    Amp_5[i] = current_5[i][1]

###  point(6)
for i in range(0, nsteps - 1):

    if np.all(np.sqrt((position[i][0] - point_x[5]) ** 2 + (position[i][1] - point_y[5]) ** 2) < radius):
        if A_random[i] > 0.3:

            current_6[i][0] = i
            current_6[i][1] = A * A_random[i]
            firing_position_x_6[i] = position[i][0]
            firing_position_y_6[i] = position[i][1]
            z_6 += 1
        else:
            current_6[i][0] = i
            current_6[i][1] = 0

    else:
        current_6[i][0] = i
        current_6[i][1] = 0

    x_6[i] = position[i][0]
    y_6[i] = position[i][1]
    t_6[i] = current_6[i][0]
    Amp_6[i] = current_6[i][1]

###  point(7)
for i in range(0, nsteps - 1):

    if np.all(np.sqrt((position[i][0] - point_x[6]) ** 2 + (position[i][1] - point_y[6]) ** 2) < radius):
        if A_random[i] > 0.3:

            current_7[i][0] = i
            current_7[i][1] = A * A_random[i]
            firing_position_x_7[i] = position[i][0]
            firing_position_y_7[i] = position[i][1]
            z_7 += 1
        else:
            current_7[i][0] = i
            current_7[i][1] = 0

    else:
        current_7[i][0] = i
        current_7[i][1] = 0

    x_7[i] = position[i][0]
    y_7[i] = position[i][1]
    t_7[i] = current_7[i][0]
    Amp_7[i] = current_7[i][1]

###  point(8)
for i in range(0, nsteps - 1):

    if np.all(np.sqrt((position[i][0] - point_x[7]) ** 2 + (position[i][1] - point_y[7]) ** 2) < radius):
        if A_random[i] > 0.3:

            current_8[i][0] = i
            current_8[i][1] = A * A_random[i]
            firing_position_x_8[i] = position[i][0]
            firing_position_y_8[i] = position[i][1]
            z_8 += 1
        else:
            current_8[i][0] = i
            current_8[i][1] = 0

    else:
        current_8[i][0] = i
        current_8[i][1] = 0

    x_8[i] = position[i][0]
    y_8[i] = position[i][1]
    t_8[i] = current_8[i][0]
    Amp_8[i] = current_8[i][1]

###  point(9)
for i in range(0, nsteps - 1):

    if np.all(np.sqrt((position[i][0] - point_x[8]) ** 2 + (position[i][1] - point_y[8]) ** 2) < radius):
        if A_random[i] > 0.3:

            current_9[i][0] = i
            current_9[i][1] = A * A_random[i]
            firing_position_x_9[i] = position[i][0]
            firing_position_y_9[i] = position[i][1]
            z_9 += 1
        else:
            current_9[i][0] = i
            current_9[i][1] = 0

    else:
        current_9[i][0] = i
        current_9[i][1] = 0

    x_9[i] = position[i][0]
    y_9[i] = position[i][1]
    t_9[i] = current_9[i][0]
    Amp_9[i] = current_9[i][1]

###  point(10)
for i in range(0, nsteps - 1):

    if np.all(np.sqrt((position[i][0] - point_x[9]) ** 2 + (position[i][1] - point_y[9]) ** 2) < radius):
        if A_random[i] > 0.3:

            current_10[i][0] = i
            current_10[i][1] = A * A_random[i]
            firing_position_x_10[i] = position[i][0]
            firing_position_y_10[i] = position[i][1]
            z_10 += 1
        else:
            current_10[i][0] = i
            current_10[i][1] = 0

    else:
        current_10[i][0] = i
        current_10[i][1] = 0

    x_10[i] = position[i][0]
    y_10[i] = position[i][1]
    t_10[i] = current_10[i][0]
    Amp_10[i] = current_10[i][1]

###  point(11)
for i in range(0, nsteps - 1):

    if np.all(np.sqrt((position[i][0] - point_x[10]) ** 2 + (position[i][1] - point_y[10]) ** 2) < radius):
        if A_random[i] > 0.3:

            current_11[i][0] = i
            current_11[i][1] = A * A_random[i]
            firing_position_x_11[i] = position[i][0]
            firing_position_y_11[i] = position[i][1]
            z_11 += 1
        else:
            current_11[i][0] = i
            current_11[i][1] = 0

    else:
        current_11[i][0] = i
        current_11[i][1] = 0

    x_11[i] = position[i][0]
    y_11[i] = position[i][1]
    t_11[i] = current_11[i][0]
    Amp_11[i] = current_11[i][1]

###  point(12)
for i in range(0, nsteps - 1):

    if np.all(np.sqrt((position[i][0] - point_x[11]) ** 2 + (position[i][1] - point_y[11]) ** 2) < radius):
        if A_random[i] > 0.3:

            current_12[i][0] = i
            current_12[i][1] = A * A_random[i]
            firing_position_x_12[i] = position[i][0]
            firing_position_y_12[i] = position[i][1]
            z_12 += 1
        else:
            current_12[i][0] = i
            current_12[i][1] = 0

    else:
        current_12[i][0] = i
        current_12[i][1] = 0

    x_12[i] = position[i][0]
    y_12[i] = position[i][1]
    t_12[i] = current_12[i][0]
    Amp_12[i] = current_12[i][1]

###  point(13)
for i in range(0, nsteps - 1):

    if np.all(np.sqrt((position[i][0] - point_x[12]) ** 2 + (position[i][1] - point_y[12]) ** 2) < radius):
        if A_random[i] > 0.3:

            current_13[i][0] = i
            current_13[i][1] = A * A_random[i]
            firing_position_x_13[i] = position[i][0]
            firing_position_y_13[i] = position[i][1]
            z_13 += 1
        else:
            current_13[i][0] = i
            current_13[i][1] = 0

    else:
        current_13[i][0] = i
        current_13[i][1] = 0

    x_13[i] = position[i][0]
    y_13[i] = position[i][1]
    t_13[i] = current_13[i][0]
    Amp_13[i] = current_13[i][1]

###  point(14)
for i in range(0, nsteps - 1):

    if np.all(np.sqrt((position[i][0] - point_x[13]) ** 2 + (position[i][1] - point_y[13]) ** 2) < radius):
        if A_random[i] > 0.3:

            current_14[i][0] = i
            current_14[i][1] = A * A_random[i]
            firing_position_x_14[i] = position[i][0]
            firing_position_y_14[i] = position[i][1]
            z_14 += 1
        else:
            current_14[i][0] = i
            current_14[i][1] = 0

    else:
        current_14[i][0] = i
        current_14[i][1] = 0

    x_14[i] = position[i][0]
    y_14[i] = position[i][1]
    t_14[i] = current_14[i][0]
    Amp_14[i] = current_14[i][1]

###  point(15)
for i in range(0, nsteps - 1):

    if np.all(np.sqrt((position[i][0] - point_x[14]) ** 2 + (position[i][1] - point_y[14]) ** 2) < radius):
        if A_random[i] > 0.3:

            current_15[i][0] = i
            current_15[i][1] = A * A_random[i]
            firing_position_x_15[i] = position[i][0]
            firing_position_y_15[i] = position[i][1]
            z_15 += 1
        else:
            current_15[i][0] = i
            current_15[i][1] = 0

    else:
        current_15[i][0] = i
        current_15[i][1] = 0

    x_15[i] = position[i][0]
    y_15[i] = position[i][1]
    t_15[i] = current_15[i][0]
    Amp_15[i] = current_15[i][1]

###  point(16)
for i in range(0, nsteps - 1):

    if np.all(np.sqrt((position[i][0] - point_x[15]) ** 2 + (position[i][1] - point_y[15]) ** 2) < radius):
        if A_random[i] > 0.3:

            current_16[i][0] = i
            current_16[i][1] = A * A_random[i]
            firing_position_x_16[i] = position[i][0]
            firing_position_y_16[i] = position[i][1]
            z_16 += 1
        else:
            current_16[i][0] = i
            current_16[i][1] = 0

    else:
        current_16[i][0] = i
        current_16[i][1] = 0

    x_16[i] = position[i][0]
    y_16[i] = position[i][1]
    t_16[i] = current_16[i][0]
    Amp_16[i] = current_16[i][1]


# ==============================================================================#

def LIF_1(_I_1=current_1[i][1], gl=0.16, Cm=0.0049):
    # Constants
    El = -0.065  # restint membrane potential [V]
    thresh = -0.050  # spiking threshold [V]

    # Experimental Setup
    # TIME
    T_1 = nsteps - 1  # total simulation length [s]
    dt = 1  # step size [s]
    time_1 = np.arange(0, T_1 + dt, dt)  # step values [s]
    # VOLTAGE
    V_1 = np.empty(len(time_1))  # array for saving Voltage history
    V_1[0] = El  # set initial to resting potential
    # CURRENT

    I_1 = Amp_1

    # Measurements
    spikes = 0  # counter for number of spikes

    # Simulation
    for i in range(1, len(time_1)):
        # use "I - V/R = C * dV/dT" to get this equation
        dV_1 = (Amp_1[i] - gl * (V_1[i - 1] - El)) / Cm
        V_1[i] = V_1[i - 1] + dV_1 * dt

        # in case we exceed threshold
        if V_1[i] > thresh:
            V_1[i - 1] = 0.04  # set the last step to spike value
            V_1[i] = El  # current step is resting membrane potential
            spikes += 1  # count spike

    return V_1


def LIF_2(_I_2=current_2[i][1], gl=0.16, Cm=0.0049):
    # Constants
    El = -0.065  # restint membrane potential [V]
    thresh = -0.050  # spiking threshold [V]

    # Experimental Setup
    # TIME
    T_2 = nsteps - 1  # total simulation length [s]
    dt = 1  # step size [s]
    time_2 = np.arange(0, T_2 + dt, dt)  # step values [s]
    # VOLTAGE
    V_2 = np.empty(len(time_2))  # array for saving Voltage history
    V_2[0] = El  # set initial to resting potential
    # CURRENT

    I_2 = Amp_2

    # Measurements
    spikes = 0  # counter for number of spikes

    # Simulation
    for i in range(1, len(time_2)):
        # use "I - V/R = C * dV/dT" to get this equation
        dV_2 = (Amp_2[i] - gl * (V_2[i - 1] - El)) / Cm
        V_2[i] = V_2[i - 1] + dV_2 * dt

        # in case we exceed threshold
        if V_2[i] > thresh:
            V_2[i - 1] = 0.04  # set the last step to spike value
            V_2[i] = El  # current step is resting membrane potential
            spikes += 1  # count spike

    return V_2


def LIF_3(_I_3=current_3[i][1], gl=0.16, Cm=0.0049):
    # Constants
    El = -0.065  # restint membrane potential [V]
    thresh = -0.050  # spiking threshold [V]

    # Experimental Setup
    # TIME
    T_3 = nsteps - 1  # total simulation length [s]
    dt = 1  # step size [s]
    time_3 = np.arange(0, T_3 + dt, dt)  # step values [s]
    # VOLTAGE
    V_3 = np.empty(len(time_3))  # array for saving Voltage history
    V_3[0] = El  # set initial to resting potential
    # CURRENT

    I_3 = Amp_3

    # Measurements
    spikes = 0  # counter for number of spikes

    # Simulation
    for i in range(1, len(time_3)):
        # use "I - V/R = C * dV/dT" to get this equation
        dV_3 = (Amp_3[i] - gl * (V_3[i - 1] - El)) / Cm
        V_3[i] = V_3[i - 1] + dV_3 * dt

        # in case we exceed threshold
        if V_3[i] > thresh:
            V_3[i - 1] = 0.04  # set the last step to spike value
            V_3[i] = El  # current step is resting membrane potential
            spikes += 1  # count spike

    return V_3

    # Experimental Setup
    # TIME
    T_4 = nsteps - 1  # total simulation length [s]
    dt = 1  # step size [s]
    time_4 = np.arange(0, T_4 + dt, dt)  # step values [s]
    # VOLTAGE
    V_4 = np.empty(len(time_4))  # array for saving Voltage history
    V_4[0] = El  # set initial to resting potential
    # CURRENT

    I_4 = Amp_4

    # Measurements
    spikes = 0  # counter for number of spikes

    # Simulation
    for i in range(1, len(time_4)):
        # use "I - V/R = C * dV/dT" to get this equation
        dV_4 = (Amp_4[i] - gl * (V_4[i - 1] - El)) / Cm
        V_4[i] = V_4[i - 1] + dV_4 * dt

        # in case we exceed threshold
        if V_4[i] > thresh:
            V_4[i - 1] = 0.04  # set the last step to spike value
            V_4[i] = El  # current step is resting membrane potential
            spikes += 1  # count spike

    return V_4


def LIF_4(_I_4=current_4[i][1], gl=0.16, Cm=0.0049):
    # Constants
    El = -0.065  # restint membrane potential [V]
    thresh = -0.050  # spiking threshold [V]

    # Experimental Setup
    # TIME
    T_4 = nsteps - 1  # total simulation length [s]
    dt = 1  # step size [s]
    time_4 = np.arange(0, T_4 + dt, dt)  # step values [s]
    # VOLTAGE
    V_4 = np.empty(len(time_4))  # array for saving Voltage history
    V_4[0] = El  # set initial to resting potential
    # CURRENT

    I_4 = Amp_4

    # Measurements
    spikes = 0  # counter for number of spikes

    # Simulation
    for i in range(1, len(time_4)):
        # use "I - V/R = C * dV/dT" to get this equation
        dV_4 = (Amp_4[i] - gl * (V_4[i - 1] - El)) / Cm
        V_4[i] = V_4[i - 1] + dV_4 * dt

        # in case we exceed threshold
        if V_4[i] > thresh:
            V_4[i - 1] = 0.04  # set the last step to spike value
            V_4[i] = El  # current step is resting membrane potential
            spikes += 1  # count spike

    return V_4


def LIF_5(_I_5=current_5[i][1], gl=0.16, Cm=0.0049):
    # Constants
    El = -0.065  # restint membrane potential [V]
    thresh = -0.050  # spiking threshold [V]

    # Experimental Setup
    # TIME
    T_5 = nsteps - 1  # total simulation length [s]
    dt = 1  # step size [s]
    time_5 = np.arange(0, T_5 + dt, dt)  # step values [s]
    # VOLTAGE
    V_5 = np.empty(len(time_5))  # array for saving Voltage history
    V_5[0] = El  # set initial to resting potential
    # CURRENT

    I_5 = Amp_5

    # Measurements
    spikes = 0  # counter for number of spikes

    # Simulation
    for i in range(1, len(time_5)):
        # use "I - V/R = C * dV/dT" to get this equation
        dV_5 = (Amp_5[i] - gl * (V_5[i - 1] - El)) / Cm
        V_5[i] = V_5[i - 1] + dV_5 * dt

        # in case we exceed threshold
        if V_5[i] > thresh:
            V_5[i - 1] = 0.04  # set the last step to spike value
            V_5[i] = El  # current step is resting membrane potential
            spikes += 1  # count spike

    return V_5


def LIF_6(_I_6=current_6[i][1], gl=0.16, Cm=0.0049):
    # Constants
    El = -0.065  # restint membrane potential [V]
    thresh = -0.050  # spiking threshold [V]

    # Experimental Setup
    # TIME
    T_6 = nsteps - 1  # total simulation length [s]
    dt = 1  # step size [s]
    time_6 = np.arange(0, T_6 + dt, dt)  # step values [s]
    # VOLTAGE
    V_6 = np.empty(len(time_6))  # array for saving Voltage history
    V_6[0] = El  # set initial to resting potential
    # CURRENT

    I_6 = Amp_6

    # Measurements
    spikes = 0  # counter for number of spikes

    # Simulation
    for i in range(1, len(time_6)):
        # use "I - V/R = C * dV/dT" to get this equation
        dV_6 = (Amp_6[i] - gl * (V_6[i - 1] - El)) / Cm
        V_6[i] = V_6[i - 1] + dV_6 * dt

        # in case we exceed threshold
        if V_6[i] > thresh:
            V_6[i - 1] = 0.04  # set the last step to spike value
            V_6[i] = El  # current step is resting membrane potential
            spikes += 1  # count spike

    return V_6


def LIF_7(_I_7=current_7[i][1], gl=0.16, Cm=0.0049):
    # Constants
    El = -0.065  # restint membrane potential [V]
    thresh = -0.050  # spiking threshold [V]

    # Experimental Setup
    # TIME
    T_7 = nsteps - 1  # total simulation length [s]
    dt = 1  # step size [s]
    time_7 = np.arange(0, T_7 + dt, dt)  # step values [s]
    # VOLTAGE
    V_7 = np.empty(len(time_7))  # array for saving Voltage history
    V_7[0] = El  # set initial to resting potential
    # CURRENT

    I_7 = Amp_7

    # Measurements
    spikes = 0  # counter for number of spikes

    # Simulation
    for i in range(1, len(time_7)):
        # use "I - V/R = C * dV/dT" to get this equation
        dV_7 = (Amp_7[i] - gl * (V_7[i - 1] - El)) / Cm
        V_7[i] = V_7[i - 1] + dV_7 * dt

        # in case we exceed threshold
        if V_7[i] > thresh:
            V_7[i - 1] = 0.04  # set the last step to spike value
            V_7[i] = El  # current step is resting membrane potential
            spikes += 1  # count spike

    return V_7


def LIF_8(_I_8=current_8[i][1], gl=0.16, Cm=0.0049):
    # Constants
    El = -0.065  # restint membrane potential [V]
    thresh = -0.050  # spiking threshold [V]

    # Experimental Setup
    # TIME
    T_8 = nsteps - 1  # total simulation length [s]
    dt = 1  # step size [s]
    time_8 = np.arange(0, T_8 + dt, dt)  # step values [s]
    # VOLTAGE
    V_8 = np.empty(len(time_8))  # array for saving Voltage history
    V_8[0] = El  # set initial to resting potential
    # CURRENT

    I_8 = Amp_8

    # Measurements
    spikes = 0  # counter for number of spikes

    # Simulation
    for i in range(1, len(time_8)):
        # use "I - V/R = C * dV/dT" to get this equation
        dV_8 = (Amp_8[i] - gl * (V_8[i - 1] - El)) / Cm
        V_8[i] = V_8[i - 1] + dV_8 * dt

        # in case we exceed threshold
        if V_8[i] > thresh:
            V_8[i - 1] = 0.04  # set the last step to spike value
            V_8[i] = El  # current step is resting membrane potential
            spikes += 1  # count spike

    return V_8


def LIF_9(_I_9=current_9[i][1], gl=0.16, Cm=0.0049):
    # Constants
    El = -0.065  # restint membrane potential [V]
    thresh = -0.050  # spiking threshold [V]

    # Experimental Setup
    # TIME
    T_9 = nsteps - 1  # total simulation length [s]
    dt = 1  # step size [s]
    time_9 = np.arange(0, T_9 + dt, dt)  # step values [s]
    # VOLTAGE
    V_9 = np.empty(len(time_9))  # array for saving Voltage history
    V_9[0] = El  # set initial to resting potential
    # CURRENT

    I_9 = Amp_9

    # Measurements
    spikes = 0  # counter for number of spikes

    # Simulation
    for i in range(1, len(time_9)):
        # use "I - V/R = C * dV/dT" to get this equation
        dV_9 = (Amp_9[i] - gl * (V_9[i - 1] - El)) / Cm
        V_9[i] = V_9[i - 1] + dV_9 * dt

        # in case we exceed threshold
        if V_9[i] > thresh:
            V_9[i - 1] = 0.04  # set the last step to spike value
            V_9[i] = El  # current step is resting membrane potential
            spikes += 1  # count spike

    return V_9


def LIF_10(_I_10=current_10[i][1], gl=0.16, Cm=0.0049):
    # Constants
    El = -0.065  # restint membrane potential [V]
    thresh = -0.050  # spiking threshold [V]

    # Experimental Setup
    # TIME
    T_10 = nsteps - 1  # total simulation length [s]
    dt = 1  # step size [s]
    time_10 = np.arange(0, T_10 + dt, dt)  # step values [s]
    # VOLTAGE
    V_10 = np.empty(len(time_10))  # array for saving Voltage history
    V_10[0] = El  # set initial to resting potential
    # CURRENT

    I_10 = Amp_10

    # Measurements
    spikes = 0  # counter for number of spikes

    # Simulation
    for i in range(1, len(time_10)):
        # use "I - V/R = C * dV/dT" to get this equation
        dV_10 = (Amp_10[i] - gl * (V_10[i - 1] - El)) / Cm
        V_10[i] = V_10[i - 1] + dV_10 * dt

        # in case we exceed threshold
        if V_10[i] > thresh:
            V_10[i - 1] = 0.04  # set the last step to spike value
            V_10[i] = El  # current step is resting membrane potential
            spikes += 1  # count spike

    return V_10


def LIF_11(_I_11=current_11[i][1], gl=0.16, Cm=0.0049):
    # Constants
    El = -0.065  # restint membrane potential [V]
    thresh = -0.050  # spiking threshold [V]

    # Experimental Setup
    # TIME
    T_11 = nsteps - 1  # total simulation length [s]
    dt = 1  # step size [s]
    time_11 = np.arange(0, T_11 + dt, dt)  # step values [s]
    # VOLTAGE
    V_11 = np.empty(len(time_11))  # array for saving Voltage history
    V_11[0] = El  # set initial to resting potential
    # CURRENT

    I_11 = Amp_11

    # Measurements
    spikes = 0  # counter for number of spikes

    # Simulation
    for i in range(1, len(time_11)):
        # use "I - V/R = C * dV/dT" to get this equation
        dV_11 = (Amp_11[i] - gl * (V_11[i - 1] - El)) / Cm
        V_11[i] = V_11[i - 1] + dV_11 * dt

        # in case we exceed threshold
        if V_11[i] > thresh:
            V_11[i - 1] = 0.04  # set the last step to spike value
            V_11[i] = El  # current step is resting membrane potential
            spikes += 1  # count spike

    return V_11


def LIF_12(_I_12=current_12[i][1], gl=0.16, Cm=0.0049):
    # Constants
    El = -0.065  # restint membrane potential [V]
    thresh = -0.050  # spiking threshold [V]

    # Experimental Setup
    # TIME
    T_12 = nsteps - 1  # total simulation length [s]
    dt = 1  # step size [s]
    time_12 = np.arange(0, T_12 + dt, dt)  # step values [s]
    # VOLTAGE
    V_12 = np.empty(len(time_12))  # array for saving Voltage history
    V_12[0] = El  # set initial to resting potential
    # CURRENT

    I_12 = Amp_12

    # Measurements
    spikes = 0  # counter for number of spikes

    # Simulation
    for i in range(1, len(time_12)):
        # use "I - V/R = C * dV/dT" to get this equation
        dV_12 = (Amp_12[i] - gl * (V_12[i - 1] - El)) / Cm
        V_12[i] = V_12[i - 1] + dV_12 * dt

        # in case we exceed threshold
        if V_12[i] > thresh:
            V_12[i - 1] = 0.04  # set the last step to spike value
            V_12[i] = El  # current step is resting membrane potential
            spikes += 1  # count spike

    return V_12


def LIF_13(_I_13=current_13[i][1], gl=0.16, Cm=0.0049):
    # Constants
    El = -0.065  # restint membrane potential [V]
    thresh = -0.050  # spiking threshold [V]

    # Experimental Setup
    # TIME
    T_13 = nsteps - 1  # total simulation length [s]
    dt = 1  # step size [s]
    time_13 = np.arange(0, T_13 + dt, dt)  # step values [s]
    # VOLTAGE
    V_13 = np.empty(len(time_13))  # array for saving Voltage history
    V_13[0] = El  # set initial to resting potential
    # CURRENT

    I_13 = Amp_13

    # Measurements
    spikes = 0  # counter for number of spikes

    # Simulation
    for i in range(1, len(time_13)):
        # use "I - V/R = C * dV/dT" to get this equation
        dV_13 = (Amp_13[i] - gl * (V_13[i - 1] - El)) / Cm
        V_13[i] = V_13[i - 1] + dV_13 * dt

        # in case we exceed threshold
        if V_13[i] > thresh:
            V_13[i - 1] = 0.04  # set the last step to spike value
            V_13[i] = El  # current step is resting membrane potential
            spikes += 1  # count spike

    return V_13


def LIF_14(_I_14=current_14[i][1], gl=0.16, Cm=0.0049):
    # Constants
    El = -0.065  # restint membrane potential [V]
    thresh = -0.050  # spiking threshold [V]

    # Experimental Setup
    # TIME
    T_14 = nsteps - 1  # total simulation length [s]
    dt = 1  # step size [s]
    time_14 = np.arange(0, T_14 + dt, dt)  # step values [s]
    # VOLTAGE
    V_14 = np.empty(len(time_14))  # array for saving Voltage history
    V_14[0] = El  # set initial to resting potential
    # CURRENT

    I_14 = Amp_14

    # Measurements
    spikes = 0  # counter for number of spikes

    # Simulation
    for i in range(1, len(time_14)):
        # use "I - V/R = C * dV/dT" to get this equation
        dV_14 = (Amp_14[i] - gl * (V_14[i - 1] - El)) / Cm
        V_14[i] = V_14[i - 1] + dV_14 * dt

        # in case we exceed threshold
        if V_14[i] > thresh:
            V_14[i - 1] = 0.04  # set the last step to spike value
            V_14[i] = El  # current step is resting membrane potential
            spikes += 1  # count spike

    return V_14


def LIF_15(_I_15=current_15[i][1], gl=0.16, Cm=0.0049):
    # Constants
    El = -0.065  # restint membrane potential [V]
    thresh = -0.050  # spiking threshold [V]

    # Experimental Setup
    # TIME
    T_15 = nsteps - 1  # total simulation length [s]
    dt = 1  # step size [s]
    time_15 = np.arange(0, T_15 + dt, dt)  # step values [s]
    # VOLTAGE
    V_15 = np.empty(len(time_15))  # array for saving Voltage history
    V_15[0] = El  # set initial to resting potential
    # CURRENT

    I_15 = Amp_15

    # Measurements
    spikes = 0  # counter for number of spikes

    # Simulation
    for i in range(1, len(time_15)):
        # use "I - V/R = C * dV/dT" to get this equation
        dV_15 = (Amp_15[i] - gl * (V_15[i - 1] - El)) / Cm
        V_15[i] = V_15[i - 1] + dV_15 * dt

        # in case we exceed threshold
        if V_15[i] > thresh:
            V_15[i - 1] = 0.04  # set the last step to spike value
            V_15[i] = El  # current step is resting membrane potential
            spikes += 1  # count spike

    return V_15


def LIF_16(_I_16=current_16[i][1], gl=0.16, Cm=0.0049):
    # Constants
    El = -0.065  # restint membrane potential [V]
    thresh = -0.050  # spiking threshold [V]

    # Experimental Setup
    # TIME
    T_16 = nsteps - 1  # total simulation length [s]
    dt = 1  # step size [s]
    time_16 = np.arange(0, T_16 + dt, dt)  # step values [s]
    # VOLTAGE
    V_16 = np.empty(len(time_16))  # array for saving Voltage history
    V_16[0] = El  # set initial to resting potential
    # CURRENT

    I_16 = Amp_16

    # Measurements
    spikes = 0  # counter for number of spikes

    # Simulation
    for i in range(1, len(time_16)):
        # use "I - V/R = C * dV/dT" to get this equation
        dV_16 = (Amp_16[i] - gl * (V_16[i - 1] - El)) / Cm
        V_16[i] = V_16[i - 1] + dV_16 * dt

        # in case we exceed threshold
        if V_16[i] > thresh:
            V_16[i - 1] = 0.04  # set the last step to spike value
            V_16[i] = El  # current step is resting membrane potential
            spikes += 1  # count spike

    return V_16


# ==============================================================================#

def I_values_1(_I_1=current_1[i][1]):
    I_1 = Amp_1
    return I_1


def I_values_2(_I_2=current_2[i][1]):
    I_2 = Amp_2
    return I_2


def I_values_3(_I_3=current_3[i][1]):
    I_3 = Amp_3
    return I_3


def I_values_4(_I_4=current_4[i][1]):
    I_4 = Amp_4
    return I_4


def I_values_5(_I_5=current_5[i][1]):
    I_5 = Amp_5
    return I_5


def I_values_6(_I_6=current_6[i][1]):
    I_6 = Amp_6
    return I_6


def I_values_7(_I_7=current_7[i][1]):
    I_7 = Amp_7
    return I_7


def I_values_8(_I_8=current_8[i][1]):
    I_8 = Amp_8
    return I_8


def I_values_9(_I_9=current_9[i][1]):
    I_9 = Amp_9
    return I_9


def I_values_10(_I_10=current_10[i][1]):
    I_10 = Amp_10
    return I_10


def I_values_11(_I_11=current_11[i][1]):
    I_11 = Amp_11
    return I_11


def I_values_12(_I_12=current_12[i][1]):
    I_12 = Amp_12
    return I_12


def I_values_13(_I_13=current_13[i][1]):
    I_13 = Amp_13
    return I_13


def I_values_14(_I_14=current_14[i][1]):
    I_14 = Amp_14
    return I_14


def I_values_15(_I_15=current_15[i][1]):
    I_15 = Amp_15
    return I_15


def I_values_16(_I_16=current_16[i][1]):
    I_16 = Amp_16
    return I_16


# ==============================================================================#
# time parameters for plotting
dt = 1  # step size [s]

T_1 = nsteps - 1  # total simulation length [s]
T_2 = nsteps - 1  # total simulation length [s]
T_3 = nsteps - 1  # total simulation length [s]
T_4 = nsteps - 1  # total simulation length [s]
T_5 = nsteps - 1  # total simulation length [s]
T_6 = nsteps - 1  # total simulation length [s]
T_7 = nsteps - 1  # total simulation length [s]
T_8 = nsteps - 1  # total simulation length [s]
T_9 = nsteps - 1  # total simulation length [s]
T_10 = nsteps - 1  # total simulation length [s]
T_11 = nsteps - 1  # total simulation length [s]
T_12 = nsteps - 1  # total simulation length [s]
T_13 = nsteps - 1  # total simulation length [s]
T_14 = nsteps - 1  # total simulation length [s]
T_15 = nsteps - 1  # total simulation length [s]
T_16 = nsteps - 1  # total simulation length [s]

time_1 = np.arange(0, T_1 + dt, dt)  # step values [s]
time_2 = np.arange(0, T_2 + dt, dt)  # step values [s]
time_3 = np.arange(0, T_3 + dt, dt)  # step values [s]
time_4 = np.arange(0, T_4 + dt, dt)  # step values [s]
time_5 = np.arange(0, T_5 + dt, dt)  # step values [s]
time_6 = np.arange(0, T_6 + dt, dt)  # step values [s]
time_7 = np.arange(0, T_7 + dt, dt)  # step values [s]
time_8 = np.arange(0, T_8 + dt, dt)  # step values [s]
time_9 = np.arange(0, T_9 + dt, dt)  # step values [s]
time_10 = np.arange(0, T_10 + dt, dt)  # step values [s]
time_11 = np.arange(0, T_11 + dt, dt)  # step values [s]
time_12 = np.arange(0, T_12 + dt, dt)  # step values [s]
time_13 = np.arange(0, T_13 + dt, dt)  # step values [s]
time_14 = np.arange(0, T_14 + dt, dt)  # step values [s]
time_15 = np.arange(0, T_15 + dt, dt)  # step values [s]
time_16 = np.arange(0, T_16 + dt, dt)  # step values [s]

I_init = 0
gl_init = 0.16
Cm_init = 0.0049

V_1 = LIF_1(_I_1=I_init, gl=gl_init, Cm=Cm_init)
I_1 = I_values_1(_I_1=I_init)

V_2 = LIF_2(_I_2=I_init, gl=gl_init, Cm=Cm_init)
I_2 = I_values_2(_I_2=I_init)

V_3 = LIF_3(_I_3=I_init, gl=gl_init, Cm=Cm_init)
I_3 = I_values_3(_I_3=I_init)

V_4 = LIF_4(_I_4=I_init, gl=gl_init, Cm=Cm_init)
I_4 = I_values_4(_I_4=I_init)

V_5 = LIF_5(_I_5=I_init, gl=gl_init, Cm=Cm_init)
I_5 = I_values_5(_I_5=I_init)

V_6 = LIF_6(_I_6=I_init, gl=gl_init, Cm=Cm_init)
I_6 = I_values_6(_I_6=I_init)

V_7 = LIF_7(_I_7=I_init, gl=gl_init, Cm=Cm_init)
I_7 = I_values_7(_I_7=I_init)

V_8 = LIF_8(_I_8=I_init, gl=gl_init, Cm=Cm_init)
I_8 = I_values_8(_I_8=I_init)

V_9 = LIF_9(_I_9=I_init, gl=gl_init, Cm=Cm_init)
I_9 = I_values_9(_I_9=I_init)

V_10 = LIF_10(_I_10=I_init, gl=gl_init, Cm=Cm_init)
I_10 = I_values_10(_I_10=I_init)

V_11 = LIF_11(_I_11=I_init, gl=gl_init, Cm=Cm_init)
I_11 = I_values_11(_I_11=I_init)

V_12 = LIF_12(_I_12=I_init, gl=gl_init, Cm=Cm_init)
I_12 = I_values_12(_I_12=I_init)

V_13 = LIF_13(_I_13=I_init, gl=gl_init, Cm=Cm_init)
I_13 = I_values_13(_I_13=I_init)

V_14 = LIF_14(_I_14=I_init, gl=gl_init, Cm=Cm_init)
I_14 = I_values_14(_I_14=I_init)

V_15 = LIF_15(_I_15=I_init, gl=gl_init, Cm=Cm_init)
I_15 = I_values_15(_I_15=I_init)

V_16 = LIF_16(_I_16=I_init, gl=gl_init, Cm=Cm_init)
I_16 = I_values_16(_I_16=I_init)

######### Plotting

axis_color = 'lightgoldenrodyellow'

fig = plt.figure("Leaky Integrate-and-Fire Neuron", figsize=(17, 7))
ax = fig.add_subplot(3, 1, 1)
plt.title("Interactive Leaky Integrate-and-Fire Neuron Simulation_1")
fig.subplots_adjust(left=0.1, bottom=0.32)

# plot lines
line = plt.plot(time_1, V_1, label="Membrane Potential")[0]
line2 = plt.plot(time_1, I_1, label="Applied Current")[0]

# add legend
plt.legend(loc="upper right")

# add axis labels
plt.ylabel("Potential [V]/ Current [A]")
plt.xlabel("Time [s]")

axis_color = 'lightgoldenrodyellow'

fig = plt.figure("Leaky Integrate-and-Fire Neuron", figsize=(14, 7))
ax = fig.add_subplot(3, 1, 3)
plt.title("Interactive Leaky Integrate-and-Fire Neuron Simulation_2")
fig.subplots_adjust(left=0.1, bottom=0.32)

# plot lines
line = plt.plot(time_2, V_2, label="Membrane Potential")[0]
line2 = plt.plot(time_2, I_2, label="Applied Current")[0]

# add legend
plt.legend(loc="upper right")

# add axis labels
plt.ylabel("Potential [V]/ Current [A]")
plt.xlabel("Time [s]")

plt.show()

######### Plotting

axis_color = 'lightgoldenrodyellow'

fig = plt.figure("Leaky Integrate-and-Fire Neuron", figsize=(17, 7))
ax = fig.add_subplot(3, 1, 1)
plt.title("Interactive Leaky Integrate-and-Fire Neuron Simulation_3")
fig.subplots_adjust(left=0.1, bottom=0.32)

# plot lines
line = plt.plot(time_3, V_3, label="Membrane Potential")[0]
line2 = plt.plot(time_3, I_3, label="Applied Current")[0]

# add legend
plt.legend(loc="upper right")

# add axis labels
plt.ylabel("Potential [V]/ Current [A]")
plt.xlabel("Time [s]")

axis_color = 'lightgoldenrodyellow'

fig = plt.figure("Leaky Integrate-and-Fire Neuron", figsize=(14, 7))
ax = fig.add_subplot(3, 1, 3)
plt.title("Interactive Leaky Integrate-and-Fire Neuron Simulation_4")
fig.subplots_adjust(left=0.1, bottom=0.32)

# plot lines
line = plt.plot(time_4, V_4, label="Membrane Potential")[0]
line2 = plt.plot(time_4, I_4, label="Applied Current")[0]

# add legend
plt.legend(loc="upper right")

# add axis labels
plt.ylabel("Potential [V]/ Current [A]")
plt.xlabel("Time [s]")

plt.show()

######### Plotting

axis_color = 'lightgoldenrodyellow'

fig = plt.figure("Leaky Integrate-and-Fire Neuron", figsize=(17, 7))
ax = fig.add_subplot(3, 1, 1)
plt.title("Interactive Leaky Integrate-and-Fire Neuron Simulation_5")
fig.subplots_adjust(left=0.1, bottom=0.32)

# plot lines
line = plt.plot(time_5, V_5, label="Membrane Potential")[0]
line2 = plt.plot(time_5, I_5, label="Applied Current")[0]

# add legend
plt.legend(loc="upper right")

# add axis labels
plt.ylabel("Potential [V]/ Current [A]")
plt.xlabel("Time [s]")

axis_color = 'lightgoldenrodyellow'

fig = plt.figure("Leaky Integrate-and-Fire Neuron", figsize=(14, 7))
ax = fig.add_subplot(3, 1, 3)
plt.title("Interactive Leaky Integrate-and-Fire Neuron Simulation_6")
fig.subplots_adjust(left=0.1, bottom=0.32)

# plot lines
line = plt.plot(time_6, V_6, label="Membrane Potential")[0]
line2 = plt.plot(time_6, I_6, label="Applied Current")[0]

# add legend
plt.legend(loc="upper right")

# add axis labels
plt.ylabel("Potential [V]/ Current [A]")
plt.xlabel("Time [s]")

plt.show()

######### Plotting

axis_color = 'lightgoldenrodyellow'

fig = plt.figure("Leaky Integrate-and-Fire Neuron", figsize=(17, 7))
ax = fig.add_subplot(3, 1, 1)
plt.title("Interactive Leaky Integrate-and-Fire Neuron Simulation_7")
fig.subplots_adjust(left=0.1, bottom=0.32)

# plot lines
line = plt.plot(time_7, V_7, label="Membrane Potential")[0]
line2 = plt.plot(time_7, I_7, label="Applied Current")[0]

# add legend
plt.legend(loc="upper right")

# add axis labels
plt.ylabel("Potential [V]/ Current [A]")
plt.xlabel("Time [s]")

axis_color = 'lightgoldenrodyellow'

fig = plt.figure("Leaky Integrate-and-Fire Neuron", figsize=(14, 7))
ax = fig.add_subplot(3, 1, 3)
plt.title("Interactive Leaky Integrate-and-Fire Neuron Simulation_8")
fig.subplots_adjust(left=0.1, bottom=0.32)

# plot lines
line = plt.plot(time_8, V_8, label="Membrane Potential")[0]
line2 = plt.plot(time_8, I_8, label="Applied Current")[0]

# add legend
plt.legend(loc="upper right")

# add axis labels
plt.ylabel("Potential [V]/ Current [A]")
plt.xlabel("Time [s]")

plt.show()

######### Plotting

axis_color = 'lightgoldenrodyellow'

fig = plt.figure("Leaky Integrate-and-Fire Neuron", figsize=(17, 7))
ax = fig.add_subplot(3, 1, 1)
plt.title("Interactive Leaky Integrate-and-Fire Neuron Simulation_9")
fig.subplots_adjust(left=0.1, bottom=0.32)

# plot lines
line = plt.plot(time_9, V_9, label="Membrane Potential")[0]
line2 = plt.plot(time_9, I_9, label="Applied Current")[0]

# add legend
plt.legend(loc="upper right")

# add axis labels
plt.ylabel("Potential [V]/ Current [A]")
plt.xlabel("Time [s]")

axis_color = 'lightgoldenrodyellow'

fig = plt.figure("Leaky Integrate-and-Fire Neuron", figsize=(14, 7))
ax = fig.add_subplot(3, 1, 3)
plt.title("Interactive Leaky Integrate-and-Fire Neuron Simulation_10")
fig.subplots_adjust(left=0.1, bottom=0.32)

# plot lines
line = plt.plot(time_10, V_10, label="Membrane Potential")[0]
line2 = plt.plot(time_10, I_10, label="Applied Current")[0]

# add legend
plt.legend(loc="upper right")

# add axis labels
plt.ylabel("Potential [V]/ Current [A]")
plt.xlabel("Time [s]")

plt.show()

######### Plotting

axis_color = 'lightgoldenrodyellow'

fig = plt.figure("Leaky Integrate-and-Fire Neuron", figsize=(17, 7))
ax = fig.add_subplot(3, 1, 1)
plt.title("Interactive Leaky Integrate-and-Fire Neuron Simulation_11")
fig.subplots_adjust(left=0.1, bottom=0.32)

# plot lines
line = plt.plot(time_11, V_11, label="Membrane Potential")[0]
line2 = plt.plot(time_11, I_11, label="Applied Current")[0]

# add legend
plt.legend(loc="upper right")

# add axis labels
plt.ylabel("Potential [V]/ Current [A]")
plt.xlabel("Time [s]")

axis_color = 'lightgoldenrodyellow'

fig = plt.figure("Leaky Integrate-and-Fire Neuron", figsize=(14, 7))
ax = fig.add_subplot(3, 1, 3)
plt.title("Interactive Leaky Integrate-and-Fire Neuron Simulation_12")
fig.subplots_adjust(left=0.1, bottom=0.32)

# plot lines
line = plt.plot(time_12, V_12, label="Membrane Potential")[0]
line2 = plt.plot(time_12, I_12, label="Applied Current")[0]

# add legend
plt.legend(loc="upper right")

# add axis labels
plt.ylabel("Potential [V]/ Current [A]")
plt.xlabel("Time [s]")

plt.show()

######### Plotting

axis_color = 'lightgoldenrodyellow'

fig = plt.figure("Leaky Integrate-and-Fire Neuron", figsize=(17, 7))
ax = fig.add_subplot(3, 1, 1)
plt.title("Interactive Leaky Integrate-and-Fire Neuron Simulation_13")
fig.subplots_adjust(left=0.1, bottom=0.32)

# plot lines
line = plt.plot(time_13, V_13, label="Membrane Potential")[0]
line2 = plt.plot(time_13, I_13, label="Applied Current")[0]

# add legend
plt.legend(loc="upper right")

# add axis labels
plt.ylabel("Potential [V]/ Current [A]")
plt.xlabel("Time [s]")

axis_color = 'lightgoldenrodyellow'

fig = plt.figure("Leaky Integrate-and-Fire Neuron", figsize=(14, 7))
ax = fig.add_subplot(3, 1, 3)
plt.title("Interactive Leaky Integrate-and-Fire Neuron Simulation_14")
fig.subplots_adjust(left=0.1, bottom=0.32)

# plot lines
line = plt.plot(time_14, V_14, label="Membrane Potential")[0]
line2 = plt.plot(time_14, I_14, label="Applied Current")[0]

# add legend
plt.legend(loc="upper right")

# add axis labels
plt.ylabel("Potential [V]/ Current [A]")
plt.xlabel("Time [s]")

plt.show()

######### Plotting

axis_color = 'lightgoldenrodyellow'

fig = plt.figure("Leaky Integrate-and-Fire Neuron", figsize=(17, 7))
ax = fig.add_subplot(3, 1, 1)
plt.title("Interactive Leaky Integrate-and-Fire Neuron Simulation_15")
fig.subplots_adjust(left=0.1, bottom=0.32)

# plot lines
line = plt.plot(time_15, V_15, label="Membrane Potential")[0]
line2 = plt.plot(time_15, I_15, label="Applied Current")[0]

# add legend
plt.legend(loc="upper right")

# add axis labels
plt.ylabel("Potential [V]/ Current [A]")
plt.xlabel("Time [s]")

axis_color = 'lightgoldenrodyellow'

fig = plt.figure("Leaky Integrate-and-Fire Neuron", figsize=(14, 7))
ax = fig.add_subplot(3, 1, 3)
plt.title("Interactive Leaky Integrate-and-Fire Neuron Simulation_16")
fig.subplots_adjust(left=0.1, bottom=0.32)

# plot lines
line = plt.plot(time_16, V_16, label="Membrane Potential")[0]
line2 = plt.plot(time_16, I_16, label="Applied Current")[0]

# add legend
plt.legend(loc="upper right")

# add axis labels
plt.ylabel("Potential [V]/ Current [A]")
plt.xlabel("Time [s]")

plt.show()

# ==============================================================================#

# modify the firing#############################################

firing_position_1 = np.zeros((z_1, 1))
firing_position_2 = np.zeros((z_2, 1))
firing_position_3 = np.zeros((z_3, 1))
firing_position_4 = np.zeros((z_4, 1))
firing_position_5 = np.zeros((z_5, 1))
firing_position_6 = np.zeros((z_6, 1))
firing_position_7 = np.zeros((z_7, 1))
firing_position_8 = np.zeros((z_8, 1))
firing_position_9 = np.zeros((z_9, 1))
firing_position_10 = np.zeros((z_10, 1))
firing_position_11 = np.zeros((z_11, 1))
firing_position_12 = np.zeros((z_12, 1))
firing_position_13 = np.zeros((z_13, 1))
firing_position_14 = np.zeros((z_14, 1))
firing_position_15 = np.zeros((z_15, 1))
firing_position_16 = np.zeros((z_16, 1))

f_p_x_1 = firing_position_x_1.ravel()
f_p_y_1 = firing_position_y_1.ravel()
f_p_x_2 = firing_position_x_2.ravel()
f_p_y_2 = firing_position_y_2.ravel()
f_p_x_3 = firing_position_x_3.ravel()
f_p_y_3 = firing_position_y_3.ravel()
f_p_x_4 = firing_position_x_4.ravel()
f_p_y_4 = firing_position_y_4.ravel()
f_p_x_5 = firing_position_x_5.ravel()
f_p_y_5 = firing_position_y_5.ravel()
f_p_x_6 = firing_position_x_6.ravel()
f_p_y_6 = firing_position_y_6.ravel()
f_p_x_7 = firing_position_x_7.ravel()
f_p_y_7 = firing_position_y_7.ravel()
f_p_x_8 = firing_position_x_8.ravel()
f_p_y_8 = firing_position_y_8.ravel()
f_p_x_9 = firing_position_x_9.ravel()
f_p_y_9 = firing_position_y_9.ravel()
f_p_x_10 = firing_position_x_10.ravel()
f_p_y_10 = firing_position_y_10.ravel()
f_p_x_11 = firing_position_x_11.ravel()
f_p_y_11 = firing_position_y_11.ravel()
f_p_x_12 = firing_position_x_12.ravel()
f_p_y_12 = firing_position_y_12.ravel()
f_p_x_13 = firing_position_x_13.ravel()
f_p_y_13 = firing_position_y_13.ravel()
f_p_x_14 = firing_position_x_14.ravel()
f_p_y_14 = firing_position_y_14.ravel()
f_p_x_15 = firing_position_x_15.ravel()
f_p_y_15 = firing_position_y_15.ravel()
f_p_x_16 = firing_position_x_16.ravel()
f_p_y_16 = firing_position_y_16.ravel()

f_p_1 = np.column_stack((f_p_x_1, f_p_y_1))
f_p_2 = np.column_stack((f_p_x_2, f_p_y_2))
f_p_3 = np.column_stack((f_p_x_3, f_p_y_3))
f_p_4 = np.column_stack((f_p_x_4, f_p_y_4))
f_p_5 = np.column_stack((f_p_x_5, f_p_y_5))
f_p_6 = np.column_stack((f_p_x_6, f_p_y_6))
f_p_7 = np.column_stack((f_p_x_7, f_p_y_7))
f_p_8 = np.column_stack((f_p_x_8, f_p_y_8))
f_p_9 = np.column_stack((f_p_x_9, f_p_y_9))
f_p_10 = np.column_stack((f_p_x_10, f_p_y_10))
f_p_11 = np.column_stack((f_p_x_11, f_p_y_11))
f_p_12 = np.column_stack((f_p_x_12, f_p_y_12))
f_p_13 = np.column_stack((f_p_x_13, f_p_y_13))
f_p_14 = np.column_stack((f_p_x_14, f_p_y_14))
f_p_15 = np.column_stack((f_p_x_15, f_p_y_15))
f_p_16 = np.column_stack((f_p_x_16, f_p_y_16))

f_p_1 = f_p_1[np.logical_not(np.logical_and(f_p_1[:, 0] == 0, f_p_1[:, 1] == 0))]
f_p_2 = f_p_2[np.logical_not(np.logical_and(f_p_2[:, 0] == 0, f_p_2[:, 1] == 0))]
f_p_3 = f_p_3[np.logical_not(np.logical_and(f_p_3[:, 0] == 0, f_p_3[:, 1] == 0))]
f_p_4 = f_p_4[np.logical_not(np.logical_and(f_p_4[:, 0] == 0, f_p_4[:, 1] == 0))]
f_p_5 = f_p_5[np.logical_not(np.logical_and(f_p_5[:, 0] == 0, f_p_5[:, 1] == 0))]
f_p_6 = f_p_6[np.logical_not(np.logical_and(f_p_6[:, 0] == 0, f_p_6[:, 1] == 0))]
f_p_7 = f_p_7[np.logical_not(np.logical_and(f_p_7[:, 0] == 0, f_p_7[:, 1] == 0))]
f_p_8 = f_p_8[np.logical_not(np.logical_and(f_p_8[:, 0] == 0, f_p_8[:, 1] == 0))]
f_p_9 = f_p_9[np.logical_not(np.logical_and(f_p_9[:, 0] == 0, f_p_9[:, 1] == 0))]
f_p_10 = f_p_10[np.logical_not(np.logical_and(f_p_10[:, 0] == 0, f_p_10[:, 1] == 0))]
f_p_11 = f_p_11[np.logical_not(np.logical_and(f_p_11[:, 0] == 0, f_p_11[:, 1] == 0))]
f_p_12 = f_p_12[np.logical_not(np.logical_and(f_p_12[:, 0] == 0, f_p_12[:, 1] == 0))]
f_p_13 = f_p_13[np.logical_not(np.logical_and(f_p_13[:, 0] == 0, f_p_13[:, 1] == 0))]
f_p_14 = f_p_14[np.logical_not(np.logical_and(f_p_14[:, 0] == 0, f_p_14[:, 1] == 0))]
f_p_15 = f_p_15[np.logical_not(np.logical_and(f_p_15[:, 0] == 0, f_p_15[:, 1] == 0))]
f_p_16 = f_p_16[np.logical_not(np.logical_and(f_p_16[:, 0] == 0, f_p_16[:, 1] == 0))]

# ==============================================================================#

# density
density_1 = z_1 / (np.pi * 100 * radius ** 2)
density_2 = z_2 / (np.pi * 100 * radius ** 2)
density_3 = z_3 / (np.pi * 100 * radius ** 2)
density_4 = z_4 / (np.pi * 100 * radius ** 2)
density_5 = z_5 / (np.pi * 100 * radius ** 2)
density_6 = z_6 / (np.pi * 100 * radius ** 2)
density_7 = z_7 / (np.pi * 100 * radius ** 2)
density_8 = z_8 / (np.pi * 100 * radius ** 2)
density_9 = z_9 / (np.pi * 100 * radius ** 2)
density_10 = z_10 / (np.pi * 100 * radius ** 2)
density_11 = z_11 / (np.pi * 100 * radius ** 2)
density_12 = z_12 / (np.pi * 100 * radius ** 2)
density_13 = z_13 / (np.pi * 100 * radius ** 2)
density_14 = z_14 / (np.pi * 100 * radius ** 2)
density_15 = z_15 / (np.pi * 100 * radius ** 2)
density_16 = z_16 / (np.pi * 100 * radius ** 2)

# density_d = nsteps / (np.pi * 100 * radius ** 2) * 0.3

density_d = nsteps / (400* length**2)

z_map = 0

if density_1 >= density_d:
    n_1 = "memorized"
    z_map += 1
else:
    n_1 = "missing"

if density_2 >= density_d:
    n_2 = "memorized"
    z_map += 1
else:
    n_2 = "missing"

if density_3 >= density_d:
    n_3 = "memorized"
    z_map += 1
else:
    n_3 = "missing"

if density_4 >= density_d:
    n_4 = "memorized"
    z_map += 1
else:
    n_4 = "missing"

if density_5 >= density_d:
    n_5 = "memorized"
    z_map += 1
else:
    n_5 = "missing"

if density_6 >= density_d:
    n_6 = "memorized"
    z_map += 1
else:
    n_6 = "missing"

if density_7 >= density_d:
    n_7 = "memorized"
    z_map += 1
else:
    n_7 = "missing"

if density_8 >= density_d:
    n_8 = "memorized"
    z_map += 1
else:
    n_8 = "missing"

if density_9 >= density_d:
    n_9 = "memorized"
    z_map += 1
else:
    n_9 = "missing"

if density_10 >= density_d:
    n_10 = "memorized"
    z_map += 1
else:
    n_10 = "missing"

if density_11 >= density_d:
    n_11 = "memorized"
    z_map += 1
else:
    n_11 = "missing"

if density_12 >= density_d:
    n_12 = "memorized"
    z_map += 1
else:
    n_12 = "missing"

if density_13 >= density_d:
    n_13 = "memorized"
    z_map += 1
else:
    n_13 = "missing"

if density_14 >= density_d:
    n_14 = "memorized"
    z_map += 1
else:
    n_14 = "missing"

if density_15 >= density_d:
    n_15 = "memorized"
    z_map += 1
else:
    n_15 = "missing"

if density_16 >= density_d:
    n_16 = "memorized"
    z_map += 1
else:
    n_16 = "missing"


if z_map > n / 2:
    state = "memorized"
else:
    state = "missing"

# ==============================================================================#
#  plotting(cell firing)


plt.plot(x_1, y_1, 'x')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title("2D random walk \n" + str(state))
plt.show()

plt.subplot(1, 4, 1)
plt.plot(f_p_1[:, 0], f_p_1[:, 1], 'x', color='r')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')

# title

tle_1_1 = "Firing time = " + str(z_1)
tle_2_1 = "density =" + str(round(density_1,5))
tle_3_1 = "status =" + str(n_1)
tle_1 = tle_1_1 + "\n" + tle_2_1 + "\n" + tle_3_1

plt.title(str(tle_1))

plt.subplot(1, 4, 2)

plt.plot(f_p_2[:, 0], f_p_2[:, 1], 'x', color='r')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')

# title

tle_1_2 = "Firing time = " + str(z_2)
tle_2_2 = "density =" + str(round(density_2,5))
tle_3_2 = "status =" + str(n_2)
tle_2 = tle_1_2 + "\n" + tle_2_2 + "\n" + tle_3_2

plt.title(str(tle_2))

plt.subplot(1, 4, 3)

plt.plot(f_p_3[:, 0], f_p_3[:, 1], 'x', color='r')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')

# title

tle_1_3 = "Firing time = " + str(z_3)
tle_2_3 = "density =" + str(round(density_3,5))
tle_3_3 = "status =" + str(n_3)
tle_3 = tle_1_3 + "\n" + tle_2_3 + "\n" + tle_3_3

plt.title(str(tle_3))

plt.subplot(1, 4, 4)

plt.plot(f_p_4[:, 0], f_p_4[:, 1], 'x', color='r')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')

# title

tle_1_4 = "Firing time = " + str(z_4)
tle_2_4 = "density =" + str(round(density_4,5))
tle_3_4 = "status =" + str(n_4)
tle_4 = tle_1_4 + "\n" + tle_2_4 + "\n" + tle_3_4

plt.title(str(tle_4))

plt.show()

plt.subplot(1, 4, 1)

plt.plot(f_p_5[:, 0], f_p_5[:, 1], 'x', color='r')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')

# title

tle_1_5 = "Firing time = " + str(z_5)
tle_2_5 = "density =" + str(round(density_5,5))
tle_3_5 = "status =" + str(n_5)
tle_5 = tle_1_5 + "\n" + tle_2_5 + "\n" + tle_3_5

plt.title(str(tle_5))

plt.subplot(1, 4, 2)

plt.plot(f_p_6[:, 0], f_p_6[:, 1], 'x', color='r')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')

# title

tle_1_6 = "Firing time = " + str(z_6)
tle_2_6 = "density =" + str(round(density_6,5))
tle_3_6 = "status =" + str(n_6)
tle_6 = tle_1_6 + "\n" + tle_2_6 + "\n" + tle_3_6

plt.title(str(tle_6))

plt.subplot(1, 4, 3)

plt.plot(f_p_7[:, 0], f_p_7[:, 1], 'x', color='r')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')

# title

tle_1_7 = "Firing time = " + str(z_7)
tle_2_7 = "density =" + str(round(density_7,5))
tle_3_7 = "status =" + str(n_7)
tle_7 = tle_1_7 + "\n" + tle_2_7 + "\n" + tle_3_7

plt.title(str(tle_7))

plt.subplot(1, 4, 4)

plt.plot(f_p_8[:, 0], f_p_8[:, 1], 'x', color='r')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')

# title

tle_1_8 = "Firing time = " + str(z_8)
tle_2_8 = "density =" + str(round(density_8,5))
tle_3_8 = "status =" + str(n_8)
tle_8 = tle_1_8 + "\n" + tle_2_8 + "\n" + tle_3_8

plt.title(str(tle_8))

plt.show()

plt.subplot(1, 4, 1)

plt.plot(f_p_9[:, 0], f_p_9[:, 1], 'x', color='r')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')

# title

tle_1_9 = "Firing time = " + str(z_9)
tle_2_9 = "density =" + str(round(density_9,5))
tle_3_9 = "status =" + str(n_9)
tle_9 = tle_1_9 + "\n" + tle_2_9 + "\n" + tle_3_9

plt.title(str(tle_9))

plt.subplot(1, 4, 2)

plt.plot(f_p_10[:, 0], f_p_10[:, 1], 'x', color='r')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')

# title

tle_1_10 = "Firing time = " + str(z_10)
tle_2_10 = "density =" + str(round(density_10,5))
tle_3_10 = "status =" + str(n_10)
tle_10 = tle_1_10 + "\n" + tle_2_10 + "\n" + tle_3_10

plt.title(str(tle_10))

plt.subplot(1, 4, 3)

plt.plot(f_p_11[:, 0], f_p_11[:, 1], 'x', color='r')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')

# title

tle_1_11 = "Firing time = " + str(z_11)
tle_2_11 = "density =" + str(round(density_11,5))
tle_3_11 = "status =" + str(n_11)
tle_11 = tle_1_11 + "\n" + tle_2_11 + "\n" + tle_3_11

plt.title(str(tle_11))

plt.subplot(1, 4, 4)

plt.plot(f_p_12[:, 0], f_p_12[:, 1], 'x', color='r')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')

# title

tle_1_12 = "Firing time = " + str(z_12)
tle_2_12 = "density =" + str(round(density_12,5))
tle_3_12 = "status =" + str(n_12)
tle_12 = tle_1_12 + "\n" + tle_2_12 + "\n" + tle_3_12

plt.title(str(tle_12))

plt.show()

plt.subplot(1, 4, 1)

plt.plot(f_p_13[:, 0], f_p_13[:, 1], 'x', color='r')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')

# title

tle_1_13 = "Firing time = " + str(z_13)
tle_2_13 = "density =" + str(round(density_13,5))
tle_3_13 = "status =" + str(n_13)
tle_13 = tle_1_13 + "\n" + tle_2_13 + "\n" + tle_3_13

plt.title(str(tle_13))

plt.subplot(1, 4, 2)

plt.plot(f_p_14[:, 0], f_p_14[:, 1], 'x', color='r')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')

# title

tle_1_14 = "Firing time = " + str(z_14)
tle_2_14 = "density =" + str(round(density_14,5))
tle_3_14 = "status =" + str(n_14)
tle_14 = tle_1_14 + "\n" + tle_2_14 + "\n" + tle_3_14

plt.title(str(tle_14))

plt.subplot(1, 4, 3)

plt.plot(f_p_15[:, 0], f_p_15[:, 1], 'x', color='r')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')

# title

tle_1_15 = "Firing time = " + str(z_15)
tle_2_15 = "density =" + str(round(density_15,5))
tle_3_15 = "status =" + str(n_15)
tle_15 = tle_1_15 + "\n" + tle_2_15 + "\n" + tle_3_15

plt.title(str(tle_15))

plt.subplot(1, 4, 4)

plt.plot(f_p_16[:, 0], f_p_16[:, 1], 'x', color='r')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')

# title

tle_1_16 = "Firing time = " + str(z_16)
tle_2_16 = "density =" + str(round(density_16,5))
tle_3_16 = "status =" + str(n_16)
tle_16 = tle_1_16 + "\n" + tle_2_16 + "\n" + tle_3_16

plt.title(str(tle_16))

plt.show()

# ==============================================================================#
