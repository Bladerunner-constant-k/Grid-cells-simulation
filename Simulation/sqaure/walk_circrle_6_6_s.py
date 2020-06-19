import cvxpy as cvx
from pylab import *
import dccp
from cffi.backend_ctypes import xrange
import numpy as np

# ==============================================================================#
# setting
nsteps = 5000
firing_prob_1 = 0.5
firing_prob_2 = 0.9

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
current_17 = np.zeros((nsteps, 2))
current_18 = np.zeros((nsteps, 2))
current_19 = np.zeros((nsteps, 2))
current_20 = np.zeros((nsteps, 2))
current_21 = np.zeros((nsteps, 2))
current_22 = np.zeros((nsteps, 2))
current_23 = np.zeros((nsteps, 2))
current_24 = np.zeros((nsteps, 2))
current_25 = np.zeros((nsteps, 2))
current_26 = np.zeros((nsteps, 2))
current_27 = np.zeros((nsteps, 2))
current_28 = np.zeros((nsteps, 2))
current_29 = np.zeros((nsteps, 2))
current_30 = np.zeros((nsteps, 2))
current_31 = np.zeros((nsteps, 2))
current_32 = np.zeros((nsteps, 2))
current_33 = np.zeros((nsteps, 2))
current_34 = np.zeros((nsteps, 2))
current_35 = np.zeros((nsteps, 2))
current_36 = np.zeros((nsteps, 2))

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
x_17 = np.zeros((nsteps, 1))
x_18 = np.zeros((nsteps, 1))
x_19 = np.zeros((nsteps, 1))
x_20 = np.zeros((nsteps, 1))
x_21 = np.zeros((nsteps, 1))
x_22 = np.zeros((nsteps, 1))
x_23 = np.zeros((nsteps, 1))
x_24 = np.zeros((nsteps, 1))
x_25 = np.zeros((nsteps, 1))
x_26 = np.zeros((nsteps, 1))
x_27 = np.zeros((nsteps, 1))
x_28 = np.zeros((nsteps, 1))
x_29 = np.zeros((nsteps, 1))
x_30 = np.zeros((nsteps, 1))
x_31 = np.zeros((nsteps, 1))
x_32 = np.zeros((nsteps, 1))
x_33 = np.zeros((nsteps, 1))
x_34 = np.zeros((nsteps, 1))
x_35 = np.zeros((nsteps, 1))
x_36 = np.zeros((nsteps, 1))

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
y_17 = np.zeros((nsteps, 1))
y_18 = np.zeros((nsteps, 1))
y_19 = np.zeros((nsteps, 1))
y_20 = np.zeros((nsteps, 1))
y_21 = np.zeros((nsteps, 1))
y_22 = np.zeros((nsteps, 1))
y_23 = np.zeros((nsteps, 1))
y_24 = np.zeros((nsteps, 1))
y_25 = np.zeros((nsteps, 1))
y_26 = np.zeros((nsteps, 1))
y_27 = np.zeros((nsteps, 1))
y_28 = np.zeros((nsteps, 1))
y_29 = np.zeros((nsteps, 1))
y_30 = np.zeros((nsteps, 1))
y_31 = np.zeros((nsteps, 1))
y_32 = np.zeros((nsteps, 1))
y_33 = np.zeros((nsteps, 1))
y_34 = np.zeros((nsteps, 1))
y_35 = np.zeros((nsteps, 1))
y_36 = np.zeros((nsteps, 1))

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
t_17 = np.zeros((nsteps, 1))
t_18 = np.zeros((nsteps, 1))
t_19 = np.zeros((nsteps, 1))
t_20 = np.zeros((nsteps, 1))
t_21 = np.zeros((nsteps, 1))
t_22 = np.zeros((nsteps, 1))
t_23 = np.zeros((nsteps, 1))
t_24 = np.zeros((nsteps, 1))
t_25 = np.zeros((nsteps, 1))
t_26 = np.zeros((nsteps, 1))
t_27 = np.zeros((nsteps, 1))
t_28 = np.zeros((nsteps, 1))
t_29 = np.zeros((nsteps, 1))
t_30 = np.zeros((nsteps, 1))
t_31 = np.zeros((nsteps, 1))
t_32 = np.zeros((nsteps, 1))
t_33 = np.zeros((nsteps, 1))
t_34 = np.zeros((nsteps, 1))
t_35 = np.zeros((nsteps, 1))
t_36 = np.zeros((nsteps, 1))

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
Amp_17 = np.zeros((nsteps, 1))
Amp_18 = np.zeros((nsteps, 1))
Amp_19 = np.zeros((nsteps, 1))
Amp_20 = np.zeros((nsteps, 1))
Amp_21 = np.zeros((nsteps, 1))
Amp_22 = np.zeros((nsteps, 1))
Amp_23 = np.zeros((nsteps, 1))
Amp_24 = np.zeros((nsteps, 1))
Amp_25 = np.zeros((nsteps, 1))
Amp_26 = np.zeros((nsteps, 1))
Amp_27 = np.zeros((nsteps, 1))
Amp_28 = np.zeros((nsteps, 1))
Amp_29 = np.zeros((nsteps, 1))
Amp_30 = np.zeros((nsteps, 1))
Amp_31 = np.zeros((nsteps, 1))
Amp_32 = np.zeros((nsteps, 1))
Amp_33 = np.zeros((nsteps, 1))
Amp_34 = np.zeros((nsteps, 1))
Amp_35 = np.zeros((nsteps, 1))
Amp_36 = np.zeros((nsteps, 1))

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
n = 36

length_setion = 2 * length / 7
radius = 1.5

point_x = np.zeros((n, 1))
point_y = np.zeros((n, 1))

for i in range(n):
    i = i
    point_x[i] = -1 * length + length_setion * (i % 6) + length_setion
    point_y[i] = length - length_setion - length_setion * (i // 6)

point = np.column_stack((point_x, point_y))

plt.figure(figsize=(5, 5))
circ = np.linspace(0, 2 * pi)
x_border = [-length, length, length, -length, -length]
y_border = [-length, -length, length, length, -length]
for i in xrange(n):
    plt.plot(point[i, 0] + radius * np.cos(circ), point[i, 1] + radius * np.sin(circ), 'b')

tag_1 = "Length =" + str(2*length)
tag_2 = "Radius =" + str(radius)
tag = tag_1 + "\n" + tag_2

plt.title(str(tag))
plt.plot(x_border, y_border, 'g')
plt.axes().set_aspect('equal')
plt.xlim([-length, length])
plt.ylim([-length, length])
plt.show()

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
z_17 = 0
z_18 = 0
z_19 = 0
z_20 = 0
z_21 = 0
z_22 = 0
z_23 = 0
z_24 = 0
z_25 = 0
z_26 = 0
z_27 = 0
z_28 = 0
z_29 = 0
z_30 = 0
z_31 = 0
z_32 = 0
z_33 = 0
z_34 = 0
z_35 = 0
z_36 = 0

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
firing_position_x_17 = np.zeros((nsteps, 1))
firing_position_y_17 = np.zeros((nsteps, 1))
firing_position_x_18 = np.zeros((nsteps, 1))
firing_position_y_18 = np.zeros((nsteps, 1))
firing_position_x_19 = np.zeros((nsteps, 1))
firing_position_y_19 = np.zeros((nsteps, 1))
firing_position_x_20 = np.zeros((nsteps, 1))
firing_position_y_20 = np.zeros((nsteps, 1))
firing_position_x_21 = np.zeros((nsteps, 1))
firing_position_y_21 = np.zeros((nsteps, 1))
firing_position_x_22 = np.zeros((nsteps, 1))
firing_position_y_22 = np.zeros((nsteps, 1))
firing_position_x_23 = np.zeros((nsteps, 1))
firing_position_y_23 = np.zeros((nsteps, 1))
firing_position_x_24 = np.zeros((nsteps, 1))
firing_position_y_24 = np.zeros((nsteps, 1))
firing_position_x_25 = np.zeros((nsteps, 1))
firing_position_y_25 = np.zeros((nsteps, 1))
firing_position_x_26 = np.zeros((nsteps, 1))
firing_position_y_26 = np.zeros((nsteps, 1))
firing_position_x_27 = np.zeros((nsteps, 1))
firing_position_y_27 = np.zeros((nsteps, 1))
firing_position_x_28 = np.zeros((nsteps, 1))
firing_position_y_28 = np.zeros((nsteps, 1))
firing_position_x_29 = np.zeros((nsteps, 1))
firing_position_y_29 = np.zeros((nsteps, 1))
firing_position_x_30 = np.zeros((nsteps, 1))
firing_position_y_30 = np.zeros((nsteps, 1))
firing_position_x_31 = np.zeros((nsteps, 1))
firing_position_y_31 = np.zeros((nsteps, 1))
firing_position_x_32 = np.zeros((nsteps, 1))
firing_position_y_32 = np.zeros((nsteps, 1))
firing_position_x_33 = np.zeros((nsteps, 1))
firing_position_y_33 = np.zeros((nsteps, 1))
firing_position_x_34 = np.zeros((nsteps, 1))
firing_position_y_34 = np.zeros((nsteps, 1))
firing_position_x_35 = np.zeros((nsteps, 1))
firing_position_y_35 = np.zeros((nsteps, 1))
firing_position_x_36 = np.zeros((nsteps, 1))
firing_position_y_36 = np.zeros((nsteps, 1))

###  point(1)

for i in range(0, nsteps - 1):

    if np.all(np.sqrt((position[i][0] - point_x[0]) ** 2 + (position[i][1] - point_y[0]) ** 2) < radius):
        if A_random[i] > firing_prob_1:

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
        if A_random[i] > firing_prob_1:

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
        if A_random[i] > firing_prob_1:

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
        if A_random[i] > firing_prob_1:

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
        if A_random[i] > firing_prob_1:

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
        if A_random[i] > firing_prob_1:

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
        if A_random[i] > firing_prob_1:

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
        if A_random[i] > firing_prob_1:

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
        if A_random[i] > firing_prob_1:

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
        if A_random[i] > firing_prob_1:

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
        if A_random[i] > firing_prob_1:

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
        if A_random[i] > firing_prob_1:

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
        if A_random[i] > firing_prob_1:

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
        if A_random[i] > firing_prob_1:

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
        if A_random[i] > firing_prob_1:

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
        if A_random[i] > firing_prob_1:

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

###  point(17)
for i in range(0, nsteps - 1):

    if np.all(np.sqrt((position[i][0] - point_x[16]) ** 2 + (position[i][1] - point_y[16]) ** 2) < radius):
        if A_random[i] > firing_prob_1:

            current_17[i][0] = i
            current_17[i][1] = A * A_random[i]
            firing_position_x_17[i] = position[i][0]
            firing_position_y_17[i] = position[i][1]
            z_17 += 1
        else:
            current_17[i][0] = i
            current_17[i][1] = 0

    else:
        current_17[i][0] = i
        current_17[i][1] = 0

    x_17[i] = position[i][0]
    y_17[i] = position[i][1]
    t_17[i] = current_17[i][0]
    Amp_17[i] = current_17[i][1]

###  point(18)
for i in range(0, nsteps - 1):

    if np.all(np.sqrt((position[i][0] - point_x[17]) ** 2 + (position[i][1] - point_y[17]) ** 2) < radius):
        if A_random[i] > firing_prob_1:

            current_18[i][0] = i
            current_18[i][1] = A * A_random[i]
            firing_position_x_18[i] = position[i][0]
            firing_position_y_18[i] = position[i][1]
            z_18 += 1
        else:
            current_18[i][0] = i
            current_18[i][1] = 0

    else:
        current_18[i][0] = i
        current_18[i][1] = 0

    x_18[i] = position[i][0]
    y_18[i] = position[i][1]
    t_18[i] = current_18[i][0]
    Amp_18[i] = current_18[i][1]

###  point(19)
for i in range(0, nsteps - 1):

    if np.all(np.sqrt((position[i][0] - point_x[18]) ** 2 + (position[i][1] - point_y[18]) ** 2) < radius):
        if A_random[i] > firing_prob_1:

            current_19[i][0] = i
            current_19[i][1] = A * A_random[i]
            firing_position_x_19[i] = position[i][0]
            firing_position_y_19[i] = position[i][1]
            z_19 += 1
        else:
            current_19[i][0] = i
            current_19[i][1] = 0

    else:
        current_19[i][0] = i
        current_19[i][1] = 0

    x_19[i] = position[i][0]
    y_19[i] = position[i][1]
    t_19[i] = current_19[i][0]
    Amp_19[i] = current_19[i][1]

###  point(20)
for i in range(0, nsteps - 1):

    if np.all(np.sqrt((position[i][0] - point_x[19]) ** 2 + (position[i][1] - point_y[19]) ** 2) < radius):
        if A_random[i] > firing_prob_1:

            current_20[i][0] = i
            current_20[i][1] = A * A_random[i]
            firing_position_x_20[i] = position[i][0]
            firing_position_y_20[i] = position[i][1]
            z_20 += 1
        else:
            current_20[i][0] = i
            current_20[i][1] = 0

    else:
        current_20[i][0] = i
        current_20[i][1] = 0

    x_20[i] = position[i][0]
    y_20[i] = position[i][1]
    t_20[i] = current_20[i][0]
    Amp_20[i] = current_20[i][1]

###  point(21)
for i in range(0, nsteps - 1):

    if np.all(np.sqrt((position[i][0] - point_x[20]) ** 2 + (position[i][1] - point_y[20]) ** 2) < radius):
        if A_random[i] > firing_prob_1:

            current_21[i][0] = i
            current_21[i][1] = A * A_random[i]
            firing_position_x_21[i] = position[i][0]
            firing_position_y_21[i] = position[i][1]
            z_21 += 1
        else:
            current_21[i][0] = i
            current_21[i][1] = 0

    else:
        current_21[i][0] = i
        current_21[i][1] = 0

    x_21[i] = position[i][0]
    y_21[i] = position[i][1]
    t_21[i] = current_21[i][0]
    Amp_21[i] = current_21[i][1]

###  point(22)
for i in range(0, nsteps - 1):

    if np.all(np.sqrt((position[i][0] - point_x[21]) ** 2 + (position[i][1] - point_y[21]) ** 2) < radius):
        if A_random[i] > firing_prob_1:

            current_22[i][0] = i
            current_22[i][1] = A * A_random[i]
            firing_position_x_22[i] = position[i][0]
            firing_position_y_22[i] = position[i][1]
            z_22 += 1
        else:
            current_22[i][0] = i
            current_22[i][1] = 0

    else:
        current_22[i][0] = i
        current_22[i][1] = 0

    x_22[i] = position[i][0]
    y_22[i] = position[i][1]
    t_22[i] = current_22[i][0]
    Amp_22[i] = current_22[i][1]

###  point(23)
for i in range(0, nsteps - 1):

    if np.all(np.sqrt((position[i][0] - point_x[22]) ** 2 + (position[i][1] - point_y[22]) ** 2) < radius):
        if A_random[i] > firing_prob_1:

            current_23[i][0] = i
            current_23[i][1] = A * A_random[i]
            firing_position_x_23[i] = position[i][0]
            firing_position_y_23[i] = position[i][1]
            z_23 += 1
        else:
            current_23[i][0] = i
            current_23[i][1] = 0

    else:
        current_23[i][0] = i
        current_23[i][1] = 0

    x_23[i] = position[i][0]
    y_23[i] = position[i][1]
    t_23[i] = current_23[i][0]
    Amp_23[i] = current_23[i][1]

###  point(24)
for i in range(0, nsteps - 1):

    if np.all(np.sqrt((position[i][0] - point_x[23]) ** 2 + (position[i][1] - point_y[23]) ** 2) < radius):
        if A_random[i] > firing_prob_1:

            current_24[i][0] = i
            current_24[i][1] = A * A_random[i]
            firing_position_x_24[i] = position[i][0]
            firing_position_y_24[i] = position[i][1]
            z_24 += 1
        else:
            current_24[i][0] = i
            current_24[i][1] = 0

    else:
        current_24[i][0] = i
        current_24[i][1] = 0

    x_24[i] = position[i][0]
    y_24[i] = position[i][1]
    t_24[i] = current_24[i][0]
    Amp_24[i] = current_24[i][1]

###  point(25)
for i in range(0, nsteps - 1):

    if np.all(np.sqrt((position[i][0] - point_x[24]) ** 2 + (position[i][1] - point_y[24]) ** 2) < radius):
        if A_random[i] > firing_prob_1:

            current_24[i][0] = i
            current_24[i][1] = A * A_random[i]
            firing_position_x_24[i] = position[i][0]
            firing_position_y_24[i] = position[i][1]
            z_24 += 1
        else:
            current_24[i][0] = i
            current_24[i][1] = 0

    else:
        current_24[i][0] = i
        current_24[i][1] = 0

    x_24[i] = position[i][0]
    y_24[i] = position[i][1]
    t_24[i] = current_24[i][0]
    Amp_24[i] = current_24[i][1]

###  point(26)
for i in range(0, nsteps - 1):

    if np.all(np.sqrt((position[i][0] - point_x[25]) ** 2 + (position[i][1] - point_y[25]) ** 2) < radius):
        if A_random[i] > firing_prob_1:

            current_26[i][0] = i
            current_26[i][1] = A * A_random[i]
            firing_position_x_26[i] = position[i][0]
            firing_position_y_26[i] = position[i][1]
            z_26 += 1
        else:
            current_26[i][0] = i
            current_26[i][1] = 0

    else:
        current_26[i][0] = i
        current_26[i][1] = 0

    x_26[i] = position[i][0]
    y_26[i] = position[i][1]
    t_26[i] = current_26[i][0]
    Amp_26[i] = current_26[i][1]

###  point(27)
for i in range(0, nsteps - 1):

    if np.all(np.sqrt((position[i][0] - point_x[26]) ** 2 + (position[i][1] - point_y[26]) ** 2) < radius):
        if A_random[i] > firing_prob_1:

            current_27[i][0] = i
            current_27[i][1] = A * A_random[i]
            firing_position_x_27[i] = position[i][0]
            firing_position_y_27[i] = position[i][1]
            z_27 += 1
        else:
            current_27[i][0] = i
            current_27[i][1] = 0

    else:
        current_27[i][0] = i
        current_27[i][1] = 0

    x_27[i] = position[i][0]
    y_27[i] = position[i][1]
    t_27[i] = current_27[i][0]
    Amp_27[i] = current_27[i][1]

###  point(28)
for i in range(0, nsteps - 1):

    if np.all(np.sqrt((position[i][0] - point_x[27]) ** 2 + (position[i][1] - point_y[27]) ** 2) < radius):
        if A_random[i] > firing_prob_1:

            current_28[i][0] = i
            current_28[i][1] = A * A_random[i]
            firing_position_x_28[i] = position[i][0]
            firing_position_y_28[i] = position[i][1]
            z_28 += 1
        else:
            current_28[i][0] = i
            current_28[i][1] = 0

    else:
        current_28[i][0] = i
        current_28[i][1] = 0

    x_28[i] = position[i][0]
    y_28[i] = position[i][1]
    t_28[i] = current_28[i][0]
    Amp_28[i] = current_28[i][1]

###  point(29)
for i in range(0, nsteps - 1):

    if np.all(np.sqrt((position[i][0] - point_x[28]) ** 2 + (position[i][1] - point_y[28]) ** 2) < radius):
        if A_random[i] > firing_prob_1:

            current_29[i][0] = i
            current_29[i][1] = A * A_random[i]
            firing_position_x_29[i] = position[i][0]
            firing_position_y_29[i] = position[i][1]
            z_29 += 1
        else:
            current_29[i][0] = i
            current_29[i][1] = 0

    else:
        current_29[i][0] = i
        current_29[i][1] = 0

    x_29[i] = position[i][0]
    y_29[i] = position[i][1]
    t_29[i] = current_29[i][0]
    Amp_29[i] = current_29[i][1]

###  point(30)
for i in range(0, nsteps - 1):

    if np.all(np.sqrt((position[i][0] - point_x[29]) ** 2 + (position[i][1] - point_y[29]) ** 2) < radius):
        if A_random[i] > firing_prob_1:

            current_30[i][0] = i
            current_30[i][1] = A * A_random[i]
            firing_position_x_30[i] = position[i][0]
            firing_position_y_30[i] = position[i][1]
            z_30 += 1
        else:
            current_30[i][0] = i
            current_30[i][1] = 0

    else:
        current_30[i][0] = i
        current_30[i][1] = 0

    x_30[i] = position[i][0]
    y_30[i] = position[i][1]
    t_30[i] = current_30[i][0]
    Amp_30[i] = current_30[i][1]

###  point(31)
for i in range(0, nsteps - 1):

    if np.all(np.sqrt((position[i][0] - point_x[5]) ** 2 + (position[i][1] - point_y[5]) ** 2) < radius):
        if A_random[i] > firing_prob_1:

            current_31[i][0] = i
            current_31[i][1] = A * A_random[i]
            firing_position_x_31[i] = position[i][0]
            firing_position_y_31[i] = position[i][1]
            z_31 += 1
        else:
            current_31[i][0] = i
            current_31[i][1] = 0

    else:
        current_31[i][0] = i
        current_31[i][1] = 0

    x_31[i] = position[i][0]
    y_31[i] = position[i][1]
    t_31[i] = current_31[i][0]
    Amp_31[i] = current_31[i][1]

###  point(32)
for i in range(0, nsteps - 1):

    if np.all(np.sqrt((position[i][0] - point_x[6]) ** 2 + (position[i][1] - point_y[6]) ** 2) < radius):
        if A_random[i] > firing_prob_1:

            current_32[i][0] = i
            current_32[i][1] = A * A_random[i]
            firing_position_x_32[i] = position[i][0]
            firing_position_y_32[i] = position[i][1]
            z_32 += 1
        else:
            current_32[i][0] = i
            current_32[i][1] = 0

    else:
        current_32[i][0] = i
        current_32[i][1] = 0

    x_32[i] = position[i][0]
    y_32[i] = position[i][1]
    t_32[i] = current_32[i][0]
    Amp_32[i] = current_32[i][1]

###  point(33)
for i in range(0, nsteps - 1):

    if np.all(np.sqrt((position[i][0] - point_x[7]) ** 2 + (position[i][1] - point_y[7]) ** 2) < radius):
        if A_random[i] > firing_prob_1:

            current_33[i][0] = i
            current_33[i][1] = A * A_random[i]
            firing_position_x_33[i] = position[i][0]
            firing_position_y_33[i] = position[i][1]
            z_33 += 1
        else:
            current_33[i][0] = i
            current_33[i][1] = 0

    else:
        current_33[i][0] = i
        current_33[i][1] = 0

    x_33[i] = position[i][0]
    y_33[i] = position[i][1]
    t_33[i] = current_33[i][0]
    Amp_33[i] = current_33[i][1]

###  point(34)
for i in range(0, nsteps - 1):

    if np.all(np.sqrt((position[i][0] - point_x[8]) ** 2 + (position[i][1] - point_y[8]) ** 2) < radius):
        if A_random[i] > firing_prob_1:

            current_34[i][0] = i
            current_34[i][1] = A * A_random[i]
            firing_position_x_34[i] = position[i][0]
            firing_position_y_34[i] = position[i][1]
            z_34 += 1
        else:
            current_34[i][0] = i
            current_34[i][1] = 0

    else:
        current_34[i][0] = i
        current_34[i][1] = 0

    x_34[i] = position[i][0]
    y_34[i] = position[i][1]
    t_34[i] = current_34[i][0]
    Amp_34[i] = current_34[i][1]

###  point(35)
for i in range(0, nsteps - 1):

    if np.all(np.sqrt((position[i][0] - point_x[9]) ** 2 + (position[i][1] - point_y[9]) ** 2) < radius):
        if A_random[i] > firing_prob_1:

            current_35[i][0] = i
            current_35[i][1] = A * A_random[i]
            firing_position_x_35[i] = position[i][0]
            firing_position_y_35[i] = position[i][1]
            z_35 += 1
        else:
            current_35[i][0] = i
            current_35[i][1] = 0

    else:
        current_35[i][0] = i
        current_35[i][1] = 0

    x_35[i] = position[i][0]
    y_35[i] = position[i][1]
    t_35[i] = current_35[i][0]
    Amp_35[i] = current_35[i][1]

###  point(36)
for i in range(0, nsteps - 1):

    if np.all(np.sqrt((position[i][0] - point_x[10]) ** 2 + (position[i][1] - point_y[10]) ** 2) < radius):
        if A_random[i] > firing_prob_1:

            current_36[i][0] = i
            current_36[i][1] = A * A_random[i]
            firing_position_x_36[i] = position[i][0]
            firing_position_y_36[i] = position[i][1]
            z_36 += 1
        else:
            current_36[i][0] = i
            current_36[i][1] = 0

    else:
        current_36[i][0] = i
        current_36[i][1] = 0

    x_36[i] = position[i][0]
    y_36[i] = position[i][1]
    t_36[i] = current_36[i][0]
    Amp_36[i] = current_36[i][1]


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


def LIF_17(_I_17=current_17[i][1], gl=0.16, Cm=0.0049):
    # Constants
    El = -0.065  # restint membrane potential [V]
    thresh = -0.050  # spiking threshold [V]

    # Experimental Setup
    # TIME
    T_17 = nsteps - 1  # total simulation length [s]
    dt = 1  # step size [s]
    time_17 = np.arange(0, T_17 + dt, dt)  # step values [s]
    # VOLTAGE
    V_17 = np.empty(len(time_17))  # array for saving Voltage history
    V_17[0] = El  # set initial to resting potential
    # CURRENT

    I_17 = Amp_17

    # Measurements
    spikes = 0  # counter for number of spikes

    # Simulation
    for i in range(1, len(time_17)):
        # use "I - V/R = C * dV/dT" to get this equation
        dV_17 = (Amp_17[i] - gl * (V_17[i - 1] - El)) / Cm
        V_17[i] = V_17[i - 1] + dV_17 * dt

        # in case we exceed threshold
        if V_17[i] > thresh:
            V_17[i - 1] = 0.04  # set the last step to spike value
            V_17[i] = El  # current step is resting membrane potential
            spikes += 1  # count spike

    return V_17


def LIF_18(_I_18=current_18[i][1], gl=0.16, Cm=0.0049):
    # Constants
    El = -0.065  # restint membrane potential [V]
    thresh = -0.050  # spiking threshold [V]

    # Experimental Setup
    # TIME
    T_18 = nsteps - 1  # total simulation length [s]
    dt = 1  # step size [s]
    time_18 = np.arange(0, T_18 + dt, dt)  # step values [s]
    # VOLTAGE
    V_18 = np.empty(len(time_18))  # array for saving Voltage history
    V_18[0] = El  # set initial to resting potential
    # CURRENT

    I_18 = Amp_18

    # Measurements
    spikes = 0  # counter for number of spikes

    # Simulation
    for i in range(1, len(time_18)):
        # use "I - V/R = C * dV/dT" to get this equation
        dV_18 = (Amp_18[i] - gl * (V_18[i - 1] - El)) / Cm
        V_18[i] = V_18[i - 1] + dV_18 * dt

        # in case we exceed threshold
        if V_18[i] > thresh:
            V_18[i - 1] = 0.04  # set the last step to spike value
            V_18[i] = El  # current step is resting membrane potential
            spikes += 1  # count spike

    return V_18


def LIF_19(_I_19=current_19[i][1], gl=0.16, Cm=0.0049):
    # Constants
    El = -0.065  # restint membrane potential [V]
    thresh = -0.050  # spiking threshold [V]

    # Experimental Setup
    # TIME
    T_19 = nsteps - 1  # total simulation length [s]
    dt = 1  # step size [s]
    time_19 = np.arange(0, T_19 + dt, dt)  # step values [s]
    # VOLTAGE
    V_19 = np.empty(len(time_19))  # array for saving Voltage history
    V_19[0] = El  # set initial to resting potential
    # CURRENT

    I_19 = Amp_19

    # Measurements
    spikes = 0  # counter for number of spikes

    # Simulation
    for i in range(1, len(time_19)):
        # use "I - V/R = C * dV/dT" to get this equation
        dV_19 = (Amp_19[i] - gl * (V_19[i - 1] - El)) / Cm
        V_19[i] = V_19[i - 1] + dV_19 * dt

        # in case we exceed threshold
        if V_19[i] > thresh:
            V_19[i - 1] = 0.04  # set the last step to spike value
            V_19[i] = El  # current step is resting membrane potential
            spikes += 1  # count spike

    return V_19


def LIF_20(_I_20=current_20[i][1], gl=0.16, Cm=0.0049):
    # Constants
    El = -0.065  # restint membrane potential [V]
    thresh = -0.050  # spiking threshold [V]

    # Experimental Setup
    # TIME
    T_20 = nsteps - 1  # total simulation length [s]
    dt = 1  # step size [s]
    time_20 = np.arange(0, T_20 + dt, dt)  # step values [s]
    # VOLTAGE
    V_20 = np.empty(len(time_20))  # array for saving Voltage history
    V_20[0] = El  # set initial to resting potential
    # CURRENT

    I_20 = Amp_20

    # Measurements
    spikes = 0  # counter for number of spikes

    # Simulation
    for i in range(1, len(time_20)):
        # use "I - V/R = C * dV/dT" to get this equation
        dV_20 = (Amp_20[i] - gl * (V_20[i - 1] - El)) / Cm
        V_20[i] = V_20[i - 1] + dV_20 * dt

        # in case we exceed threshold
        if V_20[i] > thresh:
            V_20[i - 1] = 0.04  # set the last step to spike value
            V_20[i] = El  # current step is resting membrane potential
            spikes += 1  # count spike

    return V_20


def LIF_21(_I_21=current_21[i][1], gl=0.16, Cm=0.0049):
    # Constants
    El = -0.065  # restint membrane potential [V]
    thresh = -0.050  # spiking threshold [V]

    # Experimental Setup
    # TIME
    T_21 = nsteps - 1  # total simulation length [s]
    dt = 1  # step size [s]
    time_21 = np.arange(0, T_21 + dt, dt)  # step values [s]
    # VOLTAGE
    V_21 = np.empty(len(time_21))  # array for saving Voltage history
    V_21[0] = El  # set initial to resting potential
    # CURRENT

    I_21 = Amp_21

    # Measurements
    spikes = 0  # counter for number of spikes

    # Simulation
    for i in range(1, len(time_21)):
        # use "I - V/R = C * dV/dT" to get this equation
        dV_21 = (Amp_21[i] - gl * (V_21[i - 1] - El)) / Cm
        V_21[i] = V_21[i - 1] + dV_21 * dt

        # in case we exceed threshold
        if V_21[i] > thresh:
            V_21[i - 1] = 0.04  # set the last step to spike value
            V_21[i] = El  # current step is resting membrane potential
            spikes += 1  # count spike

    return V_21


def LIF_22(_I_22=current_22[i][1], gl=0.16, Cm=0.0049):
    # Constants
    El = -0.065  # restint membrane potential [V]
    thresh = -0.050  # spiking threshold [V]

    # Experimental Setup
    # TIME
    T_22 = nsteps - 1  # total simulation length [s]
    dt = 1  # step size [s]
    time_22 = np.arange(0, T_22 + dt, dt)  # step values [s]
    # VOLTAGE
    V_22 = np.empty(len(time_22))  # array for saving Voltage history
    V_22[0] = El  # set initial to resting potential
    # CURRENT

    I_22 = Amp_22

    # Measurements
    spikes = 0  # counter for number of spikes

    # Simulation
    for i in range(1, len(time_22)):
        # use "I - V/R = C * dV/dT" to get this equation
        dV_22 = (Amp_22[i] - gl * (V_22[i - 1] - El)) / Cm
        V_22[i] = V_22[i - 1] + dV_22 * dt

        # in case we exceed threshold
        if V_22[i] > thresh:
            V_22[i - 1] = 0.04  # set the last step to spike value
            V_22[i] = El  # current step is resting membrane potential
            spikes += 1  # count spike

    return V_22


def LIF_23(_I_23=current_23[i][1], gl=0.16, Cm=0.0049):
    # Constants
    El = -0.065  # restint membrane potential [V]
    thresh = -0.050  # spiking threshold [V]

    # Experimental Setup
    # TIME
    T_23 = nsteps - 1  # total simulation length [s]
    dt = 1  # step size [s]
    time_23 = np.arange(0, T_23 + dt, dt)  # step values [s]
    # VOLTAGE
    V_23 = np.empty(len(time_23))  # array for saving Voltage history
    V_23[0] = El  # set initial to resting potential
    # CURRENT

    I_23 = Amp_23

    # Measurements
    spikes = 0  # counter for number of spikes

    # Simulation
    for i in range(1, len(time_23)):
        # use "I - V/R = C * dV/dT" to get this equation
        dV_23 = (Amp_23[i] - gl * (V_23[i - 1] - El)) / Cm
        V_23[i] = V_23[i - 1] + dV_23 * dt

        # in case we exceed threshold
        if V_23[i] > thresh:
            V_23[i - 1] = 0.04  # set the last step to spike value
            V_23[i] = El  # current step is resting membrane potential
            spikes += 1  # count spike

    return V_23


def LIF_24(_I_24=current_24[i][1], gl=0.16, Cm=0.0049):
    # Constants
    El = -0.065  # restint membrane potential [V]
    thresh = -0.050  # spiking threshold [V]

    # Experimental Setup
    # TIME
    T_24 = nsteps - 1  # total simulation length [s]
    dt = 1  # step size [s]
    time_24 = np.arange(0, T_24 + dt, dt)  # step values [s]
    # VOLTAGE
    V_24 = np.empty(len(time_24))  # array for saving Voltage history
    V_24[0] = El  # set initial to resting potential
    # CURRENT

    I_24 = Amp_24

    # Measurements
    spikes = 0  # counter for number of spikes

    # Simulation
    for i in range(1, len(time_24)):
        # use "I - V/R = C * dV/dT" to get this equation
        dV_24 = (Amp_24[i] - gl * (V_24[i - 1] - El)) / Cm
        V_24[i] = V_24[i - 1] + dV_24 * dt

        # in case we exceed threshold
        if V_24[i] > thresh:
            V_24[i - 1] = 0.04  # set the last step to spike value
            V_24[i] = El  # current step is resting membrane potential
            spikes += 1  # count spike

    return V_24


def LIF_25(_I_25=current_25[i][1], gl=0.16, Cm=0.0049):
    # Constants
    El = -0.065  # restint membrane potential [V]
    thresh = -0.050  # spiking threshold [V]

    # Experimental Setup
    # TIME
    T_25 = nsteps - 1  # total simulation length [s]
    dt = 1  # step size [s]
    time_25 = np.arange(0, T_25 + dt, dt)  # step values [s]
    # VOLTAGE
    V_25 = np.empty(len(time_25))  # array for saving Voltage history
    V_25[0] = El  # set initial to resting potential
    # CURRENT

    I_25 = Amp_25

    # Measurements
    spikes = 0  # counter for number of spikes

    # Simulation
    for i in range(1, len(time_25)):
        # use "I - V/R = C * dV/dT" to get this equation
        dV_25 = (Amp_25[i] - gl * (V_25[i - 1] - El)) / Cm
        V_25[i] = V_25[i - 1] + dV_25 * dt

        # in case we exceed threshold
        if V_25[i] > thresh:
            V_25[i - 1] = 0.04  # set the last step to spike value
            V_25[i] = El  # current step is resting membrane potential
            spikes += 1  # count spike

    return V_25


def LIF_26(_I_26=current_26[i][1], gl=0.16, Cm=0.0049):
    # Constants
    El = -0.065  # restint membrane potential [V]
    thresh = -0.050  # spiking threshold [V]

    # Experimental Setup
    # TIME
    T_26 = nsteps - 1  # total simulation length [s]
    dt = 1  # step size [s]
    time_26 = np.arange(0, T_26 + dt, dt)  # step values [s]
    # VOLTAGE
    V_26 = np.empty(len(time_26))  # array for saving Voltage history
    V_26[0] = El  # set initial to resting potential
    # CURRENT

    I_26 = Amp_26

    # Measurements
    spikes = 0  # counter for number of spikes

    # Simulation
    for i in range(1, len(time_26)):
        # use "I - V/R = C * dV/dT" to get this equation
        dV_26 = (Amp_26[i] - gl * (V_26[i - 1] - El)) / Cm
        V_26[i] = V_26[i - 1] + dV_26 * dt

        # in case we exceed threshold
        if V_26[i] > thresh:
            V_26[i - 1] = 0.04  # set the last step to spike value
            V_26[i] = El  # current step is resting membrane potential
            spikes += 1  # count spike

    return V_26


def LIF_27(_I_27=current_27[i][1], gl=0.16, Cm=0.0049):
    # Constants
    El = -0.065  # restint membrane potential [V]
    thresh = -0.050  # spiking threshold [V]

    # Experimental Setup
    # TIME
    T_27 = nsteps - 1  # total simulation length [s]
    dt = 1  # step size [s]
    time_27 = np.arange(0, T_27 + dt, dt)  # step values [s]
    # VOLTAGE
    V_27 = np.empty(len(time_27))  # array for saving Voltage history
    V_27[0] = El  # set initial to resting potential
    # CURRENT

    I_27 = Amp_27

    # Measurements
    spikes = 0  # counter for number of spikes

    # Simulation
    for i in range(1, len(time_27)):
        # use "I - V/R = C * dV/dT" to get this equation
        dV_27 = (Amp_27[i] - gl * (V_27[i - 1] - El)) / Cm
        V_27[i] = V_27[i - 1] + dV_27 * dt

        # in case we exceed threshold
        if V_27[i] > thresh:
            V_27[i - 1] = 0.04  # set the last step to spike value
            V_27[i] = El  # current step is resting membrane potential
            spikes += 1  # count spike

    return V_27


def LIF_28(_I_28=current_28[i][1], gl=0.16, Cm=0.0049):
    # Constants
    El = -0.065  # restint membrane potential [V]
    thresh = -0.050  # spiking threshold [V]

    # Experimental Setup
    # TIME
    T_28 = nsteps - 1  # total simulation length [s]
    dt = 1  # step size [s]
    time_28 = np.arange(0, T_28 + dt, dt)  # step values [s]
    # VOLTAGE
    V_28 = np.empty(len(time_28))  # array for saving Voltage history
    V_28[0] = El  # set initial to resting potential
    # CURRENT

    I_28 = Amp_28

    # Measurements
    spikes = 0  # counter for number of spikes

    # Simulation
    for i in range(1, len(time_28)):
        # use "I - V/R = C * dV/dT" to get this equation
        dV_28 = (Amp_28[i] - gl * (V_28[i - 1] - El)) / Cm
        V_28[i] = V_28[i - 1] + dV_28 * dt

        # in case we exceed threshold
        if V_28[i] > thresh:
            V_28[i - 1] = 0.04  # set the last step to spike value
            V_28[i] = El  # current step is resting membrane potential
            spikes += 1  # count spike

    return V_28


def LIF_29(_I_29=current_29[i][1], gl=0.16, Cm=0.0049):
    # Constants
    El = -0.065  # restint membrane potential [V]
    thresh = -0.050  # spiking threshold [V]

    # Experimental Setup
    # TIME
    T_29 = nsteps - 1  # total simulation length [s]
    dt = 1  # step size [s]
    time_29 = np.arange(0, T_29 + dt, dt)  # step values [s]
    # VOLTAGE
    V_29 = np.empty(len(time_29))  # array for saving Voltage history
    V_29[0] = El  # set initial to resting potential
    # CURRENT

    I_29 = Amp_29

    # Measurements
    spikes = 0  # counter for number of spikes

    # Simulation
    for i in range(1, len(time_29)):
        # use "I - V/R = C * dV/dT" to get this equation
        dV_29 = (Amp_29[i] - gl * (V_29[i - 1] - El)) / Cm
        V_29[i] = V_29[i - 1] + dV_29 * dt

        # in case we exceed threshold
        if V_29[i] > thresh:
            V_29[i - 1] = 0.04  # set the last step to spike value
            V_29[i] = El  # current step is resting membrane potential
            spikes += 1  # count spike

    return V_29


def LIF_30(_I_30=current_30[i][1], gl=0.16, Cm=0.0049):
    # Constants
    El = -0.065  # restint membrane potential [V]
    thresh = -0.050  # spiking threshold [V]

    # Experimental Setup
    # TIME
    T_30 = nsteps - 1  # total simulation length [s]
    dt = 1  # step size [s]
    time_30 = np.arange(0, T_30 + dt, dt)  # step values [s]
    # VOLTAGE
    V_30 = np.empty(len(time_30))  # array for saving Voltage history
    V_30[0] = El  # set initial to resting potential
    # CURRENT

    I_30 = Amp_30

    # Measurements
    spikes = 0  # counter for number of spikes

    # Simulation
    for i in range(1, len(time_30)):
        # use "I - V/R = C * dV/dT" to get this equation
        dV_30 = (Amp_30[i] - gl * (V_30[i - 1] - El)) / Cm
        V_30[i] = V_30[i - 1] + dV_30 * dt

        # in case we exceed threshold
        if V_30[i] > thresh:
            V_30[i - 1] = 0.04  # set the last step to spike value
            V_30[i] = El  # current step is resting membrane potential
            spikes += 1  # count spike

    return V_30


def LIF_31(_I_31=current_31[i][1], gl=0.16, Cm=0.0049):
    # Constants
    El = -0.065  # restint membrane potential [V]
    thresh = -0.050  # spiking threshold [V]

    # Experimental Setup
    # TIME
    T_31 = nsteps - 1  # total simulation length [s]
    dt = 1  # step size [s]
    time_31 = np.arange(0, T_31 + dt, dt)  # step values [s]
    # VOLTAGE
    V_31 = np.empty(len(time_31))  # array for saving Voltage history
    V_31[0] = El  # set initial to resting potential
    # CURRENT

    I_31 = Amp_31

    # Measurements
    spikes = 0  # counter for number of spikes

    # Simulation
    for i in range(1, len(time_31)):
        # use "I - V/R = C * dV/dT" to get this equation
        dV_31 = (Amp_31[i] - gl * (V_31[i - 1] - El)) / Cm
        V_31[i] = V_31[i - 1] + dV_31 * dt

        # in case we exceed threshold
        if V_31[i] > thresh:
            V_31[i - 1] = 0.04  # set the last step to spike value
            V_31[i] = El  # current step is resting membrane potential
            spikes += 1  # count spike

    return V_31


def LIF_32(_I_32=current_32[i][1], gl=0.16, Cm=0.0049):
    # Constants
    El = -0.065  # restint membrane potential [V]
    thresh = -0.050  # spiking threshold [V]

    # Experimental Setup
    # TIME
    T_32 = nsteps - 1  # total simulation length [s]
    dt = 1  # step size [s]
    time_32 = np.arange(0, T_32 + dt, dt)  # step values [s]
    # VOLTAGE
    V_32 = np.empty(len(time_32))  # array for saving Voltage history
    V_32[0] = El  # set initial to resting potential
    # CURRENT

    I_32 = Amp_32

    # Measurements
    spikes = 0  # counter for number of spikes

    # Simulation
    for i in range(1, len(time_32)):
        # use "I - V/R = C * dV/dT" to get this equation
        dV_32 = (Amp_32[i] - gl * (V_32[i - 1] - El)) / Cm
        V_32[i] = V_32[i - 1] + dV_32 * dt

        # in case we exceed threshold
        if V_32[i] > thresh:
            V_32[i - 1] = 0.04  # set the last step to spike value
            V_32[i] = El  # current step is resting membrane potential
            spikes += 1  # count spike

    return V_32


def LIF_33(_I_33=current_33[i][1], gl=0.16, Cm=0.0049):
    # Constants
    El = -0.065  # restint membrane potential [V]
    thresh = -0.050  # spiking threshold [V]

    # Experimental Setup
    # TIME
    T_33 = nsteps - 1  # total simulation length [s]
    dt = 1  # step size [s]
    time_33 = np.arange(0, T_33 + dt, dt)  # step values [s]
    # VOLTAGE
    V_33 = np.empty(len(time_33))  # array for saving Voltage history
    V_33[0] = El  # set initial to resting potential
    # CURRENT

    I_33 = Amp_33

    # Measurements
    spikes = 0  # counter for number of spikes

    # Simulation
    for i in range(1, len(time_33)):
        # use "I - V/R = C * dV/dT" to get this equation
        dV_33 = (Amp_33[i] - gl * (V_33[i - 1] - El)) / Cm
        V_33[i] = V_33[i - 1] + dV_33 * dt

        # in case we exceed threshold
        if V_33[i] > thresh:
            V_33[i - 1] = 0.04  # set the last step to spike value
            V_33[i] = El  # current step is resting membrane potential
            spikes += 1  # count spike

    return V_33


def LIF_34(_I_34=current_34[i][1], gl=0.16, Cm=0.0049):
    # Constants
    El = -0.065  # restint membrane potential [V]
    thresh = -0.050  # spiking threshold [V]

    # Experimental Setup
    # TIME
    T_34 = nsteps - 1  # total simulation length [s]
    dt = 1  # step size [s]
    time_34 = np.arange(0, T_34 + dt, dt)  # step values [s]
    # VOLTAGE
    V_34 = np.empty(len(time_34))  # array for saving Voltage history
    V_34[0] = El  # set initial to resting potential
    # CURRENT

    I_34 = Amp_34

    # Measurements
    spikes = 0  # counter for number of spikes

    # Simulation
    for i in range(1, len(time_34)):
        # use "I - V/R = C * dV/dT" to get this equation
        dV_34 = (Amp_34[i] - gl * (V_34[i - 1] - El)) / Cm
        V_34[i] = V_34[i - 1] + dV_34 * dt

        # in case we exceed threshold
        if V_34[i] > thresh:
            V_34[i - 1] = 0.04  # set the last step to spike value
            V_34[i] = El  # current step is resting membrane potential
            spikes += 1  # count spike

    return V_34


def LIF_35(_I_35=current_35[i][1], gl=0.16, Cm=0.0049):
    # Constants
    El = -0.065  # restint membrane potential [V]
    thresh = -0.050  # spiking threshold [V]

    # Experimental Setup
    # TIME
    T_35 = nsteps - 1  # total simulation length [s]
    dt = 1  # step size [s]
    time_35 = np.arange(0, T_35 + dt, dt)  # step values [s]
    # VOLTAGE
    V_35 = np.empty(len(time_35))  # array for saving Voltage history
    V_35[0] = El  # set initial to resting potential
    # CURRENT

    I_35 = Amp_35

    # Measurements
    spikes = 0  # counter for number of spikes

    # Simulation
    for i in range(1, len(time_35)):
        # use "I - V/R = C * dV/dT" to get this equation
        dV_35 = (Amp_35[i] - gl * (V_35[i - 1] - El)) / Cm
        V_35[i] = V_35[i - 1] + dV_35 * dt

        # in case we exceed threshold
        if V_35[i] > thresh:
            V_35[i - 1] = 0.04  # set the last step to spike value
            V_35[i] = El  # current step is resting membrane potential
            spikes += 1  # count spike

    return V_35


def LIF_36(_I_36=current_36[i][1], gl=0.16, Cm=0.0049):
    # Constants
    El = -0.065  # restint membrane potential [V]
    thresh = -0.050  # spiking threshold [V]

    # Experimental Setup
    # TIME
    T_36 = nsteps - 1  # total simulation length [s]
    dt = 1  # step size [s]
    time_36 = np.arange(0, T_36 + dt, dt)  # step values [s]
    # VOLTAGE
    V_36 = np.empty(len(time_36))  # array for saving Voltage history
    V_36[0] = El  # set initial to resting potential
    # CURRENT

    I_36 = Amp_36

    # Measurements
    spikes = 0  # counter for number of spikes

    # Simulation
    for i in range(1, len(time_36)):
        # use "I - V/R = C * dV/dT" to get this equation
        dV_36 = (Amp_36[i] - gl * (V_36[i - 1] - El)) / Cm
        V_36[i] = V_36[i - 1] + dV_36 * dt

        # in case we exceed threshold
        if V_36[i] > thresh:
            V_36[i - 1] = 0.04  # set the last step to spike value
            V_36[i] = El  # current step is resting membrane potential
            spikes += 1  # count spike

    return V_36


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


def I_values_17(_I_17=current_17[i][1]):
    I_17 = Amp_17
    return I_17


def I_values_18(_I_18=current_18[i][1]):
    I_18 = Amp_18
    return I_18


def I_values_19(_I_19=current_19[i][1]):
    I_19 = Amp_19
    return I_19


def I_values_20(_I_20=current_20[i][1]):
    I_20 = Amp_20
    return I_20


def I_values_21(_I_21=current_21[i][1]):
    I_21 = Amp_21
    return I_21


def I_values_22(_I_22=current_22[i][1]):
    I_22 = Amp_22
    return I_22


def I_values_23(_I_23=current_23[i][1]):
    I_23 = Amp_23
    return I_23


def I_values_24(_I_24=current_24[i][1]):
    I_24 = Amp_24
    return I_24


def I_values_25(_I_25=current_25[i][1]):
    I_25 = Amp_25
    return I_25


def I_values_26(_I_26=current_26[i][1]):
    I_26 = Amp_26
    return I_26


def I_values_27(_I_27=current_27[i][1]):
    I_27 = Amp_27
    return I_27


def I_values_28(_I_28=current_28[i][1]):
    I_28 = Amp_28
    return I_28


def I_values_29(_I_29=current_29[i][1]):
    I_29 = Amp_29
    return I_29


def I_values_30(_I_30=current_30[i][1]):
    I_30 = Amp_30
    return I_30


def I_values_31(_I_31=current_31[i][1]):
    I_31 = Amp_31
    return I_31


def I_values_32(_I_32=current_32[i][1]):
    I_32 = Amp_32
    return I_32


def I_values_33(_I_33=current_33[i][1]):
    I_33 = Amp_33
    return I_33


def I_values_34(_I_34=current_34[i][1]):
    I_34 = Amp_34
    return I_34


def I_values_35(_I_35=current_35[i][1]):
    I_35 = Amp_35
    return I_35


def I_values_36(_I_36=current_36[i][1]):
    I_36 = Amp_36
    return I_36


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
T_17 = nsteps - 1  # total simulation length [s]
T_18 = nsteps - 1  # total simulation length [s]
T_19 = nsteps - 1  # total simulation length [s]
T_20 = nsteps - 1  # total simulation length [s]
T_21 = nsteps - 1  # total simulation length [s]
T_22 = nsteps - 1  # total simulation length [s]
T_23 = nsteps - 1  # total simulation length [s]
T_24 = nsteps - 1  # total simulation length [s]
T_25 = nsteps - 1  # total simulation length [s]
T_26 = nsteps - 1  # total simulation length [s]
T_27 = nsteps - 1  # total simulation length [s]
T_28 = nsteps - 1  # total simulation length [s]
T_29 = nsteps - 1  # total simulation length [s]
T_30 = nsteps - 1  # total simulation length [s]
T_31 = nsteps - 1  # total simulation length [s]
T_32 = nsteps - 1  # total simulation length [s]
T_33 = nsteps - 1  # total simulation length [s]
T_34 = nsteps - 1  # total simulation length [s]
T_35 = nsteps - 1  # total simulation length [s]
T_36 = nsteps - 1  # total simulation length [s]

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
time_17 = np.arange(0, T_17 + dt, dt)  # step values [s]
time_18 = np.arange(0, T_18 + dt, dt)  # step values [s]
time_19 = np.arange(0, T_19 + dt, dt)  # step values [s]
time_20 = np.arange(0, T_20 + dt, dt)  # step values [s]
time_21 = np.arange(0, T_21 + dt, dt)  # step values [s]
time_22 = np.arange(0, T_22 + dt, dt)  # step values [s]
time_23 = np.arange(0, T_23 + dt, dt)  # step values [s]
time_24 = np.arange(0, T_24 + dt, dt)  # step values [s]
time_25 = np.arange(0, T_25 + dt, dt)  # step values [s]
time_26 = np.arange(0, T_26 + dt, dt)  # step values [s]
time_27 = np.arange(0, T_27 + dt, dt)  # step values [s]
time_28 = np.arange(0, T_28 + dt, dt)  # step values [s]
time_29 = np.arange(0, T_29 + dt, dt)  # step values [s]
time_30 = np.arange(0, T_30 + dt, dt)  # step values [s]
time_31 = np.arange(0, T_31 + dt, dt)  # step values [s]
time_32 = np.arange(0, T_32 + dt, dt)  # step values [s]
time_33 = np.arange(0, T_33 + dt, dt)  # step values [s]
time_34 = np.arange(0, T_34 + dt, dt)  # step values [s]
time_35 = np.arange(0, T_35 + dt, dt)  # step values [s]
time_36 = np.arange(0, T_36 + dt, dt)  # step values [s]

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

V_17 = LIF_17(_I_17=I_init, gl=gl_init, Cm=Cm_init)
I_17 = I_values_17(_I_17=I_init)

V_18 = LIF_18(_I_18=I_init, gl=gl_init, Cm=Cm_init)
I_18 = I_values_18(_I_18=I_init)

V_19 = LIF_19(_I_19=I_init, gl=gl_init, Cm=Cm_init)
I_19 = I_values_19(_I_19=I_init)

V_20 = LIF_20(_I_20=I_init, gl=gl_init, Cm=Cm_init)
I_20 = I_values_20(_I_20=I_init)

V_21 = LIF_21(_I_21=I_init, gl=gl_init, Cm=Cm_init)
I_21 = I_values_21(_I_21=I_init)

V_22 = LIF_22(_I_22=I_init, gl=gl_init, Cm=Cm_init)
I_22 = I_values_22(_I_22=I_init)

V_23 = LIF_23(_I_23=I_init, gl=gl_init, Cm=Cm_init)
I_23 = I_values_23(_I_23=I_init)

V_24 = LIF_24(_I_24=I_init, gl=gl_init, Cm=Cm_init)
I_24 = I_values_24(_I_24=I_init)

V_25 = LIF_25(_I_25=I_init, gl=gl_init, Cm=Cm_init)
I_25 = I_values_25(_I_25=I_init)

V_26 = LIF_26(_I_26=I_init, gl=gl_init, Cm=Cm_init)
I_26 = I_values_26(_I_26=I_init)

V_27 = LIF_27(_I_27=I_init, gl=gl_init, Cm=Cm_init)
I_27 = I_values_27(_I_27=I_init)

V_28 = LIF_28(_I_28=I_init, gl=gl_init, Cm=Cm_init)
I_28 = I_values_28(_I_28=I_init)

V_29 = LIF_29(_I_29=I_init, gl=gl_init, Cm=Cm_init)
I_29 = I_values_29(_I_29=I_init)

V_30 = LIF_30(_I_30=I_init, gl=gl_init, Cm=Cm_init)
I_30 = I_values_30(_I_30=I_init)
V_31 = LIF_31(_I_31=I_init, gl=gl_init, Cm=Cm_init)
I_31 = I_values_31(_I_31=I_init)

V_32 = LIF_32(_I_32=I_init, gl=gl_init, Cm=Cm_init)
I_32 = I_values_32(_I_32=I_init)

V_33 = LIF_33(_I_33=I_init, gl=gl_init, Cm=Cm_init)
I_33 = I_values_33(_I_33=I_init)

V_34 = LIF_34(_I_34=I_init, gl=gl_init, Cm=Cm_init)
I_34 = I_values_34(_I_34=I_init)

V_35 = LIF_35(_I_35=I_init, gl=gl_init, Cm=Cm_init)
I_35 = I_values_35(_I_35=I_init)

V_36 = LIF_36(_I_36=I_init, gl=gl_init, Cm=Cm_init)
I_36 = I_values_36(_I_36=I_init)

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

######### Plotting

axis_color = 'lightgoldenrodyellow'

fig = plt.figure("Leaky Integrate-and-Fire Neuron", figsize=(17, 7))
ax = fig.add_subplot(3, 1, 1)
plt.title("Interactive Leaky Integrate-and-Fire Neuron Simulation_17")
fig.subplots_adjust(left=0.1, bottom=0.32)

# plot lines
line = plt.plot(time_17, V_17, label="Membrane Potential")[0]
line2 = plt.plot(time_17, I_17, label="Applied Current")[0]

# add legend
plt.legend(loc="upper right")

# add axis labels
plt.ylabel("Potential [V]/ Current [A]")
plt.xlabel("Time [s]")

axis_color = 'lightgoldenrodyellow'

fig = plt.figure("Leaky Integrate-and-Fire Neuron", figsize=(14, 7))
ax = fig.add_subplot(3, 1, 3)
plt.title("Interactive Leaky Integrate-and-Fire Neuron Simulation_18")
fig.subplots_adjust(left=0.1, bottom=0.32)

# plot lines
line = plt.plot(time_18, V_18, label="Membrane Potential")[0]
line2 = plt.plot(time_18, I_18, label="Applied Current")[0]

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
plt.title("Interactive Leaky Integrate-and-Fire Neuron Simulation_19")
fig.subplots_adjust(left=0.1, bottom=0.32)

# plot lines
line = plt.plot(time_19, V_19, label="Membrane Potential")[0]
line2 = plt.plot(time_19, I_19, label="Applied Current")[0]

# add legend
plt.legend(loc="upper right")

# add axis labels
plt.ylabel("Potential [V]/ Current [A]")
plt.xlabel("Time [s]")

axis_color = 'lightgoldenrodyellow'

fig = plt.figure("Leaky Integrate-and-Fire Neuron", figsize=(14, 7))
ax = fig.add_subplot(3, 1, 3)
plt.title("Interactive Leaky Integrate-and-Fire Neuron Simulation_20")
fig.subplots_adjust(left=0.1, bottom=0.32)

# plot lines
line = plt.plot(time_20, V_20, label="Membrane Potential")[0]
line2 = plt.plot(time_20, I_20, label="Applied Current")[0]

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
plt.title("Interactive Leaky Integrate-and-Fire Neuron Simulation_21")
fig.subplots_adjust(left=0.1, bottom=0.32)

# plot lines
line = plt.plot(time_21, V_21, label="Membrane Potential")[0]
line2 = plt.plot(time_21, I_21, label="Applied Current")[0]

# add legend
plt.legend(loc="upper right")

# add axis labels
plt.ylabel("Potential [V]/ Current [A]")
plt.xlabel("Time [s]")

axis_color = 'lightgoldenrodyellow'

fig = plt.figure("Leaky Integrate-and-Fire Neuron", figsize=(14, 7))
ax = fig.add_subplot(3, 1, 3)
plt.title("Interactive Leaky Integrate-and-Fire Neuron Simulation_22")
fig.subplots_adjust(left=0.1, bottom=0.32)

# plot lines
line = plt.plot(time_22, V_22, label="Membrane Potential")[0]
line2 = plt.plot(time_22, I_22, label="Applied Current")[0]

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
plt.title("Interactive Leaky Integrate-and-Fire Neuron Simulation_23")
fig.subplots_adjust(left=0.1, bottom=0.32)

# plot lines
line = plt.plot(time_23, V_23, label="Membrane Potential")[0]
line2 = plt.plot(time_23, I_23, label="Applied Current")[0]

# add legend
plt.legend(loc="upper right")

# add axis labels
plt.ylabel("Potential [V]/ Current [A]")
plt.xlabel("Time [s]")

axis_color = 'lightgoldenrodyellow'

fig = plt.figure("Leaky Integrate-and-Fire Neuron", figsize=(14, 7))
ax = fig.add_subplot(3, 1, 3)
plt.title("Interactive Leaky Integrate-and-Fire Neuron Simulation_24")
fig.subplots_adjust(left=0.1, bottom=0.32)

# plot lines
line = plt.plot(time_24, V_24, label="Membrane Potential")[0]
line2 = plt.plot(time_24, I_24, label="Applied Current")[0]

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
plt.title("Interactive Leaky Integrate-and-Fire Neuron Simulation_25")
fig.subplots_adjust(left=0.1, bottom=0.32)

# plot lines
line = plt.plot(time_25, V_25, label="Membrane Potential")[0]
line2 = plt.plot(time_25, I_25, label="Applied Current")[0]

# add legend
plt.legend(loc="upper right")

# add axis labels
plt.ylabel("Potential [V]/ Current [A]")
plt.xlabel("Time [s]")

axis_color = 'lightgoldenrodyellow'

fig = plt.figure("Leaky Integrate-and-Fire Neuron", figsize=(14, 7))
ax = fig.add_subplot(3, 1, 3)
plt.title("Interactive Leaky Integrate-and-Fire Neuron Simulation_26")
fig.subplots_adjust(left=0.1, bottom=0.32)

# plot lines
line = plt.plot(time_26, V_26, label="Membrane Potential")[0]
line2 = plt.plot(time_26, I_26, label="Applied Current")[0]

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
plt.title("Interactive Leaky Integrate-and-Fire Neuron Simulation_27")
fig.subplots_adjust(left=0.1, bottom=0.32)

# plot lines
line = plt.plot(time_27, V_27, label="Membrane Potential")[0]
line2 = plt.plot(time_27, I_27, label="Applied Current")[0]

# add legend
plt.legend(loc="upper right")

# add axis labels
plt.ylabel("Potential [V]/ Current [A]")
plt.xlabel("Time [s]")

axis_color = 'lightgoldenrodyellow'

fig = plt.figure("Leaky Integrate-and-Fire Neuron", figsize=(14, 7))
ax = fig.add_subplot(3, 1, 3)
plt.title("Interactive Leaky Integrate-and-Fire Neuron Simulation_28")
fig.subplots_adjust(left=0.1, bottom=0.32)

# plot lines
line = plt.plot(time_28, V_28, label="Membrane Potential")[0]
line2 = plt.plot(time_28, I_28, label="Applied Current")[0]

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
plt.title("Interactive Leaky Integrate-and-Fire Neuron Simulation_29")
fig.subplots_adjust(left=0.1, bottom=0.32)

# plot lines
line = plt.plot(time_29, V_29, label="Membrane Potential")[0]
line2 = plt.plot(time_29, I_29, label="Applied Current")[0]

# add legend
plt.legend(loc="upper right")

# add axis labels
plt.ylabel("Potential [V]/ Current [A]")
plt.xlabel("Time [s]")

axis_color = 'lightgoldenrodyellow'

fig = plt.figure("Leaky Integrate-and-Fire Neuron", figsize=(14, 7))
ax = fig.add_subplot(3, 1, 3)
plt.title("Interactive Leaky Integrate-and-Fire Neuron Simulation_30")
fig.subplots_adjust(left=0.1, bottom=0.32)

# plot lines
line = plt.plot(time_30, V_30, label="Membrane Potential")[0]
line2 = plt.plot(time_30, I_30, label="Applied Current")[0]

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
plt.title("Interactive Leaky Integrate-and-Fire Neuron Simulation_31")
fig.subplots_adjust(left=0.1, bottom=0.32)

# plot lines
line = plt.plot(time_31, V_31, label="Membrane Potential")[0]
line2 = plt.plot(time_31, I_31, label="Applied Current")[0]

# add legend
plt.legend(loc="upper right")

# add axis labels
plt.ylabel("Potential [V]/ Current [A]")
plt.xlabel("Time [s]")

axis_color = 'lightgoldenrodyellow'

fig = plt.figure("Leaky Integrate-and-Fire Neuron", figsize=(14, 7))
ax = fig.add_subplot(3, 1, 3)
plt.title("Interactive Leaky Integrate-and-Fire Neuron Simulation_32")
fig.subplots_adjust(left=0.1, bottom=0.32)

# plot lines
line = plt.plot(time_32, V_32, label="Membrane Potential")[0]
line2 = plt.plot(time_32, I_32, label="Applied Current")[0]

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
plt.title("Interactive Leaky Integrate-and-Fire Neuron Simulation_33")
fig.subplots_adjust(left=0.1, bottom=0.32)

# plot lines
line = plt.plot(time_33, V_33, label="Membrane Potential")[0]
line2 = plt.plot(time_33, I_33, label="Applied Current")[0]

# add legend
plt.legend(loc="upper right")

# add axis labels
plt.ylabel("Potential [V]/ Current [A]")
plt.xlabel("Time [s]")

axis_color = 'lightgoldenrodyellow'

fig = plt.figure("Leaky Integrate-and-Fire Neuron", figsize=(14, 7))
ax = fig.add_subplot(3, 1, 3)
plt.title("Interactive Leaky Integrate-and-Fire Neuron Simulation_34")
fig.subplots_adjust(left=0.1, bottom=0.32)

# plot lines
line = plt.plot(time_34, V_34, label="Membrane Potential")[0]
line2 = plt.plot(time_34, I_34, label="Applied Current")[0]

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
plt.title("Interactive Leaky Integrate-and-Fire Neuron Simulation_35")
fig.subplots_adjust(left=0.1, bottom=0.32)

# plot lines
line = plt.plot(time_35, V_35, label="Membrane Potential")[0]
line2 = plt.plot(time_35, I_35, label="Applied Current")[0]

# add legend
plt.legend(loc="upper right")

# add axis labels
plt.ylabel("Potential [V]/ Current [A]")
plt.xlabel("Time [s]")

axis_color = 'lightgoldenrodyellow'

fig = plt.figure("Leaky Integrate-and-Fire Neuron", figsize=(14, 7))
ax = fig.add_subplot(3, 1, 3)
plt.title("Interactive Leaky Integrate-and-Fire Neuron Simulation_36")
fig.subplots_adjust(left=0.1, bottom=0.32)

# plot lines
line = plt.plot(time_36, V_36, label="Membrane Potential")[0]
line2 = plt.plot(time_36, I_36, label="Applied Current")[0]

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
firing_position_17 = np.zeros((z_17, 1))
firing_position_18 = np.zeros((z_18, 1))
firing_position_19 = np.zeros((z_19, 1))
firing_position_20 = np.zeros((z_20, 1))
firing_position_21 = np.zeros((z_21, 1))
firing_position_22 = np.zeros((z_22, 1))
firing_position_23 = np.zeros((z_23, 1))
firing_position_24 = np.zeros((z_24, 1))
firing_position_25 = np.zeros((z_25, 1))
firing_position_26 = np.zeros((z_26, 1))
firing_position_27 = np.zeros((z_27, 1))
firing_position_28 = np.zeros((z_28, 1))
firing_position_29 = np.zeros((z_29, 1))
firing_position_30 = np.zeros((z_30, 1))
firing_position_31 = np.zeros((z_31, 1))
firing_position_32 = np.zeros((z_32, 1))
firing_position_33 = np.zeros((z_33, 1))
firing_position_34 = np.zeros((z_34, 1))
firing_position_35 = np.zeros((z_35, 1))
firing_position_36 = np.zeros((z_36, 1))

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
f_p_x_17 = firing_position_x_17.ravel()
f_p_y_17 = firing_position_y_17.ravel()
f_p_x_18 = firing_position_x_18.ravel()
f_p_y_18 = firing_position_y_18.ravel()
f_p_x_19 = firing_position_x_19.ravel()
f_p_y_19 = firing_position_y_19.ravel()
f_p_x_20 = firing_position_x_20.ravel()
f_p_y_20 = firing_position_y_20.ravel()
f_p_x_21 = firing_position_x_21.ravel()
f_p_y_21 = firing_position_y_21.ravel()
f_p_x_22 = firing_position_x_22.ravel()
f_p_y_22 = firing_position_y_22.ravel()
f_p_x_23 = firing_position_x_23.ravel()
f_p_y_23 = firing_position_y_23.ravel()
f_p_x_24 = firing_position_x_24.ravel()
f_p_y_24 = firing_position_y_24.ravel()
f_p_x_25 = firing_position_x_25.ravel()
f_p_y_25 = firing_position_y_25.ravel()
f_p_x_26 = firing_position_x_26.ravel()
f_p_y_26 = firing_position_y_26.ravel()
f_p_x_27 = firing_position_x_27.ravel()
f_p_y_27 = firing_position_y_27.ravel()
f_p_x_28 = firing_position_x_28.ravel()
f_p_y_28 = firing_position_y_28.ravel()
f_p_x_29 = firing_position_x_29.ravel()
f_p_y_29 = firing_position_y_29.ravel()
f_p_x_30 = firing_position_x_30.ravel()
f_p_y_30 = firing_position_y_30.ravel()
f_p_x_31 = firing_position_x_31.ravel()
f_p_y_31 = firing_position_y_31.ravel()
f_p_x_32 = firing_position_x_32.ravel()
f_p_y_32 = firing_position_y_32.ravel()
f_p_x_33 = firing_position_x_33.ravel()
f_p_y_33 = firing_position_y_33.ravel()
f_p_x_34 = firing_position_x_34.ravel()
f_p_y_34 = firing_position_y_34.ravel()
f_p_x_35 = firing_position_x_35.ravel()
f_p_y_35 = firing_position_y_35.ravel()
f_p_x_36 = firing_position_x_36.ravel()
f_p_y_36 = firing_position_y_36.ravel()

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
f_p_17 = np.column_stack((f_p_x_17, f_p_y_17))
f_p_18 = np.column_stack((f_p_x_18, f_p_y_18))
f_p_19 = np.column_stack((f_p_x_19, f_p_y_19))
f_p_20 = np.column_stack((f_p_x_20, f_p_y_20))
f_p_21 = np.column_stack((f_p_x_21, f_p_y_21))
f_p_22 = np.column_stack((f_p_x_22, f_p_y_22))
f_p_23 = np.column_stack((f_p_x_23, f_p_y_23))
f_p_24 = np.column_stack((f_p_x_24, f_p_y_24))
f_p_25 = np.column_stack((f_p_x_25, f_p_y_25))
f_p_26 = np.column_stack((f_p_x_26, f_p_y_26))
f_p_27 = np.column_stack((f_p_x_27, f_p_y_27))
f_p_28 = np.column_stack((f_p_x_28, f_p_y_28))
f_p_29 = np.column_stack((f_p_x_29, f_p_y_29))
f_p_30 = np.column_stack((f_p_x_30, f_p_y_30))
f_p_31 = np.column_stack((f_p_x_31, f_p_y_31))
f_p_32 = np.column_stack((f_p_x_32, f_p_y_32))
f_p_33 = np.column_stack((f_p_x_33, f_p_y_33))
f_p_34 = np.column_stack((f_p_x_34, f_p_y_34))
f_p_35 = np.column_stack((f_p_x_35, f_p_y_35))
f_p_36 = np.column_stack((f_p_x_36, f_p_y_36))

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
f_p_17 = f_p_17[np.logical_not(np.logical_and(f_p_17[:, 0] == 0, f_p_17[:, 1] == 0))]
f_p_18 = f_p_18[np.logical_not(np.logical_and(f_p_18[:, 0] == 0, f_p_18[:, 1] == 0))]
f_p_19 = f_p_19[np.logical_not(np.logical_and(f_p_19[:, 0] == 0, f_p_19[:, 1] == 0))]
f_p_20 = f_p_20[np.logical_not(np.logical_and(f_p_20[:, 0] == 0, f_p_20[:, 1] == 0))]
f_p_21 = f_p_21[np.logical_not(np.logical_and(f_p_21[:, 0] == 0, f_p_21[:, 1] == 0))]
f_p_22 = f_p_22[np.logical_not(np.logical_and(f_p_22[:, 0] == 0, f_p_22[:, 1] == 0))]
f_p_23 = f_p_23[np.logical_not(np.logical_and(f_p_23[:, 0] == 0, f_p_23[:, 1] == 0))]
f_p_24 = f_p_24[np.logical_not(np.logical_and(f_p_24[:, 0] == 0, f_p_24[:, 1] == 0))]
f_p_25 = f_p_25[np.logical_not(np.logical_and(f_p_25[:, 0] == 0, f_p_25[:, 1] == 0))]
f_p_26 = f_p_26[np.logical_not(np.logical_and(f_p_26[:, 0] == 0, f_p_26[:, 1] == 0))]
f_p_27 = f_p_27[np.logical_not(np.logical_and(f_p_27[:, 0] == 0, f_p_27[:, 1] == 0))]
f_p_28 = f_p_28[np.logical_not(np.logical_and(f_p_28[:, 0] == 0, f_p_28[:, 1] == 0))]
f_p_29 = f_p_29[np.logical_not(np.logical_and(f_p_29[:, 0] == 0, f_p_29[:, 1] == 0))]
f_p_30 = f_p_30[np.logical_not(np.logical_and(f_p_30[:, 0] == 0, f_p_30[:, 1] == 0))]
f_p_31 = f_p_31[np.logical_not(np.logical_and(f_p_31[:, 0] == 0, f_p_31[:, 1] == 0))]
f_p_32 = f_p_32[np.logical_not(np.logical_and(f_p_32[:, 0] == 0, f_p_32[:, 1] == 0))]
f_p_33 = f_p_33[np.logical_not(np.logical_and(f_p_33[:, 0] == 0, f_p_33[:, 1] == 0))]
f_p_34 = f_p_34[np.logical_not(np.logical_and(f_p_34[:, 0] == 0, f_p_34[:, 1] == 0))]
f_p_35 = f_p_35[np.logical_not(np.logical_and(f_p_35[:, 0] == 0, f_p_35[:, 1] == 0))]
f_p_36 = f_p_36[np.logical_not(np.logical_and(f_p_36[:, 0] == 0, f_p_36[:, 1] == 0))]

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
density_17 = z_17 / (np.pi * 100 * radius ** 2)
density_18 = z_18 / (np.pi * 100 * radius ** 2)
density_19 = z_19 / (np.pi * 100 * radius ** 2)
density_20 = z_20 / (np.pi * 100 * radius ** 2)
density_21 = z_21 / (np.pi * 100 * radius ** 2)
density_22 = z_22 / (np.pi * 100 * radius ** 2)
density_23 = z_23 / (np.pi * 100 * radius ** 2)
density_24 = z_24 / (np.pi * 100 * radius ** 2)
density_25 = z_25 / (np.pi * 100 * radius ** 2)
density_26 = z_26 / (np.pi * 100 * radius ** 2)
density_27 = z_27 / (np.pi * 100 * radius ** 2)
density_28 = z_28 / (np.pi * 100 * radius ** 2)
density_29 = z_29 / (np.pi * 100 * radius ** 2)
density_30 = z_30 / (np.pi * 100 * radius ** 2)
density_31 = z_31 / (np.pi * 100 * radius ** 2)
density_32 = z_32 / (np.pi * 100 * radius ** 2)
density_33 = z_33 / (np.pi * 100 * radius ** 2)
density_34 = z_34 / (np.pi * 100 * radius ** 2)
density_35 = z_35 / (np.pi * 100 * radius ** 2)
density_36 = z_36 / (np.pi * 100 * radius ** 2)

density_d = nsteps / (400* length**2) * firing_prob_2

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

if density_17 >= density_d:
    n_17 = "memorized"
    z_map += 1
else:
    n_17 = "missing"

if density_18 >= density_d:
    n_18 = "memorized"
    z_map += 1
else:
    n_18 = "missing"

if density_19 >= density_d:
    n_19 = "memorized"
    z_map += 1
else:
    n_19 = "missing"

if density_20 >= density_d:
    n_20 = "memorized"
    z_map += 1
else:
    n_20 = "missing"

if density_21 >= density_d:
    n_21 = "memorized"
    z_map += 1
else:
    n_21 = "missing"

if density_22 >= density_d:
    n_22 = "memorized"
    z_map += 1
else:
    n_22 = "missing"

if density_23 >= density_d:
    n_23 = "memorized"
    z_map += 1
else:
    n_23 = "missing"

if density_24 >= density_d:
    n_24 = "memorized"
    z_map += 1
else:
    n_24 = "missing"

if density_25 >= density_d:
    n_25 = "memorized"
    z_map += 1
else:
    n_25 = "missing"

if density_26 >= density_d:
    n_26 = "memorized"
    z_map += 1
else:
    n_26 = "missing"

if density_27 >= density_d:
    n_27 = "memorized"
    z_map += 1
else:
    n_27 = "missing"

if density_28 >= density_d:
    n_28 = "memorized"
    z_map += 1
else:
    n_28 = "missing"

if density_29 >= density_d:
    n_29 = "memorized"
    z_map += 1
else:
    n_29 = "missing"

if density_30 >= density_d:
    n_30 = "memorized"
    z_map += 1
else:
    n_30 = "missing"

if density_31 >= density_d:
    n_31 = "memorized"
    z_map += 1
else:
    n_31 = "missing"

if density_32 >= density_d:
    n_32 = "memorized"
    z_map += 1
else:
    n_32 = "missing"

if density_33 >= density_d:
    n_33 = "memorized"
    z_map += 1
else:
    n_33 = "missing"

if density_34 >= density_d:
    n_34 = "memorized"
    z_map += 1
else:
    n_34 = "missing"

if density_35 >= density_d:
    n_35 = "memorized"
    z_map += 1
else:
    n_35 = "missing"

if density_36 >= density_d:
    n_36 = "memorized"
    z_map += 1
else:
    n_36 = "missing"

if z_map > n/3:
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

plt.subplot(1, 6, 1)
plt.plot(f_p_1[:, 0], f_p_1[:, 1], 'x', color='r')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')

# title

tle_1_1 = "Firing time = " + str(z_1)
tle_2_1 = "density =" + str(round(density_1,5))
tle_3_1 = "status =" + str(n_1)
tle_1 = tle_1_1 + "\n" + tle_2_1 + "\n" + tle_3_1

plt.title(str(tle_1))

plt.subplot(1, 6, 2)

plt.plot(f_p_2[:, 0], f_p_2[:, 1], 'x', color='r')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')

# title

tle_1_2 = "Firing time = " + str(z_2)
tle_2_2 = "density =" + str(round(density_2,5))
tle_3_2 = "status =" + str(n_2)
tle_2 = tle_1_2 + "\n" + tle_2_2 + "\n" + tle_3_2

plt.title(str(tle_2))

plt.subplot(1, 6, 3)

plt.plot(f_p_3[:, 0], f_p_3[:, 1], 'x', color='r')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')

# title

tle_1_3 = "Firing time = " + str(z_3)
tle_2_3 = "density =" + str(round(density_3,5))
tle_3_3 = "status =" + str(n_3)
tle_3 = tle_1_3 + "\n" + tle_2_3 + "\n" + tle_3_3

plt.title(str(tle_3))

plt.subplot(1, 6, 4)

plt.plot(f_p_4[:, 0], f_p_4[:, 1], 'x', color='r')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')

# title

tle_1_4 = "Firing time = " + str(z_4)
tle_2_4 = "density =" + str(round(density_4,5))
tle_3_4 = "status =" + str(n_4)
tle_4 = tle_1_4 + "\n" + tle_2_4 + "\n" + tle_3_4

plt.title(str(tle_4))

plt.subplot(1, 6, 5)

plt.plot(f_p_5[:, 0], f_p_5[:, 1], 'x', color='r')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')

# title

tle_1_5 = "Firing time = " + str(z_5)
tle_2_5 = "density =" + str(round(density_5,5))
tle_3_5 = "status =" + str(n_5)
tle_5 = tle_1_5 + "\n" + tle_2_5 + "\n" + tle_3_5

plt.title(str(tle_5))

plt.subplot(1, 6, 6)

plt.plot(f_p_6[:, 0], f_p_6[:, 1], 'x', color='r')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')

# title

tle_1_6 = "Firing time = " + str(z_6)
tle_2_6 = "density =" + str(round(density_6,5))
tle_3_6 = "status =" + str(n_6)
tle_6 = tle_1_6 + "\n" + tle_2_6 + "\n" + tle_3_6

plt.title(str(tle_6))

plt.show()

plt.subplot(1, 6, 1)

plt.plot(f_p_7[:, 0], f_p_7[:, 1], 'x', color='r')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')

# title

tle_1_7 = "Firing time = " + str(z_7)
tle_2_7 = "density =" + str(round(density_7,5))
tle_3_7 = "status =" + str(n_7)
tle_7 = tle_1_7 + "\n" + tle_2_7 + "\n" + tle_3_7

plt.title(str(tle_7))

plt.subplot(1, 6, 2)

plt.plot(f_p_8[:, 0], f_p_8[:, 1], 'x', color='r')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')

# title

tle_1_8 = "Firing time = " + str(z_8)
tle_2_8 = "density =" + str(round(density_8,5))
tle_3_8 = "status =" + str(n_8)
tle_8 = tle_1_8 + "\n" + tle_2_8 + "\n" + tle_3_8

plt.title(str(tle_8))

plt.subplot(1, 6, 3)

plt.plot(f_p_9[:, 0], f_p_9[:, 1], 'x', color='r')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')

# title

tle_1_9 = "Firing time = " + str(z_9)
tle_2_9 = "density =" + str(round(density_9,5))
tle_3_9 = "status =" + str(n_9)
tle_9 = tle_1_9 + "\n" + tle_2_9 + "\n" + tle_3_9

plt.title(str(tle_9))

plt.subplot(1, 6, 4)

plt.plot(f_p_10[:, 0], f_p_10[:, 1], 'x', color='r')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')

# title

tle_1_10 = "Firing time = " + str(z_10)
tle_2_10 = "density =" + str(round(density_10,5))
tle_3_10 = "status =" + str(n_10)
tle_10 = tle_1_10 + "\n" + tle_2_10 + "\n" + tle_3_10

plt.title(str(tle_10))

plt.subplot(1, 6, 5)

plt.plot(f_p_11[:, 0], f_p_11[:, 1], 'x', color='r')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')

# title

tle_1_11 = "Firing time = " + str(z_11)
tle_2_11 = "density =" + str(round(density_11,5))
tle_3_11 = "status =" + str(n_11)
tle_11 = tle_1_11 + "\n" + tle_2_11 + "\n" + tle_3_11

plt.title(str(tle_11))

plt.subplot(1, 6, 6)

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

plt.subplot(1, 6, 1)

plt.plot(f_p_13[:, 0], f_p_13[:, 1], 'x', color='r')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')

# title

tle_1_13 = "Firing time = " + str(z_13)
tle_2_13 = "density =" + str(round(density_13,5))
tle_3_13 = "status =" + str(n_13)
tle_13 = tle_1_13 + "\n" + tle_2_13 + "\n" + tle_3_13

plt.title(str(tle_13))

plt.subplot(1, 6, 2)

plt.plot(f_p_14[:, 0], f_p_14[:, 1], 'x', color='r')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')

# title

tle_1_14 = "Firing time = " + str(z_14)
tle_2_14 = "density =" + str(round(density_14,5))
tle_3_14 = "status =" + str(n_14)
tle_14 = tle_1_14 + "\n" + tle_2_14 + "\n" + tle_3_14

plt.title(str(tle_14))

plt.subplot(1, 6, 3)

plt.plot(f_p_15[:, 0], f_p_15[:, 1], 'x', color='r')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')

# title

tle_1_15 = "Firing time = " + str(z_15)
tle_2_15 = "density =" + str(round(density_15,5))
tle_3_15 = "status =" + str(n_15)
tle_15 = tle_1_15 + "\n" + tle_2_15 + "\n" + tle_3_15

plt.title(str(tle_15))

plt.subplot(1, 6, 4)

plt.plot(f_p_16[:, 0], f_p_16[:, 1], 'x', color='r')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')

# title

tle_1_16 = "Firing time = " + str(z_16)
tle_2_16 = "density =" + str(round(density_16,5))
tle_3_16 = "status =" + str(n_16)
tle_16 = tle_1_16 + "\n" + tle_2_16 + "\n" + tle_3_16

plt.title(str(tle_16))

plt.subplot(1, 6, 5)

plt.plot(f_p_17[:, 0], f_p_17[:, 1], 'x', color='r')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')

# title

tle_1_17 = "Firing time = " + str(z_17)
tle_2_17 = "density =" + str(round(density_17,5))
tle_3_17 = "status =" + str(n_17)
tle_17 = tle_1_17 + "\n" + tle_2_17 + "\n" + tle_3_17

plt.title(str(tle_17))

plt.subplot(1, 6, 6)

plt.plot(f_p_18[:, 0], f_p_18[:, 1], 'x', color='r')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')

# title

tle_1_18 = "Firing time = " + str(z_18)
tle_2_18 = "density =" + str(round(density_18,5))
tle_3_18 = "status =" + str(n_18)
tle_18 = tle_1_18 + "\n" + tle_2_18 + "\n" + tle_3_18

plt.title(str(tle_18))

plt.show()

plt.subplot(1, 6, 1)

plt.plot(f_p_19[:, 0], f_p_19[:, 1], 'x', color='r')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')

# title

tle_1_19 = "Firing time = " + str(z_19)
tle_2_19 = "density =" + str(round(density_19,5))
tle_3_19 = "status =" + str(n_19)
tle_19 = tle_1_19 + "\n" + tle_2_19 + "\n" + tle_3_19

plt.title(str(tle_19))

plt.subplot(1, 6, 2)

plt.plot(f_p_20[:, 0], f_p_20[:, 1], 'x', color='r')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')

# title

tle_1_20 = "Firing time = " + str(z_20)
tle_2_20 = "density =" + str(round(density_20,5))
tle_3_20 = "status =" + str(n_20)
tle_20 = tle_1_20 + "\n" + tle_2_20 + "\n" + tle_3_20

plt.title(str(tle_20))

plt.subplot(1, 6, 3)

plt.plot(f_p_21[:, 0], f_p_21[:, 1], 'x', color='r')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')

# title

tle_1_21 = "Firing time = " + str(z_21)
tle_2_21 = "density =" + str(round(density_21,5))
tle_3_21 = "status =" + str(n_21)
tle_21 = tle_1_21 + "\n" + tle_2_21 + "\n" + tle_3_21

plt.title(str(tle_21))

plt.subplot(1, 6, 4)

plt.plot(f_p_22[:, 0], f_p_22[:, 1], 'x', color='r')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')

# title

tle_1_22 = "Firing time = " + str(z_22)
tle_2_22 = "density =" + str(round(density_22,5))
tle_3_22 = "status =" + str(n_22)
tle_22 = tle_1_22 + "\n" + tle_2_22 + "\n" + tle_3_22

plt.title(str(tle_22))

plt.subplot(1, 6, 5)

plt.plot(f_p_23[:, 0], f_p_23[:, 1], 'x', color='r')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')

# title

tle_1_23 = "Firing time = " + str(z_23)
tle_2_23 = "density =" + str(round(density_23,5))
tle_3_23 = "status =" + str(n_23)
tle_23 = tle_1_23 + "\n" + tle_2_23 + "\n" + tle_3_23

plt.title(str(tle_23))

plt.subplot(1, 6, 6)

plt.plot(f_p_24[:, 0], f_p_24[:, 1], 'x', color='r')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')

# title

tle_1_24 = "Firing time = " + str(z_24)
tle_2_24 = "density =" + str(round(density_24,5))
tle_3_24 = "status =" + str(n_24)
tle_24 = tle_1_24 + "\n" + tle_2_24 + "\n" + tle_3_24

plt.title(str(tle_24))

plt.show()

plt.subplot(1, 6, 1)

plt.plot(f_p_25[:, 0], f_p_25[:, 1], 'x', color='r')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')

# title

tle_1_25 = "Firing time = " + str(z_25)
tle_2_25 = "density =" + str(round(density_25,5))
tle_3_25 = "status =" + str(n_25)
tle_25 = tle_1_25 + "\n" + tle_2_25 + "\n" + tle_3_25

plt.title(str(tle_25))

plt.subplot(1, 6, 2)

plt.plot(f_p_26[:, 0], f_p_26[:, 1], 'x', color='r')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')

# title

tle_1_26 = "Firing time = " + str(z_26)
tle_2_26 = "density =" + str(round(density_26,5))
tle_3_26 = "status =" + str(n_26)
tle_26 = tle_1_26 + "\n" + tle_2_26 + "\n" + tle_3_26

plt.title(str(tle_26))

plt.subplot(1, 6, 3)

plt.plot(f_p_27[:, 0], f_p_27[:, 1], 'x', color='r')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')

# title

tle_1_27 = "Firing time = " + str(z_27)
tle_2_27 = "density =" + str(round(density_27,5))
tle_3_27 = "status =" + str(n_27)
tle_27 = tle_1_27 + "\n" + tle_2_27 + "\n" + tle_3_27

plt.title(str(tle_27))

plt.subplot(1, 6, 4)

plt.plot(f_p_28[:, 0], f_p_28[:, 1], 'x', color='r')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')

# title

tle_1_28 = "Firing time = " + str(z_28)
tle_2_28 = "density =" + str(round(density_28,5))
tle_3_28 = "status =" + str(n_28)
tle_28 = tle_1_28 + "\n" + tle_2_28 + "\n" + tle_3_28

plt.title(str(tle_28))

plt.subplot(1, 6, 5)

plt.plot(f_p_29[:, 0], f_p_29[:, 1], 'x', color='r')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')

# title

tle_1_29 = "Firing time = " + str(z_29)
tle_2_29 = "density =" + str(round(density_29,5))
tle_3_29 = "status =" + str(n_29)
tle_29 = tle_1_29 + "\n" + tle_2_29 + "\n" + tle_3_29

plt.title(str(tle_29))

plt.subplot(1, 6, 6)

plt.plot(f_p_30[:, 0], f_p_30[:, 1], 'x', color='r')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')

# title

tle_1_30 = "Firing time = " + str(z_30)
tle_2_30 = "density =" + str(round(density_30,5))
tle_3_30 = "status =" + str(n_30)
tle_30 = tle_1_30 + "\n" + tle_2_30 + "\n" + tle_3_30

plt.title(str(tle_30))

plt.show()

plt.subplot(1, 6, 1)

plt.plot(f_p_31[:, 0], f_p_31[:, 1], 'x', color='r')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')

# title

tle_1_31 = "Firing time = " + str(z_31)
tle_2_31 = "density =" + str(round(density_31,5))
tle_3_31 = "status =" + str(n_31)
tle_31 = tle_1_31 + "\n" + tle_2_31 + "\n" + tle_3_31

plt.title(str(tle_31))

plt.subplot(1, 6, 2)

plt.plot(f_p_32[:, 0], f_p_32[:, 1], 'x', color='r')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')

# title

tle_1_32 = "Firing time = " + str(z_32)
tle_2_32 = "density =" + str(round(density_32,5))
tle_3_32 = "status =" + str(n_32)
tle_32 = tle_1_32 + "\n" + tle_2_32 + "\n" + tle_3_32

plt.title(str(tle_32))

plt.subplot(1, 6, 3)

plt.plot(f_p_33[:, 0], f_p_33[:, 1], 'x', color='r')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')

# title

tle_1_33 = "Firing time = " + str(z_33)
tle_2_33 = "density =" + str(round(density_33,5))
tle_3_33 = "status =" + str(n_33)
tle_33 = tle_1_33 + "\n" + tle_2_33 + "\n" + tle_3_33

plt.title(str(tle_33))

plt.subplot(1, 6, 4)

plt.plot(f_p_34[:, 0], f_p_34[:, 1], 'x', color='r')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')

# title

tle_1_34 = "Firing time = " + str(z_34)
tle_2_34 = "density =" + str(round(density_34,5))
tle_3_34 = "status =" + str(n_34)
tle_34 = tle_1_34 + "\n" + tle_2_34 + "\n" + tle_3_34

plt.title(str(tle_34))

plt.subplot(1, 6, 5)

plt.plot(f_p_35[:, 0], f_p_35[:, 1], 'x', color='r')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')

# title

tle_1_35 = "Firing time = " + str(z_35)
tle_2_35 = "density =" + str(round(density_35,5))
tle_3_35 = "status =" + str(n_35)
tle_35 = tle_1_35 + "\n" + tle_2_35 + "\n" + tle_3_35

plt.title(str(tle_35))

plt.subplot(1, 6, 6)

plt.plot(f_p_36[:, 0], f_p_36[:, 1], 'x', color='r')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')

# title

tle_1_36 = "Firing time = " + str(z_36)
tle_2_36 = "density =" + str(round(density_36,5))
tle_3_36 = "status =" + str(n_36)
tle_36 = tle_1_36 + "\n" + tle_2_36 + "\n" + tle_3_36

plt.title(str(tle_36))

plt.show()
# ==============================================================================#
