import cvxpy as cvx
from pylab import *
import dccp
from cffi.backend_ctypes import xrange
import numpy as np
# ==============================================================================#
# setting

firing_prob_2 = 0.7
firing_prob_1 = 0.3

n = 36
n_srt = np.sqrt(n)
R = 30
w = 12
h = 12
A = 0.005  # A
Vr = -65  # mV
tau = 10  # ms

output_1 = 0
output_2 = 0
output_3 = 0

count_c = 0

count_e = 0

count_t = 0

# ==============================================================================#
run = 100
k_1 = 0
k_2 = 0
k_3 = 0
nsteps = 3500
a = []
# Random walk and counting
# ==============================================================================#

for j in range(run):

    A_random = np.random.random_sample(nsteps)
    position = np.zeros((nsteps, 2))

    x_t = np.zeros((n, nsteps))
    y_t = np.zeros((n, nsteps))
    x_e = np.zeros((n, nsteps))
    y_e = np.zeros((n, nsteps))
    x_c = np.zeros((n, nsteps))
    y_c = np.zeros((n, nsteps))

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

    length_setion = 2 * length / (n_srt + 1)
    radius = 1.5

    point_x_t = np.zeros((n, 1))
    point_y_t = np.zeros((n, 1))
    point_x_e = np.zeros((n, 1))
    point_y_e = np.zeros((n, 1))
    point_x_c = np.zeros((n, 1))
    point_y_c = np.zeros((n, 1))

    for i in range(n):
        i = i
        variable = i // n_srt
        if variable % 2 == 0:
            point_x_t[i] = -1 * length + length_setion * (i % n_srt) + length_setion
            point_y_t[i] = length - length_setion - length_setion * (i // n_srt)
        else:
            point_x_t[i] = -1 * length + length_setion * (i % n_srt) + length_setion + (length_setion / np.sqrt(3))
            point_y_t[i] = length - length_setion - length_setion * (i // n_srt)

    point_e = np.column_stack((point_x_e, point_y_e))
    point_t = np.column_stack((point_x_t, point_y_t))
    point_c = np.column_stack((point_x_c, point_y_c))

    z_t = np.zeros((n, 1))
    z_e = np.zeros((n, 1))
    z_c = np.zeros((n, 1))

    firing_position_x_t = np.zeros((n, nsteps))
    firing_position_y_t = np.zeros((n, nsteps))
    firing_position_x_e = np.zeros((n, nsteps))
    firing_position_y_e = np.zeros((n, nsteps))
    firing_position_x_c = np.zeros((n, nsteps))
    firing_position_y_c = np.zeros((n, nsteps))

    for j in range(n):
        for i in range(0, nsteps - 1):
            if np.all(
                    np.sqrt((position[i][0] - point_x_t[j]) ** 2 + (position[i][1] - point_y_t[j]) ** 2) < radius):
                if A_random[i] > firing_prob_1:
                    firing_position_x_t[j][i] = position[i][0]
                    firing_position_y_t[j][i] = position[i][1]
                    z_t[j] += 1
            x_t[j][i] = position[i][0]
            y_t[j][i] = position[i][1]

    # modify the firing#############################################

    firing_position_t = []

    for i in range(n):
        firing_position_t.append(np.zeros((1, int(z_t[i]))))

    f_p_x_t = firing_position_x_t.ravel()
    f_p_y_t = firing_position_y_t.ravel()

    f_p_t = np.column_stack((f_p_x_t, f_p_y_t))

    f_p_t = f_p_t[np.logical_not(np.logical_and(f_p_t[:, 0] == 0, f_p_t[:, 1] == 0))]

    # ==============================================================================#

    # density

    density_t = z_t / (np.pi * 100 * radius ** 2)

    density_d = nsteps / (400 * length ** 2) * firing_prob_2

    z_map_t = 0

    n_t = []

    for i in range(n):
        if density_t[i] >= density_d:
            n_t.append("memorized")
            z_map_t += 1
        else:
            n_t.append("missing")
    a.append(str(round(z_map_t/n, 4)))

b ={i:a.count(i) for i in a}






print(b)


# ==============================================================================#
#  plotting(cell firing)

# title_1 = "square :" + str(state_e)
# title_2 = "\n hexagonal :" + str(state_t)
# title_3 = "\n circle :" + str(state_c)
#
# title = title_1 + title_2 + title_3
#
# plt.plot(x_t[1], y_t[1], 'x')
# plt.xlabel('X-axis')
# plt.ylabel('Y-axis')
# plt.title("2D random walk \n" + str(title))
# plt.show()
#
# circ = np.linspace(0, 2 * pi)
# x_border = [-length, length, length, -length, -length]
# y_border = [-length, -length, length, length, -length]
#
# plt.figure(figsize=(5, 5))
# for i in xrange(n):
#     plt.plot(point_t[i, 0] + radius * np.cos(circ), point_t[i, 1] + radius * np.sin(circ), 'b')
#
# plt.title(" hexagonal :" + str(state_t))
# plt.plot(x_border, y_border, 'g')
# plt.axes().set_aspect('equal')
# plt.xlim([-length, length])
# plt.ylim([-length, length])
#
# l = cvx.max(cvx.max(cvx.abs(c), axis=1) + r).value * 2
# pi = np.pi
# ratio = pi * cvx.sum(cvx.square(r)).value / cvx.square(l).value
#
# circ = np.linspace(0, 2 * pi)
# x_border = [-l / 2, l / 2, l / 2, -l / 2, -l / 2]
# y_border = [-l / 2, -l / 2, l / 2, l / 2, -l / 2]
#
# plt.figure(figsize=(5, 5))
# for i in xrange(n):
#     plt.plot(c[i, 0].value + 1.5 * np.cos(circ), c[i, 1].value + 1.5 * np.sin(circ), 'b')
#
# plt.title(str("circle :" + str(state_c)))
# plt.plot(x_border, y_border, 'g')
# plt.axes().set_aspect('equal')
# plt.xlim([-l / 2, l / 2])
# plt.ylim([-l / 2, l / 2])
# plt.show()
#
# plt.figure(figsize=(5, 5))
# for i in xrange(n):
#     plt.plot(point_e[i, 0] + radius * np.cos(circ), point_e[i, 1] + radius * np.sin(circ), 'b')
#
# plt.title("square :" + str(state_e))
# plt.plot(x_border, y_border, 'g')
# plt.axes().set_aspect('equal')
# plt.xlim([-length, length])
# plt.ylim([-length, length])
#
# circ = np.linspace(0, 2 * pi)
# x_border = [-length, length, length, -length, -length]
# y_border = [-length, -length, length, length, -length]
# plt.show()

# plt.subplot(1, 3, 1)
# plt.plot(f_p_t[:, 0], f_p_t[:, 1], 'x', color='r')
# plt.xlabel('X-axis')
# plt.ylabel('Y-axis')
# plt.title("hexagonal :" + str(state_t))
#
# plt.subplot(1, 3, 2)
# plt.plot(f_p_e[:, 0], f_p_e[:, 1], 'x', color='r')
# plt.xlabel('X-axis')
# plt.ylabel('Y-axis')
# plt.title("square :" + str(state_e))
#
# plt.subplot(1, 3, 3)
# plt.plot(f_p_c[:, 0], f_p_c[:, 1], 'x', color='r')
# plt.xlabel('X-axis')
# plt.ylabel('Y-axis')
# plt.title(" circle :" + str(state_c))
# plt.show()

# ==============================================================================#
