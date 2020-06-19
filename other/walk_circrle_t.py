import cvxpy as cvx
from pylab import *
import dccp
from cffi.backend_ctypes import xrange
import numpy as np

# ==============================================================================#
# setting
nsteps = 5000

n = 16
n_srt = np.sqrt(n)
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

current_t_1 = np.zeros((n, nsteps))
current_t_2 = np.zeros((n, nsteps))
x_t = np.zeros((n, nsteps))
y_t = np.zeros((n, nsteps))
t_t = np.zeros((n, nsteps))
Amp_t = np.zeros((n, nsteps))

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
length_setion = 2 * length / (n_srt + 1)
radius = 1.5

point_x_t = np.zeros((n, 1))
point_y_t = np.zeros((n, 1))

for i in range(n):
    i = i
    variable = i // n_srt
    if variable % 2 == 0:
        point_x_t[i] = -1 * length + length_setion * (i % n_srt) + length_setion
        point_y_t[i] = length - length_setion - length_setion * (i // n_srt)
    else:
        point_x_t[i] = -1 * length + length_setion * (i % n_srt) + length_setion + (length_setion / np.sqrt(3))
        point_y_t[i] = length - length_setion - length_setion * (i // n_srt)

point_t = np.column_stack((point_x_t, point_y_t))

circ = np.linspace(0, 2 * pi)
x_border = [-length, length, length, -length, -length]
y_border = [-length, -length, length, length, -length]

plt.figure(figsize=(5, 5))
for i in xrange(n):
    plt.plot(point_t[i, 0] + radius * np.cos(circ), point_t[i, 1] + radius * np.sin(circ), 'b')

tag_1 = "Length =" + str(2 * length)
tag_2 = "Radius =" + str(radius)
tag = tag_1 + "\n" + tag_2

plt.title(str(tag))
plt.plot(x_border, y_border, 'g')
plt.axes().set_aspect('equal')
plt.xlim([-length, length])
plt.ylim([-length, length])
plt.show()

# Integrate and fire model

z_t = np.zeros((n, 1))

firing_position_x_t = np.zeros((n, nsteps))
firing_position_y_t = np.zeros((n, nsteps))

for j in range(n):
    for i in range(0, nsteps - 1):
        if np.all(np.sqrt((position[i][0] - point_x_t[j]) ** 2 + (position[i][1] - point_y_t[j]) ** 2) < radius):
            if A_random[i] > firing_prob_1:
                firing_position_x_t[j][i] = position[i][0]
                firing_position_y_t[j][i] = position[i][1]
                z_t[j] += 1
        x_t[j][i] = position[i][0]
        y_t[j][i] = position[i][1]


# ==============================================================================#

# ==============================================================================#

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

if z_map_t > n / n_srt:
    state_t = "memorized"
else:
    state_t = "missing"

# ==============================================================================#
#  plotting(cell firing)


plt.plot(x_t[1], y_t[1], 'x')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title("2D random walk \n" + str(state_t))
plt.show()

tle_1 = []
tle_2 = []
tle_3 = []

tle_t = []
plt_x_tt = []

# tle_1_1 = "Firing time = " + str(z_1)
# tle_2_1 = "density =" + str(round(density_1,5))
# tle_3_1 = "status =" + str(n_1)
# tle_1 = tle_1_1 + "\n" + tle_2_1 + "\n" + tle_3_1


plt.plot(f_p_t[:, 0], f_p_t[:, 1], 'x', color='r')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.show()

# ==============================================================================#
