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

x_1 = np.zeros((nsteps, 1))
x_2 = np.zeros((nsteps, 1))
x_3 = np.zeros((nsteps, 1))
x_4 = np.zeros((nsteps, 1))

y_1 = np.zeros((nsteps, 1))
y_2 = np.zeros((nsteps, 1))
y_3 = np.zeros((nsteps, 1))
y_4 = np.zeros((nsteps, 1))

t_1 = np.zeros((nsteps, 1))
t_2 = np.zeros((nsteps, 1))
t_3 = np.zeros((nsteps, 1))
t_4 = np.zeros((nsteps, 1))

Amp_1 = np.zeros((nsteps, 1))
Amp_2 = np.zeros((nsteps, 1))
Amp_3 = np.zeros((nsteps, 1))
Amp_4 = np.zeros((nsteps, 1))

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
n = 4
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

firing_position_x_1 = np.zeros((nsteps, 1))
firing_position_y_1 = np.zeros((nsteps, 1))
firing_position_x_2 = np.zeros((nsteps, 1))
firing_position_y_2 = np.zeros((nsteps, 1))
firing_position_x_3 = np.zeros((nsteps, 1))
firing_position_y_3 = np.zeros((nsteps, 1))
firing_position_x_4 = np.zeros((nsteps, 1))
firing_position_y_4 = np.zeros((nsteps, 1))

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


# ==============================================================================#
# time parameters for plotting
dt = 1  # step size [s]

T_1 = nsteps - 1  # total simulation length [s]
T_2 = nsteps - 1  # total simulation length [s]
T_3 = nsteps - 1  # total simulation length [s]
T_4 = nsteps - 1  # total simulation length [s]

time_1 = np.arange(0, T_1 + dt, dt)  # step values [s]
time_2 = np.arange(0, T_2 + dt, dt)  # step values [s]
time_3 = np.arange(0, T_3 + dt, dt)  # step values [s]
time_4 = np.arange(0, T_4 + dt, dt)  # step values [s]

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

# ==============================================================================#

# modify the firing#############################################

firing_position_1 = np.zeros((z_1, 1))
firing_position_2 = np.zeros((z_2, 1))
firing_position_3 = np.zeros((z_3, 1))
firing_position_4 = np.zeros((z_4, 1))

f_p_x_1 = firing_position_x_1.ravel()
f_p_y_1 = firing_position_y_1.ravel()
f_p_x_2 = firing_position_x_2.ravel()
f_p_y_2 = firing_position_y_2.ravel()
f_p_x_3 = firing_position_x_3.ravel()
f_p_y_3 = firing_position_y_3.ravel()
f_p_x_4 = firing_position_x_4.ravel()
f_p_y_4 = firing_position_y_4.ravel()

f_p_1 = np.column_stack((f_p_x_1, f_p_y_1))
f_p_2 = np.column_stack((f_p_x_2, f_p_y_2))
f_p_3 = np.column_stack((f_p_x_3, f_p_y_3))
f_p_4 = np.column_stack((f_p_x_4, f_p_y_4))

f_p_1 = f_p_1[np.logical_not(np.logical_and(f_p_1[:, 0] == 0, f_p_1[:, 1] == 0))]
f_p_2 = f_p_2[np.logical_not(np.logical_and(f_p_2[:, 0] == 0, f_p_2[:, 1] == 0))]
f_p_3 = f_p_3[np.logical_not(np.logical_and(f_p_3[:, 0] == 0, f_p_3[:, 1] == 0))]
f_p_4 = f_p_4[np.logical_not(np.logical_and(f_p_4[:, 0] == 0, f_p_4[:, 1] == 0))]

# ==============================================================================#

# density
density_1 = z_1 / (np.pi * 100 * radius ** 2)
density_2 = z_2 / (np.pi * 100 * radius ** 2)
density_3 = z_3 / (np.pi * 100 * radius ** 2)
density_4 = z_4 / (np.pi * 100 * radius ** 2)

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

if z_map > n/2:
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

plt.subplot(1, 2, 1)
plt.plot(f_p_1[:, 0], f_p_1[:, 1], 'x', color='r')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')

# title

tle_1_1 = "Firing time = " + str(z_1)
tle_2_1 = "density =" + str(round(density_1,5))
tle_3_1 = "status =" + str(n_1)
tle_1 = tle_1_1 + "\n" + tle_2_1 + "\n" + tle_3_1

plt.title(str(tle_1))

plt.subplot(1, 2, 2)

plt.plot(f_p_2[:, 0], f_p_2[:, 1], 'x', color='r')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')

# title

tle_1_2 = "Firing time = " + str(z_2)
tle_2_2 = "density =" + str(round(density_2,5))
tle_3_2 = "status =" + str(n_2)
tle_2 = tle_1_2 + "\n" + tle_2_2 + "\n" + tle_3_2

plt.title(str(tle_2))

plt.show()

plt.subplot(1, 2, 1)

plt.plot(f_p_3[:, 0], f_p_3[:, 1], 'x', color='r')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')

# title

tle_1_3 = "Firing time = " + str(z_3)
tle_2_3 = "density =" + str(round(density_3,5))
tle_3_3 = "status =" + str(n_3)
tle_3 = tle_1_3 + "\n" + tle_2_3 + "\n" + tle_3_3

plt.title(str(tle_3))

plt.subplot(1, 2, 2)

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

# ==============================================================================#
