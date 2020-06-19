import cvxpy as cvx
from pylab import *

# ==============================================================================#
# setting
nsteps = 5000

position = np.zeros((nsteps, 2))

# class current:
#     def __init__(self, current):
#         self.current = current
#
#     current = np.zeros((nsteps, 2))


current = np.zeros((nsteps, 2))
x = np.zeros((nsteps, 1))
y = np.zeros((nsteps, 1))
t = np.zeros((nsteps, 1))
Amp = np.zeros((nsteps, 1))
z = 0

R = 30
count = 0
w = 12
h = 12
region = 4
A = 0.005  # A
Vr = -65  # mV
tau = 10  # ms

A_random = np.random.random_sample(nsteps)

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
n = 2
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

# plt.figure(figsize=(5, 5))
# circ = np.linspace(0, 2 * pi)
# x_border = [-l / 2, l / 2, l / 2, -l / 2, -l / 2]
# y_border = [-l / 2, -l / 2, l / 2, l / 2, -l / 2]
# for i in xrange(n):
#     plt.plot(c[i, 0].value + r[i] * np.cos(circ), c[i, 1].value + r[i] * np.sin(circ), 'b')
# plt.plot(x_border, y_border, 'g')
# plt.axes().set_aspect('equal')
# plt.xlim([-l / 2, l / 2])
# plt.ylim([-l / 2, l / 2])
# plt.show()


point_x = np.zeros((n, 1))
point_y = np.zeros((n, 1))

for i in range(n):
    point_x[i] = c.value[i][0]
    point_y[i] = c.value[i][1]

point = np.column_stack((point_x, point_y))
point = sorted(point, key=lambda k: [k[1], k[0]], reverse=True)

# Integrate and fire model

firing_position_x_1 = np.zeros((nsteps, 1))
firing_position_y_1 = np.zeros((nsteps, 1))
firing_position_x_2 = np.zeros((nsteps, 1))
firing_position_y_2 = np.zeros((nsteps, 1))


### first point

radius_compare = np.zeros((nsteps, 1))

for i in range(0, nsteps - 1):

    radius_compare[i] = np.sqrt((position[i][0] - point_x[0]) ** 2 + (position[i][1] - point_y[0]) ** 2)

    if np.all(radius_compare[i] < radius):
        if A_random[i] > 0.3:

            current[i][0] = i
            current[i][1] = A * A_random[i]
            firing_position_x_1[i] = position[i][0]
            firing_position_y_1[i] = position[i][1]
            z += 1
        else:
            current[i][0] = i
            current[i][1] = 0

    else:
        current[i][0] = i
        current[i][1] = 0

### second point

# for i in range(0, nsteps - 1):
#     if np.all(np.sqrt((position[i][0] - point_x[0]) ** 2 + (position[i][1] - point_y[0]) ** 2) < radius):
#         if A_random[i] > 0.3:
#
#             current[i][0] = i
#             current[i][1] = A * A_random[i]
#             firing_position_x_2[i] = position[i][0]
#             firing_position_y_2[i] = position[i][1]
#             z += 1
#         else:
#             current[i][0] = i
#             current[i][1] = 0
#
#     else:
#         current[i][0] = i
#         current[i][1] = 0

    x[i] = position[i][0]
    y[i] = position[i][1]
    t[i] = current[i][0]
    Amp[i] = current[i][1]

t = np.linspace(0, nsteps, nsteps)


# ==============================================================================#

def LIF(_I=current[i][1], gl=0.16, Cm=0.0049):
    # Constants
    El = -0.065  # restint membrane potential [V]
    thresh = -0.050  # spiking threshold [V]

    # Experimental Setup
    # TIME
    T = nsteps - 1  # total simulation length [s]
    dt = 1  # step size [s]
    time = np.arange(0, T + dt, dt)  # step values [s]
    # VOLTAGE
    V = np.empty(len(time))  # array for saving Voltage history
    V[0] = El  # set initial to resting potential
    # CURRENT

    I = Amp

    # Measurements
    spikes = 0  # counter for number of spikes

    # Simulation
    for i in range(1, len(time)):
        # use "I - V/R = C * dV/dT" to get this equation
        dV = (Amp[i] - gl * (V[i - 1] - El)) / Cm
        V[i] = V[i - 1] + dV * dt

        # in case we exceed threshold
        if V[i] > thresh:
            V[i - 1] = 0.04  # set the last step to spike value
            V[i] = El  # current step is resting membrane potential
            spikes += 1  # count spike

    return V


def I_values(_I=current[i][1]):
    I = Amp
    return I


# ==============================================================================#

def start_LIF_sim():
    # time parameters for plotting
    T = nsteps - 1  # total simulation length [s]
    dt = 1  # step size [s]
    time = np.arange(0, T + dt, dt)  # step values [s]

    # initial parameters
    I_init = 0
    gl_init = 0.16
    Cm_init = 0.0049

    # update functions for lines
    V = LIF(_I=I_init, gl=gl_init, Cm=Cm_init)
    I = I_values(_I=I_init)

    ######### Plotting
    axis_color = 'lightgoldenrodyellow'

    fig = plt.figure("Leaky Integrate-and-Fire Neuron", figsize=(14, 7))
    ax = fig.add_subplot(111)
    plt.title("Interactive Leaky Integrate-and-Fire Neuron Simulation")
    fig.subplots_adjust(left=0.1, bottom=0.32)

    # plot lines
    line = plt.plot(time, V, label="Membrane Potential")[0]
    line2 = plt.plot(time, I, label="Applied Current")[0]

    # add legend
    plt.legend(loc="upper right")

    # add axis labels
    plt.ylabel("Potential [V]/ Current [A]")
    plt.xlabel("Time [s]")

    plt.show()


# if __name__ == '__main__':
#     start_LIF_sim()

# ==============================================================================#

# modify the firing#############################################

firing_position = np.zeros((z, 2))

f_p_x = firing_position_x_1.ravel()
f_p_y = firing_position_y_1.ravel()

f_p = np.column_stack((f_p_x, f_p_y))

f_p = f_p[np.logical_not(np.logical_and(f_p[:, 0] == 0, f_p[:, 1] == 0))]
# ==============================================================================#

# density
density = z / 6400
density_d = nsteps / 6400 * 0.3

if density >= density_d:
    n = "memorized"
else:
    n = "missing"

print(point)

# ==============================================================================#
# plotting

plt.plot(t, Amp)
plt.xlabel('time')
plt.ylabel('Amp')
plt.title('t-A')
plt.show()

plt.subplot(121)
plt.plot(x, y, 'x')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title("2D random walk")

plt.subplot(122)
plt.plot(f_p[:, 0], f_p[:, 1], 'x', color='r')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')

# title

t_1 = "Firing time = " + str(z)
t_2 = "density =" + str(density)
t_3 = "status =" + str(n)
t = t_1 + "\n" + t_2 + "\n" + t_3

plt.title(str(t))

plt.show()
# ==============================================================================#
