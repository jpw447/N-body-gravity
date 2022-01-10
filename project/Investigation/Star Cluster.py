import numpy as np
import matplotlib.pyplot as plt
from N_body_simulator import N_body_solver
'''
Simple code creating a random, seeded star cluster of bodies of mass varying between 0.8 and 1.2 Earth masses.
This will take approximately 61 seconds to run.
'''
# Default parameters, along with axis boundaries
M = 6e28
N = 10
AU = 1.496e11
limit = 8
ax_limit = limit/2

# Arbitrarily chosen seed to make a nice-looking star cluster evolution.
np.random.seed(55)

# Set mass range to be 0.8 to 1.2 solar masses instead of 0 to 1 solar masses
# Color them by mass
masses = np.random.uniform(0.8,1.2, N)*M
x_positions = (np.random.rand(N)*limit*AU) - (limit/2)*AU
y_positions = (np.random.rand(N)*limit*AU) - (limit/2)*AU
z_positions = (np.random.rand(N)*limit*AU) - (limit/2)*AU

fig_cluster = plt.figure(figsize=(8,6))
ax_cluster = fig_cluster.gca(projection="3d")

ax_cluster.plot(x_positions/AU, y_positions/AU, z_positions/AU, 'rx')
ax_cluster.set_xlabel("$x$ (AU)",fontsize=16)
ax_cluster.set_ylabel("$y$ (AU)",fontsize=16)
ax_cluster.set_zlabel("$z$ (AU)",fontsize=16)
ax_cluster.set_xlim(-ax_limit, ax_limit)
ax_cluster.set_ylim(-ax_limit, ax_limit)
ax_cluster.set_zlim(-ax_limit, ax_limit)

# Creating initial conditions array with all stars stationary
ic = np.zeros(6*N)
for i in range(N):
    x = 6*i
    y = x+1
    z = x+2
    vx = x+3
    vy = x+4
    vz = x+5
    ic[x] = x_positions[i]
    ic[y] = y_positions[i]
    ic[z] = z_positions[i]
    ic[vx] = 1
    ic[vy] = 0
    ic[vz] = 0

# Creating time array
year_constant = 60**2 * 24 *365.25
number_of_points = 50000
number_of_years = 10
t_max = number_of_years * year_constant
t_array = np.linspace(0, t_max, number_of_points) # In seconds

# Solving the system and plotting it in 3D space
solution, system_energy = N_body_solver(N, ic, masses, t_array)

for i in range(N):
        x_start = 6*i
        y_start = x_start+1
        z_start = x_start+2
        x = solution[:,x_start]/AU
        y = solution[:,y_start]/AU
        z = solution[:,z_start]/AU
        ax_cluster.plot(x, y, z, label="Star "+str(i+1)+" path")

ax_cluster.legend(bbox_to_anchor=(0, 0.5))
plt.show()