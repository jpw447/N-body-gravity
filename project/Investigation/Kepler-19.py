import numpy as np
import matplotlib.pyplot as plt
from N_body_simulator import N_body_solver
from fourier_transforms import fourier_func
from energy_calculator import system_energy

'''
This script produced figures 3 and 4 in the report.
This script produces a simulation of the Kepler-19 system, whose planets were detected by transit photometry,
transit timing variation and the radial velocity technique. This simulation will attempt to recover the
orbital periods of the planets by measuring the radial velocity technique and Fourier transforms.

Because not all of the planets have a measured semi-major axis, it is approximated using Kepler's Third Law.

Parameters used for the system were retrieved directly from: https://exoplanetarchive.ipac.caltech.edu/overview/Kepler-19%20b#planet_Kepler-19-b_collapsible
'''
# Parameters of the system
G = 6.67e-11
M_sol = 1.989e30
M_earth = 5.972e24
AU = 1.496e11
days_to_seconds = (60**2) * 24
# Inclination angle. If statement to prevent zero offset and having to redefine axes later for visual clarity. Not adjusted in final report.
i = 0
if i == np.pi/2:
    cosine = 0
else:
    cosine = np.cos(i)

# Creating initial conditions for the system
M_star = 0.936*M_sol
ic_star = [0, 0, 0, 0, 0, 0]

M_b = 8.4*M_earth
period_b = 9.287 * days_to_seconds
SMA_b = ((period_b**2 * G* M_star)/(4*np.pi**2))**(1/3)
v_b = np.sqrt(G*M_star/SMA_b)
v_b_y = v_b*cosine
v_b_z = v_b*np.sin(i)
ic_b = [SMA_b, 0, 0, 0, v_b_y, v_b_z]

M_c = 13.1*M_earth
period_c = 28.731 * days_to_seconds
SMA_c = ((period_c**2 * G* M_star)/(4*np.pi**2))**(1/3)
v_c = np.sqrt(G*M_star/SMA_c)
v_c_y = v_c*cosine
v_c_z = v_c*np.sin(i)
ic_c = [SMA_c, 0, 0, 0, v_c_y, v_c_z]

M_d = 22.5*M_earth
period_d = 62.95 * days_to_seconds
SMA_d = ((period_d**2 * G* M_star)/(4*np.pi**2))**(1/3)
v_d = np.sqrt(G*M_star/SMA_d)
v_d_y = v_d*cosine
v_d_z = v_d*np.sin(i)
ic_d = [SMA_d, 0, 0, 0, v_d_y, v_d_z]

# Initial conditions array
ic = ic_star + ic_b + ic_c + ic_d
masses = [M_star, M_b, M_c, M_d]
names = ["Kepler-19", "Kepler-19b", "Kepler-19c", "Kepler-19d"]

# Creating time array and calling the function
N = 4
T = 16*period_d 
number_of_points = 10000
t_array = np.linspace(0, T, number_of_points) # In seconds

solution, initial_energy = N_body_solver(N, ic, masses, t_array)

# Creating figures
# 3d system
fig_kepler_19 = plt.figure(figsize=(8,6))
ax_kepler_19 = fig_kepler_19.gca(projection="3d")
ax_kepler_19.set_xlabel("$x$ (AU)", fontsize=14)
ax_kepler_19.set_ylabel("$y$ (AU)", fontsize=14)
ax_kepler_19.set_zlabel("$z$ (AU)", fontsize=14)

# Radial velocity (vy) and Fourier Transform subplot
fig_radial_velocity, ax_radial_velocity = plt.subplots(1, 2, figsize=(12,6))
plt.subplots_adjust(wspace=0.3, hspace=0.3, top=0.85)
ax_radial_velocity[0].set_xlabel("$t$ (days)", fontsize=16)
ax_radial_velocity[0].set_ylabel("Line of Sight Radial Velocity (m/s)", fontsize=16)
ax_radial_velocity[1].set_xlabel("Frequency (s$^{-1}$)", fontsize=16)
ax_radial_velocity[1].set_ylabel("Power (W)", fontsize=16)

# Energy conservation
fig_energy = plt.figure(figsize=(8,6))
ax_energy = fig_energy.gca()
ax_energy.set_xlabel("$t$ (days)", fontsize=16)
ax_energy.set_ylabel("% Energy change", fontsize=16)

# Creating radial velocity plot
t_days = t_array/days_to_seconds # Time converted to days for plots
vy_star = solution[:,4]
ax_radial_velocity[0].plot(t_days, vy_star, 'r')

# Fourier transform of radial velocity
# Note: to reproduce figure 4 from the report, the axis need to be manually adjusted in the window
power, frequencies = fourier_func(vy_star, T)
ax_radial_velocity[1].plot(frequencies[1:], power[1:], label="Frequency Intensity")
max_power = np.max(power[1:])
ax_radial_velocity[1].vlines(1/period_b, 0, max_power, colors="red", linestyles="dashed")
ax_radial_velocity[1].vlines(1/period_c, 0, max_power, colors="red", linestyles="dashed")
ax_radial_velocity[1].vlines(1/period_d, 0, max_power, colors="red", linestyles="dashed")
ax_radial_velocity[1].legend()

# Looping through solution to plot all bodies
for i in range(N):
        if masses[i] != 0:
            x_start = 6*i
            y_start = x_start+1
            z_start = x_start+2
            x = solution[:,x_start]/AU
            y = solution[:,y_start]/AU
            z = solution[:,z_start]/AU
            ax_kepler_19.plot(x, y, z, label=names[i])
            ax_kepler_19.plot(x[0], y[0], z[0], marker='o')
        else:
            pass
ax_kepler_19.legend()

# Total energy change and plot
energy_over_time, KE, PE = system_energy(solution, masses, N, number_of_points)
energy_change = (energy_over_time-initial_energy)/initial_energy
ax_energy.plot(t_days, energy_change, 'r')

plt.show()