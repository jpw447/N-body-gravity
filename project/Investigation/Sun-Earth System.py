import numpy as np
import matplotlib.pyplot as plt
from N_body_simulator import N_body_solver
from energy_calculator import system_energy

'''
This simulates a basic 2-body system of the Earth orbiting the sun. It was used to produce figures 1 and 2 in the report.
Initial conditions data retrieved from https://nssdc.gsfc.nasa.gov/planetary/factsheet/earthfact.html and https://nssdc.gsfc.nasa.gov/planetary/factsheet/sunfact.html
'''

# Setting constants for the system
G = 6.67e-11
AU = 1.496e11
M_sun = 2e30
M_earth = 6e24

# Establishing initial conditions
x_earth = 1*AU
v_earth = np.sqrt(G*M_sun/x_earth)
sun_ic = [0,0,0,0,0,0]
earth_ic = [x_earth,0,0,0,v_earth,0]
ic = sun_ic + earth_ic

N = 3 # int(len(ic)/6)
masses = [M_sun, M_earth]
names = ["Sun", "Earth"]

# Creating time array and grabbing solution
year_constant = 60**2 * 24 *365.25
number_of_points = 5000
number_of_years = 5
t_max = number_of_years * year_constant
t_array = np.linspace(0, t_max, number_of_points) # In seconds

solution, initial_energy = N_body_solver(N, ic, masses, t_array)

# Converting back to days for the purposes of plotting
t_days = t_array/year_constant

# Creating figure and plotting
fig_2d_system = plt.figure(figsize=(8,6))
ax_2d_system = fig_2d_system.gca()
ax_2d_system.set_xlabel("$x$ (AU)", fontsize=18)
ax_2d_system.set_ylabel("$y$ (AU)", fontsize=18)
ax_2d_system.set_aspect("equal")

# Position versus time plot for Earth
fig_phase = plt.figure(figsize=(8,6))
ax_phase = fig_phase.gca()
ax_phase.set_xlabel("$t$ (years)", fontsize=18)
ax_phase.set_ylabel("$x$ (AU)", fontsize=18)

# Energy conservation
fig_energy = plt.figure(figsize=(8,6))
ax_energy = fig_energy.gca()
ax_energy.set_xlabel("$t$ (years)", fontsize=18)
ax_energy.set_ylabel("% Energy Change", fontsize=18)

# Calculates the % energy change over the simulation, checking conservation
energy_over_time, KE, PE = system_energy(solution, masses, N, number_of_points)
energy_change = (energy_over_time-initial_energy)/initial_energy

ax_energy.plot(t_days, energy_change)

# Plotting each body in the solution
for i in range(N):
        x_start = 6*i
        y_start = x_start+1
        x = solution[:,x_start]/AU
        y = solution[:,y_start]/AU
        ax_2d_system.plot(x, y, label=names[i])
        if i == 1:
                ax_phase.plot(t_days, x)
        else:
                pass
ax_2d_system.legend()
plt.show()