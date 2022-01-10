import numpy as np
import matplotlib.pyplot as plt
from N_body_simulator import N_body_solver

# Setting constants
G = 6.67e-11
AU = 1.496e11
M_sun = 2e30
M_earth = 6e24
M_moon = 7.346e22

# Establishing initial conditions
r_earth = 1*AU
r_moon = 3.844e8
v_earth = np.sqrt(G*M_sun/r_earth)
v_moon = np.sqrt(G*M_earth/r_moon) + v_earth
sun_ic = [0,0,0, 0,0,0]
earth_ic = [r_earth,0,0, 0,v_earth,0]
moon_ic = [r_earth+r_moon,0,0, 0,v_moon,0]
ic = sun_ic + earth_ic + moon_ic

N = int(len(ic)/6)
masses = [M_sun, M_earth, M_moon]
names = ["Sun", "Earth", "Moon"]

# Creating time array
year_constant = 60**2 * 24 *365.25
number_of_points = 5000
number_of_years = 20
t_max = number_of_years * year_constant
t_array = np.linspace(0, t_max, number_of_points)

solution, system_energy = N_body_solver(N, ic, masses, t_array)

# Positions of each body
x_sun = solution[:,0]
y_sun = solution[:,1]
x_earth = solution[:,6]
y_earth = solution[:,7]
x_moon = solution[:,12]
y_moon = solution[:,13]

# Calculating separations to show the moon orbits the earth periodically
earth_sun_separation = np.sqrt( (x_earth-x_sun)**2 + (y_earth-y_sun)**2 )
moon_sun_separation = np.sqrt( (x_moon-x_sun)**2 + (y_moon-y_sun)**2 )

fig_separation = plt.figure(figsize=(8,6))
ax_separation = fig_separation.gca()
ax_separation.plot(t_array/year_constant, moon_sun_separation/AU, 'r--', label="Moon-Sun separation")
ax_separation.plot(t_array/year_constant, earth_sun_separation/AU, 'k', label="Earth-Sun separation")
ax_separation.set_xlabel("$t$ (years)")
ax_separation.set_ylabel("Body Separation (AU)")
ax_separation.legend()

# Creating figure and plotting
fig_2d_system = plt.figure(figsize=(8,6))
ax_2d_system = fig_2d_system.gca()

for i in range(N):
        x_start = 6*i
        y_start = x_start+1
        corrected_x = solution[:,x_start]/AU - solution[:,0]/AU
        corrected_y = solution[:,y_start]/AU - solution[:,1]/AU
        ax_2d_system.plot(corrected_x, corrected_y, label=names[i])
ax_2d_system.set_aspect("equal")
plt.show()