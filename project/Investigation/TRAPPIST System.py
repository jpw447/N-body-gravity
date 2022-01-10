import numpy as np
import matplotlib.pyplot as plt
from N_body_simulator import N_body_solver
from energy_calculator import system_energy

'''
The basic TRAPPIST system and the disturbed TRAPPIST system. Used to produce figures 5, 6, 7 and 8 in the report.
All of the planets initially have very low eccentricity (e<0.02) so can be treated as simple Keplerian orbits.
Initial condition data retrieved from: https://exoplanetarchive.ipac.caltech.edu/overview/TRAPPIST-1 
'''

# Constants and establishing initial conditions for each body
G = 6.67e-11
M_sol = 1.989e30
M_earth = 5.972e24
AU = 1.496e11
days_to_seconds = (60**2) * 24
# Inclination angle. If statement to prevent zero offset and redefining axes later
i = np.pi/2
if i == np.pi/2:
    cosine = 0
else:
    cosine = np.cos(i)

M_star = 0.898*M_sol
ic_star = [0,0,0, 0,0,0]

M_b = 1.374*M_earth
period_b = 1.510826*days_to_seconds
SMA_b = ((period_b**2 * G* M_star)/(4*np.pi**2))**(1/3)
v_b = np.sqrt(G*M_star/SMA_b) 
ic_b = [SMA_b,0,0, 0,v_b,0]

M_c = 1.308*M_earth
period_c = 2.421937*days_to_seconds
SMA_c = ((period_c**2 * G* M_star)/(4*np.pi**2))**(1/3)
v_c = np.sqrt(G*M_star/SMA_c)
ic_c = [SMA_c,0,0, 0,v_c,0]

M_d = 0.388*M_earth
period_d = 4.049219*days_to_seconds
SMA_d = ((period_d**2 * G* M_star)/(4*np.pi**2))**(1/3)
v_d = np.sqrt(G*M_star/SMA_d)
ic_d = [SMA_d,0,0, 0,v_d,0]

M_e = 0.692*M_earth
period_e = 6.101013*days_to_seconds
SMA_e = ((period_e**2 * G* M_star)/(4*np.pi**2))**(1/3)
v_e = np.sqrt(G*M_star/SMA_e)
ic_e = [SMA_e,0,0, 0,v_e,0]

M_f = 1.039*M_earth
period_f = 9.20754*days_to_seconds
SMA_f = ((period_f**2 * G* M_star)/(4*np.pi**2))**(1/3)
v_f = np.sqrt(G*M_star/SMA_f)
ic_f = [SMA_f,0,0, 0,v_f,0]

M_g = 1.321*M_earth
period_g = 12.352446*days_to_seconds
SMA_g = ((period_g**2 * G* M_star)/(4*np.pi**2))**(1/3)
v_g = np.sqrt(G*M_star/SMA_g)
ic_g = [SMA_g,0,0, 0,v_g,0]

M_h = 0.326*M_earth
period_h = 18.772866*days_to_seconds
SMA_h = ((period_h**2 * G* M_star)/(4*np.pi**2))**(1/3)
v_h = np.sqrt(G*M_star/SMA_h)
ic_h = [SMA_h,0,0, 0,v_h,0]

# Adding a Jupiter-sized planet between e and f, but on the other side of the star and travelling in -y direction
M_jupiter = 317.83*M_earth
SMA_jupiter = -np.mean([SMA_e,SMA_f])
v_jupiter = -np.sqrt(G*M_star/abs(SMA_jupiter))
ic_jupiter = [SMA_jupiter,0,0, 0,v_jupiter,0]

# Initial conditions, masses and names for the undisturbed TRAPPIST system
ic = ic_star + ic_b + ic_c + ic_d + ic_e + ic_f + ic_g + ic_h
masses = [M_star, M_b, M_c, M_d, M_e, M_f, M_g, M_h]
names = ["TRAPPIST-1", "TRAPPIST-1b", "TRAPPIST-1c", "TRAPPIST-1d", "TRAPPIST-1e", "TRAPPIST-1f", "TRAPPIST-1g", "TRAPPIST-1h"]

# Creating time array and calling the function
N = int(len(masses))
T = 4*period_g
number_of_points = 100000   
t_array = np.linspace(0, T, number_of_points) # In seconds

# Creating plots
# 2d system plot
fig_trappist_1, ax_trappist_1 = plt.subplots(1, 2, figsize=(12,8))
plt.subplots_adjust(wspace=0.3, hspace=0.5, top=0.85)
for axis in ax_trappist_1:
    axis.set_xlabel("$x$ (AU)", fontsize=16)
    axis.set_ylabel("$y$ (AU)", fontsize=16)
    axis.set_aspect("equal")
    axis.set_xlim(-0.18, 0.18)
    axis.set_ylim(-0.18, 0.18)

# Plots of separation between TRAPPIST-1 and planets
fig_separation = plt.figure(figsize=(8,6))
ax_separation = fig_separation.gca()
ax_separation.set_xlabel("$t$ (days)", fontsize=16)
ax_separation.set_ylabel("$r$ (AU)", fontsize=16)

# Phase space plots
fig_poincare, ax_poincare = plt.subplots(1, 2, figsize=(12,6))
plt.subplots_adjust(wspace=0.3, hspace=0.5, top=0.85)
for axis in ax_poincare:
    axis.set_xlabel("$x$ (AU)", fontsize=16)
    axis.set_ylabel("$v_{x}$ (ms$^{-1}$)", fontsize=16)
    axis.set_xlim(-0.18, 0.18)
    axis.set_ylim(-140000, 140000)
plt.tight_layout()

# Converting to days for the plots
t_days = t_array/days_to_seconds

def system_solver(N, ic, masses, t_array, axis_system, axis_separation, axis_poincare, disturbed=False):
    '''
    Simple function that calls N_body_solver to solve the system, and proceeds to systematically plot each relevant body on the relevant axis.
    This is used to prevent repetition later in the code.
    '''
    solution, initial_energy = N_body_solver(N, ic, masses, t_array)

    # Looping through solution to plot all bodies
    x_star = solution[:,0]/AU
    y_star = solution[:,1]/AU
    z_star = solution[:,2]/AU
    for i in range(N):
        # When system is not disturbed by Jupiter
        name = names[i]
        x_start = 6*i
        y_start = x_start+1
        z_start = x_start+2
        x = solution[:,x_start]/AU - x_star
        y = solution[:,y_start]/AU - y_star
        z = solution[:,z_start]/AU - z_star
        if disturbed == False:
            # Special Jupiter plot
            if name == "Jupiter":
                axis_system.plot(x, y, color="cyan", linestyle="dashed", label=name)
                axis_system.plot(x[0], y[0], color="cyan", marker="o")
            # Phase space and separation plots for 1f and 1g
            elif (name == "TRAPPIST-1f") or (name == "TRAPPIST-1e"):
                # Velocities needed for phase space for only these bodies
                vx_start = x_start+3
                vx = solution[:,vx_start]
                separation = np.sqrt( x**2 + y**2 + z**2 )
                if name == "TRAPPIST-1f":
                    axis_system.plot(x, y, 'black', label=name)
                    axis_system.plot(x[0], y[0], 'ko')
                    axis_separation.plot(t_days, separation, 'k--', label=name)
                    axis_poincare.plot(x, vx, 'black', label=name)
                    axis_poincare.plot(x[0], vx[0], 'ko')
                else:
                    axis_system.plot(x, y, 'red', label=name)
                    axis_system.plot(x[0], y[0], 'ro')
                    axis_separation.plot(t_days, separation, 'r--', label=name)
                    axis_poincare.plot(x, vx, 'red', label=name)
                    axis_poincare.plot(x[0], vx[0], 'ro')
            # Plots the planet normally if not Jupiter, 1f or 1g
            else:
                axis_system.plot(x, y, label=name)
        # When system is disturbed by Jupiter, only plot Jupiter, 1f and 1g
        else:
            if (name == "TRAPPIST-1f") or (name == "TRAPPIST-1e"):
                vx_start = x_start+3
                vx = solution[:,vx_start]
                separation = np.sqrt( x**2 + y**2 + z**2 )
                if name == "TRAPPIST-1f":
                    axis_system.plot(x, y, 'black', label=name)
                    axis_system.plot(x[0], y[0], 'ko')
                    axis_separation.plot(t_days, separation, 'black', label=name+" disturbed")
                    axis_poincare.plot(x, vx, 'black', label=name)
                    axis_poincare.plot(x[0], vx[0], 'ko')
                else:
                    axis_system.plot(x, y, 'red', label=name)
                    axis_system.plot(x[0], y[0], 'ro')
                    axis_separation.plot(t_days, separation, 'red', label=name+" disturbed")
                    axis_poincare.plot(x, vx, 'red', label=name)
                    axis_poincare.plot(x[0], vx[0], 'ro')
            elif name == "Jupiter":
                # Special Jupiter plot again
                axis_system.plot(x, y, color="cyan", linestyle="dashed", label=name)
                axis_system.plot(x[0], y[0], color="cyan", marker="o")
            else:
                pass
                
# Running for regular TRAPPIST system and one with Jupiter in it
system_solver(N, ic, masses, t_array, ax_trappist_1[0], ax_separation, ax_poincare[0])
# Updating system to include Jupiter mass
ic += ic_jupiter
names.append("Jupiter")
masses.append(M_jupiter)
N = int(len(masses))
system_solver(N, ic, masses, t_array, ax_trappist_1[1], ax_separation, ax_poincare[1], disturbed=True)

# Adding legends, some of which are moved to stop the data being obscured
ax_trappist_1[1].legend(bbox_to_anchor=(0, 0.05))
ax_separation.legend()
ax_poincare[1].legend(bbox_to_anchor=(0, 0.05))
plt.show()