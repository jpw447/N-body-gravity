import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

'''
Hand-crafted solution to the 2 body problem in 3D. This is a redundant file and may not produce the correct solution.
It was not used in the final report, but existed as a study into the 2 body problem.
'''

if __name__ == "__main__":
    # Constants
    G = 6.67408e-11
    M1 = 2e30
    M2 = 6e24
    AU = 1.496e11
    
    # Position intitial conditions
    x1_init = 0
    y1_init = 0
    z1_init = 0

    x2_init = 1*AU
    y2_init = 0
    z2_init = 0
    
    # Separations
    M1_M2_separation = np.sqrt( (x1_init - x2_init)**2 + (y1_init - y2_init)**2 + (z1_init - z2_init)**2 )

    # Velocities    
    vx1_init = 0
    vy1_init = 0
    vz1_init = 0
    vx2_init = 0
    vy2_init = np.sqrt(G*M1/M1_M2_separation)
    vz2_init = 0

    # Energies
    M1_KE_init = 0.5 * M1 * (vx1_init*vx1_init + vy1_init*vy1_init + vz1_init*vz1_init)
    M2_KE_init = 0.5 * M2 * (vx2_init*vx2_init + vy2_init*vy2_init + vz2_init*vz2_init)
    PE_init = -G*M1*M2/M1_M2_separation
    KE_init = M1_KE_init + M2_KE_init
    initial_energy = KE_init + PE_init
    
    # Time array
    year_constant = 60**2 * 24 * 365.35
    number_of_years = 10
    number_of_points = 10000
    t_max = number_of_years * year_constant
    t_array = np.linspace(0, t_max, number_of_points)

    # Lists containin intial conditions (parameters) and important constants.
    # These appear in a certain order here, and the order must be adheredt to 
    # everywhere else you create a list like this - in the function passed to
    # odeint in both input and output, and what odeint outputs.
    initial_parameters =[x1_init, y1_init, z1_init,
                         x2_init, y2_init, z2_init,
                         vx1_init, vy1_init, vz1_init,
                         vx2_init, vy2_init, vz2_init]
    constants = [G, M1, M2]

    def field_function(parameters, t, constants):
        '''
        Function that takes input parameters (initial conditions) and constants, as well as at time array.
        Returns a list containing the field of differential equations for each derivative.

        Args:
            parameters: list with initial conditions, containing positions and velocities of 2 bodies
            t: time array used by ode_int
            constants: list containing constants such as Gravitational Constant and masses

        Returns:
            field: list containing the derivatives for the system
        '''
        x1, y1, z1, x2, y2, z2, vx1, vy1, vz1, vx2, vy2, vz2 = parameters
        G, M1, M2 = constants

        # Separation of bodies  
        r_12 = ((x1-x2)*(x1-x2) + (y1-y2)*(y1-y2) + (z1-z2)*(z1-z2))**0.5

        # ODEs for star
        x1_dot = vx1
        vx1_dot = -(G*M2/r_12**3) * (x1-x2)
        y1_dot = vy1
        vy1_dot = -(G*M2/r_12**3) * (y1-y2)
        z1_dot = vz1
        vz1_dot = -(G*M2/r_12**3) * (z1-z2)

        # ODEs for Earth
        x2_dot = vx2
        vx2_dot = -(G*M1/r_12**3) * (x2-x1)
        y2_dot = vy2
        vy2_dot = -(G*M1/r_12**3) * (y2-y1)
        z2_dot = vz2
        vz2_dot = -(G*M2/r_12**3) * (z2-z1)

        # Returning ODEs as a list
        field = [x1_dot, y1_dot, z1_dot,
                 x2_dot, y2_dot, z2_dot,
                 vx1_dot, vy1_dot, vz1_dot,
                 vx2_dot, vy2_dot, vz2_dot]
        return field

    # Passing function to odeint and retrieving planet and star positions
    # Tighter tolerances results in smaller energy deviations
    solution = odeint(field_function, initial_parameters, t_array, args=(constants,), rtol=1e-13) 
    
    # solution = array containing 8 columns for x_star, y_star etc. in order they appear in field_function. Each row is t value
    # Columns:
    # 0: x1
    # 1: y1
    # 2: z1
    # 3: x2
    # 4: y2
    # 5: z1
    # 6: vx1
    # 7: vy1
    # 8: vz1
    # 9: vx2
    # 10: vy2
    # 11: vz2

    # Velocities
    vx1, vy1, vz1 = solution[:,6], solution[:,7], solution[:, 8]
    v1_squared = vx1*vx1 + vy1*vy1
    vx2, vy2, vz2 = solution[:,9], solution[:,10], solution[:,11]
    v2_squared = vx2*vx2 + vy2*vy2 + vz2*vz2
    
    # Positions and separation
    x1, y1, z1 = solution[:,0], solution[:,1], solution[:,2]
    x2, y2, z2 = solution[:,3], solution[:,4], solution[:,5]
    r = np.sqrt ( (x1 - x2)**2 + (y1 - y2)**2 + (z1 - z2)**2 )
    # Converting to AU for plot
    x1_AU, y1_AU, z1_AU = x1/AU, y1/AU, z1/AU
    x2_AU, y2_AU, z2_AU = x2/AU, y2/AU, z2/AU

    # Finds maximum position of any body, and adds 20% for plot limits
    maxima = []
    for position in [abs(x1_AU), abs(x2_AU), abs(y1_AU), abs(y2_AU), abs(z1_AU), abs(z2_AU)]:
        maxima.append(np.max(position))
    ax_planet_limits = np.max(maxima) + np.max(maxima)*0.2
    print("Axis limits are ±"+str(ax_planet_limits))

    # Energies
    KE = 0.5*M1*v1_squared + 0.5*M2*v2_squared
    PE = -G*M1*M2/r
    energy_difference = (KE + PE - initial_energy)/initial_energy

    # Creating figures and axes
    fig_planet = plt.figure(figsize=(8,6))
    ax_planet = fig_planet.gca(projection="3d")
    fig_sinusoid = plt.figure(figsize=(8,6))
    ax_sinusoid = fig_sinusoid.gca()
    fig_system_energy = plt.figure(figsize=(8,6))
    ax_system_energy = fig_system_energy.gca()

    # Creating plots
    # Planet position x versus y
    ax_planet.plot(x1_AU, y1_AU, z1_AU, 'b')
    ax_planet.plot(x2_AU, y2_AU, z2_AU, 'r')
    ax_planet.set_title("Planet position", fontsize=20)
    ax_planet.set_xlabel("$x_{\\bigoplus}$ (AU)", fontsize=16)
    ax_planet.set_ylabel("$y_{\\bigoplus}$ (AU)", fontsize=16)
    ax_planet.set_xlim(-ax_planet_limits, ax_planet_limits)
    ax_planet.set_ylim(-ax_planet_limits, ax_planet_limits)
    ax_planet.set_zlim(-ax_planet_limits, ax_planet_limits)

    t_array = t_array / year_constant # Converting time back to years

    # Plotting planet position against time
    sinusoid = np.cos(2*np.pi*t_array)
    ax_sinusoid.plot(t_array, x2_AU, 'k')
    ax_sinusoid.plot(t_array, sinusoid, 'r--')
    ax_sinusoid.set_title("Planet Position Versus Time", fontsize=20)
    ax_sinusoid.set_xlabel("$t$ (years)", fontsize=16)
    ax_sinusoid.set_ylabel("$x_{\\bigoplus}$ (AU)", fontsize=16)

    # Plotting energies
    ax_system_energy.plot(t_array, energy_difference, 'k')
    ax_system_energy.set_title("Energy Change over Time", fontsize=20)
    ax_system_energy.set_xlabel("$t$ (years)", fontsize=16)
    ax_system_energy.set_ylabel("Energy Change (J)", fontsize=16)
    plt.show()