import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

'''
Hand-crafted solution to the 3 body problem in 3D. This is a redundant file and may not produce the correct solution.
It was not used in the final report, but existed as a study into the 3 body problem.
'''

if __name__ == "__main__":
    # Constants
    G = 6.67408e-11
    M1 = 1.989e30
    M2 = 5.683e26
    M3 = 1.898e27
    AU = 1.496e11
    
    # Get division by zero otherwise
    zero = 1e-6
    
    # Position intitial conditions
    x1_init = 0
    y1_init = 0
    z1_init = 0

    x2_init = 9.5*AU
    y2_init = 0
    z2_init = 0

    x3_init = 5.2*AU
    y3_init = 0
    z3_init = 0
    
    # Separations
    M1_M2_separation = np.sqrt( (x1_init - x2_init)**2 + (y1_init - y2_init)**2 + (z1_init - z2_init)**2 )
    M1_M3_separation = np.sqrt( (x1_init - x3_init)**2 + (y1_init - y3_init)**2 + (z1_init - z3_init)**2 )
    M2_M3_separation = np.sqrt( (x2_init - x3_init)**2 + (y2_init - y3_init)**2 + (z2_init - z3_init)**2 )

    # Velocities    
    vx1_init = 0
    vy1_init = 0
    vz1_init = 0

    vx2_init = 0
    vy2_init = np.sqrt(G*M1/M1_M2_separation)
    vz2_init = 0

    vx3_init = 0
    vy3_init = np.sqrt(G*M1/M1_M3_separation)
    vz3_init = 0

    # Energies
    M1_KE_init = 0.5 * M1 * (vx1_init*vx1_init + vy1_init*vy1_init + vz1_init*vz1_init)
    M2_KE_init = 0.5 * M2 * (vx2_init*vx2_init + vy2_init*vy2_init + vz2_init*vz2_init)
    M3_KE_init = 0.5 * M3 * (vx3_init*vx3_init + vy3_init*vy3_init + vz3_init*vz3_init)
    PE_init = -G*M1*M2/M1_M2_separation - G*M1*M3/M1_M3_separation - G*M2*M3/M2_M3_separation
    KE_init = M1_KE_init + M2_KE_init + M3_KE_init
    initial_energy = KE_init + PE_init
    
    # Time array7
    x2_year = 29
    year_constant = 60**2 * 24 * 365.35
    number_of_years = x2_year*2
    number_of_points = 20000
    t_max = number_of_years * year_constant
    t_array = np.linspace(0, t_max, number_of_points)

    # Lists containin intial conditions (parameters) and important constants.
    # These appear in a certain order here, and the order must be adheredt to 
    # everywhere else you create a list like this - in the function passed to
    # odeint in both input and output, and what odeint outputs.
    initial_parameters =[x1_init, y1_init, z1_init,
                         x2_init, y2_init, z2_init,
                         x3_init, y3_init, z3_init,    
                         vx1_init, vy1_init, vz1_init,
                         vx2_init, vy2_init, vz2_init,
                         vx3_init, vy3_init, vz3_init]
    constants = [G, M1, M2, M3]

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
        x1, y1, z1, x2, y2, z2, x3, y3, z3, vx1, vy1, vz1, vx2, vy2, vz2, vx3, vy3, vz3 = parameters
        G, M1, M2, M3 = constants

        # Close encounters could be causing bus, by creating huge numbers.
        # Needs some checking.

        # Separation of bodies  
        r_12 = ((x1-x2)*(x1-x2) + (y1-y2)*(y1-y2) + (z1-z2)*(z1-z2))**0.5
        r_13 = ((x1-x3)*(x1-x3) + (y1-y3)*(y1-y3) + (z1-z3)*(z1-z3))**0.5
        r_23 = ((x2-x3)*(x2-x3) + (y2-y3)*(y2-y3) + (z2-z3)*(z2-z3))**0.5

        # ODEs for M1
        x1_dot = vx1
        vx1_dot = -(G*M2/r_12**3) * (x1-x2) - (G*M3/r_13**3) * (x1-x3)
        y1_dot = vy1
        vy1_dot = -(G*M2/r_12**3) * (y1-y2) - (G*M3/r_13**3) * (y1-y3)
        z1_dot = vz1
        vz1_dot = -(G*M2/r_12**3) * (z1-z2) - (G*M3/r_13**3) * (z1-z3)

        # ODEs for M2
        x2_dot = vx2
        vx2_dot = -(G*M1/r_12**3) * (x2-x1) - (G*M3/r_23**3) * (x2-x3)
        y2_dot = vy2
        vy2_dot = -(G*M1/r_12**3) * (y2-y1) - (G*M3/r_23**3) * (y2-y3)
        z2_dot = vz2
        vz2_dot = -(G*M1/r_12**3) * (z2-z1) - (G*M3/r_23**3) * (z2-z3)

        # ODEs for M3
        x3_dot = vx3
        vx3_dot = -(G*M1/r_13**3) * (x3-x1) - (G*M2/r_23**3) * (x3-x2)
        y3_dot = vy3
        vy3_dot = -(G*M1/r_13**3) * (y3-y1) - (G*M2/r_23**3) * (y3-y2)
        z3_dot = vz3
        vz3_dot = -(G*M1/r_13**3) * (z3-z1) - (G*M2/r_23**3) * (z3-z2)

        # Returning ODEs as a list

        field =[x1_dot, y1_dot, z1_dot,
                x2_dot, y2_dot, z2_dot,
                x3_dot, y3_dot, z3_dot,
                vx1_dot, vy1_dot, vz1_dot,
                vx2_dot, vy2_dot, vz2_dot,
                vx3_dot, vy3_dot, vz3_dot]
        return field

    # Passing function to odeint and retrieving planet and star positions
    # Tighter tolerances results in smaller energy deviations
    solution = odeint(field_function, initial_parameters, t_array, args=(constants,), full_output=1, rtol=1e-13)[0]
    
    # solution = array containing 8 columns for x_star, y_star etc. in order they appear in field_function. Each row is t value
    # Columns:
    # 0: x1
    # 1: y1
    # 2: z1
    # 3: x2
    # 4: y2
    # 5: z2
    # 6: x3
    # 7: y3
    # 8: z3
    # 9: vx1
    # 10: vy1
    # 11: vz1
    # 12: vx2
    # 13: vy2
    # 14: vz2
    # 15: vx3
    # 16: vy3
    # 17: vz3

    # Velocities
    vx1, vy1, vz1 = solution[:,9], solution[:,10], solution[:,11]
    v1_squared = vx1*vx1 + vy1*vy1 + vz1*vz1
    vx2, vy2, vz2 = solution[:,12], solution[:,13], solution[:,14]
    v2_squared = vx2*vx2 + vy2*vy2 + vz2*vz2 
    vx3, vy3, vz3 = solution[:,15], solution[:,16], solution[:,17]
    v3_squared = vx3*vx3 + vy3*vy3 + vz3*vz3
    
    # Positions
    x1, y1, z1 = solution[:,0], solution[:,1], solution[:,2]
    x2, y2, z2 = solution[:,3], solution[:,4], solution[:,5]
    x3, y3, z3 = solution[:,6], solution[:,7], solution[:,8]
    r_12 = np.sqrt ( (x1 - x2)**2 + (y1 - y2)**2 + (z1 - z2)**2 )
    r_13 = np.sqrt ( (x1 - x3)**2 + (y1 - y3)**2 + (z1 - z3)**2 )
    r_23 = np.sqrt ( (x2 - x3)**2 + (y2 - y3)**2 + (z2 - z3)**2 )

    # Converting to AU for plot
    x1_AU, y1_AU, z1_AU = x1/AU, y1/AU, z1/AU
    x2_AU, y2_AU, z2_AU = x2/AU, y2/AU, z2/AU
    x3_AU, y3_AU, z3_AU = x3/AU, y3/AU, z3/AU

    # Energies
    KE = 0.5*M1*v1_squared + 0.5*M2*v2_squared + 0.5*M3*v3_squared
    PE = -G*M1*M2/r_12 - G*M1*M3/r_13 - G*M2*M3/r_23
    energy_difference = (KE + PE - initial_energy)/initial_energy
    print(initial_energy)

    # Creating figures and axes
    # Plotting separations between bodies
    fig_separations = plt.figure(figsize=(8,6))
    ax_separations = fig_separations.gca()
    ax_separations.plot(t_array, r_12, 'k', label="M1-M2 separation")
    ax_separations.plot(t_array, r_13, 'g--', label="M1-M3 separation")
    ax_separations.plot(t_array, r_23, 'r', label="M2-M3 separation")
    ax_separations.legend()
    ax_separations.set_title("Separations between bodies")
    ax_separations.set_xlabel("Time (yrs)")
    ax_separations.set_ylabel("Separation (AU)")

    t_array = t_array / year_constant # Converting time back to years
    
    fig_sinusoid = plt.figure(figsize=(8,6))
    ax_sinusoid = fig_sinusoid.gca()

    # Plotting planet position against time
    sinusoid = (x2_init/AU)*np.cos(2*np.pi/(x2_year)*t_array)
    ax_sinusoid.plot(t_array, x2_AU, 'k')
    ax_sinusoid.plot(t_array, sinusoid, 'r--')
    ax_sinusoid.set_title("M2 Position Versus Time", fontsize=20)
    ax_sinusoid.set_xlabel("$t$ (years)", fontsize=16)
    ax_sinusoid.set_ylabel("$x_{\\bigoplus}$ (AU)", fontsize=16)

    # Difference between theory and simulation
    fig_phase = plt.figure(figsize=(8,6))
    ax_phase = fig_phase.gca()
    phase_difference = sinusoid-x2_AU
    ax_phase.plot(t_array, phase_difference, 'k')
    ax_phase.set_title("Prediction VS Simluation for M2", fontsize=20)
    ax_phase.set_xlabel("$t$ (years)", fontsize=16)
    ax_phase.set_ylabel("$x_{\\bigoplus}$ (AU)", fontsize=16)

    # Plotting energies
    fig_system_energy = plt.figure(figsize=(8,6))
    ax_system_energy = fig_system_energy.gca()
    ax_system_energy.plot(t_array, energy_difference, 'k')
    ax_system_energy.set_title("Energy Change over Time", fontsize=20)
    ax_system_energy.set_xlabel("$t$ (years)", fontsize=16)
    ax_system_energy.set_ylabel("Energy Change (J)", fontsize=16)

    # Planet position x versus y
    fig_system = plt.figure(figsize=(8,6))
    ax_system = fig_system.gca(projection="3d")
    ax_system.plot(x1_AU, y1_AU, z1_AU, 'b', label="M1={} kg".format(M1))
    ax_system.plot(x2_AU, y2_AU, z2_AU, 'k', label="M2={} kg".format(M2))
    ax_system.plot(x3_AU, y3_AU, z3_AU, 'r', label="M3={} kg".format(M3))
    ax_system.legend()
    ax_system.set_title("Body positions", fontsize=20)
    ax_system.set_xlabel("$x_{\\bigoplus}$ (AU)", fontsize=16)
    ax_system.set_ylabel("$y_{\\bigoplus}$ (AU)", fontsize=16)
    
    plt.show()
# #%%
# for i in range(len(x1_AU)):
#     body_1 = ax_system.plot(x1_AU[i], y1_AU[i], z1_AU[i], 'bx')
#     plt.pause(0.0001)
#     body_1 = body_1.pop()
#     body_1.remove()

