import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

'''
Hand-crafted solution to the 3 body problem in 2D for the Sun, Jupiter and Saturn. This is a redundant file and may not produce the correct solution.
It was not used in the final report, but existed as a study into the 3 body problem.
'''

if __name__ == "__main__":
    # Constants
    G = 6.67408e-11
    M1 = 1.989e30
    M2 = 5.683e26
    M3 = 1.898e27
    AU = 1.496e11
    MU = 384400e3
    
    # Get division by zero otherwise
    zero = 1e-6
    
    # Position intitial conditions
    x1_init = zero
    y1_init = zero

    x2_init = 1.434e12
    y2_init = zero

    x3_init = 778.5e9
    y3_init = zero
    
    # Separations
    M1_M2_separation = np.sqrt( (x1_init - x2_init)**2 + (y1_init - y2_init)**2 )
    M1_M3_separation = np.sqrt( (x1_init - x3_init)**2 + (y1_init - y3_init)**2 )
    M2_M3_separation = np.sqrt( (x2_init - x3_init)**2 + (y2_init - y3_init)**2 )

    # Velocities    
    vx1_init = zero
    vy1_init = zero
    vx2_init = zero
    vy2_init = np.sqrt(G*M1/M1_M2_separation)
    vx3_init = zero
    vy3_init = np.sqrt(G*M1/M1_M3_separation)
    v1_squared = vx1_init*vx1_init + vy1_init*vy1_init
    v2_squared = vx2_init*vx2_init + vy2_init*vy2_init
    v3_squared = vx3_init*vx3_init + vy3_init*vy3_init

    # Energies
    M1_KE_init = 0.5 * M1 * v1_squared
    M2_KE_init = 0.5 * M2 * v2_squared
    M3_KE_init = 0.5 * M3 * v3_squared
    PE_init = -G*M1*M2/M1_M2_separation - G*M1*M3/M1_M3_separation - G*M2*M3/M2_M3_separation
    KE_init = M1_KE_init + M2_KE_init + M3_KE_init
    initial_energy = KE_init + PE_init

    # Angular momentum
    # Calculations are incorrect, since v is always positive this way
    L1_init = M1 * np.sqrt(v1_squared) * np.sqrt( x1_init*x1_init + y1_init*y1_init )
    L2_init = M2 * np.sqrt(v2_squared) * np.sqrt( x2_init*x2_init + y2_init*y2_init )
    L3_init = M3 * np.sqrt(v3_squared) * np.sqrt( x3_init*x3_init + y3_init*y3_init )
    L_init = L1_init + L2_init + L3_init
    
    # Time array7
    x2_year = 29
    year_constant = 60**2 * 24 * 365.35
    number_of_years = 29
    number_of_points = 40000
    t_max = number_of_years * year_constant
    t_array = np.linspace(0, t_max, number_of_points)

    # Lists containin intial conditions (parameters) and important constants.
    # These appear in a certain order here, and the order must be adheredt to 
    # everywhere else you create a list like this - in the function passed to
    # odeint in both input and output, and what odeint outputs.
    initial_parameters =[x1_init, y1_init, vx1_init, vy1_init,
                         x2_init, y2_init, vx2_init, vy2_init,
                         x3_init, y3_init, vx3_init, vy3_init]
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
        x1, y1, vx1, vy1, x2, y2, vx2, vy2, x3, y3, vx3, vy3 = parameters
        G, M1, M2, M3 = constants

        # Close encounters could be causing bus, by creating huge numbers.
        # Needs some checking.

        # Separation of bodies  
        r_12 = ((x1-x2)*(x1-x2) + (y1-y2)*(y1-y2))**0.5
        r_13 = ((x1-x3)*(x1-x3) + (y1-y3)*(y1-y3))**0.5
        r_23 = ((x2-x3)*(x2-x3) + (y2-y3)*(y2-y3))**0.5

        # ODEs for M1
        x1_dot = vx1
        vx1_dot = -(G*M2/r_12**3) * (x1-x2) - (G*M3/r_13**3) * (x1-x3)
        y1_dot = vy1
        vy1_dot = -(G*M2/r_12**3) * (y1-y2) - (G*M3/r_13**3) * (y1-y3)

        # ODEs for M2
        x2_dot = vx2
        vx2_dot = -(G*M1/r_12**3) * (x2-x1) - (G*M3/r_23**3) * (x2-x3)
        y2_dot = vy2
        vy2_dot = -(G*M1/r_12**3) * (y2-y1) - (G*M3/r_23**3) * (y2-y3)

        # ODEs for M3
        x3_dot = vx3
        vx3_dot = -(G*M1/r_13**3) * (x3-x1) - (G*M2/r_23**3) * (x3-x2)
        y3_dot = vy3
        vy3_dot = -(G*M1/r_13**3) * (y3-y1) - (G*M2/r_23**3) * (y3-y2)

        # Returning ODEs as a list
        field =[x1_dot, y1_dot, vx1_dot, vy1_dot,
                x2_dot, y2_dot, vx2_dot, vy2_dot,
                x3_dot, y3_dot, vx3_dot, vy3_dot]
        return field

    # Passing function to odeint and retrieving planet and star positions
    # Tighter tolerances results in smaller energy deviations
    solution = odeint(field_function, initial_parameters, t_array, args=(constants,), full_output=1, rtol=1e-13)[0]
    
    # solution = array containing 8 columns for x_star, y_star etc. in order they appear in field_function. Each row is t value
    # Columns:
    # 0: x1
    # 1: y1
    # 2: vx1
    # 3: vy1
    # 4: x2
    # 5: y2
    # 6: vx2
    # 7: vy2
    # 8: x3
    # 9: y3
    # 10: vx3
    # 11: vy3

    # Velocities
    vx1, vy1 = solution[:,2], solution[:,3]
    v1_squared = vx1*vx1 + vy1*vy1
    vx2, vy2 = solution[:,6], solution[:,7]
    v2_squared = vx2*vx2 + vy2*vy2
    vx3, vy3 = solution[:,10], solution[:,11]
    v3_squared = vx3*vx3 + vy3*vy3
    
    # Positions
    x1, y1 = solution[:,0], solution[:,1]
    x2, y2 = solution[:,4], solution[:,5]
    x3, y3 = solution[:,8], solution[:,9]
    r_12 = np.sqrt ( (x1 - x2)**2 + (y1 - y2)**2 )
    r_13 = np.sqrt ( (x1 - x3)**2 + (y1 - y3)**2 )
    r_23 = np.sqrt ( (x2 - x3)**2 + (y2 - y3)**2 )

    # Converting to AU for plot
    x1_AU, y1_AU = x1/AU, y1/AU
    x2_AU, y2_AU = x2/AU, y2/AU 
    x3_AU, y3_AU = x3/AU, y3/AU

    # Centres of mass
    x_COM = (M1*x1_AU + M2*x2_AU + M3*x3_AU)/(M1+M2+M3)
    y_COM = (M1*y1_AU + M2*y2_AU + M3*y3_AU)/(M1+M2+M3)

    # Angular momentum
    L1 = M1*np.sqrt(v1_squared) * np.sqrt( x1_AU*x1_AU + y1_AU*y1_AU )
    L2 = M2*np.sqrt(v2_squared) * np.sqrt( x2_AU*x2_AU + y2*AU*y2_AU )
    L3 = M3*np.sqrt(v3_squared) * np.sqrt( x3_AU*x3_AU + y3*AU*y3_AU )
    L = L1 + L2 + L3
    delta_L = (L-L_init)/L_init

    # Energies
    KE = 0.5*M1*v1_squared + 0.5*M2*v2_squared + 0.5*M3*v3_squared
    PE = -G*M1*M2/r_12 - G*M1*M3/r_13 - G*M2*M3/r_23
    energy_difference = (KE + PE - initial_energy)/initial_energy
    print(initial_energy)

    # Creating figures and axes
    fig_planet = plt.figure(figsize=(8,6))
    ax_planet = fig_planet.gca()
    fig_sinusoid = plt.figure(figsize=(8,6))
    ax_sinusoid = fig_sinusoid.gca()
    fig_system_energy = plt.figure(figsize=(8,6))
    ax_system_energy = fig_system_energy.gca()
    fig_phase = plt.figure(figsize=(8,6))
    ax_phase = fig_phase.gca()
    fig_separations = plt.figure(figsize=(8,6))
    ax_separations = fig_separations.gca()
    fig_COM = plt.figure(figsize=(8,6))
    ax_COM = fig_COM.gca()
    fig_momentum = plt.figure(figsize=(8,6))
    ax_momentum = fig_momentum.gca()

    # Plotting separations between bodies
    ax_separations.plot(t_array, r_12, 'k', label="M1-M2 separation")
    ax_separations.plot(t_array, r_13, 'g--', label="M1-M3 separation")
    ax_separations.plot(t_array, r_23, 'r', label="M2-M3 separation")
    ax_separations.legend()
    ax_separations.set_title("Separations between bodies")
    ax_separations.set_xlabel("Time (yrs)")
    ax_separations.set_ylabel("Separation (AU)")

    # Creating plots
    # Planet position x versus y
    ax_planet.plot(x1_AU, y1_AU, 'b', label="M1")
    ax_planet.plot(x2_AU, y2_AU, 'k', label="M2")
    ax_planet.plot(x3_AU, y3_AU, 'r', label="M3")
    ax_planet.legend()
    ax_planet.set_title("Body positions", fontsize=20)
    ax_planet.set_xlabel("$x_{\\bigoplus}$ (AU)", fontsize=16)
    ax_planet.set_ylabel("$y_{\\bigoplus}$ (AU)", fontsize=16)
    ax_planet.set_aspect('equal')

    t_array = t_array / year_constant # Converting time back to years

    # Plotting planet position against time
    sinusoid = (x2_init/AU)*np.cos(2*np.pi/(x2_year)*t_array)
    ax_sinusoid.plot(t_array, x2_AU, 'k')
    ax_sinusoid.plot(t_array, sinusoid, 'r--')
    ax_sinusoid.set_title("M2 Position Versus Time", fontsize=20)
    ax_sinusoid.set_xlabel("$t$ (years)", fontsize=16)
    ax_sinusoid.set_ylabel("$x_{\\bigoplus}$ (AU)", fontsize=16)

    # Difference between theory and simulation
    phase_difference = sinusoid-x2_AU
    ax_phase.plot(t_array, phase_difference, 'k')
    ax_phase.set_title("Prediction VS Simluation for M2", fontsize=20)
    ax_phase.set_xlabel("$t$ (years)", fontsize=16)
    ax_phase.set_ylabel("$x_{\\bigoplus}$ (AU)", fontsize=16)

    # Plotting energies
    ax_system_energy.plot(t_array, energy_difference, 'k')
    ax_system_energy.set_title("Percentage Change of Energy over Time", fontsize=20)
    ax_system_energy.set_xlabel("$t$ (years)", fontsize=16)
    ax_system_energy.set_ylabel("Energy Change (%)", fontsize=16)

    # Centre of mass
    ax_COM.plot(t_array, x_COM, 'r', label="$x_{COM}$")
    ax_COM.plot(t_array, y_COM, 'b', label="$y_{COM}$")
    ax_COM.set_title("Centres of mass for system")
    ax_COM.set_xlabel("$t$ (years)")
    ax_COM.legend()

    # Momentum
    ax_momentum.plot(t_array, L, 'r')
    ax_momentum.set_title("Percentage change of Momentum over time")
    ax_momentum.set_xlabel("$t$ (years)")
    ax_momentum.set_ylabel("Momentum Change (%)")

    plt.show()