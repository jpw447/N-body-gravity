import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

'''
Hand-crafted solution to the 2 body problem in 2D for the Sun Jupiter system. This is a redundant file and may not produce the correct solution.
It was not used in the final report, but existed as a study into the 2 body problem.
'''

def two_body_field_function(parameters, t, constants):
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
        x1, y1, vx1, vy1, x2, y2, vx2, vy2 = parameters
        G, M1, M2 = constants

        # Separation of bodies  
        r_12 = ((x1-x2)*(x1-x2) + (y1-y2)*(y1-y2))**0.5
        # r_13 = ((x1-x3)*(x1-x3) + (y1-y3)*(y1-y3))**0.5
        # r_23 = ((x2-x3)*(x2-x3) + (y2-y3)*(y2-y3))**0.5

        # ODEs for star
        x1_dot = vx1
        vx1_dot = -(G*M2/r_12**3) * (x1-x2)
        y1_dot = vy1
        vy1_dot = -(G*M2/r_12**3) * (y1-y2)

        # ODEs for Earth
        x2_dot = vx2
        vx2_dot = -(G*M1/r_12**3) * (x2-x1)
        y2_dot = vy2
        vy2_dot = -(G*M1/r_12**3) * (y2-y1)

        # Returning ODEs as a list
        field = [x1_dot, y1_dot, vx1_dot, vy1_dot,
                x2_dot, y2_dot, vx2_dot, vy2_dot]
        return field

def three_body_field_function(parameters, t, constants):
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
        vy1_dot = -(G*M2/r_12**3) * (y1-y2) - (G*M3/r_13**3) * (x1-x3)

        # ODEs for M2
        x2_dot = vx2
        vx2_dot = -(G*M1/r_12**3) * (x2-x1) - (G*M3/r_23**3) * (x2-x3)
        y2_dot = vy2
        vy2_dot = -(G*M1/r_12**3) * (y2-y1) - (G*M3/r_23**3) * (x2-x3)

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

if __name__ == "__main__":
    # Constants
    G = 6.67408e-11
    M1 = 1.989e30
    M2 = 1.898e27
    M3 = 5.683e26
    AU = 1.496e11
    MU = 384400e3
    
    # Get division by zero otherwise
    zero = 1e-6
    
    # Position intitial conditions
    x1_init = zero
    y1_init = zero

    x2_init = 778.5e9
    y2_init = zero

    x3_init = 1.434e12
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
    L1_init = M1 * np.sqrt(v1_squared) * np.sqrt( x1_init*x1_init + y1_init*y1_init )
    L2_init = M2 * np.sqrt(v2_squared) * np.sqrt( x2_init*x2_init + y2_init*y2_init )
    L3_init = M3 * np.sqrt(v3_squared) * np.sqrt( x3_init*x3_init + y3_init*y3_init )
    L_init = L1_init + L2_init + L3_init
    
    # Time array
    year_constant = 60**2 * 24 * 365.35
    points_per_year = 1000
    number_of_years = 26*5
    number_of_points = number_of_years*points_per_year
    t_max = number_of_years * year_constant
    t_array = np.linspace(0, t_max, number_of_points)

    # Lists containin intial conditions (parameters) and important constants.
    # These appear in a certain order here, and the order must be adheredt to 
    # everywhere else you create a list like this - in the function passed to
    # odeint in both input and output, and what odeint outputs.
    initial_parameters_two_body = [x1_init, y1_init, vx1_init, vy1_init,
                         x2_init, y2_init, vx2_init, vy2_init]

    initial_parameters_three_body = [x1_init, y1_init, vx1_init, vy1_init,
                                 x2_init, y2_init, vx2_init, vy2_init,
                                 x3_init, y3_init, vx3_init, vy3_init]
    constants_two_body = [G, M1, M2]
    constants_three_body = [G, M1, M2, M3]

    # Passing function to odeint and retrieving planet and star positions
    # Tighter tolerances results in smaller energy deviations
    solution_jupiter = odeint(two_body_field_function, initial_parameters_two_body, t_array, args=(constants_two_body,), rtol=1e-10) 
    solution_jupiter_saturn = odeint(three_body_field_function, initial_parameters_three_body, t_array, args=(constants_three_body,), rtol=1e-10) 
    
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

    # Velocities
    vx1_jupiter, vy1_jupiter = solution_jupiter[:,2], solution_jupiter[:,3]
    v1_squared_jupiter = vx1_jupiter*vx1_jupiter + vy1_jupiter*vy1_jupiter
    vx2_jupiter, vy2_jupiter = solution_jupiter[:,6], solution_jupiter[:,7]
    v2_squared_jupiter = vx2_jupiter*vx2_jupiter + vy2_jupiter*vy2_jupiter
    
    # Positions
    x1_jupiter, y1_jupiter = solution_jupiter[:,0], solution_jupiter[:,1]
    x2_jupiter, y2_jupiter = solution_jupiter[:,4], solution_jupiter[:,5]
    r_jupiter = np.sqrt ( (x1_jupiter - x2_jupiter)**2 + (y1_jupiter - y2_jupiter)**2 )
    x1_AU_jupiter, y1_AU_jupiter = x1_jupiter/AU, y1_jupiter/AU
    x2_AU_jupiter, y2_AU_jupiter = x2_jupiter/AU, y2_jupiter/AU # Converting to AU for plot

    # Energies
    KE = 0.5*M1*v1_squared_jupiter + 0.5*M2*v2_squared_jupiter
    PE = -G*M1*M2/r_jupiter
    energy_difference_jupiter = (KE + PE - initial_energy)/initial_energy

    # Creating figures and axes
    fig_planet = plt.figure(figsize=(8,6))
    ax_planet = fig_planet.gca()
    fig_sinusoid = plt.figure(figsize=(8,6))
    ax_sinusoid = fig_sinusoid.gca()
    fig_resonance = plt.figure(figsize=(8,6))
    ax_resonance = fig_resonance.gca()

    t_array = t_array / year_constant # Converting time back to years

    fig_phase = plt.figure(figsize=(8,6))
    ax_phase = fig_phase.gca()

    x1, y1 = solution_jupiter_saturn[:,0], solution_jupiter_saturn[:,1]
    x2, y2 = solution_jupiter_saturn[:,4], solution_jupiter_saturn[:,5]
    x3, y3 = solution_jupiter_saturn[:,8], solution_jupiter_saturn[:,9]
    r_12 = np.sqrt( (x1 - x2)**2 + (y1 - y2)**2 )
    r_13 = np.sqrt( (x1 - x3)**2 + (y1 - y3)**2 )
    r_23 = np.sqrt( (x2 - x3)**2 + (y2 - y3)**2 )
    x1_AU, y1_AU = x1/AU, y1/AU
    x2_AU, y2_AU = x2/AU, y2/AU # Converting to AU for plot
    x3_AU, y3_AU = x3/AU, y3/AU

    ax_planet.set_title("Jupiter, Saturn and Sun System")
    ax_planet.plot(x1_AU, y1_AU, 'b', label="Sun")
    ax_planet.plot(x2_AU, y2_AU, 'r', label="Jupiter")
    ax_planet.plot(x3_AU, y3_AU, 'g', label="Saturn")
    ax_planet.legend()

    ax_sinusoid.set_title("Orbit Cycle for Jupiter and Jupiter-Saturn Systems")
    ax_sinusoid.plot(t_array, x2_AU_jupiter, 'k--', label="Jupiter System")
    ax_sinusoid.plot(t_array, x2_AU, 'r', label="Jupiter-Saturn System")
    ax_sinusoid.legend()

    ax_resonance.set_title("Saturn and Jupiter Cycles")
    ax_resonance.plot(t_array, x2_AU, 'r', label="Jupiter")
    ax_resonance.plot(t_array, x3_AU, 'g', label="Saturn")
    ax_resonance.legend()

    ax_phase.set_title("Comparing Orbital Positions for Solo and Coupled Systems")
    ax_phase.plot(t_array, r_jupiter, 'r', label="Jupiter System")
    ax_phase.plot(t_array, r_12, 'g', label="Jupiter-Saturn System")
    ax_phase.legend()
    plt.show()
#%%
plt.close()
fig = plt.figure(figsize=(8,6))
ax = fig.gca()
for i in range(len(x2_AU)):
    x1 = x1_AU[i]
    x2 = x2_AU[i]
    x3 = x3_AU[i]
    y1 = y1_AU[i]
    y2 = y2_AU[i]
    y3 = y3_AU[i]
    ax.plot(x1, y1, 'bx')
    ax.plot(x2, y2, 'rx')
    ax.plot(x3, y3, 'gx')
    plt.pause(0.0001)
