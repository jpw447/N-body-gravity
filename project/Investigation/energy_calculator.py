import numpy as np

G = 6.67e-11

def velocity_squared(solution, N, number_of_points):
    '''
    A function that returns the square of all velocities at all times for all bodies in the system by cycling through the solution and grabbing each velocity vector.
    Args:
        solution: array containing the solution of an N-body system evolving under gravity. Created from N_body_solver in the 
                    N_body_simulator.py module.
        N: integer number of bodies.
        number_of_points: integer number of time steps used in N_body_solver simulation.
    Returns:
        velocities: array containing the square velocities of each body within the system for each point in time.
    '''
    velocities = np.zeros((N, number_of_points))
    for i in range(N):
        vx_start = 6*i + 3
        vy_start = vx_start + 1
        vz_start = vx_start + 2
        vx_squared = solution[:,vx_start]**2
        vy_squared = solution[:,vy_start]**2
        vz_squared = solution[:,vz_start]**2
        v_squared = vx_squared + vy_squared + vz_squared
        velocities[i,:] = v_squared
        
    return velocities

def PE_calculator(N, masses, x_positions, y_positions, z_positions):
    '''
    A simple function that calculates the potential energy of the system given N bodies. Used in system_energy function below.
    This is a vectorised calculation.
    Args:
        N: integer number of bodies.
        masses: array containing masses in order of their initial conditions given to N_body_solver.
        x_positions: array containing all x positions for all bodies across all time.
        y_positions: as above, for y.
        z_positions: as above, for z.
    Returns:
        PE: array of system potential energy across all time.
    '''
    PE = 0
    # Cycles through each body i
    for i in range(N):
        # Then every other body, skipping itself and preventing double counting
        for j in range(i, N):
            if j == i:
                pass
            else:
                separation = np.sqrt( (x_positions[i,:] - x_positions[j,:])**2 +\
                                    (y_positions[i,:] - y_positions[j,:])**2 +\
                                    (z_positions[i,:] - z_positions[j,:])**2 )
                PE += -G*masses[i]*masses[j]/separation
    return PE

def system_energy(solution, masses, N, number_of_points):
    '''
    This function calculates the energy of the system over the run-time of the simulation. It grabs positions for each body over all time,
    calculates the potential energy of the system and kinetic energy of the bodies, and adds them. The function then returns the system
    energy over all time.
    Args:
        solution: array containing the positions and velocities of all bodies, produced by N_body_simulator.
        masses: array with the masses of each body.
        N: integer number of bodies in the simulation.
        number_of_points: integer number of time steps

    Returns:
        system_energy: array containing the system energy at every point in time.
    '''
    # Grabbing all positions
    x_positions = np.zeros([N, number_of_points])
    y_positions = np.zeros([N, number_of_points])
    z_positions = np.zeros([N, number_of_points])
    for i in range(N):
        x_start = 6*i
        y_start = x_start+1
        z_start = x_start+2
        x_positions[i,:] = solution[:, x_start]
        y_positions[i,:] = solution[:, y_start]
        z_positions[i,:] = solution[:, z_start]

    # PE term
    PE = PE_calculator(N, masses, x_positions, y_positions, z_positions)

    # Calculating square velocity and kinetic energy
    velocities_squared = velocity_squared(solution, N, number_of_points)
    kinetic_energy = np.zeros((N, number_of_points))
    for i in range(N):
        mass = masses[i]
        kinetic_energy[i,:] = 0.5 * mass * velocities_squared[i,:]
    kinetic_energy = kinetic_energy.sum(axis=0)

    system_energy = kinetic_energy + PE
    return system_energy, kinetic_energy, PE