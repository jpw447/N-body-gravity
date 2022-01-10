import numpy as np
from scipy.integrate import odeint
import time

G = 6.67e-11

class body_class():
    '''
    Class used to store information about each body being considered. This was initially written to attempt to solve problems using classes,
    but problems were encountered using classes with odeint. So, although technically redundant since using classes proved unnecessary,
    it made some sections of the code shorter and potentially quicker.
    Classes were not written out due to time constraints on the project and are only used within this module.
    '''
    # Defining a body given its identifier (name or number) and its initial conditions (parameters)
    def __init__(self, identifier, parameters, mass):
         # This order of parameters determines the order they must be returned in the function given to odeint
        x, y, z, vx, vy, vz = parameters
        self.number = identifier
        self.mass = mass
        self.x_pos = x
        self.y_pos = y
        self.z_pos = z
        self.vx = vx
        self.vy = vy
        self.vz = vz
        self.KE = 0.5*self.mass*(vx**2 + vy**2 + vz**2)
    
    def separation(self, other_body):
        '''
        Calculates the distance between two bodies, self and other_body
        Args:
            other_body: class instance containing attritubtes of the body to which the distance is calculated.
        
        Returns:
        r : float
            The magnitude of distance vector r to reference body "other_body"
        '''
        r = np.sqrt( (self.x_pos - other_body.x_pos)**2 + \
                     (self.y_pos - other_body.y_pos)**2 + \
                     (self.z_pos - other_body.z_pos)**2 )
        return r
    
    def potential_energy(self, other_body):
        r = self.separation(other_body)
        PE = -G*self.mass*other_body.mass/r
        return PE

    def acceleration_due_to(self, other_body):
        '''
        Finds the acceleration of self due to another body, other_body. Calculates acceleration in x, y, z. If in 2D, then z component is 0 anyway.
        Args:
            other_body: class instance whose influence is being considered in the calculation.
        Returns:
            a_x: acceleration of self in x direction due to other_body.
            a_y: as above, in y direction.
            a_z: as above, in z direction.
        '''
        m_other = other_body.mass
        x, y, z = self.x_pos, self.y_pos, self.z_pos
        x_other, y_other, z_other = other_body.x_pos, other_body.y_pos, other_body.z_pos
        r = self.separation(other_body)
        a_x = -(G*m_other/(r**3)) * (x - x_other)
        a_y = -(G*m_other/(r**3)) * (y - y_other)
        a_z = -(G*m_other/(r**3)) * (z - z_other)

        return a_x, a_y, a_z


def calculate_system_energy(N, bodies_list):
    '''
    Calculates the total energy = kinetic energy + potential energy of the system. Different to energy_calculator module, which handles all positions across all time.
    This particular function deals with only the initial positions using classes.
    Args:
        N: integer number of bodies.
        bodies_list: list of class instances, containing the class method that calculates the potential energy between it and another body.
    '''
    # KE calculated using class methods. PE calculated by summing over bodies.
    KE = sum(body.KE for body in bodies_list)
    PE = 0
    for i in range(N):
        body = bodies_list[i]
        for other_body in bodies_list[i+1:]:
            PE += body.potential_energy(other_body)
    return KE, PE

def field_function(initial_conditions, t, N, masses):
    '''
        Function that takes a list of body objects, as well as a time array. Used exclusively to solve the system using odeint.
        Args:
            parameters: list with initial conditions, containing positions and velocities of 2 bodies
            t: time array used by ode_int
            constants: list containing constants such as Gravitational Constant and masses

        Returns:
            field: array containing the derivatives for the system
    '''
    field = np.zeros(6*N)
    parameters = []
    for i in range(N):
        parameters.append(initial_conditions[6*i:6*(i+1)])
    # Manual summation over every body. This could be vectorised, but wasn't due to time constraints on the project.
    for i in range(N):
        x, y, z, vx, vy, vz = parameters[i]
        ax = 0
        ay = 0
        az = 0
        # Looping through every other body, skipping itself
        for j in range(N):
            if j == i:
                pass
            else:
                # Caclulates accelerations
                x_other, y_other, z_other, vx_other, vy_other, vz_other = parameters[j]
                m_other = masses[j]
                r = np.sqrt( (x-x_other)**2 + (y-y_other)**2 + (z-z_other)**2 )
                ax += -(G*m_other/(r**3)) * (x - x_other)
                ay += -(G*m_other/(r**3)) * (y - y_other)
                az += -(G*m_other/(r**3)) * (z - z_other)

        # Differential equations
        x_dot = vx
        vx_dot = ax
        y_dot = vy
        vy_dot = ay
        z_dot = vz
        vz_dot = az
        field[6*i:6*(i+1)] = x_dot, y_dot, z_dot, vx_dot, vy_dot, vz_dot
    return field

def N_body_solver(N, initial_conditions, masses, time_array):
    '''
    Function that solves for the equations of motion, x(t), y(t), z(t) for N bodies in 3d. If 2d is required, then z and vz should be set to 0.
    It also calculates and returns the initial total energy of the system, and prints to the console how long solving the system took.
    Args:
        N: integer which specifies the number of bodies in the system. Often a good idea to calculate given your initial conditions.
        initial_conditions: array containing the initial conditions for the system, in format [x,y,z,vx,vy,vz]. There must be 6 initial conditons for every body.
        masses: array containing the masses, in the order their initial conditions appear in the initial_conditions array.
        time_array: array passed to odeint over which to solve the system.
    Returns:
        solution: array 6N wide and as long as there are number of points in time_array. Contains solutions for x(t), vx(t) etc.
                    in the same order as initial_conditions. I.e. [x(t), y(t), z(t), vx(t), vy(t), vz(t)] etc.
        system_energy: float which gives the total energy (KE+PE) of the system, given the initial conditions. Useful for checking energy conservation.

    Example:
        The sun and earth system initial conditions should look like this:
            ic = numpy.array([x_sun, y_sun, z_sun, vx_sun, vy_sun, vz_sun, x_earth, y_earth, z_earth, vx_earth, vy_earth, vz_earth])
        N_body_solver will then return an array in the order above (x,y,z,vx,vy,vz), regardless of input.
        See Sun-Earth System.py for a basic example on how the function is called and used.
    '''
    if type(N) is not int:
        print("N must be an integer, and cannot be type "+str(type(N)))
        return
    size = 6*N
    # Checks that enough initial conditions and masses are given for the number of bodies specified. If not, exits the function
    if size != len(initial_conditions):
        print("Invalid number of initial conditions entered. {} were provided, where {} are required. Input all positions followed by all bodies, in order of x, y, z.".format(len(initial_conditions), size))
        return
    if len(masses) != N:
        print("Number of bodies and masses given do not correspond. "+str(len(masses))+" masses were given, yet N="+str(N)+" was specified.")
        return

    # Creating class instances for every body in the system and giving it the initial conditions. Useful to keep code clean elsewhere
    # bodies_list is zero-indexed, i.e. starts with body_0
    parameters = [None]*size
    bodies_list = []
    for i in range(N):
        parameters = initial_conditions[6*i:6*(i+1)]
        exec("global body_"+str(i)+"\n" + \
             "body_"+str(i)+"=body_class(i, parameters, masses[i])\n" + \
              "bodies_list.append(body_"+str(i)+")")
    
    # Calculating initial energy of the system
    KE, PE = calculate_system_energy(N, bodies_list)
    system_energy = KE + PE

    # Solving the system for positions and velocities as functions of time.
    start = time.time()
    solution = odeint(field_function, initial_conditions, time_array, args=(N, masses), rtol=1e-12)
    duration = np.round(time.time() - start, 2)
    print("Solving the system took "+str(duration)+" seconds.")

    return solution, system_energy