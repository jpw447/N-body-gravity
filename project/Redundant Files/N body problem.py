#%%
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import time

G = 6.67e-11

class body_class():
    # Defining a body given its identifier (name or number) and its
    # initial conditions (parameters)
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
        Parameters:
        ------------
        other_body : object
            Class containing attritubtes of the body to which the distance is calculated.
        
        Returns:
        ------------
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
        '''
        m_other = other_body.mass
        mass = self.mass
        x, y, z = self.x_pos, self.y_pos, self.z_pos
        x_other, y_other, z_other = other_body.x_pos, other_body.y_pos, other_body.z_pos
        
        r = self.separation(other_body)

        a_x = -(G*m_other/(r**3)) * (x - x_other)
        a_y = -(G*m_other/(r**3)) * (y - y_other)
        a_z = -(G*m_other/(r**3)) * (z - z_other)

        return a_x, a_y, a_z


def calculate_system_energy(N, bodies_list):
    '''
    Calculates the total energy = kinetic energy + potential energy of the system.
    '''
    KE = sum(body.KE for body in bodies_list)
    PE = 0
    counter = 0
    for i in range(N):
        print("Examining body "+str(i))
        body = bodies_list[i]
        for other_body in bodies_list[i+1:]:
            counter += 1
            print("Interaction number "+str(counter))
            PE += body.potential_energy(other_body)
    return KE, PE

def field_function(initial_conditions, t, N, masses, bodies_list):
    '''
        Function that takes a list of body objects, as well as a time array.
        Returns a list containing the field of differential equations for each derivative.

        Args:
            parameters: list with initial conditions, containing positions and velocities of 2 bodies
            t: time array used by ode_int
            constants: list containing constants such as Gravitational Constant and masses

        Returns:
            field: list containing the derivatives for the system
    '''
    field = np.zeros(6*N)
    parameters = []
    for i in range(N):
        parameters.append(initial_conditions[6*i:6*(i+1)])
    
    # Manual summation over every body
    for i in range(N):
        x, y, z, vx, vy, vz = parameters[i]
        ax = 0
        ay = 0
        az = 0
        # Looping through every other body
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
                
                # Redundant method using classes
                # body = bodies_list[i]
                # other_body = bodies_list[j]
                # a_x, a_y, a_z = body.acceleration_due_to(other_body)
                # ax += a_x
                # ay += a_y
                # az += a_z

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
    size = 6*N
    # Checks that enough initial conditions and masses are given for the number of bodies specified. 
    if size != len(initial_conditions):
        print("Invalid number of initial conditions entered. {} were provided, where {} are required.\
            Input all positions followed by all bodies, in order of x, y, z.".format(len(initial_conditions), size))
        return
    if len(masses) != N:
        print("Number of bodies and masses given do not correspond. "+str(len(masses))+" masses were given, yet N="+str(N)+" was specified.")
        return

    # Creating objects for every body in the system and giving it the initial conditions.
    # Each body is zero-indexed, i.e. starts from body_0
    parameters = [None]*size
    bodies_list = []
    for i in range(N):
        parameters = initial_conditions[6*i:6*(i+1)]
        exec("global body_"+str(i)+"\n" + \
             "body_"+str(i)+"=body_class(i, parameters, masses[i])\n" + \
              "bodies_list.append(body_"+str(i)+")")
    
    # Calculating energy of the system
    KE, PE = calculate_system_energy(N, bodies_list)
    system_energy = KE +PE

    # Solving the system for positions and velocities over time. Returns all quantities calculated in an 2d by N by number_of_points array
    start = time.time()
    solution = odeint(field_function, initial_conditions, time_array, args=(N, masses, bodies_list), rtol=1e-12)
    duration = np.round(time.time() - start, 2)
    print("The calculation took "+str(duration)+" seconds.")

    return solution, system_energy

# Constants
year_constant = 60**2 * 24 * 365.35
G = 6.67e-11
AU = 1.496e11

# Masses
M_sun = 1.989e30
M_mercury = 3.3011e23
M_venus = 4.8675e24
M_earth = 5.972e24
M_moon = 7.346e22
M_mars = 6.4171e23
M_jupiter = 1.898e27
M_saturn = 5.834e26
M_uranus = 8.6813e25
M_neptune = 1.02413e24

# Initial positions
r_mercury = 0.387*AU
r_venus = 0.723*AU
r_earth = 1*AU
r_moon = 3.844e8
r_mars = 1.524*AU
r_jupiter = 5.2*AU
r_saturn = 9.5*AU
r_uranus = 19.201*AU
r_neptune = 30.048*AU

# Initial velocities
v_mercury_init = np.sqrt(G*M_sun/r_mercury)
v_venus_init = np.sqrt(G*M_sun/r_venus)
v_earth_init = np.sqrt(G*M_sun/r_earth)
v_moon_init =  np.sqrt(G*M_earth/r_moon) + v_earth_init
v_mars_init = np.sqrt(G*M_sun/r_mars)
v_jupiter_init = np.sqrt(G*M_sun/r_jupiter)
v_saturn_init = np.sqrt(G*M_sun/r_saturn)
v_uranus_init = np.sqrt(G*M_sun/r_uranus)
v_neptune_init = np.sqrt(G*M_sun/r_neptune)


# Initial conditions format: [x, y, z, vx, vy, vz]
sun_initial_conditions = [0,0,0,0,0,0]
mercury_initial_conditions = [r_mercury, 0, 0, 0, v_mercury_init, 0]
venus_initial_conditions = [r_venus, 0, 0, 0, v_venus_init, 0]
earth_initial_conditions = [r_earth, 0, 0, 0, v_earth_init, 0]
moon_initial_conditions = [r_earth+r_moon, 0, 0, 0, v_moon_init, 0]
mars_initial_conditions = [r_mars, 0, 0, 0, v_mars_init, 0]
jupiter_initial_conditions = [r_jupiter, 0, 0, 0, v_jupiter_init, 0]
saturn_initial_conditons = [r_saturn, 0, 0, 0, v_saturn_init, 0]
uranus_initial_conditions = [r_uranus, 0, 0, 0, v_uranus_init, 0]
neptune_initial_conditions = [r_neptune, 0, 0, 0, v_neptune_init, 0]

# Burrau's Problem
# This reproduces known results over t_max = 10
M1_initial_conditions = [1, 3, 0, 0, 0, 0]
M2_initial_conditions = [1, -1, 0, 0, 0, 0]
M3_initial_conditions = [-2, -1, 0, 0, 0, 0]
M1 = 3
M2 = 5
M3 = 4

ic = sun_initial_conditions+earth_initial_conditions+moon_initial_conditions
masses = np.array([M_sun, M_earth, M_moon])
names = np.array(["Sun", "Earth", "Moon"])

# Creating time array
number_of_points = 10000
time_period = 1
t_max = time_period * year_constant
t_array = np.linspace(0, t_max, number_of_points)
number_of_bodies = int(len(ic)/6)

alpha, system_energy = N_body_solver(N=number_of_bodies, initial_conditions=ic, masses=masses, time_array=t_array)

#%%
# Creating figure, axes and plots

def plotter_2d(fig):
    axis = fig.gca()
    for i in range(number_of_bodies):
        x_start = 6*i
        y_start = x_start+1
        axis.plot(alpha[:,x_start]/AU, alpha[:,y_start]/AU, label=names[i])
    return axis

fig_system = plt.figure(figsize=(8,6))
ax_system = fig_system.gca(projection="3d")
for i in range(number_of_bodies):
    x_start = 6*i
    y_start = x_start+1
    z_start = x_start+2
    ax_system.plot(alpha[:,x_start]/AU - alpha[:,0]/AU, alpha[:,y_start]/AU, alpha[:,z_start]/AU, label=names[i])
ax_system.set_xlabel("$x_{\\bigoplus}$")
ax_system.set_ylabel("$y_{\\bigoplus}$")
ax_system.set_zlabel("$z_{\\bigoplus}$")
ax_system.set_title("The Solar System over a period of "+str(time_period)+" years (3D)")
ax_system.set_xlim(-1.1, 1.1)
ax_system.set_ylim(-1.1, 1.1)
ax_system.set_zlim(-1.1, 1.1)
ax_system.legend()

fig_2d_system = plt.figure(figsize=(8,6))
ax_2d_system = plotter_2d(fig_2d_system)
ax_2d_system.set_title("The Solar System over a period of "+str(time_period)+" years (2D)")
ax_2d_system.set_xlabel("$x_{\\bigoplus}$")
ax_2d_system.set_ylabel("$y_{\\bigoplus}$")
# ax_2d_system.set_aspect("equal")
ax_2d_system.legend()
plt.show()

def velocity_squared(solution):
    velocities = np.zeros((number_of_bodies, number_of_points))
    
    for i in range(number_of_bodies):
        vx_start = 6*i + 3
        vy_start = vx_start + 1
        vz_start = vx_start + 2
        vx_squared = solution[:,vx_start]**2
        vy_squared = solution[:,vy_start]**2
        vz_squared = solution[:,vz_start]**2
        v_squared = vx_squared + vy_squared + vz_squared
        velocities[i,:] = v_squared
        
    return velocities
velocities_squared = velocity_squared(alpha)

# Array of total KE at any given moment
kinetic_energy = sum(0.5 * np.outer(masses.transpose(),velocities_squared))

# Doing it by hand
kinetic_energy_manual = np.zeros(shape=(np.shape(velocities_squared)))
for i in range(number_of_bodies):
    mass = masses[i]
    kinetic_energy_manual[i] = 0.5 * mass * velocities_squared[i,:]
if kinetic_energy.all() == kinetic_energy_manual.all():
    print("Success!")
else:
    print("There's a disparity between methods")

# Also need to calculate potential energy. That can also be vectorised, probably