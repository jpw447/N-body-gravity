[![Open in Visual Studio Code](https://classroom.github.com/assets/open-in-vscode-f059dc9a6f8d3a56e377f745f24479a46679e63a5d9fe6f495e02850cd0d8118.svg)](https://classroom.github.com/online_ide?assignment_repo_id=6385273&assignment_repo_type=AssignmentRepo)
# ScientificComputingProject-jpw447
# N-body Gravity Simulation (Joe Williams)
## Description
---
This is a project creating an gravitational N-body simulation for the 3rd year module "Scientific Computing" as part of my Physics MSci at the University of Nottingham. This was produced over 5 weeks between 15/11/2021 and 17/12/2021.
By studying a system evolving under gravity using Newton's Law of Gravity and Newton's 2nd Law, a system of differential equations can be produced which describe the system well. In the case of 2 bodies, these are analytically solvable for the motion of the bodies in time, whereas for 3 bodies or more, they become analytically solvable. For this reason, numerical integration is required to find a "solution".

For this project, systems of coupled differential equations were found by hand for 2 and 3 body systems, and then solved numerically in Python using the `odeint` module from the `scipy` library. Examples of this can be seen in `Redundant Files/2D Systems/Sun-Earth-Moon System.py`. A solver was then coded to solve any 3D system with N bodies of any period of time, producing the `N_body_simulator.py` library, whose module `N_body_solver` actually solves the system. The documentation can be read in any instance of the `N_body_simulator.py` file in this repository.

Using this module, the Sun-Earth system was simulated, succeeded by the Kepler-19 system and the TRAPPIST-1 system, which was disturbed by an interloping Jupiter mass and studied. Its influence and impact on two surrounding bodies, TRAPPIST-1e and TRAPPIST-1f, was examined, with a key focus on how TRAPPIST-1e's orbit was affected with a varying interloper mass between 0.1 and 14.1 Jupiter masses. The figures and captions produced for this can be found in `N Body Simulations.pdf`; note not every figure is reproducible, since the scripts have been changed multiple times and may not resemble their form when the figure was made.

Due to some inefficiencies that could be improved by vectorising operations or introducting the `Numba` library to `N_body_simulator`, `N_body_solver` can take several seconds to solve a system. For example, the Kepler-19 system takes around 16 seconds to run. The maximum time provided to the time array given to `N_body_solver` primarily dictates how long solving the system can run.

## Important Notes regarding `N_body_solver`
---
The module works well in the vast majority of cases. It does, however, exhibit issues in close encounters. If too many or too few time steps are provided to `odeint`, then the integrator can crash and cause everything to apparently fall to the origin. This occurs in extremely close encounters, since the integrator struggles to keep up with the provided time steps. This can cause a massive acceleration of a body due to Newton's Law of Gravity being an inverse square law, and therefore a short, local violation of conservation of energy. This means future orbits beyond a close encounter can be unreliable.

![crash](https://i.imgur.com/kwRtJZu.png)

This is an example of the code running well (left) and crashing (right). This was induced by providing an insufficient number of time steps for the simulation to accurately solve the system to illustrate what a crash can look like. `odeint` will provide an output like:
```
ODEintWarning: Excess work done on this call (perhaps wrong Dfun type). Run with full_output = 1 to get quantitative information.  warnings.warn(warning_msg, ODEintWarning)
```
and the solver will report the system was solved in a time period shorter than expected. To solve this, try running the simulation over a shorter time period or providing more time steps.

The code is also not fantastically efficient. It runs within reasonable timeframes, all of which are a minute or under except for `TRAPPIST - Jupiter Study.py`, which takes around 30 minutes to run, depending on the parameters provided.

The run-time for each file is dominated primarily by the N_body_solver module, which calculates the solution to the system given some initial conditions. All other operations take a comparably negligible amount of time. Whenever N_body_solver is called, it will output how long the solution took to calculate to the console.

## Requirements
---
This project uses the `numpy`, `matplotlib`, `scipy` and `time` libraries. It additionally uses the libraries `N_body_simulator`, `fourier_transforms` and `energy_calculator`, which have been also been written using the aforementioned libraries. When running a file containing `N_body_solver`, the library `N_body_simulator` must be in the same folder, or the working directory changed so the library can be found. To prevent this issue on other computers, the same instance of `N_body_simulator` exists in each folder where it is called within a file.

## How to Run the Code and `N_body_solver`
---
In VSCode, which was used to develop this project, simply run the file using Ctrl+F5 and wait. When the system has been solved, a message such as "`Solving the system took 61.11 seconds.`" will be outputted. With the files in this repository, plots will subsequently be made. For some files with many data points, such as `TRAPPIST - Jupiter Study.py`, the figures can take some time to appear.

The initial conditions of each body can be changed by adjusting the variable values as you see them in the file, although the initial conditions arrays have been constructed to be in the correct format for `N_body_solver`, described in the module documentation. All initial conditions follow the format:

[x1, y1, z1, vx1, vy1, vz1, x2, y2, z, ...]

where x1 is the x position of body 1, vx1 the x-component of the velocity for body 1, x2 the x position of body 2 and so on. As an example, `Sun-Earth System.py` contains:

```
x_earth = 1*AU
v_earth = np.sqrt(G*M_sun/x_earth)
sun_ic = [0,0,0,0,0,0]
earth_ic = [x_earth,0,0,0,v_earth,0]
ic = sun_ic + earth_ic

N = int(len(ic)/6)
masses = [M_sun, M_earth]
names = ["Sun", "Earth"]
```

By changing the value of `x_earth`, you can set where Earth is on the x axis initially. The sun has initial conditions at (x,y,z)=(0,0,0) and is at rest. The earth is at (x,y,z)=(1AU,0,0) and has an orbital velocity of $+\sqrt{\frac{GM_{sun}}{1AU}}$ in the y direction. N is automatically calculated in this instance, and masses and names can be appended if additional bodies are supplied.

If you encounter an error in running the code that is not `odeint` related, check the output in the console - `N_body_solver` may tell you what went wrong in a printed statement, as it has possibly rejected your input based on an insufficient number of initial conditions for a specified number of bodies. For instance, if 12 initial conditions are given and N=3 is specified, the console will read:

```
Invalid number of initial conditions entered. 12 were provided, where 18 are required. Input all positions followed by all bodies, in order of x, y, z.
```
Followed by the Python error in the file. In the provided files in the `Investigation` folder, this will not happen. `N_body_solver` will not recognise if you accidentally input a velocity as a position, etc.

Suggested files to test run are `Sun-Earth System.py`, `Kepler-19.py` and `TRAPPIST System.py`. This is due to the fact that they run in approximately 90 seconds or less and produce interesting figures to investigate. Other fies in the `Redundant Files` folder can also be run and should work, but may produce incorrect physical results and none of the figures produced by these files were used in the final report. They exist here purely for interest and for my own reference and comparison as I was coding.

**NOTE:** If you are coding your own file and wish to use `N_body_solver`, `energy_calculator` or `fourier_transforms` then you **MUST** put these files in the same folder as your code, or change your working directory so that these libraries can be found.

## Figures
---
Most of the figures in the report are reproducible. Figure 5 is not, however, since it came from a previous iteration of `TRAPPIST System.py` which no longer produces this figure. Instead, Figure 6 will be produced. If you run the codes as they are, they will produce the figures using the initial conditions written in the file already.

| Figure | File to run |
| ------ | ------- |
| 1 | `Sun-Earth System.py` |
| 2 | `Sun-Earth System.py` |
| 3 | `Kepler-19.py` |
| 4 | `Kepler-19.py` | 
| 5 | N/A |
| 6 | `TRAPPIST System.py` |
| 7 | `TRAPPIST System.py` |
| 8 | `TRAPPIST System.py` |
| 9 | `TRAPPIST - Jupiter Study.py` | 
| 10 | `Star Cluster.py` |

It is worth noting `TRAPPIST - Jupiter Study.py` produces some redundant figures, since it is an edited copy of `TRAPPIST System.py`.

## Expected Run-times
---
These are the runtimes found from running the scripts on my personal computer in VSCode. Some tests show that Spyder is capable of running some scripts a few seconds faster, although the difference is not significant.
| File | Expected Runtime |
| --- | --- | 
| `Sun-Earth System.py` | 0.15s |
| `Kepler-19.py` | 23.3s |
| `Star Cluster.py` | 61s | 
| `TRAPPIST System.py` | 84s | 
| `TRAPPIST - Juputer Study.py` | 30 mins+ |
