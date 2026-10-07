import numpy as np

# GMSH VARIABLES -------------------------------------------

run_gui = False  # Show the mesh in GMSH GUI


# Thruster / collimator variables:
# TODO note that these are arbitrary values at this stage. 
# Check with team mech re: geometry of thruster.
# Choose magnets as well. 

# Note that these are arbitrary values
r_c = 0.0125  # m, = 12.5mm, cathode radius
r_coll = 0.05  # m, = 3cm, collimator radius
r_a = 2 * r_c  # m, = 25mm, anode radius
r_trig = 0.001  # m = 1mm, trigger radius
kappa = 0.03  # m, = 3cm, cathode-collimator distance
l_c = r_c  # m = 12.5mm, length of cathode
l_coll = 0.08  # m, = 10cm, length of collimator
l_a = l_c + kappa  # m, = 42.5mm, length of anode
l_trig = l_c * 3 / 2  # m, = 18.75mm, length of trigger


n_mag = 3  # number of pairs of magnets
h_mag = 0.0025  # m, = 2.5mm, height of magnet
l_mag = l_coll  # m, = 10cm, length of magnet
w_mag = 2 * h_mag  # m, = 5mm, width of magnet


l_area = l_coll + l_a # m, length of area around collimator. Must encompass the whole collimator-anode shape.
r_area = r_coll + l_coll # m, radius of area around collimator. Must extend at least a magnet-length away from the collimator to allow A->0 at this boundary

# Plasma variables:

debye_length = 0.005


# Model variables:

size_min = 0.001
size_max = debye_length


# File variables:

meshname = "collimator.msh"  # Mesh will be saved to this file


# FENICSX VARIABLES -----------------------------------------------------------------

V_c = 0 # V, electric potential at surface of cathode
V_a = 3000 # V, electric potential at surface of anode


t_trig = 5e-3 # s, time at which the trigger stops firing
def V_trig(t): # V, electric potential at surface of trigger as a function of time
    '''
    Args:
        - t: time (in seconds)
    Returns:
        - electric potential at surface of trigger at time t
    '''
    if t < 0:
        raise ValueError("Time must be greater than or equal to zero")
    elif t < t_trig: # While triggering
        return 3000
    else: # Not triggering
        return V_c

n_p = 10 # Number of plasma ions in the simulation
q_ion = 3.1 # Mean charge on a molybdenum ion (Yushkov et al, 2000)
m_ion = 95.95 # g mol^-1 mass of molybdenum ion
v_ion = 1.74e4 # ms^-1 ion velocity (Yushkov et al, 2000) TODO Not ND specific yet

# PYVISTA VARIABLES

show_plot = True          
# set False for an interactive window. 
# This isn't going to work on Fermi (no GPU?) but would probably work on NS computers in future.

glyph_size = 0.05 # arrow size scale for 3d quiverplot

# CONSTANTS ----------------------------------------------------------------

mu0 = 4 * np.pi * 1e-7
mu_r_magnet = 1.05  # Dimensionless, for NdFeB magnet (source: https://www.advancedmagnets.com/custom-magnets/)

Br = 1.2 # T, magnetic field remanence, e.g. N42 NdFeB