import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

try: import pyvista
except ModuleNotFoundError:
    pass # has conflicts with gmsh - don't need this for gmsh_collimator script

# ANDERS VARIABLES / FUNCTIONS -----------------------------
class AndersVars():
    def __init__(self):
        self.z_dist = 9e-2 # m, cathode-transparency distance for results in fig 4, Anders et al, 1996

    def distribute_particles_uncollimated(self, n_p:int, v_total:float, random_state:int=None)->np.array:
        '''
        Generates an array of n_p particles each with total velocity v_total.
        The particles' directions are randomly distributed according to the uncollimated 
        cobalt film deposition distribution from figure 4, Anders et. al, 1996. 
        Args:
            - n_p: total number of particles to generate
            - v_total: magnitude of velocity for the particles. Note: at this stage, all particles are generated with the same velocity
            - random_state: Optional (default = None). Seed for rng engine. 
        Output: 
            - (n_p, 3) array of particle velocities in cartesian coordinates
        '''

        # Create pdf with 2d film distribution for plasma particles
        anders = AndersUncollimated(name="anders_uncollimated")
        dist_samples = anders.rvs(size=n_p, random_state=random_state)

        # This gives the 2d radial component of position at a height of z = 9cm from the cathode surface
        pos_cyl = np.zeros((n_p, 3)) # [r, θ, z]

        pos_cyl[:, 0] = dist_samples # m
        pos_cyl[:, 2] = self.z_dist # m

        # This distribution is symmetrical around all angles theta, so we expect that the 
        # angular distribution will be linearly distributed from [0, 2 pi)
        if random_state is None:
            angle = stats.uniform.rvs(loc=0, scale=np.pi, size=n_p, random_state=random_state)
        else:
            angle = stats.uniform.rvs(loc=0, scale=np.pi, size=n_p, random_state=random_state+1) # Require different random seed for this generator

        pos_cyl[:, 1] = angle # rad

        # Convert to cartesian coords.
        pos_cart = np.zeros_like(pos_cyl)
        # pos_cart[:, 0] = pos_cyl[:,0] * np.cos(pos_cyl[:, 1]) # m, x
        # pos_cart[:, 1] = pos_cyl[:,0] * np.sin(pos_cyl[:, 1]) # m, y
        # pos_cart[:, 2] = pos_cyl[:,2] # m, z 

        # Note that the particles are travelling along the z axis, so we rotate these coordinates like so:
        # axes 0, 1, 2 = Z, X, Y

        pos_cart[:, 2] = pos_cyl[:,0] * np.cos(pos_cyl[:, 1]) # m, z
        pos_cart[:, 1] = pos_cyl[:,0] * np.sin(pos_cyl[:, 1]) # m, x
        pos_cart[:, 0] = pos_cyl[:,2] # m, y

        # Normalise to get contribution of x, y, z to total velocity
        norm = np.linalg.norm(pos_cart, axis=1)
        pos_cart = pos_cart / norm[:, None]

        # Final velocity
        pos_cart = pos_cart * v_total

        return pos_cart

class AndersUncollimated(stats.rv_continuous):
    """Piecewise exponential - quadratic - exponential distribution
    (Anders et al., params from Lev Thomson's Matlab code)."""

    # Prefixed with p_ to avoid clashing with rv_continuous attributes (.a, .b, etc.)
    p_xL, p_xR = -0.01791, 0.02035
    p_A, p_B = 38.75, 49.65      # left exponential
    p_D, p_E = 40.51, -41.8      # right exponential
    p_g, p_h, p_k = -0.4401, 0.002474, 0.001523   # quadratic

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._P = lambda x: self.p_g*x**3/3 + self.p_h*x**2/2 + self.p_k*x  # antiderivative

        poly = lambda x: self.p_g*x**2 + self.p_h*x + self.p_k
        self.polyL, self.polyR = poly(self.p_xL), poly(self.p_xR)
        self.p_c = np.log(self.polyL/self.p_A) - self.p_B*self.p_xL
        self.p_f = np.log(self.polyR/self.p_D) - self.p_E*self.p_xR

        # Unnormalised areas of each region
        self.areaL = self.polyL / self.p_B
        self.areaM = self._P(self.p_xR) - self._P(self.p_xL)
        self.areaR = self.polyR / (-self.p_E)
        self.Z = self.areaL + self.areaM + self.areaR

    def _pdf(self, x):
        x = np.asarray(x, dtype=float)
        out = np.piecewise(
            x,
            [x < self.p_xL, x > self.p_xR],
            [lambda x: self.p_A*np.exp(self.p_B*x + self.p_c),
             lambda x: self.p_D*np.exp(self.p_E*x + self.p_f),
             lambda x: self.p_g*x**2 + self.p_h*x + self.p_k],
        )
        return out / self.Z

    def _cdf(self, x):
        x = np.asarray(x, dtype=float)
        out = np.piecewise(
            x,
            [x < self.p_xL, x > self.p_xR],
            [lambda x: self.polyL/self.p_B * np.exp(self.p_B*(x - self.p_xL)),
             lambda x: self.Z - self.polyR/(-self.p_E) * np.exp(self.p_E*(x - self.p_xR)),
             lambda x: self.areaL + self._P(x) - self._P(self.p_xL)],
        )
        return out / self.Z

# GMSH VARIABLES -------------------------------------------

class CollimatorVars():
    def __init__(self):

        # Thruster / collimator variables:
        # TODO note that these are arbitrary values at this stage. 
        # Check with team mech re: geometry of thruster.
        # Choose magnets as well. 

        # Note that these are arbitrary values
        self.r_c = 0.0125  # m, = 12.5mm, cathode radius
        self.r_coll = 0.05  # m, = 3cm, collimator radius
        self.r_a = 2 * self.r_c  # m, = 25mm, anode radius
        self.r_trig = 0.001  # m = 1mm, trigger radius
        self.kappa = 0.03  # m, = 3cm, cathode-collimator distance
        self.l_c = self.r_c  # m = 12.5mm, length of cathode
        self.l_coll = 0.08  # m, = 10cm, length of collimator
        self.l_a = self.l_c + self.kappa  # m, = 42.5mm, length of anode
        self.l_trig = self.l_c * 3 / 2  # m, = 18.75mm, length of trigger

        self.n_mag = 3  # number of pairs of magnets
        self.h_mag = 0.0025  # m, = 2.5mm, height of magnet
        self.l_mag = self.l_coll  # m, = 10cm, length of magnet
        self.w_mag = 2 * self.h_mag  # m, = 5mm, width of magnet

        self.l_area = self.l_coll + self.l_a # m, length of area around collimator. Must encompass the whole collimator-anode shape.
        self.r_area = self.r_coll + self.l_coll # m, radius of area around collimator. Must extend at least a magnet-length away from the collimator to allow A->0 at this boundary

        # Plasma variables:
        self.debye_length = 0.005


        # Model variables:
        self.size_min = 0.001
        self.size_max = self.debye_length


        # File variables:
        self.meshname = "collimator.msh"  # Mesh will be saved to this file


# FENICSX VARIABLES -----------------------------------------------------------------

class PlasmaVars():
    def __init__(self):

        self.V_c = 0 # V, electric potential at surface of cathode
        self.V_a = 3000 # V, electric potential at surface of anode
        self.v_ion_cathode = 15e3 # m s^-1, initial ion velocity at the cathode

        self.n_p = 100 # Number of plasma ions in the simulation (note: this value will be rounded up if it doesn't divide evenly into the trigger surface cells)
        self.q_ion = 3.1 # Mean charge on a molybdenum ion (Yushkov et al, 2000)
        self.m_ion = 95.95 # g mol^-1 mass of molybdenum ion
        self.v_ion = 1.74e4 # ms^-1 ion velocity (Yushkov et al, 2000) TODO Not ND specific yet

        self.t_trig = 5e-3 # s, time at which the trigger stops firing

    def V_trig(self, t): # V, electric potential at surface of trigger as a function of time
        '''
        Args:
            - t: time (in seconds)
        Returns:
            - electric potential at surface of trigger at time t
        '''
        if t < 0:
            raise ValueError("Time must be greater than or equal to zero")
        elif t < self.t_trig: # While triggering
            return 3000
        else: # Not triggering
            return self.V_c


# PYVISTA VARIABLES
class PlottingVars():
        def __init__(self):
            self.show_plot = True 
            self.run_gui = False # Show the mesh in GMSH GUI     
            # set False for an interactive window. 
            # This isn't going to work on Fermi (no GPU?) but would probably work on NS computers in future.

            self.glyph_size = 0.05 # arrow size scale for 3d quiverplot

        def render_particles(self, fname, view, filled, outline, cloud, t):
            '''
            Plotting function to render particles within a mesh, then save the plot to a png.
            Args:
                - fmame: filename for png
                - view: "view_isometric", "view_xy", "view_yz" or "view_xz". Which angle to screenshot the plot at.
                - filled: the mesh of the filled cells
                - cloud: the particles
                - outline: the mesh surrounding the filled cells (i.e., the rest of the collimator)
                - t: time at which particles are plotted (ms)
            '''
            p = pyvista.Plotter(off_screen=plotting_vars.run_gui, window_size=(1400, 1000))
            p.add_mesh(outline, style="wireframe", color="lightgrey", opacity=0.3)
            p.add_mesh(filled, color="lightblue", opacity=0.15, show_edges=True)
            p.add_mesh(cloud, color="red", point_size=6, render_points_as_spheres=True)
            p.add_axes()
            getattr(p, view)()
            p.add_title(f"Positions of plasma ions at time t = {str(t)} ms")
            p.screenshot('images/'+fname)
            p.close()
            return

# CONSTANTS ----------------------------------------------------------------

class Constants():
    def __init__(self):
        self.mu0 = 4 * np.pi * 1e-7
        self.mu_r_magnet = 1.05  # Dimensionless, for NdFeB magnet (source: https://www.advancedmagnets.com/custom-magnets/)

        self.Br = 1.2 # T, magnetic field remanence, e.g. N42 NdFeB

# Instantiate these classes for import into scripts

anders_vars = AndersVars()
collimator_vars = CollimatorVars()
plasma_vars = PlasmaVars()
plotting_vars = PlottingVars()
constants = Constants()