# Collimator mesh setup
# Written for python 3.9.25 by Lev Thomson

import gmsh
import numpy as np

# VARIABLES --------------------

run_gui = False # Show the mesh in GMSH GUI

# Thruster / collimator variables:
# TODO note that these are almost entirely dummy variables at this stage. 
# Check with ¿team mech? re: geometry of thruster.
# Do some googling to find collimator & magnet geometry

r_c = 0.0125 # m, = 12.5mm, cathode radius
r_coll = 0.05 # m, = 3cm, collimator radius
r_a = 2*r_c # m, = 25mm, anode radius
r_trig = 0.0005 # m = 0.5mm, trigger radius
kappa = 0.03 # m, = 3cm, cathode-collimator distance
l_c = r_c # m = 12.5mm, length of cathode
l_coll = 0.08 # m, = 10cm, length of collimator
l_a = l_c+kappa # m, = 42.5mm, length of anode
l_trig = l_c * 3/2 # m, = 18.75mm, length of trigger

n_mag = 3 # number of pairs of magnets TODO set range
h_mag = 0.0025 # m, = 2.5mm, height of magnet
l_mag = l_coll # m, = 10cm, length of magnet
w_mag = 2*h_mag # m, = 5mm, width of magnet

# Plasma variables:
debye_length = 0.1 # TODO dummy values for now - define as a field eventually

# Model variables:
size_min = 0.001 # TODO dummy values for now - define as a field eventually
size_max = debye_length

# FUNCTIONS --------------------

def add_magnet_pair(model:gmsh.model, x:float, y:float, z:float, dx:float, dy:float, dz:float, angle:float, last_tag:int, magnet_list:list)->gmsh.model:
    '''
    Adds a pair of magnets at a given point on the collimator-thruster mesh. 
    The magnet is a rectangle defined by a point (x', y', z') and the extents along the x-, y- and z-axes, and is rotated by angle. 
    The second magnet is added diagonally opposite the first.
        Args:
        model: the model to be updated
        x: x-coordinate of one corner of box
        y: y-coordinate of one corner of box
        z: z-coordinate of one corner of box
        dx: x-coordinate of other corner of box
        dy: y-coordinate of other corner of box
        dz: z-coordinate of other corner of box
        angle: (radians) the rectangle is rotated about the x-axis by this angle
        last_tag: the tag of the entity to add the magnets to

    Returns:
        model: updated gmsh model
        last_tag: the tag of the updated model
        magnet_list: list of magnet surfaces by [(dimension=2, tag)] (for defining field later)
    '''
    # Add first magnet
    mag_1_vol = model.occ.addBox(x,-y,-z, dx, dy, dz, tag=last_tag+1)
    model.occ.rotate([(3, mag_1_vol)], x+dx/2,0,0, 1,0,0, angle)

    # Add second magnet opposite the first
    mag_2_vol = model.occ.addBox(x,-y,-z, dx, dy, dz, tag=last_tag+2)
    model.occ.rotate([(3, mag_2_vol)], x+dx/2,0,0, 1,0,0, angle+np.pi)

    gmsh.model.occ.synchronize() # Cut both at the same time - reduces synchronise calls. 
    
    model.occ.cut([(3, last_tag)], [(3, mag_1_vol)], tag=last_tag+3, removeObject=True, removeTool=False) # Cut first magnet
    model.occ.cut([(3, last_tag+3)], [(3, mag_2_vol)], tag=last_tag+4, removeObject=True, removeTool=False) # Cut second magnet

    mag_1_surf = model.getBoundary([(3, mag_1_vol)], combined=False, oriented=False)
    mag_2_surf = model.getBoundary([(3, mag_2_vol)], combined=False, oriented=False)
    magnet_list.append(mag_1_surf)
    magnet_list.append(mag_2_surf)

    gmsh.model.occ.remove([(3, mag_1_vol)], recursive=False) # Then remove the volumes leaving the bounding surfaces intact
    gmsh.model.occ.remove([(3, mag_2_vol)], recursive=False)

    #collimator_surf = model.getBoundary([(3, mag_1_vol)], combined=False, oriented=False)

    # UP TO HERE trying to remove the magnet surface that touches the collimator

    gmsh.model.occ.synchronize()

    # Update tag
    last_tag = last_tag+4
    return model, last_tag, magnet_list

# MAIN -------------------------

# ~~~~ Set parameters ~~~~

gmsh.initialize()
gmsh.model.add("collimator")

#gmsh.option.setNumber("Mesh.MeshSizeMin", size_min) # Set a global minimum and maximum mesh size
#gmsh.option.setNumber("Mesh.MeshSizeMax", size_max)

# ~~~~ Define geometry ~~~~

vol_coll = gmsh.model.occ.addCylinder(l_a,0,0, l_coll,0,0, r_coll, 1) # Collimator cylinder
vol_a = gmsh.model.occ.addCylinder(0,0,0, l_a,0,0, r_a,2) # Anode cylinder
vol_coll_a = gmsh.model.occ.fuse([(3, vol_coll)], [(3, vol_a)], 3)[0] # Fuse collimator to anode. Should look like 2 cylinders stacked on top of each other

# Then remove the cathode from the anode cylinder
vol_c = gmsh.model.occ.addCylinder(0,0,0, l_c,0,0, r_c, 4)
vol_trig = gmsh.model.occ.addCylinder(0,0,0, l_trig,0,0, r_trig, 5)
vol_c_trig = gmsh.model.occ.fuse([(3, vol_c)], [(3, vol_trig)], 6)[0] # Fuse cathode and trigger
vol_thruster = gmsh.model.occ.cut(vol_coll_a, vol_c_trig, 7)[0] # Cut the cathode-trigger out of the anode-collimator assembly

print("Made collimator")

# Add magnets:
if n_mag == 0:
    pass
elif n_mag > 0:
    last_tag = vol_thruster[0][1] # Tag of most-recently created entity
    angle_array = np.linspace(0, np.pi, n_mag+1)[0:n_mag] # Generate array of equally spaced magnet pairs around spherical cathode surface
    magnet_list = []
    for loop_idx, angle in enumerate(angle_array):
        gmsh.model, last_tag, magnet_list = add_magnet_pair(gmsh.model, l_a, r_coll, w_mag/2, l_mag, h_mag, w_mag, angle, last_tag, magnet_list)

gmsh.model.occ.synchronize()

print("Added magnets")

# ~~~~ Define mesh field ~~~~

# Get magnet boundary
magnet_array = np.zeros((n_mag*2*6))
for mag_idx, magnet in enumerate(magnet_list):
    for surf_idx, surface in enumerate(magnet):
        magnet_array[6*mag_idx+surf_idx] = surface[1]

# Eliminate surfaces that are shared by all magnets - this causes the mesh to be fine near the ends of the collimator
unique_surfs_vals, unique_surfs_counts = np.unique(magnet_array, return_counts=True)

for val_idx, value in enumerate(unique_surfs_vals):
    if unique_surfs_counts[val_idx] == n_mag*2: # If shared by all magnets
        magnet_array = magnet_array[magnet_array != value] # Remove all instances of that surface



gmsh.model.mesh.field.add("Distance", 1)
gmsh.model.mesh.field.setNumbers(1, "SurfacesList", magnet_array)
gmsh.model.mesh.field.setNumber(1, "Sampling", 1000)

gmsh.model.mesh.field.add("Threshold", 2)
gmsh.model.mesh.field.setNumber(2, "InField", 1) # Create a mesh field that defines mesh size based on proximity to magnets
gmsh.model.mesh.field.setNumber(2, "SizeMin", size_min)
gmsh.model.mesh.field.setNumber(2, "SizeMax", size_max)
gmsh.model.mesh.field.setNumber(2, "DistMin", 0)
gmsh.model.mesh.field.setNumber(2, "DistMax", 0.0005) # 0.5mm
gmsh.model.mesh.field.setAsBackgroundMesh(2)

for surf in magnet_array:
    gmsh.model.occ.remove([(2, surf)], recursive=False)

print("Defined mesh")

# ~~~~ Save & output to GUI ~~~~

gmsh.model.occ.synchronize()

print("Generating mesh...")

# ~~~~ Parallelize ~~~~

gmsh.option.setNumber("General.NumThreads", 5) # Use multithreading to generate mesh faster
# gmsh.option.setNumber("Mesh.Algorithm", 10)
gmsh.option.setNumber("Mesh.Algorithm3D", 4) # HXT - works better with multithreading. Front-Delunay has issues with overlapping facets on the mesh

# ~~~~ Mesh ~~~~

gmsh.model.mesh.generate(3)

gmsh.write("collimator.msh")

if run_gui:
    gmsh.fltk.run()

gmsh.finalize()

print("Done")