"""
Boris pusher for the collimator simulation
import numpy as np
import dolfinx
import dolfinx.geometry
"""
# ----------------------------------------------------------------------------
# 1. Boris push (pure numpy, vectorised over particles)
# ----------------------------------------------------------------------------
def boris_push(v, B, E, q, m, dt):
    """
    Advance velocities one step with the Boris scheme.
    Args:
        - v: (N, 3) array, particle's velocities
        - B: (N, 3) array, magnetic field at particles' positions
        - E : (N, 3) array, electric field at particles' positions
    A negative dt runs the scheme backwards (used for the initial half step back).
    """
    qmdt2 = q * dt / (2.0 * m)
    v_minus = v + qmdt2 * E                      # first half electric step
    t = qmdt2 * B                                # rotation vector
    s = 2.0 * t / (1.0 + np.sum(t * t, axis=1, keepdims=True))
    v_prime = v_minus + np.cross(v_minus, t)
    v_plus = v_minus + np.cross(v_prime, s)      # magnetic rotation
    return v_plus + qmdt2 * E                    # second half electric step


# ----------------------------------------------------------------------------
# 2. Time step selection
# ----------------------------------------------------------------------------
def choose_dt(B, mesh, v_max, q, m, steps_per_gyro=20, cell_fraction=0.5):
    """
    dt <= T_cyclotron / steps_per_gyro   and   dt <= cell_fraction * h_min / v_max
    """
    B_mag_max = np.linalg.norm(B.x.array.real.reshape(-1, 3), axis=1).max()
    T_cyc = 2.0 * np.pi * m / (q * B_mag_max)
    tdim = mesh.topology.dim
    n_local = mesh.topology.index_map(tdim).size_local
    # TODO(check): dolfinx.cpp.mesh.h signature in 0.7.2
    h = dolfinx.cpp.mesh.h(mesh._cpp_object, tdim, np.arange(n_local, dtype=np.int32))
    dt_gyro = T_cyc / steps_per_gyro
    dt_cell = cell_fraction * h.min() / v_max
    r_L = m * v_max / (q * B_mag_max)
    print(f"B_max = {B_mag_max:.3g} T | T_cyc = {T_cyc:.3g} s | r_L(v_max) = {r_L:.3g} m")
    print(f"h_min = {h.min():.3g} m | dt_gyro = {dt_gyro:.3g} s | dt_cell = {dt_cell:.3g} s")
    return min(dt_gyro, dt_cell)


# ----------------------------------------------------------------------------
# 3. Standalone test: uniform B, one particle, compare with analytic orbit
# ----------------------------------------------------------------------------
def test_gyration(q, m, B0=1.0, v0=2e4, n_gyros=5, steps_per_gyro=50):
    B = np.array([[0.0, 0.0, B0]])
    E = np.zeros((1, 3))
    T = 2 * np.pi * m / (q * B0)
    r_L = m * v0 / (q * B0)
    dt = T / steps_per_gyro
    v = np.array([[v0, 0.0, 0.0]])               # velocity at t = 0
    x = np.zeros((1, 3))                         # position at t = 0
    v = boris_push(v, B, E, q, m, -0.5 * dt)     # leapfrog stagger:ß v at t = -dt/2
    xs = []
    for _ in range(n_gyros * steps_per_gyro):
        v = boris_push(v, B, E, q, m, dt)
        x = x + v * dt
        xs.append(x[0].copy())
    xs = np.array(xs)
    # orbit centre for q>0, B along +z, v along +x is at (0, -r_L)
    radii = np.linalg.norm(xs[:, :2] - np.array([0.0, -r_L]), axis=1)
    print(f"|v| drift: {abs(np.linalg.norm(v) - v0) / v0:.2e} (should be small, eg ~1e-15)")
    print(f"Larmor radius analytic {r_L:.5g} m, simulated {radii.mean():.5g} +/- {radii.std():.2e}")

# UP TO HERE
# ----------------------------------------------------------------------------
# 4. LEoPart interface
# ----------------------------------------------------------------------------
def _delete_slots(ptcls, slots):
    """Delete particles by slot index. global_to_local is re-queried for every
    particle because cell-local indices can shift after each deletion."""
    for p in np.asarray(slots, dtype=np.int64).tolist():
        cell, p_local = ptcls.global_to_local(p)
        ptcls.delete_particle(cell, p_local)


def _points_inside_mesh(mesh, bb_tree, pts):
    """Boolean mask: True where pts (N,3) lie inside some cell of the (serial) mesh."""
    pts = np.ascontiguousarray(pts, dtype=np.float64)
    candidates = dolfinx.geometry.compute_collisions_points(bb_tree, pts)
    colliding = dolfinx.geometry.compute_colliding_cells(mesh, candidates, pts)
    return np.diff(colliding.offsets) > 0


def advance_particles(ptcls, mesh, bb_tree, pidxs, x_new,
                      cell_tag_array, absorbing_tags):
    """
    Write new positions for the live particles `pidxs`, then:
      1. delete particles that left the mesh (escaped),
      2. relocate the rest into their new cells,
      3. delete particles that landed in an absorbing (solid) cell region.
    Returns (n_escaped, n_absorbed).
    """
    # 1. write positions. data() is a view onto LeoPart's memory; assign through it.
    xd = ptcls.x().data()
    xd[pidxs] = x_new

    # 2. my own escape test, because relocate_bbox does not document what it does
    #    with a point that is outside every cell.   TODO(check) its behaviour
    inside = _points_inside_mesh(mesh, bb_tree, x_new)
    escaped = pidxs[~inside]
    _delete_slots(ptcls, escaped)
    survivors = pidxs[inside]

    # 3. relocate survivors (never pass deleted slots: "does not check for deleted")
    if survivors.size:
        ptcls.relocate_bbox_on_proc(mesh._cpp_object, survivors.tolist())

    # 4. absorption: re-query live particles in case relocation reassigned slots
    #    TODO(check): whether relocate_bbox_on_proc keeps slot indices stable
    live = np.asarray(ptcls.active_pidxs(), dtype=np.int64)
    n_absorbed = 0
    if live.size:
        cells_now = np.asarray(ptcls.particle_to_cell())[live]
        in_solid = np.isin(cell_tag_array[cells_now], absorbing_tags)
        n_absorbed = int(in_solid.sum())
        if n_absorbed:
            _delete_slots(ptcls, live[in_solid])

    return int(escaped.size), n_absorbed


# ----------------------------------------------------------------------------
# 5. Main loop (serial)
# ----------------------------------------------------------------------------
def run_simulation(mesh, ptcls, B, cell_tags, physicalgroups,
                   n_steps, dt, q, m, E=None, 
                   absorbing_cell_groups=("cathode", "area_surface", "magnets_north", "magnets_south"), # TODO also want reflecting cell groups! "anode", "area_surface", "collimator_surface",  PROBABLY ABSORBING THINGS INSIDE IT. LIKELY SOURCE OF ERROR
                   stagger_initial_velocity=True,
                   output_every=100):
    """
    B : dolfinx Function in DG0 vector space
    E : optional dolfinx Function in DG0 vector space; None means no electric field
    absorbing_cell_groups : cell regions that are solid. A particle ending up in one
        of these cells is counted as absorbed and removed.
    stagger_initial_velocity : treat the stored v as v(t=0) and step it back to
        v(-dt/2), as leapfrog/Boris expects. Set False if v is already staggered.
    """
    assert mesh.comm.size == 1, "serial only: escape detection assumes one process owns the whole mesh"

    tdim = mesh.topology.dim
    imap = mesh.topology.index_map(tdim)
    n_cells = imap.size_local + imap.num_ghosts

    cell_tag_array = np.zeros(n_cells, dtype=np.int32)
    cell_tag_array[cell_tags.indices] = cell_tags.values
    absorbing_tags = np.array([physicalgroups[g]["tag"] for g in absorbing_cell_groups])

    bb_tree = dolfinx.geometry.bb_tree(mesh, tdim)

    B_arr = B.x.array.real.reshape(-1, 3)
    E_arr = None if E is None else E.x.array.real.reshape(-1, 3)

    def sample_fields(pidxs):
        cells = np.asarray(ptcls.particle_to_cell())[pidxs]
        B_p = B_arr[cells]                                   # DG0 -> piecewise constant
        E_p = np.zeros_like(B_p) if E_arr is None else E_arr[cells]
        return B_p, E_p

    n_start = len(ptcls.active_pidxs())
    tally = {"escaped": 0, "absorbed": 0}

    if stagger_initial_velocity:
        pidxs = np.asarray(ptcls.active_pidxs(), dtype=np.int64)
        vd = ptcls.field("v").data()
        B_p, E_p = sample_fields(pidxs)
        vd[pidxs] = boris_push(vd[pidxs], B_p, E_p, q, m, -0.5 * dt)

    for step in range(n_steps):
        pidxs = np.asarray(ptcls.active_pidxs(), dtype=np.int64)
        if pidxs.size == 0:
            print(f"step {step}: no particles left")
            break

        xd = ptcls.x().data()
        vd = ptcls.field("v").data()
        x = xd[pidxs].copy()
        v = vd[pidxs].copy()
        B_p, E_p = sample_fields(pidxs)

        v_new = boris_push(v, B_p, E_p, q, m, dt)
        vd[pidxs] = v_new
        x_new = x + v_new * dt

        n_esc, n_abs = advance_particles(ptcls, mesh, bb_tree, pidxs, x_new,
                                         cell_tag_array, absorbing_tags)
        tally["escaped"] += n_esc
        tally["absorbed"] += n_abs

        if step % output_every == 0:
            live = np.asarray(ptcls.active_pidxs(), dtype=np.int64)
            v_now = ptcls.field("v").data()[live]
            speed = np.linalg.norm(v_now, axis=1)
            print(f"step {step:6d} t={(step + 1) * dt:.3e}s  alive={live.size}/{n_start}  "
                  f"escaped={tally['escaped']} absorbed={tally['absorbed']}  "
                  f"<|v|>={speed.mean() if speed.size else 0:.6g}  "
                  f"mean v={v_now.mean(axis=0) if speed.size else 0}")
            # With E=None, each particle's |v| must stay constant: use as a regression test.

    return tally