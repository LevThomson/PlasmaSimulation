"""
Boris pusher for the collimator simulation
"""
import numpy as np
import dolfinx
import matplotlib.pyplot as plt
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
    v_minus = v + qmdt2 * E # first half electric step
    t = qmdt2 * B # rotation vector
    s = 2.0 * t / (1.0 + np.sum(t * t, axis=1, keepdims=True))
    v_prime = v_minus + np.cross(v_minus, t)
    v_plus = v_minus + np.cross(v_prime, s) # magnetic rotation
    return v_plus + qmdt2 * E # second half electric step


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
# def test_gyration(q, m, B0=1.0, v0=2e4, n_gyros=5, steps_per_gyro=50):
#     B = np.array([[0.0, 0.0, B0]])
#     E = np.zeros((1, 3))
#     T = 2 * np.pi * m / (q * B0)
#     r_L = m * v0 / (q * B0)
#     dt = T / steps_per_gyro
#     v = np.array([[v0, 0.0, 0.0]])               # velocity at t = 0
#     x = np.zeros((1, 3))                         # position at t = 0
#     v = boris_push(v, B, E, q, m, -0.5 * dt)     # leapfrog stagger:ß v at t = -dt/2
#     xs = []
#     for _ in range(n_gyros * steps_per_gyro):
#         v = boris_push(v, B, E, q, m, dt)
#         x = x + v * dt
#         xs.append(x[0].copy())
#     xs = np.array(xs)
#     # orbit centre for q>0, B along +z, v along +x is at (0, -r_L)
#     radii = np.linalg.norm(xs[:, :2] - np.array([0.0, -r_L]), axis=1)
#     print(f"|v| drift: {abs(np.linalg.norm(v) - v0) / v0:.2e} (should be small, eg ~1e-15)")
#     print(f"Larmor radius analytic {r_L:.5g} m, simulated {radii.mean():.5g} +/- {radii.std():.2e}")

def test_gyration(q, m, B0=1.0, v0=2e4, n_gyros=5, steps_per_gyro=50, plot=True):
    B = np.array([[0.0, 0.0, B0]])
    E = np.zeros((1, 3))
    T = 2 * np.pi * m / (q * B0)
    r_L = m * v0 / (q * B0)
    dt = T / steps_per_gyro

    v = np.array([[v0, 0.0, 0.0]]) # velocity at t = 0
    x = np.zeros((1, 3)) # position at t = 0
    v = boris_push(v, B, E, q, m, -0.5 * dt) # leapfrog stagger: v at t = -dt/2

    n_steps = n_gyros * steps_per_gyro
    xs = np.empty((n_steps, 3))
    speeds = np.empty(n_steps + 1)
    speeds[0] = np.linalg.norm(v) # speed after the half-step stagger

    for i in range(n_steps):
        v = boris_push(v, B, E, q, m, dt)
        x = x + v * dt
        xs[i] = x[0]
        speeds[i + 1] = np.linalg.norm(v)

    # orbit centre for q>0, B along +z, v along +x is at (0, -r_L)
    centre = np.array([0.0, -r_L])
    radii = np.linalg.norm(xs[:, :2] - centre, axis=1)

    KE = 0.5 * m * speeds**2
    dKE = np.diff(KE) # energy change per step [J]
    rel_drift = (speeds - v0) / v0 # cumulative relative |v| error

    print(f"|v| drift: {abs(speeds[-1] - v0) / v0:.2e} (should be small, eg ~1e-15)")
    print(f"Larmor radius analytic {r_L:.5g} m, simulated {radii.mean():.5g} +/- {radii.std():.2e}")
    print(f"Max |dKE| per step: {np.abs(dKE).max():.2e} J  (KE0 = {KE[0]:.3e} J)")

    if plot:
        fig, axs = plt.subplots(1, 2, figsize=(10, 4.5))

        # 1) Particle path vs analytic circle
        theta = np.linspace(0, 2 * np.pi, 400)
        axs[0].plot(centre[0] + r_L * np.cos(theta), centre[1] + r_L * np.sin(theta),
                    "k--", lw=1, label="Analytic Larmor orbit")
        axs[0].plot(xs[:, 0], xs[:, 1], ".-", ms=3, lw=0.8, label="Boris pusher")
        axs[0].plot(0, 0, "ro", label="Start")
        axs[0].set_aspect("equal")
        axs[0].set_xlabel("x [m]")
        axs[0].set_ylabel("y [m]")
        axs[0].set_title(f"Gyration in B = {B0} T ẑ")
        axs[0].legend(fontsize=8)

        # # 2) Energy change per step
        # steps = np.arange(1, n_steps + 1)
        # axs[1].plot(steps, dKE / KE[0], lw=0.8)
        # axs[1].set_xlabel("Step")
        # axs[1].set_ylabel(r"$\Delta KE / KE_0$ per step")
        # axs[1].set_title("Energy change per step")
        # axs[1].ticklabel_format(axis="y", style="sci", scilimits=(0, 0))

        # 3) Cumulative relative speed drift
        axs[1].plot(np.arange(n_steps + 1), rel_drift, lw=0.8)
        axs[1].set_xlabel("Step")
        axs[1].set_ylabel(r"$(|v| - v_0)/v_0$")
        axs[1].set_title("Cumulative |v| drift")
        axs[1].ticklabel_format(axis="y", style="sci", scilimits=(0, 0))

        fig.tight_layout()
        plt.savefig("images/gyration_test.png")

    return xs, speeds

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
    Returns: 
    - n_escaped 
    - n_absorbed
    """
    # write positions. data() is a view onto LeoPart's memory, can be used to assign values to particles
    xd = ptcls.x().data()
    xd[pidxs] = x_new

    # remove whatever particles have left the mesh
    inside = _points_inside_mesh(mesh, bb_tree, x_new)
    escaped = pidxs[~inside]
    _delete_slots(ptcls, escaped)
    survivors = pidxs[inside]

    # relocate survivors
    if survivors.size:
        ptcls.relocate_bbox_on_proc(mesh._cpp_object, survivors.tolist())

    # absorption: re-query live particles in case relocation reassigned slots
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
                   absorbing_cell_groups=("cathode", "area_surface", "magnets_north", "magnets_south"), # TODO also want reflecting cell groups! "anode", "area_surface", "collimator_surface". Previously this was absorbing the particles almost immediately, which was the source of the error.
                   stagger_initial_velocity=True,
                   output_every=100,
                   on_step=None):
    """
    Simulates particles in the collimator environment.
    Args:
    - on_step: optional callback on_step(n_steps_done, t, ptcls). Called once before
        the first step (n=0, t=0) and again after every completed step. Pass a
        particle_animation.TrajectoryRecorder to record frames for an animation.
    - B: dolfinx Function in DG0 vector space
    - E: optional dolfinx Function in DG0 vector space; None means no electric field
    - absorbing_cell_groups: cell regions that are solid. A particle ending up in one
        of these cells is counted as absorbed and removed.
    - stagger_initial_velocity: treat the stored v as v(t=0) and step it back to
        v(-dt/2), as leapfrog/Boris expects. Set False if v is already staggered.
    """
    assert mesh.comm.size == 1, "parallel not supported"

    tdim = mesh.topology.dim
    imap = mesh.topology.index_map(tdim)
    n_cells = imap.size_local + imap.num_ghosts

    cell_tag_array = np.zeros(n_cells, dtype=np.int32)
    cell_tag_array[cell_tags.indices] = cell_tags.values
    absorbing_tags = np.array([physicalgroups[g]["tag"] for g in absorbing_cell_groups if g in physicalgroups])

    bb_tree = dolfinx.geometry.bb_tree(mesh, tdim)

    B_arr = B.x.array.real.reshape(-1, 3)
    E_arr = None if E is None else E.x.array.real.reshape(-1, 3)

    def sample_fields(pidxs):
        cells = np.asarray(ptcls.particle_to_cell())[pidxs]
        B_p = B_arr[cells] # DG0 -> piecewise constant
        E_p = np.zeros_like(B_p) if E_arr is None else E_arr[cells]
        return B_p, E_p

    n_start = len(ptcls.active_pidxs())
    tally = {"escaped": 0, "absorbed": 0}

    if stagger_initial_velocity:
        pidxs = np.asarray(ptcls.active_pidxs(), dtype=np.int64)
        vd = ptcls.field("v").data()
        B_p, E_p = sample_fields(pidxs)
        vd[pidxs] = boris_push(vd[pidxs], B_p, E_p, q, m, -0.5 * dt)

    if on_step is not None:
        on_step(0, 0.0, ptcls) # frame 0: initial state

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

        if on_step is not None:
            on_step(step + 1, (step + 1) * dt, ptcls)

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

 
# ----------------------------------------------------------------------------
# 1. Recording
# ----------------------------------------------------------------------------
class TrajectoryRecorder:
    """
    Callable for run_simulation(on_step=...). Snapshots the live particles at t = 0 and
    after every `every`-th step. Each particle keeps its LeoPart slot index as a stable
    id, so trails can follow individual particles even as others are removed.
    """
 
    def __init__(self, every=1):
        self.every = max(int(every), 1)
        self.steps, self.times = [], []
        self.ids, self.x, self.v = [], [], []
 
    def __call__(self, n_done, t, ptcls):
        if n_done % self.every != 0:
            return
        ids = np.asarray(ptcls.active_pidxs(), dtype=np.int64)
        x = np.array(ptcls.x().data())[ids] # make a copy instead of viewing the data, so it saves correctly
        v = np.array(ptcls.field("v").data())[ids]
        if x.shape[1] == 2: # 2-D mesh: pad z with zeros
            x = np.c_[x, np.zeros(len(x))]
        self.steps.append(int(n_done))
        self.times.append(float(t))
        self.ids.append(ids)
        self.x.append(x)
        self.v.append(v)
 
    @property
    def n_frames(self):
        return len(self.times)
 
    def bounds(self):
        """(xmin, xmax, ymin, ymax, zmin, zmax) over every recorded position."""
        pts = np.concatenate([x for x in self.x if len(x)]) if any(len(x) for x in self.x) \
            else np.zeros((1, 3))
        lo, hi = pts.min(axis=0), pts.max(axis=0)
        return (lo[0], hi[0], lo[1], hi[1], lo[2], hi[2])
 
    # -- persistence, so a long run can be rendered later / elsewhere -------------
    def save(self, path):
        counts = np.array([len(i) for i in self.ids], dtype=np.int64)
        cat = lambda L, shape: np.concatenate(L) if counts.sum() else np.zeros(shape)
        np.savez_compressed(
            path, every=self.every, steps=np.array(self.steps), times=np.array(self.times),
            counts=counts, ids=cat(self.ids, (0,)), x=cat(self.x, (0, 3)), v=cat(self.v, (0, 3)))
 
    @classmethod
    def load(cls, path):
        d = np.load(path)
        rec = cls(every=int(d["every"]))
        rec.steps, rec.times = d["steps"].tolist(), d["times"].tolist()
        edges = np.concatenate([[0], np.cumsum(d["counts"])])
        for a, b in zip(edges[:-1], edges[1:]):
            rec.ids.append(d["ids"][a:b])
            rec.x.append(d["x"][a:b])
            rec.v.append(d["v"][a:b])
        return rec
 
 
# ----------------------------------------------------------------------------
# 2. Static scenery (needs dolfinx; kept separate so render_animation does not)
# ----------------------------------------------------------------------------
DEFAULT_REGION_STYLES = {
    "magnets_north": dict(color="dimgray", opacity=0.85),
    "magnets_south": dict(color="dimgray", opacity=0.85),
    "anode": dict(color="steelblue", opacity=0.35),
    "cathode": dict(color="darkorange", opacity=0.6),
    "collimator_surface": dict(color="dimgray", opacity=0.6),
}
 
 
def build_background(mesh, cell_tags, physicalgroups, region_styles=None, outline=True):
    """
    Returns a list of (pyvista_mesh, add_mesh_kwargs) for the static parts of the scene:
    the domain bounding box plus the surfaces of the cell regions named in
    `region_styles` (defaults: magnets, anode, cathode).
    """
    import pyvista
    import dolfinx.plot
 
    styles = DEFAULT_REGION_STYLES if region_styles is None else region_styles
    tdim = mesh.topology.dim
    topology, cell_types, geometry = dolfinx.plot.vtk_mesh(mesh, tdim)
    grid = pyvista.UnstructuredGrid(topology, cell_types, geometry)
 
    scene = []
    if outline:
        scene.append((grid.outline(), dict(color="black", line_width=1)))
    for name, style in styles.items():
        if name not in physicalgroups:
            continue
        cells = cell_tags.find(physicalgroups[name]["tag"])
        if len(cells) == 0:
            continue
        scene.append((grid.extract_cells(cells).extract_surface(), dict(style)))
    return scene
 
 
# ----------------------------------------------------------------------------
# 3. Rendering
# ----------------------------------------------------------------------------
def _time_label(t, t_max):
    if t_max < 1e-6:
        return f"{t * 1e9:.1f} ns"
    if t_max < 1e-3:
        return f"{t * 1e6:.2f} µs"
    return f"{t * 1e3:.3f} ms"
 
 
def _scalar(v, color_by):
    if color_by == "speed":
        return np.linalg.norm(v, axis=1)
    axis = {"vx": 0, "vy": 1, "vz": 2}[color_by]
    return v[:, axis]
 
 
def _frame_camera(plotter, bounds):
    """Keep the current viewing direction, but centre on `bounds` and back off just far
    enough to fit them (reset_camera(bounds=...) is not reliable across PyVista versions)."""
    b = np.asarray(bounds, dtype=float).reshape(3, 2)
    centre = b.mean(axis=1)
    radius = 0.5 * np.linalg.norm(b[:, 1] - b[:, 0])
    cam = plotter.camera
    direction = np.array(cam.position) - np.array(cam.focal_point)
    direction /= np.linalg.norm(direction)
    dist = radius / np.sin(np.radians(cam.view_angle) / 2.0)
    cam.focal_point = centre
    cam.position = centre + direction * dist
    plotter.reset_camera_clipping_range()
 
 
def _trail_polydata(rec, frame, length):
    """Line segments joining each particle's last `length` recorded positions."""
    import pyvista
    segs = []
    for j in range(max(1, frame - length + 1), frame + 1):
        _, ia, ib = np.intersect1d(rec.ids[j - 1], rec.ids[j], return_indices=True)
        if len(ia):
            segs.append(np.stack([rec.x[j - 1][ia], rec.x[j][ib]], axis=1).reshape(-1, 3))
    if not segs:
        return None
    pts = np.concatenate(segs)
    n = len(pts) // 2
    lines = np.column_stack([np.full(n, 2), np.arange(0, 2 * n, 2), np.arange(1, 2 * n, 2)])
    return pyvista.PolyData(pts, lines=lines.ravel())
 
 
def render_animation(rec, filename="particles.gif", background=None,
                     color_by="vx", cmap="cividis", clim=None,
                     fps=15, frame_stride=1, trail_length=0, trail_color="dimgray",
                     point_size=4, view="isometric", camera_bounds="particles", zoom=1.0,
                     window_size=(1000, 750), title="Plasma ions in the collimator", writer="auto"):
    """
    Render the recorded frames to .gif (default) or .mp4.
    args:
    - background: list from build_background(); None draws particles only
    - color_by: "vx" (thrust axis), "vy", "vz" or "speed"
    - frame_stride: render every n-th recorded frame (1 = every timestep)
    - trail_length: draw each particle's last N steps as a line (0 = off)
    - trail_color: colour of those lines
    - view: "isometric", "xy", "xz" or "yz"
    - camera_bounds: "particles" frames the recorded trajectories (recommended when the
                    mesh domain is much larger than the collimator), None frames the whole
                    scene, or pass (xmin, xmax, ymin, ymax, zmin, zmax)
    - clim: fixed colour limits (min, max); default spans the whole run
    - writer: how frames become a file
                    "auto"    Pillow (inbuilt PyVista package) or png.
                    "pillow"  GIF assembled with Pillow
                    "png"     just write numbered PNG frames to <filename stem>_frames/,
                              assemble later with frames_to_gif() or ffmpeg
    """
    import pyvista
 
    if rec.n_frames == 0:
        raise ValueError("Recorder is empty: pass it to run_simulation(on_step=...) first")
 
    frames = sorted(set(range(0, rec.n_frames, max(int(frame_stride), 1))) | {rec.n_frames - 1})
    if len(frames) > 600 and filename.lower().endswith(".gif"):
        print(f"Note: {len(frames)} frames will make a very large GIF; consider "
              "frame_stride > 1, fewer steps, or an .mp4 filename.")
 
    # colour limits fixed across the whole run so colours mean the same in every frame
    if clim is None:
        allv = [_scalar(v, color_by) for v in rec.v if len(v)]
        lo = min(a.min() for a in allv) if allv else 0.0
        hi = max(a.max() for a in allv) if allv else 1.0
        clim = (lo, hi) if hi > lo else (lo - 0.5, hi + 0.5)
    label = {"vx": "vx, thrust (m/s)", "vy": "vy (m/s)", "vz": "vz (m/s)",
             "speed": "speed (m/s)"}[color_by]
 
    plotter = pyvista.Plotter(off_screen=True, window_size=list(window_size))
    for scene_mesh, kwargs in (background or []):
        plotter.add_mesh(scene_mesh, reset_camera=False, **kwargs)
    if background is None:
        plotter.add_axes()
 
    getattr(plotter, f"view_{view}")()
    if camera_bounds == "particles":
        b = np.array(rec.bounds(), dtype=float).reshape(3, 2)
        pad = 0.25 * max(np.ptp(b, axis=1).max(), 1e-6) # never a zero-size box
        b[:, 0] -= pad
        b[:, 1] += pad
        _frame_camera(plotter, b.ravel())
    elif camera_bounds is not None:
        _frame_camera(plotter, np.asarray(camera_bounds, dtype=float))
    if zoom != 1.0:
        plotter.camera.zoom(zoom)
 
    is_mp4 = filename.lower().endswith(".mp4")
    if writer == "auto":
        writer = "png" if is_mp4 else "pillow"
    if is_mp4 and writer == "pillow":
        raise ValueError("Pillow cannot write .mp4. Use writer='png' and run ffmpeg on the "
                         "frames, or give a .gif filename.")
 
    pil_frames, frame_dir = [], None
    if writer == "png":
        import os
        frame_dir = os.path.splitext(filename)[0] + "_frames"
        os.makedirs(frame_dir, exist_ok=True)
    elif writer != "pillow":
        raise ValueError(f"unknown writer {writer!r}")
 
    n0, t_max = max(len(rec.ids[0]), 1), max(rec.times)
    bar_added = False
    for k in frames:
        pos, vel = rec.x[k], rec.v[k]
        if len(pos):
            cloud = pyvista.PolyData(pos.astype(float))
            cloud["values"] = _scalar(vel, color_by)
            plotter.add_mesh(
                cloud, name="particles", scalars="values", cmap=cmap, clim=clim,
                point_size=point_size, render_points_as_spheres=True, reset_camera=False,
                show_scalar_bar=not bar_added,
                scalar_bar_args=dict(title=label + "\n", vertical=True, height=0.6,
                                     position_x=0.88, position_y=0.2))
            bar_added = True
        else:
            try:
                plotter.remove_actor("particles")
            except Exception:
                pass
 
        if trail_length > 0:
            trails = _trail_polydata(rec, k, trail_length)
            if trails is not None:
                plotter.add_mesh(trails, name="trails", color=trail_color, opacity=0.6,
                                 line_width=1.5, reset_camera=False)
            else:
                try:
                    plotter.remove_actor("trails")
                except Exception:
                    pass
 
        plotter.add_text(
            f"{title}\nt = {_time_label(rec.times[k], t_max)}  (step {rec.steps[k]})\n"
            f"alive: {len(pos)} / {n0}",
            name="hud", position="upper_left", font_size=10)
        if writer == "png":
            plotter.screenshot(os.path.join(frame_dir, f"frame_{k:06d}.png"))
        else: # pillow: keep 8-bit frames
            from PIL import Image
            img = Image.fromarray(np.asarray(plotter.screenshot(return_img=True)))
            pil_frames.append(img.convert("P", palette=Image.Palette.ADAPTIVE, colors=256))
 
    plotter.close()
    if writer == "pillow":
        pil_frames[0].save(filename, save_all=True, append_images=pil_frames[1:],
                           duration=int(round(1000 / fps)), loop=0, disposal=2)
        print(f"Wrote {len(frames)} frames to {filename}")
    elif writer == "png":
        print(f"Wrote {len(frames)} PNG frames to {frame_dir}/\n"
              f"  GIF:  frames_to_gif('{frame_dir}', '{filename}', fps={fps})\n"
              f"  MP4:  ffmpeg -framerate {fps} -i {frame_dir}/frame_%06d.png "
              "-pix_fmt yuv420p out.mp4   (frame numbers have gaps if frame_stride > 1; "
              "use -pattern_type glob -i '*.png' instead)")
    else:
        print(f"Wrote {len(frames)} frames to {filename}")
    return filename
 
 
def frames_to_gif(frame_dir, filename, fps=15):
    """Assemble numbered PNGs into a GIF using only Pillow (run in any environment)."""
    import glob
    import os
    from PIL import Image
    paths = sorted(glob.glob(os.path.join(frame_dir, "*.png")))
    if not paths:
        raise FileNotFoundError(f"no PNG frames in {frame_dir}")
    imgs = [Image.open(p).convert("RGB").convert("P", palette=Image.Palette.ADAPTIVE, colors=256)
            for p in paths]
    imgs[0].save(filename, save_all=True, append_images=imgs[1:],
                 duration=int(round(1000 / fps)), loop=0, disposal=2)
    print(f"Wrote {len(imgs)} frames to {filename}")
 
 
def export_vtk(rec, folder):
    """
    Write each frame as particles_NNNNNN.vtp plus particles.pvd (a time series ParaView
    opens as one animated dataset). Point arrays: v (vector), speed, id. Needs pyvista only
    to build the files; ParaView/VTK can then animate and export GIF/MP4 itself.
    """
    import os
    import pyvista
    os.makedirs(folder, exist_ok=True)
    entries = []
    for k in range(rec.n_frames):
        if len(rec.x[k]) == 0:
            continue # nothing to show this frame
        cloud = pyvista.PolyData(rec.x[k].astype(float))
        cloud["v"] = rec.v[k]
        cloud["speed"] = np.linalg.norm(rec.v[k], axis=1)
        cloud["id"] = rec.ids[k]
        name = f"particles_{k:06d}.vtp"
        cloud.save(os.path.join(folder, name))
        entries.append(f'    <DataSet timestep="{rec.times[k]:.9e}" part="0" file="{name}"/>')
    with open(os.path.join(folder, "particles.pvd"), "w") as f:
        f.write('<?xml version="1.0"?>\n<VTKFile type="Collection" version="0.1">\n'
                '  <Collection>\n' + "\n".join(entries) + "\n  </Collection>\n</VTKFile>\n")
    print(f"Wrote {len(entries)} frames + particles.pvd to {folder}/")
 
 
# ----------------------------------------------------------------------------
# Simulate & animate
# ----------------------------------------------------------------------------
def animate_run(mesh, ptcls, B, cell_tags, physicalgroups, n_steps, dt,
                q_ion, m_ion, filename="particles.gif", record_every=1, 
                save_npz=None, run_kwargs=None, region_styles=None, **render_kwargs):
    """
    Runs the Boris pusher loop and then animates the particles. 
    Args:
    - run_kwargs   : extra keyword arguments for run_simulation (E=..., q=..., m=..., ...)
    - save_npz     : path to also save the raw recording (for re-rendering later)
    - render_kwargs: anything render_animation accepts (trail_length, fps, view, ...)
    Returns:
    - Recorder
    - Tally
    """
    rec = TrajectoryRecorder(every=record_every)
    tally = run_simulation(mesh, ptcls, B, cell_tags, physicalgroups,
                           n_steps=n_steps, dt=dt, q=q_ion, m=m_ion,
                           on_step=rec, **(run_kwargs or {}))
    if save_npz:
        rec.save(save_npz)
    background = build_background(mesh, cell_tags, physicalgroups, region_styles)
    render_animation(rec, filename, background=background, **render_kwargs)
    return rec, tally
