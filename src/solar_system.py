import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# ============================================================
# SOLAR SYSTEM N-BODY SIMULATOR V1
# ============================================================
# 9 bodies, orbital-elements initialization, N-body gravity,
# conservation diagnostics, 2D plots, and full-system 3D animation.
#
# NOTE:
# The orbital elements below are representative starting values.
# They are not precision epoch-specific ephemeris states.
# ============================================================

# ============================================================
# 1. CONSTANTS
# ============================================================

G = 6.67430e-11
AU = 1.495978707e11

SECONDS_PER_DAY = 86400.0
DAYS_PER_YEAR = 365.25
SECONDS_PER_YEAR = SECONDS_PER_DAY * DAYS_PER_YEAR


# ============================================================
# 2. BODY DATABASE
# ============================================================

bodies = [
    {"name": "Sun", "mass": 1.98847e30, "a_AU": 0.0, "e": 0.0,
     "i_deg": 0.0, "Omega_deg": 0.0, "omega_deg": 0.0, "nu_deg": 0.0},

    {"name": "Mercury", "mass": 3.3011e23, "a_AU": 0.3871, "e": 0.2056,
     "i_deg": 7.005, "Omega_deg": 48.33, "omega_deg": 29.12, "nu_deg": 40.0},

    {"name": "Venus", "mass": 4.8675e24, "a_AU": 0.7233, "e": 0.0068,
     "i_deg": 3.3947, "Omega_deg": 76.68, "omega_deg": 54.89, "nu_deg": 120.0},

    {"name": "Earth", "mass": 5.9722e24, "a_AU": 1.0000, "e": 0.0167,
     "i_deg": 0.000, "Omega_deg": 0.0, "omega_deg": 102.94, "nu_deg": 180.0},

    {"name": "Mars", "mass": 6.4171e23, "a_AU": 1.5237, "e": 0.0934,
     "i_deg": 1.850, "Omega_deg": 49.56, "omega_deg": 286.50, "nu_deg": 250.0},

    {"name": "Jupiter", "mass": 1.89813e27, "a_AU": 5.2028, "e": 0.0489,
     "i_deg": 1.304, "Omega_deg": 100.47, "omega_deg": 273.87, "nu_deg": 60.0},

    {"name": "Saturn", "mass": 5.6834e26, "a_AU": 9.5388, "e": 0.0565,
     "i_deg": 2.485, "Omega_deg": 113.66, "omega_deg": 339.39, "nu_deg": 200.0},

    {"name": "Uranus", "mass": 8.6810e25, "a_AU": 19.1914, "e": 0.0472,
     "i_deg": 0.773, "Omega_deg": 74.01, "omega_deg": 96.99, "nu_deg": 300.0},

    {"name": "Neptune", "mass": 1.02413e26, "a_AU": 30.0611, "e": 0.0086,
     "i_deg": 1.770, "Omega_deg": 131.79, "omega_deg": 276.34, "nu_deg": 20.0}
]

number_of_bodies = len(bodies)
masses = np.array([body["mass"] for body in bodies], dtype=float)
total_mass = np.sum(masses)
sun_index = 0
M_sun = masses[sun_index]

print()
print("=" * 60)
print("SOLAR SYSTEM N-BODY SIMULATOR V1")
print("=" * 60)
print()
print("Number of bodies:", number_of_bodies)
print("Total system mass:", total_mass, "kg")


# ============================================================
# 3. ORBITAL ELEMENTS -> CARTESIAN STATE
# ============================================================

def orbital_elements_to_state(a, e, inclination, Omega, omega, nu, mu):
    """Return inertial-frame position [m] and velocity [m/s]."""

    if a <= 0.0:
        raise ValueError("Semi-major axis must be positive.")

    if not (0.0 <= e < 1.0):
        raise ValueError("V1 requires 0 <= eccentricity < 1.")

    p = a * (1.0 - e**2)

    r = p / (1.0 + e * np.cos(nu))

    r_pqw = np.array([
        r * np.cos(nu),
        r * np.sin(nu),
        0.0
    ])

    v_pqw = np.sqrt(mu / p) * np.array([
        -np.sin(nu),
        e + np.cos(nu),
        0.0
    ])

    cO, sO = np.cos(Omega), np.sin(Omega)
    ci, si = np.cos(inclination), np.sin(inclination)
    co, so = np.cos(omega), np.sin(omega)

    rotation_matrix = np.array([
        [
            cO * co - sO * so * ci,
            -cO * so - sO * co * ci,
            sO * si
        ],
        [
            sO * co + cO * so * ci,
            -sO * so + cO * co * ci,
            -cO * si
        ],
        [
            so * si,
            co * si,
            ci
        ]
    ])

    position = rotation_matrix @ r_pqw
    velocity = rotation_matrix @ v_pqw

    return position, velocity


# ============================================================
# 4. INITIAL STATES FROM ORBITAL ELEMENTS
# ============================================================

initial_positions_unshifted = np.zeros((number_of_bodies, 3), dtype=float)
initial_velocities_unshifted = np.zeros((number_of_bodies, 3), dtype=float)

for i, body in enumerate(bodies):

    if body["name"] == "Sun":
        continue

    a = body["a_AU"] * AU
    e = body["e"]

    inclination = np.deg2rad(body["i_deg"])
    Omega = np.deg2rad(body["Omega_deg"])
    omega = np.deg2rad(body["omega_deg"])
    nu = np.deg2rad(body["nu_deg"])

    mu = G * (M_sun + body["mass"])

    position, velocity = orbital_elements_to_state(
        a, e, inclination, Omega, omega, nu, mu
    )

    initial_positions_unshifted[i] = position
    initial_velocities_unshifted[i] = velocity


# ============================================================
# 5. BARYCENTRIC POSITION CORRECTION
# ============================================================

initial_barycenter = (
    np.sum(
        masses[:, None] * initial_positions_unshifted,
        axis=0
    ) / total_mass
)

print()
print("=" * 60)
print("INITIAL BARYCENTER BEFORE SHIFT")
print("=" * 60)
print()
print(initial_barycenter, "m")

initial_positions = (
    initial_positions_unshifted - initial_barycenter
)

barycenter_after_shift = (
    np.sum(
        masses[:, None] * initial_positions,
        axis=0
    ) / total_mass
)

print()
print("=" * 60)
print("BARYCENTER AFTER POSITION SHIFT")
print("=" * 60)
print()
print(barycenter_after_shift, "m")



# ============================================================
# 6. CONVERT HELIOCENTRIC VELOCITIES TO BARYCENTRIC VELOCITIES
# ============================================================

# Planet velocities currently represent velocities relative to
# the Sun. Find the Sun's barycentric velocity so total momentum
# of the complete system is zero.

planet_momentum = np.sum(
    masses[1:, None] * initial_velocities_unshifted[1:],
    axis=0
)

sun_barycentric_velocity = -planet_momentum / total_mass

# Transform every planet velocity into the barycentric frame.
initial_velocities = (
    initial_velocities_unshifted
    + sun_barycentric_velocity
)

# The Sun's heliocentric velocity is zero, so its barycentric
# velocity is the frame velocity calculated above.
initial_velocities[sun_index] = sun_barycentric_velocity

initial_total_momentum = np.sum(
    masses[:, None] * initial_velocities,
    axis=0
)

print()
print("=" * 60)
print("INITIAL TOTAL MOMENTUM")
print("=" * 60)
print()
print(initial_total_momentum, "kg m/s")



# ============================================================
# 7. STORE BODY STATES
# ============================================================

for i, body in enumerate(bodies):
    body["position"] = initial_positions[i].copy()
    body["velocity"] = initial_velocities[i].copy()


# ============================================================
# 8. INITIAL STATE VECTOR
# ============================================================

initial_positions = np.array(
    [body["position"] for body in bodies],
    dtype=float
)

initial_velocities = np.array(
    [body["velocity"] for body in bodies],
    dtype=float
)

initial_state = np.concatenate([
    initial_positions.flatten(),
    initial_velocities.flatten()
])

print()
print("Number of state variables:", len(initial_state))


# ============================================================
# 9. N-BODY EQUATIONS
# ============================================================

def n_body_equations(t, state, masses):

    n = len(masses)

    positions = state[:3 * n].reshape(n, 3)
    velocities = state[3 * n:].reshape(n, 3)

    accelerations = np.zeros_like(positions)

    for i in range(n):
        for j in range(n):

            if i == j:
                continue

            r_ij = positions[j] - positions[i]
            distance = np.linalg.norm(r_ij)

            if distance == 0.0:
                raise RuntimeError(
                    "Two bodies have zero separation."
                )

            accelerations[i] += (
                G * masses[j] * r_ij / distance**3
            )

    return np.concatenate([
        velocities.flatten(),
        accelerations.flatten()
    ])


# ============================================================
# 10. SIMULATION SETTINGS
# ============================================================

simulation_years = 20.0
t_start = 0.0
t_end = simulation_years * SECONDS_PER_YEAR

number_of_output_points = 4000

t_eval = np.linspace(
    t_start,
    t_end,
    number_of_output_points
)

print()
print("=" * 60)
print("SIMULATION SETTINGS")
print("=" * 60)
print()
print("Duration:", simulation_years, "years")
print("Output points:", number_of_output_points)


# ============================================================
# 11. NUMERICAL INTEGRATION
# ============================================================

print()
print("=" * 60)
print("STARTING N-BODY INTEGRATION")
print("=" * 60)
print()

solution = solve_ivp(
    lambda t, state: n_body_equations(
        t, state, masses
    ),
    (t_start, t_end),
    initial_state,
    t_eval=t_eval,
    rtol=1e-9,
    atol=1e-9
)

print()
print("=" * 60)
print("INTEGRATION RESULT")
print("=" * 60)
print()
print("Integration successful:", solution.success)
print("Solver message:", solution.message)
print("Number of output points:", len(solution.t))


# ============================================================
# 12. EXTRACT TRAJECTORIES
# ============================================================

positions = solution.y[:3 * number_of_bodies]
velocities = solution.y[3 * number_of_bodies:]

trajectories = {}
velocity_history = {}

for i, body in enumerate(bodies):

    p_start = 3 * i
    p_end = p_start + 3

    v_start = 3 * i
    v_end = v_start + 3

    trajectories[body["name"]] = (
        positions[p_start:p_end].T
    )

    velocity_history[body["name"]] = (
        velocities[v_start:v_end].T
    )

time_years = solution.t / SECONDS_PER_YEAR

trajectories_AU = {
    name: trajectory / AU
    for name, trajectory in trajectories.items()
}


# ============================================================
# 13. BARYCENTER HISTORY
# ============================================================

barycenter = np.zeros((len(solution.t), 3))

for body in bodies:
    barycenter += (
        body["mass"]
        * trajectories[body["name"]]
    )

barycenter /= total_mass

barycenter_AU = barycenter / AU


# ============================================================
# 14. MOMENTUM CONSERVATION
# ============================================================

total_momentum_history = np.zeros(
    (len(solution.t), 3)
)

for body in bodies:
    total_momentum_history += (
        body["mass"]
        * velocity_history[body["name"]]
    )

initial_momentum = total_momentum_history[0]

momentum_error = np.linalg.norm(
    total_momentum_history - initial_momentum,
    axis=1
)

earth_index = next(
    i for i, body in enumerate(bodies)
    if body["name"] == "Earth"
)

momentum_scale = (
    masses[earth_index]
    * np.linalg.norm(initial_velocities[earth_index])
)

relative_momentum_error = (
    momentum_error / momentum_scale
)


# ============================================================
# 15. ENERGY CONSERVATION
# ============================================================

kinetic_energy = np.zeros(len(solution.t))

for body in bodies:

    velocity = velocity_history[body["name"]]

    speed_squared = np.sum(
        velocity**2,
        axis=1
    )

    kinetic_energy += (
        0.5
        * body["mass"]
        * speed_squared
    )


potential_energy = np.zeros(len(solution.t))

for i in range(number_of_bodies):

    for j in range(i + 1, number_of_bodies):

        body_i = bodies[i]
        body_j = bodies[j]

        r_i = trajectories[body_i["name"]]
        r_j = trajectories[body_j["name"]]

        distance = np.linalg.norm(
            r_j - r_i,
            axis=1
        )

        potential_energy += (
            -G
            * body_i["mass"]
            * body_j["mass"]
            / distance
        )


total_energy = kinetic_energy + potential_energy
initial_energy = total_energy[0]

relative_energy_error = (
    np.abs(total_energy - initial_energy)
    / np.abs(initial_energy)
)


# ============================================================
# 16. ANGULAR MOMENTUM CONSERVATION
# ============================================================

total_L = np.zeros((len(solution.t), 3))

for body in bodies:

    position = trajectories[body["name"]]
    velocity = velocity_history[body["name"]]

    momentum = body["mass"] * velocity

    total_L += np.cross(
        position,
        momentum
    )

angular_momentum_error = np.linalg.norm(
    total_L - total_L[0],
    axis=1
)

angular_momentum_scale = np.linalg.norm(
    total_L[0]
)

relative_angular_momentum_error = (
    angular_momentum_error
    / angular_momentum_scale
)


# ============================================================
# 17. FINAL DIAGNOSTICS
# ============================================================

print()
print("=" * 60)
print("FINAL DIAGNOSTICS")
print("=" * 60)
print()

print(
    "Final barycenter:",
    barycenter[-1],
    "m"
)

print()

print(
    "Initial total momentum:",
    total_momentum_history[0],
    "kg m/s"
)

print(
    "Final total momentum:",
    total_momentum_history[-1],
    "kg m/s"
)

print(
    "Maximum relative momentum error:",
    np.max(relative_momentum_error)
)

print()

print(
    "Initial total energy:",
    total_energy[0],
    "J"
)

print(
    "Final total energy:",
    total_energy[-1],
    "J"
)

print(
    "Maximum relative energy error:",
    np.max(relative_energy_error)
)

print()

print(
    "Initial angular momentum:",
    total_L[0],
    "kg m^2/s"
)

print(
    "Final angular momentum:",
    total_L[-1],
    "kg m^2/s"
)

print(
    "Maximum relative angular momentum error:",
    np.max(relative_angular_momentum_error)
)


# ============================================================
# 18. FINAL POSITIONS
# ============================================================

print()
print("=" * 60)
print("FINAL POSITIONS")
print("=" * 60)
print()

for body in bodies:
    print(
        body["name"],
        ":",
        trajectories[body["name"]][-1],
        "m"
    )


# ============================================================
# 19. FULL SOLAR SYSTEM 2D PLOT
# ============================================================

plt.figure(figsize=(10, 10))

for name, trajectory in trajectories_AU.items():
    plt.plot(
        trajectory[:, 0],
        trajectory[:, 1],
        label=name
    )

plt.plot(
    barycenter_AU[:, 0],
    barycenter_AU[:, 1],
    "--",
    label="Barycenter"
)

plt.xlabel("X [AU]")
plt.ylabel("Y [AU]")

plt.title(
    "Full Solar System — 20 Year N-Body Simulation"
)

plt.axis("equal")
plt.grid()
plt.legend()

plt.show()


# ============================================================
# 20. INNER SOLAR SYSTEM 2D PLOT
# ============================================================

inner_bodies = [
    "Sun",
    "Mercury",
    "Venus",
    "Earth",
    "Mars"
]

plt.figure(figsize=(10, 10))

for name in inner_bodies:

    trajectory = trajectories_AU[name]

    plt.plot(
        trajectory[:, 0],
        trajectory[:, 1],
        label=name
    )

plt.xlabel("X [AU]")
plt.ylabel("Y [AU]")

plt.title(
    "Inner Solar System — 20 Year N-Body Simulation"
)

plt.xlim(-2.0, 2.0)
plt.ylim(-2.0, 2.0)

plt.axis("equal")
plt.grid()
plt.legend()

plt.show()


# ============================================================
# 21. CLEAN 3D FULL SOLAR SYSTEM ANIMATION — ALL 9 BODIES
# ============================================================

# Display range: Neptune's orbit is approximately 30 AU.
full_plot_limit = 32.0

fig = plt.figure(figsize=(13, 10), facecolor="#10141c")
ax = fig.add_subplot(111, projection="3d")
ax.set_facecolor("#10141c")

ax.set_xlim(-full_plot_limit, full_plot_limit)
ax.set_ylim(-full_plot_limit, full_plot_limit)
ax.set_zlim(-full_plot_limit, full_plot_limit)
ax.set_box_aspect((1, 1, 1))

ax.set_xlabel("X [AU]", color="white", labelpad=10)
ax.set_ylabel("Y [AU]", color="white", labelpad=10)
ax.set_zlabel("Z [AU]", color="white", labelpad=10)
ax.set_title("Solar System N-Body Simulation", color="white", pad=20, fontsize=16)
ax.tick_params(colors="#d5d9e0", labelsize=8)

# Keep the 3D panes subtle so the trajectories remain the focus.
for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
    axis.pane.set_facecolor((0.08, 0.10, 0.14, 1.0))
    axis.pane.set_edgecolor((0.25, 0.29, 0.36, 0.7))

planet_colors = {
    "Sun": "#FFD166",
    "Mercury": "#B8B8B8",
    "Venus": "#F4A261",
    "Earth": "#4EA8DE",
    "Mars": "#E76F51",
    "Jupiter": "#D6A77A",
    "Saturn": "#E9D8A6",
    "Uranus": "#72DDF7",
    "Neptune": "#4361EE",
}

# Larger markers improve visibility; the Sun is deliberately prominent.
body_sizes = {
    "Sun": 25,
    "Mercury": 7,
    "Venus": 9,
    "Earth": 10,
    "Mars": 8,
    "Jupiter": 14,
    "Saturn": 13,
    "Uranus": 11,
    "Neptune": 11,
}

all_bodies = [body["name"] for body in bodies]
body_markers = {}
orbit_trails = {}
inner_bodies = {"Mercury", "Venus", "Earth", "Mars"}

for name in all_bodies:
    body_markers[name], = ax.plot(
        [], [], [], marker="o", linestyle="None",
        markersize=body_sizes[name], color=planet_colors[name],
        markeredgecolor="white" if name == "Sun" else planet_colors[name],
        markeredgewidth=0.6, label=name, zorder=5,
    )

    # Thicker, clearer orbit trails for the inner planets.
    orbit_trails[name], = ax.plot(
        [], [], [], color=planet_colors[name],
        linewidth=1.35 if name in inner_bodies else 0.85,
        alpha=0.9 if name in inner_bodies else 0.58,
    )

# Small fixed Sun glow effect using a larger, transparent marker.
sun_glow, = ax.plot(
    [], [], [], marker="o", linestyle="None",
    markersize=34, color="#FFD166", alpha=0.16,
)

ax.legend(
    loc="upper left", bbox_to_anchor=(1.02, 1.0),
    frameon=True, facecolor="#171d28", edgecolor="#424b5a",
    labelcolor="white", fontsize=9,
)

# Make grid lines subtle.
ax.grid(True, alpha=0.18)

# ------------------------------------------------------------
# INTERACTIVE CAMERA CONTROLS
# ------------------------------------------------------------
# Scroll wheel: smooth zoom in/out around the current view.
# Left-click + drag: rotate the 3D view (Matplotlib default).
# Middle-click + drag: pan the view.
# Press R: reset the camera to the full Solar System view.
# These callbacks change only the camera, not the simulation.
initial_view_limits = {
    "x": (-full_plot_limit, full_plot_limit),
    "y": (-full_plot_limit, full_plot_limit),
    "z": (-full_plot_limit, full_plot_limit),
}
_pan_state = {"active": False, "last_x": None, "last_y": None}


def _set_limits_about_center(scale):
    """Scale the current 3D view smoothly around its center."""
    for axis_name, getter, setter in (
        ("x", ax.get_xlim3d, ax.set_xlim3d),
        ("y", ax.get_ylim3d, ax.set_ylim3d),
        ("z", ax.get_zlim3d, ax.set_zlim3d),
    ):
        low, high = getter()
        center = 0.5 * (low + high)
        half_range = 0.5 * (high - low) * scale
        # Avoid a zero-sized view after repeated zoom-ins.
        half_range = max(half_range, 0.015)
        setter(center - half_range, center + half_range)
    fig.canvas.draw_idle()


def _on_scroll(event):
    if event.inaxes is not ax:
        return
    # Smaller scale means zoom in; larger means zoom out.
    if event.button == "up":
        _set_limits_about_center(0.82)
    elif event.button == "down":
        _set_limits_about_center(1.0 / 0.82)


def _on_mouse_press(event):
    # Middle mouse button pans without interfering with left-drag rotation.
    if event.inaxes is ax and event.button == 2:
        _pan_state.update(active=True, last_x=event.x, last_y=event.y)


def _on_mouse_release(event):
    if event.button == 2:
        _pan_state.update(active=False, last_x=None, last_y=None)


def _on_mouse_move(event):
    if not _pan_state["active"] or event.inaxes is not ax:
        return
    last_x, last_y = _pan_state["last_x"], _pan_state["last_y"]
    if last_x is None or last_y is None:
        return

    # Convert pixel movement to a fraction of the current view width.
    xlim = ax.get_xlim3d()
    ylim = ax.get_ylim3d()
    zlim = ax.get_zlim3d()
    width = max(event.canvas.get_width_height()[0], 1)
    height = max(event.canvas.get_width_height()[1], 1)
    dx = -(event.x - last_x) / width * (xlim[1] - xlim[0])
    dy = -(event.y - last_y) / height * (ylim[1] - ylim[0])

    ax.set_xlim3d(xlim[0] + dx, xlim[1] + dx)
    ax.set_ylim3d(ylim[0] + dy, ylim[1] + dy)
    # Shift depth as well for a consistent 3D pan gesture.
    ax.set_zlim3d(zlim[0] + dy, zlim[1] + dy)
    _pan_state.update(last_x=event.x, last_y=event.y)
    fig.canvas.draw_idle()


def _on_key_press(event):
    if event.key and event.key.lower() == "r":
        ax.set_xlim3d(*initial_view_limits["x"])
        ax.set_ylim3d(*initial_view_limits["y"])
        ax.set_zlim3d(*initial_view_limits["z"])
        ax.view_init(elev=30, azim=-60)
        fig.canvas.draw_idle()


fig.canvas.mpl_connect("scroll_event", _on_scroll)
fig.canvas.mpl_connect("button_press_event", _on_mouse_press)
fig.canvas.mpl_connect("button_release_event", _on_mouse_release)
fig.canvas.mpl_connect("motion_notify_event", _on_mouse_move)
fig.canvas.mpl_connect("key_press_event", _on_key_press)
ax.view_init(elev=30, azim=-60)

def initialize():
    for name in all_bodies:
        body_markers[name].set_data([], [])
        body_markers[name].set_3d_properties([])
        orbit_trails[name].set_data([], [])
        orbit_trails[name].set_3d_properties([])

    sun_glow.set_data([], [])
    sun_glow.set_3d_properties([])
    return list(body_markers.values()) + list(orbit_trails.values()) + [sun_glow]


def update(frame):
    for name in all_bodies:
        trajectory = trajectories_AU[name]
        x, y, z = trajectory[frame]

        body_markers[name].set_data([x], [y])
        body_markers[name].set_3d_properties([z])

        # Accumulating trails show the path travelled so far.
        trail = trajectory[:frame + 1]
        orbit_trails[name].set_data(trail[:, 0], trail[:, 1])
        orbit_trails[name].set_3d_properties(trail[:, 2])

    sun_position = trajectories_AU["Sun"][frame]
    sun_glow.set_data([sun_position[0]], [sun_position[1]])
    sun_glow.set_3d_properties([sun_position[2]])

    ax.set_title(
        "Solar System N-Body Simulation\n"
        f"Simulation time: {time_years[frame]:.2f} years",
        color="white", pad=20, fontsize=16,
    )

    return list(body_markers.values()) + list(orbit_trails.values()) + [sun_glow]


animation = FuncAnimation(
    fig,
    update,
    frames=len(solution.t),
    init_func=initialize,
    interval=20,
    blit=False,
    repeat=False,
)

plt.tight_layout()
plt.show()


# ============================================================
# END OF V1
# ============================================================
