import math
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

#--- Core physics definitions ---
class Body:
    """Represents a body with position, velocity, mass, and orbit history."""
    def __init__(self, x, y, radius, mass, vx=0, vy=0):
        self.x = x
        self.y = y
        self.radius = radius
        self.mass = mass
        self.vx = vx
        self.vy = vy
        self.orbit = [(x, y)]

    def reset_orbit(self):
        self.orbit = [(self.x, self.y)]

#--- Physics utilities ---
def gravitational_force(b1, b2, G=1.0):
    dx = b2.x - b1.x
    dy = b2.y - b1.y
    dist2 = dx*dx + dy*dy + 1e-8
    dist = math.sqrt(dist2)
    F = G * b1.mass * b2.mass / dist2
    return (F * dx / dist, F * dy / dist)

def recenter(bodies):
    """Remove any net drift by recentering positions & velocities to the COM."""
    M = sum(b.mass for b in bodies)
    x_cm = sum(b.x * b.mass for b in bodies) / M
    y_cm = sum(b.y * b.mass for b in bodies) / M
    vx_cm = sum(b.vx * b.mass for b in bodies) / M
    vy_cm = sum(b.vy * b.mass for b in bodies) / M
    for b in bodies:
        b.x -= x_cm
        b.y -= y_cm
        b.vx -= vx_cm
        b.vy -= vy_cm

#--- Velocity-Verlet with subsampling & optional recenter ---
def simulate_verlet(bodies, dt, steps,
                     G=1.0,
                     max_records=500,
                     recenter_every=1):
    """
    Integrate with velocity-Verlet, recenter every `recenter_every` steps,
    and only keep ~max_records points per body by subsampling.
    """
    for b in bodies:
        b.reset_orbit()
    record_skip = max(1, steps // max_records)

    # initial accelerations
    acc = {b: [0.0, 0.0] for b in bodies}
    for i, b1 in enumerate(bodies):
        for b2 in bodies[i+1:]:
            fx, fy = gravitational_force(b1, b2, G)
            acc[b1][0] += fx / b1.mass
            acc[b1][1] += fy / b1.mass
            acc[b2][0] -= fx / b2.mass
            acc[b2][1] -= fy / b2.mass

    for i in range(1, steps+1):
        # Velocity half-step
        for b in bodies:
            b.vx += 0.5 * acc[b][0] * dt
            b.vy += 0.5 * acc[b][1] * dt
        # Position full-step
        for b in bodies:
            b.x += b.vx * dt
            b.y += b.vy * dt
        # Compute new accelerations
        acc_new = {b: [0.0, 0.0] for b in bodies}
        for j, b1 in enumerate(bodies):
            for b2 in bodies[j+1:]:
                fx, fy = gravitational_force(b1, b2, G)
                acc_new[b1][0] += fx / b1.mass
                acc_new[b1][1] += fy / b1.mass
                acc_new[b2][0] -= fx / b2.mass
                acc_new[b2][1] -= fy / b2.mass
        # Velocity half-step
        for b in bodies:
            b.vx += 0.5 * acc_new[b][0] * dt
            b.vy += 0.5 * acc_new[b][1] * dt
        # Recentering
        if i % recenter_every == 0:
            recenter(bodies)
        acc = acc_new
        # Record orbit
        if i % record_skip == 0:
            for b in bodies:
                b.orbit.append((b.x, b.y))

#--- Orbit factories ---
def stable_nonhierarchical():
    """
    First stable non-hierarchical 3-body orbit:
    m1=0.800, m2=0.756, m3=1.000,
    x1=-0.135024519775613, v1=2.51505829297841, v2=0.316396261593079, T=5.16008719949432
    """
    m1, m2, m3 = 0.800, 0.756, 1.000
    x1 = -0.135024519775613
    v1 = 2.51505829297841
    v2 = 0.316396261593079
    r = 0.05
    return [
        Body(x1, 0.0, r, m1, vx=0.0, vy=v1),
        Body(1.0, 0.0, r, m2, vx=0.0, vy=v2),
        Body(0.0, 0.0, r, m3, vx=0.0, vy=-(m1*v1 + m2*v2)/m3)
    ]

def divergent_case():
    """Example divergent three-body: high velocities cause escape paths."""
    m, r = 1.0, 0.05
    return [
        Body(-1.0, 0.0, r, m, vx=0.0, vy=1.0),
        Body(1.0, 0.0, r, m, vx=0.0, vy=-1.0),
        Body(0.0, 1.0, r, m, vx=1.0, vy=0.0)
    ]

def convergent_case():
    """Example convergent three-body: zero velocities lead to collapse."""
    m, r = 1.0, 0.05
    return [
        Body(-1.0, 0.0, r, m),
        Body(1.0, 0.0, r, m),
        Body(0.0, 1.5, r, m)
    ]

#--- Utilities for plotting colors ---
COMMON_COLORS = ['green', 'blue', 'red']

#--- Animation routine with dynamic bounds and fixed colors ---
def animate_solution(bodies, title,
                     dt=0.001,
                     steps=1000,
                     max_frames=500,
                     recenter_every=1):
    simulate_verlet(
        bodies, dt, steps,
        max_records=max_frames,
        recenter_every=recenter_every
    )
    colors = COMMON_COLORS[:len(bodies)]
    # determine dynamic plot limits
    xs_all, ys_all = [], []
    for b in bodies:
        xs, ys = zip(*b.orbit)
        xs_all.extend(xs)
        ys_all.extend(ys)
    x_min, x_max = min(xs_all), max(xs_all)
    y_min, y_max = min(ys_all), max(ys_all)
    margin_x = (x_max - x_min) * 0.1 or 1.0
    margin_y = (y_max - y_min) * 0.1 or 1.0

    fig, ax = plt.subplots(figsize=(6,6))
    fig.suptitle(title, fontsize=16)
    ax.set_xlim(x_min - margin_x, x_max + margin_x)
    ax.set_ylim(y_min - margin_y, y_max + margin_y)
    ax.axhline(0, color='gray', linewidth=0.5)
    ax.axvline(0, color='gray', linewidth=0.5)

    def update(frame):
        ax.clear()
        ax.set_xlim(x_min - margin_x, x_max + margin_x)
        ax.set_ylim(y_min - margin_y, y_max + margin_y)
        ax.axhline(0, color='gray', linewidth=0.5)
        ax.axvline(0, color='gray', linewidth=0.5)
        for color, b in zip(colors, bodies):
            xs, ys = zip(*b.orbit[:frame+1])
            ax.plot(xs, ys, color=color, linewidth=1)
            ax.scatter(xs[-1], ys[-1], color=color, s=50)

    ani = FuncAnimation(fig, update, frames=len(bodies[0].orbit), interval=20)
    plt.show()

#--- Static final-outcome plot with dynamic bounds and fixed colors ---
def plot_final_outcome(bodies, title,
                       dt,
                       steps,
                       recenter_every,
                       max_records,
                       filename='final_outcome.png'):
    simulate_verlet(
        bodies,
        dt,
        steps,
        max_records=max_records,
        recenter_every=recenter_every
    )
    colors = COMMON_COLORS[:len(bodies)]
    # gather all points
    xs_all, ys_all = [], []
    for b in bodies:
        xs, ys = zip(*b.orbit)
        xs_all.extend(xs)
        ys_all.extend(ys)
    x_min, x_max = min(xs_all), max(xs_all)
    y_min, y_max = min(ys_all), max(ys_all)
    margin_x = (x_max - x_min) * 0.1 or 1.0
    margin_y = (y_max - y_min) * 0.1 or 1.0

    fig, ax = plt.subplots(figsize=(6,6))
    fig.suptitle(title, fontsize=16)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_xlim(x_min - margin_x, x_max + margin_x)
    ax.set_ylim(y_min - margin_y, y_max + margin_y)
    # no grid lines

    for color, b in zip(colors, bodies):
        xs, ys = zip(*b.orbit)
        ax.plot(xs, ys, color=color, linewidth=1)
        ax.scatter(xs[-1], ys[-1], color=color, s=100, edgecolor='k')

    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.show()
    print(f"Saved final outcome to {filename!r}")

#--- Main entry ---
if __name__ == '__main__':
    # Stable non-hierarchical demo
    bodies = stable_nonhierarchical()
    T = 5.16008719949432
    dt = 0.00001          # ~5 160 steps per period
    steps = int(T / dt)
    title = f'Stable Non-hierarchical Orbit (T={T:.6f})'
    animate_solution(
        bodies,
        title=title,
        dt=dt,
        steps=steps,
        max_frames=800,
        recenter_every=5
    )
    plot_final_outcome(
        bodies,
        title=title + ' – Final Outcome',
        dt=dt,
        steps=steps,
        recenter_every=5,
        max_records=800,
        filename='stable_nonhierarchical_final.png'
    )
    # Divergent case
    bodies = divergent_case()
    plot_final_outcome(
        bodies,
        title='Divergent Case – Final Outcome',
        dt=dt,
        steps=steps,
        recenter_every=5,
        max_records=800,
        filename='divergent_final.png'
    )
    # Convergent case
    bodies = convergent_case()
    plot_final_outcome(
        bodies,
        title='Convergent Case – Final Outcome',
        dt=dt,
        steps=steps,
        recenter_every=5,
        max_records=800,
        filename='convergent_final.png'
    )
