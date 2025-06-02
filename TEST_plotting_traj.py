import math
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

#--- Core physics definitions ---
class Body:
    """Represents a body with position, velocity, mass, and orbit history."""
    def __init__(self, x, y, radius, mass, vx=0.0, vy=0.0):
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
    return F * dx / dist, F * dy / dist

def recenter(bodies):
    """Recenters simulation to center-of-mass frame."""
    M = sum(b.mass for b in bodies)
    x_cm = sum(b.x * b.mass for b in bodies) / M
    y_cm = sum(b.y * b.mass for b in bodies) / M
    vx_cm = sum(b.vx * b.mass for b in bodies) / M
    vy_cm = sum(b.vy * b.mass for b in bodies) / M
    for b in bodies:
        b.x -= x_cm; b.y -= y_cm
        b.vx -= vx_cm; b.vy -= vy_cm

#--- Damped Velocity-Verlet integrator ---
def simulate_verlet_damped(bodies, dt, steps, G=1.0, damping=0.01,
                            max_records=1000, recenter_every=1):
    """
    Integrate bodies under gravity with linear damping on velocities.
    damping: coefficient for drag force F_drag = -damping * v.
    """
    for b in bodies:
        b.reset_orbit()
    skip = max(1, steps // max_records)

    # initial accelerations
    acc = {b: [0.0, 0.0] for b in bodies}
    for i, b1 in enumerate(bodies):
        for b2 in bodies[i+1:]:
            fx, fy = gravitational_force(b1, b2, G)
            acc[b1][0] += fx / b1.mass
            acc[b1][1] += fy / b1.mass
            acc[b2][0] -= fx / b2.mass
            acc[b2][1] -= fy / b2.mass

    # integration loop
    for i in range(1, steps + 1):
        # half-step velocity
        for b in bodies:
            b.vx += 0.5 * acc[b][0] * dt
            b.vy += 0.5 * acc[b][1] * dt
        # position update
        for b in bodies:
            b.x += b.vx * dt
            b.y += b.vy * dt
        # compute new accelerations including damping
        new_acc = {b: [0.0, 0.0] for b in bodies}
        for j, b1 in enumerate(bodies):
            for b2 in bodies[j+1:]:
                fx, fy = gravitational_force(b1, b2, G)
                new_acc[b1][0] += fx / b1.mass
                new_acc[b1][1] += fy / b1.mass
                new_acc[b2][0] -= fx / b2.mass
                new_acc[b2][1] -= fy / b2.mass
        # apply damping
        for b in bodies:
            new_acc[b][0] += -damping * b.vx / b.mass
            new_acc[b][1] += -damping * b.vy / b.mass
        # half-step velocity
        for b in bodies:
            b.vx += 0.5 * new_acc[b][0] * dt
            b.vy += 0.5 * new_acc[b][1] * dt
        # recenter
        if i % recenter_every == 0:
            recenter(bodies)
        acc = new_acc
        # record orbit
        if i % skip == 0:
            for b in bodies:
                b.orbit.append((b.x, b.y))

#--- Orbit factory with damping-induced spiral ---
def damped_spiral_converge(R=1.0, damping=0.02):
    """
    Three equal masses at equilateral triangle vertices, given exact circular
    velocity. Damping then causes gradual inward spiral and eventual convergence.
    """
    angles = [0, 2*math.pi/3, 4*math.pi/3]
    bodies = []
    # circular speed for each around COM: v_circ = sqrt(G*M_total/(3*R))
    M_total = 3.0
    v_circ = math.sqrt(1.0 * M_total / (3 * R))
    r = 0.05
    for theta in angles:
        x = R * math.cos(theta)
        y = R * math.sin(theta)
        # unit tangential
        tx, ty = -math.sin(theta), math.cos(theta)
        vx = v_circ * tx
        vy = v_circ * ty
        bodies.append(Body(x, y, r, 1.0, vx=vx, vy=vy))
    return bodies, damping

#--- Plotting utility ---
COMMON_COLORS = ['green', 'blue', 'red']

def plot_final_outcome(bodies, title, dt, steps,
                       damping, recenter_every, max_records, filename):
    simulate_verlet_damped(bodies, dt, steps,
                            damping=damping,
                            max_records=max_records,
                            recenter_every=recenter_every)
    # collect orbits
    xs_all, ys_all = [], []
    for b in bodies:
        xs, ys = zip(*b.orbit)
        xs_all.extend(xs); ys_all.extend(ys)
    xmin, xmax = min(xs_all), max(xs_all)
    ymin, ymax = min(ys_all), max(ys_all)
    mx = (xmax - xmin) * 0.1 or 1.0
    my = (ymax - ymin) * 0.1 or 1.0

    fig, ax = plt.subplots(figsize=(6,6))
    ax.set_title(title)
    ax.set_xlim(xmin - mx, xmax + mx)
    ax.set_ylim(ymin - my, ymax + my)
    ax.set_xlabel('x'); ax.set_ylabel('y')
    for color, b in zip(COMMON_COLORS, bodies):
        xs, ys = zip(*b.orbit)
        ax.plot(xs, ys, color=color)
        ax.scatter(xs[-1], ys[-1], color=color, s=80, edgecolor='k')
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.show()
    print(f"Saved '{filename}'")

#--- Main execution ---
if __name__ == '__main__':
    # Damped spiral convergence demo
    R = 1.0
    bodies, damping = damped_spiral_converge(R, damping=0.02)
    # estimate circular period ~ 2*pi*sqrt(3R/3)
    T = 2 * math.pi * math.sqrt(R)
    total_time = 8 * T
    dt = total_time / 10000
    steps = 10000
    plot_final_outcome(
        bodies,
        title='Damped Spiral Convergence',
        dt=dt,
        steps=steps,
        damping=damping,
        recenter_every=5,
        max_records=1000,
        filename='damped_spiral_converge.png'
    )
