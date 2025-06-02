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

#--- Velocity-Verlet integration ---
def simulate_verlet(bodies, dt, steps,
                     G=1.0,
                     max_records=500,
                     recenter_every=1):
    """
    Integrate with velocity-Verlet, recenter every `recenter_every` steps,
    and keep ~max_records points per body by subsampling.
    """
    for b in bodies:
        b.reset_orbit()
    skip = max(1, steps // max_records)

    # initial accelerations
    acc = {b: [0.0, 0.0] for b in bodies}
    for i, b1 in enumerate(bodies):
        for b2 in bodies[i+1:]:
            fx, fy = gravitational_force(b1, b2, G)
            acc[b1][0] += fx/b1.mass
            acc[b1][1] += fy/b1.mass
            acc[b2][0] -= fx/b2.mass
            acc[b2][1] -= fy/b2.mass

    for step in range(1, steps+1):
        # half-step velocity
        for b in bodies:
            b.vx += 0.5*acc[b][0]*dt
            b.vy += 0.5*acc[b][1]*dt
        # full-step position
        for b in bodies:
            b.x += b.vx*dt
            b.y += b.vy*dt
        # compute new acc
        new_acc = {b: [0.0, 0.0] for b in bodies}
        for i, b1 in enumerate(bodies):
            for b2 in bodies[i+1:]:
                fx, fy = gravitational_force(b1, b2, G)
                new_acc[b1][0] += fx/b1.mass
                new_acc[b1][1] += fy/b1.mass
                new_acc[b2][0] -= fx/b2.mass
                new_acc[b2][1] -= fy/b2.mass
        # half-step velocity
        for b in bodies:
            b.vx += 0.5*new_acc[b][0]*dt
            b.vy += 0.5*new_acc[b][1]*dt
        # recentre
        if step % recenter_every == 0:
            recenter(bodies)
        acc = new_acc
        # record
        if step % skip == 0:
            for b in bodies:
                b.orbit.append((b.x, b.y))

#--- Orbit factories ---
def stable_nonhierarchical():
    """Stable 3-body periodic orbit demo."""
    m1, m2, m3 = 0.8, 0.756, 1.0
    x1 = -0.135024519775613
    v1 = 2.51505829297841
    v2 = 0.316396261593079
    r = 0.05
    return [
        Body(x1, 0.0, r, m1, vy=v1),
        Body(1.0, 0.0, r, m2, vy=v2),
        Body(0.0, 0.0, r, m3, vy=-(m1*v1+m2*v2)/m3)
    ]

def divergent_case():
    """Divergent 3-body: three bodies escape."""
    m, r = 1.0, 0.05
    return [
        Body(-1.0, 0.0, r, m, vy=1.0),
        Body(1.0, 0.0, r, m, vy=-1.0),
        Body(0.0, 1.0, r, m, vx=1.0)
    ]

def collision_three():
    """
    Three bodies, but only two initially move toward each other; third is static.
    Bodies A & B head-on collide; C sits off-axis.
    """
    m, r = 1.0, 0.05
    # A & B on x-axis, moving inward
    A = Body(-1.0, 0.0, r, m, vx=0.5)
    B = Body(1.0, 0.0, r, m, vx=-0.5)
    # C static above
    C = Body(0.0, 1.5, r, m, vx=0.0, vy=0.0)
    return [A, B, C]

#--- Plotting ---
COMMON_COLORS = ['green', 'blue', 'red']

def plot_final_outcome(bodies, title,
                       dt, steps,
                       recenter_every, max_records,
                       filename='final.png'):
    simulate_verlet(bodies, dt, steps,
                     max_records=max_records,
                     recenter_every=recenter_every)
    # gather orbits
    xs_all, ys_all = [], []
    for b in bodies:
        xs, ys = zip(*b.orbit)
        xs_all.extend(xs); ys_all.extend(ys)
    xmin, xmax = min(xs_all), max(xs_all)
    ymin, ymax = min(ys_all), max(ys_all)
    mx = (xmax-xmin)*0.1 or 1.0
    my = (ymax-ymin)*0.1 or 1.0

    fig, ax = plt.subplots(figsize=(6,6))
    fig.suptitle(title)
    ax.set_xlim(xmin-mx, xmax+mx)
    ax.set_ylim(ymin-my, ymax+my)
    ax.set_xlabel('x'); ax.set_ylabel('y')
    for color, b in zip(COMMON_COLORS, bodies):
        xs, ys = zip(*b.orbit)
        ax.plot(xs, ys, color=color)
        ax.scatter(xs[-1], ys[-1], color=color, s=80, edgecolor='k')
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.show()
    print(f"Saved '{filename}'")

#--- Main ---
if __name__ == '__main__':
    # Collision demo: two bump, third static
    bodies = collision_three()
    plot_final_outcome(bodies,
                       title='Two-Body Head-On Collision with Third Body',
                       dt=0.005,
                       steps=2000,
                       recenter_every=1,
                       max_records=500,
                       filename='three_body_collision.png')
