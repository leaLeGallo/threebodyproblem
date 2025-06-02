# animations_solutions.py
import math
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

#--- Core physics definitions ---
class Body:
    """Represents a body with position, velocity, mass, and orbit history."""
    def __init__(self, x, y, radius, color, mass, vx=0, vy=0):
        self.x = x
        self.y = y
        self.radius = radius
        self.color = color
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
    x_cm  = sum(b.x  * b.mass for b in bodies) / M
    y_cm  = sum(b.y  * b.mass for b in bodies) / M
    vx_cm = sum(b.vx * b.mass for b in bodies) / M
    vy_cm = sum(b.vy * b.mass for b in bodies) / M
    for b in bodies:
        b.x  -= x_cm
        b.y  -= y_cm
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
    # reset history
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
        # half-step vel
        for b in bodies:
            b.vx += 0.5 * acc[b][0] * dt
            b.vy += 0.5 * acc[b][1] * dt
        # full-step pos
        for b in bodies:
            b.x += b.vx * dt
            b.y += b.vy * dt
        # new accel
        acc_new = {b: [0.0, 0.0] for b in bodies}
        for j, b1 in enumerate(bodies):
            for b2 in bodies[j+1:]:
                fx, fy = gravitational_force(b1, b2, G)
                acc_new[b1][0] += fx / b1.mass
                acc_new[b1][1] += fy / b1.mass
                acc_new[b2][0] -= fx / b2.mass
                acc_new[b2][1] -= fy / b2.mass
        # half-step vel
        for b in bodies:
            b.vx += 0.5 * acc_new[b][0] * dt
            b.vy += 0.5 * acc_new[b][1] * dt
        if i % recenter_every == 0:
            recenter(bodies)
        acc = acc_new

        # record
        if i % record_skip == 0:
            for b in bodies:
                b.orbit.append((b.x, b.y))

#--- Butterfly I factory ---
def butterfly_I():
    """Šuvakov–Dmitrašinović I.2.A ‘Butterfly I’ (equal masses)."""
    m, r = 1.0, 0.05
    v1x, v1y =  0.306893,  0.125507
    v3x, v3y = -2*v1x,   -2*v1y
    b1 = Body(-1.0, 0.0, r, 'red',   m, vx=v1x, vy=v1y)
    b2 = Body( 1.0, 0.0, r, 'green', m, vx=v1x, vy=v1y)
    b3 = Body( 0.0, 0.0, r, 'blue',  m, vx=v3x, vy=v3y)
    return [b1, b2, b3]

#--- Moth (VIIb.4.A) factory ---
def moth_VIIb():
    """Šuvakov–Dmitrašinović VIIb.4.A ‘Moth’ (equal masses)."""
    m, r = 1.0, 0.05
    v1x, v1y =  0.537956,  0.341458
    v3x, v3y = -2*v1x,   -2*v1y
    b1 = Body(-1.0, 0.0, r, 'magenta', m, vx=v1x, vy=v1y)
    b2 = Body( 1.0, 0.0, r, 'cyan',    m, vx=v1x, vy=v1y)
    b3 = Body( 0.0, 0.0, r, 'orange',  m, vx=v3x, vy=v3y)
    return [b1, b2, b3]

#--- Animation routine ---
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
    colors = [b.color for b in bodies]
    N = len(bodies[0].orbit)

    fig, ax = plt.subplots(figsize=(6,6))
    fig.suptitle(title, fontsize=16)
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    ax.axhline(0, color='black', linewidth=0.5)
    ax.axvline(0, color='black', linewidth=0.5)

    def update(frame):
        ax.clear()
        ax.axhline(0, color='black', linewidth=0.5)
        ax.axvline(0, color='black', linewidth=0.5)
        for color, b in zip(colors, bodies):
            xs, ys = zip(*b.orbit[:frame+1])
            ax.plot(xs, ys, color=color, linewidth=1)
            ax.scatter(xs[-1], ys[-1], color=color, s=50)
        ax.set_xlim(-2, 2)
        ax.set_ylim(-2, 2)

    ani = FuncAnimation(fig, update, frames=N, interval=20)
    plt.show()


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
    r = 0.05  # body radius for plotting

    b1 = Body(x1, 0.0, r, 'red',   m1, vx=0.0,      vy=v1)
    b2 = Body(1.0,  0.0, r, 'green', m2, vx=0.0,      vy=v2)
    # enforce zero total momentum:
    b3 = Body(0.0,  0.0, r, 'blue',  m3,
              vx=0.0,
              vy=-(m1*v1 + m2*v2)/m3)
    return [b1, b2, b3]


#--- Main entry ---
if __name__ == '__main__':
    bodies = stable_nonhierarchical()
    T      = 5.16008719949432
    dt     = 0.00001          # ~5 160 steps per period
    steps  = int(T / dt)
    title  = f'Stable Non-hierarchical Orbit (T={T:.6f})'

    animate_solution(
        bodies,
        title=title,
        dt=dt,
        steps=steps,
        max_frames=800,      # keep ~800 points per body
        recenter_every=5     # recenter every 5 steps
    )

