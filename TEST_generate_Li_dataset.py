import math
import random
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np

# --- Constants -------------------------------------------------------
G = 1.0
EPS = 1e-8       # softening constant for both force and potential
ENERGY_TOL = 1e-5  # tolerance for positive-energy divergence test

# --- Physics / integration (Velocity‑Verlet) ------------------------------
class Body:
    def __init__(self, x, y, mass, vx, vy, radius=0.1, color=None):
        self.x, self.y = x, y
        self.mass = mass
        self.vx, self.vy = vx, vy
        self.radius = radius
        self.color = color or 'black'
        # histories
        self.orbit = [(x, y)]
        self.vel_hist = [(vx, vy)]

    def reset(self):
        self.orbit = [(self.x, self.y)]
        self.vel_hist = [(self.vx, self.vy)]


def gravitational_force(b1, b2):
    dx = b2.x - b1.x
    dy = b2.y - b1.y
    dist2 = dx*dx + dy*dy + EPS
    dist = math.sqrt(dist2)
    F = G * b1.mass * b2.mass / dist2
    return F * dx / dist, F * dy / dist


def recenter(bodies):
    """Shift coordinates so that the centre of mass is at the origin"""
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


def simulate_verlet(bodies, dt, steps, recenter_every=10, max_frames=None):
    """Velocity‑Verlet integrator with optional periodic recentring."""
    record_skip = 1 if max_frames is None else max(1, steps // max_frames)

    # reset histories
    for b in bodies:
        b.reset()

    # initial accelerations
    acc = {b: [0.0, 0.0] for b in bodies}
    for i, b1 in enumerate(bodies):
        for b2 in bodies[i+1:]:
            fx, fy = gravitational_force(b1, b2)
            acc[b1][0] += fx / b1.mass
            acc[b1][1] += fy / b1.mass
            acc[b2][0] -= fx / b2.mass
            acc[b2][1] -= fy / b2.mass

    for i in range(steps):
        # half‑kick
        for b in bodies:
            b.vx += 0.5 * acc[b][0] * dt
            b.vy += 0.5 * acc[b][1] * dt
        # drift
        for b in bodies:
            b.x += b.vx * dt
            b.y += b.vy * dt
        # new accelerations
        new_acc = {b: [0.0, 0.0] for b in bodies}
        for j, b1 in enumerate(bodies):
            for b2 in bodies[j+1:]:
                fx, fy = gravitational_force(b1, b2)
                new_acc[b1][0] += fx / b1.mass
                new_acc[b1][1] += fy / b1.mass
                new_acc[b2][0] -= fx / b2.mass
                new_acc[b2][1] -= fy / b2.mass
        # half‑kick
        for b in bodies:
            b.vx += 0.5 * new_acc[b][0] * dt
            b.vy += 0.5 * new_acc[b][1] * dt

        # periodic recentre (optional)
        if recenter_every and (i + 1) % recenter_every == 0:
            recenter(bodies)

        # save histories
        if i % record_skip == 0:
            for b in bodies:
                b.orbit.append((b.x, b.y))
                b.vel_hist.append((b.vx, b.vy))

        acc = new_acc


# --- Outcome classification ------------------------------------------------

def total_energy(bodies, t):
    """Return the total mechanical energy of the system at index *t*."""
    # kinetic
    E = 0.0
    for b in bodies:
        vx, vy = b.vel_hist[t]
        E += 0.5 * b.mass * (vx*vx + vy*vy)
    # potential
    for i, bi in enumerate(bodies):
        xi, yi = bi.orbit[t]
        for bj in bodies[i+1:]:
            xj, yj = bj.orbit[t]
            dx = xi - xj
            dy = yi - yj
            r = math.sqrt(dx*dx + dy*dy + EPS)
            E -= G * bi.mass * bj.mass / r
    return E


def segments_intersect(p, q, r, s, radius_sum):
    """Quick‑and‑dirty test whether two line segments of radius *radius_sum* intersect."""
    px, py = p; qx, qy = q; rx, ry = r; sx, sy = s
    vx, vy = qx - px, qy - py
    wx, wy = sx - rx, sy - ry
    dx, dy = px - rx, py - ry
    a = (vx - wx)**2 + (vy - wy)**2
    b = 2 * ((vx - wx)*dx + (vy - wy)*dy)
    c = dx*dx + dy*dy - radius_sum*radius_sum
    disc = b*b - 4*a*c
    if disc < 0:
        return False
    t = max(0.0, min(1.0, (-b - math.sqrt(disc)) / (2*a))) if a else 0.0
    cx = dx + (vx - wx)*t
    cy = dy + (vy - wy)*t
    return (cx*cx + cy*cy) <= radius_sum*radius_sum


def classify_outcome(bodies, escape_R=8.0):
    """Return 'convergence', 'divergence' or 'stable' for one simulated run."""
    n = len(bodies[0].orbit)

    # 1) collision check ----------------------------------------------------
    for t in range(n - 1):
        for i, b1 in enumerate(bodies):
            for b2 in bodies[i+1:]:
                r_sum = b1.radius + b2.radius
                if segments_intersect(
                        b1.orbit[t], b1.orbit[t+1],
                        b2.orbit[t], b2.orbit[t+1],
                        r_sum):
                    return "convergence"

    # 2) escape / hyperbolic orbit check ------------------------------------
    for t in range(n):
        # hyperbolic if total energy > tolerance
        if total_energy(bodies, t) > ENERGY_TOL:
            return "divergence"
        # loose distance‑plus‑radial‑velocity gate
        for b in bodies:
            x, y = b.orbit[t]
            vx, vy = b.vel_hist[t]
            if (x*x + y*y > escape_R*escape_R) and (x*vx + y*vy > 0):
                return "divergence"

    return "stable"


# --- Dataset generation ----------------------------------------------------

def make_random_dataset(n_per_class=100, dt=1e-5, max_frames=500, out_txt="dataset.txt"):
    """Generate a balanced dataset of convergent/divergent initial conditions."""
    wanted = {"convergence": n_per_class, "divergence": n_per_class}
    have   = {k: 0 for k in wanted}
    rows   = []

    # parameter ranges — narrow window around the figure‑eight ----------------
    m1_min, m1_max = 0.800, 0.820
    m2_min, m2_max = 0.751, 0.760
    m3_val = 1.000
    x1_min, x1_max = -0.1385, -0.1335
    v1_min, v1_max = 2.4800, 2.5200
    v2_min, v2_max = 0.3160, 0.3320
    T_min, T_max  = 5.16008719949432, 5.23790395158023

    while any(have[k] < wanted[k] for k in wanted):
        m1 = random.uniform(m1_min, m1_max)
        m2 = random.uniform(m2_min, m2_max)
        m3 = m3_val
        x1 = random.uniform(x1_min, x1_max)
        v1 = random.uniform(v1_min, v1_max)
        v2 = random.uniform(v2_min, v2_max)
        T  = random.uniform(T_min, T_max)
        steps_T = max(1, int(T / dt))

        # set up bodies — COM at origin after one initial recenter ----------
        b1 = Body(x1, 0.0, m1, 0.0, v1, radius=0.1, color='red')
        b2 = Body(1.0, 0.0, m2, 0.0, v2, radius=0.1, color='green')
        vy3 = -(m1*v1 + m2*v2) / m3
        b3 = Body(0.0, 0.0, m3, 0.0, vy3, radius=0.1, color='blue')
        bodies = [b1, b2, b3]
        recenter(bodies)

        # integrate -----------------------------------------------------
        simulate_verlet(
            bodies, dt, steps_T,
            recenter_every=0,
            max_frames=max_frames,
        )

        label = classify_outcome(bodies)
        if label not in wanted:
            continue
        if have[label] < wanted[label]:
            rows.append({
                "m1": f"{m1:.3f}", "m2": f"{m2:.3f}", "m3": f"{m3:.3f}",
                "x1": f"{x1:.14e}", "v1": f"{v1:.14e}",
                "v2": f"{v2:.14e}", "T": f"{T:.14e}",
                "stability": "C" if label == "convergence" else "D",
            })
            have[label] += 1
            print(f"{label}: {have[label]}/{wanted[label]}")

    # write file --------------------------------------------------------
    with open(out_txt, "w") as f:
        f.write("m1 m2 m3 x1 v1 v2 T stability\n")
        for r in rows:
            f.write(
                f"{r['m1']} {r['m2']} {r['m3']} "
                f"{r['x1']} {r['v1']} {r['v2']} {r['T']} {r['stability']}\n"
            )
    print(f"Done — wrote {len(rows)} rows to {out_txt}")
    return rows


# --- Simple visual sanity check -------------------------------------------

def animate_random_cases(dt, out_txt="dataset.txt", max_frames=500, interval=20):
    """Pick one convergent and one divergent case from *out_txt* and animate."""
    df = pd.read_csv(out_txt, sep=r"\s+")
    picks = {
        "convergent": df[df.stability=="C"].sample(1).iloc[0],
        "divergent": df[df.stability=="D"].sample(1).iloc[0],
    }
    empty = np.empty((0,2))

    for outcome, row in picks.items():
        m1, m2, m3 = map(float, (row.m1, row.m2, row.m3))
        x1, v1, v2     = map(float, (row.x1, row.v1, row.v2))
        T = float(row.T)
        steps_T = max(1, int(T / dt))

        b1 = Body(x1, 0.0, m1, 0.0, v1, radius=0.1, color='red')
        b2 = Body(1.0, 0.0, m2, 0.0, v2, radius=0.1, color='green')
        vy3 = -(m1*v1 + m2*v2) / m3
        b3 = Body(0.0, 0.0, m3, 0.0, vy3, radius=0.1, color='blue')
        bodies = [b1, b2, b3]
        recenter(bodies)

        simulate_verlet(
            bodies, dt, steps_T,
            recenter_every=max(1, steps_T // max_frames),
            max_frames=max_frames,
        )

        fig, ax = plt.subplots(figsize=(5,5))
        ax.set_title(f"{outcome.capitalize()} case")
        ax.set_xlim(-5,5); ax.set_ylim(-5,5)
        ax.axhline(0, color='gray', lw=0.5)
        ax.axvline(0, color='gray', lw=0.5)

        lines, dots = [], []
        for c in ('red','green','blue'):
            ln, = ax.plot([], [], color=c, lw=1)
            dtp = ax.scatter([], [], s=50, color=c)
            lines.append(ln); dots.append(dtp)

        def init():
            for ln, dtp in zip(lines, dots):
                ln.set_data([], [])
                dtp.set_offsets(empty)
            return lines + dots

        def update(frame):
            for ln, dtp, b in zip(lines, dots, bodies):
                xs, ys = zip(*b.orbit[:frame+1])
                ln.set_data(xs, ys)
                dtp.set_offsets((xs[-1], ys[-1]))
            return lines + dots

        ani = FuncAnimation(
            fig, update,
            frames=len(bodies[0].orbit),
            init_func=init,
            interval=interval,
            blit=True,
        )
        plt.show()


if __name__ == "__main__":
    # build a tiny test set and visualize one of each
    make_random_dataset(n_per_class=1, dt=1e-5, max_frames=500)
    animate_random_cases(dt=1e-5, max_frames=500, interval=20)
