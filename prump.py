import math
import random
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from IPython.display import HTML


class Body:
    """Represents a body with methods to compute forces and displacements."""

    def __init__(self, x, y, radius, color, mass, vx=0.0, vy=0.0):
        self.x = x
        self.y = y
        self.radius = radius
        self.color = color
        self.mass = mass
        self.vx = vx
        self.vy = vy
        # store complete trajectory (for analysis & animation)
        self.orbit = [(x, y)]


# -----------------------------------------------------------------------------
# basic physics helpers
# -----------------------------------------------------------------------------

def update_position(body: Body, dt: float):
    body.x += body.vx * dt
    body.y += body.vy * dt


def update_velocity(body: Body, force: tuple[float, float], dt: float):
    ax = force[0] / body.mass
    ay = force[1] / body.mass
    body.vx += ax * dt
    body.vy += ay * dt


def gravitational_force(body1: Body, body2: Body, G: float = 1.0):
    dx = body2.x - body1.x
    dy = body2.y - body1.y
    distance_sq = dx * dx + dy * dy + 1e-9   # softening epsilon
    distance = math.sqrt(distance_sq)

    force_mag = G * body1.mass * body2.mass / distance_sq
    fx = force_mag * dx / distance
    fy = force_mag * dy / distance
    return fx, fy


# -----------------------------------------------------------------------------
# simulation driver
# -----------------------------------------------------------------------------

def simulate(bodies: list[Body], dt: float, steps: int):
    """Integrate forward *steps* times, storing flattened state every step."""
    records: list[list[float]] = []
    for _ in range(steps):
        # 1) forces & velocity update (symplectic Euler)
        for b in bodies:
            total_fx = total_fy = 0.0
            for o in bodies:
                if o is not b:
                    fx, fy = gravitational_force(b, o)
                    total_fx += fx
                    total_fy += fy
            update_velocity(b, (total_fx, total_fy), dt)

        # 2) position update & trajectory storage
        for b in bodies:
            update_position(b, dt)
            b.orbit.append((b.x, b.y))

        # 3) flattened snapshot for ML
        flat_state: list[float] = []
        for b in bodies:
            flat_state.extend([b.x, b.y, b.vx, b.vy])
        records.append(flat_state)

    return records


# -----------------------------------------------------------------------------
# outcome classification helpers
# -----------------------------------------------------------------------------

def segments_intersect(p: tuple[float, float], q: tuple[float, float],
                       r: tuple[float, float], s: tuple[float, float],
                       radius_sum: float) -> bool:
    """Return True if the two closed line segments ever come within *radius_sum*."""

    px, py = p
    qx, qy = q
    rx, ry = r
    sx, sy = s

    # relative motion of b1 vs b2 in one timestep
    vx, vy = qx - px, qy - py
    wx, wy = sx - rx, sy - ry
    dx, dy = px - rx, py - ry

    a = (vx - wx) ** 2 + (vy - wy) ** 2
    b = 2 * ((vx - wx) * dx + (vy - wy) * dy)
    c = dx ** 2 + dy ** 2 - radius_sum ** 2

    disc = b * b - 4 * a * c
    if disc < 0:        # never comes that close
        return False

    # time (0..1) of closest approach along relative trajectory
    t = max(0.0, min(1.0, (-b - math.sqrt(disc)) / (2 * a))) if a else 0.0
    cx = dx + (vx - wx) * t
    cy = dy + (vy - wy) * t
    return cx * cx + cy * cy <= radius_sum * radius_sum


def is_escaping(b: Body, R: float) -> bool:
    """Simple geometric escape test: outside radius *R* and still moving outward."""
    r2 = b.x * b.x + b.y * b.y
    if r2 <= R * R:
        return False
    return (b.x * b.vx + b.y * b.vy) > 0.0


def classify_outcome(bodies: list[Body], *, G: float = 1.0, escape_R: float = 8.0):
    """Return one of "convergence", "divergence", or "stable" for the final state."""

    # 1) collision check across segment pairs
    col_d = bodies[0].radius + bodies[1].radius
    n_steps = len(bodies[0].orbit)
    for t in range(n_steps - 1):  # note the -1 (we look at segments t..t+1)
        for i, b1 in enumerate(bodies):
            for b2 in bodies[i + 1:]:
                if segments_intersect(b1.orbit[t], b1.orbit[t + 1],
                                      b2.orbit[t], b2.orbit[t + 1],
                                      col_d):
                    return "convergence"

    # 2) geometric escape – any body is already on its way out of bounds
    if any(is_escaping(b, escape_R) for b in bodies):
        return "divergence"

    # 3) energetic escape – per-body energy > 0 implies that body is unbound
    for i, bi in enumerate(bodies):
        Ki = 0.5 * bi.mass * (bi.vx ** 2 + bi.vy ** 2)
        Vi = 0.0
        for j, bj in enumerate(bodies):
            if j == i:
                continue
            rij = math.hypot(bi.x - bj.x, bi.y - bj.y) + 1e-6
            Vi -= G * bi.mass * bj.mass / rij
        if Ki + Vi > 0.0:
            return "divergence"

    # 4) nothing dramatic happened – treat as (meta-)stable over the simulated interval
    return "stable"


# -----------------------------------------------------------------------------
# dataset generation utility
# -----------------------------------------------------------------------------

def make_dataset_per_class(target: int = 1,
                           *,
                           steps: int = 200,
                           dt: float = 0.01,
                           out_csv: str = "dataset.csv") -> pd.DataFrame:
    """Generate a balanced csv with *target* examples of each class."""

    wanted = {"stable": target, "divergence": target, "convergence": target}
    have = {k: 0 for k in wanted}
    rows: list[dict[str, float | str]] = []

    while any(have[k] < wanted[k] for k in wanted):
        # random initial configuration – lower speeds for faster filling
        bodies = [
            Body(random.uniform(-1, 1),
                 random.uniform(-1, 1),
                 0.1,
                 color,
                 random.uniform(0.5, 2.0),
                 random.uniform(-0.3, 0.3),
                 random.uniform(-0.3, 0.3))
            for color in ("red", "green", "blue")
        ]

        trajectory = simulate(bodies, dt, steps)
        label = classify_outcome(bodies)

        if have[label] >= wanted[label]:
            continue  # already got enough of this class – discard

        # flatten trajectory -> single row
        row: dict[str, float | str] = {}
        for t, state in enumerate(trajectory):
            for i in range(3):
                x, y, vx, vy = state[4 * i: 4 * i + 4]
                row[f"x{i + 1}_t{t}"] = x
                row[f"y{i + 1}_t{t}"] = y
                row[f"vx{i + 1}_t{t}"] = vx
                row[f"vy{i + 1}_t{t}"] = vy
        row["label"] = label

        rows.append(row)
        have[label] += 1
        print(f"Collected {have[label]}/{wanted[label]} of {label}")

    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    print("\nFinal counts:", have)
    return df


# -----------------------------------------------------------------------------
# quick demo – generate tiny balanced set & animate one sample per class
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    # 1) tiny dataset – raise target for real work
    df = make_dataset_per_class(target=1)
    print(df.label.value_counts())

    # 2) animate one example per class
    steps = 200
    colors = ["red", "green", "blue"]

    for label in ["stable", "divergence", "convergence"]:
        sample = df[df.label == label].sample(1).iloc[0]

        # rebuild orbits
        orbits: list[list[tuple[float, float]]] = []
        for i in (1, 2, 3):
            orbit_i = [(sample[f"x{i}_t{t}"], sample[f"y{i}_t{t}"]) for t in range(steps)]
            orbits.append(orbit_i)

        # animation callback
        def animate_case(frame, orbits=orbits):
            ax.clear()
            ax.axhline(0, color="black", linewidth=0.5)
            ax.axvline(0, color="black", linewidth=0.5)
            for c, orbit in zip(colors, orbits):
                xs, ys = zip(*orbit[: frame + 1])
                ax.plot(xs, ys, color=c, linewidth=2)
                ax.scatter(xs[-1], ys[-1], color=c, s=100)

        fig, ax = plt.subplots(figsize=(5, 5))
        fig.suptitle(label.capitalize(), fontsize=14)
        ax.set_xlim(-5, 5)
        ax.set_ylim(-5, 5)

        ani = FuncAnimation(fig, animate_case, frames=steps, interval=50)
        plt.show()
