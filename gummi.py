import math
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from IPython.display import HTML
import random
import pandas as pd


class Body:
    """Represents a body with methods to compute forces and displacements."""

    def __init__(self, x, y, radius, color, mass, vx=0, vy=0):
        self.x = x
        self.y = y
        self.radius = radius
        self.color = color
        self.mass = mass
        self.vx = vx
        self.vy = vy
        self.orbit = [(x, y)]


def update_position(body, dt):
    body.x += body.vx * dt
    body.y += body.vy * dt


def update_velocity(body, force, dt):
    ax = force[0] / body.mass
    ay = force[1] / body.mass
    body.vx += ax * dt
    body.vy += ay * dt


def gravitational_force(body1, body2):
    G = 1.0
    dx = body2.x - body1.x
    dy = body2.y - body1.y
    distance_squared = dx**2 + dy**2
    distance = math.sqrt(distance_squared) + 1e-6  # prevent divide by zero
    force_magnitude = G * body1.mass * body2.mass / distance_squared
    force_x = force_magnitude * dx / distance
    force_y = force_magnitude * dy / distance
    return (force_x, force_y)


def simulate(bodies, dt, steps):
    records = []
    for _ in range(steps):
        for b in bodies:
            total_fx, total_fy = 0.0, 0.0
            for o in bodies:
                if o is not b:
                    fx, fy = gravitational_force(b, o)
                    total_fx += fx
                    total_fy += fy
            update_velocity(b, (total_fx, total_fy), dt)

        for b in bodies:
            update_position(b, dt)
            b.orbit.append((b.x, b.y))

        flat = []
        for b in bodies:
            flat += [b.x, b.y, b.vx, b.vy]
        records.append(flat)

    return records


def classify_outcome(bodies, G=1.0):
    col_d = bodies[0].radius + bodies[1].radius
    n_steps = len(bodies[0].orbit)
    for t in range(n_steps):
        for i, b1 in enumerate(bodies):
            x1, y1 = b1.orbit[t]
            for b2 in bodies[i+1:]:
                x2, y2 = b2.orbit[t]
                if math.hypot(x1 - x2, y1 - y2) < col_d:
                    return "convergence"

    K = sum(0.5 * b.mass * (b.vx**2 + b.vy**2) for b in bodies)
    V = 0.0
    for i, b1 in enumerate(bodies):
        for b2 in bodies[i+1:]:
            dx = b1.x - b2.x
            dy = b1.y - b2.y
            r = math.hypot(dx, dy) + 1e-6
            V -= G * b1.mass * b2.mass / r

    if K + V > 0:
        return "divergence"

    return "stable"


def make_dataset_per_class(target=1, steps=5000, dt=0.01, out_csv="dataset.csv"):
    targets = {"stable": target, "divergence": target, "convergence": target}
    counts = {k: 0 for k in targets}
    all_rows = []

    while any(counts[label] < target for label in targets):
        bodies = [
            Body(random.uniform(-1, 1), random.uniform(-1, 1), 0.1, color,
                 random.uniform(0.5, 2), random.uniform(-1, 1), random.uniform(-1, 1))
            for color in ("red", "green", "blue")
        ]

        traj = simulate(bodies, dt, steps)
        label = classify_outcome(bodies)

        if counts[label] < target:
            row = {}
            for t, state in enumerate(traj):
                for i in range(3):
                    x, y, vx, vy = state[4*i:4*i+4]
                    row[f"x{i+1}_t{t}"] = x
                    row[f"y{i+1}_t{t}"] = y
                    row[f"vx{i+1}_t{t}"] = vx
                    row[f"vy{i+1}_t{t}"] = vy
            row["label"] = label
            all_rows.append(row)
            counts[label] += 1
            print(f"Collected {counts[label]}/{target} of {label}")

    df = pd.DataFrame(all_rows)
    df.to_csv(out_csv, index=False)
    print("\nFinal counts:", counts)
    return df


if __name__ == "__main__":
    df = make_dataset_per_class()
    print(df.label.value_counts())

    steps = 5000
    colors = ["red", "green", "blue"]

    for label in ["stable", "divergence", "convergence"]:
        sample = df[df.label == label].sample(1).iloc[0]
        orbits = [
            [(sample[f"x{i}_t{t}"], sample[f"y{i}_t{t}"]) for t in range(steps)]
            for i in (1, 2, 3)
        ]

        def animate_case(frame, orbits=orbits):
            ax.clear()
            ax.axhline(0, color='black', linewidth=0.5)
            ax.axvline(0, color='black', linewidth=0.5)
            for color, orbit in zip(colors, orbits):
                xs, ys = zip(*orbit[:frame+1])
                ax.plot(xs, ys, color=color, linewidth=2)
                ax.scatter(xs[-1], ys[-1], color=color, s=100)

        fig, ax = plt.subplots(figsize=(5, 5))
        fig.suptitle(label.capitalize(), fontsize=14)
        ax.set_xlim(-5, 5)
        ax.set_ylim(-5, 5)

        ani = FuncAnimation(fig, animate_case, frames=steps, interval=50)
        plt.show()
