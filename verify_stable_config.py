import math
import random
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import os

# Define the Body class
class Body:
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
    body.orbit.append((body.x, body.y))

def update_velocity(body, force, dt):
    ax = force[0] / body.mass
    ay = force[1] / body.mass
    body.vx += ax * dt
    body.vy += ay * dt

def gravitational_force(body1, body2):
    G = 1.0
    dx = body2.x - body1.x
    dy = body2.y - body1.y
    distance_squared = dx**2 + dy**2 + 1e-6
    distance = math.sqrt(distance_squared)
    force_magnitude = G * body1.mass * body2.mass / distance_squared
    fx = force_magnitude * dx / distance
    fy = force_magnitude * dy / distance
    return fx, fy

def simulate(bodies, dt, steps):
    for _ in range(steps):
        for b in bodies:
            fx_total = fy_total = 0.0
            for other in bodies:
                if other is not b:
                    fx, fy = gravitational_force(b, other)
                    fx_total += fx
                    fy_total += fy
            update_velocity(b, (fx_total, fy_total), dt)
        for b in bodies:
            update_position(b, dt)

def generate_stable_configuration():
    r = 1.0
    m = 1.0
    theta = [0, 2 * math.pi / 3, 4 * math.pi / 3]
    positions = [(r * math.cos(t), r * math.sin(t)) for t in theta]
    v = 0.5
    velocities = [(-v * math.sin(t), v * math.cos(t)) for t in theta]
    bodies = [
        Body(
            x + random.uniform(-0.05, 0.05),
            y + random.uniform(-0.05, 0.05),
            0.1,
            color,
            m,
            vx + random.uniform(-0.02, 0.02),
            vy + random.uniform(-0.02, 0.02)
        )
        for (x, y), (vx, vy), color in zip(positions, velocities, ["red", "green", "blue"])
    ]
    return bodies

def animate_and_save(bodies, steps, i, trail_length=100, interval=10, out_dir="gifs"):
    os.makedirs(out_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)
    ax.set_aspect('equal')
    ax.set_title(f"Stable Configuration {i + 1}")
    trails = [[] for _ in bodies]

    def animate(frame):
        ax.clear()
        ax.set_xlim(-3, 3)
        ax.set_ylim(-3, 3)
        ax.set_aspect('equal')
        ax.set_title(f"Stable Configuration {i + 1}")

        for idx, b in enumerate(bodies):
            trail = b.orbit[max(0, frame - trail_length):frame + 1]
            xs, ys = zip(*trail)
            alpha_step = 1 / len(xs)
            for j in range(len(xs) - 1):
                ax.plot(xs[j:j + 2], ys[j:j + 2], color=b.color, alpha=alpha_step * j)
            ax.scatter(xs[-1], ys[-1], color=b.color, s=80)

    ani = FuncAnimation(fig, animate, frames=steps, interval=interval)
    out_path = os.path.join(out_dir, f"stable_config_{i + 1}.gif")
    ani.save(out_path, writer=PillowWriter(fps=1000 // interval))
    print(f"Saved: {out_path}")
    plt.close(fig)

def run_simulations():
    steps = 500  # fewer steps for speed
    dt = 0.01
    for i in range(5):
        print(f"Simulating config {i + 1}")
        bodies = generate_stable_configuration()
        simulate(bodies, dt, steps)
        animate_and_save(bodies, steps, i)

if __name__ == "__main__":
    run_simulations()
