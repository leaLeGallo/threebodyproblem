import math
import random
import pandas as pd

# Body class
class Body:
    def __init__(self, x, y, mass, vx, vy, color="black"):
        self.x = x
        self.y = y
        self.mass = mass
        self.vx = vx
        self.vy = vy
        self.color = color
        self.orbit = [(x, y)]

def gravitational_force(b1, b2, G=1.0):
    dx = b2.x - b1.x
    dy = b2.y - b1.y
    dist_sq = dx**2 + dy**2 + 1e-6
    dist = math.sqrt(dist_sq)
    force = G * b1.mass * b2.mass / dist_sq
    fx = force * dx / dist
    fy = force * dy / dist
    return fx, fy

def simulate(bodies, steps=5000, dt=0.01):
    for _ in range(steps):
        forces = []
        for i, b1 in enumerate(bodies):
            fx_total, fy_total = 0, 0
            for j, b2 in enumerate(bodies):
                if i != j:
                    fx, fy = gravitational_force(b1, b2)
                    fx_total += fx
                    fy_total += fy
            forces.append((fx_total, fy_total))

        for i, b in enumerate(bodies):
            fx, fy = forces[i]
            ax = fx / b.mass
            ay = fy / b.mass
            b.vx += ax * dt
            b.vy += ay * dt
            b.x += b.vx * dt
            b.y += b.vy * dt
            b.orbit.append((b.x, b.y))

    return bodies

# Utility for perturbation
def perturb(val, eps=1e-2):
    return val + random.uniform(-eps, eps)

# Known stable configuration: Lagrange triangle orbit
def make_lagrange_config():
    r = 1.0
    m = 1.0
    angles = [0, 2 * math.pi / 3, 4 * math.pi / 3]
    positions = [(r * math.cos(a), r * math.sin(a)) for a in angles]
    v_mag = 0.5
    velocities = [(-v_mag * math.sin(a), v_mag * math.cos(a)) for a in angles]
    bodies = [Body(perturb(x), perturb(y), m, perturb(vx), perturb(vy)) 
              for (x, y), (vx, vy) in zip(positions, velocities)]
    return bodies

# Known stable configuration: Euler linear orbit
def make_euler_config():
    m = 1.0
    bodies = [
        Body(perturb(-1.0), 0.0, m, 0.0, perturb(0.3)),
        Body(perturb(0.0), 0.0, m, 0.0, perturb(0.0)),
        Body(perturb(1.0), 0.0, m, 0.0, perturb(-0.3)),
    ]
    return bodies

# Known stable configuration: figure-eight (approx)
def make_figure8_config():
    # Moore’s 3-body figure-eight initial conditions
    return [
        Body(perturb(-0.97000436), perturb(0.24308753), 1.0, perturb(0.4662036850), perturb(0.4323657300)),
        Body(perturb(0.97000436), perturb(-0.24308753), 1.0, perturb(0.4662036850), perturb(0.4323657300)),
        Body(perturb(0.0), perturb(0.0), 1.0, perturb(-0.93240737), perturb(-0.86473146)),
    ]

# Divergent configurations
def make_divergent_config():
    return [
        Body(random.uniform(-1, 1), random.uniform(-1, 1), 1.0, random.uniform(5, 10), random.uniform(5, 10)),
        Body(random.uniform(-1, 1), random.uniform(-1, 1), 1.0, random.uniform(5, 10), random.uniform(5, 10)),
        Body(random.uniform(-1, 1), random.uniform(-1, 1), 1.0, random.uniform(5, 10), random.uniform(5, 10)),
    ]

# Convergent configurations
def make_convergent_config():
    return [
        Body(-0.1, 0.0, 1.0, 0.5, 0.0),
        Body(0.1, 0.0, 1.0, -0.5, 0.0),
        Body(0.0, 0.2, 1.0, 0.0, -0.5),
    ]

# Generate dataset
def generate_configurations():
    data = {"stable": [], "divergent": [], "convergent": []}

    for _ in range(2):
        data["stable"].append(make_lagrange_config())
        data["stable"].append(make_euler_config())
        data["stable"].append(make_figure8_config())

    for _ in range(5):
        data["divergent"].append(make_divergent_config())
        data["convergent"].append(make_convergent_config())

    # Trim stable to 5
    data["stable"] = data["stable"][:5]

    return data

# Collect simulation results
def simulate_all(data, steps=5000, dt=0.01):
    all_results = []
    for label, configs in data.items():
        for idx, config in enumerate(configs):
            sim = simulate(config, steps=steps, dt=dt)
            result = {
                "label": label,
                "initial_conditions": [(b.x, b.y, b.vx, b.vy, b.mass) for b in config],
                "trajectories": [b.orbit for b in config],
            }
            all_results.append(result)
    return all_results

# Example usage:
if __name__ == "__main__":
    dataset = generate_configurations()
    full_trajectories = simulate_all(dataset, steps=3000, dt=0.01)
    # At this point, `full_trajectories` contains everything
