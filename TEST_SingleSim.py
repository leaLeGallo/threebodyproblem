#!/usr/bin/env python3
"""
three_body_rebound_bounce.py
-------------------------------------------
3‑body planar simulation with collision‐safe hard‑sphere bounces
and an optional Matplotlib animation.

• Integrator  : WHFast (symplectic, dt = 0.005)
• Collision   : “line” search  +  “hardsphere” resolver (e = 1)
• Radii       : 0.10 (same for all three bodies)
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import rebound

# ───────── simulation / animation parameters ─────────
G          = 2.0
mass       = 2.0
radius     = 0.10                       # hard‑sphere radius
r0         = [(-1.0, 0.0), (1.0, 0.0), (0.0, 0.8)]   # ← wider spread
v0         = [(0.0,  0.35), (0.0, -0.35), (0.0, 0.0)]

dt         = 0.005                      # small enough for collision detection
n_steps    = 6_000
step_skip  = 4

ANIMATE    = True                       # pop‑up animation window?
SAVE_MOVIE = False                      # write MP4/GIF?
movie_path = Path("orbit_bounce.mp4")
fps        = 60
max_trail  = 150
# ─────────────────────────────────────────────────────


# ---------- build simulation -------------------------------------------------
def restitution_const(sim_ptr, v_n):
    """Constant coefficient‑of‑restitution: perfectly elastic (e = 1)."""
    return 1.0

def build_sim():
    sim = rebound.Simulation()
    sim.G = G
    sim.integrator = "whfast"
    sim.dt = dt

    for (x, y), (vx, vy) in zip(r0, v0):
        sim.add(m=mass, x=x, y=y, vx=vx, vy=vy, r=radius)

    sim.move_to_com()

    # --- collision settings ---
    sim.collision = "line"               # detects overlaps during each step :contentReference[oaicite:0]{index=0}
    sim.collision_resolve = "hardsphere" # bounce, not merge/halt :contentReference[oaicite:1]{index=1}
    sim.coefficient_of_restitution = restitution_const
    return sim


# ---------- integrate & store coordinates ------------------------------------
sim = build_sim()
n_bodies = len(sim.particles)
coords   = np.zeros((n_steps, n_bodies, 2))

for i in range(n_steps):
    sim.integrate(sim.t + dt)
    coords[i] = [[p.x, p.y] for p in sim.particles]

coords_anim = coords[::step_skip]
n_frames    = coords_anim.shape[0]


# ---------- OPTIONAL matplotlib animation ------------------------------------
if ANIMATE or SAVE_MOVIE:
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_aspect("equal")
    ax.set_xlabel("x"); ax.set_ylabel("y")
    ax.set_title("3‑Body Hard‑Sphere Bounce (REBOUND)")

    # live positions (marker only) + fading trails
    scatters = [ax.plot([], [], marker="o", lw=0,
                        color=colors[k % len(colors)])[0]
                for k in range(n_bodies)]
    trails   = [ax.plot([], [], lw=0.8, alpha=0.7,
                        color=colors[k % len(colors)])[0]
                for k in range(n_bodies)]

    margin = 1.2
    ax.set_xlim(coords[:, :, 0].min()*margin, coords[:, :, 0].max()*margin)
    ax.set_ylim(coords[:, :, 1].min()*margin, coords[:, :, 1].max()*margin)

    def init():
        for art in scatters + trails:
            art.set_data([], [])
        return scatters + trails

    def update(frame):
        for k in range(n_bodies):
            x, y = coords_anim[frame, k]
            scatters[k].set_data([x], [y])           # wrap scalars ➜ no error
            start = max(frame - max_trail, 0)
            trails[k].set_data(coords_anim[start:frame+1, k, 0],
                               coords_anim[start:frame+1, k, 1])
        return scatters + trails

    ani = animation.FuncAnimation(fig, update, frames=n_frames,
                                  init_func=init, interval=1000/fps,
                                  blit=True)

    if SAVE_MOVIE:
        writer = (animation.FFMpegWriter(fps=fps, bitrate=-1)
                  if movie_path.suffix == ".mp4" else "pillow")
        ani.save(movie_path, dpi=150, writer=writer)
        print(f"Movie saved → {movie_path.resolve()}")

    if ANIMATE and not SAVE_MOVIE:
        plt.show()
