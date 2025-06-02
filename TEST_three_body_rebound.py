#!/usr/bin/env python3
"""
three_body_sim.py ─ A full‑featured dataset generator for planar three‑body chaos
=========================================================================

• Generates random initial conditions in 2‑D that respect centre‑of‑mass balance.
• Integrates each system with one of REBOUND's integrators (default: symplectic
  Leapfrog), conserving energy well over long times.
• Classifies every run on‑the‑fly as
      ─ "stable"     … no escape, no merger within t_max
      ─ "divergent"  … at least one body crosses esc_radius
      ─ "merger"     … any pair comes closer than merge_dist
• Saves all trajectories (variable length) and labels in a compressed .npz file
  that you can load directly in PyTorch or NumPy.
• Parallelises across CPU cores for fast generation of thousands of runs.

Usage ─────────────────────────────────────────────────────────────────────
    python three_body_sim.py --n_runs 5000 --out data/three_body.npz \
                             --t_max 100 --integrator ias15 --njobs 8

The output file contains two entries:
    ─ trajectories : numpy.ndarray dtype=object; each element is a  (T_i × 6) array
    ─ labels       : numpy.ndarray of strings  ("stable" | "divergent" | "merger")

-------------------------------------------------------------------------
"""

from __future__ import annotations
import argparse
import math
import multiprocessing as mp
from pathlib import Path
from typing import Tuple, List

import numpy as np
import rebound

###############################################################################
#                              Physics helpers                               #
###############################################################################

def random_planar_ic(
    r_max: float = 2.0,
    v_scale: float = 0.3,
    masses: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    rng: np.random.Generator | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (positions, velocities) in centre‑of‑mass frame.

    * positions : (3,3) array, Z component = 0
    * velocities: (3,3) array, Z component = 0
    """
    if rng is None:
        rng = np.random.default_rng()

    # Uniform in disk: r = R_max * sqrt(U)
    r = r_max * np.sqrt(rng.random(3))
    theta = rng.random(3) * 2.0 * math.pi
    x = r * np.cos(theta)
    y = r * np.sin(theta)

    # Isotropic planar velocities
    vx, vy = v_scale * rng.normal(size=(2, 3))

    # Assemble 3‑D vectors (z = 0)
    pos = np.column_stack([x, y, np.zeros(3)])
    vel = np.column_stack([vx, vy, np.zeros(3)])

    m = np.asarray(masses)[:, None]

    # Shift to centre‑of‑mass (CoM) frame
    pos -= (m * pos).sum(axis=0) / m.sum()
    vel -= (m * vel).sum(axis=0) / m.sum()

    return pos, vel


def integrate_three_body(
    seed: int,
    t_max: float,
    dt: float,
    esc_radius: float,
    merge_dist: float,
    integrator: str,
) -> Tuple[np.ndarray, str]:
    """Simulate one system and return (trajectory, outcome)."""
    rng = np.random.default_rng(seed)
    pos, vel = random_planar_ic(rng=rng)

    sim = rebound.Simulation()
    sim.G = 1.0
    sim.integrator = integrator
    sim.dt = dt  # ignored by IAS15 but set anyway

    for i in range(3):
        sim.add(m=1.0, x=pos[i, 0], y=pos[i, 1], z=0.0,
                vx=vel[i, 0], vy=vel[i, 1], vz=0.0)

    sim.move_to_com()

    # Pre‑allocate (rough upper bound) to avoid Python list overhead
    est_steps = int(t_max / dt) + 1
    traj = np.empty((est_steps, 6), dtype=np.float32)

    outcome = "stable"
    step = 0
    while sim.t < t_max:
        # Record planar positions
        p0, p1, p2 = sim.particles[0:3]
        traj[step] = [p0.x, p0.y, p1.x, p1.y, p2.x, p2.y]
        step += 1

        # Divergence: any body beyond esc_radius
        if any(p.x**2 + p.y**2 > esc_radius**2 for p in (p0, p1, p2)):
            outcome = "divergent"
            break

        # Merger: any pair closer than merge_dist
        if (
            (p0.x - p1.x) ** 2 + (p0.y - p1.y) ** 2 < merge_dist**2 or
            (p0.x - p2.x) ** 2 + (p0.y - p2.y) ** 2 < merge_dist**2 or
            (p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2 < merge_dist**2
        ):
            outcome = "merger"
            break

        sim.integrate(sim.t + dt)
    else:
        # loop exhausted → t_max reached with no merger/divergence
        outcome = "stable"

    return traj[:step].copy(), outcome

###############################################################################
#                           Batch‑generation driver                           #
###############################################################################

def _worker(args):
    """Helper for multiprocessing Pool."""
    seed, cfg = args
    return integrate_three_body(
        seed=seed,
        t_max=cfg.t_max,
        dt=cfg.dt,
        esc_radius=cfg.esc_radius,
        merge_dist=cfg.merge_dist,
        integrator=cfg.integrator,
    )


def generate_dataset(cfg) -> None:
    seeds = np.random.SeedSequence(cfg.master_seed).spawn(cfg.n_runs)
    payload = [(s_entropy.entropy, cfg) for s_entropy in seeds]

    with mp.Pool(processes=cfg.njobs) as pool:
        results = pool.map(_worker, payload)

    trajectories, labels = zip(*results)
    labels = np.asarray(labels)

    out_path = Path(cfg.out).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, trajectories=np.asarray(trajectories, dtype=object), labels=labels)
    print(f"✹ Saved {cfg.n_runs} runs -> {out_path} (classes: stable/divergent/merger = "
          f"{np.unique(labels, return_counts=True)[1]})")

###############################################################################
#                                   CLI                                       #
###############################################################################

def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Generate labelled three‑body trajectories with REBOUND")
    p.add_argument("--n_runs", type=int, default=1000, help="Number of simulations to run (default: 1000)")
    p.add_argument("--t_max", type=float, default=50.0, help="Integration time span (default: 50)")
    p.add_argument("--dt", type=float, default=1e-2, help="Timestep for symplectic integrators (default: 0.01)")
    p.add_argument("--esc_radius", type=float, default=10.0, help="Escape radius criterion")
    p.add_argument("--merge_dist", type=float, default=0.05, help="Merger distance criterion")
    p.add_argument("--integrator", choices=["leapfrog", "ias15", "whfast"], default="leapfrog",
                   help="REBOUND integrator to use (default: leapfrog)")
    p.add_argument("--out", default="three_body_dataset.npz", help="Output .npz file path")
    p.add_argument("--njobs", type=int, default=mp.cpu_count(), help="Parallel worker processes (default: all cores)")
    p.add_argument("--master_seed", type=int, default=2025, help="Seed for SeedSequence spawning (reproducibility)")
    return p

###############################################################################
#                                   main                                      #
###############################################################################

def main():
    cfg = build_arg_parser().parse_args()
    print("⏳ Generating data … this may take a moment…")
    generate_dataset(cfg)


if __name__ == "__main__":
    main()
