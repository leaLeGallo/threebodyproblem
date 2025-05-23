import numpy as np
import pandas as pd

# ─── 1) Load the .npz ───────────────────────────────────────────────────────
data = np.load('threebody_dataset2.npz', allow_pickle=True)

X        = data['X']         # shape (N_runs, min_steps, 6)
y        = data['y']         # shape (N_runs,)
labels   = data['labels']    # list of stability labels
masses   = data['masses']    # shape (N_runs, 3)
init_pos = data['init_pos']  # shape (N_runs, 3, 2)
init_vel = data['init_vel']  # shape (N_runs, 3, 2)

# ─── 2) Print “first line” (initial conditions + label) ──────────────────
i = 0
print("Run #0 initial conditions:")
print(f"  masses    = {masses[i]}")
print(f"  init_pos  = b1{init_pos[i,0]}, b2{init_pos[i,1]}, b3{init_pos[i,2]}")
print(f"  init_vel  = b1{init_vel[i,0]}, b2{init_vel[i,1]}, b3{init_vel[i,2]}")
print(f"  stability = {labels[y[i]]}")

# ─── 3) Print “first full simulation” (entire trajectory) ───────────────
print(f"\nRun #0 full trajectory (x,y for each body at each stored step):")
print("  Columns = [x1, y1,  x2, y2,  x3, y3]")
print(X[i])   # this will print a min_steps×6 array
