import numpy as np
import pandas as pd

# ─── 1) Load the data ──────────────────────────────────────────────
data = np.load('threebody_dataset2.npz', allow_pickle=True)
print("Keys in archive:", data.files)
# ['X','y','labels','masses','init_pos','init_vel']

# ─── 2) Grab each array ───────────────────────────────────────────
X        = data['X']         # shape (N_runs, min_steps, 6)
y        = data['y']         # shape (N_runs,)
labels   = data['labels']    # list of strings
masses   = data['masses']    # shape (N_runs, 3)
init_pos = data['init_pos']  # shape (N_runs, 3, 2)
init_vel = data['init_vel']  # shape (N_runs, 3, 2)

# Print shapes
print(f"X.shape      = {X.shape}")
print(f"y.shape      = {y.shape}")
print(f"masses.shape = {masses.shape}")
print(f"init_pos     = {init_pos.shape}")
print(f"init_vel     = {init_vel.shape}")

# ─── 3) Inspect a single run ──────────────────────────────────────
i = 0  # first sample
print(f"\nSample #{i}")
print("  label       =", labels[y[i]])
print("  masses      =", masses[i])
print("  init_pos b1 =", init_pos[i,0],  "b2 =", init_pos[i,1], "b3 =", init_pos[i,2])
print("  init_vel b1 =", init_vel[i,0],  "b2 =", init_vel[i,1], "b3 =", init_vel[i,2])

# ─── 4) Tabulate all initial conditions ───────────────────────────
# Build a DataFrame for easy viewing
df_init = pd.DataFrame({
    'm1': masses[:,0],
    'm2': masses[:,1],
    'm3': masses[:,2],
    'b1_x0': init_pos[:,0,0],
    'b1_y0': init_pos[:,0,1],
    'b2_x0': init_pos[:,1,0],
    'b2_y0': init_pos[:,1,1],
    'b3_x0': init_pos[:,2,0],
    'b3_y0': init_pos[:,2,1],
    'b1_vx0': init_vel[:,0,0],
    'b1_vy0': init_vel[:,0,1],
    'b2_vx0': init_vel[:,1,0],
    'b2_vy0': init_vel[:,1,1],
    'b3_vx0': init_vel[:,2,0],
    'b3_vy0': init_vel[:,2,1],
    'stability': [labels[idx] for idx in y]
})
print("\nInitial‐conditions table:")
print(df_init.head())

# ─── 5) Accessing the trajectory time‐series ───────────────────────
# e.g. the x‐coordinate of body 1 for run i:
x1_traj = X[i, :, 0]  
print(f"\nRun {i} has {x1_traj.size} time‐steps; here are the first 10 x1’s:\n", x1_traj[:10])
