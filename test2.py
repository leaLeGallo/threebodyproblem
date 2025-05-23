# animate_row250.py
import numpy as np
import matplotlib.pyplot as plt

# ─── 1) Load your dataset ────────────────────────────────────────────────
data       = np.load('threebody_dataset2.npz', allow_pickle=True)
X, y, labs = data['X'], data['y'], data['labels']
N_runs     = X.shape[0]

# ─── 2) Choose the run you want ──────────────────────────────────────────
run_idx = 199
if not (0 <= run_idx < N_runs):
    raise IndexError(f"Dataset has {N_runs} runs; cannot pick run {run_idx}")
print(f"Animating run #{run_idx} (label = {labs[y[run_idx]]!r})")

# ─── 3) Extract its three orbits ────────────────────────────────────────
traj    = X[run_idx]         # shape (min_steps, 6)
x1, y1  = traj[:,0], traj[:,1]
x2, y2  = traj[:,2], traj[:,3]
x3, y3  = traj[:,4], traj[:,5]
num_frm = len(x1)

# ─── 4) Set up the figure ───────────────────────────────────────────────
plt.ion()
fig, ax = plt.subplots(figsize=(6,6))
ax.set_aspect('equal', 'box')

# Zoom to the envelope of this single run
all_x = np.hstack([x1, x2, x3])
all_y = np.hstack([y1, y2, y3])
pad   = 0.1
xmin, xmax = all_x.min(), all_x.max()
ymin, ymax = all_y.min(), all_y.max()
ax.set_xlim(xmin - pad*(xmax-xmin), xmax + pad*(xmax-xmin))
ax.set_ylim(ymin - pad*(ymax-ymin), ymax + pad*(ymax-ymin))

ax.set_title(f"Run #{run_idx} (label={labs[y[run_idx]]!r})")
ax.set_xlabel("x")
ax.set_ylabel("y")

# ─── 5) Brute-force animation ────────────────────────────────────────────
for frame in range(num_frm):
    ax.clear()
    # draw trails
    ax.plot(x1[:frame], y1[:frame], '-', color='red')
    ax.plot(x2[:frame], y2[:frame], '-', color='green')
    ax.plot(x3[:frame], y3[:frame], '-', color='blue')
    # draw current positions
    ax.scatter(x1[frame], y1[frame], color='red',   s=50)
    ax.scatter(x2[frame], y2[frame], color='green', s=50)
    ax.scatter(x3[frame], y3[frame], color='blue',  s=50)
    ax.set_aspect('equal', 'box')
    plt.pause(0.02)

plt.ioff()
plt.show()
