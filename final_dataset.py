import numpy as np
import pandas as pd
from tqdm import tqdm
from generate_Li_dataset import Body, recenter, verlet

# ─── 1) Load the initial‐conditions ─────────────────────────────────────────
df = pd.read_csv(
    'dataset.txt',
    delim_whitespace=True,
    dtype={
        'm1': float, 'm2': float, 'm3': float,
        'x1': float, 'v1': float, 'v2': float,
        'T':  float, 'stability': str
    }
)

# ─── 2) Set up simulation parameters ───────────────────────────────────────
dt          = 1e-4
store_each  = 100         # match your original decimation
periods     = 10          # integrate for 10 periods
N_runs      = len(df)

# Precompute how many stored steps each run would produce
n_steps_list = [
    int(periods * row['T'] / dt / store_each) + 1
    for _, row in df.iterrows()
]
# We’ll truncate to the shortest run so X has a fixed time‐axis
min_steps = min(n_steps_list)

# ─── 3) Preallocate arrays ─────────────────────────────────────────────────
# We'll store only positions here: (x1,y1,x2,y2,x3,y3) so 6 features
X = np.zeros((N_runs, min_steps, 6), dtype=np.float32)

# Map your labels ('C','D','S',…) to integers 0…K-1
labels = sorted(df['stability'].unique())
label_map = {lab:i for i,lab in enumerate(labels)}
y = np.zeros((N_runs,), dtype=np.int64)

# ─── 4) Run simulations ────────────────────────────────────────────────────
for i, row in tqdm(df.iterrows(), total=N_runs, desc="Building dataset"):
    # Unpack
    m1, m2, m3 = row['m1'], row['m2'], row['m3']
    x1, v1, v2, T = row['x1'], row['v1'], row['v2'], row['T']
    
    # Initialize bodies
    b1 = Body(x1,   0.0, m1,  0.0, v1, r=0.1, col='r')
    b2 = Body(1.0,  0.0, m2,  0.0, v2, r=0.1, col='g')
    vy3 = -(m1*v1 + m2*v2) / m3
    b3 = Body(0.0,  0.0, m3,  0.0, vy3, r=0.1, col='b')
    bodies = [b1, b2, b3]
    recenter(bodies)
    
    # Integrate
    steps = int(periods * T / dt)
    verlet(bodies, dt, steps, recenter_every=0, store_each=store_each)
    
    # Collect orbits, convert to arrays
    orb1 = np.array(b1.orbit[:min_steps], dtype=np.float32)  # shape (min_steps, 2)
    orb2 = np.array(b2.orbit[:min_steps], dtype=np.float32)
    orb3 = np.array(b3.orbit[:min_steps], dtype=np.float32)
    
    # Fill X: [x1,y1 | x2,y2 | x3,y3]
    X[i,:,0:2] = orb1
    X[i,:,2:4] = orb2
    X[i,:,4:6] = orb3
    
    # Label
    y[i] = label_map[row['stability']]

# ─── 5) Save as a compressed .npz ──────────────────────────────────────────
np.savez_compressed('threebody_dataset.npz',
                    X=X,         # shape (N_runs, min_steps, 6)
                    y=y,         # shape (N_runs,)
                    labels=labels)

print(f"Built dataset with {N_runs} samples, each {min_steps} time‐steps long.")
print("Labels mapping:", label_map)
print("Saved to threebody_dataset.npz")
