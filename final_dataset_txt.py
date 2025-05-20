import pandas as pd
from tqdm import tqdm
from newnewfart import Body, recenter, verlet

# 1) Load your data
df = pd.read_csv(
    'dataset.txt',
    delim_whitespace=True,
    dtype={
        'm1': float, 'm2': float, 'm3': float,
        'x1': float, 'v1': float, 'v2': float,
        'T': float, 'stability': str
    }
)

# 2) Prepare containers
records = []
dt = 0.0001

# Build a counter for all labels that appear in your file
counts = {lab: 0 for lab in df['stability'].unique()}

# 3) Simulate with progress bar
for idx, row in tqdm(df.iterrows(), total=len(df), desc="Simulating runs"):
    # Unpack parameters
    m1, m2, m3 = row['m1'], row['m2'], row['m3']
    x1, v1, v2, T = row['x1'], row['v1'], row['v2'], row['T']
    
    # Initialize bodies
    b1 = Body(x1,   0.0, m1,  0.0, v1, r=0.1, col='r')
    b2 = Body(1.0,  0.0, m2,  0.0, v2, r=0.1, col='g')
    vy3 = -(m1*v1 + m2*v2) / m3
    b3 = Body(0.0,  0.0, m3,  0.0, vy3, r=0.1, col='b')
    bodies = [b1, b2, b3]
    recenter(bodies)
    
    # Integrate and store every 100 steps
    steps = int(10 * T / dt)
    verlet(bodies, dt, steps, recenter_every=0, store_each=100)
    
    # Extract trajectories
    traj = {
        'b1_orbit': bodies[0].orbit,
        'b2_orbit': bodies[1].orbit,
        'b3_orbit': bodies[2].orbit,
        'b1_vel':   bodies[0].velhist,
        'b2_vel':   bodies[1].velhist,
        'b3_vel':   bodies[2].velhist,
    }
    
    # Count using the original label (could be 'C', 'D', 'S', etc.)
    orig_lab = row['stability']
    counts[orig_lab] += 1
    
    # Record everything
    records.append({
        'm1': m1, 'm2': m2, 'm3': m3,
        'x1': x1, 'v1': v1, 'v2': v2, 'T': T,
        'stability': orig_lab,
        'trajectories': traj
    })

# 4) Save your augmented dataset as a .txt
out_path = 'dataset_with_trajectories.txt'
pd.DataFrame(records).to_csv(
    out_path,
    sep=' ',    # space‐delimited
    index=False
)

# 5) Print a final tally for every label
print(f"\nFinished {len(records)} runs. Results:")
for label, ct in counts.items():
    print(f"  → {label:11s}: {ct}")
