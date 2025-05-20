import numpy as np

# 1) Load the dataset
data = np.load('threebody_dataset.npz')

X      = data['X']       # shape (N_runs, T, 6)
y      = data['y']       # shape (N_runs,)
labels = data['labels']  # array of your stability codes

# 2) Print the “first line” (i.e. the first sample)
print("First sample (trajectories):")
print(X[0])               # this is a (T × 6) array of [x1,y1,x2,y2,x3,y3]

print("\nIts label index and code:")
print(y[0], "→", labels[int(y[0])])
