import numpy as np
import json
import reservoirpy as rpy
from reservoirpy.nodes import Reservoir
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import RidgeClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.metrics import classification_report

# ─── 1) Load & preprocess the trajectory data ───────────────────────────────
with open('dataset_with_trajectories.json','r') as f:
    records = json.load(f)

N   = len(records)
T   = min(len(rec['trajectories']['b1_orbit']) for rec in records)
labels = sorted({rec['stability'] for rec in records})
label_map = {lab:i for i,lab in enumerate(labels)}

# build X with pos+vel → 12 features
X = np.zeros((N, T, 12), dtype=np.float32)
y = np.zeros((N,), dtype=np.int64)

for i, rec in enumerate(records):
    traj = rec['trajectories']
    # positions
    orb1 = np.array(traj['b1_orbit'][:T])
    orb2 = np.array(traj['b2_orbit'][:T])
    orb3 = np.array(traj['b3_orbit'][:T])
    # velocities
    vel1 = np.array(traj['b1_vel'][:T])
    vel2 = np.array(traj['b2_vel'][:T])
    vel3 = np.array(traj['b3_vel'][:T])
    X[i,:,:2]   = orb1
    X[i,:,2:4]  = orb2
    X[i,:,4:6]  = orb3
    X[i,:,6:8]  = vel1
    X[i,:,8:10] = vel2
    X[i,:,10:12]= vel3
    y[i] = label_map[rec['stability']]

# normalize
flat = X.reshape(-1,12)
mu, sig = flat.mean(0), flat.std(0)+1e-8
X = ((flat - mu)/sig).reshape(N,T,12)

# train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ─── 2) ESN Classifier wrapper ──────────────────────────────────────────────
class ESNClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, units=200, sr=0.9, lr=0.3, input_scaling=0.1, alpha=1.0):
        self.units = units
        self.sr = sr
        self.lr = lr
        self.input_scaling = input_scaling
        self.alpha = alpha

    def fit(self, X, y):
        # build reservoir + readout
        rpy.set_seed(0)
        self.reservoir_ = Reservoir(
            self.units, sr=self.sr, lr=self.lr, input_scaling=self.input_scaling
        )
        # compute average-pooled states
        N,T,F = X.shape
        states = np.zeros((N, self.units), dtype=np.float32)
        for i in range(N):
            all_s = self.reservoir_.run(X[i], reset=True)  # (T,units)
            states[i] = all_s.mean(axis=0)
        # train ridge
        self.readout_ = RidgeClassifier(alpha=self.alpha, class_weight='balanced')
        self.readout_.fit(states, y)
        return self

    def predict(self, X):
        N,T,F = X.shape
        states = np.zeros((N, self.units), dtype=np.float32)
        for i in range(N):
            all_s = self.reservoir_.run(X[i], reset=True)
            states[i] = all_s.mean(axis=0)
        return self.readout_.predict(states)

# ─── 3) Setup Randomized Search ─────────────────────────────────────────────
param_dist = {
    'units': [100, 300, 500],
    'sr': np.linspace(0.5, 1.2, 8),
    'lr': np.linspace(0.1, 0.7, 7),
    'input_scaling': [0.05, 0.1, 0.2, 0.5],
    'alpha': [0.1, 1.0, 10.0]
}

esn_clf = ESNClassifier()
search = RandomizedSearchCV(
    esn_clf,
    param_distributions=param_dist,
    n_iter=20,
    cv=3,
    scoring='f1_macro',
    random_state=42,
    verbose=2,
    n_jobs=1
)

# ─── 4) Run the search ─────────────────────────────────────────────────────
search.fit(X_train, y_train)

print("Best parameters:", search.best_params_)
print("\nValidation performance:")
print(search.best_score_)

# ─── 5) Final evaluation on hold-out test set ─────────────────────────────
best = search.best_estimator_
y_pred = best.predict(X_test)
print("\nTest-set results:")
print(classification_report(y_test, y_pred, target_names=labels))
