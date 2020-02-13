import numpy as np
import time
n_samples, n_features = int(1e5), 5
rng = np.random.RandomState(0)
y = rng.randn(n_samples)
X = rng.randn(n_samples, n_features)

from my_ridge import Ridge
clf = Ridge(alpha=1.0)
t_start_scratch = time.time()
clf.fit(X, y)
t_end_scratch = time.time()
print("Scratch model weights: ", clf.coef_)


from sklearn.linear_model import Ridge
clf = Ridge(alpha=1.0)
t_start = time.time()
clf.fit(X, y)
t_end = time.time()
print("Sklearn model weights: ", clf.coef_)

print("Scratch model takes %f time, and Sklearn model takes %f time"%(t_end_scratch - t_start_scratch, t_end - t_start))