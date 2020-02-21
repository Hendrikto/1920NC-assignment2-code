import numpy as np


def classify(z):
    return int(z[0] >= 0.7 or (z[0] <= 0.3 and z[1] >= -0.2 - z[0]))


def generate_instances(n=400):
    X = np.random.uniform(-1, 1, (n, 2))
    y = np.apply_along_axis(classify, axis=1, arr=X)
    return X, y
