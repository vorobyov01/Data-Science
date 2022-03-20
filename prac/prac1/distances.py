import numpy as np


def euclidean_distance(X, Y):
    M1 = (X ** 2) @ np.ones((Y.shape[1], Y.shape[0]))  # -> (N, D) @ (D, M) = (N, M)
    M2 = np.ones((X.shape[0], X.shape[1])) @ (Y.T ** 2)  # -> (N, D) @ (D, M) = (N, M)
    M3 = -2 * (X @ Y.T)  # -> (N, M)
    return np.sqrt(M1 + M2 + M3)  # -> (N, M)
# source: https://www.robots.ox.ac.uk/~albanie/notes/Euclidean_distance_trick.pdf


def cosine_distance(X, Y):
    num = np.dot(X, Y.T)
    p1 = np.sqrt(np.sum(X ** 2, axis=1))[:, np.newaxis]
    p2 = np.sqrt(np.sum(Y ** 2, axis=1))[np.newaxis, :]
    return 1.0 - num / (p1 * p2)
# source: https://towardsdatascience.com/cosine-similarity-matrix-using-broadcasting-in-python-2b1998ab3ff3
