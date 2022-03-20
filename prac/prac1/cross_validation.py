import numpy as np
from nearest_neighbors import KNNClassifier
import sklearn.neighbors
import sklearn.metrics


def kfold(n, n_folds):
    idx_array = []
    indices = np.arange(n)
    fold_sizes = np.full(n_folds, n // n_folds, dtype=int)
    fold_sizes[:n % n_folds] += 1
    current = 0
    for fold_size in fold_sizes:
        start, stop = current, current + fold_size
        idx_train = np.hstack((indices[0:start], indices[stop:]))
        idx_test = indices[start:stop]
        idx_array.append((idx_train, idx_test))
        current = stop
    return idx_array
# source: sklearn's github


def knn_cross_val_score(X, y, k_list, score, cv, **kwargs):
    X = np.array(X)
    y = np.array(y)
    clf = KNNClassifier(k=k_list[0], **kwargs)
    if cv is None:
        cv = kfold(X.shape[0], 3)
    cv_scores = {k: np.zeros(len(cv)) for k in k_list}
    for i, (train, test) in enumerate(cv):
        clf.k = max(k_list)
        clf.fit(X[train], y[train])
        neighbours_dist, neighbours_class = clf.find_kneighbors(X[test], return_distance=True)
        neighbours_class = np.apply_along_axis(lambda i: clf.y[i], axis=0, arr=neighbours_class)
        for k in k_list:
            clf.k = k
            ans = pretrained_predict(clf, X[test], neighbours_dist[:, :k], neighbours_class[:, :k])
            cv_scores[k][i] = accuracy(ans, y[test])
    return cv_scores


def pretrained_predict(clf, X, neighbours_dist, neighbours_class):
        prediction = np.zeros(X.shape[0])
        if not clf.weights:
            for i in range(X.shape[0]):
                counts = np.bincount(neighbours_class[i, :])
                prediction[i] = np.argmax(counts)
        else:
            for i in range(X.shape[0]):
                counts = np.zeros(len(np.unique(clf.y)))
                for j in range(clf.k):
                    counts[neighbours_class[i][j]] += 1 / (neighbours_dist[i][j] + 1e-5)
                prediction[i] = np.argmax(counts)
        return prediction


def accuracy(a, b):
    return np.sum(a == b) / a.shape[0]
