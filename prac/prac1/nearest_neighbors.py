import sklearn.neighbors
import numpy as np
from distances import euclidean_distance, cosine_distance


class KNNClassifier:
    def __init__(self, k=5, strategy='my_own', metric='euclidean', weights=False, test_block_size=100):
        self.k = k
        self.strategy = strategy
        self.metric = metric
        self.weights = weights
        self.test_block_size = test_block_size
        self.supervised_model = None

        if strategy in ('brute', 'kd_tree', 'ball_tree'):
            self.supervised_model = sklearn.neighbors.NearestNeighbors(
                                                    n_neighbors=k,
                                                    algorithm=strategy,
                                                    metric=metric)
        elif strategy != 'my_own':
            raise TypeError('Wrong strategy!')

    def fit(self, X, y):
        if self.strategy in ('brute', 'kd_tree', 'ball_tree'):
            self.supervised_model.fit(X)
            self.y = y
        else:
            self.X = X
            self.y = y

    def find_kneighbors(self, X, return_distance):
        if self.strategy in ('brute', 'kd_tree', 'ball_tree'):
            return self.supervised_model.kneighbors(X, self.k, return_distance)
        distances = np.zeros((X.shape[0], self.k))
        if self.metric == 'euclidean':
            distances = euclidean_distance(X, self.X)
        elif self.metric == 'cosine':
            distances = cosine_distance(X, self.X)
        else:
            raise TypeError('Wrong metric!')

        neighbours_idx = np.zeros((X.shape[0], self.k))
        neighbours_dist = np.zeros((X.shape[0], self.k))
        for i in range(X.shape[0]):
            neighbours_idx[i] = distances[i].argsort()[:self.k]
        neighbours_idx = neighbours_idx.astype(int)
        for i in range(X.shape[0]):
            neighbours_dist[i] = distances[i][neighbours_idx[i]]
        if return_distance:
            return neighbours_dist, neighbours_idx
        else:
            return neighbours_idx

    def predict(self, X):
        y_hat = np.array([])
        for batch in test_block_generator(X, self.test_block_size):
            neighbours_dist, neighbours_idx = self.find_kneighbors(batch, return_distance=True)
            neighbours_class = np.apply_along_axis(lambda i: self.y[i], axis=0, arr=neighbours_idx)
            prediction = np.zeros(batch.shape[0])
            if not self.weights:
                for i in range(batch.shape[0]):
                    counts = np.bincount(neighbours_class[i, :])
                    prediction[i] = np.argmax(counts)
            else:
                for i in range(batch.shape[0]):
                    counts = np.zeros(len(np.unique(self.y)))
                    for j in range(self.k):
                        counts[neighbours_class[i][j]] += 1 / (neighbours_dist[i][j] + 1e-5)
                    prediction[i] = np.argmax(counts)
            y_hat = np.hstack((y_hat, prediction))
        return y_hat


def test_block_generator(X, test_block_size):
    current = 0
    while current < X.shape[0]:
        start, stop = current, current + test_block_size
        current += test_block_size
        yield X[start:stop]
