import numpy as np
import scipy
from scipy.sparse import csr_matrix, spdiags


class BaseSmoothOracle:
    """
    Базовый класс для реализации оракулов.
    """
    def func(self, w):
        """
        Вычислить значение функции в точке w.
        """
        raise NotImplementedError('Func oracle is not implemented.')

    def grad(self, w):
        """
        Вычислить значение градиента функции в точке w.
        """
        raise NotImplementedError('Grad oracle is not implemented.')


class BinaryLogistic(BaseSmoothOracle):
    def __init__(self, l2_coef=0):
        self.l2_coef = l2_coef

    def func(self, X, y, w):
        M = y * (X @ w)
        # loss = np.mean(-np.log(scipy.special.expit(M))) + (self.l2_coef / 2.0) * np.dot(w, w)
        loss = np.mean(np.logaddexp(0, -M)) + (self.l2_coef / 2.0) * np.dot(w, w)
        return loss

    def grad(self, X, y, w):
        M = y * (X @ w)
        if type(X) == scipy.sparse.csr.csr_matrix:
            crutch = -X.multiply(y.reshape(-1, 1))
            dw = crutch.multiply((scipy.special.expit(M) * np.exp(-np.clip(M, -709, 709)))[:, np.newaxis])
            dw = np.mean(dw, axis=0)
            dw += self.l2_coef * w
        else:
            crutch = -y[:, np.newaxis] * X
            dw = np.mean((scipy.special.expit(M) * np.exp(-np.clip(M, -709, 709)))[:, np.newaxis] * crutch, axis=0)
            dw += self.l2_coef * w
        return np.asarray(dw).reshape(-1)
