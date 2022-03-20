from oracles import BinaryLogistic
import numpy as np
import scipy
import time
from tqdm import tqdm
from sklearn.metrics import accuracy_score
from sklearn.utils import shuffle


class GDClassifier:
    def __init__(
        self, loss_function='binary_logistic', step_alpha=1, step_beta=0,
        tolerance=1e-6, max_iter=1000, **kwargs
    ):
        if loss_function == 'binary_logistic':
            self.loss_function = BinaryLogistic(kwargs['l2_coef'])
        else:
            raise TypeError('wrong loss function')
        self.step_alpha = step_alpha
        self.step_beta = step_beta
        self.tolerance = tolerance
        self.max_iter = max_iter

    def fit(self, X, y, X_val=None, y_val=None, w_0=None, trace=False):
        if X_val is None or y_val is None and trace:
            X_val = X
            y_val = y
        if trace:
            history = {'time': [], 'func': [], 'accuracy': []}
        if w_0 is None:
            self.w = np.zeros(X.shape[1])
        else:
            self.w = w_0

        for k in range(1, self.max_iter + 1):
            start_time = time.time()
            eta = self.step_alpha / k ** self.step_beta
            new_w = self.w - eta * self.loss_function.grad(X, y, self.w)
            current_func = self.loss_function.func(X, y, self.w)
            new_func = self.loss_function.func(X, y, new_w)
            if trace:
                history['func'].append(round(current_func, 12))
                history['time'].append(time.time() - start_time)
                history['accuracy'].append(accuracy_score(y_val, self.predict(X_val)))
            if abs(current_func - new_func) < self.tolerance:
                if trace:
                    history['func'].append(round(new_func, 12))
                    history['time'].append(0)
                    history['accuracy'].append(accuracy_score(y_val, self.predict(X_val)))
                break
            self.w = new_w
        if trace:
            return history

    def predict(self, X):
        return np.sign(X @ self.w)

    def predict_proba(self, X):
        return scipy.special.expit(X @ self.w)

    def get_objective(self, X, y):
        return self.loss_function.func(X, y, self.w)

    def get_gradient(self, X, y):
        return self.loss_function.grad(X, y, self.w)

    def get_weights(self):
        return self.w


class SGDClassifier(GDClassifier):
    def __init__(
        self, loss_function='binary_logistic', batch_size=1024, step_alpha=1, step_beta=0,
        tolerance=1e-6, max_iter=1000, random_seed=153, **kwargs
    ):
        """
        loss_function - строка, отвечающая за функцию потерь классификатора.
        Может принимать значения:
        - 'binary_logistic' - бинарная логистическая регрессия
        batch_size - размер подвыборки, по которой считается градиент
        step_alpha - float, параметр выбора шага из текста задания
        step_beta- float, параметр выбора шага из текста задания
        tolerance - точность, по достижении которой, необходимо прекратить оптимизацию
        Необходимо использовать критерий выхода по модулю разности соседних значений функции:
        если |f(x_{k+1}) - f(x_{k})| < tolerance: то выход
        max_iter - максимальное число итераций (эпох)
        random_seed - в начале метода fit необходимо вызвать np.random.seed(random_seed).
        Этот параметр нужен для воспроизводимости результатов на разных машинах.
        **kwargs - аргументы, необходимые для инициализации
        """
        if loss_function == 'binary_logistic':
            self.loss_function = BinaryLogistic(kwargs['l2_coef'])
        else:
            raise TypeError('wrong loss function')
        self.batch_size = batch_size
        self.step_alpha = step_alpha
        self.step_beta = step_beta
        self.tolerance = tolerance
        self.max_iter = max_iter
        np.random.seed(random_seed)

    def fit(self, X, y, X_val=None, y_val=None, w_0=None, trace=False, log_freq=1):
        if X_val is None or y_val is None and trace:
            X_val = X
            y_val = y
        history = {'epoch_num': [], 'time': [], 'func': [], 'accuracy': [], 'weights_diff': []}
        if w_0 is None:
            self.w = np.zeros(X.shape[1])
        else:
            self.w = w_0

        num_iter = 0
        obj_processed = 0
        old_weights = self.w
        curr_epoch = 0
        start_time = time.time()

        while num_iter < self.max_iter:
            for k, (X_batch, y_batch) in enumerate(self.get_batch(X, y)):
                
                eta = self.step_alpha / (round(obj_processed / X.shape[0]) + 1) ** self.step_beta
                new_w = self.w - eta * self.loss_function.grad(X_batch, y_batch, self.w)
                ### hot fix
                if np.sum(np.isnan(self.w)):
                    print("nan in weights", num_iter)
                ###
                current_func = self.loss_function.func(X_batch, y_batch, self.w)
                new_func = self.loss_function.func(X_batch, y_batch, new_w)
                if abs(current_func - new_func) < self.tolerance:
                    break
                self.w = new_w
                num_iter += 1
                obj_processed += self.batch_size
                if obj_processed / X.shape[0] - curr_epoch >= log_freq:
                    curr_epoch = obj_processed / X.shape[0]
                    if trace:
                        history['epoch_num'].append(obj_processed / X.shape[0])
                        history['func'].append(round(current_func, 12))
                        history['time'].append(time.time() - start_time)
                        history['accuracy'].append(accuracy_score(y_val, self.predict(X_val)))
                        history['weights_diff'].append(np.linalg.norm(old_weights - self.w))
                        old_weights = self.w
                        start_time = time.time()
        return history


    def get_batch(self, X, y):
        sh_X, sh_y = shuffle(X, y)
        current = 0
        while current < X.shape[0]:
            start, stop = current, current + self.batch_size
            current += self.batch_size
            yield sh_X[start:stop], sh_y[start:stop]
