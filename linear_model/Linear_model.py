# __Author__:Zcc
import logging

import autograd.numpy as np
from autograd import grad

from base.base import BaseEstimator
from metrics.metrics import mean_squared_error, binary_crossentropy,\
     categorical_crossentropy

np.random.seed(2000)


class BasicRegression(BaseEstimator):
    def __init__(
            self,
            lr=0.001,
            penalty='None',
            C=0.01,
            tolerance=0.0001,
            max_iters=100,
            mini_batch_size=1):
        """
        Parameters
        ----------
        lr : float, default 0.001
            Learning rate.
        penalty : str, {'l1','l2','None'}, default None
            Regularization function name.
        C : float, default 0.01
            The regularization coefficient.
        tolerance : float, default 0.001
            If the gradient descent updates are smaller than `tolerance`, then
            stop optimization process.
        max_iters : int, default 10000
            The maximun number of iterations
        """
        self.lr = lr
        self.penalty = penalty
        self.C = C
        self.tolerance = tolerance
        self.max_iters = max_iters
        self.errors = []
        self.theta = []
        self.n_samples, self.n_features = None, None
        self.cost_func = None
        self.mini_batch_size = mini_batch_size

    def _batch_size(self):
        if self.mini_batch_size <= 0 or self.mini_batch_size > self.n_samples:
            raise ValueError('batch_size value error')

    def _loss(self, w):
        """加正则的损失函数，每个算法不同
        """
        raise NotImplementedError()

    def init_cost(self):
        raise NotImplementedError()

    def _add_penalty(self, loss, w):
        if self.penalty == "l1":  # 带L1正则的损失函数1阶不可导，没什么卵用。。
            loss += self.C * np.abs(w[:-1]).sum()
        elif self.penalty == "l2":
            loss += 0.5 * self.C * (w[:-1] ** 2).mean()
        return loss

    def _cost(self, X, y, theta):
        """计算当前参数下损失函数的值
        """
        prediction = X.dot(theta)
        error = self.cost_func(y, prediction)
        return error

    @staticmethod
    def _add_intercept(X):
        b = np.ones([X.shape[0], 1])
        return np.concatenate([b, X], axis=1)

    def fit(self, X, y=None):
        self._setup_input(X, y)
        self.init_cost()
        self.n_samples, self.n_features = X.shape
        self._batch_size()

        # 初始化权重和截距
        self.theta = np.random.normal(size=(self.n_features + 1), scale=0.5)

        # X增加截距列，全部为1
        self.X = self._add_intercept(self.X)

        self._train()

    def _gradient_descent(self):
        theta = self.theta
        errors = [self._cost(self.X, self.y, theta)]

        if self.y.ndim == 1 and self.X.ndim == 2:
            arr = np.hstack([self.X, self.y[:, np.newaxis]])
        else:
            arr = np.hstack([self.X, self.y])
        if self.penalty == 'l1':
            raise NotImplementedError('L1正则优化算法未实现')

        for i in range(1, self.max_iters + 1):
            np.random.shuffle(arr)
            mini_batches = (arr[k:k + self.mini_batch_size] for k in range(0, self.n_samples, self.mini_batch_size))
            for mini_batch in mini_batches:
                self._tmp_x = mini_batch[:, :-1]
                self._tmp_y = mini_batch[:, -1]
                cost_d = grad(self._loss)  # 对w求导
                delta = cost_d(theta)      # 将theta代入求导后的公式
                theta -= self.lr * delta  # 更新theta

            errors.append(self._cost(self.X, self.y, theta))
            logging.info('Iteration %s, error %s' % (i, errors[i]))

            error_diff = np.abs(errors[i - 1] - errors[i])
            if error_diff < self.tolerance:
                logging.info('Convergence has reached.')
                break
        return theta, errors  # 最终的theta和errors列表

    def _train(self):
        self.theta, self.errors = self._gradient_descent()
        logging.info(' Theta: %s' % self.theta.flatten())

    def _predict(self, X=None):
        X = self._add_intercept(X)
        return X.dot(self.theta)


class LinearRegression(BasicRegression):

    def init_cost(self):
        self.cost_func = mean_squared_error

    def _loss(self, w):
        loss = self.cost_func(self._tmp_y, np.dot(self._tmp_x, w))
        return self._add_penalty(loss, w)


class LogisticRegression(BasicRegression):

    def init_cost(self):
        self.cost_func = binary_crossentropy

    @staticmethod
    def sigmoid(x):
        return 0.5*(np.tanh(x) + 1)

    def _loss(self, w):
        loss = self.cost_func(self._tmp_y, self.sigmoid(np.dot(self._tmp_x, w)))
        return self._add_penalty(loss, w)

    def _predict(self, X=None):
        X = self._add_intercept(X)
        prob = self.sigmoid(X.dot(self.theta))
        return np.vstack([1 - prob, prob]).T

    def predict_proba(self,X=None):
        return self._predict(X)


