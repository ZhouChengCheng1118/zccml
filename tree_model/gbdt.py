# __Author__:Zcc

import autograd.numpy as np
from autograd import elementwise_grad

from base.base import BaseEstimator
from tree_model.cart import DecisionTreeRegression


class GradientBoosting(BaseEstimator):
    """Gradient boosting trees with cart."""

    def __init__(self, n_estimators=10, learning_rate=0.1, max_features=10, max_depth=2, min_samples_split=10):
        self.min_samples_split = min_samples_split
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.max_features = max_features
        self.n_estimators = n_estimators
        self.trees = []
        self.loss = None

    def fit(self, X, y=None):
        self._setup_input(X, y)
        self._train()

    def _train(self):
        # Initialize model with zeros
        y_pred = np.zeros(self.n_samples, np.float32)

        for n in range(self.n_estimators):
            residuals = -elementwise_grad(self.loss)(y_pred, self.y)  # 计算负梯度
            tree = DecisionTreeRegression(max_features=self.max_features, max_depth=self.max_depth,
                                          min_samples_split=self.min_samples_split)
            # Pass multiple target values to the tree learner
            targets = {
                # 负梯度值作为拟合目标
                'y': residuals,
                # Actual target values
                'actual': self.y,
                # Predictions from previous step
                'y_pred': y_pred
            }
            tree.fit(self.X, targets)
            predictions = tree.predict(self.X)
            y_pred += self.learning_rate * predictions  # 每棵树结果相加
            self.trees.append(tree)

    def _predict(self, X=None):
        y_pred = np.zeros(X.shape[0], np.float32)

        for i, tree in enumerate(self.trees):
            y_pred += self.learning_rate * tree.predict(X)
        return y_pred

    def predict(self, X=None):
        return self._predict(X)


class GradientBoostingRegressor(GradientBoosting):

    @staticmethod
    def mean_squared_error(predicted, actual):
        return np.mean((predicted - actual)**2)

    def fit(self, X, y=None):
        self.loss = self.mean_squared_error
        super(GradientBoostingRegressor, self).fit(X, y)


class GradientBoostingClassifier(GradientBoosting):

    @staticmethod
    def binary_crossentropy(predicted, actual):
        return np.log(1 + np.exp(-actual * predicted))

    def fit(self, X, y=None):
        # Convert labels from {0, 1} to {-1, 1}
        y = (y * 2) - 1
        self.loss = self.binary_crossentropy
        super(GradientBoostingClassifier, self).fit(X, y)

    def predict(self, X=None):
        return 0.5*(np.tanh(self._predict(X)) + 1)  # 输出的值通过sigmod函数转换

