# __Author__:Zcc

import numpy as np

from base.base import BaseEstimator
from tree_model.cart import DecisionTreeRegression, DecisionTreeClassifier


class RandomForest(BaseEstimator):
    def __init__(self, n_estimators=10, max_features=None, min_samples_split=10, max_depth=None):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.n_estimators = n_estimators
        self.trees = []

    def fit(self, X, y=None):
        self._setup_input(X, y)
        if self.max_features is None:
            self.max_features = int(np.sqrt(X.shape[1]))
        else:
            assert (X.shape[1] > self.max_features)
        self._train()

    def _train(self):
        for tree in self.trees:
            tree.fit(self.X, self.y)

    def _predict(self, X=None):
        raise NotImplementedError()


class RandomForestClassifier(RandomForest):
    def __init__(self, n_estimators=10, max_features=None, min_samples_split=10, max_depth=5):
        super(RandomForestClassifier, self).__init__(n_estimators=n_estimators, max_features=max_features,
                                                     min_samples_split=min_samples_split, max_depth=max_depth)

        for i in range(self.n_estimators):
            self.trees.append(DecisionTreeClassifier(max_features=self.max_features, min_samples_split=self.min_samples_split,
                                                     max_depth=self.max_depth))

    def _predict(self, X=None):
        y_shape = np.unique(self.y).shape[0]  # y有多少类
        predictions = np.zeros(X.shape[0])

        for i in range(X.shape[0]):
            row_pred = np.zeros(y_shape)  # 2个值的一维数组
            for tree in self.trees:
                row_pred[tree.predict_row(X[i, :])] += 1

            predictions[i] = row_pred.argmax()

        return predictions


class RandomForestRegressor(RandomForest):
    def __init__(self, n_estimators=10, max_features=None, min_samples_split=10, max_depth=5):
        super(RandomForestRegressor, self).__init__(n_estimators=n_estimators, max_features=max_features,
                                                    min_samples_split=min_samples_split, max_depth=max_depth)
        # Initialize empty regression trees
        for _ in range(self.n_estimators):
            self.trees.append(DecisionTreeRegression(max_features=self.max_features, min_samples_split=self.min_samples_split,
                                                     max_depth=self.max_depth))

    def _predict(self, X=None):
        predictions = np.zeros((X.shape[0], self.n_estimators))  # 每列都是一颗树
        for i, tree in enumerate(self.trees):
            predictions[:, i] = tree.predict(X)
        return predictions.mean(axis=1)





















