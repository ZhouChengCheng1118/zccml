# __Author__:Zcc
import random


import numpy as np
from tree_model.index import gini, mse_criterion, split, split_dataset

from base.base import BaseEstimator

random.seed(222)


class Tree(BaseEstimator):

    def __init__(self, max_features, min_samples_split, max_depth, criterion):
        self.impurity = None
        self.threshold = None
        self.column_index = None
        self.outcome = None
        self.criterion = criterion

        self.left_child = None
        self.right_child = None
        self.max_features = max_features
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth

    @property
    def is_terminal(self):
        #  当child节点为None时返回True
        return not bool(self.left_child and self.right_child)

    def _find_splits(self, column):
        #  找到该特征所有的分割点，这里仅针对连续特征
        split_values = set()

        x_unique = list(np.unique(column))
        for i in range(1, len(x_unique)):
            # Find a point between two values
            average = (x_unique[i - 1] + x_unique[i]) / 2.0
            split_values.add(average)

        return list(split_values)

    def _if_end(self, *args, **kwargs):
        raise NotImplementedError

    def _find_best_split(self, X, y, max_features):
        """找到最佳的特征和分割点"""

        # 列抽样
        subset = random.sample(range(0, X.shape[1]), max_features)
        min_criterion, min_col, max_val = None, None, None

        for column in subset:
            split_values = self._find_splits(X[:, column])
            for value in split_values:
                splits = split(X[:, column], y['y'], value)
                criterion = self.criterion(y['y'], splits)

                if (min_criterion is None) or (criterion < min_criterion):
                    min_col, min_val, min_criterion = column, value, criterion
        return min_col, min_val, min_criterion

    def _grow_child(self, *args, **kwargs):
        """生长子树
        """
        raise NotImplementedError

    def fit(self, X, y=None):

        if not isinstance(y, dict): # 将y转换成字典
            y = {'y': y}

        try:
            assert (X.shape[0] > self.min_samples_split) # 限制最小划分的样本数

            assert (self.max_depth > 0)  # 限制树的深度

            assert (len(np.unique(y['y'])) != 1)  # 判断节点的y是否都是同一类

            if self.max_features is None:
                self.max_features = X.shape[1]

            # 当前样本下分割的最佳特征，值和基尼系数
            column, value, criterion = self._find_best_split(X, y, self.max_features)

            assert self._if_end(criterion)  # 根据分类和回归的不同，定义基尼系数或均方差的最大值

            self.column_index = column
            self.threshold = value
            self.impurity = criterion

            # 分割数据集，分成左右2个X和y
            left_X, right_X, left_y, right_y = split_dataset(X, y, column, value)

            # 递归的生长子树
            self._grow_child(left_X, left_y, right_X, right_y)

        except AssertionError:
            self._calculate_leaf_value(y)

    def _calculate_leaf_value(self, y):
        """计算叶子节点的值
        """
        raise NotImplementedError

    def predict_row(self, row):
        """预测某一个样本"""
        if not self.is_terminal:  # 首先判断根节点是否是叶节点
            if row[self.column_index] < self.threshold:
                return self.left_child.predict_row(row)
            else:
                return self.right_child.predict_row(row)
        return self.outcome

    def predict(self, X=None):
        result = np.zeros(X.shape[0])
        for i in range(X.shape[0]):
            result[i] = self.predict_row(X[i, :])
        return result


class DecisionTreeClassifier(Tree):

    def __init__(self, max_features=None, min_samples_split=10, max_depth=10, criterion=None, min_gini=0.01):
        super(DecisionTreeClassifier, self).__init__(max_features=max_features, min_samples_split=min_samples_split,
                                                     max_depth=max_depth, criterion=criterion)
        self.min_gini = min_gini
        self.criterion = gini

    def _if_end(self, criterion):
        return criterion > self.min_gini

    def _grow_child(self, left_X, left_y, right_X, right_y):
        """生长子树
        """
        self.left_child = DecisionTreeClassifier(self.max_features, self.min_samples_split, self.max_depth - 1, self.criterion,
                                                 self.min_gini)
        self.left_child.fit(left_X, left_y)

        self.right_child = DecisionTreeClassifier(self.max_features, self.min_samples_split, self.max_depth - 1, self.criterion,
                                                  self.min_gini)

        self.right_child.fit(right_X, right_y)

    def _calculate_leaf_value(self, y):
        """计算叶节点的概率值,输出概率最高的一类"""
        self.outcome = np.bincount(y['y']).argmax()


class DecisionTreeRegression(Tree):

    def __init__(self, max_features=None, min_samples_split=10, max_depth=None, criterion=None, min_mse=0.):
        super(DecisionTreeRegression, self).__init__(max_features=max_features, min_samples_split=min_samples_split,
                                                     max_depth=max_depth, criterion=criterion)
        self.min_mse = min_mse
        self.criterion = mse_criterion

    def _if_end(self, criterion):
        return criterion > self.min_mse

    def _grow_child(self, left_X, left_y, right_X, right_y):
        """生长子树
        """
        self.left_child = DecisionTreeRegression(self.max_features, self.min_samples_split, self.max_depth - 1, self.criterion,
                                                 self.min_mse)
        self.left_child.fit(left_X, left_y)

        self.right_child = DecisionTreeRegression(self.max_features, self.min_samples_split, self.max_depth - 1, self.criterion,
                                                  self.min_mse)
        self.right_child.fit(right_X, right_y)

    def _calculate_leaf_value(self, y):
        """计算叶节点的概率值"""
        self.outcome = np.mean(y['y'])







































