# __Author__:Zcc
from scipy.linalg import svd
import numpy as np
import logging

from base.base import BaseEstimator


class PCA(BaseEstimator):
    y_required = False

    def __init__(self, n_components, solver='svd'):
        """
        Parameters
        ----------
        n_components : int
        solver : str, default 'svd'
            {'svd', 'eigen'}
        """
        self.n_components = n_components
        self.solver = solver
        self.components = None  # 特征向量组成的矩阵
        self.mean = None  # 每列的均值

    def fit(self, X, y=None):
        self.mean = np.mean(X, axis=0)  # 计算每一列的均值，返回一个一维数组
        self._decompose(X)

    def _decompose(self, X):
        """
        得到前n个特征向量矩阵
        """
        X = X.copy()
        X -= self.mean  # 每列均值化

        if self.solver == 'svd': # x可以不是方阵，所以不用计算协方差矩阵
            _, s, Vh = svd(X, full_matrices=True)
            s = np.real(s)
            Vh = np.real(Vh)
        elif self.solver == 'eigen':
            s,Vh = np.linalg.eig(np.cov(X.T))
            Vh = np.real(Vh.T)
            s = np.real(s)

        s_squared = s ** 2  # 奇异值s的平方等于特征值，s默认从大到小排列，s是新的协方差矩阵(对角矩阵)的主对角线元素，即新数据集每列的方差，s一般不会为负数
        variance_ratio = s_squared / (s_squared).sum()
        logging.info('Explained variance ratio: %s' % (variance_ratio[0:self.n_components]))
        self.components = Vh[0:self.n_components]

    def transform(self, X):
        X = X.copy()
        X -= self.mean
        return np.dot(X, self.components.T)

    def _predict(self, X=None):
        return self.transform(X)
