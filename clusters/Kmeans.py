# __Author__:Zcc
import random
import numpy as np
from base import BaseEstimator
from metrics.distance import euclidean_distance

random.seed(222)


class KMeans(BaseEstimator):
    y_required = False

    def __init__(self, K=3, max_iters=100, init='random'):
        self.K = K
        self.max_iters = max_iters
        self.init = init
        self.clusters = [[] for _ in range(self.K)]
        self.centroids = []

    def _initialize_centroids(self,init):
        """初始化质心."""

        if init == 'random':
            self.centroids = [self.X[i] for i in
                              random.sample(range(self.n_samples),self.K)]
        elif init == '++':
            self.centroids = [random.choice(self.X)]
            while len(self.centroids) < self.K:
                self.centroids.append(self._choose_next_center())
        else:
            raise ValueError('未知类型')

    def _choose_next_center(self):
        distance = self._dist_from_centers()  # 得到每个样本点到最近的一个质心点的距离
        probs = distance / distance.sum()  # 距离越远的比值越大
        cumprobs = probs.cumsum()   # 累加值，比值越大，差越大
        r = random.random()  # 0-1随机值
        ind = np.where(cumprobs >= r)[0][0]  # 返回比r大的第一个值索引，距离越远的概率越大
        return self.X[ind]

    def _dist_from_centers(self):
        """计算每个样本点到最近的一个质心点的距离，返回一个一维数组
        """
        return np.array([min([euclidean_distance(i,c) for c in self.centroids]) for i in self.X])

    def _predict(self, X=None):
        self._initialize_centroids(self.init)
        centroids = self.centroids

        for _ in range(self.max_iters):
            self._assign(centroids)  # 分配所有样本点到最近的类中
            centroids_old = centroids
            # 每个类得到新的质心点
            centroids = [self._get_centroid(cluster) for cluster in self.clusters]

            if self._is_converged(centroids_old, centroids):  # 判断是否收敛
                break

        self.centroids = centroids  # 最终的质心点

        return self._get_predictions()

    def _assign(self, centroids):
        """分配所有样本点到最近的质心点
        """
        for row in range(self.n_samples):
            for i,cluster in enumerate(self.clusters):
                if row in cluster:  # 如果当前样本点在某个clusters中，先去除，最后通过计算在决定添加到哪个类中
                    self.clusters[i].remove(row)
                    break

            closest = self._closest(row, centroids) # 得到该样本点最近质心点索引
            self.clusters[closest].append(row)

    def _closest(self, fpoint, centroids):
        """
        输入一个样本点的索引和一个质心点列表，返回一个最近的质心点索引
        """
        closest_index = None
        closest_distinct = None
        for i, point in enumerate(centroids):
            dist = euclidean_distance(self.X[fpoint], point)
            if closest_index is None or dist < closest_distinct:
                closest_index = i
                closest_distinct = dist
        return closest_index

    def _is_converged(self, centroids_old, centroids):
        """每个类的新老质心点距离是否为0，返回True或者False
        """
        distance = 0
        for i in range(self.K):
            distance += euclidean_distance(centroids_old[i],centroids[i])
        return distance == 0

    def _get_centroid(self, cluster):
        """给定某一类cluster（索引），对样本每个features求均值,得到新的质心点
        """
        return [np.mean(np.take(self.X[:, i ], cluster)) for i in range(self.n_features)]

    def _get_predictions(self):
        """
        把self.clusters每个样本所属的类别按顺序转换成predictions
        """
        predictions = np.empty(self.n_samples)

        for i, cluster in enumerate(self.clusters):
            for index in cluster:
                predictions[index] = i
        return predictions






















