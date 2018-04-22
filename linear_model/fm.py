# __Author__:Zcc
from base.base import BaseEstimator
from metrics.metrics import mean_squared_error, binary_crossentropy
import autograd.numpy as np
from autograd import elementwise_grad

np.random.seed(9999)


class BaseFM(BaseEstimator):
    def __init__(
            self,
            n_components=10,
            max_iter=100,
            init_stdev=0.1,
            learning_rate=0.01,
            reg_v=0.1,
            reg_w=0.5,
            reg_w0=0.1):
        """Simplified factorization machines implementation using SGD optimizer."""
        self.reg_w0 = reg_w0
        self.reg_w = reg_w
        self.reg_v = reg_v
        self.n_components = n_components
        self.lr = learning_rate
        self.init_stdev = init_stdev  # 初始化v的方差
        self.max_iter = max_iter
        self.loss = None
        self.loss_grad = None

    def fit(self, X, y=None):
        self._setup_input(X, y)
        # bias
        self.wo = np.random.randn()
        # Feature weights
        self.w = np.random.randn(self.n_features)
        # Factor weights
        self.v = np.random.normal(
            scale=self.init_stdev, size=(
                self.n_features, self.n_components))
        self._train()

    def _train(self):
        for epoch in range(self.max_iter):
            y_pred = self._predict(self.X)  # 每次更新时使用上次的参数值
            # 损失函数对预测函数的值求导，后续会乘以预测函数对各个参数求偏导的值
            loss = self.loss_grad(self.y, y_pred)
            # loss乘以每列X的特征值，即为损失函数对w的偏导,这里是所有样本的偏导和，还需要除以样本数
            w_grad = np.dot(loss, self.X) / float(self.n_samples)
            # 预测函数对w0的偏导为1，所以loss就是每个样本下，损失函数对w0的偏导(loss里面所有值乘以1)，loss求均值表示沿着所有样本的平均梯度方向更新w0
            self.wo -= self.lr * (loss.mean() + 2 *
                                  self.reg_w0 * self.wo)  # 加上正则项
            self.w -= self.lr * (w_grad + 2 * self.reg_w * self.w)
            self._factor_step(loss)

    def _factor_step(self, loss):
        for ix, x in enumerate(self.X):  # 每个样本更新一次v矩阵
            for i in range(self.n_features):
                # 某列v的偏导数，矩阵计算
                v_grad = loss[ix] * \
                    (x.dot(self.v).dot(x[i]) - self.v[i] * x[i] ** 2)
                # 每次更新1列的v
                self.v[i] -= self.lr * (v_grad + 2 * self.reg_v * self.v[i])

    def _predict(self, X=None):
        linear_output = np.dot(X, self.w)
        #  这里改写了公式，降低时间复杂度
        factors_output = np.sum(np.dot(X, self.v) **
                                2 - np.dot(X ** 2, self.v ** 2), axis=1) / 2.
        return self.wo + linear_output + factors_output


class FMRegressor(BaseFM):
    def fit(self, X, y=None):
        self.loss = mean_squared_error
        self.loss_grad = elementwise_grad(mean_squared_error)
        super(FMRegressor, self).fit(X, y)


class FMClassifier(BaseFM):
    def fit(self, X, y=None):
        self.loss = binary_crossentropy
        self.loss_grad = elementwise_grad(binary_crossentropy)
        super(FMClassifier, self).fit(X, y)

    def predict(self, X=None):
        predictions = self._predict(X)
        return np.sign(predictions)
