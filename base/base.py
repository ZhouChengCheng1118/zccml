# __Author__:Zcc
import numpy as np

class BaseEstimator(object):
    X = None
    y = None
    y_required = True
    fit_required = True
    def _setup_input(self,X,y=None):
        """校验格式
        """
        if not isinstance(X,np.ndarray):
            X = np.array(X)

        if X.size == 0:
            raise ValueError('数据集大小不能为0')

        if X.ndim == 1:
            self.n_samples, self.n_features = 1,X.shape[0]
        else:
            self.n_samples, self.n_features = X.shape[0],X.shape[1]

        self.X = X

        if self.y_required:
            if y is None:
                raise ValueError('缺少参数y')

            if not isinstance(y,np.ndarray):
                y = np.array(y)

            if y.size == 0:
                raise ValueError('目标变量个数必须大于0')

        self.y = y

    def fit(self,X,y=None):
        self._setup_input(X, y)

    def predict(self, X=None):
        if not isinstance(X,np.ndarray):
            X = np.array(X)

        if self.X is not None or not self.fit_required:
            scores = self._predict(X)
            if len(scores.shape) == 1:
                return scores
            else:
                return scores.argmax(axis=1)  # 二分类的情况
        else:
            raise ValueError('还没有训练模型')

    def _predict(self, X=None):
        raise NotImplementedError()













