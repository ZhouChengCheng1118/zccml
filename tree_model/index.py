# __Author__:Zcc
import numpy as np
from scipy import stats


def f_entropy(y):
    #  输入y列，转换成每个类的概率
    p = np.bincount(y) / float(y.shape[0])

    ep = stats.entropy(p)  # 计算经验熵
    if ep == -float('inf'):
        return 0.0
    return ep


def information_gain(y, splits, rate = False):
    #  splits是切割后的y列,rate是否计算信息增益率
    splits_entropy = sum([f_entropy(split) * (float(split.shape[0]) / y.shape[0]) for split in splits])
    if rate:
        feature_entropy = [float(split.shape[0])/y.shape[0] for split in splits]
        return (f_entropy(y) - splits_entropy)/stats.entropy(feature_entropy)

    return f_entropy(y) - splits_entropy


def f_gini(y):
    #  基尼系数
    return 1-np.sum((np.bincount(y) / float(y.shape[0])) ** 2)


def gini(y, splits):
    return sum([f_gini(split) * (float(split.shape[0]) / y.shape[0]) for split in splits])


def mse_criterion(y, splits):
    y_mean = np.mean(y)
    return sum([np.sum((split - y_mean) ** 2) * (float(split.shape[0]) / y.shape[0]) for split in splits])


def split(X, y, value):
    """给定一个value和feature，根据feature是否大于value分成两部分，返回两部分y的值
    """
    left_mask = (X < value)
    right_mask = (X >= value)
    return y[left_mask], y[right_mask]


def get_split_mask(X, column, value):
    """给定一个数据集，列名，值，根据该列是否大于某个值，返回2个bool数组
    """
    left_mask = (X[:, column] < value)
    right_mask = (X[:, column] >= value)
    return left_mask, right_mask


def split_dataset(X, y, column, value, return_X=True):
    """给定X数据集和列名，返回切分后的2个y值字典
    """
    left_mask, right_mask = get_split_mask(X, column, value)

    left, right = {}, {}
    left['y'] = y['y'][left_mask]
    right['y'] = y['y'][right_mask]

    if return_X:
        left_X, right_X = X[left_mask], X[right_mask]
        return left_X, right_X, left, right
    else:
        return left, right


def _find_splits(X):
    """返回每列的所有分割点
    """
    split_values = set()
    x_unique = list(np.unique(X))

    for i in range(1, len(x_unique)):
        average = (x_unique[i-1] + x_unique[i]) / 2.0
        split_values.add(average)

    return list(split_values)











