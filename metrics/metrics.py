# __Author__:Zcc
import autograd.numpy as np

EPS = 1e-15


def squared_error(actual, predicted):
    return (actual - predicted) ** 2


def mean_squared_error(actual, predicted):
    return np.mean(squared_error(actual, predicted))


def binary_crossentropy(actual, predicted):
    """二分类logloss
    """
    predicted = np.clip(predicted, EPS, 1 - EPS)
    return -np.mean(actual * np.log(predicted) +
                    (1 - actual) * np.log(1 - predicted))


def categorical_crossentropy(actual, predicted):
    """多分类logloss,要先OneHotEncoder
    """
    predicted = np.clip(predicted, EPS, 1 - EPS)
    loss = -np.sum(actual * np.log(predicted))
    return loss / float(actual.shape[0])


def accuracy(actual, predicted):
    return (actual == predicted).sum()/float(actual.shape[0])





