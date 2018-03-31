# __Author__:Zcc
import logging
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_regression, make_classification, make_blobs

from linear_model.Linear_model import LinearRegression, LogisticRegression
from metrics.metrics import mean_squared_error, accuracy
from clusters.Kmeans import KMeans
import numpy as np
logging.basicConfig(level=logging.ERROR)


def regression():
    # Generate a random regression problem
    X, y = make_regression(n_samples=1000, n_features=20,
                           n_informative=10, n_targets=1, noise=0.05,
                           random_state=1111, bias=0.5)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25,
                                                        random_state=1111)

    model = LinearRegression(lr=0.01, max_iters=2000, penalty='l2', C=0.03)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    print('regression mse', mean_squared_error(y_test, predictions))


def classification():
    # Generate a random binary classification problem.
    X, y = make_classification(n_samples=1000, n_features=20,
                               n_informative=10, random_state=1111,
                               n_classes=2, class_sep=2.5, )
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1,
                                                        random_state=1111)

    model = LogisticRegression(lr=0.01, max_iters=10, penalty='l2', C=0.01)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    print(len(predictions.shape))
    print('classification accuracy', accuracy(y_test, predictions))
    #print(y_test[:10],predictions[:10])

def kmeans_example():
    X, y = make_blobs(centers=3, n_samples=500, n_features=2,
                      shuffle=True, random_state=42)
    clusters = len(np.unique(y))
    k = KMeans(K=clusters, max_iters=150, init='++')
    k.fit(X)
    k.predict()
    print(k.clusters)



if __name__=='__main__':
    #regression()
    #classification()
    kmeans_example()



