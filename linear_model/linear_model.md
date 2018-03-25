## Tips：
### SGD不能优化带L1正则的损失函数，sklearn中使用truncated gradient algorithm实现L1正则的优化算法，也可以使用Proximal Algorithm.
### logistic多分类的损失函数是categorical_crossentropy，优化的theta是一个[n_class,n_features+1]的矩阵，经过softmax函数映射后每个样本得到一个概率向量。

