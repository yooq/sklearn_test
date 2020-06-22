import numpy as np
from sklearn import datasets
from sklearn.semi_supervised import LabelPropagation,LabelSpreading

# LabelPropagation,LabelSpreading  都可以做半监督学习
# LabelPropagation不改变有标记数据的原始标记
# LabelSpreadind可以一定比例的改变有标记数据的原始标记，最小化一个带有正规项的损失函数，对噪声鲁棒。

'''

    rbf (\exp(-\gamma |x-y|^2), \gamma > 0). \gamma 通过关键字 gamma 来指定。
    knn (1[x' \in kNN(x)]). k 通过关键字 n_neighbors 来指定。

'''
label_prop_model = LabelPropagation(kernel='rbf')

iris = datasets.load_iris()
rng = np.random.RandomState(42)
random_unlabeled_points = rng.rand(len(iris.target)) < 0.3
labels = np.copy(iris.target)

labels[random_unlabeled_points] = -1

label_prop_model.fit(iris.data, labels)

pre = label_prop_model.predict(iris.data)

for i,j,k in zip(pre,labels,iris.target):
    if i!=k:
        print(i,k)

print(label_prop_model.transduction_)

# label_prop_model.transduction_[]    []可填写需要预测样本的下标
