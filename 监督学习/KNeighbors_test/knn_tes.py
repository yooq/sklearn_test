from sklearn.neighbors import NearestNeighbors
import numpy as np
X = np.array([[-2, -1], [-3, -2], [1, 1], [2, 1], [-1, 3]])
'''
判断数据集中，与当前样本距离最近的k个样本，最近距离搜索方式kd_tree 或ball_tree 
KD 树的方法对于低维度 (D < 20) 近邻搜索非常快, 当 D 增长到很大时, 效率变低: 这就是所谓的 “维度灾难” 的一种体现
Ball 为了解决 KD 树在高维上效率低下的问题, ball 树 数据结构就被研发出来了

k为3时，第0个样本与0,1,2,样本最近，第4个样本与2,3,4样本最近。
[[0 1 2]
 [1 0 2]
 [2 3 4]
 [3 2 4]
 [4 2 3]]
 
[[1. 1. 1. 0. 0.]
 [1. 1. 1. 0. 0.]
 [0. 0. 1. 1. 1.]
 [0. 0. 1. 1. 1.]
 [0. 0. 1. 1. 1.]]

'''
nbrs = NearestNeighbors(n_neighbors=3, algorithm='kd_tree').fit(X)
distances, indices = nbrs.kneighbors(X)
# print(indices)
# print(nbrs.kneighbors_graph(X).toarray())

'''使用KDTree查找最近邻'''
from sklearn.neighbors import KDTree
import numpy as np
X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
kdt = KDTree(X, leaf_size=30, metric='euclidean')
d=kdt.query(X, k=2, return_distance=False)
# print(d)

'''最近邻分类'''
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn import neighbors, datasets

n_neighbors = 15
iris = datasets.load_iris()

X = iris.data[:, :2]
y = iris.target

h = .02  # step size in the mesh

cmap_light = ListedColormap(['orange', 'cyan', 'cornflowerblue'])
cmap_bold = ListedColormap(['darkorange', 'c', 'darkblue'])

for weights in ['uniform', 'distance']:
    '''
    uniform 每个点同等重要
    distance 按其距离的倒数加权点。在这种情况下，查询点的近邻将具有比更远的邻居有更大的影响力。
    '''
    clf = neighbors.KNeighborsClassifier(n_neighbors, weights=weights)
    clf.fit(X, y)
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

    Z = Z.reshape(xx.shape)
    plt.figure()
    plt.pcolormesh(xx, yy, Z, cmap=cmap_light)

    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold,
                edgecolor='k', s=20)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.title("3-Class classification (k = %i, weights = '%s')"
              % (n_neighbors, weights))
plt.show()


'''最近邻回归
KNN算法不仅可以用于分类，还可以用于回归。通过找出一个样本的k个最近邻居，
将这些邻居的某个（些）属性的平均值赋给该样本，就可以得到该样本对应属性的值。
'''
np.random.seed(0)
X = np.sort(5 * np.random.rand(40, 1), axis=0)
T = np.linspace(0, 5, 500)[:, np.newaxis]
y = np.sin(X).ravel()

y[::5] += 1 * (0.5 - np.random.rand(8)) #每隔5个，y上面增加噪音
n_neighbors = 5

for i, weights in enumerate(['uniform', 'distance']):
    knn = neighbors.KNeighborsRegressor(n_neighbors, weights=weights)
    y_ = knn.fit(X, y).predict(T)

    plt.subplot(2, 1, i + 1)
    plt.scatter(X, y, color='darkorange', label='data')
    plt.plot(T, y_, color='navy', label='prediction')
    plt.axis('tight')
    plt.legend()
    plt.title("KNeighborsRegressor (k = %i, weights = '%s')" % (n_neighbors,
                                                                weights))
plt.tight_layout()
plt.show()

'''
邻域成分分析NCA,NeighborhoodComponentsAnalysis,其目的是提高最近邻分类相对于标准欧氏距离的准确性。该算法直接最大化训练集上k近邻(KNN)得分的随机变量，还可以拟合数据的低维线性投影
可以自然地处理多类问题，而不需要增加模型的大小，并且不引入需要用户进行微调的额外参数。
'''
from sklearn.neighbors import (NeighborhoodComponentsAnalysis,
KNeighborsClassifier)
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y,
stratify=y, test_size=0.7, random_state=42)
nca = NeighborhoodComponentsAnalysis(random_state=42)
knn = KNeighborsClassifier(n_neighbors=3)
nca_pipe = Pipeline([('nca', nca), ('knn', knn)])
nca_pipe.fit(X_train, y_train)
print(nca_pipe.score(X_test, y_test))
