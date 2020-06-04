'''
优点
    速度: 是混合模型学习算法中最快的算法．
    无偏差性: 这个算法仅仅只是最大化可能性，并不会使均值偏向于0，或是使聚类大小偏向于可能适用或者可能不适用的特殊结构。
缺点：
    奇异性: 当每个混合模型没有足够多的点时，会很难去估算对应的协方差矩阵，
            同时该算法会发散并且去寻找具有无穷大似然函数值的解，除非人为地正则化相应的协方差。
    分量的数量: 这个算法总是会使用它所能用的全部分量，所以在缺失外部线索的情况下，
            需要留存数据或者信息理论标准来决定用多少个分量

'''

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from sklearn import mixture

n_samples = 300

np.random.seed(0)

shifted_gaussian = np.random.randn(n_samples, 2) + np.array([20, 20])

C = np.array([[0., -0.7], [3.5, .7]])
stretched_gaussian = np.dot(np.random.randn(n_samples, 2), C)

X_train = np.vstack([shifted_gaussian, stretched_gaussian])

clf = mixture.GaussianMixture(n_components=20, covariance_type='full')
clf.fit(X_train)

x = np.linspace(-20., 30.)
y = np.linspace(-20., 40.)
X, Y = np.meshgrid(x, y)
XX = np.array([X.ravel(), Y.ravel()]).T
Z = -clf.score_samples(XX)
Z = Z.reshape(X.shape)

CS = plt.contour(X, Y, Z, norm=LogNorm(vmin=1.0, vmax=1000.0),
                 levels=np.logspace(0, 3, 10))
CB = plt.colorbar(CS, shrink=0.8, extend='both')
plt.scatter(X_train[:, 0], X_train[:, 1], .8)

plt.title('Negative log-likelihood predicted by a GMM')
plt.axis('tight')
plt.show()
