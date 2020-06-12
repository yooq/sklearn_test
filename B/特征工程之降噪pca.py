'''
保持维度，去掉噪音

'''

from sklearn.decomposition import PCA
from sklearn.datasets import load_digits

data =load_digits()

pca = PCA(n_components=0.5,svd_solver='full')
x = pca.fit_transform(data.data)
print(x.shape)
print(x)

x_ = pca.inverse_transform(x)
print(x_.shape)
print(x_)
