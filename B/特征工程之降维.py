'''
PCA（主成分分析）：
    以方差作为衡量指标，使用特征值分解来找新的特征空间。
    PCA通过线性变换将原始数据变换为一组各维度线性无关的表示。
   高维降低到低维：
        将一组N维向量降为K维（K大于0，小于N），其目标是选择K个单位（模为1）正交基，
        使得原始数据变换到这组基上后（投影），各字段两两间协方差为0（各字段正交），而字段的方差则尽可能大（在正交的约束下，取最大的K个方差）。



SVD（特征值分解）：
    SVD是一种强大的降维工具，同时也用于去噪，或图片压缩，本质上SVD是使用奇异值分解
'''


from sklearn.decomposition import PCA,TruncatedSVD
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt


data = load_iris()
X = data.data
Y = data.target
print(X.shape)

pca = PCA(n_components=2) #n_components 表示维度，可以根据 pca.explained_variance_ratio_ 对元数据信息占比来选择合适的值
#***************************************#
pca = PCA(n_components='mle')  # mle 自动选择最好的特征个数
pca = PCA(n_components=0.97,svd_solver='full') #至少保留元数据信息的97%，
# svd_slover--[randomized（适合巨大矩阵,效果好于full）,full（完整的SVD，数据量小好用）,auto(自行选择用哪种),arpack(使用稀疏矩阵)]

#***************************************#



X_features = pca.fit_transform(X)


# plt.figure()
# colors =['red','black','orange']
# for i in [0,1,2]:
#     plt.scatter(X[Y==i,0],X[Y==i,1],alpha=0.9,c=colors[i],label=data.target_names[i])
# plt.legend()
# plt.show()

print('variance_',pca.explained_variance_) #查看每一位方差
print('ratio',pca.explained_variance_ratio_) #新特征每个维度所保留元数据信息量占比，返回[0,1]
print('sum',pca.explained_variance_ratio_.sum()) #即为新数据保留元数据信息占比
print('****************************')
print('com',pca.components_)  # U,S,V = SVD  奇异值分解后的V矩阵

