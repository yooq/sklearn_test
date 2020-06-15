from sklearn.cluster import KMeans
from sklearn.datasets import load_wine
from sklearn.metrics import accuracy_score,silhouette_samples,silhouette_score,calinski_harabaz_score

data = load_wine()
X = data.data
print(X.shape)
'''初始化质心'''



#tol表示 两次inertia变化差值，小于这个差值就停止迭代
cluster = KMeans(n_clusters=3,init='k-means++',random_state=0,max_iter=100,tol=1e-4).fit(X)  #init默认是kmeans++,迭代更快，迭代次数少

y_pre = cluster.labels_  #查看各个样本标签
y_pre = cluster.fit_predict(X) #与上面是一样的
#、、区别
# 第一种所有数据都要聚类才能得到每个样本的标签
# 第二种可以先用部分数据聚类，在把剩下数据分到类簇里面去，计算消耗少

center = cluster.cluster_centers_
print(center)

inertia = cluster.inertia_ #聚类结果中所有样本到自己簇心距离的平方和。。。并不一定是模型的评估指标

'''如何评价一个聚类好不好,两种情况'''
    #数据带有一定标签

    # 数据不带一点点标签--   轮廓系数
        # --轮廓系数：簇内稠密程度（样本与所在簇中其他样本点距离的均值 a），簇间稀疏程度(样本所在簇与最近的一个簇，样本与该簇中所有点的距离均值)
        # 轮廓系数：s=（b-a）/max(a,b)...max(a,b)表示a，b中最大值。。。。a>b表示效果不好
        # 故s取值（-1,1）. 大于0效果就不错

score = silhouette_score(X,y_pre)
print(score)
s_exam = silhouette_samples(X,y_pre)
print(s_exam)


   # 数据不带一点点标签--   卡林斯基
        #矩阵数据之间离散度越高，那么迹就越大。 迹：nxn矩阵的对角元素之和，或者特征值之和
        #卡林斯基系数为  [（组间迹）/(组内迹)] *[(N-K)/(k-1)]  N为总数据数，k为类别数。
        #系数无范围，系数越大越好

ca_score = calinski_harabaz_score(X,y_pre)
print(ca_score)
