from sklearn import datasets
iris = datasets.load_iris()
from sklearn.naive_bayes import GaussianNB,MultinomialNB,ComplementNB,BernoulliNB

gnb = GaussianNB() #高斯贝朴素叶斯---特征的可能性(即概率)假设为高斯分布
mnb = MultinomialNB() #多项式朴素贝叶斯--服从多项分布数据的朴素贝叶斯算法，也是用于文本分类经典算法；多项式分布是二项式分布的扩展，不同的是多项式分布中，每次实验有n种结果。
cnb = ComplementNB() #补充贝朴素叶斯----特别适用于不平衡数据集
bnb = BernoulliNB() #伯努利朴素贝叶斯-----有多个特征，但每个特征 都假设是一个二元 (Bernoulli, boolean) 变量  0-1分布
y_pred_gnb = gnb.fit(iris.data, iris.target).predict(iris.data)
y_pred_mnb = mnb.fit(iris.data, iris.target).predict(iris.data)
y_pred_cnb = mnb.fit(iris.data, iris.target).predict(iris.data)
y_pred_bnb= mnb.fit(iris.data, iris.target).predict(iris.data)


print("Gauss---Number of mislabeled points out of a total %d points : %d" % (iris.data.shape[0],(iris.target != y_pred_gnb).sum()))
print("Multi---Number of mislabeled points out of a total %d points : %d" % (iris.data.shape[0],(iris.target != y_pred_mnb).sum()))
print("Com---Number of mislabeled points out of a total %d points : %d" % (iris.data.shape[0],(iris.target != y_pred_cnb).sum()))
print("Ber---Number of mislabeled points out of a total %d points : %d" % (iris.data.shape[0],(iris.target != y_pred_bnb).sum()))
