'''在训练之前平衡您的数据集，以防止决策树偏向于主导类.可以通过从每个类中抽取相等数量的样本来进行类平衡'''
from sklearn import tree
from sklearn.datasets import load_iris
from sklearn import tree
X = [[0, 0], [1, 1]]
Y = [0, 1]
clf = tree.DecisionTreeClassifier()
clf = clf.fit(X, Y)
print(clf.predict_proba([[2., 2.]])) # [[0. 1.]]
print(clf.predict([[2., 2.]])) # [1]

# 多分类
iris = load_iris()
clf_mul = tree.DecisionTreeClassifier()
clf_mul = clf_mul.fit(iris.data, iris.target)
print(clf_mul.predict([[5.7, 4.4, 1.5, 0.4]]))
print(clf_mul.predict_proba([[5.7, 4.4, 1.5, 0.4]])) #[[1. 0. 0.]]

'''回归'''
from sklearn import tree
X = [[0, 0], [2, 2]]
y = [0.5, 2.5]
clf = tree.DecisionTreeRegressor()
clf = clf.fit(X, y)
pre = clf.predict([[0, 1]])
print(pre)
clf.export()

