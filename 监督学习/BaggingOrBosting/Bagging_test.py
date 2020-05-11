from sklearn.svm import SVC
from sklearn.ensemble import BaggingClassifier, AdaBoostRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import make_classification
from sklearn import tree

# bagging
X, y = make_classification(n_samples=100, n_features=4,
                           n_informative=4, n_redundant=0,
                           random_state=0, n_classes=4,shuffle=False)
clf_svm = BaggingClassifier(base_estimator=KNeighborsClassifier(),
                        n_estimators=10, random_state=0).fit(X, y)

clf_knn = BaggingClassifier(base_estimator=KNeighborsClassifier(),
                        n_estimators=10, random_state=0).fit(X, y)

clf_tr = BaggingClassifier(base_estimator=tree.DecisionTreeClassifier(),
                        n_estimators=10, random_state=0).fit(X, y)

pre = clf_svm.predict_proba([[0, 0, 0, 0]])
pre_knn = clf_knn.predict_proba([[0, 0, 0, 0]])
pre_knn_ = clf_knn.predict([[0, 0, 0, 0]])

pre_tr = clf_knn.predict_proba([[0, 0, 0, 0]])
print(pre)
print(pre_knn)
print(pre_knn_)
print(pre_tr)


# 决策树，随机森林，极限随机森林
from sklearn.model_selection import cross_val_score
from sklearn.datasets import make_blobs
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

X, y = make_blobs(n_samples=10000, n_features=10, centers=100,
    random_state=0)

clf = DecisionTreeClassifier(max_depth=None, min_samples_split=2,
    random_state=0)
scores = cross_val_score(clf, X, y, cv=5)
scores.mean()

clf = RandomForestClassifier(n_estimators=10, max_depth=None,
   min_samples_split=2, random_state=0)
scores = cross_val_score(clf, X, y, cv=5)
scores.mean()

clf = ExtraTreesClassifier(n_estimators=10, max_depth=None,
    min_samples_split=2, random_state=0)
scores = cross_val_score(clf, X, y, cv=5)
scores.mean()

# adboost
from sklearn.model_selection import cross_val_score
from sklearn.datasets import load_iris
from sklearn.ensemble import AdaBoostClassifier

iris = load_iris()

clf = AdaBoostClassifier(n_estimators=100).fit(iris.data,iris.target)
scores = cross_val_score(clf, iris.data, iris.target,cv=3)
print(scores.mean())

print(clf.predict_proba([[4.9,2.4,3.3,1. ]]))
print(clf.predict([[4.9,2.4,3.3,1. ]]))

# 使用 AdaBoost.R2 算法证明了回归。
# Fit regression model
regr_1 = DecisionTreeRegressor(max_depth=4)
regr_2 = AdaBoostRegressor(DecisionTreeRegressor(max_depth=4),
                          n_estimators=300)

# GBDT
# GradientBoostingClassifier和GradientBoostingRegressor
from sklearn.datasets import make_hastie_10_2
from sklearn.ensemble import GradientBoostingClassifier

X, y = make_hastie_10_2(random_state=0)
X_train, X_test = X[:2000], X[2000:]
y_train, y_test = y[:2000], y[2000:]
clf = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0,
    max_depth=1, random_state=0).fit(X_train, y_train)
clf.score(X_test, y_test)
