from sklearn.neural_network import MLPClassifier

# 分类

X = [[0., 0.], [1., 1.]]
y = [0, 1]
clf = MLPClassifier(solver='lbfgs', alpha=1e-5,
                 hidden_layer_sizes=(5, 2), random_state=1,warm_start=True)
# 如果warm_start=True就表示就是在模型训练的过程中，在前一阶段的训练结果上继续训练

model = clf.fit(X, y)
print(model) #模型

pre = clf.predict([[2., 2.], [-1., -2.]]) #预测
print(pre)
print(model.coefs_) #参数
print(clf.predict_proba([[2., 2.], [1., 2.]])) #预测
print(clf.predict([[1., 2.],[0., 0.]]))


# 回归

from sklearn.neural_network import MLPRegressor
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
X, y = make_regression(n_samples=200, random_state=1)
X_train, X_test, y_train, y_test = train_test_split(X, y,random_state=1)
regr = MLPRegressor(random_state=1, max_iter=500).fit(X_train, y_train)
print('回归')
print('11',X_test[:2][1])
print('22',regr.predict([X_test[:2][1]])) #一个值输出
print('33',regr.predict(X_test[:2])) #多个值（2）输出
print('44',regr.score(X_test, y_test))



