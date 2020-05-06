from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets
X_0,y_0=datasets.load_iris(return_X_y=True)
X=X_0[y_0<2,2:]
y=y_0[y_0<2]

# 划分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

# 使用 sklearn 的 LogisticRegression 作为模型，其中有 penalty，solver，dual 几个比较重要的参数，不同的参数有不同的准确率，这里为了简便都使用默认的，详细的请参考 sklearn 文档
model = LogisticRegression(solver='liblinear')

# 拟合
model.fit(X, y)

# 预测测试集
predictions = model.predict(X_test)

# 打印准确率
print('测试集准确率：', accuracy_score(y_test, predictions))


x_ = np.linspace(0, 7, 10000)
y_ = np.linspace(0, 7, 10000)

x_,y_ = np.meshgrid(x_,y_)
# custom_cmap = ListedColormap(['#EF9A9A', '#FFF59D', '#90CAF9'])
y_predict = model.predict(np.c_[x_.ravel(),y_.ravel()])
zz = y_predict.reshape(x_.shape)
plt.contourf(x_, y_,zz,camp=plt.cm.coolwarm,s=20,edgecolors='k')

plt.scatter(X[y==1,0],X[y==1,1],color='red')
plt.scatter(X[y==0,0],X[y==0,1],color='blue')
plt.show()


x_p=np.linspace(0, 7, 10000)
plt.scatter(X[y==1,0],X[y==1,1],color='red')
plt.scatter(X[y==0,0],X[y==0,1],color='blue')
plt.plot(x_p,(-model.coef_[0][0]*x_p-model.intercept_)/model.coef_[0][1],'green')
plt.show()
