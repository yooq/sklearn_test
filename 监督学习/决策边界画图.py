import numpy as np
import matplotlib.pyplot  as plt
from sklearn import datasets

iris = datasets.load_iris()
x = iris.data
y = iris.target
x=x[y<2,:2]
y=y[y<2]
plt.scatter(x[y==0,0],x[y==0,1],color='red')
plt.scatter(x[y==1,0],x[y==1,1],color='blue')

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

X_train,X_test,Y_train,Y_test =train_test_split(x,y,test_size=0.3)
reg = LogisticRegression(solver='liblinear')
reg.fit(X_train,Y_train)
# print(reg.coef_) #[[ 1.92995462 -3.1864954 ]]

# 画决策边界(比较规则的决策面)
def func(x):
    return (-reg.intercept_-reg.coef_[0][0]*x)/reg.coef_[0][1]
x1_plt = np.linspace(4,8,100000)
x2_plt = func(x1_plt)
plt.plot(x1_plt,x2_plt)
plt.show()

# 比较不规则的决策面
def func2(model,x):
    x0,x1 = np.meshgrid(
        np.linspace(x[0],x[1],int(x[1]-x[0])*2000),
        np.linspace(x[3],x[2],int(x[3]-x[2])*2000)
    )
    x_new = np.c_[x0.ravel(), x1.ravel()]
    y_predict = model.predict(x_new)
    zz = y_predict.reshape(x0.shape)
    from matplotlib.colors import ListedColormap
    custom_cmap = ListedColormap(['#EF9A9A', '#FFF59D', '#90CAF9'])
    plt.contourf(x0, x1, zz, cmap=custom_cmap)

func2(reg,x=[4,7.5,1.5,4.5])
plt.scatter(X_train[Y_train==0,0],X_train[Y_train==0,1],color='blue')
plt.scatter(X_train[Y_train==1,0],X_train[Y_train==1,1],color='red')

plt.show()


# 多类别不规则决策面，以knn举例
from sklearn.neighbors import KNeighborsClassifier
x_ = iris.data[:,:2]
y_ = iris.target
knn = KNeighborsClassifier(n_neighbors=3, algorithm='ball_tree').fit(x_,y_)
func2(knn,x=[4,8,1.5,4.5])
plt.scatter(x_[y_==0,0],x_[y_==0,1])
plt.scatter(x_[y_==1,0],x_[y_==1,1])
plt.scatter(x_[y_==2,0],x_[y_==2,1])

plt.show()
