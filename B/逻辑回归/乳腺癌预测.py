from sklearn.linear_model import LogisticRegression as LR
from sklearn.datasets import  load_breast_cancer
from sklearn.metrics import  accuracy_score
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
import numpy as np


if __name__ == '__main__':
    data = load_breast_cancer()
    X = data.data
    Y = data.target

    X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.3)


    for i in ['l1', 'l2']:
        acc_list = []
        acc_list_ = []
        for c in np.linspace(0.05,1.0,9):
            # 可选l2和l1，，，l2使得参数不会为0，使得每个特征都发挥作用。。l1使部分特征为零
            #  liblinear  只支持二分类
            model_lr = LR(penalty=i,C=c,solver='liblinear',class_weight='balance')
            model_lr.fit(X_train,Y_train)

            score = model_lr.score(X_test,Y_test)

            pre_y = model_lr.predict(X_test)

            acc = accuracy_score(y_true=Y_test,y_pred=pre_y)
            acc_ =accuracy_score(y_true=Y_train,y_pred=model_lr.predict(X_train))
            acc_list.append(acc)
            acc_list_.append(acc_)
            # print('acc: ',acc)

        plt.figure()

        plt.plot(np.linspace(0.05,1.0,9),acc_list,'red',label='test')
        plt.plot(np.linspace(0.05,1.0,9),acc_list_,'blue',label='train')
        plt.title(i)
    plt.show()
