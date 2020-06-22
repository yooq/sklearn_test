# -*- coding: utf-8 -*-
import  xgboost as xgb
from xgboost import XGBRegressor as XGBR
from xgboost import XGBClassifier as XGBC
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split,KFold,cross_val_score,learning_curve
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.linear_model import LinearRegression as LR
from sklearn.metrics import mean_squared_error  as mse,SCORERS
import matplotlib.pyplot as plt


data = load_boston()

X = data.data
Y = data.target

X_trian ,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.3)

'''
objective  
            表示损失函数类型。共四种
                        ：binary:logistic 对数损失，二分类时使用
                        ：reg:linear 均方误差（回归时使用）
                        ：binary：hinge 支持向量机的损失函数，二分类用
                        ：multi.softmax 多分类使用
            
回归用效果好
    reg_alpha   L1的系数[0,]
    reg_lambda  L2 的系数[0,]
            
'''

sk_xgb_model =XGBR(
            n_estimators=20
            ,random_state=420
            ,booster='gblinear'
            ,objective='reg:linear'
            ,reg_lambda=0.3
            ,gamma=0.4    #阻止 树继续生长从而导致过拟合，树的结构之差大于gamma则生长 [0,]
            # ,max_depth=

            )
for i in range(5,10):
    train_size,train_score,test_score = learning_curve(sk_xgb_model,X,Y,cv=5)  #返回训练尺寸，训练分数，测试分数

    plt.scatter(train_size,[i.mean() for i in train_score],marker='s')
    plt.plot(train_size,[i.mean() for i in train_score],label='true')

    plt.scatter(train_size,[i.mean() for i in test_score],marker='s')
    plt.plot(train_size,[i.mean() for i in test_score],label='pre')
    plt.title('cv = {}'.format(i))
    plt.legend()
    plt.show()



'''泛化误差  =  bias**2 + var +噪音（可以不考虑）      使得最小'''

# bias = []
# vars = []
# ge =[]
#
# for  i  in range(10,200,20):
#     reslut = cross_val_score(XGBR(n_estimators=i,random_state=420),X,Y,cv=10)
#
#     bias.append(reslut.mean())
#     vars.append(reslut.var())
#
#     ge.append((1-reslut.mean()**2 +reslut.var()))
#
# index = ge.index(max(ge))
#
# x = [i for i in range(10,200,20)]
# plt.figure()
# plt.scatter(x[index],max(ge),marker='s')
#
# plt.plot(x,ge,label = 'n_estimators')
#
# plt.vlines(x[index],0,max(ge), colors="c", linestyles="dashed")
# plt.hlines(max(ge),0,x[index],colors="c", linestyles="dashed")
#
# plt.annotate('({} , {})'.format(x[index],round(max(ge),3)),
#              (x[index],max(ge)-0.2)
#              )
# plt.xlim((0,200))
# plt.ylim((0,1.2))
#
# plt.legend()
# plt.show()


'''
subsample   （0,1]  表示抽取数据的比例

# 有放回的抽取，能够有效的减轻过拟合
'''



'''
迭代决策树。

learning_rate  [0,1]
'''

'''
弱评估器

booster 可选模型 gbliner(线性模型有奇效),gbtree,dart
'''

'''
scale_pos_weight= 负样本/正样本 ,float
示正样本所占比例,默认是1

'''
