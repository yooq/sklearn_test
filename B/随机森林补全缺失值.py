
'''
作用：用来填补缺失值

'''
import random

from numpy import ravel
from sklearn.datasets import load_boston
from  sklearn.impute import  SimpleImputer  #常用填补缺失值方法
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor


data =load_boston()

df =pd.DataFrame(data.data)
# print(df)


# 编一些缺失值
def drop_value(dataframe):
    index_del = [i for i in range(0, 13)]
    index_del_ = [random.sample(index_del, 1)[0:4] for i in range(0, 4)]
    l=list(set(ravel(index_del_)))
    for i in l:
        dataframe = dataframe.drop(i, axis=0,inplace=False)
    return dataframe

df_miss = df.apply(drop_value,axis=1)
print(df_miss)

'''
使用SimpleImputer 来补全缺失值
'''
imp_mean = SimpleImputer(missing_values=np.nan,strategy='mean')# 实例化，也可以填充常数
# imp_mean = SimpleImputer(missing_values=np.nan,strategy='constant',fill_value=0) 填充常数0
df_ = imp_mean.fit_transform(df_miss) #训练 + 导出
df_ =pd.DataFrame(df_)
# print(df_)
# print(df_.isnull()) #寻找空值


'''
用回顾预测缺失值。
从缺失值最少列开始，其他缺失值暂且用0代替。
将缺失值所在列作为y，确实y的样本作为测试样本，没有确实y的样本作为训练样本。
将原来其他所有列以及lable列放在一起作为特征进行训练。
'''
# print(df_miss)

miss_num = df_miss.isnull().sum()
sort_miss_num = np.argsort(miss_num)  #带索引的排序,返回的是索引
sort_miss_num = sort_miss_num.values
print(sort_miss_num)

df13 = pd.DataFrame(data.target,columns=[13])
# print(df13)

for i in sort_miss_num:
    print(i)
    df_new = df_miss.iloc[:,df_miss.columns!=i]
    df_new[13]=df13
    imp_mean = SimpleImputer(missing_values=np.nan,strategy='constant',fill_value=0)
    df_new = imp_mean.fit_transform(df_new)
    # print(df_new.shape)
    y_lable = df_miss.iloc[:,df_miss.columns==i]
    # print(y_lable)
    y_lable_trian = y_lable[y_lable[i].notnull()]

    y_lable_test = y_lable[y_lable[i].isnull()]
    # print(y_lable_test)
    X_train = df_new[y_lable_trian.index]
    print(X_train.shape)
    X_test = df_new[y_lable_test.index]
    print(X_test.shape)

    model =RandomForestRegressor(random_state=2,n_estimators=10)
    model.fit(X_train,y_lable_trian)
    y_lable_test_ = model.predict(X_test)
    print('m',df_miss.iloc[:,i].isnull())
    df_miss.loc[df_miss.iloc[:,i].isnull(),i]=y_lable_test_    #布尔索引用loc



print(df_miss)
