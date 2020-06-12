import random
from pprint import pprint

from numpy import ravel
from sklearn.datasets import load_boston
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split,cross_val_score

if __name__ == '__main__':
    data =load_boston()

    model =RandomForestRegressor(n_estimators=10,random_state=0)
    cross = cross_val_score(model,data.data,data.target,cv=10,scoring='neg_mean_squared_error')
    import sklearn
    # print(cross)
    # 模型指标评估列表
    k = sorted(sklearn.metrics.SCORERS.keys())
    # pprint(k)

