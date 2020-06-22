
'''

读取数据             设置参数          训练模型                  预测结果
xgb.DMatrix()       param={}        xgb.train(param)          xgb.predict()

'''
import  xgboost as xgb
from xgboost import XGBRegressor as XGBR
from xgboost import XGBClassifier as XGBC
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split,KFold,cross_val_score
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.linear_model import LinearRegression as LR
from sklearn.metrics import mean_squared_error  as mse,SCORERS


data = load_boston()

X = data.data
Y = data.target

X_trian ,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.3)

sk_xgb_model =XGBR(n_estimators=100,random_state=0).fit(X_trian,Y_train)

pre1 = sk_xgb_model.predict(X_test)
score1 = sk_xgb_model.score(X_test,Y_test)
mse = mse(y_true=Y_test,y_pred=pre1)
important = sk_xgb_model.feature_importances_

print('pre:  ',pre1)
print('score1:  ',score1)
print('mse:  ',mse)
print('important:  ',important)
print('mean:   ',Y.mean())

print(SCORERS.keys())   #所有可用的评估指标
