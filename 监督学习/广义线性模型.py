from sklearn import linear_model
import numpy as np

# 基于普通最小二乘法的线性模型 复杂度(n_samples,n_features)-->O(n_sample*n_features*n_features)
# 必须满秩，趋向于奇异矩阵，可能产生大的方差

train_x = np.random.normal(size=(10000,2))
train_y = np.random.normal(size=(10000,1))
reg = linear_model.LinearRegression()
reg.fit(train_x,train_y) #train
test_x=train_x[-200:]
pre_y = reg.predict(test_x) #predict
print('ALS: ')
print('W: ',reg.coef_) #W
print('bias: ',reg.intercept_) #bias

# 岭回归  岭系数最小化的是 带罚项 的残差平方和 复杂度和ALS一样
reg_lin = linear_model.Ridge(alpha=0.3)  #Linear least squares with l2 regularization.
reg_lin.fit(train_x,train_y)
print('岭回归： ')
print('W:',reg_lin.coef_)
print('bias',reg_lin.intercept_)

# 交叉验证
reg_cv = linear_model.RidgeCV(alphas=[0.1,1.0,10.0],cv=None) #十折交叉验证,留一
reg_cv.fit(train_x,train_y)
print('cv')
print('w: ',reg_cv.coef_)
print('bias: ',reg_cv.intercept_)

# lasso

reg_lasso = linear_model.Lasso(alpha = 0.5)
reg_lasso.fit(train_x,train_y)
print('w: ',reg_lasso.coef_)
print('b: ',reg_lasso.intercept_)

reg_lasso_cv = linear_model.LassoCV(alphas=[0.1,1.0,10.0],cv=None).fit(train_x,train_y) #作用于高维数据集
print('w: ',reg_lasso_cv.coef_)

reg_lasso_cv_lar = linear_model.LassoLarsCV(cv=10).fit(train_x,train_y) #作用于低维数据集
print('w: ',reg_lasso_cv_lar.coef_)
