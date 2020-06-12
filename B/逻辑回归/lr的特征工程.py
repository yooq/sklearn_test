'''
嵌入法高效

'''

from  sklearn.linear_model import LogisticRegression as LR
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.feature_selection import SelectFromModel

data = load_breast_cancer()

lr = LR()

X = SelectFromModel(lr,norm_order=1).fit_transform(data.data,data.target)  #使用l1范式来筛选，去除l1范式下未能起到作用的特征
score = cross_val_score(lr,X,data.target,cv=5)
lr = lr.fit(X,data.target)
print(score.mean())


