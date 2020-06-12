from sklearn.metrics import  recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
import numpy as  np



y_pred = [0, 2, 1, 3]
y_true = [0, 1, 2, 3]
print(accuracy_score(y_true, y_pred))  # 0.5
print(accuracy_score(y_true, y_pred, normalize=False))  # 2

# 在具有二元标签指示符的多标签分类案例中
print(accuracy_score(np.array([[0, 1], [1, 1]]), np.ones((2, 2))))  # 0.5




y_true = [0, 1, 2, 0, 1, 2]
y_pred = [0, 2, 1, 0, 0, 1]
print(precision_score(y_true, y_pred, average='macro'))  # 0.2222222222222222
print(precision_score(y_true, y_pred, average='micro'))  # 0.3333333333333333
print(precision_score(y_true, y_pred, average='weighted'))  # 0.2222222222222222
print(precision_score(y_true, y_pred, average=None))  # [0.66666667 0.         0.        ]





y_true = [0, 1, 2, 0, 1, 2]
y_pred = [0, 2, 1, 0, 0, 1]
print(recall_score(y_true, y_pred, average='macro'))  # 0.3333333333333333
print(recall_score(y_true, y_pred, average='micro'))  # 0.3333333333333333
print(recall_score(y_true, y_pred, average='weighted'))  # 0.3333333333333333
print(recall_score(y_true, y_pred, average=None))  # [1. 0. 0.]




y_true = [0, 1, 2, 0, 1, 2]
y_pred = [0, 2, 1, 0, 0, 1]
print(f1_score(y_true, y_pred, average='macro'))  # 0.26666666666666666
print(f1_score(y_true, y_pred, average='micro'))  # 0.3333333333333333
print(f1_score(y_true, y_pred, average='weighted'))  # 0.26666666666666666
print(f1_score(y_true, y_pred, average=None))  # [0.8 0.  0. ]
