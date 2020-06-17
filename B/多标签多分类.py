from sklearn.datasets import make_classification

from sklearn.multioutput import MultiOutputClassifier,MultiOutputRegressor

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from sklearn.utils import shuffle

import numpy as np

X, y1 = make_classification(n_samples=100, n_features=10, n_informative=3, n_classes=3, random_state=1)


y2 = shuffle(y1, random_state=1)

y3 = shuffle(y1, random_state=2)

Y = np.vstack((y1, y2, y3)).T

n_samples, n_features = X.shape

n_outputs = Y.shape[1] # 3

n_classes = 3

forest = SVC(kernel='rbf',random_state=0)

# classif = OneVsRestClassifier(SVC(kernel='linear'))
multi_target_forest = MultiOutputClassifier(forest, n_jobs=-1)

y_pre = multi_target_forest.fit(X, Y).predict(X)
# print(y_pre)
