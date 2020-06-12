from sklearn.datasets import load_boston
import pydotplus as pydotplus
from  sklearn import  tree
from sklearn.model_selection import train_test_split,cross_val_score

if __name__ == '__main__':
    data = load_boston()


    model = tree.DecisionTreeRegressor(
        criterion='mse'
        , random_state=30
        , splitter='random'
    )

    cv = cross_val_score(model,data.data,data.target,cv=5,scoring='neg_mean_squared_error')  #scoring 默认R^2
    print(cv)

