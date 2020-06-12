from sklearn.datasets import load_boston
import pydotplus as pydotplus
from  sklearn import  tree
from sklearn.model_selection import train_test_split

if __name__ == '__main__':
    data = load_boston()
    # print(data)
    X_train,X_test,Y_train,Y_test = train_test_split(data.data,data.target,test_size=0.3)

    model = tree.DecisionTreeRegressor(
        criterion='mse'
        , random_state=30
        , splitter='random'
    )
    model.fit(X_train,Y_train)
    score= model.score(X_test,Y_test)
    score_ = model.score(X_train,Y_train)
    print(score)
    print(score_)


    data_log = tree.export_graphviz(model,feature_names=data.feature_names,filled=True,rounded=True)
    # 决策树可视化
    graph = pydotplus.graph_from_dot_data(data_log)

    ###保存图像到pdf文件
    graph.write_pdf("2222.pdf")
