import pydotplus as pydotplus
from  sklearn import  tree
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split

if __name__ == '__main__':
    data = load_wine()

    X_train,X_test,Y_train,Y_test = train_test_split(data.data,data.target,test_size=0.3)


    model = tree.DecisionTreeClassifier(
                                    criterion='entropy'
                                    ,random_state=30
                                    ,splitter='random'
                                     # 以下是剪枝
                                    ,max_depth=4   #最大深度，对于高纬度，样本量少，有奇效
                                    ,min_samples_leaf=10 #每个节点至少拥有的样本数
                                    ,min_samples_split=20 #每个非叶子结点至少拥有样本数
                                    ,min_impurity_split=0.2
                                    ,min_impurity_decrease=0.2#这两个参数是一样的，都是用来限制信息熵的变化，如果信息熵改变的值小于这个值，就不再生长
                                    )

    model.fit(X_train,Y_train)
    score= model.score(X_test,Y_test)
    score_ = model.score(X_train,Y_train)
    print(score)
    print(score_)

    data_log = tree.export_graphviz(model,feature_names=data.feature_names,class_names=['0','1','2'],filled=True,rounded=True)
    # 决策树可视化
    graph = pydotplus.graph_from_dot_data(data_log)

    ###保存图像到pdf文件
    graph.write_pdf("treetwo_cut22.pdf")


    # 使用了的特征，使用了特征对应值不为0，没使用的特征为0
    print([*zip(data.feature_names,model.feature_importances_)])



