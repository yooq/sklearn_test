import pydotplus as pydotplus
from sklearn import tree
from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.tree import DecisionTreeClassifier


if __name__ == '__main__':
    data = load_wine()


    X_train,X_test,Y_train,Y_test = train_test_split(data.data,data.target,test_size=0.3)



    model_rand = RandomForestClassifier(n_estimators=5,random_state=2,oob_score=True)  #oob  (out of bag)是指，在构建随机森林的时候，其实只有百分之62左右的数据被拿去训练了，剩下没有去训练的数据就是袋外数据，可以直接作为测试集

    # 如果使用袋外数据集，那么，训练的时候就不需要切分数据及
    model_rand.fit(data.data,data.target)
    print(model_rand.oob_score_)

    print(model_rand.feature_importances_)

    # model_rand.fit(X_train,Y_train)

    # 查看所有的树
    for i in range(len(model_rand.estimators_)):
        print(model_rand.estimators_[i])




    # score = model_rand.score(X_test,Y_test)
    # print(score)
    #


'''
回归
'''
