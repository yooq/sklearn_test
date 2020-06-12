import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
import pydotplus as pydotplus
from  sklearn import  tree


df = pd.read_csv('mytrain.csv')
print(df.info())


df.drop(['Cabin','Name','Ticket'],inplace=True,axis=1)
print(df.head(5))
print(df['Age'])

# 补充缺失值
df['Age']=df['Age'].fillna(df['Age'].mean())
print(df.info())

# 删除有缺失值的行
df = df.dropna()
print(df.info())

#查看列里不同值,并处理
print(df['Embarked'].unique())
lable = df['Embarked'].unique().tolist()
df['Embarked'] = df['Embarked'].apply(lambda x:lable.index(x))

lable_sex = df['Sex'].unique().tolist()
df['Sex'] = df['Sex'].apply(lambda x:lable_sex.index(x))


X= df.iloc[:,df.columns !='Survived']
Y= df.iloc[:,df.columns =='Survived']


X_train ,X_test, Y_train,Y_test = train_test_split(X,Y,test_size=0.3)

# 重新索引
X_train.index = range(X_train.shape[0])

for  i in [X_train,X_test,Y_train,Y_test]:
    i.index =range(i.shape[0])


# prelist = []
# trainlist = []

# for i in range(0,10):
#     model = tree.DecisionTreeClassifier(random_state=25
#                                         ,max_depth=i+1
#                                         , min_samples_leaf=10 #每个节点至少拥有的样本数
#                                         ,min_samples_split=20
#                                         )
#     # model.fit(X_train,Y_train)
#     score = model.score(X_test, Y_test)
#
#     score = cross_val_score(model,X,Y,cv=10).mean()
#
#     score_ = model.score(X_train,Y_train)
#
#     # 决策树可视化
#     data_log = tree.export_graphviz(model, feature_names=X.columns, class_names=['0', '1'], filled=True,
#                                     rounded=True)
#     graph = pydotplus.graph_from_dot_data(data_log)
#
#     prelist.append(score)
#     trainlist.append(score_)
#
#     ###保存图像到pdf文件
#     graph.write_pdf('taitannike'+str(i+1)+'.pdf')
#
#
# import matplotlib.pyplot as plt
#
# plt.plot([i+1 for i in range(10)],prelist,color='red')
# plt.plot([i+1 for i in range(10)],trainlist,color='blue')
# plt.legend()
# plt.show()




# 网格搜索

param ={
'criterion':('gini','entropy')
,'max_depth':[i for i in range(2)]
,'splitter':('random','best')
,'min_samples_split':[i for i in range(9,11,)]
,'min_samples_leaf':[i for i in range(4,6)]
}
model1 = tree.DecisionTreeClassifier(random_state=25)
GS = GridSearchCV(model1,param,cv=3)
GS = GS.fit(X_train,Y_train)

print(GS.score(X_test,Y_test))
print(GS.best_params_)
print(GS.best_score_)
