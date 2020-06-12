from sklearn import cluster
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split

'''
训练数据上的标签可以在 labels_ 

'''

data = load_wine()

X_trian ,X_test,Y_train,Y_test = train_test_split(data.data,data.target,test_size=0.3)

model = cluster.KMeans(n_clusters=3)
model.fit_transform(X_trian,Y_train)
score = model.score(X_test[0:2],Y_test[0:2])
print(score)
print(model.labels_)

y_pre = model.predict(X_test[0:2])


# from sklearn.metrics import  recall_score
# from sklearn.metrics import precision_score
# from sklearn.metrics import accuracy_score
# from sklearn.metrics import f1_score
#
#
# acc =  accuracy_score(Y_test,y_pre)
# print('acc',acc)
#
# preacc = precision_score(Y_test,y_pre,average='micro')
# print('preacc',preacc)
#
# recall = recall_score(Y_test,y_pre,average='micro')
# print('recall',recall)
#
# f1 = f1_score(Y_test,y_pre,average='micro')
# print('f1',f1)
