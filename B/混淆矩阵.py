'''
混淆矩阵
                          预测值
                  1                  0

        1        11（TP）         10(FN)
真实值
        0        01(FP)           00(TN)


acc = (11+00)/all
pre = (11)/(11+01)          ----查准率，精确度，在所有被预测为正类中，有多少事真正类
recall = (11)/(11+10)       -----召回率，所有真正类中，被预测为正类的有多少

f1 = 2/[1/(p)+1/(recall)] = 2*p*r/(p+r)    [0,1]接近1模型最好

Specificity = TN/(FP+TN)   ---特异度 ，真正0类中，有多少预测成0类，可理解为召回率召回0类

FPR = 1-Specificity        ----假正率


'''
from sklearn.metrics import confusion_matrix,accuracy_score,precision_score,f1_score,recall_score
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import matplotlib.pyplot as plt

data =load_iris()

X_train,X_test,Y_train,Y_test = train_test_split(data.data,data.target,test_size=0.3)
svm_model =SVC(kernel='rbf',random_state=0,probability=True)
svm_model.fit(X_train,Y_train)
y_pre = svm_model.predict(X_test)

y_pre_ = svm_model.predict_proba(X_test)    #返回样本属于每个类别的概率，必须在训练模型时，设置 probability=True

y_pre__ = svm_model.predict_log_proba(X_test)
print(y_pre_)
# print(y_pre__)
#
svm_model.decision_function(X_test)  #返回的是样本到决策边界的距离

cfm = confusion_matrix(y_pred=y_pre,y_true=Y_test)
# print(cfm)
plt.matshow(cfm, cmap=plt.cm.gray)
plt.show()



# 'micro':通过先计算总体的TP，FN和FP的数量，再计算F1
# 'macro':分布计算每个类别的F1，然后做平均（各类别F1的权重相同）
# ‘weighted’，按加权（每个标签的真实实例数）平均，这可以解决标签不平衡问题，可能导致f1分数不在precision于recall之间。

acc = accuracy_score(y_true=Y_test,y_pred=y_pre)
pre = precision_score(y_pred=y_pre,y_true=Y_test,average='weighted')
recall = recall_score(y_pred=y_pre,y_true=Y_test,average='macro')
f1 = f1_score(y_pred=y_pre,y_true=Y_test,average='macro')

print('acc: ',acc,'pre: ',pre,'f1: ',f1)




