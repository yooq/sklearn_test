'''
ROC曲线  是 不同阈值下   假正率为横坐标，召回率为纵坐标
        假正率 = 01/（00+01）,即在0类下，预测不正确的0除以所有0类
        阈值  启用probability=True 参数，每个样本属于不同类别的概率，当概率大于阈值时属于那一类
        from sklearn.metrics import roc_curve

AUC面积：面积越大模型效果越好
       from sklearn.metrics import roc_auc_score

'''
from sklearn.metrics import confusion_matrix,accuracy_score,precision_score,f1_score,roc_curve,roc_auc_score
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import matplotlib.pyplot as plt

data =load_iris()

X_train,X_test,Y_train,Y_test = train_test_split(data.data,data.target,test_size=0.3)
svm_model =SVC(kernel='rbf',random_state=0,probability=True)
svm_model.fit(X_train,Y_train)
y_pre = svm_model.predict(X_test)
y_pre_ = svm_model.predict_proba(X_test)

FPR,RECALL,thresholds =  roc_curve(y_true=Y_test,y_score=y_pre_[:,1],pos_label=1)
# y_score=y_pre_[:,1]   也可以用svm_model.decision_function(X_test)作为参数，根据距离决策边界来分类

auc = roc_auc_score(Y_test,y_score=y_pre_,multi_class='ovo',labels=[0,1,2],max_fpr=1.0)  #这里是多分类，用multi_class参数
print(auc)

import  matplotlib.pyplot as plt

plt.plot(FPR,RECALL,color='red',label = 'auc = %0.2f'%auc)
plt.plot([0,1],[0,1],color ='black',linestyle = '--')
plt.xlabel('FPR')
plt.ylabel('recall')
plt.legend(loc='lower right')

plt.show()


'''
寻找合理的阈值

recall - FPR  差值的最大值，此时阈值即为最佳

'''
