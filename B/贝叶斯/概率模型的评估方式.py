'''
不说准确度。

布里尔分数  =  1/n *西格玛（p-o）平方
            就是真实{0,1}与预测之间[0,1]的差的平方均值,越小越好
            from  sklearn.metrics import brier_score_loss as bsl



对数似然函数 （常用）  表损失，故越小越好
             from sklearn.metrics import log_loss

'''

from  sklearn.metrics import brier_score_loss as bsl


bsl(Y_test,pre,pos_label=1)

# *******************************************

from sklearn.metrics import log_loss

log_loss(y_true=,y_pred=,)


'''
可靠性曲线

    分箱：
        每个箱子中真实的类所占比例，为该箱子真实概率 ，为纵坐标
        每个箱子预测概率的均值为这个箱子的预测概率，为横坐标
        from sklearn.calibration import  calibration_curve

'''
from sklearn.calibration import  calibration_curve
y,x = calibration_curve(y_true=,y_prob=,n_bins=箱子数目,normalize=是否归一化)  #返回横纵坐标



'''
预测概率直方图

ax.hist() 设置参数，直方图会自己分箱

'''


'''
概率校准
'''
from sklearn.calibration import CalibratedClassifierCV

model = CalibratedClassifierCV(base_estimator=基础模型,method=校准方式只有两种,cv=)
model.fit()
model.predict()
model.get_params()

