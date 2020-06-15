'''
特征工程------特征提取，特征创造，特征选择

'''


'''
特征选择目标：不损害算法效果的前提下，降低模型训练时间。
特征选择的方法：
        过滤法：
                方差过滤法-----消除方差很小的特征，对模型学习数据分布，区分样本的帮助不大
        卡方检验：
                常用于分类，离散标签
                非负特征与标签之间的卡方统计量，越高说明相关性越高。。。如果存在负值特征，可以现做数据平移归一等。先处理
        F检验
        互信息
'''
# 过滤法
from sklearn.feature_selection import  VarianceThreshold
select = VarianceThreshold( threshold=1) #默认 threshold=0. 过滤掉方差小于 threshold 的特征，如果想留下一半特征可以领 threshold等于方差中位数
p=0.8
select_ =  VarianceThreshold(p*(1-p)) #若特征是伯努利随机变量,若某一个值占特征的p(这里假设是0.8)，则删除该特征。



# 卡方检验
from sklearn.feature_selection import SelectKBest,chi2

x_ =SelectKBest(chi2,k=300).fit_transform(x,y)  #chi2 卡方检验方法，也可以填F检验，选择300个特征,这样不能保证效果

# 最好是选取显著性小于0.05的特征，及p值小于0.05 说明两组数据是相关的

k,p = chi2(x,y)  #返回卡方值和p值, 统计看看p<0.05的有多少个特征，然后作为k的参数。



# F检验
#F检验的本质是两组数据的线性关系，假设’数据不存在显著的线性关系‘，返回F值，P值，P<=0.05表示假设不成立，两组数据存在显著的线性相关。
# 可用于分类数据和回归数据的处理，有不同的函数
# F检验在服从正太分布的时候效果比较好
from sklearn.feature_selection import f_classif,f_regression

F,p = f_classif(x,y) #处理离散标签数据，用作分类。

F,p = f_regression(x,y) #处理连续型数据，用作回归。

# 往往都是先根据P值，然后保留与y相关性高的特征,共有多少个，以便确定k值
select = SelectKBest(f_classif,k=??)


# 互信息
# 用来捕捉标签与特征之间的任意关系，可用作回归与分类,返回一个相关值[0,1]，0表示不相关，1表示完全相关
from sklearn.feature_selection import mutual_info_classif,mutual_info_regression

fe = mutual_info_classif(x,y) #得到一组相关性的值，统计一下大于0的有多少，即可确定k值
select = SelectKBest(mutual_info_classif,K=？？？)



#           嵌入法

# 对       逻辑回归            效过不错
# 原理：模型反复的训练和调整。
# 基础算法模型训练后，可以返回各个特征的重要程度，过滤掉小于阈值重要性的特征，重新训练模型，评估效果
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier as RCF
from sklearn.model_selection import cross_val_score
import  numpy as np

RCF_ = RCF(n_estimators=10,random_state=2) #实例化一个算法模型
x_features = SelectFromModel(RCF_,threshold=0.05).fit_transform(x,y) #返回的是保留下的特征，这里threshold是一个超参数
#------------我分割--------
RCF_.fit(x,y)
importance = RCF_.feature_importances_ #返回各个特征在模型中的作用

k = np.linspace(0,max(importance),20) #选取20个阈值，分别作为threshold,来进行模型训练
for i in k:
    x_features = SelectFromModel(RCF_, threshold=i).fit_transform(x, y)  # 返回的是保留下的特征，这里threshold是一个超参数

    cros = cross_val_score(RCF_,x_features,y,cv=5)  #评估模型



# 包装法，
# 综合了统计法和嵌入法 ，对   支持向量机    有奇效
from sklearn.feature_selection import RFE
RCF_ = RCF(n_estimators=10,random_state=2) #基础模型
x_features_ = RFE(RCF_,n_features_to_select=340,step=50).fit(x,y) #transform后返回特征。。。保留340个特征，每次迭代减少50个特征

x_features.support_ #这是一个bool型的矩阵，表示对应得特征是否被使用
x_features.ranking_ #这是一个表示每个特征重要性排名的列表，靠前的重要

x_features = x_features_.transform(x) #返回特征
