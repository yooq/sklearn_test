from sklearn.preprocessing import LabelEncoder,OneHotEncoder,LabelBinarizer

list_name=[['赤'],['橙'],['黄']]

labelencoder = LabelEncoder()
list_enconder = labelencoder.fit_transform(list_name)
print(list_enconder)


l =['吃撑','黄绿','青蓝紫']
labelbinar = LabelBinarizer()
list_lba = labelbinar.fit_transform(l)
print(list_lba)
# [[1 0 0]
#  [0 0 1]
#  [0 1 0]]




onehot = OneHotEncoder(categories='auto')
result = onehot.fit_transform(list_name)
print(result.toarray())

cc = onehot.fit(list_name).get_feature_names()  #查看编码后每一列对应特征  ['x0_橙' 'x0_赤' 'x0_黄']
print(cc)




from sklearn.preprocessing import KBinsDiscretizer



# 二值化,将连续特征转换成二值特征，比如将年龄转换成0,1.多少岁为0，多少岁为1
from sklearn.preprocessing import Binarizer
a = Binarizer(threshold=10)


# 分箱，二值化的拓展
from sklearn.preprocessing import KBinsDiscretizer
a  = KBinsDiscretizer(n_bins=3,encode='onehot',strategy='uniform')  #参数1：分3个箱，参数2：以何种编码展示。参数3：以何种方法分箱，有聚类方法

