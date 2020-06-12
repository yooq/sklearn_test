from sklearn.preprocessing import StandardScaler

data = [[-1,2],[0,2],[-0.5,6],[0,10],[1,18]]

stand = StandardScaler()
result = stand.fit_transform(data)
result_ = stand.inverse_transform(result)  #将数据还原
print(result)

# 查看模型的方差和均值
var = stand.var_
mean = stand.mean_
print(var , mean)

# 查看结果的方差和均值
mean_ = result.mean()
var_ = result.std()
print(var_ , mean_)

