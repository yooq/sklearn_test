from sklearn.preprocessing import  MinMaxScaler
data = [[-1,2],[0,2],[-0.5,6],[0,10],[1,18]]

scaler = MinMaxScaler()
# scaler = scaler.fit(data)
# scaler.transform(data)

'''
fit_transform()  等于 fit(),transform()
'''
result = scaler.fit_transform(data)  #训练和转换归一化
print(result)

inver = scaler.inverse_transform(result)  # 将归一化结果 转回 原数据
print(inver)

scaler = MinMaxScaler(feature_range=[0,10])  #将归一化到feature_range范围内，默认是[0,1].
result1 = scaler.fit_transform(data)
print(result1)

# scaler.partial_fit(data)  # 当数据了很多的时候，fit可能会报错表示自己计算不过来，这个时候partial_fit
