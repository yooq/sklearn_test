'''
核心
https://blog.csdn.net/FelixWang0515/article/details/94629025
'''


clf1 = svm.SVC(C=1,kernel='linear')
clf1.fit(l_d, l_c)
clf0 = svm.SVC(C=1,kernel='linear')  # 这里的核函数使用 linear， 惩罚参数的值设为1，可以尝试其他值
clf0.fit(l_d, l_c)
lu_c_0 = clf0.predict(lu_d)  # clf0直接使用有标签数据训练，训练完成对测试集进行分类
u_c_new = clf1.predict(u_d)  # 这里直接使用有标签数据训练得到的SVM模型对无标签数据进行分类，将其分类结果作为无标签数据的类别
cu, cl = 0.0001, 1   # 初始化有标签数据无标签数据重要程度的折中参数
sample_weight = np.ones(n)  # 样本权重， 直接让有标签数据的权重为Cl,无标签数据的权重为Cu
# print(sample_weight)
# print()
sample_weight[len(l_c):] = cu
# print(sample_weight)
id_set = np.arange(len(u_d))
while cu < cl:
    lu_c = np.concatenate((l_c, u_c_new))  # 70
    clf1.fit(lu_d, lu_c, sample_weight=sample_weight)
    while True:
        u_c_new = clf1.predict(u_d)  #  类别预测
        u_dist = clf1.decision_function(u_d)  #  表示点到当前超平面的距离
        norm_weight = np.linalg.norm(clf1.coef_)  # 权重数组
        epsilon = 1 - u_dist * u_c_new * norm_weight
        plus_set, plus_id = epsilon[u_c_new > 0], id_set[u_c_new > 0]  # 全部正例样本
        minus_set, minus_id = epsilon[u_c_new < 0], id_set[u_c_new < 0]  # negative labelled samples
        plus_max_id, minus_max_id = plus_id[np.argmax(plus_set)], minus_id[np.argmax(minus_set)]
        a, b = epsilon[plus_max_id], epsilon[minus_max_id]
        if a > 0 and b > 0 and a + b > 2:  # 若存在一对未标记样本，其标记指派不同，并且松弛向量相加的值大于2则以为分类错误的可能性很大，需要将二者的分类标签互换，重新训练
            u_c_new[plus_max_id], u_c_new[minus_max_id] = -u_c_new[plus_max_id], -u_c_new[minus_max_id]
            lu_c = np.concatenate((l_c, u_c_new))
            clf1.fit(lu_d, lu_c, sample_weight=sample_weight)  #翻转错误样本，重新训练
        else:
            break
    cu = min(cu * 2, cl) # 更新折中参数
    sample_weight[len(l_c):] = cu # 更新权重
lu_c = np.concatenate((l_c, u_c_new))
test_c1 = clf0.predict(test_d)
test_c2 = clf1.predict(test_d)
score1 = clf0.score(test_d,test_c) # SVM的模型精度
score2 = clf1.score(test_d,test_c) # TSVM的模型精度
