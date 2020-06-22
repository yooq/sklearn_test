import pickle

pickle.dumps(model,open('name','wb'))   #保存模型


model = pickle.load(open('name','rb'))  #加载模型
