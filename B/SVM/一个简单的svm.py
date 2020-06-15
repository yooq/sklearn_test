from matplotlib.colors import ListedColormap
from sklearn.svm import SVC
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
import numpy as np

X,Y = make_blobs(n_samples=50,centers=2,random_state=0,cluster_std=0.6)


'''决策边界可视化'''

def plot_svc_decision_funtion(model,X,Y,s,ax=None):
    '''
    画决策边界
    :param model:
    :param ax:
    :return:
    '''
    plt.scatter(X[:, 0], X[:, 1], c=Y, s=s, cmap='rainbow') #数据散点图

    if ax is None:
        ax= plt.gca() #获取图例
    #找出 x，y轴最大最小值，用来画网格
    xlimit = ax.get_xlim()  #   x轴最大最小值
    ylimit = ax.get_ylim()  #   y轴最大最小值

    axisx = np.linspace(xlimit[0],xlimit[1],30) #   网格横坐标
    axisy = np.linspace(ylimit[0],ylimit[1],30) #   网格纵坐标

    axisx,axisy = np.meshgrid(axisx,axisy)  #网格矩阵
    xy = np.vstack([axisx.ravel(),axisy.ravel()]).T

    Z = model.decision_function(xy).reshape(axisx.shape) #计算样本到决策边界的距离

    #线图
    ax.contour(axisx,axisy,Z,colors='k',levels=[-1,0,1],alpha=0.5,linestyles=['--','-','--'])

    # 填充颜色图ax
    # custom_cmap = ListedColormap(['red', 'blue'])
    # ax.contourf(axisx,axisy,Z,alpha=0.5, cmap=custom_cmap)

    ax.set_xlim(xlimit)
    ax.set_ylim(ylimit)
    plt.show()


#模型
model_svc = SVC(kernel='linear').fit(X,Y)

plot_svc_decision_funtion(model_svc,X=X,Y=Y,s=50)





