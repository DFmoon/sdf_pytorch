#-*-coding:utf-8-*_
import torch
import torch.nn.functional as F     #nn是torch中的神经网络模块
from torch.autograd import Variable
import matplotlib.pyplot as plt

# 做一些假数据来观看图像
x=torch.linspace(-5, 5, 200)    #在-5-5之间取200个点的数据
x=Variable(x)                   #把数据放进variable
x_np=x.data.numpy()             #torch的格式不能被plt识别，换成numpy

#几种常用的激励函数
y_relu=F.relu(x).data.numpy()
y_sigmoid=F.sigmoid(x).data.numpy()
y_tanh=F.tanh(x).data.numpy()
y_softplus=F.softplus(x).data.numpy()
#y_softmax=F.softmax(x)  softmax比较特殊，不能直接显示，它是关于概率的，用于分类

#开始画图
plt.figure(1, figsize=(8, 6))
plt.subplot(221)
plt.plot(x_np, y_relu, c='red', label='relu')
plt.ylim((-1, 5))           #设置坐标轴的取值范围
plt.legend(loc='best')      #自动调整图例的位置

plt.subplot(222)
plt.plot(x_np, y_sigmoid, c='red', label='sigmoid')
plt.ylim((-0.2, 1.2))
plt.legend(loc='best')

plt.subplot(223)
plt.plot(x_np, y_tanh, c='red', label='tanh')
plt.ylim((-1.2, 1.2))
plt.legend(loc='best')

plt.subplot(224)
plt.plot(x_np, y_softplus, c='red', label='softplus')
plt.ylim((-0.2, 6))
plt.legend(loc='best')

plt.show()