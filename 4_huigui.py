#-*-coding:utf-8-*_
import torch
import torch.nn.functional as F     #nn是torch中的神经网络模块
from torch.autograd import Variable
import matplotlib.pyplot as plt

# 做一些假数据
x=torch.unsqueeze(torch.linspace(-1,1,100),dim=1)    #把一维的数据变成二维
y=x.pow(2)+0.2*torch.rand(x.size())                  #y是x的平方加上噪点的影响

x,y=Variable(x),Variable(y)     #把数据变成variable的形式，神经网络只能识别variable

#plt.scatter(x.data.numpy(),y.data.numpy())          #打印散点图
#plt.show()

#创建一个神经网络
class Net(torch.nn.Module):                     #继承torch的Module模块
    def __init__(self,n_feature,n_hidden,n_output):     #搭建层所需要的信息
        super(Net,self).__init__()                      #继承 __init__ 功能
        self.hidden=torch.nn.Linear(n_feature,n_hidden) #隐藏层线性输出，名为hidden，层信息都是self模块的属性
        self.predict=torch.nn.Linear(n_hidden,n_output) #输出层线性输出，名为predict
        
    def forward(self,x):        #前向传递的过程，把前面的流程信息组合起来，神经网络分析出输出值
        x=F.relu(self.hidden(x))#激励函数(隐藏层的线性值)
        r=self.predict(x)
        return r
    
net=Net(1,10,1)
print(net)

plt.ion()                       #设置实时打印
plt.show()

#调用优化模块中的SGD优化神经网络参数
optimizer=torch.optim.SGD(net.parameters(),lr=0.5)      #传入net的参数，并给定学习效率（一般小于1）

#定义损失函数，计算误差
loss_func=torch.nn.MSELoss()                            #均方差

for t in range(100):            #给定训练步数
    prediction=net(x)           #给net训练数据x，输出预测值
    loss=loss_func(prediction,y)#计算误差，两者位置不能换
    
    optimizer.zero_grad()       #清空上一步的残余更新参数值（梯度）
    loss.backward()             #误差反向传播，计算参数更新值
    optimizer.step()            #将参数更新值施加到net的parameters上
    
    if t % 5 == 0:              #每学习5步，打印一次图像信息
        plt.cla()
        plt.scatter(x.data.numpy(), y.data.numpy())     #原始信息
        plt.plot(x.data.numpy(), prediction.data.numpy(), 'r-', lw=5)       #预测信息
        plt.text(0.5, 0, 'Loss=%.4f' % loss.data[0], fontdict={'size': 20, 'color':  'red'})
        plt.pause(0.5)
plt.ioff()                       #关闭实时打印
plt.show()
