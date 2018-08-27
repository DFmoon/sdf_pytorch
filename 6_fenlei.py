#-*-coding:utf-8-*_
import torch
import torch.nn.functional as F     #nn是torch中的神经网络模块
from torch.autograd import Variable
import matplotlib.pyplot as plt

#假数据
n_data=torch.ones(100,2)         #数据的基本形态
x0=torch.normal(2*n_data,1)      #类型0的数据
y0=torch.zeros(100)              #类型0的标签
x1=torch.normal(-2*n_data,1)     #类型1的数据
y1=torch.ones(100)               #类型1的标签

#注意x，y数据的数据形式是一定要像下面一样(torch.cat是在合并数据)
x=torch.cat((x0,x1),0).type(torch.FloatTensor)  #FloatTensor=32-bit floating合并数据
y=torch.cat((y0,y1),).type(torch.LongTensor)    #LongTensor=64-bit integer合并标签

#torch只能在Variable上训练，所以把它们变成Variable
x, y = Variable(x), Variable(y)

#简化的神经网络代码
net=torch.nn.Sequential(
    torch.nn.Linear(2,10),
    torch.nn.ReLU(),
    torch.nn.Linear(10,2),
)

print(net)

plt.ion()                       #设置实时打印
plt.show()

#调用优化模块中的SGD优化神经网络参数
optimizer=torch.optim.SGD(net.parameters(),lr=0.02)     #传入net的参数，并给定学习效率（一般小于1）

#定义损失函数，计算误差
loss_func=torch.nn.CrossEntropyLoss()                   #输出的是属于每一个类别对应的概率

for t in range(100):            #给定训练步数
    out=net(x)                  #给net训练数据x，输出预测值
    loss=loss_func(out,y)       #计算误差
    
    optimizer.zero_grad()       #清空上一步的残余更新参数值（梯度）
    loss.backward()             #误差反向传播，计算参数更新值
    optimizer.step()            #将参数更新值施加到net的parameters上
    
    if t % 2 == 0:              #每学习2步，打印一次图像信息
        plt.cla()
        #经过softmax激励函数得到概率后取最大概率才是预测值，即所属的类别
        prediction=torch.max(F.softmax(out),1)[1]
        pred_y=prediction.data.numpy().squeeze()
        target_y=y.data.numpy()
        plt.scatter(x.data.numpy()[:,0],x.data.numpy()[:,1],c=pred_y,s=100,lw=0,cmap='RdYlGn')
        accuracy=sum(pred_y==target_y)/200              #预测中有多少和真实值一样
        plt.text(1.5,-4,'Accuracy=%.2f'%accuracy,fontdict={'size':20,'color':'red'})
        plt.pause(1)

plt.ioff()                       #关闭实时打印
plt.show()