#-*-coding:utf-8-*_
import torch
import torch.nn.functional as F     #nn是torch中的神经网络模块
from torch.autograd import Variable
import matplotlib.pyplot as plt

#假数据
x=torch.unsqueeze(torch.linspace(-1,1,100),dim=1)    #把一维的数据变成二维
y=x.pow(2)+0.2*torch.rand(x.size())                  #y是x的平方加上噪点的影响
x=Variable(x,requires_grad=False)
y=Variable(y,requires_grad=False)

#保存神经网络
def save():
    net1=torch.nn.Sequential(
        torch.nn.Linear(1,10),
        torch.nn.ReLU(),
        torch.nn.Linear(10,1),
    )
    optimizer=torch.optim.SGD(net1.parameters(),lr=0.5)
    loss_func=torch.nn.MSELoss()  
    
    for t in range(500):            #给定训练步数
        prediction=net1(x)
        loss=loss_func(prediction,y)
        
        optimizer.zero_grad()       #清空上一步的残余更新参数值（梯度）
        loss.backward()             #误差反向传播，计算参数更新值
        optimizer.step()            #将参数更新值施加到net的parameters上
    #出图
    plt.figure(1,figsize=(10,3))
    plt.subplot(131)
    plt.title('net1')
    plt.scatter(x.data.numpy(),y.data.numpy())
    plt.plot(x.data.numpy(),prediction.data.numpy(),'r-',lw=5)
        
    torch.save(net1,'net.pkl')      #保存整个神经网络
    torch.save(net1.state_dict(),'net_params.pkl')          #仅保存节点参数
    
#提取神经网络    
def restore_net():
    net2=torch.load('net.pkl')
    prediction = net2(x)
    
    #出图
    plt.subplot(132)
    plt.title('net2')
    plt.scatter(x.data.numpy(), y.data.numpy())
    plt.plot(x.data.numpy(), prediction.data.numpy(), 'r-', lw=5)
    
def restore_params():
    net3=torch.nn.Sequential(
        torch.nn.Linear(1,10),
        torch.nn.ReLU(),
        torch.nn.Linear(10,1),
    )
    net3.load_state_dict(torch.load('net_params.pkl'))
    prediction = net3(x)
    
    #出图
    plt.subplot(133)
    plt.title('net3')
    plt.scatter(x.data.numpy(), y.data.numpy())
    plt.plot(x.data.numpy(), prediction.data.numpy(), 'r-', lw=5)
    plt.show()
    
save()
restore_net()
restore_params()