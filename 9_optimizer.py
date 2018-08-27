#-*-coding:utf-8-*-
import torch
import torch.utils.data as Data
import torch.nn.functional as F
from torch.autograd import Variable
import matplotlib.pyplot as plt

LR=0.01
BATCH_SIZE=32
EPOCH=12

x=torch.unsqueeze(torch.linspace(-1,1,1000),dim=1)
y=x.pow(2)+0.1*torch.normal(torch.zeros(*x.size()))
x,y=Variable(x),Variable(y)

class Net(torch.nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.hidden=torch.nn.Linear(1,20)
        self.predict=torch.nn.Linear(20,1)
        
    def forward(self,x):        #前向传递的过程，把前面的流程信息组合起来，神经网络分析出输出值
        x=F.relu(self.hidden(x))#激励函数(隐藏层的线性值)
        r=self.predict(x)
        return r

#建立四个神经网络放入列表    
net_SGD=Net()
net_Momentum=Net()
net_RMSprop=Net()
net_Adam=Net()
nets=[net_SGD,net_Momentum,net_RMSprop,net_Adam]

#分别用四种优化器优化参数，结果放入列表
opt_SGD=torch.optim.SGD(net_SGD.parameters(),lr=LR)
opt_Momentum=torch.optim.SGD(net_Momentum.parameters(),lr=LR,momentum=0.8)
opt_RMSprop=torch.optim.RMSprop(net_RMSprop.parameters(),lr=LR,alpha=0.9)
opt_Adam=torch.optim.Adam(net_Adam.parameters(),lr=LR,betas=(0.9,0.99))
optimizers=[opt_SGD,opt_Momentum,opt_RMSprop,opt_Adam]

#记录误差列表
lost_func=torch.nn.MSELoss()
losses_his=[[],[],[],[]]

for epoch in range(EPOCH):
    print(epoch)
    for net,opt,l_his in zip(nets,optimizers,losses_his):
        output=net(x)
        loss=lost_func(output,y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        l_his.append(loss.data[0])

labels = ['SGD', 'Momentum', 'RMSprop', 'Adam']
for i, l_his in enumerate(losses_his):
    #plt.subplot(i+221)
    plt.plot(l_his, label=labels[i])
plt.legend(loc='best')
plt.xlabel('Steps')
plt.ylabel('Loss')
plt.ylim((0, 0.2))
plt.show()
        