#-*-coding:utf-8-*-
import torch
import torch.utils.data as Data
from torch import nn
from torch.autograd import Variable
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

#Hyper Parameters
EPOCH=1
BATCH_SIZE=64
TIME_STEP=28        #对于28步中的每一步读取其一行的信息
INPUT_SIZE=28       #一行28个像素点
LR=0.01
DOWNLOAD_MNIST=False

train_data=dsets.MNIST(root='./mnist',train=True,transform=transforms.ToTensor(),download=DOWNLOAD_MNIST)   #转成tensor
train_loader=Data.DataLoader(dataset=train_data,batch_size=BATCH_SIZE,shuffle=True)     #用于批训练

test_data=dsets.MNIST(root='./mnist',train=False,transform=transforms.ToTensor())
test_x=Variable(test_data.test_data,volatile=True).type(torch.FloatTensor)[:2000]/255.
test_y=test_data.test_labels.numpy().squeeze()[:2000]

class RNN(nn.Module):
    def __init__(self):
        super(RNN,self).__init__()
        self.rnn=nn.LSTM(
            input_size=INPUT_SIZE,          #代表28个时间点
            hidden_size=64,                 #隐藏的神经元个数
            num_layers=2,
            batch_first=True,
        )
        self.out=nn.Linear(64,10)           #输出层10表示最后有10个分类
    def forward(self,x):
        r_out,(h_n,h_c)=self.rnn(x,None)    #x:(batch_size,time_step,input_size)
        # r_out shape (batch, time_step, output_size)
        # h_n shape (n_layers, batch, hidden_size)
        # h_c shape (n_layers, batch, hidden_size)        
        out=self.out(r_out[:,-1,:])         #选最后一个时刻的r_out，即time_step的维度设置为-1
        return out

rnn=RNN()
print(rnn)

optimizer=torch.optim.Adam(rnn.parameters(),lr=LR)
loss_func=nn.CrossEntropyLoss()

for epoch in range(EPOCH):
    for step,(x,y) in enumerate(train_loader):
        b_x=Variable(x.view(-1,28,28))      #reshape x to (batch_size,time_step,input_size)
        b_y=Variable(y)
        output=rnn(b_x)
        loss=loss_func(output,b_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if step%50==0:
            test_output=rnn(test_x)
            pred_y=torch.max(test_output,1)[1].data.numpy().squeeze()
            accuracy=float(sum(pred_y==test_y))/float(test_y.size)
            print('Epoch:',epoch,'|train loss:%.4f'%loss.data[0],'|accuracy:%.4f'%accuracy)

test_output=rnn(test_x[:10].view(-1,28,28))
pred_y=torch.max(test_output,1)[1].data.numpy().squeeze()
print(pred_y,'prediction number')
print(test_y[:10],'real number')        