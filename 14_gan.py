#-*-coding:utf-8-*-
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.autograd import Variable

#Hyper Parameters
BATCH_SIZE=64
LR_G=0.0001     #learning rate for generator
LR_D=0.0001     #learning rate for discriminator
N_IDEAS=5       #Generator的随机想法数量
ART_COMPONENTS=15       #画一元二次曲线时总共取的点数
PAINT_POINTS=np.vstack([np.linspace(-1,1,ART_COMPONENTS) for _ in range(BATCH_SIZE)])

#需要学习生成的图像实例
#plt.plot(PAINT_POINTS[0],2*np.power(PAINT_POINTS[0],2)+1,c='#74BCFF',lw=3,label='upper bound') #曲线1
#plt.plot(PAINT_POINTS[0],1*np.power(PAINT_POINTS[0],2)+0,c='#FF9359',lw=3,label='lower bound') #曲线2
#plt.legend(loc='upper right')
#plt.show()

#需要学习生成的最终目标
def artist_works():
    a=np.random.uniform(1,2,size=BATCH_SIZE)[:,np.newaxis]  #a为一元二次函数的二次项系数
    paintings=a*np.power(PAINT_POINTS,2)+(a-1)              #生成的是曲线1和曲线2中间的部分
    paintings=torch.from_numpy(paintings).float()           #转换成torch的形式
    return Variable(paintings)      #放在Variable中

G=nn.Sequential(                    #Generator
    nn.Linear(N_IDEAS, 128),        #5个随机想法
    nn.ReLU(),
    nn.Linear(128,ART_COMPONENTS),  #从随机想法生成的15个y的值
)

D=nn.Sequential(                    #Discriminator
    nn.Linear(ART_COMPONENTS,128),  #接收Generator生成的东西
    nn.ReLU(),
    nn.Linear(128,1),               #输出对于接收到的东西的判别
    nn.Sigmoid(),                   #将判别结果转成百分比
)

opt_D=torch.optim.Adam(D.parameters(),lr=LR_D)
opt_G=torch.optim.Adam(G.parameters(),lr=LR_G)

plt.ion()

for step in range(5000):
    artist_paintings=artist_works()           #返回目标域
    G_ideas=Variable(torch.randn(BATCH_SIZE,N_IDEAS))
    G_paintings=G(G_ideas)                    #生成图像
    
    prob_artist0=D(artist_paintings)          #分析artist_paintings中有多少是来自目标（100%）
    prob_artist1=D(G_paintings)               #分析G_paintings中有多少是来自目标

    D_loss=-(torch.mean(torch.log(prob_artist0)+torch.log(1.-prob_artist1)))
    G_loss=torch.mean(torch.log(1.-prob_artist1))

    opt_D.zero_grad()
    D_loss.backward(retain_graph=True)    #保留相关参数给下一次反向传递，注意是retain_graph
    opt_D.step()

    opt_G.zero_grad()
    G_loss.backward()
    opt_G.step()
    
    if step % 50 == 0:  # plotting
        plt.cla()       # Clear axis
        plt.plot(PAINT_POINTS[0],G_paintings.data.numpy()[0],c='#4AD631',lw=3,label='Generated painting',)
        plt.plot(PAINT_POINTS[0],2*np.power(PAINT_POINTS[0],2)+1,c='#74BCFF',lw=3,label='upper bound')
        plt.plot(PAINT_POINTS[0],1*np.power(PAINT_POINTS[0],2)+0,c='#FF9359',lw=3,label='lower bound')
        plt.text(-.5,2.3,'D accuracy=%.2f (0.5 for D to converge)'% prob_artist0.data.numpy().mean(),fontdict={'size':12})
        plt.text(-.5,2,'D score= %.2f (-1.38 for G to converge)'% -D_loss.data.numpy(),fontdict={'size':12})
        plt.ylim((0,3));
        plt.legend(loc='upper right',fontsize=12);
        plt.draw();     # re-draw the figure
        plt.pause(0.01)
plt.ioff()
plt.show()        