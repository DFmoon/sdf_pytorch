#-*-coding:utf-8-*-
import torch
import torch.utils.data as Data                   #批训练的模块

BATCH_SIZE=5                    #定义批训练的一批的数据量

x = torch.linspace(1, 10, 10)       # this is x data (torch tensor)
y = torch.linspace(10, 1, 10)       # this is y data (torch tensor)

torch_dataset = Data.TensorDataset(x, y) 

loader=Data.DataLoader(
    dataset=torch_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=2
    )              #使训练批量化，shuffle=True打乱数据排序，num_workers给定用多少个进程提取数据
#要在Windows中使用num_workers实现multiprocessing时，必须在if __name__ == '__main__':的结构下，否则就会出错

if __name__=='__main__':        #不在loader中使用num_workers参数时可以去掉这一行
    for epoch in range(3):          #数据整体训练三次
        for step,(batch_x,batch_y) in enumerate (loader):               #step为2，总共10个数据，每批5个
            #这里就是训练的地方...
            print('Epoch:',epoch,'|Step:',step,'|batch x:',batch_x.numpy(),'|batch y:',batch_y.numpy())
