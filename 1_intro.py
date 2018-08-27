#-*-coding:utf-8-*-
import torch        #神经网络模块，动态
import numpy as np 

#numpy和torch数据的相互转换
np_data=np.arange(6).reshape((2,3))
torch_data=torch.from_numpy(np_data)
np_data2=torch_data.numpy()
print('numpy_data:\n',np_data,'\n')
print('torch_data:\n',torch_data,'\n')
print('numpy_data2:\n',np_data2,'\n\n')

#abs
data=[-1,-1,3,2]
tensor=torch.FloatTensor(data)
print(np.abs(data),'\n')
print(torch.abs(tensor),'\n')

#矩阵相乘
data2=[[-1,-1],[3,2]]
tensor2=torch.FloatTensor(data2)
data3=tensor2.numpy()
data4=np.array(tensor2)
print(np.matmul(data2,data2),'\n')      #np.matmul(data2,data2)就是,data2.dot(data2)
print(torch.mm(tensor2,tensor2),'\n')   #torch的data2.dot(data2)形式运算出来的是一个数字
print('data3:\n',data3)
print('data4:\n',data4)