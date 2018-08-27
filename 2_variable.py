#-*-coding:utf-8-*-
import torch
from torch.autograd import Variable

#神经网络中的参数都是variable变量的形式，以下将tensor放入variable中
tensor=torch.FloatTensor([[1,2],[3,4]])
variable=Variable(tensor,requires_grad=True)    #requires_grad=True将variable放入反向传播过程，将会计算梯度

t_out=torch.mean(tensor*tensor)             #求均值
v_out=torch.mean(variable*variable)         #v_out=1/4*sum(var^2)=1/4*(1*1+2*2+3*3+4*4)=7.5
print(t_out)
print(v_out)

v_out.backward()            #计算误差反向传递
print(variable.grad)        #反向传递后的variable的梯度,grad=d(v_out)/d(var)=1/4*2*variable=variable/2(对v_out求导)

print(variable.data.numpy())        #variable.data就是tensor
