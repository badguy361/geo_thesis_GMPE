# https://blog.csdn.net/qq_41858347/article/details/107783932
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import transforms
import numpy as np
import matplotlib.pyplot as plt
import torchvision.utils as vutils
import PIL.Image as Image

def get_circle_dataset(n = 1000): #產生二維數據
    dr,angle = np.random.randn(n)*0.05 , 1.5*np.pi*np.random.rand(n)
    r = dr+1.3
    x = r*np.sin(angle)
    y = r*np.cos(angle)
    data = np.concatenate([x[np.newaxis,:],y[np.newaxis,:]]).T
    return data

circle_data = get_circle_dataset()
plt.scatter(circle_data[:,0],circle_data[:,1])

#################################################

latent_size = 16 # input layer

# Discriminator
D = nn.Sequential( 
    nn.Linear(2, 64),
    nn.ReLU(),
    nn.Linear(64, 64),
    nn.ReLU(),
    nn.Linear(64, 64),
    nn.ReLU(),
    nn.Linear(64, 1), # 輸出一個概率值
    nn.Sigmoid()) 

# Generator 
G = nn.Sequential(
    nn.Linear(latent_size, 32),
    nn.Tanh(),
    nn.Linear(32, 32),
    nn.Tanh(),
    nn.Linear(32, 32),
    nn.Tanh(),
    nn.Linear(32, 2))

#################################################

from torch.utils.data import DataLoader,TensorDataset

batch_size = 500
num_epochs = 500

real_data = torch.FloatTensor(circle_data)
data_set = TensorDataset(real_data)
data_loader = DataLoader(dataset=data_set, batch_size=batch_size, shuffle=True)
loss_func = nn.MSELoss()
d_optimizer = torch.optim.Adam(D.parameters(),0.0002)
g_optimizer = torch.optim.Adam(G.parameters(),0.0002)


# Start training
g_losses = []
d_losses = []


total_step = len(data_loader)
for epoch in range(num_epochs):
    for step, x in enumerate(data_loader):
        x = x[0]
        real_labels = torch.ones(batch_size, 1)
        fake_labels = torch.zeros(batch_size, 1)
        # 为真实数据计算BCEloss
        outputs = D(x)
        d_loss_real = loss_func(outputs, real_labels)
        # 为假数据计算BCEloss
        z = torch.randn(batch_size, latent_size)
        fake_data = G(z)
        outputs = D(fake_data)
        d_loss_fake = loss_func(outputs, fake_labels)
        
        # 训练，只训练分类器
        d_loss = d_loss_real + d_loss_fake
        d_optimizer.zero_grad()
        g_optimizer.zero_grad()
        d_loss.backward()
        d_optimizer.step()
        
        d_losses.append(d_loss.item())
        
        z = torch.randn(batch_size, latent_size)
        fake_data = G(z)
        outputs = D(fake_data)
        # 这里让生成器学习让损失函数朝着真样本的一侧移动
        g_loss = loss_func(outputs, real_labels)
        
        g_losses.append(g_loss.item())
        
        # 训练生成器
        d_optimizer.zero_grad()
        g_optimizer.zero_grad()
        g_loss.backward()
        g_optimizer.step()
    
    
    if (epoch+1)%10==0:
        # 从生成器采样生成一些假数据点
        z = torch.randn(1000, latent_size)
        with torch.no_grad():
            fake_data = G(z)
        fake_x,fake_y = fake_data[:,0].numpy(),fake_data[:,1].numpy()
        real_x,real_y = real_data[:,0].numpy(),real_data[:,1].numpy()

        step = 0.02
        x = np.arange(-2,2,step)
        y = np.arange(-2,2,step)

        #将原始数据变成网格数据形式
        X,Y = np.meshgrid(x,y)
        n,m = X.shape
        #写入函数，z是大写
        inputs = torch.stack([torch.FloatTensor(X),torch.FloatTensor(Y)])
        inputs = inputs.permute(1,2,0)
        inputs = inputs.reshape(-1,2)
        with torch.no_grad():
            Z = D(inputs)
        Z = Z.numpy().reshape(n,m)
        
        plt.figure(figsize=(7,6))
        plt.title('Discriminator probablity')
        cset = plt.contourf(X,Y,Z,100)
        plt.colorbar(cset)
        plt.show()
        
        plt.figure(figsize=(6,6))

        plt.scatter(real_x,real_y,c = 'w', edgecolor='b')
        plt.scatter(fake_x,fake_y,c = 'r')
        plt.title('Scatter epoch %d'%(epoch+1))
        #contour = plt.contour(X,Y,Z,1)
        #plt.clabel(contour,colors='k')
        plt.show()
        
plt.figure(figsize=(10,4))
plt.plot(g_losses,label='Generator')
plt.plot(d_losses,label='Discriminator')
plt.legend()
