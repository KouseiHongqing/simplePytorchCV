'''
函数说明: 
Author: hongqing
Date: 2021-09-03 11:31:51
LastEditTime: 2021-09-16 15:06:29
'''
import numpy as np
import matplotlib.pyplot as plt
from cnn_utils import *

%matplotlib inline
np.random.seed(1)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import dataloader,Dataset
X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = load_dataset()

X_train = X_train_orig/255.
X_test = X_test_orig/255.
Y_train = convert_to_one_hot(Y_train_orig, 6).T
Y_test = convert_to_one_hot(Y_test_orig, 6).T
print ("训练样本数 = " + str(X_train.shape[0]))
print ("测试样本数 = " + str(X_test.shape[0]))
X_train = np.transpose(X_train,[0,3,1,2])
X_test = np.transpose(X_test,[0,3,1,2])
Y_train = Y_train.argmax(axis=1)
Y_test = Y_test.argmax(axis=1)
print ("X_train的维度: " + str(X_train.shape))
print ("Y_train的维度: " + str(Y_train.shape))
print ("X_test的维度: " + str(X_test.shape))
print ("Y_test的维度: " + str(Y_test.shape))
conv_layers = {}

np.argmax(Y_train,)
class myDataset(Dataset):
    def __init__(self,X,Y):
        self.X= X
        self.Y= Y

    def __getitem__(self, index):
        return self.X[index],self.Y[index]

    def __len__(self):
        return self.X.shape[0]
dataset = myDataset(X_train,Y_train)
datas = dataloader.DataLoader(dataset,64,True)

class mynet(nn.Module):
    def __init__(self):
        super().__init__()
        #(1080, 64, 64, 3)
        self.conv1 = nn.Conv2d(3,8,4,padding='same') #64*64*8
        self.poll1 = nn.MaxPool2d(8,8) #8*8*8
        self.conv2 = nn.Conv2d(8,16,2,padding='same') # 8*8*16
        self.poll2 = nn.MaxPool2d(4,4) # 2*2*16
        self.dense = nn.Linear(64,6)
        

    def forward(self,x):
        x = F.relu(self.conv1(x))
        x = self.poll1(x)
        x = F.relu(self.conv2(x))
        x = self.poll2(x).reshape(-1,64)
        x = self.dense(x)
        return x


learning_rate=0.009
num_epochs=100

net = mynet()
optimizer = torch.optim.Adam(net.parameters(),lr=learning_rate)
criterion = nn.CrossEntropyLoss()
costs=[]
for i in range(num_epochs):
    minibatch_cost = 0.
    for index,(data,label) in enumerate(datas):
        out = net(data.float())
        optimizer.zero_grad()
        loss = criterion(out,label)
        loss.backward()
        optimizer.step()
        minibatch_cost+=loss.item()
    costs.append(minibatch_cost)
    print('epoch:{} finished,cost:{}'.format(i,minibatch_cost))

plt.plot(np.squeeze(costs))
plt.ylabel('cost')
plt.xlabel('iterations (per tens)')
plt.title("Learning rate =" + str(learning_rate))
plt.show()
# 计算预测精准度
predict_op = torch.max(net(torch.tensor(X_test,dtype=torch.float)),dim=1)[1].numpy()
correct_prediction = np.sum(predict_op==Y_test)
accuracy = correct_prediction/len(Y_test)

tpredict_op = torch.max(net(torch.tensor(X_train,dtype=torch.float)),dim=1)[1].numpy()
tcorrect_prediction = np.sum(tpredict_op==Y_train)
taccuracy = tcorrect_prediction/len(Y_train)
print("训练集预测精准度:", taccuracy)
print("测试集预测精准度:", accuracy)
