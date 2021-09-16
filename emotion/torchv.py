'''
函数说明: 
Author: hongqing
Date: 2021-09-03 11:31:51
LastEditTime: 2021-09-03 15:34:54
'''
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
from kt_utils import *
%matplotlib inline
np.random.seed(1)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import dataloader,Dataset
X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = load_dataset()

X_train = X_train_orig/255.
X_test = X_test_orig/255.
Y_train = Y_train_orig.T.reshape(-1)
Y_test = Y_test_orig.T.reshape(-1)
print ("训练样本数 = " + str(X_train.shape[0]))
print ("测试样本数 = " + str(X_test.shape[0]))
X_train = np.transpose(X_train,[0,3,1,2])
X_test = np.transpose(X_test,[0,3,1,2])
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
datas = dataloader.DataLoader(dataset,50,True)

class mynet(nn.Module):
    def __init__(self):
        super().__init__()
        #(600, 3,64, 64)
        self.conv1 = nn.Conv2d(3,32,7,padding=3) #64*64*32
        self.bn = nn.BatchNorm2d(32)
        self.poll1 = nn.MaxPool2d(2,2) #32*32*32
        
        self.dense = nn.Linear(32*32*32,2)
        

    def forward(self,x):
        x = F.relu(self.bn(self.conv1(x)))
        x = self.poll1(x).reshape(-1,32*32*32)
        x = self.dense(x)
        return x


learning_rate=0.009
num_epochs=40

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

import cv2
## 这里改成你自己的图片 跑测试
img_path = 'images/my_image.jpg'
img1 = cv2.imread(img_path)
img1 = cv2.resize(img1,dsize=(64,64),interpolation=cv2.INTER_CUBIC) 
imshow(img1)
img1 = np.transpose(img1,[2,0,1])
img1 = img1[np.newaxis,:]
x1 = net(torch.FloatTensor(img1))
if(x1.argmax().item()==0):
    print('哭')
else:
    print('笑')