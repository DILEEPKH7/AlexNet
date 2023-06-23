import matplotlib.pyplot as plt
import numpy as np

import torch
from torch import nn, optim
from torchvision import datasets, transforms

import zipfile

import shutil
import os
import pandas as pd

device = ("cuda" if torch.cuda.is_available() else "cpu")
print(device)

class Alexnet(nn.Module):

    def __init__(self,num_classes: int=2):
        super(Alexnet, self).__init__()

        self.convolutional = nn.Sequential(
            nn.Conv2d(3,96, kernel_size = 11, stride=4, padding =2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),

            nn.Conv2d(96,256, kernel_size = 5, padding =2),
            nn.ReLU(inplace = True),
            nn.MaxPool2d(kernel_size=3, stride=2),

            nn.Conv2d(256,384, kernel_size = 3, padding =1),
            nn.ReLU(inplace = True),
            nn.Conv2d(384,384, kernel_size = 3, padding =1),
            nn.ReLU(inplace = True),

            nn.Conv2d(384,256, kernel_size = 3, padding =1),
            nn.ReLU(inplace = True),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6,6))

        self.linear = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256*6*6,4096),
            nn.ReLU(inplace=True),
            
            nn.Dropout(),
            nn.Linear(4096,4096),
            nn.ReLU(inplace=True),

            nn.Linear(4096,2)
    )

    def forward(self,x):
        x = self.convolutional(x)
        x = self.avgpool(x)

        x = torch.flatten(x,1)
        x = self.linear(x)

        return torch.sigmoid(x)

model = Alexnet()
print(model)

df = pd.read_csv("Users/khdil/Desktop/My_files/Github/AlexNet/list_attr_celeba.csv")
df = df[['image_id','Smiling']]

print(df.iterrows())

num = 1500
s0,s1 = 0,0

for i, (_,i_row) in enumerate(df.iterrows()):
    if s0<num:
        if i_row['Smiling'] == 1:
            s0+=1
            shutil.copyfile('Users/khdil/Desktop/My_files/Github/AlexNet/celeba/img_align_celeba/' + i_row['image_id'], 'Users/khdil/Desktop/My_files/Github/AlexNet/data/smile/' + i_row['image_id'])
    
    if s1<num:
        if i_row['Smiling'] == -1:
            s1+=1
            shutil.copyfile('Users/khdil/Desktop/My_files/Github/AlexNet/celeba/img_align_celeba/' + i_row['image_id'], 'Users/khdil/Desktop/My_files/Github/AlexNet/data/no_smile/' + i_row['image_id'])         

img_list = os.listdir('Users/khdil/Desktop/My_files/Github/AlexNet/data/smile/')
img_list.extend(os.listdir('Users/khdil/Desktop/My_files/Github/AlexNet/data/no_smile/'))

print("Images: ", len(img_list))

transform = transforms.Compose([
    transforms.Resize((64,64)),
    transforms.ToTensor()
])

celeba_data = datasets.ImageFolder('Users/khdil/Desktop/My_files/Github/AlexNet/data/',transform=transform)

print(celeba_data.classes)
print(len(celeba_data))

train_size = int(len(img_list)*0.75)
test_size = len(img_list)-train_size


train_set, test_set = torch.utils.data.random_split(celeba_data, [train_size,test_size])

trainLoader = torch.utils.data.DataLoader(train_set, batch_size=64, shuffle=True)
testLoader = torch.utils.data.DataLoader(train_set, batch_size=64, shuffle=True)

optimizer = optim.Adam(model.parameters(), lr=0.0001)
criterion = nn.CrossEntropyLoss()

epochs = 10
train_loss =[]

for epoch in range(epochs):
    total_train_loss = 0 

    for idx, (image, label) in enumerate(trainLoader):
        optimizer.zero_grad()

        pred = model(image)
        loss = criterion(pred, label)

        total_train_loss += loss.item()
        
        loss.backward()
        optimizer.step()

total_train_loss = total_train_loss/(idx+1)
train_loss.append(total_train_loss)

plt.plot(train_loss)

testiter = iter(testLoader)
images, labels = testiter._next()

with torch.no_grad():
    pred = model(images)

print(pred.shape)

images_np = [i.cpu() for i in images]
class_names = celeba_data.classes

print(class_names)

fig = plt.figure(figsize=(15,7))
fig.subplot_adjust(left=0, right=1, bottom = 0, top=1, hspace = 0.05, wspace = 0.05)

for i in range(50):
    ax= fig.add_subplot(5,10,i+1, xticks=[], yticks=[])
    ax.imshow(images_np[i].permute(1,2,0), cmap=plt.cm.gray_r, interpolation='nearest')

    if labels[i] == torch.max(pred[i],0)[1]:
        ax.text(0,3,class_names[torch.max(pred[i],0)[1]],color='blue')
    else:
        ax.text(0,3,class_names[torch.max(pred[i],0)[1]],color='red')






