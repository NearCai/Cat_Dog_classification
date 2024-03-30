import os
import torch
import torchvision
import numpy as np
from PIL import Image
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter

from net import *


data_transform = transforms.Compose([
    transforms.Resize((256,256)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.504, 0.504, 0.383),std=(0.312, 0.294, 0.327))
])

train_data = datasets.ImageFolder(root="",
transform=data_transform)

train_data_size = len(train_data)
print(f'训练数据集的长度为:%d' %train_data_size)

train_dataloader = DataLoader(dataset=train_data,batch_size=64,shuffle=True)

net = Animal_Net()
net.load_state_dict(torch.load("net2_train_87.pth"))

loss_fn = nn.CrossEntropyLoss()

optim = torch.optim.Adam(params=net.parameters(),lr=0.0003)



train_step = 0
test_step = 0

epoch = 10

writer = SummaryWriter('../net2_train')

for i in range(epoch):
    print(f'--------第%d轮训练开始--------' %(i+1))

    net.train()
    for data in train_dataloader:
        imgs, targets = data

        output = net(imgs)
        loss = loss_fn(output, targets)

        optim.zero_grad()
        loss.backward()
        optim.step()

        train_step += 1
        if train_step % 100 == 0:
            print(f'训练次数：{train_step}，Loss：{loss.item()}')
            writer.add_scalar('train_loss',loss.item(),train_step)

    torch.save(net.state_dict(), f"net2_train_{i+111}.pth")

writer.close()

